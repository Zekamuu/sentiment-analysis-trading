"""Phase 5 — add sentiment (the actual experiment).

Re-runs the IDENTICAL price-only pipeline from Phase 4 with the lagged sentiment
features added as the exogenous X in ARIMAX — same ARIMA order, same threshold-
tuning procedure, same backtest. The ONLY difference is whether sentiment is
supplied. Then compares the sentiment model against the price-only control and the
Phase-1 baselines on net return per trade, hit rate, and cumulative net return.

A weak/null result (sentiment ~ price-only) is a legitimate, complete finding
(Design §10), and is exactly what Phase 3's Granger null predicted.

Run:  uv run python check_phase5.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.backtest import (
    always_buy,
    buy_and_hold,
    fixed_hold_backtest,
    random_decisions,
)
from src.config import load_config
from src.data import load_ohlcv
from src.features import SENTIMENT_FEATURES, TARGET
from src.model import decision_rule, one_step_forecasts, select_order, tune_threshold
from src.splits import chronological_split


def run_model(endog, exog, n_train, order, split, close, N, cost):
    """Forecast -> tune threshold on train -> backtest on test. Returns (result,
    decisions, forecasts, dir_acc)."""
    fc = one_step_forecasts(endog, n_train, order, exog=exog)
    train_prices = close.loc[split.train_index]
    thr = tune_threshold(fc.loc[split.train_index], train_prices, N, cost)
    test_prices = close.loc[split.test_index[0]: split.test_index[-1] + pd.Timedelta(days=N + 3)]
    test_fc = fc.loc[split.test_index]
    dec = decision_rule(test_fc, thr)
    res = fixed_hold_backtest(dec, test_prices, N, cost)
    actual = endog.loc[split.test_index]
    traded = dec != 0
    dir_acc = (float((np.sign(test_fc[traded]) == np.sign(actual[traded])).mean())
               if traded.any() else float("nan"))
    return res, thr, dir_acc


def main() -> int:
    cfg = load_config()
    N, cost = cfg["holding_period_N"], cfg["transaction_cost"]
    table = pd.read_parquet(cfg["data"]["processed_path"])
    close = load_ohlcv(cfg["data"]["ohlcv_path"])["close"]

    endog = table[TARGET]
    exog = table[SENTIMENT_FEATURES]
    split = chronological_split(table.index, cfg["split"]["train_frac"])
    n_train = len(split.train_index)

    # Same ARIMA order for both models (selected price-only on train) so the only
    # difference is the sentiment exog (Design §5.2 ablation).
    order = select_order(endog.iloc[:n_train])
    print(f"ARIMA order {order} | train {n_train} / test {len(split.test_index)} "
          f"| test {split.test_index[0].date()} -> {split.test_index[-1].date()} "
          f"| N={N}, cost={cost}\n")

    price_res, price_thr, price_da = run_model(endog, None, n_train, order, split, close, N, cost)
    sent_res, sent_thr, sent_da = run_model(endog, exog, n_train, order, split, close, N, cost)
    print(f"price-only : threshold {price_thr:.5f}, directional acc {price_da:.3f}")
    print(f"+sentiment : threshold {sent_thr:.5f}, directional acc {sent_da:.3f}")

    bnh = buy_and_hold(close.loc[split.test_index], cost)
    tp = close.loc[split.test_index[0]: split.test_index[-1] + pd.Timedelta(days=N + 3)]
    ab = fixed_hold_backtest(always_buy(tp), tp, N, cost)
    rnd = fixed_hold_backtest(random_decisions(tp, seed=cfg["random_seed"]), tp, N, cost)

    tbl = pd.DataFrame([
        sent_res.as_row("sentiment_ARIMAX"),
        price_res.as_row("price_only_ARIMA"),
        bnh.as_row("buy_and_hold"),
        ab.as_row("always_buy"),
        rnd.as_row("random"),
    ]).set_index("strategy")
    tbl["avg_net_return"] = tbl["avg_net_return"].round(5)
    tbl["hit_rate"] = tbl["hit_rate"].round(3)
    tbl["cumulative_net_return"] = tbl["cumulative_net_return"].round(4)
    print("\n" + tbl.to_string())

    tbl.to_csv("results/phase5_comparison.csv")
    print("\nSaved comparison -> results/phase5_comparison.csv")

    # Verdict (after costs).
    d_cum = sent_res.cumulative_net_return - price_res.cumulative_net_return
    beat = sent_res.cumulative_net_return > price_res.cumulative_net_return
    print(f"\nSentiment vs price-only: cumulative {sent_res.cumulative_net_return:+.4f} "
          f"vs {price_res.cumulative_net_return:+.4f} (Δ {d_cum:+.4f}) — "
          f"{'sentiment adds value' if beat else 'sentiment does NOT beat price-only'}.")
    print("Read alongside directional accuracy and the Phase 3 Granger null before "
          "claiming a signal — small trade counts make cumulative noisy.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
