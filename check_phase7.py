"""Phase 7 — LLM (CryptoBERT) + VADER fusion vs VADER-only and the price control.

Isolated experiment compared back to Phase 5: identical pipeline and ARIMA order,
the ONLY difference is which sentiment features enter ARIMAX as exog —
  - none           (price-only control)
  - VADER          (Phase 5)
  - VADER+CryptoBERT confidence-weighted fusion (this phase)
Reports the comparison table, directional accuracy, and a cumulative-return figure.

Run:  uv run python check_phase7.py
"""
from __future__ import annotations

import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.backtest import always_buy, buy_and_hold, fixed_hold_backtest, random_decisions
from src.config import load_config
from src.data import load_ohlcv
from src.features import SENTIMENT_FEATURES, TARGET
from src.model import decision_rule, one_step_forecasts, select_order, tune_threshold
from src.splits import chronological_split

FIG = "results/phase7_cumulative.png"


def run_model(endog, exog, n_train, order, split, close, N, cost):
    fc = one_step_forecasts(endog, n_train, order, exog=exog)
    thr = tune_threshold(fc.loc[split.train_index], close.loc[split.train_index], N, cost)
    tp = close.loc[split.test_index[0]: split.test_index[-1] + pd.Timedelta(days=N + 3)]
    test_fc = fc.loc[split.test_index]
    dec = decision_rule(test_fc, thr)
    res = fixed_hold_backtest(dec, tp, N, cost)
    traded = dec != 0
    da = (float((np.sign(test_fc[traded]) == np.sign(endog.loc[split.test_index][traded])).mean())
          if traded.any() else float("nan"))
    return res, da


def daily_equity(res, ti):
    return res.equity_curve.reindex(ti).ffill().fillna(0.0)


def main() -> int:
    cfg = load_config()
    N, cost = cfg["holding_period_N"], cfg["transaction_cost"]
    vader_tbl = pd.read_parquet(cfg["data"]["processed_path"])
    llm_tbl = pd.read_parquet(cfg["data"]["processed_path"].replace(
        "modeling_table.parquet", "modeling_table_llm.parquet"))
    close = load_ohlcv(cfg["data"]["ohlcv_path"])["close"]

    endog = vader_tbl[TARGET]
    split = chronological_split(vader_tbl.index, cfg["split"]["train_frac"])
    n_train = len(split.train_index)
    order = select_order(endog.iloc[:n_train])
    ti = split.test_index
    print(f"ARIMA order {order} | train {n_train} / test {len(ti)} "
          f"| test {ti[0].date()} -> {ti[-1].date()}\n")

    price_res, price_da = run_model(endog, None, n_train, order, split, close, N, cost)
    vader_res, vader_da = run_model(endog, vader_tbl[SENTIMENT_FEATURES], n_train, order, split, close, N, cost)
    comb_res, comb_da = run_model(endog, llm_tbl[SENTIMENT_FEATURES], n_train, order, split, close, N, cost)

    bnh = buy_and_hold(close.loc[ti], cost)
    tp = close.loc[ti[0]: ti[-1] + pd.Timedelta(days=N + 3)]
    ab = fixed_hold_backtest(always_buy(tp), tp, N, cost)
    rnd = fixed_hold_backtest(random_decisions(tp, seed=cfg["random_seed"]), tp, N, cost)

    print(f"directional accuracy — price {price_da:.3f}, VADER {vader_da:.3f}, "
          f"combined {comb_da:.3f}")

    da = {"combined_VADER+LLM": comb_da, "vader_only": vader_da, "price_only": price_da}
    tbl = pd.DataFrame([
        comb_res.as_row("combined_VADER+LLM"),
        vader_res.as_row("vader_only"),
        price_res.as_row("price_only"),
        bnh.as_row("buy_and_hold"),
        ab.as_row("always_buy"),
        rnd.as_row("random"),
    ]).set_index("strategy")
    tbl["avg_net_return"] = tbl["avg_net_return"].round(5)
    tbl["hit_rate"] = tbl["hit_rate"].round(3)
    tbl["cumulative_net_return"] = tbl["cumulative_net_return"].round(4)
    tbl["directional_acc"] = [round(da.get(s, float("nan")), 3) for s in tbl.index]
    print("\n" + tbl.to_string())
    tbl.to_csv("results/phase7_comparison.csv")

    fig, ax = plt.subplots(figsize=(10, 6))
    for name, res, st in [("Combined (VADER+LLM)", comb_res, dict(lw=2.4, color="#9467bd")),
                          ("VADER only", vader_res, dict(lw=2.0, color="#d62728")),
                          ("Price-only", price_res, dict(lw=2.0, color="#1f77b4")),
                          ("Buy & hold", bnh, dict(lw=1.6, color="#2ca02c", ls="--"))]:
        ax.plot(ti, daily_equity(res, ti) * 100, label=name, **st)
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"Phase 7 — LLM+VADER fusion vs VADER vs price-only "
                 f"({ti[0].date()} → {ti[-1].date()})")
    ax.set_ylabel("Cumulative net return (%)"); ax.set_xlabel("Date")
    ax.legend(loc="lower left"); ax.grid(alpha=0.3)
    fig.autofmt_xdate(); fig.tight_layout(); fig.savefig(FIG, dpi=150)
    print(f"\nSaved -> {FIG}, results/phase7_comparison.csv")

    better_than_vader = comb_res.cumulative_net_return > vader_res.cumulative_net_return
    better_than_price = comb_res.cumulative_net_return > price_res.cumulative_net_return
    print(f"\nCombined vs VADER: {'better' if better_than_vader else 'NOT better'}; "
          f"combined vs price-only: {'beats' if better_than_price else 'does NOT beat'}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
