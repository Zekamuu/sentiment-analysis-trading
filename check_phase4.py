"""Phase 4 — price-only ARIMA baseline (the control group).

Fit ARIMA on the training-slice returns, forecast the test slice chronologically
(leak-free), threshold into buy/sell/no-trade (threshold tuned on TRAIN only), run
through the Phase-1 backtest, and report every metric next to the baselines. These
numbers are the bar sentiment must clear in Phase 5.

Run:  uv run python check_phase4.py
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
from src.features import TARGET
from src.model import decision_rule, one_step_forecasts, select_order, tune_threshold
from src.splits import chronological_split


def main() -> int:
    cfg = load_config()
    N = cfg["holding_period_N"]
    cost = cfg["transaction_cost"]
    table = pd.read_parquet(cfg["data"]["processed_path"])
    close = load_ohlcv(cfg["data"]["ohlcv_path"])["close"]

    endog = table[TARGET]  # r_{t+1} indexed by decision bar t
    split = chronological_split(table.index, cfg["split"]["train_frac"])
    n_train = len(split.train_index)
    print(f"Target series: {len(endog)} bars | train {n_train} / test {len(split.test_index)} "
          f"| cutoff {split.cutoff.date()} | N={N}, cost={cost}")

    # --- Fit + chronological one-step forecasts (params from train only) -------
    order = select_order(endog.iloc[:n_train])
    print(f"Selected ARIMA order (p,d,q) = {order} by AIC on train returns")
    forecasts = one_step_forecasts(endog, n_train, order)

    # --- Tune threshold on TRAIN, apply to TEST ------------------------------
    train_prices = close.loc[split.train_index]            # no buffer => no test price used
    thr = tune_threshold(forecasts.loc[split.train_index], train_prices, N, cost)
    print(f"Tuned decision threshold (train only) = {thr:.5f}")

    # Test prices need N extra bars so the last decision can exit.
    test_prices = close.loc[split.test_index[0]: split.test_index[-1] + pd.Timedelta(days=N + 3)]
    test_fc = forecasts.loc[split.test_index]
    model_dec = decision_rule(test_fc, thr)
    model_res = fixed_hold_backtest(model_dec, test_prices, N, cost)

    # Directional accuracy on the test slice (sign of forecast vs realized return).
    actual = table[TARGET].loc[split.test_index]
    traded = model_dec != 0
    dir_acc = float((np.sign(test_fc[traded]) == np.sign(actual[traded])).mean()) if traded.any() else float("nan")

    # --- Baselines on the same test slice (Design §5.2) ----------------------
    bnh = buy_and_hold(close.loc[split.test_index], cost)
    ab = fixed_hold_backtest(always_buy(test_prices), test_prices, N, cost)
    rnd = fixed_hold_backtest(random_decisions(test_prices, seed=cfg["random_seed"]),
                              test_prices, N, cost)

    rows = [
        model_res.as_row("price_only_ARIMA"),
        bnh.as_row("buy_and_hold"),
        ab.as_row("always_buy"),
        rnd.as_row("random"),
    ]
    tbl = pd.DataFrame(rows).set_index("strategy")
    tbl["avg_net_return"] = tbl["avg_net_return"].round(5)
    tbl["hit_rate"] = tbl["hit_rate"].round(3)
    tbl["cumulative_net_return"] = tbl["cumulative_net_return"].round(4)

    print(f"\nTest slice {split.test_index[0].date()} -> {split.test_index[-1].date()} "
          f"({len(split.test_index)} bars)")
    print(f"Price-only directional accuracy (traded bars): {dir_acc:.3f}\n")
    print(tbl.to_string())

    out = "results/phase4_price_only.csv"
    tbl.to_csv(out)
    # Save the model's test equity curve + decisions for the Phase 6 plots.
    pd.DataFrame({"forecast": test_fc, "decision": model_dec,
                  "actual_fwd_return": actual}).to_csv("results/phase4_model_test.csv")
    print(f"\nSaved metrics -> {out} (control group for Phase 5)")

    beats = (model_res.cumulative_net_return > bnh.cumulative_net_return,
             model_res.cumulative_net_return > rnd.cumulative_net_return)
    print(f"\nPrice-only vs buy&hold: {'beats' if beats[0] else 'does NOT beat'}; "
          f"vs random: {'beats' if beats[1] else 'does NOT beat'}.")
    print("This is the control. Phase 5 adds sentiment and changes nothing else.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
