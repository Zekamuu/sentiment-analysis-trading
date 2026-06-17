"""Phase 1 checkpoint verifier — prove the evaluation harness is HONEST.

Runs four things and asserts the critical properties:
  1. Chronological split (70/30) on the BTC daily closes over the tweet window.
  2. Leak-free scaler: fit on train only, and a positive leakage assertion.
  3. Backtest mechanics on synthetic prices (constant / trending) — sanity.
  4. THE honesty test: a random decision signal, over many seeds, must be
     break-even-or-negative after costs and must not systematically beat
     buy-and-hold. If it looks profitable, the harness has a look-ahead bug.

Run:  uv run python check_phase1.py
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from src.backtest import (
    BUY,
    SELL,
    always_buy,
    buy_and_hold,
    fixed_hold_backtest,
    random_decisions,
)
from src.config import load_config
from src.data import load_ohlcv
from src.splits import TrainScaler, chronological_split

OK, FAIL = "[ OK ]", "[FAIL]"


def main() -> int:
    cfg = load_config()
    N = cfg["holding_period_N"]
    cost = cfg["transaction_cost"]
    train_frac = cfg["split"]["train_frac"]
    failures = 0

    # ---- Real price universe: BTC daily close over the tweet window ---------
    ohlcv = load_ohlcv(cfg["data"]["ohlcv_path"])
    window = ohlcv.loc["2022-06-08":"2023-05-28"]
    close = window["close"]
    print(f"Price universe: {len(close)} daily bars "
          f"{close.index.min().date()} -> {close.index.max().date()}  (N={N}, cost={cost})")

    # ---- 1. Chronological split --------------------------------------------
    split = chronological_split(close.index, train_frac)
    contiguous = split.train_index.max() < split.cutoff <= split.test_index.min()
    no_overlap = len(split.train_index.intersection(split.test_index)) == 0
    if contiguous and no_overlap:
        print(f"{OK} chronological split: {len(split.train_index)} train / "
              f"{len(split.test_index)} test, cutoff {split.cutoff.date()}, no overlap")
    else:
        print(f"{FAIL} split is not a clean earlier/later partition"); failures += 1

    # ---- 2. Leak-free scaler -----------------------------------------------
    feat = close.to_frame("close")
    scaler = TrainScaler().fit(feat.loc[split.train_index])
    train_mean = feat.loc[split.train_index].mean().iloc[0]
    if np.isclose(scaler.mean_.iloc[0], train_mean):
        print(f"{OK} scaler fit on TRAIN only (mean={train_mean:.2f} ignores test data)")
    else:
        print(f"{FAIL} scaler mean does not match train slice"); failures += 1
    try:
        scaler.assert_fit_was_leak_free(split.test_index)
        print(f"{OK} leakage assertion: no test timestamp in fitted stats")
    except AssertionError as e:
        print(f"{FAIL} {e}"); failures += 1

    # ---- 3. Backtest mechanics on synthetic prices -------------------------
    flat = pd.Series(100.0, index=pd.RangeIndex(50))
    r_flat = fixed_hold_backtest(always_buy(flat), flat, N, cost)
    if np.isclose(r_flat.avg_net_return, -cost) and r_flat.n_trades == len(flat) - N:
        print(f"{OK} flat market: avg net = -cost = {r_flat.avg_net_return:+.4f}, "
              f"{r_flat.n_trades} trades (last {N} bar(s) excluded → no look-ahead)")
    else:
        print(f"{FAIL} flat-market mechanics wrong: {r_flat}"); failures += 1

    up = pd.Series(100.0 * (1.01 ** np.arange(50)), index=pd.RangeIndex(50))
    r_long = fixed_hold_backtest(always_buy(up), up, N, cost)
    r_short = fixed_hold_backtest(pd.Series(SELL, index=up.index), up, N, cost)
    if r_long.avg_net_return > 0 > r_short.avg_net_return:
        print(f"{OK} trending market: long profits ({r_long.avg_net_return:+.4f}), "
              f"short loses ({r_short.avg_net_return:+.4f}) — directions correct")
    else:
        print(f"{FAIL} long/short directions wrong"); failures += 1

    # ---- 4. THE honesty test: random signal on the real test slice ----------
    test_px = close.loc[split.test_index]
    bnh = buy_and_hold(test_px, cost)

    seeds = range(1000)
    avg_nets, hit_rates, cum_rets, beats = [], [], [], 0
    for s in seeds:
        res = fixed_hold_backtest(random_decisions(test_px, seed=s), test_px, N, cost)
        avg_nets.append(res.avg_net_return)
        hit_rates.append(res.hit_rate)
        cum_rets.append(res.cumulative_net_return)
        beats += int(res.cumulative_net_return > bnh.cumulative_net_return)

    mean_avg = float(np.mean(avg_nets))
    mean_hit = float(np.mean(hit_rates))
    median_cum = float(np.median(cum_rets))
    beat_frac = beats / len(seeds)

    print()
    print(f"Buy-and-hold over test slice: cumulative net = {bnh.cumulative_net_return:+.4f}")
    print(f"Random signal over {len(seeds)} seeds:")
    print(f"    mean avg net return / trade = {mean_avg:+.5f}  (cost = {cost})")
    print(f"    mean hit rate               = {mean_hit:.3f}")
    print(f"    median cumulative net       = {median_cum:+.4f}")
    print(f"    fraction beating buy&hold   = {beat_frac:.3f}")

    # Critical assertions.
    if mean_avg < 0:
        print(f"{OK} random is net-NEGATIVE after costs (≈ -cost) — no free money")
    else:
        print(f"{FAIL} random is profitable after costs → LOOK-AHEAD BUG"); failures += 1
    if 0.45 <= mean_hit <= 0.55:
        print(f"{OK} random hit rate ≈ 0.5 — signal is genuinely uninformative")
    else:
        print(f"{FAIL} random hit rate {mean_hit:.3f} is not ≈ 0.5"); failures += 1
    if beat_frac < 0.5:
        print(f"{OK} random does not systematically beat buy&hold ({beat_frac:.1%})")
    else:
        print(f"{FAIL} random beats buy&hold too often ({beat_frac:.1%})"); failures += 1

    print()
    if failures:
        print(f"{FAIL} Phase 1 NOT complete: {failures} check(s) failed — the harness "
              f"is not trustworthy yet.")
        return 1
    print(f"{OK} Phase 1 checkpoint passed — the backtest is honest. Safe to build "
          f"models on top of it.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
