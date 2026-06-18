"""Phase 3 — pre-build validation gate.

Hard gates (must pass): target is a return; leak-free split is wired.
Informational (recorded either way): ADF stationarity; Granger causality of each
sentiment feature on the return, run on the TRAIN slice only so the test set stays
unseen. A weak/null Granger result is a legitimate finding — it sets honest
expectations, it does NOT stop the project.

Run:  uv run python check_phase3.py
"""
from __future__ import annotations

import sys

import pandas as pd

from src.config import load_config
from src.features import SENTIMENT_FEATURES, TARGET
from src.splits import chronological_split
from src.validate import (
    adf_test,
    check_leak_free_split,
    check_target_is_return,
    granger_min_pvalue,
)

OK, FAIL, INFO = "[ OK ]", "[FAIL]", "[info]"
MAXLAG = 5


def main() -> int:
    cfg = load_config()
    table = pd.read_parquet(cfg["data"]["processed_path"])
    train_frac = cfg["split"]["train_frac"]
    print(f"Modeling table: {len(table)} bars "
          f"{table.index.min().date()} -> {table.index.max().date()}\n")

    failures = 0

    # ---- Hard gate 1: target is a return -----------------------------------
    ok, msg = check_target_is_return(table)
    print(f"{OK if ok else FAIL} GATE 1 — {msg}")
    failures += 0 if ok else 1

    # ---- Hard gate 2: leak-free split wired --------------------------------
    ok, msg = check_leak_free_split(table, train_frac)
    print(f"{OK if ok else FAIL} GATE 2 — {msg}")
    failures += 0 if ok else 1

    if failures:
        print(f"\n{FAIL} Hard gate failed — fix before proceeding (pathway Phase 3).")
        return 1

    # Everything below is informational and uses the TRAIN slice only.
    split = chronological_split(table.index, train_frac)
    train = table.loc[split.train_index]
    rows = []

    # ---- Informational 3: ADF stationarity ---------------------------------
    print(f"\n{INFO} ADF stationarity (train slice; stationary = reject unit root, p<0.05):")
    adf_target = adf_test(train[TARGET])
    print(f"    {TARGET:14s} p={adf_target['pvalue']:.4f} "
          f"-> {'stationary' if adf_target['stationary'] else 'NON-stationary'}")
    rows.append({"check": "adf", "series": TARGET, **adf_target})

    nonstationary = set()
    for f in SENTIMENT_FEATURES:
        r = adf_test(train[f])
        if not r["stationary"]:
            nonstationary.add(f)
        print(f"    {f:14s} p={r['pvalue']:.4f} "
              f"-> {'stationary' if r['stationary'] else 'NON-stationary (will difference for Granger)'}")
        rows.append({"check": "adf", "series": f, **r})

    # ---- Informational 4: Granger causality --------------------------------
    print(f"\n{INFO} Granger causality: does lagged sentiment predict the return "
          f"beyond its own past? (train slice, maxlag={MAXLAG})")
    any_signal = False
    for f in SENTIMENT_FEATURES:
        diff = f in nonstationary
        g = granger_min_pvalue(train[TARGET], train[f], maxlag=MAXLAG, difference_feature=diff)
        sig = g["min_pvalue"] < 0.05
        any_signal = any_signal or sig
        tag = "<-- predictive (p<0.05)" if sig else "no significant lead"
        print(f"    {f:14s} min p={g['min_pvalue']:.4f} @ lag {g['best_lag']}"
              f"{' (differenced)' if diff else ''}  {tag}")
        rows.append({"check": "granger", "series": f, "min_pvalue": g["min_pvalue"],
                     "best_lag": g["best_lag"], "differenced": diff})

    # Save for the writeup (Phase 6).
    out = "results/phase3_validation.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    print(f"\nSaved validation results -> {out}")

    print()
    print(f"{OK} GATES PASSED. Granger verdict: "
          + ("sentiment shows a lead on the return — worth testing."
             if any_signal else
             "no significant sentiment lead — expect a weak/null sentiment result "
             "(a legitimate outcome; the experiment still proceeds)."))
    return 0


if __name__ == "__main__":
    sys.exit(main())
