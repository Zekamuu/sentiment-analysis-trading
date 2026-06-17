# Build Pathway & Implementation Plan

**Companion to:** the System Design Sheet (v2). This document is the *how and in what order*; the design sheet is the *what and why*. Section references (e.g. "Design §5") point to that document.

**Guiding principle:** build the measuring stick before the thing it measures. You will construct and validate the evaluation harness — with a *random* signal — before you train any model. That way a good-looking result can't fool you, because the harness was proven honest before the model existed. This is the single habit that separates this project from the one with the suspiciously perfect chart.

**Overall arc:** get the dumbest possible version running end-to-end first (random signal → backtest → number), then add real features, then a price-only model, then sentiment, then optional stretch goals. Each phase ends with a concrete checkpoint you must pass before moving on.

---

## Phase 0 — Setup & Decisions

**Goal:** a clean workspace and three decisions locked.

Steps:
1. Set up a repository and a Python environment. Pin versions. Suggested core deps: `pandas`, `numpy`, `statsmodels`, `pmdarima`, `matplotlib`, `vaderSentiment`. (Add `arch` only if you attempt the optional GARCH filter.)
2. Lock three parameters and write them in a config file so they live in one place:
   - **Asset** — start with BTC.
   - **Bar frequency** — whatever your data supports (1-minute is fine).
   - **Holding period N** — how many bars a trade stays open (e.g. 10).
3. Acquire data. You already have a merged minute-level sentiment+price CSV from the reference project; you can reuse it as your dataset, or rebuild your own from a tweets dataset + OHLCV. Either is fine — just record exactly which file and time range you used.

**Checkpoint:** repo runs, dependencies import, config file holds asset/frequency/N, and you can load the dataset into a DataFrame indexed by a UTC timestamp.

---

## Phase 1 — The Evaluation Harness (build this FIRST)

**Goal:** a working, honest backtest you trust — tested against a known-nonsense signal — before any model exists.

Steps:
1. **Chronological split utility.** A function that splits the data by time into train / test (e.g. first 70% / last 30%). Never random. Return the split indices, not shuffled rows.
2. **Leak-free scaling utility.** Fit any normalization on the *training* slice only; apply those fixed parameters to the test slice. Write it so it's impossible to accidentally fit on the full series.
3. **Fixed-holding-period backtest** (Design §5.1). Input: a series of decisions (buy / sell / no-trade) and the price series. For each `buy` at bar *t*, compute the return from price[t] to price[t+N], subtract a fixed transaction cost, and collect it. Output: average net return per trade, hit rate, trade count, cumulative net return.
4. **Baselines** (Design §5.2): buy-and-hold, always-buy, and random buy/sell. Implement them as decision series your same backtest can consume.
5. **Transaction cost** (Design §5.3): a single constant (e.g. a small % per trade), read from config.

**Checkpoint (critical):** feed the backtest a *random* decision signal. After costs, it should hover around zero or negative, and it should **not** beat buy-and-hold except by chance. If a random signal looks profitable, your harness has a bug (often a look-ahead in how entry/exit prices are indexed) — fix it now, because every later result depends on this being honest.

---

## Phase 2 — Feature Engineering

**Goal:** a single modeling table, correctly time-aligned, with the target defined as a return.

Steps:
1. **Per-bar sentiment features (VADER).** Score each post, aggregate to the bar: mean sentiment, plus dispersion (variance) and post volume. (Design §4.4–4.5.) Keep it to VADER for now — the LLM branch is a stretch goal.
2. **Price features.** Compute log returns from close (Design §4.6). Optionally add realized volatility and volume change.
3. **Define the target.** `target = next-period return r_{t+1}`. This is the non-negotiable choice — never the price level (Design §1, rule 1).
4. **Lag the features.** Shift every feature so the row used to predict bar *t* contains only information through *t−1* (Design §4.8). Verify a "last-hour" sentiment window for bar *t* includes no post timestamped at or after *t*.
5. **Assemble & join** sentiment + price features onto one timestamp index; handle empty bars explicitly (Design §4.7).

**Checkpoint:** one DataFrame where each row has lagged features and a next-period-return target, no NaNs in the modeling columns, and a spot-check confirming row *t*'s features predate *t*.

---

## Phase 3 — Pre-Build Validation Gate

**Goal:** confirm the experiment is sound and worth running (Design §6). These are cheap and make excellent writeup material.

Steps:
1. **Confirm target = return** (not price). One assertion.
2. **Confirm leak-free split** is wired into the workflow.
3. **Stationarity (ADF test)** on the return target and on the sentiment features; difference/normalize anything non-stationary (Design §4.6).
4. **Granger causality**: does lagged sentiment help predict the return beyond past returns alone? Use `statsmodels` `grangercausalitytests`.

**Checkpoint:** steps 1–2 pass (hard gate — fix before proceeding). Step 4 is informational: record the result whichever way it goes. A weak or reverse-causal result just sets honest expectations; it does not stop the project.

---

## Phase 4 — Price-Only Baseline Model

**Goal:** an ARIMAX (the "X" empty/price-only) forecasting the return, run through your harness. This is the bar that sentiment must clear.

Steps:
1. Fit ARIMA on the training-slice returns; select (p, d, q) with ACF/PACF + AIC/BIC, `d` likely 0 (Design §4.9).
2. Forecast returns across the test slice (chronologically — refit or roll forward, never peek ahead).
3. Apply the decision rule: threshold the forecasted return into buy / sell / no-trade (Design §4.10). Tune the threshold on the training slice only.
4. Run the result through the Phase-1 backtest. Record every metric vs. the baselines.

**Checkpoint:** a full set of numbers for the price-only model. Save them — this is your control group.

---

## Phase 5 — Add Sentiment (the actual experiment)

**Goal:** ARIMAX with sentiment as exogenous regressors; measure whether sentiment adds anything.

Steps:
1. Add the lagged sentiment features as the exogenous `X` in ARIMAX. Keep the regressor count modest (Design §4.7) to avoid overfitting.
2. Re-run the identical forecast → decision → backtest pipeline. Change nothing else, so the only difference vs. Phase 4 is the sentiment features.
3. **Compare** the sentiment model against (a) the price-only model from Phase 4 and (b) the Phase-1 baselines, on net return per trade, hit rate, and cumulative net return.

**Checkpoint:** a clean comparison table. If sentiment beats price-only and the baselines after costs — great, you found a signal. If it doesn't — that is a legitimate, complete result (Design §10, "honest expectation"). Either way you have a finished, defensible project.

---

## Phase 6 — Analysis & Writeup

**Goal:** turn results into a clear story.

Steps:
1. Plot cumulative net return of each strategy on one chart (your model, price-only, buy-and-hold, random). Plot **returns/PnL**, not predicted-vs-true price levels.
2. Report the metrics table and the Granger result.
3. Write the limitations section straight from Design §10 (bots/manipulation, single asset, no position sizing, directional prediction is hard). Naming these earns credit.
4. State the conclusion honestly, tied to the numbers and the baselines.

**Checkpoint:** someone reading only your writeup can tell what you built, how you tested it, and whether sentiment helped — without seeing the code.

---

## Phase 7 — Stretch Goals (only after Phases 0–6 are done)

Tackle in roughly this order, each as an isolated experiment compared back to the Phase-5 result:
- **LLM feature branch** (Design §4.3): replace/augment VADER with a finance-tuned transformer's sentiment output; reduce dimensionality before feeding ARIMAX.
- **GARCH volatility filter** (Design §4.9): skip trades when forecasted volatility is extreme; report whether it improves net results.
- **Sell/short side**: extend the decision rule beyond long-only.
- **Second asset** (e.g. ETH): re-run the whole pipeline; note survivorship caveats if you generalize further.

---

## Suggested Repository Structure

```
crypto-sentiment/
├── config.yaml              # asset, frequency, N, transaction cost, threshold, data path
├── data/                    # raw + processed (do not commit large raw files)
├── src/
│   ├── features.py          # sentiment aggregation, price features, lagging, target
│   ├── splits.py            # chronological split + leak-free scaling
│   ├── backtest.py          # fixed-hold backtest + baselines
│   ├── model.py             # ARIMAX fit / forecast / decision rule
│   └── validate.py          # ADF, Granger, leakage assertions
├── notebooks/               # exploration + final analysis/plots
├── results/                 # metrics tables, figures
└── README.md
```

---

## One-Page Sequence Checklist

- [ ] Phase 0 — env, config (asset / frequency / N), data loaded
- [ ] Phase 1 — chronological split, leak-free scaler, backtest + baselines; **random signal ≈ break-even after costs**
- [ ] Phase 2 — modeling table with lagged features and a **return** target, no leakage
- [ ] Phase 3 — target=return ✓, leak-free split ✓, ADF done, Granger recorded
- [ ] Phase 4 — price-only ARIMAX numbers saved (the control)
- [ ] Phase 5 — sentiment ARIMAX compared to price-only + baselines after costs
- [ ] Phase 6 — PnL/returns plots, metrics table, honest conclusion, limitations
- [ ] Phase 7 — optional: LLM features, GARCH filter, short side, second asset

**If you do nothing else right, get these two right:** the target is a return (not a price), and the evaluation is leak-free and out-of-sample. Everything else is refinement.
