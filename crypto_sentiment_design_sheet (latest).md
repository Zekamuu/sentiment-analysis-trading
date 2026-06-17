# Crypto Sentiment-Driven Trading Signal — System Design Sheet

**Project:** Sentiment analysis pipeline for cryptocurrency trading decisions
**Architecture type:** Feature pipeline feeding a time-series return forecaster that produces a buy/sell signal, evaluated with a fixed-holding-period backtest
**Status:** Design draft (v2 — scoped for an undergraduate project)
**Scope note:** Goal is a working, honestly-evaluated end-to-end prototype on historical data, not a production trading system. Position sizing, portfolio control, and live execution are explicitly out of scope (see Section 10).

---

## 1. Objective

Build an end-to-end pipeline that ingests social-media chatter and market price data, converts unstructured text into a quantified sentiment signal, fuses it with engineered price features, and feeds the result into an **ARIMAX** model that forecasts the **next-period return**. The forecasted return is turned into a **buy / sell (or no-trade)** decision, and the decision is evaluated with a **fixed-holding-period backtest** (open on the signal, close a fixed number of bars later, e.g. 10 minutes).

The core hypothesis: aggregate social sentiment carries short-horizon predictive information about crypto returns that improves a forecast over price history alone.

**Two design rules are non-negotiable — the project is meaningless without them:**

1. **Forecast returns, not price levels.** Predicting the *price* at short horizons is a trap: consecutive prices are nearly identical, so any model trivially scores well by echoing the last price (this is exactly why a naive "prediction vs. true price" chart can look near-perfect while containing no real signal). The target must be the next-period **return**, which removes that persistence and forces the model to predict something real.
2. **Evaluate leak-free and out-of-sample.** Train on an earlier time slice, test on a strictly later one (never shuffle time). Fit any scaler/normalizer on the training slice **only**. Ensure features for predicting bar *t* use information through *t−1* only. See Sections 4.8 and 5.

Everything else in this document is either a core component or an optional enhancement; these two rules are the pass/fail gate.

---

## 2. High-Level Architecture

The system has **two parallel feeds** that converge before modeling:

- **Text feed** — social media → ingestion → cleanup → (tokenize + LLM features) **and** (lexicon sentiment) → time aggregation
- **Market feed** — finance API (OHLCV) → stationarity transform (log returns)

These merge at **Combine**, then flow through feature lagging → ARIMAX (return forecast) → decision rule → buy/sell, with a fixed-holding-period backtest measuring effectiveness.

```
Social media ──► Ingestion ──► Cleanup ──┬──► Tokenization ──► LLM feature extraction ──┐
                                          │                                              ├──► Time aggregation ──┐
                                          └──► Sentiment classifier (VADER) ─────────────┘                       │
                                                                                                                 ├──► Combine ──► Feature lagging ──► ARIMAX ──► Decision rule ──► Buy/Sell ──► Fixed-hold backtest
   Finance API (OHLCV) ──► Stationarity transform (log returns) ──────────────────────────────────────────────┘                (forecast return)   (threshold)                  (open → close +N bars)
```

---

## 3. Data Sources

| Source | Content | Cadence | Notes |
|---|---|---|---|
| Social media (e.g. X/Twitter) | Posts, author metadata (follower count, verified status), timestamps | Streaming / polled | Need author follower counts for influence weighting; collect ticker/cashtag and keyword matches ($BTC, "bitcoin", etc.) |
| Market data API | Open, High, Low, Close, Volume (OHLCV) | Per-bar (e.g. 1m / 1h / 1d) | Crypto trades 24/7 — no market-close gaps; align bar boundaries to UTC |

**Design note:** keep the two feeds on a shared, explicit time index from the start. Misalignment between sentiment timestamps and price bars is the most common source of leakage and silent bugs downstream.

---

## 4. Pipeline Components

### 4.1 Data Ingestion
- **Purpose:** Gather all raw social posts into a single landing zone.
- **Input:** Streaming API / firehose / scraped posts.
- **Output:** Raw, immutable records (post text, author, follower count, timestamp, source).
- **Implementation notes:**
  - Land raw data first, transform later (raw → bronze → silver layers). Keep the untouched raw copy so you can reprocess when cleaning/feature logic changes.
  - Capture follower count *at time of post* — it's needed later for influence weighting and changes over time.
  - Deduplicate on post ID; retweets/reposts should be flagged, not silently merged.

### 4.2 Data Cleanup
- **Purpose:** Normalize text into a model-ready form.
- **Typical operations:** strip URLs, handle emojis/cashtags, normalize casing where appropriate, drop bots/spam, language-filter, remove duplicates and near-duplicates.
- **Design decisions:**
  - **Do not over-clean.** VADER (see 4.4) is tuned for social-media style — it *uses* capitalization, punctuation emphasis ("!!!"), and emoji as signal. Aggressive lowercasing/punctuation stripping destroys information it relies on. Consider maintaining two cleaned variants: a light-clean stream for the lexicon model and a heavier-clean stream for the LLM/tokenizer path.
  - Bot/spam filtering matters a lot here — coordinated shilling can dominate raw sentiment.

### 4.3 Branch A — Tokenization → LLM Feature Extraction

**Tokenization (e.g. byte-pair encoding)**
- **Purpose:** Convert cleaned text into subword tokens the LLM expects.
- **Notes:** Use the tokenizer that ships with the chosen feature-extraction model — BPE/SentencePiece vocab must match the model, or embeddings are meaningless. Watch token-length truncation limits.

**LLM feature extraction layer**
- **Purpose:** Produce dense semantic features (embeddings or model-derived sentiment/aspect scores) that capture nuance a lexicon misses — sarcasm, context, multi-token entities, slang.
- **Output options:**
  - Pooled embedding vector per post (e.g. last-hidden-state mean-pool), or
  - A fine-tuned classifier head emitting a sentiment/probability per post (e.g. a crypto/finance-tuned transformer such as a FinBERT-style model).
- **Design decisions:**
  - Embeddings give richness but add dimensionality — you'll need to reduce (PCA / a small learned projection) before feeding a time-series model, which expects a modest number of exogenous regressors.
  - This path is the expensive one (GPU/cost). Decide batch vs. real-time inference based on latency budget.

### 4.4 Branch B — Sentiment Classification (lexicon)
- **Purpose:** Fast, cheap, interpretable per-post sentiment score.
- **Example model:** VADER — rule/lexicon-based, purpose-built for social media.
- **Output:** A scalar per post, e.g. compound score in [-1, +1].
  - *Example from the diagram:* a post like "BTC looking very strong today" → ~ +0.90.
- **Why keep both A and B:** they're complementary. VADER is transparent, instant, and a strong baseline; the LLM path captures context VADER misses. Running both lets you (a) sanity-check, (b) feed both as features, and (c) fall back if the LLM path is down.
- **Caveat:** general-purpose lexicons mis-handle crypto slang ("rekt", "diamond hands", "to the moon", "rug"). Consider extending the VADER lexicon with a domain dictionary.

### 4.5 Time Aggregation (influence-weighted)
- **Purpose:** Collapse many per-post scores into one sentiment value per time bar, aligned to the price bars.
- **Operations:** resample to the model's bar frequency (e.g. hourly), then aggregate post-level scores into a bar-level feature.
- **Influence weighting (key design choice):** weight each post by author influence (follower count is the simplest proxy) so high-reach accounts move the aggregate more than a 3-follower account:

  ```
  bar_sentiment = Σ (wᵢ · sᵢ) / Σ wᵢ
      where sᵢ = post sentiment, wᵢ = influence weight (e.g. log(1 + followers))
  ```
- **Design decisions:**
  - Use **log-scaled** weights — raw follower counts are extremely heavy-tailed; one mega-account would otherwise drown out the crowd.
  - Emit more than one feature per bar: weighted mean sentiment, **post volume** (chatter count itself is predictive), sentiment dispersion/variance, and share of positive vs. negative. Volume spikes often matter as much as polarity.
  - Decide handling of empty bars (no posts) — forward-fill, zero, or NaN-and-impute. This choice interacts with stationarity and the model.
- **Scope note:** the basic log-follower weighting above is fine to include as a feature. More elaborate influence modeling and manipulation-aware weighting are out of scope for this project (Section 10) — a simple weighted mean plus post-volume and dispersion features is enough.

### 4.6 Market Branch — Stationarity Transform (log returns)
- **Purpose:** ARIMA-family models assume (weak) stationarity. Raw price is non-stationary (trending, unit root); the log-return transform yields a roughly stationary series.

  ```
  rₜ = ln(Pₜ) − ln(Pₜ₋₁)
  ```
- **Implementation notes:**
  - Verify stationarity with an **Augmented Dickey-Fuller** (or KPSS) test rather than assuming it; difference further only if needed (this is effectively the "I" / `d` term in ARIMA).
  - Engineer companion features from OHLCV: realized volatility, volume change, high-low range. These can also enter ARIMAX as exogenous regressors.

### 4.7 Combine (feature fusion)
- **Purpose:** Join the aggregated sentiment features and the market features onto a single aligned time index → the modeling table.
- **Output schema (per bar t):** features `[log_return_t, volatility_t, volume_t, weighted_sentiment_t, post_volume_t, sentiment_dispersion_t, llm_features..., ...]`, plus the **prediction target = next-period return `r_{t+1}`** (the label the model learns to forecast).
- **Design decisions:**
  - Inner-join on the time index; be explicit about timezone (UTC) and bar-boundary convention.
  - Scale/normalize exogenous regressors as the model requires.
  - Watch dimensionality — too many exogenous regressors relative to sample size overfits ARIMAX badly.

### 4.8 Feature Lagging (look-ahead-bias prevention)
- **Purpose:** Guarantee the model only uses information available **strictly before** the bar it predicts. As noted in the diagram: prevent the model from seeing day *t*; only use data up to *t−1*.
- **Why it's critical:** the single most common (and fatal) mistake in financial ML is leakage — letting a feature contain information from the future. A backtest with leakage looks spectacular and loses money live.
- **Implementation notes:**
  - Shift all exogenous features by ≥1 bar so predicting return at *t* uses sentiment/price features through *t−1*.
  - Be careful that aggregation windows don't straddle the boundary (a "last hour" sentiment feature for bar *t* must not include any post timestamped in *t*).
  - Enforce this in code, not by convention — a single util that builds lagged frames, tested.

### 4.9 ARIMAX Model
- **Purpose:** Forecast the **next-period return** using the autoregressive structure of returns **plus** exogenous regressors (the "X") — the sentiment and market features. The forecasted return is the quantity the decision rule acts on.
- **Form:** ARIMA(p, d, q) with exogenous inputs Xₜ. Features are the exogenous variables; `d` is typically 0 because the target is already a (stationary) return, not a price.
- **Design decisions:**
  - Target is the return, never the price level (see Section 1, rule 1).
  - Select (p, d, q) via ACF/PACF + information criteria (AIC/BIC); check residuals are roughly white noise.
  - Validate with a **chronological train/test split** (or walk-forward), never a random shuffle — time order must be preserved.
  - ARIMAX is linear and interpretable, which is ideal for a first project; nonlinear models (gradient-boosted trees, small sequence nets) are a later iteration, not a starting point.

- **Why ARIMAX and not GARCH-X here:** the two model different things. **ARIMAX models the conditional *mean* return** — i.e. how much and in which *direction* price is expected to move — which is exactly what a buy/sell decision needs. **GARCH-X models conditional *variance*** — how *big* the next move will be, regardless of direction — so it cannot, on its own, produce a directional trade signal. The research literature suggests the *variance* is the more predictable target (sentiment tends to forecast volatility better than direction), so a volatility project would prefer GARCH-X; but because this project's deliverable is a directional buy/sell call, ARIMAX is the correct fit. Practical consequence: **expect the directional signal to be weak.** That is an expected, honest result for this problem — reporting it clearly (with the baselines in Section 5) is a successful outcome, not a failure. Optionally, a GARCH-style volatility estimate can be added later as a *secondary* filter (skip trading when expected volatility is extreme), but that is an enhancement, not part of the core.

### 4.10 Decision Rule
- **Purpose:** Convert the forecasted return into an action. Deliberately simple for this scope.
- **Rule:** if forecasted next-period return > +threshold → **buy**; if < −threshold → **sell** (or, for a long-only simplification, just **no-trade**); otherwise **no-trade**. The threshold is a single tunable parameter (tune it on the training slice only).
- **Design decisions:**
  - Long-only (buy vs. no-trade) is the simplest valid version and avoids the complications of shorting; add the sell/short side only if you want to.
  - A nonzero threshold matters: forecasts near zero are noise, and trading on them just pays fees. The threshold is effectively the project's one risk control.

### 4.11 Buy/Sell Decision (output)
- **Output:** A per-bar action (buy / sell / no-trade) plus the forecasted return that produced it, logged for evaluation.
- **Downstream:** feeds the fixed-holding-period backtest in Section 5. No live execution in this project.

---

## 5. Evaluation Methodology

This is where the project earns its credibility. A result is only meaningful if the evaluation is leak-free, out-of-sample, cost-aware, and compared against dumb baselines.

### 5.1 Fixed-Holding-Period Backtest

The core evaluation, deliberately simple:

1. Run the model across the **test slice** (a strictly later time period than training).
2. Every time the decision rule fires **buy**, record the return from the entry price at bar *t* to the price *N* bars later (e.g. *N* = 10 minutes); close the trade then regardless of anything else. (For a **sell/short**, the trade profits from a price fall over the same window.)
3. Subtract a **transaction cost** per trade (see 5.3).
4. Aggregate across all trades: average net return per trade, **hit rate** (% of trades profitable after costs), and cumulative net return.

This needs no portfolio logic, position sizing, or stop-losses — one trade at a time, fixed exit. Roughly 20 lines of code.

### 5.2 Baselines (mandatory)

A strategy number means nothing in isolation. Report it next to:
- **Buy-and-hold** — would you have done better just holding the asset over the test period?
- **Always-trade / coin-flip** — entering on every bar (or a random buy/sell each bar), same fixed-hold exit and costs. Isolates whether the *signal* adds anything beyond trading activity.
- **Price-only model** — the same ARIMAX with the sentiment features removed. This is the key ablation: if the sentiment features don't beat price-only, sentiment added no value — a legitimate and reportable finding.

### 5.3 Transaction Costs

Include a simple, fixed per-trade cost (a small percentage capturing fee + spread; state the assumption explicitly). This is one line of code and it is essential: a signal that trades constantly can look profitable gross and lose money net. Many published high-frequency sentiment strategies stop being profitable once realistic costs are applied, so omitting costs would make the evaluation misleading.

### 5.4 Reported Metrics

At minimum: net return per trade, hit rate, number of trades, cumulative net return vs. each baseline. Optionally a directional-accuracy figure (did the sign of the forecast match the sign of the realized return). Avoid leaning on price-level error metrics (RMSE on price) — they are the misleading numbers that make a do-nothing model look good.

---

## 6. Pre-Build Validation Gate (go / no-go)

Cheap checks to run **before** investing in the full build, each of which is also good material for the writeup:

1. **Target is a return, not a price.** Confirm the label is `r_{t+1}`. (Pass/fail — non-negotiable.)
2. **Leak-free split wired up.** Chronological split, scaler fit on training only, features lagged to *t−1*. (Pass/fail — non-negotiable.)
3. **Lead-lag / Granger check.** Does sentiment at *t−1* help predict the return at *t* beyond what past returns already explain? A single function call in `statsmodels`. Whether the answer is yes or no, it tells you up front whether a tradeable signal is plausible — and reverse causality (price moving sentiment rather than the other way around) is a known finding worth reporting either way.

If checks 1–2 don't pass, fix them before anything else. Check 3 is informational, not a hard stop — a weak result there simply sets honest expectations.

---

## 7. Data Sourcing & Test Harness

The pipeline should never hard-code a single data source. Both development cost and live reliability depend on being able to swap the *origin* of posts without touching anything downstream of ingestion. The strategy is to develop against a replayed dataset, then promote to a third-party reseller, then (optionally) to the official X API — all behind one stable interface.

### 7.1 Source Abstraction Layer

Define a single `PostSource` interface that every backend implements. Everything downstream (cleanup → sentiment → aggregation) consumes its output and never knows or cares which backend produced it.

```
interface PostSource:
    stream(start, end) -> iterator[Post]   # yields posts in timestamp order

# Post — the normalized internal record every backend must emit
Post = {
    post_id:        str        # stable unique id (dedupe key)
    author_id:      str
    timestamp:      datetime    # UTC, tz-aware
    text:           str
    follower_count: int         # author reach AT TIME OF POST
    verified:       bool         # see 7.4 — normalize meaning across sources
    verified_type:  enum{none, legacy_notable, paid, business, unknown}
    source:         enum{dataset, reseller, official}
    raw:            dict         # untouched original payload, for reprocessing
}
```

The contract: **whatever the backend, it produces `Post` records on this schema.** Swapping sources becomes a one-line config change (`source: dataset` → `source: reseller`), not a refactor. This is the single most important design decision for a smooth transition.

### 7.2 Backends Behind the Interface

| Backend | Use phase | What it does |
|---|---|---|
| `DatasetReplaySource` | Development, CI, backtest | Reads a static labeled file (Parquet/CSV) and *replays* rows in timestamp order, simulating a live stream. Deterministic and free. |
| `ResellerSource` | Staging, early live | Calls a third-party provider (Apify, GetXAPI, Netrows, etc.); maps their profile fields onto `Post`. Cheaper than official, historical access, coverage caveats. |
| `OfficialApiSource` | Live (optional) | Calls the X API directly (pay-per-use as of Feb 2026). Highest fidelity, highest cost, strictest terms. |

Because all three emit the same `Post`, your cleanup/sentiment/aggregation code is written and tested **once** against the dataset backend and runs unchanged on the others.

### 7.3 Recommended Datasets for the Replay Backend

| Dataset | Why it fits | Metadata included |
|---|---|---|
| Crypto-influencer set (52 influencers, 2021–2023, via Apify) | Mirrors the influence-weighting design closely | Compound sentiment, per-tweet importance/influence coefficient, polarity, prices for BTC/ETH/BNB |
| "Cryptocurrency Tweets" (Kaggle, 2022–2023) | General crypto chatter with sentiment | User content, sentiment, timestamps |
| "Bitcoin tweets – 16M, sentiment tagged" (Kaggle/HF) | Volume for stress-testing throughput | Sentiment label (lighter on author metadata) |

Use a metadata-rich set (influence/follower fields present) as the primary dev fixture, since the influence-weighted aggregation in 4.5 depends on those fields existing.

### 7.4 Normalizing `verified` Across Sources (important)

"Verified" does **not** mean the same thing across eras or providers:
- Pre-2023 datasets: verified = the old notability badge (a genuine influence signal).
- Post-2023 / live: verified = a paid subscription, available to anyone — a much weaker signal.

The abstraction layer must map each source onto the `verified_type` enum so the influence model treats them consistently. **Recommendation:** weight influence primarily on (log) follower count, and treat `verified_type` as a secondary, source-aware feature rather than a raw boolean — otherwise a model trained on pre-2023 verified semantics will misbehave on live paid-verified data.

### 7.5 Phased Transition Plan

1. **Phase 0 — Dataset only.** Build and validate the entire pipeline against `DatasetReplaySource`. Free, deterministic, repeatable. All unit/integration tests run here.
2. **Phase 1 — Shadow reseller.** Point `ResellerSource` at a small live window *in parallel* with the dataset. Diff the normalized `Post` streams to catch field-mapping bugs (e.g. their follower field is a string, their timestamp is epoch-ms). No model changes — just validating the adapter.
3. **Phase 2 — Reseller live.** Promote `ResellerSource` to the real feed once the adapter is verified. Keep a dataset replay as the regression fixture.
4. **Phase 3 — Official API (optional).** Swap or supplement with `OfficialApiSource` only if data fidelity/coverage justifies the cost. Same interface, so this is a config flip plus a billing decision.

A reseller→official cutover should be reversible: keep both adapters, gate by config, and fail back to the cheaper source if the expensive one errors.

### 7.6 Test Harness

The dataset backend doubles as the test rig — this is the payoff of the abstraction.

- **Deterministic replay.** Same fixture in → same features/predictions out. Any diff is a real regression, not data noise.
- **Look-ahead-bias tests (ties to 4.8).** Build golden fixtures with hand-placed posts straddling bar boundaries and assert that the feature row for bar *t* contains **nothing** timestamped ≥ *t*. This is the test that protects against the most dangerous class of bug; it's only practical with a controllable source.
- **Simulated clock.** Replay should advance a virtual clock rather than wall-clock time, so a year of data tests in seconds and time-dependent logic (windows, lags, gating) is exercised exactly.
- **Adapter conformance tests.** A shared test suite every backend must pass (valid `Post` schema, monotonic timestamps, dedupe on `post_id`, UTC tz-awareness). Run it against the reseller adapter before trusting it in Phase 1.
- **Edge-case fixtures.** Empty bars (no posts), bot floods, a single mega-follower post, missing follower counts, non-English text — assert the aggregation and gating behave sensibly.
- **Throughput test.** Run the 16M-tweet set through to surface performance ceilings before live volume does.

---

## 8. Cross-Cutting Concerns

- **Leakage prevention** (covered in 4.8 and Section 5) is the top priority — build and test the lagging and the train/test split before trusting any backtest number.
- **Reproducibility:** version the data, cleaning code, model config, threshold, and random seeds. A backtest you can't reproduce isn't evidence. Fixing a random seed and committing the exact dataset slice is enough for this project.
- **Compute budget:** the LLM feature path (4.3) is the expensive part. If it's slow or costly, the VADER-only version (Section 10) is a fully valid project on its own.
- **Data quality / manipulation (as a limitation):** social sentiment is adversarial — pump-and-dump shilling and bot swarms can dominate raw sentiment. Detecting this is out of scope (Section 10), but call it out as a known threat to the signal's validity in your writeup.

---

## 9. Suggested Tech Stack (indicative)

| Layer | Candidate tooling |
|---|---|
| Data sourcing | `PostSource` abstraction with swappable backends; dataset replay (Parquet/CSV), reseller adapters (Apify/GetXAPI/Netrows), official X API |
| Ingestion | Platform streaming API / Kafka or a queue; object storage for raw landing |
| Processing | Python, pandas/Polars; Spark if volume warrants |
| Tokenization | HuggingFace `tokenizers` (BPE/SentencePiece) matched to the model |
| LLM features | HuggingFace Transformers (a finance/crypto-tuned classifier or embedding model) |
| Lexicon sentiment | `vaderSentiment` (with a custom crypto lexicon extension) |
| Modeling | `statsmodels` SARIMAX / `pmdarima` for ARIMAX (return target); `arch` for an optional GARCH volatility filter |
| Evaluation | Custom fixed-holding-period backtest in pandas; `statsmodels` `grangercausalitytests` for the pre-build lead-lag check |
| Orchestration | Airflow / Prefect / Dagster (overkill for the prototype — a script or notebook is fine to start) |
| Storage | Time-series store or partitioned Parquet keyed by UTC bar |

---

## 10. Scope, Assumptions & Limitations

**Assumptions / decisions locked for this version**
- Single asset (e.g. BTC). Multi-asset is a later fan-out, not part of this project.
- One fixed bar frequency shared across both feeds (start with the resolution your dataset supports — e.g. 1-minute).
- Forecast horizon = the fixed holding period (e.g. predict the next-period return, hold N bars, then close).
- Target is the next-period **return**; decision is **buy / sell / no-trade** via a single threshold.
- Evaluation is a fixed-holding-period backtest on historical data with transaction costs and baselines (Section 5).

**Explicitly out of scope (acknowledge as limitations, do not build)**
- **Position sizing & portfolio control** — one trade at a time, fixed exit. No capital allocation logic.
- **Live execution** — historical backtest only; no broker/exchange integration or latency engineering.
- **Bot / manipulation detection & manipulation-aware influence weighting** — real and important (pump-and-dumps, bot swarms), but a research project in itself. Note it as a known vulnerability of the sentiment signal.
- **Multi-coin survivorship handling** — only relevant once you go multi-asset.
- **Regime-switching models** — a known way the sentiment↔return relationship shifts over time; out of scope but worth a sentence in the writeup.
- **GARCH-X / volatility modeling as the primary model** — appropriate if the goal were *volatility* forecasting (sentiment predicts variance better than direction), but this project's deliverable is a directional decision, so ARIMAX on returns is the right core. A GARCH volatility *filter* is an optional enhancement only (Section 4.9).
- **Live data sourcing (reseller / official API)** — the `PostSource` abstraction and phased plan in Section 7 are documented for completeness, but a single historical dataset replay is all this project needs. The reseller/official phases are future work.

**Honest expectation:** short-horizon directional prediction of crypto returns is genuinely hard, and the literature suggests public sentiment often lags price or only helps at very short horizons before costs. A weak or null result, reported clearly against the baselines, is a legitimate and complete outcome for this project — the value is in the correct methodology, not in a profitable strategy.

**Remaining question to settle before building**
- **LLM feature branch vs. VADER-only.** The full pipeline includes both a lexicon (VADER) and an LLM feature path. A defensible minimal version uses VADER only (cheaper, simpler, and what most comparable student projects do); the LLM branch is a strong stretch goal once the VADER end-to-end version works and is evaluated. Decide based on your time and compute budget.
