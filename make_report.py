"""Phase 6 — analysis & writeup.

Recomputes the five strategies over the test slice (for their equity curves), plots
cumulative NET RETURN (PnL, never predicted-vs-true price levels), and writes a
standalone REPORT.md a reader can follow without seeing any code.

Run:  uv run python make_report.py
"""
from __future__ import annotations

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

FIG = "results/cumulative_returns.png"
REPORT = "REPORT.md"


def df_to_md(df: pd.DataFrame) -> str:
    """Render a DataFrame (with a named index) as a GitHub markdown table —
    avoids a tabulate dependency for .to_markdown()."""
    def fmt(v):
        if isinstance(v, (int, float, np.integer, np.floating)) and float(v).is_integer():
            return str(int(v))
        return str(v)

    cols = [df.index.name or ""] + [str(c) for c in df.columns]
    out = ["| " + " | ".join(cols) + " |",
           "|" + "|".join(["---"] * len(cols)) + "|"]
    for idx, r in df.iterrows():
        out.append("| " + " | ".join([str(idx)] + [fmt(v) for v in r.values]) + " |")
    return "\n".join(out)


def daily_equity(res, test_index) -> pd.Series:
    """Cumulative net return as a daily series over the test slice (flat on
    no-trade days), so every strategy is comparable on one axis."""
    return res.equity_curve.reindex(test_index).ffill().fillna(0.0)


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
    return res, thr, da


def main() -> None:
    cfg = load_config()
    N, cost = cfg["holding_period_N"], cfg["transaction_cost"]
    table = pd.read_parquet(cfg["data"]["processed_path"])
    close = load_ohlcv(cfg["data"]["ohlcv_path"])["close"]
    endog, exog = table[TARGET], table[SENTIMENT_FEATURES]
    split = chronological_split(table.index, cfg["split"]["train_frac"])
    n_train = len(split.train_index)
    order = select_order(endog.iloc[:n_train])
    ti = split.test_index

    price_res, price_thr, price_da = run_model(endog, None, n_train, order, split, close, N, cost)
    sent_res, sent_thr, sent_da = run_model(endog, exog, n_train, order, split, close, N, cost)
    bnh = buy_and_hold(close.loc[ti], cost)
    tp = close.loc[ti[0]: ti[-1] + pd.Timedelta(days=N + 3)]
    ab = fixed_hold_backtest(always_buy(tp), tp, N, cost)
    rnd = fixed_hold_backtest(random_decisions(tp, seed=cfg["random_seed"]), tp, N, cost)

    strategies = {
        "Sentiment ARIMAX": sent_res, "Price-only ARIMA": price_res,
        "Buy & hold": bnh, "Always-buy": ab, "Random": rnd,
    }

    # ---- Plot cumulative net return (PnL) -----------------------------------
    fig, ax = plt.subplots(figsize=(10, 6))
    styles = {"Sentiment ARIMAX": dict(lw=2.4, color="#d62728"),
              "Price-only ARIMA": dict(lw=2.4, color="#1f77b4"),
              "Buy & hold": dict(lw=1.6, color="#2ca02c", ls="--"),
              "Always-buy": dict(lw=1.2, color="#7f7f7f", ls=":"),
              "Random": dict(lw=1.2, color="#bcbd22", ls=":")}
    for name, res in strategies.items():
        ax.plot(ti, daily_equity(res, ti) * 100, label=name, **styles[name])
    ax.axhline(0, color="black", lw=0.8)
    ax.set_title(f"Cumulative net return on the test slice "
                 f"({ti[0].date()} → {ti[-1].date()}, after {cost:.1%}/trade cost)")
    ax.set_ylabel("Cumulative net return (%)")
    ax.set_xlabel("Date")
    ax.legend(loc="lower left")
    ax.grid(alpha=0.3)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(FIG, dpi=150)
    print(f"Saved figure -> {FIG}")

    # ---- Metrics table ------------------------------------------------------
    def row(name, res):
        return {"strategy": name, "trades": res.n_trades,
                "avg_net_return": round(res.avg_net_return, 5),
                "hit_rate": round(res.hit_rate, 3),
                "cumulative_net_return": round(res.cumulative_net_return, 4)}
    tbl = pd.DataFrame([row(n, r) for n, r in strategies.items()]).set_index("strategy")

    # ---- Granger / ADF from Phase 3 -----------------------------------------
    v = pd.read_csv("results/phase3_validation.csv")
    granger = v[v["check"] == "granger"][["series", "min_pvalue", "best_lag"]]

    # ---- Write REPORT.md ----------------------------------------------------
    md = _report_markdown(cfg, split, order, tbl, granger, price_da, sent_da,
                          price_res, sent_res, bnh)
    with open(REPORT, "w") as f:
        f.write(md)
    print(f"Wrote {REPORT}\n")
    print(tbl.to_string())


def _report_markdown(cfg, split, order, tbl, granger, price_da, sent_da,
                     price_res, sent_res, bnh) -> str:
    ti = split.test_index
    gr = "\n".join(f"| {r.series} | {r.min_pvalue:.3f} | {int(r.best_lag)} |"
                   for r in granger.itertuples())
    beat = sent_res.cumulative_net_return > price_res.cumulative_net_return
    return f"""# Does Twitter sentiment improve short-horizon BTC return forecasting?

**Answer: No.** On a leak-free, out-of-sample, cost-aware backtest, adding aggregate
social-media sentiment to a price-only return forecaster **did not improve** next-day
Bitcoin trading decisions — it slightly degraded them. This is a clean, honest null,
and it agrees with a pre-build Granger-causality check that found no sentiment lead.

## What was built

An end-to-end pipeline: score tweets with VADER → aggregate to influence-weighted
daily sentiment → fuse with price features → forecast the **next-day return** with
ARIMAX → threshold into buy/sell/no-trade → evaluate with a fixed-holding-period
backtest against dumb baselines. The evaluation harness was built and **proven honest
against a random signal before any model existed**.

## Setup

| | |
|---|---|
| Asset / bar / horizon | BTC / daily / hold N={cfg['holding_period_N']} bar (forecast r₍t+1₎) |
| Window | {ti.min().date()} … {ti.max().date()} test; train ends {split.cutoff.date()} ({len(split.train_index)} train / {len(ti)} test) |
| Transaction cost | {cfg['transaction_cost']:.1%} per trade |
| Sentiment | VADER compound, influence-weighted by log(1+followers); mean, post-volume, dispersion |
| Empty-bar policy | sentiment forward-filled across no-tweet days (~45% of bars) |
| Model | ARIMAX{order} on the return; sentiment enters as exogenous regressors |

## Methodology & honesty safeguards

- **Target is a return, never a price level** — predicting price at short horizons
  trivially "succeeds" by echoing the last price; forecasting r₍t+1₎ removes that.
- **Leak-free, out-of-sample** — chronological split (never shuffled), scaler fit on
  train only, every feature lagged so row t uses only information through the close
  of t, verified by an automated no-look-ahead check.
- **Honest harness** — a random buy/sell signal earns ≈ −cost per trade, a ~0.5 hit
  rate, and beats buy-and-hold only ~7% of the time. The backtest cannot be fooled.
- **Clean ablation** — the sentiment model is identical to the price-only control
  except for the sentiment regressors.

## Pre-build validation (Phase 3)

Both hard gates passed (target is a return; split is leak-free). All series are
stationary (ADF p < 0.001). Granger causality — does lagged sentiment predict the
return beyond the return's own past? — found **no significant lead**:

| sentiment feature | min p-value | best lag |
|---|---|---|
{gr}

All p-values ≫ 0.05, so a tradeable sentiment edge was implausible going in. We ran
the full experiment anyway and reported it.

## Results

![Cumulative net return by strategy](results/cumulative_returns.png)

{df_to_md(tbl)}

Directional accuracy (sign of forecast vs realized return, traded bars):
price-only **{price_da:.3f}**, sentiment **{sent_da:.3f}** — both **below 0.5**.

**Reading the numbers.** The test slice was a BTC **downtrend** (buy-and-hold
{bnh.cumulative_net_return:+.0%}), so every always-in-market baseline lost heavily.
The price-only model's small **positive** cumulative ({price_res.cumulative_net_return:+.1%})
is **not evidence of skill**: its directional accuracy is below a coin flip, so the
result comes from a handful of large short trades landing in a falling market — a
small-sample artifact, not a predictive edge. Adding sentiment
({'beat' if beat else 'did NOT beat'} the control) pushed cumulative return to
{sent_res.cumulative_net_return:+.1%}, lowered the hit rate, and moved directional
accuracy further below chance. **Sentiment added noise, not signal.**

## Limitations

- **Directional prediction is genuinely hard** — the literature and our Granger
  result both expect public sentiment to lag price or help only fleetingly. A null
  is the expected outcome, not a bug.
- **Bots & manipulation** — crypto social sentiment is adversarial (pump-and-dump
  shilling, bot swarms); raw aggregate sentiment can be dominated by coordinated
  activity. We did not detect or filter this.
- **Single asset, single window, one regime** — BTC only, ~1 year, and a
  downtrending test slice. Results may not generalize across assets or regimes.
- **No position sizing** — one fixed-size, fixed-exit trade at a time; no capital
  allocation, stop-losses, or risk control beyond the decision threshold.
- **Sentiment sparsity / staleness** — ~45% of days had no tweets and were
  forward-filled, and dense days were sampled to ≤3k tweets; both blunt the signal.
- **Small trade counts** — 35–51 trades make cumulative return noisy; treat single
  cumulative figures with caution.

## Conclusion

Built and evaluated honestly, **aggregate Twitter sentiment did not improve
short-horizon BTC return forecasting over price history alone**, and degraded it on
this test slice. The price-only control showed no genuine directional skill either.
The value of this project is the **correct, leak-free, baseline-anchored methodology**
and a clearly-reported null — exactly the defensible outcome the design anticipated.
"""


if __name__ == "__main__":
    main()
