"""One-time LLM scoring: run CryptoBERT over the windowed sample (Phase 7).

Reads the same sampled tweets VADER uses, scores each post with the transformer,
and writes per-post LLM sentiment + confidence aligned to the sample's row order.
Expensive — run once (downloads the model on first use). Needs the `llm` extra:
    uv sync --extra llm

Run:  uv run python prepare_llm.py
"""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from src.config import load_config
from src.llm_sentiment import DEFAULT_MODEL, LLMScorer


def main() -> None:
    cfg = load_config()
    tweets_path = Path(cfg["data"]["tweets_path"])
    out = tweets_path.with_name("tweets_llm_scores.parquet")

    tweets = pd.read_parquet(tweets_path)  # columns: date, text, followers (row order = key)
    print(f"Scoring {len(tweets):,} posts with {DEFAULT_MODEL} ...")
    scorer = LLMScorer()
    print(f"  device={scorer.device}, bull_idx={scorer.bull_idx}, bear_idx={scorer.bear_idx}")

    signed, conf = scorer.score(tweets["text"].astype(str).tolist(), batch_size=128)

    pd.DataFrame({"llm_signed": signed, "llm_conf": conf},
                 index=tweets.index).to_parquet(out)
    print(f"\nWrote {len(signed):,} LLM scores -> {out}")
    print(f"  signed mean {signed.mean():+.3f} (std {signed.std():.3f}), "
          f"confidence mean {conf.mean():.3f}")


if __name__ == "__main__":
    main()
