"""Branch A — tokenization + LLM feature extraction (Design §4.3).

Scores each post with a finance/crypto-tuned transformer (CryptoBERT by default),
emitting a signed sentiment in [-1, +1] plus the model's confidence. This is the
expensive path (GPU/MPS), kept behind the optional `llm` dependency extra and run
as a one-off preprocessing step (see prepare_llm.py).

  llm_signed = P(bullish) - P(bearish)        # directional sentiment, like VADER
  llm_conf   = max class probability           # used for confidence-weighted fusion
"""
from __future__ import annotations

import numpy as np

DEFAULT_MODEL = "ElKulako/cryptobert"


class LLMScorer:
    """Batched transformer sentiment scorer. Loads lazily; uses MPS if available."""

    def __init__(self, model_name: str = DEFAULT_MODEL, device: str | None = None,
                 max_length: int = 64) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.max_length = max_length
        self.device = device or ("mps" if torch.backends.mps.is_available() else "cpu")
        self.tok = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.to(self.device).eval()

        # Resolve which logit index is bullish vs bearish from the model config;
        # fall back to CryptoBERT's documented order {0:Bearish,1:Neutral,2:Bullish}.
        id2label = {i: str(l).lower() for i, l in self.model.config.id2label.items()}
        self.bull_idx = next((i for i, l in id2label.items()
                              if "bull" in l or "pos" in l), 2)
        self.bear_idx = next((i for i, l in id2label.items()
                              if "bear" in l or "neg" in l), 0)

    def score(self, texts: list[str], batch_size: int = 128,
              progress_every: int = 200) -> tuple[np.ndarray, np.ndarray]:
        """Return (signed, confidence) arrays aligned to `texts`."""
        torch = self.torch
        signed, conf = [], []
        n_batches = (len(texts) + batch_size - 1) // batch_size
        with torch.no_grad():
            for b, i in enumerate(range(0, len(texts), batch_size)):
                batch = [t if isinstance(t, str) else "" for t in texts[i:i + batch_size]]
                enc = self.tok(batch, padding=True, truncation=True,
                               max_length=self.max_length, return_tensors="pt").to(self.device)
                probs = torch.softmax(self.model(**enc).logits, dim=-1)
                signed.append((probs[:, self.bull_idx] - probs[:, self.bear_idx]).cpu().numpy())
                conf.append(probs.max(dim=-1).values.cpu().numpy())
                if progress_every and (b % progress_every == 0 or b == n_batches - 1):
                    print(f"  scored batch {b + 1}/{n_batches} "
                          f"({min(i + batch_size, len(texts)):,}/{len(texts):,} posts)", flush=True)
        return np.concatenate(signed), np.concatenate(conf)
