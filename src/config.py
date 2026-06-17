"""Load the single config file so locked parameters live in exactly one place.

Build Pathway Phase 0, step 2.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# Repo root = parent of the src/ directory holding this file.
ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG_PATH = ROOT / "config.yaml"


def load_config(path: str | Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    """Read config.yaml into a dict. Relative data paths are resolved to the repo root."""
    path = Path(path)
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)

    # Resolve data paths relative to the repo root so scripts work from any cwd.
    data = cfg.get("data", {})
    for key in ("tweets_path", "ohlcv_path", "processed_path"):
        if key in data and data[key]:
            p = Path(data[key])
            data[key] = str(p if p.is_absolute() else (ROOT / p))
    cfg["data"] = data
    return cfg


if __name__ == "__main__":
    import json

    print(json.dumps(load_config(), indent=2))
