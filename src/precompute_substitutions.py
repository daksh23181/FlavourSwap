"""Deliverable 3 output - substitution table for every vocabulary ingredient.

For each ingredient: the top-10 *similar* and top-10 *healthier* substitutes from the tuned hybrid
model (models/hybrid_config.json), with the explanation fields the app shows (embedding cosine,
shared flavor compounds, nutrient deltas per 100 g, health-score change).

Output: data/processed/substitutions.csv
Run after src/evaluate_substitutions.py:  python src/precompute_substitutions.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flavorswap import paths  # noqa: E402
from flavorswap.recommender import HybridWeights, Recommender  # noqa: E402

TOP_K = 10


def main() -> None:
    t0 = time.time()
    weights = HybridWeights.load()
    rec = Recommender.load(weights)
    frames = []
    for n, source in enumerate(rec.names, 1):
        for mode, fn in (("similar", rec.similar), ("healthier", rec.healthier)):
            df = fn(source, TOP_K)
            if df.empty:
                continue
            df.insert(0, "mode", mode)
            df.insert(0, "source", source)
            frames.append(df)
        if n % 500 == 0:
            print(f"  {n}/{len(rec.names)} ({time.time() - t0:.0f}s)")
    out = pd.concat(frames, ignore_index=True)
    out.to_csv(paths.PROCESSED / "substitutions.csv", index=False)
    n_healthy = out[out["mode"] == "healthier"].source.nunique()
    print(f"Wrote {len(out):,} rows: similar for {out[out['mode'] == 'similar'].source.nunique():,} "
          f"ingredients, healthier for {n_healthy:,} ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
