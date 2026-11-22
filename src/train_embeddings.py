"""Deliverable 2 (part 1) - Word2Vec ingredient embeddings learned from recipe co-occurrence.

Each recipe is one "sentence" whose tokens are its canonical ingredients (vocabulary only). The
context window is larger than the longest recipe and window shrinking is disabled, so every other
ingredient of a recipe is context and ingredient order is irrelevant. Ingredients used in the same
kinds of recipes end up close together, which is the signal later used for substitution.

Model selection uses an intrinsic criterion that is independent of Deliverable 3's evaluation:
food-group coherence of nearest neighbours (labels from USDA / Ahn categories only, never from
name keywords). Known substitution pairs are deliberately NOT used here so that the Hit@k
evaluation in Deliverable 3 stays unbiased.

Outputs:
    models/ingredient_w2v.model         full gensim model (selected configuration)
    models/ingredient_vectors.kv        KeyedVectors (what downstream code loads)
    data/processed/ingredient_neighbors.csv   top-20 cosine neighbours per ingredient
    reports/embedding_report.md         grid results, coherence vs. baseline, example neighbours

Run:  python src/train_embeddings.py
"""

from __future__ import annotations

import json
import sys
import time
import zlib
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import Word2Vec

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flavorswap import paths  # noqa: E402

SEED = 42
WINDOW = 50          # > longest recipe (43 canonical ingredients)
EPOCHS = 30
NEGATIVE = 10
GRID = [             # (sg, vector_size, sample)
    (0, 100, 1e-3),
    (0, 100, 1e-4),
    (1, 100, 1e-3),
    (1, 100, 1e-4),
]
EVAL_MIN_RECIPES = 50   # evaluate coherence only on ingredients with stable vectors
K = 10
PROBES = [
    "butter", "olive oil", "sugar", "honey", "milk", "heavy cream", "sour cream", "egg",
    "all purpose flour", "chicken breast", "ground beef", "bacon", "shrimp", "parmesan cheese",
    "cilantro", "basil", "garlic", "soy sauce", "lemon juice", "white wine", "brown rice",
    "spaghetti", "black bean", "walnut", "cinnamon", "cumin", "tofu", "mayonnaise",
]


def stable_hash(s: str) -> int:
    """Deterministic replacement for Python's salted str hash (gensim uses it to seed vectors)."""
    return zlib.crc32(s.encode("utf-8"))


def load_corpus() -> tuple[list[list[str]], pd.DataFrame]:
    rec = pd.read_parquet(paths.PROCESSED / "recipes.parquet", columns=["ingredients_in_vocab"])
    corpus = [list(x) for x in rec.ingredients_in_vocab if len(x) >= 2]
    ing = pd.read_csv(paths.PROCESSED / "ingredients.csv")
    return corpus, ing


def train(corpus, sg: int, size: int, sample: float) -> Word2Vec:
    return Word2Vec(
        sentences=corpus, vector_size=size, window=WINDOW, shrink_windows=False, min_count=1,
        sg=sg, negative=NEGATIVE, sample=sample, epochs=EPOCHS, seed=SEED, workers=1,
        hashfxn=stable_hash,
    )


def neighbour_coherence(kv, labels: dict[str, str], k: int = K) -> tuple[float, float]:
    """Mean share of an ingredient's k nearest *labelled* neighbours that share its food group,
    and the chance level for the same label distribution."""
    words = [w for w in labels if w in kv.key_to_index]
    idx = np.array([kv.key_to_index[w] for w in words])
    vecs = kv.get_normed_vectors()[idx]
    sims = vecs @ vecs.T
    np.fill_diagonal(sims, -np.inf)
    lab = np.array([labels[w] for w in words])
    top = np.argsort(-sims, axis=1)[:, :k]
    score = float((lab[top] == lab[:, None]).mean())
    _, counts = np.unique(lab, return_counts=True)
    # chance: neighbour drawn from the other n-1 items
    n = len(lab)
    chance = float(sum(c / n * (c - 1) / (n - 1) for c in counts))
    return score, chance


def main() -> None:
    paths.MODELS.mkdir(exist_ok=True)
    (paths.ROOT / "reports").mkdir(exist_ok=True)
    corpus, ing = load_corpus()
    n_tokens = sum(len(r) for r in corpus)
    print(f"Corpus: {len(corpus):,} recipes, {n_tokens:,} ingredient tokens, "
          f"max length {max(map(len, corpus))}")
    assert max(map(len, corpus)) < WINDOW

    eval_set = ing[(ing.n_recipes >= EVAL_MIN_RECIPES) & ing.food_group_source.isin(["usda", "ahn"])
                   & (ing.food_group != "Other")]
    labels = dict(zip(eval_set.ingredient, eval_set.food_group))
    print(f"Coherence evaluation set: {len(labels)} ingredients")

    results, models = [], {}
    for sg, size, sample in GRID:
        t0 = time.time()
        m = train(corpus, sg, size, sample)
        score, chance = neighbour_coherence(m.wv, labels)
        name = f"{'skipgram' if sg else 'cbow'}_d{size}_s{sample:g}"
        results.append({"config": name, "sg": sg, "vector_size": size, "sample": sample,
                        f"coherence@{K}": round(score, 4), "chance": round(chance, 4),
                        "train_seconds": round(time.time() - t0, 1)})
        models[name] = m
        print(f"  {name:22} coherence@{K}={score:.4f} (chance {chance:.4f})  {time.time() - t0:.0f}s")

    res = pd.DataFrame(results).sort_values(f"coherence@{K}", ascending=False)
    best = res.iloc[0]["config"]
    model = models[best]
    print(f"Selected: {best}")

    model.save(str(paths.MODELS / "ingredient_w2v.model"))
    model.wv.save(str(paths.MODELS / "ingredient_vectors.kv"))
    (paths.MODELS / "ingredient_w2v_config.json").write_text(json.dumps({
        "selected": best, "window": WINDOW, "epochs": EPOCHS, "negative": NEGATIVE, "seed": SEED,
        "shrink_windows": False, "min_count": 1, "corpus_recipes": len(corpus),
        "corpus_tokens": n_tokens, "vocab_size": len(model.wv), "grid": results,
    }, indent=2))

    # ---------------- neighbours table ----------------
    kv = model.wv
    rows = []
    for w in kv.index_to_key:
        for rank, (nb, sim) in enumerate(kv.most_similar(w, topn=20), 1):
            rows.append((w, rank, nb, round(float(sim), 4)))
    nbr = pd.DataFrame(rows, columns=["ingredient", "rank", "neighbor", "cosine"])
    nbr.to_csv(paths.PROCESSED / "ingredient_neighbors.csv", index=False)

    write_report(res, best, kv, ing, labels)
    print("Done.")


def write_report(res: pd.DataFrame, best: str, kv, ing: pd.DataFrame, labels) -> None:
    # per-group coherence for the selected model
    words = [w for w in labels if w in kv.key_to_index]
    normed = kv.get_normed_vectors()
    idx = np.array([kv.key_to_index[w] for w in words])
    sims = normed[idx] @ normed[idx].T
    np.fill_diagonal(sims, -np.inf)
    lab = np.array([labels[w] for w in words])
    top = np.argsort(-sims, axis=1)[:, :K]
    same = (lab[top] == lab[:, None]).mean(axis=1)
    per_group = (pd.DataFrame({"group": lab, "coherence": same})
                 .groupby("group").agg(n=("coherence", "size"), coherence=("coherence", "mean"))
                 .sort_values("coherence", ascending=False))
    share = pd.Series(lab).value_counts(normalize=True)
    per_group["chance"] = [share[g] for g in per_group.index]

    lines = [
        "# FlavorSwap - Deliverable 2 embedding report",
        "",
        "Generated by `src/train_embeddings.py`.",
        "",
        "## Set-up",
        f"- Corpus: Food.com recipes with >= 2 vocabulary ingredients; each recipe is one sentence.",
        f"- Window {WINDOW} (> longest recipe) with `shrink_windows=False`: every co-ingredient is "
        "context, ingredient order is irrelevant.",
        f"- {EPOCHS} epochs, {NEGATIVE} negative samples, seed {SEED}, single worker, deterministic hash "
        "(bit-for-bit reproducible).",
        f"- Model selection metric: **food-group coherence@{K}** - share of an ingredient's {K} nearest "
        f"neighbours (cosine) in the same food group, over {len(labels)} ingredients with >= "
        f"{EVAL_MIN_RECIPES} recipes whose group comes from USDA/Ahn categories (not name keywords). "
        "Substitution pairs are not used here, keeping Deliverable 3's Hit@k evaluation unbiased.",
        "",
        "## Configuration grid",
        "",
        res.to_markdown(index=False),
        "",
        f"Selected: **{best}**.",
        "",
        "## Coherence by food group (selected model)",
        "",
        per_group.round(3).to_markdown(),
        "",
        "## Nearest neighbours of probe ingredients",
        "",
        "| ingredient | 8 nearest neighbours (cosine) |",
        "|---|---|",
    ]
    for p in PROBES:
        if p in kv.key_to_index:
            nb = ", ".join(f"{w} ({s:.2f})" for w, s in kv.most_similar(p, topn=8))
            lines.append(f"| **{p}** | {nb} |")
    lines.append("")
    (paths.ROOT / "reports" / "embedding_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
