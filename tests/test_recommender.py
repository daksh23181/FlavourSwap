"""Invariant and regression tests for the hybrid substitution recommender (Deliverable 3).

Run:  python tests/test_recommender.py      (exits non-zero on failure)
Requires the pipeline outputs (build_dataset, train_embeddings, evaluate_substitutions).
"""

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from flavorswap.recommender import (  # noqa: E402
    MAX_WATER_DIFF, SEASONING_GROUP, Recommender, UnknownIngredient,
)

FAILURES: list[str] = []


def check(cond: bool, msg: str) -> None:
    if not cond:
        FAILURES.append(msg)
        print("FAIL", msg)


def main() -> int:
    rec = Recommender.load()
    w = rec.weights
    groups = rec.ing.food_group.to_numpy()
    rng = np.random.default_rng(0)
    sample = list(rng.choice(rec.names[:1500], 150, replace=False)) + ["butter", "sugar", "sour cream", "salt"]

    for s in sample:
        i = rec.index[s]
        sim = rec.similar(s, 10)
        check(len(sim) == 10, f"{s}: similar() returned {len(sim)} rows")
        check(s not in set(sim.ingredient), f"{s}: source recommended to itself")
        check(bool((np.diff(sim.score.to_numpy()) <= 1e-9).all()), f"{s}: similar scores not sorted")
        hl = rec.healthier(s, 10)
        for t in hl.ingredient:
            j = rec.index[t]
            need = rec.required_improvement(i, w.min_reduction)
            check(rec.health[i] - rec.health[j] >= need - 1e-12, f"{s}->{t}: not healthier by the required margin")
            check(abs(rec.water[i] - rec.water[j]) <= MAX_WATER_DIFF, f"{s}->{t}: per-100 g comparison not comparable")
            check(bool(rec.group_ok(i, w.min_group_prob)[j]), f"{s}->{t}: implausible food-group transition")
            check(groups[i] == SEASONING_GROUP or groups[j] != SEASONING_GROUP, f"{s}->{t}: seasoning as healthier swap")

    h = {n: rec.health[rec.index[n]] for n in ["brown rice", "white rice", "whole wheat flour", "all purpose flour",
                                              "olive oil", "butter", "margarine", "shortening"]}
    check(h["brown rice"] < h["white rice"], "health: brown rice should beat white rice")
    check(h["whole wheat flour"] < h["all purpose flour"], "health: whole wheat flour should beat all purpose flour")
    check(h["olive oil"] < h["butter"] < h["margarine"], "health: olive oil < butter < margarine")
    check(h["butter"] < h["shortening"], "health: butter should beat shortening (trans fat)")

    check("margarine" in set(rec.similar("butter", 10).ingredient), "butter: margarine missing from similar")
    check("honey" in set(rec.healthier("sugar", 10).ingredient), "sugar: honey missing from healthier")
    check("light sour cream" in set(rec.healthier("sour cream", 10).ingredient), "sour cream: light sour cream missing")
    check("garlic" not in set(rec.healthier("white rice", 10).ingredient), "white rice: garlic offered as healthier")
    check("egg" not in set(rec.similar("butter", 10).ingredient), "butter: egg offered as similar")
    check("white rice" not in set(rec.healthier("brown rice", 10).ingredient), "brown rice: white rice offered as healthier")
    check(rec.healthier("tomato", 10).empty, "tomato: whole vegetable should get no health swaps")
    check(rec.healthier("apple", 10).empty, "apple: whole fruit should get no health swaps")
    check("orange" in set(rec.healthier("orange juice", 10).ingredient), "orange juice: whole orange missing")
    for low in ["ice", "vinegar", "lemon juice", "parsley", "dry white wine"]:
        check(rec.healthier(low, 10).empty, f"{low}: nothing to limit, should get no health swaps")
    check(rec.ing.food_group[rec.index["soy sauce"]] == "Sauces, Condiments & Soups", "soy sauce should be a condiment")
    check(rec.ing.food_group[rec.index["lemon zest"]] == "Herbs & Spices", "lemon zest should be a seasoning")

    # free-text input as typed by app users
    for text, expected in [("Butter", "butter"), ("eggs", "egg"), ("Large Eggs", "egg"), ("garlic cloves", "garlic"),
                           ("sour creme", "sour cream"), ("olive oil ", "olive oil"), ("all-purpose flour", "all purpose flour"),
                           ("chiken breast", "chicken breast"), ("fresh basil leaves", "basil")]:
        got = rec.resolve(text)
        check(got == expected, f"resolve({text!r}) = {got!r}, expected {expected!r}")
    try:
        rec.similar("xyzzy")
        check(False, "unknown ingredient did not raise")
    except UnknownIngredient as e:
        check(isinstance(e, KeyError), "UnknownIngredient should be a KeyError")
    check(list(rec.healthier("white rice", 5).columns) == list(rec.similar("butter", 1).columns),
          "healthier/similar tables must share columns (also when empty)")

    n = len(FAILURES)
    print("all recommender checks passed" if n == 0 else f"{n} failures")
    return 1 if n else 0


if __name__ == "__main__":
    sys.exit(main())
