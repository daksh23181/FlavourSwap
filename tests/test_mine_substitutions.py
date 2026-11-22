"""Regression tests for the review-substitution miner (Deliverable 3 ground truth).

Run:  python tests/test_mine_substitutions.py      (exits non-zero on failure)
Requires data/processed/ingredients.csv (python src/build_dataset.py).
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from mine_substitutions import Miner  # noqa: E402

RECIPE = frozenset({
    "sour cream", "butter", "margarine", "milk", "ricotta cheese", "water", "onion", "peanut oil",
    "salt", "half and half", "chicken broth", "all purpose flour", "vegetable oil", "egg", "sugar",
    "cream of mushroom soup", "monterey jack cheese", "onion powder", "ground beef", "cream cheese",
    "hazelnut", "oat bran", "shortening",
})

CASES = [
    # accepted, with direction (old, new)
    ("i used greek yogurt instead of sour cream and it was great", ("sour cream", "greek yogurt")),
    ("used cottage cheese instead of ricotta", ("ricotta cheese", "cottage cheese")),
    ("i used 2 cups of chopped walnuts instead of the butter", ("butter", "walnut")),
    ("i substituted the butter with applesauce", ("butter", "applesauce")),
    ("replaced the oil with applesauce", ("vegetable oil", "applesauce")),
    ("instead of milk, i used orange juice", ("milk", "orange juice")),
    ("i used half and half instead of milk", ("milk", "half and half")),
    ("i used applesauce as a substitute for oil", ("vegetable oil", "applesauce")),
    ("i used fat free sour cream instead of regular sour cream", ("sour cream", "fat free sour cream")),
    ("i used oil instead of butter and it came out great", ("butter", "vegetable oil")),
    ("made it with ground turkey instead of ground beef", ("ground beef", "ground turkey")),
    ("i sauteed the onions in olive oil instead of butter", ("butter", "olive oil")),
    ("with finely chopped walnuts instead of hazelnuts", ("hazelnut", "walnut")),
    ("1/2 cup of wheat germ instead of oat bran", ("oat bran", "wheat germ")),
    ("i used 3/4 cup butter instead of shortening", ("shortening", "butter")),
    ("i add lots of pepper instead of salt", ("salt", "black pepper")),
    ("with olive instead of peanut oil - delicious", ("peanut oil", "olive oil")),
    ("i used cream of celery instead of cream of mushroom soup",
     ("cream of mushroom soup", "cream of celery soup")),
    # rejected
    ("i would not substitute margarine for butter", None),               # negation
    ("i used butter and sugar instead of margarine", None),              # coordinated new
    ("i used butter, sugar instead of margarine", None),                 # list before new
    ("i used dried minced onions rather than onion powder", None),       # truncated new
    ("i used comte cheese instead of monterey jack", None),              # truncated new
    ("added mushrooms to the gravy instead of onions", None),            # locative attachment
    ("i used a little broth to soak the onions instead of water", None),  # verb object
    ("used milk to mix with eggs instead of water", None),               # mixed with ...
    ("i also make it vegan by substituting the milk with coconut or soy milk", None),  # alternatives
    ("baked it instead of grilling", None),                              # not ingredients
    ("i used turkey instead of chicken", None),                          # old not in recipe
]


def main() -> int:
    vocab = set(pd.read_csv(ROOT / "data" / "processed" / "ingredients.csv").ingredient)
    miner = Miner(vocab, {1: RECIPE})
    failures = 0
    for text, expected in CASES:
        got = miner.mine_sentence(text, 1)
        pair = (got[0][0], got[0][1]) if got else None
        if pair != expected:
            failures += 1
            print(f"FAIL {text!r}: got {pair}, expected {expected}")
    # "substituted A for B" with both in the recipe has unrecoverable direction -> skipped
    both = Miner(vocab, {1: frozenset({"honey", "sugar"})}).mine_sentence("i substituted honey for the sugar", 1)
    single = Miner(vocab, {1: frozenset({"sugar"})}).mine_sentence("i substituted honey for the sugar", 1)
    if both or [(a, b) for a, b, *_ in single] != [("sugar", "honey")]:
        failures += 1
        print("FAIL ambiguous 'substituted A for B' handling")
    print(f"{len(CASES) + 1 - failures}/{len(CASES) + 1} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
