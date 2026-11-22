"""Regression tests for ingredient-line parsing and gram conversion (Deliverable 5).

Run:  python tests/test_quantities.py      (exits non-zero on failure)
Requires data/processed/ingredients.csv and data/raw/usda_sr_legacy/food_portion.csv.
"""

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from flavorswap.quantities import ITEM_DEFAULTS, PortionTable, parse_line  # noqa: E402

# (raw line, canonical ingredient, expected grams or None when it must not convert)
CASES = [
    ("1 1/2  cups    flour", "all purpose flour", 187.5),
    ("2   large    eggs", "egg", 100.0),
    ("3   eggs", "egg", 150.0),                                   # bare count -> large eggs
    ("1   cup    butter or 1   cup    margarine, softened ", "butter", 227.0),   # first alternative
    ("1   stick butter", "butter", 113.0),
    ("1 (14   ounce) can   sweetened condensed milk", "sweetened condensed milk", 396.9),
    ("2 (15   ounce) cans   chicken broth", "chicken broth", 850.5),
    ("1/2  teaspoon    salt", "salt", 3.0),
    ("1   pinch salt", "salt", 0.375),
    ("1   lb    ground beef", "ground beef", 453.6),
    ("3   cloves    garlic, minced", "garlic", 9.0),
    ("1   medium    onion, chopped", "onion", 110.0),
    ("1       onion, chopped", "onion", 110.0),
    ("3/4  cup   firmly packed brown sugar", "brown sugar", 165.0),   # "packed", not "unpacked"
    ("1 cup brown sugar", "brown sugar", 220.0),              # brown sugar is measured packed
    ("2 cups cheddar cheese, shredded", "cheddar cheese", 226.0),   # form-specific cup weight
    ("1 cup onion, chopped", "onion", 160.0),
    ("250   g    sugar", "sugar", 250.0),
    ("1   tablespoon    olive oil", "olive oil", 13.5),
    ("8   fl oz milk", "milk", 244.0),
    ("1       lemon, juice of", "lemon juice", 48.0),             # USDA "1 lemon yields"
    ("3 -4   stalks    celery, diced ", "celery", None),          # checked separately (range)
    ("  salt", "salt", None),                                     # no amount -> None
]


def main() -> int:
    portions = PortionTable(pd.read_csv(ROOT / "data" / "raw" / "usda_sr_legacy" / "food_portion.csv"))
    ing = pd.read_csv(ROOT / "data" / "processed" / "ingredients.csv").dropna(subset=["usda_fdc_id"])
    fid = dict(zip(ing.ingredient, ing.usda_fdc_id.astype(int)))
    failures = 0
    for line, name, expected in CASES:
        g = portions.to_grams(fid[name], parse_line(line), ITEM_DEFAULTS.get(name))
        if expected is None:
            continue
        if g is None or abs(g - expected) / expected > 0.02:
            failures += 1
            print(f"FAIL {line!r} ({name}): got {g}, expected {expected}")
    p = parse_line("3 -4   stalks    celery, diced ")
    if not (p.amount == 3.5 and p.unit == "stalks"):
        failures += 1
        print(f"FAIL range parse: {p}")
    if parse_line("  salt").amount is not None or portions.to_grams(fid["salt"], parse_line("  salt")) is not None:
        failures += 1
        print("FAIL: a line without amount must not convert")
    p = parse_line("8 (8   inch)    flour tortillas")
    if not (p.amount == 8 and p.unit is None):
        failures += 1
        print(f"FAIL: '(8 inch)' is a size, not an amount: {p}")
    n = len(CASES) + 3
    print(f"{n - failures}/{n} passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
