"""Link canonical recipe ingredients to (a) Ahn et al. flavor-network ingredients and
(b) USDA FoodData Central SR Legacy foods.

Both linkers are precision-first:
  * a hand-curated override table (data/curation/*.csv) always wins;
  * automatic matches must contain the ingredient's head noun, and are tagged with the method
    used so downstream users can filter on confidence.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from .normalize import _singular

# ============================================================================================
# Ahn et al. (2011) flavor network
# ============================================================================================

# Modifiers that may be dropped when falling back from "red onion" to Ahn's "onion".
# They describe colour/form/processing/fat level but not a different food.
AHN_SAFE_MODIFIERS = {
    "red", "white", "yellow", "green", "purple", "golden", "black", "brown",
    "fresh", "frozen", "canned", "dried", "dry", "cooked", "raw", "whole", "baby", "large", "small",
    "chopped", "diced", "sliced", "minced", "crushed", "grated", "shredded", "ground", "mashed",
    "plain", "unsalted", "salted", "sweet", "sharp", "mild", "aged", "light", "lite", "low", "fat",
    "free", "non", "reduced", "skim", "part", "lean", "extra", "boneless", "skinless", "toasted",
    "roasted", "organic", "unsweetened", "sweetened", "seedless", "pitted", "peeled", "ripe",
    "1%", "2%", "sodium", "less", "spanish", "vidalia", "italian", "english", "hothouse",
    "granny", "smith", "russet", "yukon", "gold", "new", "idaho", "roma", "plum", "cherry", "grape",
    "kalamata", "navel", "bartlett", "bosc", "anjou", "medjool", "button", "cremini", "portabella",
    "crimini", "portobello", "jumbo", "sea", "tiger", "cold", "smoked",
}

# Generic Ahn entries that must never be reached by the head-noun fallback: "roasted red pepper"
# must not become Piper "pepper", "bay leaf" must not become the generic "leaf".
AHN_HEAD_BLOCK = {"pepper", "leaf", "seed", "nut", "bean", "root", "plant", "fruit", "flower",
                  "vegetable", "wood", "oil", "juice", "cereal", "grain", "meat", "fish", "animal"}


def _key(name: str) -> str:
    """Comparable key: underscores->spaces, singular last token."""
    toks = name.replace("_", " ").lower().split()
    if toks:
        toks[-1] = _singular(toks[-1])
    return " ".join(toks)


@dataclass
class AhnIndex:
    by_key: dict[str, str]        # key -> ahn name (underscored)
    categories: dict[str, str]    # ahn name -> category
    n_compounds: dict[str, int]   # ahn name -> number of compounds


def build_ahn_index(ingr_info: pd.DataFrame, ingr_comp: pd.DataFrame) -> AhnIndex:
    n = ingr_comp.groupby("ingredient_id").compound_id.nunique()
    by_key: dict[str, str] = {}
    cats, ncomp = {}, {}
    for r in ingr_info.itertuples(index=False):
        cats[r.name] = r.category
        ncomp[r.name] = int(n.get(r.id, 0))
        if ncomp[r.name] == 0:
            continue  # an Ahn ingredient with no compounds carries no flavor information
        by_key.setdefault(_key(r.name), r.name)
    return AhnIndex(by_key, cats, ncomp)


def match_ahn(ingredient: str, idx: AhnIndex, overrides: dict[str, str | None]) -> tuple[str | None, str]:
    """Return (ahn_name or None, method)."""
    if ingredient in overrides:
        target = overrides[ingredient]
        if target is None:
            return None, "manual_none"
        if target not in idx.categories:
            raise ValueError(f"Ahn override for {ingredient!r} -> {target!r} is not an Ahn ingredient")
        return target, "manual"

    k = _key(ingredient)
    if k in idx.by_key:
        return idx.by_key[k], "exact"

    toks = k.split()
    # Longest suffix whose dropped prefix consists only of safe modifiers.
    for i in range(1, len(toks)):
        dropped, rest = toks[:i], " ".join(toks[i:])
        if rest in AHN_HEAD_BLOCK:
            continue
        if rest in idx.by_key and all(t in AHN_SAFE_MODIFIERS for t in dropped):
            return idx.by_key[rest], "head_noun"
    return None, "unmatched"


# ============================================================================================
# USDA FoodData Central (SR Legacy)
# ============================================================================================

NUTRIENTS = {  # FDC nutrient id -> output column (all per 100 g edible portion)
    1008: "energy_kcal",
    1003: "protein_g",
    1004: "fat_g",
    1258: "sat_fat_g",
    1005: "carbs_g",
    2000: "sugar_g",
    1079: "fiber_g",
    1093: "sodium_mg",
    1253: "cholesterol_mg",
    1257: "trans_fat_g",
    1051: "water_g",
}

# Categories that hold composite dishes, branded products or baby food: never auto-matched.
USDA_EXCLUDED_CATEGORIES = {
    "Baby Foods", "Fast Foods", "Meals, Entrees, and Side Dishes", "Restaurant Foods",
    "Branded Food Products Database", "Quality Control Materials",
    "American Indian/Alaska Native Foods",
}

_PROCESS_WORDS = {
    "cooked", "boiled", "roasted", "fried", "braised", "stewed", "baked", "broiled", "grilled",
    "simmered", "microwaved", "steamed", "pan", "heated", "toasted", "drained", "prepared",
}
_STATE_WORDS = {"canned", "frozen", "dehydrated", "dried", "dry", "powder", "concentrate", "juice"}
# Parts / by-products / composite forms: a different food unless the ingredient names them.
_PART_WORDS = {"skin", "giblet", "giblets", "carcass", "roll", "loaf", "blend", "variety", "product", "by",
               "restaurant", "spread", "substitute", "imitation", "meatless", "flavored", "mix"}
# Nutritionally different variants that should only match when the ingredient asks for them.
_VARIANT_WORDS = {"reduced", "low", "free", "nonfat", "lite", "light", "lowfat", "sodium", "calcium",
                  "fortified", "instant", "diet", "unsweetened", "sweetened", "salted", "soy"}
_STOP = {"and", "or", "with", "without", "of", "in", "the", "a", "to", "for", "from", "added", "ns", "as"}
_TOKEN = re.compile(r"[a-z0-9%]+")


def _tokens(text: str) -> list[str]:
    return [_singular(t) for t in _TOKEN.findall(text.lower().replace("-", " "))]


@dataclass
class UsdaFood:
    fdc_id: int
    description: str
    category: str
    tokens: frozenset
    first_seg: frozenset
    first_word: str
    has_brand: bool


def build_usda_index(food: pd.DataFrame, categories: pd.DataFrame) -> list[UsdaFood]:
    """Candidate foods (excluded categories removed)."""
    cat = dict(zip(categories.id, categories.description))
    out = []
    for r in food.itertuples(index=False):
        c = cat.get(r.food_category_id, "")
        if c in USDA_EXCLUDED_CATEGORIES:
            continue
        segs = [s.strip() for s in r.description.split(",")]
        toks = _tokens(r.description)
        first = _tokens(segs[0])
        # Brand names are written in capitals in SR Legacy ("KRAFT", "PILLSBURY").
        has_brand = bool(re.search(r"\b[A-Z][A-Z'&]{2,}\b", r.description))
        out.append(UsdaFood(int(r.fdc_id), r.description, c, frozenset(toks), frozenset(first),
                            first[0] if first else "", has_brand))
    return out


def score_usda(ingredient: str, f: UsdaFood) -> tuple[float, float]:
    """Return (score, coverage). Score is -inf when the head noun is missing."""
    t = [w for w in _tokens(ingredient) if w not in _STOP]
    if not t:
        return float("-inf"), 0.0
    head = t[-1]
    if head not in f.tokens:
        return float("-inf"), 0.0
    tset = set(t)
    matched = tset & f.tokens
    coverage = len(matched) / len(tset)
    extra = len({w for w in f.tokens if w not in _STOP} - tset)
    # Words in the leading term that the ingredient lacks ("Sugar-apples" for "sugar",
    # "Cinnamon buns" for "cinnamon") signal a different food.
    first_extra = len({w for w in f.first_seg if w not in _STOP} - tset)

    s = 3.0 * coverage
    s += 1.5 if head in f.first_seg else 0.0
    s += 1.0 if f.first_word in tset else 0.0
    s -= 0.12 * extra
    s -= 1.0 * first_extra
    want_process = bool(tset & _PROCESS_WORDS)
    if not want_process:
        if "raw" in f.tokens:
            s += 0.6
        if f.tokens & _PROCESS_WORDS:
            s -= 0.6
    for w in _STATE_WORDS:
        if w in f.tokens and w not in tset:
            s -= 0.5
    for w in _VARIANT_WORDS:
        if w in f.tokens and w not in tset:
            s -= 0.8
    for w in _PART_WORDS:
        if w in f.tokens and w not in tset:
            s -= 1.0
    if f.has_brand:
        s -= 1.5
    return s, coverage


def invert_usda_index(index: list[UsdaFood]) -> dict[str, list[UsdaFood]]:
    """token -> foods containing it (an ingredient's head noun must occur in the description)."""
    inv: dict[str, list[UsdaFood]] = {}
    for f in index:
        for t in f.tokens:
            inv.setdefault(t, []).append(f)
    return inv


def match_usda(ingredient: str, inv: dict[str, list[UsdaFood]], by_id: dict[int, UsdaFood],
               overrides: dict[str, int | None]) -> tuple[int | None, str, float]:
    """Return (fdc_id or None, method, score). `by_id` must cover every SR Legacy food so that
    manual overrides may point anywhere (even into categories the automatic matcher skips)."""
    if ingredient in overrides:
        fid = overrides[ingredient]
        if fid is None:
            return None, "manual_none", float("nan")
        if fid not in by_id:
            raise ValueError(f"USDA override for {ingredient!r}: fdc_id {fid} not found / excluded")
        return fid, "manual", float("nan")

    toks = [w for w in _tokens(ingredient) if w not in _STOP]
    if not toks:
        return None, "unmatched", float("nan")
    best, best_s, best_cov = None, float("-inf"), 0.0
    for f in inv.get(toks[-1], []):
        s, cov = score_usda(ingredient, f)
        if s > best_s or (s == best_s and best is not None and len(f.description) < len(best.description)):
            best, best_s, best_cov = f, s, cov
    if best is None or best_s == float("-inf"):
        return None, "unmatched", float("nan")
    if best_cov == 1.0 and best_s >= 4.0:
        return best.fdc_id, "auto_high", best_s
    if best_cov >= 0.5 and best_s >= 2.5:
        return best.fdc_id, "auto_low", best_s
    return None, "unmatched", best_s
