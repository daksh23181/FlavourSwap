"""Ingredient quantities -> grams (Deliverable 5).

The Food.com "recipes with search terms" dump keeps the raw ingredient lines with amounts and units
("1 1/2 cups all-purpose flour", "1 (14 ounce) can sweetened condensed milk", "2 large eggs").
This module parses a line into (amount, unit, text) and converts it to grams for a specific USDA
food using that food's own household-measure weights from SR Legacy `food_portion.csv`
(e.g. 1 cup flour = 125 g, 1 stick butter = 113 g, 1 large egg = 50 g, 1 clove garlic = 3 g).

Nothing is guessed: if a quantity cannot be converted for that food, `to_grams` returns None and
the recipe is reported as unresolved.
"""

from __future__ import annotations

import html
import re
from dataclasses import dataclass

import pandas as pd

# ---------------------------------------------------------------------------------- units
MASS_G = {"g": 1.0, "gram": 1.0, "grams": 1.0, "gr": 1.0, "kg": 1000.0, "kilogram": 1000.0,
          "kilograms": 1000.0, "oz": 28.3495, "ounce": 28.3495, "ounces": 28.3495,
          "lb": 453.592, "lbs": 453.592, "pound": 453.592, "pounds": 453.592}
# volume expressed in US cups
VOLUME_CUP = {"cup": 1.0, "cups": 1.0, "c": 1.0,
              "tablespoon": 1 / 16, "tablespoons": 1 / 16, "tbsp": 1 / 16, "tbs": 1 / 16, "tbl": 1 / 16,
              "teaspoon": 1 / 48, "teaspoons": 1 / 48, "tsp": 1 / 48,
              "ml": 1 / 236.588, "milliliter": 1 / 236.588, "milliliters": 1 / 236.588,
              "millilitre": 1 / 236.588, "millilitres": 1 / 236.588, "cl": 10 / 236.588,
              "dl": 100 / 236.588, "l": 1000 / 236.588, "liter": 1000 / 236.588,
              "liters": 1000 / 236.588, "litre": 1000 / 236.588, "litres": 1000 / 236.588,
              "pint": 2.0, "pints": 2.0, "quart": 4.0, "quarts": 4.0, "gallon": 16.0, "gallons": 16.0,
              "pinch": 1 / 768, "pinches": 1 / 768, "dash": 1 / 384, "dashes": 1 / 384,
              "drop": 1 / 4800, "drops": 1 / 4800, "smidgen": 1 / 1536}
FLUID_OUNCE_CUP = 1 / 8
# count descriptors -> USDA portion modifier words to look for (in order of preference)
COUNT_UNITS = {
    "large": ["large"], "medium": ["medium"], "small": ["small"],
    "extra-large": ["extra large", "large"], "jumbo": ["jumbo", "extra large", "large"],
    "clove": ["clove"], "cloves": ["clove"], "stick": ["stick"], "sticks": ["stick"],
    "slice": ["slice"], "slices": ["slice"], "stalk": ["stalk"], "stalks": ["stalk"],
    "rib": ["stalk", "rib"], "ribs": ["stalk", "rib"], "head": ["head"], "heads": ["head"],
    "bunch": ["bunch"], "bunches": ["bunch"], "sprig": ["sprig"], "sprigs": ["sprig"],
    "leaf": ["leaf"], "leaves": ["leaf"], "ear": ["ear"], "ears": ["ear"],
    "fillet": ["fillet"], "fillets": ["fillet"], "piece": ["piece"], "pieces": ["piece"],
    "can": ["can"], "cans": ["can"], "package": ["package"], "packages": ["package"],
    "jar": ["jar"], "jars": ["jar"], "container": ["container"], "containers": ["container"],
    "envelope": ["envelope", "packet"], "envelopes": ["envelope", "packet"],
    "packet": ["packet", "envelope"], "packets": ["packet", "envelope"],
    "bottle": ["bottle"], "bottles": ["bottle"], "box": ["box", "package"], "boxes": ["box", "package"],
    "bag": ["bag", "package"], "bags": ["bag", "package"], "breast": ["breast"], "breasts": ["breast"],
    "thigh": ["thigh"], "thighs": ["thigh"], "whole": ["whole", "medium", "large"],
}
# portion modifiers that describe a single whole item, used when a line has a bare count ("2 eggs")
ITEM_WORDS = ["medium", "large", "whole", "small", "unit", "item", "fruit", "each", "piece", "egg",
              "clove", "ear", "stalk", "breast", "fillet", "slice", "yields"]   # "1 lemon yields" (juice)
VOLUME_OR_MASS_PREFIXES = ("cup", "tbsp", "tsp", "tablespoon", "teaspoon", "oz", "fl oz", "lb",
                           "quart", "pint", "cubic", "dash", "pinch", "serving", "nlea", "liter")
# US recipes mean large eggs when they say "2 eggs"
ITEM_DEFAULTS = {"egg": ["large"], "egg white": ["large"], "egg yolk": ["large"]}
FORM_WORDS = ["shredded", "grated", "chopped", "diced", "sliced", "minced", "packed", "unpacked",
              "melted", "crumbled", "mashed", "whole", "ground", "cubed", "pieces", "halves", "pureed",
              "sifted"]
PACKAGE_WORDS = {"can", "cans", "package", "packages", "pkg", "jar", "jars", "container",
                 "containers", "box", "boxes", "bag", "bags", "bottle", "bottles", "envelope",
                 "envelopes", "packet", "packets", "carton", "cartons", "tub", "tubs", "loaf"}

_NUM = r"(?:\d+\s+\d+/\d+|\d+/\d+|\d+(?:\.\d+)?)"
_QTY = re.compile(rf"^\s*({_NUM})(?:\s*-\s*({_NUM}))?")
_PAREN_SIZE = re.compile(rf"^\s*\(\s*({_NUM})\s*-?\s*([a-zA-Z\.]+(?:\s+oz)?)\s*\)\s*")
_TAGS = re.compile(r"<[^>]+>")
_WS = re.compile(r"\s+")
_ALT = re.compile(r"\s+or\s+(?=\d)")       # "1 cup butter or 1 cup margarine" -> first alternative


def _num(s: str) -> float:
    s = s.strip()
    if " " in s:
        whole, frac = s.split(None, 1)
        return float(whole) + _num(frac)
    if "/" in s:
        a, b = s.split("/", 1)
        return float(a) / float(b) if float(b) else float("nan")
    return float(s)


@dataclass
class ParsedLine:
    amount: float | None      # number of units (None = no amount given, e.g. "salt to taste")
    unit: str | None          # normalised unit word, None for a bare count
    text: str                 # remaining lower-case text (ingredient + preparation words)


def clean_line(line: str) -> str:
    s = html.unescape(_TAGS.sub(" ", str(line)))
    s = _ALT.split(s)[0]
    return _WS.sub(" ", s).strip()


def parse_line(line: str) -> ParsedLine:
    s = clean_line(line)
    m = _QTY.match(s)
    if not m:
        return ParsedLine(None, None, s.lower())
    a = _num(m.group(1))
    if m.group(2):
        a = (a + _num(m.group(2))) / 2           # range "3 -4" -> 3.5
    rest = s[m.end():]
    # package with size: "1 (14 ounce) can ...", "2 (15 oz) cans ..."
    p = _PAREN_SIZE.match(rest)
    if p:
        inner_unit = p.group(2).lower().rstrip(".")
        after = rest[p.end():]
        first = after.split()[0].lower().rstrip(",.") if after.split() else ""
        if inner_unit in ("fl oz", "fluid") or inner_unit in MASS_G or inner_unit in VOLUME_CUP:
            size = _num(p.group(1))
            text = after
            if first in PACKAGE_WORDS:
                text = after.split(None, 1)[1] if len(after.split()) > 1 else ""
            unit = "fl oz" if inner_unit in ("fl oz", "fluid") else inner_unit
            return ParsedLine(a * size, unit, text.lower().strip())
        rest = after                              # "(8 inch) tortillas": size is not an amount
    words = rest.split()
    if not words:
        return ParsedLine(a, None, "")
    w0 = words[0].lower().rstrip(".,")
    if w0 in ("fl", "fluid") and len(words) > 1 and words[1].lower().rstrip(".,") in ("oz", "ounce", "ounces"):
        return ParsedLine(a, "fl oz", " ".join(words[2:]).lower())
    if w0 == "extra" and len(words) > 1 and words[1].lower().rstrip(".,") == "large":
        return ParsedLine(a, "extra-large", " ".join(words[2:]).lower())
    if w0 in MASS_G or w0 in VOLUME_CUP or w0 in COUNT_UNITS:
        return ParsedLine(a, w0, " ".join(words[1:]).lower())
    return ParsedLine(a, None, rest.lower().strip())


# ---------------------------------------------------------------------------------- portions
class PortionTable:
    """USDA household measures per food: grams per cup and per countable item."""

    def __init__(self, portions: pd.DataFrame):
        p = portions.copy()
        p["modifier"] = p.modifier.fillna("").str.lower().str.strip()
        p = p[(p.amount > 0) & (p.gram_weight > 0)]
        p["g_per_unit"] = p.gram_weight / p.amount
        self.by_food = {fid: list(zip(g.modifier, g.g_per_unit)) for fid, g in p.groupby("fdc_id")}

    def grams_per_cup(self, fid: int, text: str) -> float | None:
        rows = self.by_food.get(fid, [])
        forms = [f for f in FORM_WORDS if re.search(rf"\b{f}\b", text)]   # "packed" != "unpacked"
        cands = []
        for mod, g in rows:
            if mod.startswith("cup"):
                cands.append((mod, g))
            elif mod.startswith(("tbsp", "tablespoon")):
                cands.append((mod, g * 16))
            elif mod.startswith(("tsp", "teaspoon")):
                cands.append((mod, g * 48))
            elif mod.startswith("fl oz"):
                cands.append((mod, g * 8))
            elif mod.startswith("quart"):
                cands.append((mod, g / 4))
        if not cands:
            return None
        cands.sort(key=lambda c: not c[0].startswith("cup"))   # direct cup weights are most precise
        for f in forms:                               # match the stated form ("shredded", "packed")
            hit = [g for mod, g in cands if re.search(rf"\b{f}\b", mod)]
            if hit:
                return hit[0]
        plain = [g for mod, g in cands if mod in ("cup", "tbsp", "tsp", "tablespoon", "teaspoon")]
        return plain[0] if plain else cands[0][1]

    def grams_per_item(self, fid: int, words: list[str]) -> float | None:
        """Weight of one item described by the first matching word ("large", "clove", "stick",
        "can", ...). Volume/weight measures ("cup, chopped", "oz") are never treated as items."""
        rows = self.by_food.get(fid, [])
        for w in words:
            for mod, g in rows:
                head = re.split(r"[,(]", mod)[0].strip()
                if head.startswith(VOLUME_OR_MASS_PREFIXES):
                    continue
                if re.search(rf"\b{re.escape(w)}\b", head):
                    return g
        return None

    def to_grams(self, fid: int, parsed: ParsedLine, item_default: list[str] | None = None) -> float | None:
        a, unit, text = parsed.amount, parsed.unit, parsed.text
        if a is None or not a == a or a <= 0:
            return None
        if unit in MASS_G:
            return a * MASS_G[unit]
        if unit == "fl oz":
            cup = self.grams_per_cup(fid, text)
            return a * FLUID_OUNCE_CUP * cup if cup else None
        if unit in VOLUME_CUP:
            cup = self.grams_per_cup(fid, text)
            return a * VOLUME_CUP[unit] * cup if cup else None
        if unit in COUNT_UNITS:
            g = self.grams_per_item(fid, COUNT_UNITS[unit])
            return a * g if g else None
        # bare count: "2 eggs", "1 green pepper"
        g = self.grams_per_item(fid, (item_default or []) + ITEM_WORDS)
        return a * g if g else None
