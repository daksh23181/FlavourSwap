"""Health scoring shared by Deliverables 3-5.

Health score per 100 g (lower = healthier), in %Daily-Value units:

    energy/2000 kcal + saturated fat/20 g + free sugars/50 g + sodium/2300 mg + trans fat/2.2 g
    - fiber/28 g

Sugars count only as *free* sugars (WHO): sugars inside intact fruit and vegetables and the milk
sugar of plain dairy are not penalised; juices, syrups and sweetened products are.

The choice of nutrients follows established nutrient-profiling systems such as Nutri-Score
(penalise energy, saturated fat, sugars and sodium; credit fiber; total fat is NOT penalised, so
olive oil or nuts are not punished for unsaturated fat). Reference values are the FDA Daily Values
(21 CFR 101.9; total sugars against the 50 g added-sugars DV); trans fat uses the WHO limit
(< 1% of energy = 2.2 g/day at 2,000 kcal). The score can be negative for very high-fiber foods.

SR Legacy often leaves trans fat unreported. `estimate_trans_fat` fills a separate *estimated*
column (the raw column is never modified):
    reported                      -> the reported value, except a reported 0 for a fatty ruminant-fat
                                     food (label rounding; natural trans fat is ~3-5% of milk/beef fat),
                                     which is estimated as below
    total fat <= 0.5 g            -> 0 (trans fat cannot exceed total fat)
    possible industrial trans fat -> unknown (margarine / shortening / spread / topping-type products,
                                     and baked goods, snacks and sweets, which may contain partially
                                     hydrogenated oil)
    otherwise                     -> total fat x the median trans/fat ratio of SR Legacy foods in the
                                     same USDA category (natural ruminant trans fat in dairy and beef ~3-5%
                                     of fat; ~0 for plant foods)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

LIMIT = {"energy_kcal": 2000.0, "sat_fat_g": 20.0, "sugar_g": 50.0, "sodium_mg": 2300.0,
         "trans_fat_g_est": 2.2}
CREDIT = {"fiber_g": 28.0}
INDUSTRIAL_FAT = (r"hydrogenated|margarine|shortening|spread|frosting|imitation|non-?dairy|creamer|"
                  r"whipped topping|dessert topping")
# Animal fats that USDA files under "Fats and Oils" carry the natural trans fat of their source.
ANIMAL_FAT_SOURCE = [(r"^Butter\b|\bghee\b", "Dairy and Egg Products"),
                     (r"\btallow\b|\bsuet\b", "Beef Products"),
                     (r"^Lard\b", "Pork Products"),
                     (r"^Fat, chicken\b", "Poultry Products")]
# Negated phrases are removed from USDA descriptions before the patterns below are applied
# ("unsweetened", "not sweetened", "without salt" must not count as sweetened / salted).
NEGATED = (r"\b(?:unsweetened|not sweetened|no sugar added|without added sugar|sugar[- ]free|unsalted|"
           r"without salt|no salt added|salt[- ]free)\b")
# Fruit/vegetable products whose sugars are free sugars (WHO): juices, syrups, sweetened products.
FREE_SUGAR_PRODUCTS = (r"\b(?:juices?|syrups?|sweetened|candied|maraschino|jams?|jelly|jellies|preserves?|"
                       r"nectars?|marmalade|pie filling|sauces?|catsup|ketchup|drinks?|cocktails?|glazed?|"
                       r"sugars?|honey)\b")
# Processing that adds salt, fat or sugar to an otherwise whole plant food.
PROCESSED_PLANT = (r"\b(?:salt|salted|butter|oils?|fried|refried|baked|pork|sauces?|chips|spread|paste|"
                   r"cream|creamed|breaded|batter|pickled?|sauerkraut|catsup|ketchup|maraschino|mashed)\b")
WHOLE_PLANT_GROUPS = {"Fruits", "Vegetables", "Legumes, Nuts & Seeds"}
# Dairy products with added sugar (plain milk, cream, cheese, plain yogurt only contain milk sugar).
SWEETENED_DAIRY = (r"\b(?:sweetened|flavou?red|vanilla|fruit|chocolate|strawberry|ice creams?|condensed|"
                   r"frozen|pudding|eggnog|dessert|whipped topping|sugars?)\b")
RUMINANT_CATEGORIES = {"Dairy and Egg Products", "Beef Products", "Lamb, Veal, and Game Products"}
TRANS_UNKNOWN_CATEGORIES = {"Baked Products", "Snacks", "Sweets", "Fast Foods", "Restaurant Foods",
                            "Meals, Entrees, and Side Dishes"}


def category_trans_ratios(fat_trans: pd.DataFrame) -> tuple[dict[str, float], float]:
    """Median trans/fat ratio per USDA category over foods with fat >= 1 g and reported trans fat,
    excluding industrial-fat products. `fat_trans` has columns fat_g, trans_fat_g, category,
    description (one row per SR Legacy food)."""
    ok = (fat_trans.fat_g >= 1) & fat_trans.trans_fat_g.notna() & \
        ~fat_trans.description.str.contains(INDUSTRIAL_FAT, case=False, regex=True)
    r = fat_trans[ok].assign(ratio=lambda d: d.trans_fat_g / d.fat_g)
    return r.groupby("category").ratio.median().to_dict(), float(r.ratio.median())


def estimate_trans_fat(ing: pd.DataFrame, ratios: dict[str, float], overall: float) -> tuple[pd.Series, pd.Series]:
    """Return (trans_fat_g_est, method) for the ingredient table (needs fat_g, trans_fat_g,
    usda_category, usda_description, usda_fdc_id)."""
    est = pd.Series(np.nan, index=ing.index)
    method = pd.Series("no_usda_link", index=ing.index, dtype=object)
    linked = ing.usda_fdc_id.notna()
    ratio_cat = ing.usda_category.copy()
    for pattern, cat in ANIMAL_FAT_SOURCE:
        ratio_cat[ing.usda_description.fillna("").str.contains(pattern, case=False, regex=True)] = cat
    # A reported 0 g for a fatty ruminant-fat food (milk fat, beef, lamb) is a label-rounding
    # artefact (natural trans fat is ~3-5% of their fat), e.g. "Butter, Clarified butter (ghee)".
    ruminant = ratio_cat.isin(RUMINANT_CATEGORIES) & \
        ~ing.usda_description.fillna("").str.startswith("Egg")
    implausible_zero = linked & (ing.trans_fat_g == 0) & ruminant & (ing.fat_g >= 5)
    reported = linked & ing.trans_fat_g.notna() & ~implausible_zero
    est[reported] = ing.trans_fat_g[reported]
    method[reported] = "reported"
    # trans fat <= total fat: a (near) fat-free food has no trans fat, whatever its category
    no_fat = linked & ~reported & (ing.fat_g <= 0.5)
    est[no_fat] = 0.0
    method[no_fat] = "implied_by_zero_fat"
    reported = reported | no_fat
    industrial = ing.usda_description.fillna("").str.contains(INDUSTRIAL_FAT, case=False, regex=True) | \
        ing.usda_category.isin(TRANS_UNKNOWN_CATEGORIES)
    unknown = linked & ~reported & industrial
    method[unknown] = "unknown_possible_industrial"
    rest = linked & ~reported & ~industrial & ing.fat_g.notna()
    est[rest] = ing.fat_g[rest] * ratio_cat[rest].map(ratios).fillna(overall)
    method[rest] = "category_ratio"
    method[implausible_zero & rest] = "category_ratio_reported_zero_implausible"
    return est, method


def _desc(ing: pd.DataFrame) -> pd.Series:
    """Lower-case USDA description with negated phrases removed."""
    return ing.usda_description.fillna("").str.lower().str.replace(NEGATED, " ", regex=True)


def intrinsic_sugar_only(ing: pd.DataFrame) -> pd.Series:
    """True where the sugars are intrinsic (inside intact fruit/vegetables) or milk sugar of plain dairy.

    The WHO sugar guideline limits *free* sugars - sugars added to foods plus those in honey, syrups
    and fruit juices - and explicitly excludes sugars in intact fruit, vegetables and milk. Juices,
    syrups and sweetened or flavored products keep the sugar penalty."""
    desc = _desc(ing)
    plant = ing.food_group.isin(["Fruits", "Vegetables"]) & ~desc.str.contains(FREE_SUGAR_PRODUCTS, case=False, regex=True)
    dairy = ing.food_group.eq("Dairy") & ~desc.str.contains(SWEETENED_DAIRY, case=False, regex=True)
    return plant | dairy


def whole_plant_food(ing: pd.DataFrame) -> pd.Series:
    """Unprocessed fruit, vegetables, legumes, nuts and seeds - the foods dietary guidelines (and
    Nutri-Score's fruit/vegetable/legume/nut credit) already recommend. FlavorSwap does not rank them
    against each other for health; processed forms (juices, syrup-packed, salted, refried, ...) are
    not whole plant foods."""
    desc = _desc(ing)
    return ing.food_group.isin(WHOLE_PLANT_GROUPS) & ing.usda_fdc_id.notna() & \
        ~desc.str.contains(FREE_SUGAR_PRODUCTS, case=False, regex=True) & \
        ~desc.str.contains(PROCESSED_PLANT, case=False, regex=True)


def _score_inputs(ing: pd.DataFrame) -> pd.DataFrame:
    """Score components per 100 g. Missing sugars / saturated fat / fiber are filled only when
    logically implied (each is bounded by its parent value: sugars and fiber <= carbohydrate,
    saturated fat <= total fat). Sugars count only when they are free sugars (`intrinsic_sugar_only`)."""
    df = ing[list(LIMIT) + list(CREDIT) + ["carbs_g", "fat_g"]].astype(float).copy()
    df.loc[intrinsic_sugar_only(ing).to_numpy(), "sugar_g"] = 0.0
    df.loc[df.sugar_g.isna() & (df.carbs_g <= 0.5), "sugar_g"] = 0.0
    df.loc[df.fiber_g.isna() & (df.carbs_g <= 0.5), "fiber_g"] = 0.0
    df.loc[df.sat_fat_g.isna() & (df.fat_g <= 0.5), "sat_fat_g"] = 0.0
    return df


def limit_scores(ing: pd.DataFrame) -> np.ndarray:
    """The 'nutrients to limit' part of the health score (energy, saturated fat, free sugars, sodium,
    trans fat; no fiber credit) per 100 g; NaN if any component is unknown."""
    df = _score_inputs(ing)
    return sum(df[c] / v for c, v in LIMIT.items()).to_numpy()


def health_scores(ing: pd.DataFrame) -> np.ndarray:
    """Health score per 100 g (NaN if any component is unknown): `limit_scores` minus the fiber credit."""
    df = _score_inputs(ing)
    return limit_scores(ing) - sum(df[c] / v for c, v in CREDIT.items()).to_numpy()
