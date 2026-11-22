"""Deliverable 5 (step 1) - ingredient amounts in grams for every recipe, validated against Food.com.

Food.com's ingredient *names* (RAW_recipes) carry no quantities. The "recipes with search terms"
dump of the same site (same recipe ids) keeps the raw lines ("1 1/2 cups flour"), the number of
servings and the serving size in grams. Each line is matched to its ingredient, normalised with the
Deliverable 1 normaliser, and converted to grams with that ingredient's USDA household measures.

Validation (no Food.com number is used to *fit* anything):
  * reconstructed mass per serving  vs. Food.com serving size (g)
  * reconstructed kcal per serving  vs. Food.com calories per serving
A recipe enters the analysis set only if every quantified main ingredient was converted and its
reconstructed calories are within CALORIE_TOLERANCE of Food.com's value.

Outputs
  data/processed/recipe_ingredient_amounts.parquet   one row per ingredient line
  data/processed/recipe_amounts_summary.parquet      one row per recipe (+ validation fields)
  reports/quantity_validation.md

Run:  python src/build_recipe_quantities.py
"""

from __future__ import annotations

import ast
import json
import sys
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flavorswap import paths  # noqa: E402
from flavorswap.normalize import NON_FOOD, canonicalize  # noqa: E402
from flavorswap.quantities import ITEM_DEFAULTS, PortionTable, clean_line, parse_line  # noqa: E402

CALORIE_TOLERANCE = 0.20          # |reconstructed / Food.com - 1| for the analysis set
MAX_SERVINGS = 100
MINOR_GROUPS = {"Herbs & Spices"}  # unconvertible lines of these are negligible by mass
NUTRIENTS = ["energy_kcal", "fat_g", "sat_fat_g", "trans_fat_g_est", "sugar_g", "sodium_mg",
             "fiber_g", "protein_g", "carbs_g"]


@lru_cache(maxsize=None)
def canon(name: str) -> tuple[str, ...]:
    return tuple(c for c in canonicalize(name) if c not in NON_FOOD)


def match_line(line_lc: str, names: list[str]) -> str | None:
    """The recipe's ingredient name that occurs in the line (longest wins)."""
    hits = [n for n in names if n and n in line_lc]
    return max(hits, key=len) if hits else None


def main() -> None:
    t0 = time.time()
    ing = pd.read_csv(paths.PROCESSED / "ingredients.csv")
    vocab = set(ing.ingredient)
    fid = dict(zip(ing.ingredient, ing.usda_fdc_id))
    group = dict(zip(ing.ingredient, ing.food_group))
    nut = ing.set_index("ingredient")[NUTRIENTS]
    portions = PortionTable(pd.read_csv(paths.USDA_DIR / "food_portion.csv"))

    rec = pd.read_parquet(paths.PROCESSED / "recipes.parquet",
                          columns=["recipe_id", "name", "cuisine", "calories", "calories_outlier"])
    terms = pd.read_csv(paths.RAW / "foodcom" / "recipes_w_search_terms.csv",
                        usecols=["id", "ingredients", "ingredients_raw_str", "servings", "serving_size"])
    terms = terms[terms.id.isin(rec.recipe_id)].reset_index(drop=True)
    print(f"{len(terms):,} of {len(rec):,} recipes have raw ingredient lines")

    line_rows, rec_rows = [], []
    for r in terms.itertuples(index=False):
        names = [str(n).lower() for n in ast.literal_eval(r.ingredients)]
        try:
            lines = [l for l in json.loads(r.ingredients_raw_str) if str(l).strip()]
        except (ValueError, TypeError):
            lines = []
        counts = {"ok": 0, "no_amount": 0, "unresolved_major": 0, "unresolved_minor": 0}
        for k, line in enumerate(lines):
            parsed = parse_line(line)
            lc = clean_line(line).lower()
            name = match_line(lc, names)
            cands = canon(name) if name else ()
            ingredient = cands[0] if len(cands) == 1 else None
            if ingredient is None:
                status = "no_amount" if parsed.amount is None else "unmatched"
            elif parsed.amount is None:
                status = "no_amount"
            elif ingredient not in vocab or pd.isna(fid.get(ingredient)):
                status = "no_usda"
            else:
                g = portions.to_grams(int(fid[ingredient]), parsed, ITEM_DEFAULTS.get(ingredient))
                status = "ok" if g is not None else "unconverted"
            grams = g if status == "ok" else np.nan
            if status == "ok":
                counts["ok"] += 1
            elif status == "no_amount":
                counts["no_amount"] += 1
            elif ingredient is not None and group.get(ingredient) in MINOR_GROUPS:
                counts["unresolved_minor"] += 1
            else:
                counts["unresolved_major"] += 1
            line_rows.append((r.id, k, line, ingredient, parsed.amount, parsed.unit, grams, status))
        ss = pd.to_numeric(pd.Series([r.serving_size]).str.extract(r"\((-?\d+(?:\.\d+)?)\s*g\)")[0],
                           errors="coerce").iloc[0]
        rec_rows.append((r.id, len(lines), r.servings, ss, *counts.values()))
    print(f"Parsed {len(line_rows):,} lines ({time.time() - t0:.0f}s)")

    lines_df = pd.DataFrame(line_rows, columns=["recipe_id", "line_no", "raw_line", "ingredient", "amount",
                                                "unit", "grams", "status"])
    summ = pd.DataFrame(rec_rows, columns=["recipe_id", "n_lines", "servings", "serving_size_g", "n_ok",
                                           "n_no_amount", "n_unresolved_major", "n_unresolved_minor"])

    # ---- per-serving reconstruction from converted lines
    ok = lines_df[lines_df.status == "ok"].merge(summ[["recipe_id", "servings"]], on="recipe_id")
    ok = ok[(ok.servings >= 1) & (ok.servings <= MAX_SERVINGS)]
    ok["g_serving"] = ok.grams / ok.servings
    vals = nut.reindex(ok.ingredient).to_numpy() * ok.g_serving.to_numpy()[:, None] / 100.0
    per = pd.DataFrame(vals, columns=[f"recon_{c}" for c in NUTRIENTS])
    per["recipe_id"] = ok.recipe_id.to_numpy()
    per["recon_mass_g"] = ok.g_serving.to_numpy()
    # min_count=1 keeps NaN when a nutrient is unknown for every ingredient; a single unknown
    # value (e.g. sugar not reported) is flagged below rather than silently treated as zero
    recon = per.groupby("recipe_id").sum(min_count=1)
    missing = per.drop(columns=["recon_mass_g"]).set_index("recipe_id").isna().groupby(level=0).any()
    missing.columns = [c.replace("recon_", "missing_") for c in missing.columns]

    summ = summ.merge(recon, left_on="recipe_id", right_index=True, how="left") \
               .merge(missing, left_on="recipe_id", right_index=True, how="left") \
               .merge(rec, on="recipe_id", how="left")
    summ["kcal_ratio"] = summ.recon_energy_kcal / summ.calories
    summ["mass_ratio"] = summ.recon_mass_g / summ.serving_size_g
    summ["fully_resolved"] = (summ.n_unresolved_major == 0) & (summ.n_ok > 0) & \
        summ.servings.between(1, MAX_SERVINGS)
    summ["validated"] = summ.fully_resolved & ~summ.calories_outlier.fillna(True) & (summ.calories > 0) & \
        ((summ.kcal_ratio - 1).abs() <= CALORIE_TOLERANCE) & ~summ.missing_energy_kcal.fillna(True)

    lines_df.to_parquet(paths.PROCESSED / "recipe_ingredient_amounts.parquet", index=False)
    summ.to_parquet(paths.PROCESSED / "recipe_amounts_summary.parquet", index=False)
    write_report(lines_df, summ, len(rec))
    print(f"Done ({time.time() - t0:.0f}s): {summ.fully_resolved.sum():,} fully resolved, "
          f"{summ.validated.sum():,} validated")


def write_report(lines: pd.DataFrame, s: pd.DataFrame, n_recipes: int) -> None:
    st = lines.status.value_counts()
    fr = s[s.fully_resolved & (s.calories > 0) & ~s.calories_outlier.fillna(True)]
    kc = fr.kcal_ratio.replace([np.inf, -np.inf], np.nan).dropna()
    ms = fr.mass_ratio.replace([np.inf, -np.inf], np.nan).dropna()
    corr_kcal = np.corrcoef(np.log1p(fr.recon_energy_kcal.clip(lower=0)), np.log1p(fr.calories))[0, 1]
    v = s[s.validated]

    def q(x):
        return " / ".join(f"{x.quantile(p):.2f}" for p in (0.1, 0.25, 0.5, 0.75, 0.9))

    lines_md = [
        "# Deliverable 5 - recipe quantities and their validation",
        "",
        "Generated by `src/build_recipe_quantities.py`.",
        "",
        "## Source",
        f"- {len(s):,} of {n_recipes:,} recipes have raw ingredient lines (Food.com \"recipes with search terms\" "
        "dump, joined on recipe id).",
        f"- {len(lines):,} non-empty ingredient lines.",
        "",
        "| line status | lines | share |", "|---|---|---|",
        *[f"| {k} | {n:,} | {n / len(lines):.1%} |" for k, n in st.items()],
        "",
        "`ok` = converted to grams with the ingredient's USDA household measures; `no_amount` = no quantity "
        "given (e.g. \"salt to taste\", counted as 0 g - Food.com's own calculator cannot count it either); "
        "`no_usda` = ingredient without a reliable nutrient link; `unconverted` = unit not available for "
        "that food; `unmatched` = line could not be tied to an ingredient name.",
        "",
        "## Validation against Food.com",
        f"Recipes whose quantified main ingredients were all converted: **{s.fully_resolved.sum():,}** "
        "(unconvertible herb/spice lines are negligible by mass and allowed).",
        "",
        "Ratio of reconstructed to Food.com values (percentiles 10 / 25 / 50 / 75 / 90):",
        "",
        "| quantity | percentiles | within +-20% |", "|---|---|---|",
        f"| kcal per serving | {q(kc)} | {((kc - 1).abs() <= 0.2).mean():.1%} |",
        f"| grams per serving vs. serving size | {q(ms)} | {((ms - 1).abs() <= 0.2).mean():.1%} |",
        "",
        f"Correlation of log calories per serving (reconstructed vs. Food.com): **r = {corr_kcal:.3f}**.",
        "",
        f"**Analysis set**: fully resolved, not a calorie outlier, reconstructed calories within "
        f"+-{CALORIE_TOLERANCE:.0%} of Food.com's -> **{len(v):,} recipes**, median |kcal error| "
        f"{(v.kcal_ratio - 1).abs().median():.1%}.",
        "",
        "Analysis-set recipes per cuisine:",
        "",
        "| cuisine | recipes |", "|---|---|",
        *[f"| {k} | {n:,} |" for k, n in v.cuisine.value_counts().items()],
        "",
    ]
    (paths.ROOT / "reports" / "quantity_validation.md").write_text("\n".join(lines_md), encoding="utf-8")


if __name__ == "__main__":
    main()
