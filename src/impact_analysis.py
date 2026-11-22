"""Deliverable 5 (step 2) - recipe-level impact of healthier substitutions, compared across cuisines.

Sample: the validated recipes of src/build_recipe_quantities.py (every quantified main ingredient
converted to grams and reconstructed calories within 20% of Food.com's own value).

Swaps: every quantified ingredient gets the model's first *healthier* suggestion (the one the app
shows first; Deliverable 3 healthier mode), swapped at equal mass. Herbs, spices and salt are never
swapped: they are used by the gram, so replacing e.g. 3 g of salt with 3 g of marjoram is not a
meaningful substitution (cutting salt is a different intervention). Three scenarios:
  * best single swap       - the one model swap that lowers the recipe's health score the most (what a
                             cook would most likely do);
  * all swaps              - every ingredient that has a healthier substitute is swapped (upper bound);
  * reviewer-reported swap - real-world check: only swaps that >= 2 Food.com reviewers reported making
                             and that are healthier by the same criterion; the best one per recipe.

Outcomes per serving: calories, total fat, saturated fat, sugars, sodium (absolute and %), the
health score, and flavor retained:
  * pair flavor similarity   - Ochiai overlap of the compound sets of original and substitute;
  * recipe aroma retention   - share of the recipe's flavor compounds (union over its ingredients)
                               still present after the swaps.

Outputs
  data/processed/recipe_swap_impact.parquet   one row per recipe x scenario
  data/processed/recipe_swaps.parquet         one row per applied swap
  reports/impact_report.md, reports/figures/impact_*.png, outputs/impact_dashboard.html

Run:  python src/impact_analysis.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flavorswap import paths  # noqa: E402
from flavorswap.nutrition import health_scores  # noqa: E402
from flavorswap.recommender import Recommender  # noqa: E402

SEED = 42
SEASONING_GROUP = "Herbs & Spices"
REVIEW_MIN_USERS = 2
SCENARIOS = ["best single swap", "all swaps", "reviewer-reported swap"]
N_BOOT = 2000
MIN_CUISINE_RECIPES = 75
EXCLUDED_FROM_COMPARISON = {"Unspecified", "Mixed"}
# outcome -> (nutrient column, unit, minimum original amount for a meaningful % change)
OUTCOMES = {
    "Calories": ("energy_kcal", "kcal", 20.0),
    "Total fat": ("fat_g", "g", 1.0),
    "Saturated fat": ("sat_fat_g", "g", 0.5),
    "Sugars": ("sugar_g", "g", 1.0),
    "Sodium": ("sodium_mg", "mg", 20.0),
}
NUTRIENT_COLS = sorted({c for c, _, _ in OUTCOMES.values()})


def nutrient_table(ing: pd.DataFrame) -> pd.DataFrame:
    """Per-100 g nutrients with the same logical fills as the health score (a value bounded by a
    parent that is ~0 is 0)."""
    t = ing.set_index("ingredient")[NUTRIENT_COLS + ["carbs_g"]].astype(float).copy()
    t.loc[t.sugar_g.isna() & (t.carbs_g <= 0.5), "sugar_g"] = 0.0
    t.loc[t.sat_fat_g.isna() & (t.fat_g <= 0.5), "sat_fat_g"] = 0.0
    return t[NUTRIENT_COLS]


def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    rec = Recommender.load()
    ing = rec.ing
    nut = nutrient_table(ing)
    health = pd.Series(health_scores(ing), index=ing.ingredient)

    summ = pd.read_parquet(paths.PROCESSED / "recipe_amounts_summary.parquet")
    sample = summ[summ.validated][["recipe_id", "name", "cuisine", "servings"]].copy()
    lines = pd.read_parquet(paths.PROCESSED / "recipe_ingredient_amounts.parquet",
                            columns=["recipe_id", "ingredient", "grams", "status"])
    lines = lines[(lines.status == "ok") & lines.recipe_id.isin(sample.recipe_id)]
    amounts = (lines.groupby(["recipe_id", "ingredient"]).grams.sum().reset_index()
               .merge(sample[["recipe_id", "servings"]], on="recipe_id"))
    amounts["g"] = amounts.grams / amounts.servings          # grams per serving
    print(f"Sample: {len(sample):,} recipes, {len(amounts):,} quantified ingredient entries")

    # ---- first healthier suggestion for every ingredient in the sample
    best_swap = {}
    for name in sorted(amounts.ingredient.unique()):
        h = rec.healthier(name, 1)
        if not h.empty:
            best_swap[name] = h.ingredient.iloc[0]
    print(f"{len(best_swap):,} of {amounts.ingredient.nunique():,} ingredients have a healthier substitute "
          f"({time.time() - t0:.0f}s)")

    seasonings = set(ing.ingredient[ing.food_group == SEASONING_GROUP])
    amounts["substitute"] = amounts.ingredient.map(best_swap).where(~amounts.ingredient.isin(seasonings))

    def with_health_change(df: pd.DataFrame) -> pd.DataFrame:
        """Change in the recipe's health score per serving if the ingredient is swapped (equal mass)."""
        df = df.copy()
        df["d_health"] = df.g / 100 * (health.reindex(df.substitute).to_numpy() - health.reindex(df.ingredient).to_numpy())
        return df

    sw = with_health_change(amounts[amounts.substitute.notna()])
    single = sw.loc[sw.groupby("recipe_id").d_health.idxmin()]

    # real-world check: healthier swaps that Food.com reviewers actually reported
    pairs = pd.read_csv(paths.PROCESSED / "substitution_pairs.csv")
    pairs = pairs[pairs.n_users >= REVIEW_MIN_USERS]
    cand = []
    for src, grp in pairs.groupby("source"):
        if src not in rec.index or src in seasonings:
            continue
        ok = rec.healthier_mask(rec.index[src], rec.weights.min_reduction)
        cand += [(src, t) for t in grp.target if t in rec.index and ok[rec.index[t]]]
    cand = pd.DataFrame(cand, columns=["ingredient", "substitute"])
    rv = with_health_change(amounts.drop(columns="substitute").merge(cand, on="ingredient"))
    reviewer = rv.loc[rv.groupby("recipe_id").d_health.idxmin()]
    print(f"{len(cand):,} healthier reviewer-reported pairs; {reviewer.recipe_id.nunique():,} recipes can use one")
    scenarios = {"best single swap": single, "all swaps": sw, "reviewer-reported swap": reviewer}

    # ---- flavor: compound sets and pair similarities
    comp_sets = {n: frozenset(rec.flav_bin[i].indices) for n, i in rec.index.items() if rec.flav_has[i]}

    def pair_flavor(a: str, b: str) -> float:
        if a not in comp_sets or b not in comp_sets:
            return np.nan
        A, B = comp_sets[a], comp_sets[b]
        return len(A & B) / np.sqrt(len(A) * len(B))

    recipe_ings = amounts.groupby("recipe_id").ingredient.apply(list).to_dict()
    rows, swap_rows = [], []
    base = (amounts.assign(**{c: amounts.g / 100 * nut.reindex(amounts.ingredient)[c].to_numpy() for c in NUTRIENT_COLS})
            .groupby("recipe_id")[NUTRIENT_COLS].sum(min_count=1))
    base_complete = amounts.assign(**{c: nut.reindex(amounts.ingredient)[c].notna().to_numpy() for c in NUTRIENT_COLS}) \
        .groupby("recipe_id")[NUTRIENT_COLS].all()
    base_health = (amounts.g / 100 * health.reindex(amounts.ingredient).to_numpy()).groupby(amounts.recipe_id).sum(min_count=1)

    for scen, s in scenarios.items():
        s = s.copy()
        for c in NUTRIENT_COLS:
            s[f"d_{c}"] = s.g / 100 * (nut.reindex(s.substitute)[c].to_numpy() - nut.reindex(s.ingredient)[c].to_numpy())
        s["pair_flavor"] = [pair_flavor(a, b) for a, b in zip(s.ingredient, s.substitute)]
        s["pair_context"] = [float(rec.emb[rec.index[a]] @ rec.emb[rec.index[b]]) for a, b in zip(s.ingredient, s.substitute)]
        s["scenario"] = scen
        swap_rows.append(s)
        agg = s.groupby("recipe_id").agg(**{f"d_{c}": (f"d_{c}", lambda x: x.sum(min_count=1)) for c in NUTRIENT_COLS},
                                         d_health=("d_health", "sum"), n_swaps=("ingredient", "size"),
                                         swapped_g=("g", "sum"))
        swaps_by_recipe = s.groupby("recipe_id").apply(lambda d: dict(zip(d.ingredient, d.substitute)))
        for rid in sample.recipe_id:
            row = {"recipe_id": rid, "scenario": scen}
            mapping = swaps_by_recipe.get(rid, {})
            row["n_swaps"] = int(agg.n_swaps.get(rid, 0))
            row["d_health"] = float(agg.d_health.get(rid, 0.0))
            row["health_before"] = float(base_health.get(rid, np.nan))
            for c in NUTRIENT_COLS:
                row[f"{c}_before"] = float(base.loc[rid, c]) if rid in base.index else np.nan
                row[f"{c}_complete"] = bool(base_complete.loc[rid, c]) if rid in base_complete.index else False
                row[f"d_{c}"] = float(agg[f"d_{c}"].get(rid, 0.0)) if mapping else 0.0
            before = set().union(*[comp_sets[n] for n in recipe_ings.get(rid, []) if n in comp_sets]) \
                if any(n in comp_sets for n in recipe_ings.get(rid, [])) else set()
            after_ings = [mapping.get(n, n) for n in recipe_ings.get(rid, [])]
            after = set().union(*[comp_sets[n] for n in after_ings if n in comp_sets]) if before else set()
            row["aroma_compounds_before"] = len(before)
            row["aroma_retained"] = len(before & after) / len(before) if before else np.nan
            rows.append(row)
    impact = pd.DataFrame(rows).merge(sample[["recipe_id", "name", "cuisine"]], on="recipe_id")
    for label, (c, unit, floor) in OUTCOMES.items():
        ok = impact[f"{c}_complete"] & (impact[f"{c}_before"] >= floor)
        impact[f"pct_{c}"] = np.where(ok, 100 * impact[f"d_{c}"] / impact[f"{c}_before"], np.nan)
    swaps = pd.concat(swap_rows, ignore_index=True)
    impact.to_parquet(paths.PROCESSED / "recipe_swap_impact.parquet", index=False)
    swaps.to_parquet(paths.PROCESSED / "recipe_swaps.parquet", index=False)
    print(f"Impact computed ({time.time() - t0:.0f}s)")

    report(impact, swaps, rng)
    print(f"Done ({time.time() - t0:.0f}s)")


# ============================================================================================
# statistics
# ============================================================================================
def boot_mean_ci(x: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, np.nan
    idx = rng.integers(0, len(x), size=(N_BOOT, len(x)))
    means = x[idx].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def cuisine_table(impact: pd.DataFrame, scenario: str, rng) -> pd.DataFrame:
    d = impact[impact.scenario == scenario]
    counts = d.cuisine.value_counts()
    keep = [c for c, n in counts.items() if n >= MIN_CUISINE_RECIPES and c not in EXCLUDED_FROM_COMPARISON]
    groups = [("All recipes", d)] + [(c, d[d.cuisine == c]) for c in keep]
    rows = []
    for name, g in groups:
        row = {"cuisine": name, "recipes": len(g), "with a swap": (g.n_swaps > 0).mean()}
        for label, (c, unit, _) in OUTCOMES.items():
            x = g[f"pct_{c}"].to_numpy(dtype=float)
            lo, hi = boot_mean_ci(x, rng)
            row[f"{label} %"] = np.nanmean(x)
            row[f"{label} % CI"] = (lo, hi)
            row[f"{label} median %"] = np.nanmedian(x)
            row[f"{label} abs"] = np.nanmean(g[f"d_{c}"].to_numpy(dtype=float))
            row[f"{label} abs median"] = np.nanmedian(g[f"d_{c}"].to_numpy(dtype=float))
        a = g.aroma_retained.to_numpy(dtype=float)
        lo, hi = boot_mean_ci(a, rng)
        row["Aroma retained"] = np.nanmean(a)
        row["Aroma retained CI"] = (lo, hi)
        rows.append(row)
    return pd.DataFrame(rows)


def kruskal(impact: pd.DataFrame, scenario: str, cuisines: list[str]) -> pd.DataFrame:
    d = impact[(impact.scenario == scenario) & impact.cuisine.isin(cuisines)]
    rows = []
    for label, (c, _, _) in list(OUTCOMES.items()) + [("Aroma retained", ("aroma_retained", "", 0))]:
        col = f"pct_{c}" if label != "Aroma retained" else "aroma_retained"
        samples = [g[col].dropna().to_numpy() for _, g in d.groupby("cuisine")]
        samples = [x for x in samples if len(x) >= 2]
        h, p = stats.kruskal(*samples)
        n = sum(len(x) for x in samples)
        eps2 = (h - len(samples) + 1) / (n - len(samples))       # epsilon-squared effect size
        rows.append({"outcome": label, "H": h, "p": p, "epsilon^2": max(eps2, 0.0), "recipes": n})
    return pd.DataFrame(rows)


# ============================================================================================
# report + figures
# ============================================================================================
AUDIT_FILE = "impact_swap_audit.csv"


def swap_audit_lines(s1: pd.DataFrame, v: pd.DataFrame) -> list[str]:
    """Report section for the manual audit of the most frequent best single swaps
    (reports/impact_swap_audit.csv: ingredient, substitute, plausible yes/no, note)."""
    path = paths.ROOT / "reports" / AUDIT_FILE
    if not path.exists():
        return []
    audit = pd.read_csv(path)
    m = s1[["recipe_id", "ingredient", "substitute"]].merge(
        audit[["ingredient", "substitute", "plausible"]], on=["ingredient", "substitute"], how="left")
    judged = m[m.plausible.notna()]
    ok_ids = set(judged.recipe_id[judged.plausible.eq("yes")])
    sub = v[v.recipe_id.isin(ok_ids)]
    bad = (judged[judged.plausible.eq("no")].groupby(["ingredient", "substitute"]).size()
           .sort_values(ascending=False).head(8))
    lines = [
        "## Plausibility of the applied swaps (manual audit)",
        f"The {len(audit):,} most frequent best single swaps (`reports/{AUDIT_FILE}`) were judged by hand: would a "
        "cook accept this as a substitute in a typical recipe? They cover "
        f"{len(judged) / len(m):.1%} of recipes with a best single swap; **{judged.plausible.eq('yes').mean():.1%}** "
        "of these recipes received a plausible swap. Implausible ones: "
        + "; ".join(f"{a} -> {b} ({n})" for (a, b), n in bad.items()) + ".",
        "",
        f"Robustness check - best single swap restricted to the {len(sub):,} recipes whose swap was judged plausible:",
        "",
        "| outcome | mean change per serving | median |",
        "|---|---|---|",
        *[f"| {label} | {signed(np.nanmean(sub[f'pct_{c}']))}% | {signed(np.nanmedian(sub[f'pct_{c}']))}% |"
          for label, (c, _, _) in OUTCOMES.items()],
        f"| Aroma retained | {100 * np.nanmean(sub.aroma_retained):.1f}% | {100 * np.nanmedian(sub.aroma_retained):.1f}% |",
        "",
    ]
    return lines


def signed(x: float, d: int = 1) -> str:
    """Signed number; a value that rounds to zero is printed as 0 (never -0.0)."""
    r = round(float(x), d)
    return f"{0.0:.{d}f}" if r == 0 else f"{r:+.{d}f}"


def fmt_ci(ci) -> str:
    lo, hi = ci
    return f"[{lo:+.1f}, {hi:+.1f}]"


def report(impact: pd.DataFrame, swaps: pd.DataFrame, rng) -> None:
    (paths.ROOT / "reports" / "figures").mkdir(parents=True, exist_ok=True)
    tables = {s: cuisine_table(impact, s, rng) for s in SCENARIOS}
    single = tables["best single swap"]
    compared = [c for c in single.cuisine if c != "All recipes"]
    kw = {s: kruskal(impact, s, compared) for s in tables}

    from impact_dashboard import build as write_dashboard  # noqa: E402
    from impact_figures import write_static_figures  # noqa: E402
    write_static_figures(tables, paths.ROOT / "reports" / "figures")
    write_dashboard(tables, impact, swaps, paths.OUTPUTS / "impact_dashboard.html")

    def outcome_table(t: pd.DataFrame) -> str:
        cols = ["cuisine", "recipes", "with a swap"]
        out = t[cols].copy()
        out["with a swap"] = (100 * out["with a swap"]).map(lambda x: f"{x:.1f}%")
        for label in OUTCOMES:
            out[f"{label} (mean %, 95% CI)"] = [f"{m:+.1f} {fmt_ci(ci)}" for m, ci in zip(t[f"{label} %"], t[f"{label} % CI"])]
        out["Aroma retained"] = [f"{100 * m:.1f}% {fmt_ci(tuple(100 * v for v in ci))}" for m, ci in
                                 zip(t["Aroma retained"], t["Aroma retained CI"])]
        return out.to_markdown(index=False)

    def abs_table(t: pd.DataFrame) -> str:
        out = t[["cuisine"]].copy()
        for label, (_, unit, _) in OUTCOMES.items():
            d = 0 if unit == "mg" else 1
            out[f"{label} ({unit}/serving)"] = [f"{signed(m, d)} ({signed(md, d)})" for m, md in
                                                zip(t[f"{label} abs"], t[f"{label} abs median"])]
        return out.to_markdown(index=False)

    s1 = swaps[swaps.scenario == "best single swap"]
    top_swaps = (s1.groupby(["ingredient", "substitute"]).size().sort_values(ascending=False).head(15)
                 .reset_index(name="recipes"))
    per_cuisine = []
    for c in compared:
        ids = set(impact[(impact.scenario == "best single swap") & (impact.cuisine == c)].recipe_id)
        top = s1[s1.recipe_id.isin(ids)].groupby(["ingredient", "substitute"]).size().sort_values(ascending=False).head(3)
        per_cuisine.append({"cuisine": c, "most common best swaps": "; ".join(
            f"{a} -> {b} ({n})" for (a, b), n in top.items())})
    pair = swaps.groupby("scenario")[["pair_flavor", "pair_context"]].mean()
    all_row = single[single.cuisine == "All recipes"].iloc[0]
    v = impact[impact.scenario == "best single swap"]
    audit_lines = swap_audit_lines(s1, v)
    summ = pd.read_parquet(paths.PROCESSED / "recipe_amounts_summary.parquet", columns=["validated", "kcal_ratio"])
    kcal_err = (summ.kcal_ratio[summ.validated] - 1).abs().median()

    lines = [
        "# FlavorSwap - Deliverable 5: recipe-level impact of healthier substitutions",
        "",
        "Generated by `src/impact_analysis.py` (quantities: `src/build_recipe_quantities.py`, "
        "validation: `reports/quantity_validation.md`).",
        "",
        "## Method",
        f"- **Sample**: {impact.recipe_id.nunique():,} Food.com recipes whose ingredient amounts were converted to "
        "grams and whose reconstructed calories per serving are within 20% of Food.com's own value "
        f"(median error {kcal_err:.1%}).",
        "- **Swaps**: for every quantified ingredient, the first suggestion of FlavorSwap's healthier mode "
        "(Deliverable 3), swapped at **equal mass**. *Best single swap* = the one swap that lowers the recipe's "
        "health score the most; *all swaps* = every ingredient that has a healthier substitute (upper bound); "
        f"*reviewer-reported swap* = real-world check using only healthier swaps that >= {REVIEW_MIN_USERS} "
        "Food.com reviewers reported making (best one per recipe).",
        "- **Herbs, spices and salt are never swapped** (they still count in the recipe's nutrition): they are "
        "used by the gram, so an equal-mass swap such as 3 g salt -> 3 g marjoram is not a meaningful "
        "substitution. Reducing salt is a separate intervention and is not modelled here.",
        "- **Outcomes** per serving from USDA values: calories, total fat, saturated fat, total sugars (the "
        "health score that picks the swaps counts only free sugars), sodium. "
        "% changes are computed only for recipes whose original amount is meaningful (>= 20 kcal, 1 g fat, "
        "0.5 g saturated fat, 1 g sugars, 20 mg sodium) and whose nutrient data is complete.",
        "- **Flavor retained**: *aroma retention* = share of the recipe's flavor compounds (union over its "
        "ingredients, Ahn et al.) still present after the swaps; *pair flavor similarity* = compound overlap "
        "of each original/substitute pair.",
        f"- **Cuisines**: Food.com cuisine tags; cuisines with >= {MIN_CUISINE_RECIPES} sample recipes are "
        "compared (\"Unspecified\" and \"Mixed\" are only in *All recipes*). 95% CIs: bootstrap over "
        f"recipes ({N_BOOT:,} resamples). Differences between cuisines: Kruskal-Wallis test.",
        "",
        "## Headline (best single swap, all sample recipes)",
        f"- {all_row['with a swap']:.1%} of recipes have at least one (non-seasoning) ingredient with a "
        "healthier substitute.",
        *[f"- {label}: mean change **{all_row[f'{label} %']:+.1f}%** per serving (95% CI "
          f"{fmt_ci(all_row[f'{label} % CI'])}), median {signed(all_row[f'{label} median %'])}%."
          for label in OUTCOMES],
        f"- Aroma retention: **{100 * all_row['Aroma retained']:.1f}%** of the recipe's flavor compounds are kept "
        f"on average; mean pair flavor similarity of the swaps {pair.loc['best single swap', 'pair_flavor']:.2f}, "
        f"recipe-context similarity {pair.loc['best single swap', 'pair_context']:.2f}.",
        f"- Mean health-score change per serving: {v.d_health.mean():+.3f} %DV units.",
        "",
        "![per-cuisine change](figures/impact_by_cuisine_single.png)",
        "",
        *audit_lines,
        "## Best single swap - by cuisine",
        "Mean % change per serving with 95% CI:",
        "",
        outcome_table(single),
        "",
        "Absolute change per serving, mean (median):",
        "",
        abs_table(single),
        "",
        "Differences between cuisines (Kruskal-Wallis across the compared cuisines):",
        "",
        kw["best single swap"].assign(p=lambda d: d.p.map(lambda x: f"{x:.1e}")).round(4).to_markdown(index=False),
        "",
        "Most common best swaps per cuisine:",
        "",
        pd.DataFrame(per_cuisine).to_markdown(index=False),
        "",
        "## Real-world check: best reviewer-reported swap - by cuisine",
        "Only healthier swaps that Food.com reviewers actually made. If the model's swaps were unrealistic, "
        "their impact would differ sharply from this scenario.",
        "",
        outcome_table(tables["reviewer-reported swap"]),
        "",
        "![per-cuisine change, reviewer-reported swaps](figures/impact_by_cuisine_reviewer.png)",
        "",
        "Differences between cuisines:",
        "",
        kw["reviewer-reported swap"].assign(p=lambda d: d.p.map(lambda x: f"{x:.1e}")).round(4).to_markdown(index=False),
        "",
        "## All swaps (upper bound) - by cuisine",
        "",
        outcome_table(tables["all swaps"]),
        "",
        "![per-cuisine change, all swaps](figures/impact_by_cuisine_all.png)",
        "",
        "Differences between cuisines:",
        "",
        kw["all swaps"].assign(p=lambda d: d.p.map(lambda x: f"{x:.1e}")).round(4).to_markdown(index=False),
        "",
        "## Most common best single swaps (all cuisines)",
        "",
        top_swaps.to_markdown(index=False),
        "",
        "## Limitations",
        "- Equal-mass swaps per 100 g: real cooks may adjust amounts (e.g. less oil than butter); texture and "
        "baking chemistry are not modelled.",
        "- Ingredient amounts come from the recipe text; cooking losses (water, fat drained) are ignored, as "
        "in Food.com's own nutrition calculation.",
        "- The sample is limited to recipes whose amounts could be reconstructed and validated; recipes with "
        "unquantified or unlinked main ingredients are excluded, which may under-represent some cuisines.",
        "- Aroma retention uses compound presence (Ahn et al.), not concentrations; ingredients without a flavor "
        "profile do not contribute.",
        "- Cuisine labels are Food.com user tags.",
        "- % changes are averaged over recipes; a recipe with a small original amount (e.g. 1-2 g sugars per "
        "serving) can show a large % increase, which widens the CIs of small cuisines (e.g. Japanese sugars). "
        "Absolute means can likewise be driven by single extreme recipes (e.g. a homemade soy sauce recipe in "
        "Japanese sodium); medians are reported next to them.",
        "",
    ]
    (paths.ROOT / "reports" / "impact_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
