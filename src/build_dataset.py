"""Deliverable 1 - curated dataset linking cleaned recipe ingredients to flavor compounds and
nutrient values.

Inputs  (data/raw/, see download_data.py):
    Food.com RAW_recipes.csv, Ahn et al. 2011 flavor network, USDA FDC SR Legacy.
Curation tables (data/curation/):
    ahn_links.csv, usda_links.csv  - hand-checked links that override automatic matching.
Outputs (data/processed/):
    recipes.parquet            one row per recipe: canonical ingredients, per-serving nutrition, cuisine
    ingredients.csv            one row per vocabulary ingredient: frequency, flavor link, nutrient link
                               and nutrients per 100 g
    ingredient_compounds.csv   long table: vocabulary ingredient -> flavor compound
    compounds.csv              flavor compounds (id, name, CAS number)
    ingredient_aliases.csv     raw Food.com ingredient string -> canonical ingredient(s)
    reports/dataset_report.md  coverage and quality statistics

Run:  python src/build_dataset.py
"""

from __future__ import annotations

import ast
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from flavorswap import paths  # noqa: E402
from flavorswap.matching import (  # noqa: E402
    NUTRIENTS, build_ahn_index, build_usda_index, invert_usda_index, match_ahn, match_usda,
)
from flavorswap.food_groups import food_group  # noqa: E402
from flavorswap.nutrition import category_trans_ratios, estimate_trans_fat  # noqa: E402
from flavorswap.normalize import canonicalize, canonicalize_recipe  # noqa: E402

VOCAB_MIN_RECIPES = 20  # an ingredient must appear in at least this many recipes

# Food.com 'nutrition' column: calories (#) then % daily value (PDV) per serving.
NUTRITION_COLS = ["calories", "total_fat_pdv", "sugar_pdv", "sodium_pdv", "protein_pdv",
                  "sat_fat_pdv", "carbs_pdv"]

# Food.com cuisine tags -> analysis label. Food.com tags are hierarchical (e.g. 'mexican' sits under
# 'north-american', 'chinese' under 'asian'), so specific cuisines take precedence over regions.
CUISINE_SPECIFIC = {
    "italian": "Italian", "mexican": "Mexican", "tex-mex": "Mexican", "chinese": "Chinese",
    "indian": "Indian", "french": "French", "greek": "Greek", "thai": "Thai",
    "japanese": "Japanese", "korean": "Korean", "vietnamese": "Vietnamese", "german": "German",
    "spanish": "Spanish", "middle-eastern": "Middle Eastern", "lebanese": "Middle Eastern",
    "turkish": "Middle Eastern", "iranian-persian": "Middle Eastern", "moroccan": "North African",
    "egyptian": "North African", "caribbean": "Caribbean", "cuban": "Caribbean",
    "cajun": "Cajun/Creole", "creole": "Cajun/Creole", "southern-united-states": "Southern US",
    "irish": "British/Irish", "english": "British/Irish", "scottish": "British/Irish",
    "welsh": "British/Irish",
}
CUISINE_REGIONAL = [  # checked in order when no specific cuisine tag is present
    ("asian", "Asian (other)"), ("european", "European (other)"), ("african", "African (other)"),
    ("south-american", "Latin American"), ("central-american", "Latin American"),
    ("north-american", "North American"), ("american", "North American"),
    ("canadian", "North American"),
]


def assign_cuisine(tags: list[str]) -> tuple[str, list[str]]:
    specific = sorted({CUISINE_SPECIFIC[t] for t in tags if t in CUISINE_SPECIFIC})
    if len(specific) == 1:
        return specific[0], specific
    if len(specific) > 1:
        return "Mixed", specific
    tagset = set(tags)
    for tag, label in CUISINE_REGIONAL:
        if tag in tagset:
            return label, []
    return "Unspecified", []


def load_recipes() -> pd.DataFrame:
    df = pd.read_csv(paths.RAW_RECIPES)
    df["ingredients_raw"] = df["ingredients"].map(ast.literal_eval)
    df["tags"] = df["tags"].map(ast.literal_eval)
    nut = np.array(df["nutrition"].map(ast.literal_eval).tolist(), dtype=float)
    for i, c in enumerate(NUTRITION_COLS):
        df[c] = nut[:, i]
    df["ingredients"] = df["ingredients_raw"].map(canonicalize_recipe)
    cz = df["tags"].map(assign_cuisine)
    df["cuisine"] = cz.map(lambda x: x[0])
    df["cuisine_tags"] = cz.map(lambda x: x[1])
    df = df.rename(columns={"id": "recipe_id"})
    return df


def load_ahn() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    ii = pd.read_csv(paths.AHN_INGR, sep="\t")
    ii.columns = ["id", "name", "category"]
    cc = pd.read_csv(paths.AHN_COMP, sep="\t")
    cc.columns = ["compound_id", "compound_name", "cas_number"]
    ic = pd.read_csv(paths.AHN_INGR_COMP, sep="\t")
    ic.columns = ["ingredient_id", "compound_id"]
    assert ic.ingredient_id.isin(ii.id).all() and ic.compound_id.isin(cc.compound_id).all()
    return ii, cc, ic.drop_duplicates()


def load_usda() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    food = pd.read_csv(paths.USDA_DIR / "food.csv")
    cats = pd.read_csv(paths.USDA_DIR / "food_category.csv")
    fn = pd.read_csv(paths.USDA_DIR / "food_nutrient.csv", usecols=["fdc_id", "nutrient_id", "amount"])
    nut = (fn[fn.nutrient_id.isin(NUTRIENTS)]
           .pivot_table(index="fdc_id", columns="nutrient_id", values="amount", aggfunc="first")
           .rename(columns=NUTRIENTS))
    nut = nut[[c for c in NUTRIENTS.values()]]
    return food, cats, nut


def read_curation(path: Path, value_col: str) -> tuple[dict, dict]:
    cur = pd.read_csv(path, dtype=str, keep_default_na=False)
    if cur.ingredient.duplicated().any():
        raise ValueError(f"duplicate ingredients in {path.name}: "
                         f"{cur.ingredient[cur.ingredient.duplicated()].tolist()}")
    links = {r.ingredient: (getattr(r, value_col) or None) for r in cur.itertuples()}
    notes = {r.ingredient: r.note for r in cur.itertuples()}
    return links, notes


def resolve_usda_descriptions(links: dict[str, str | None], food: pd.DataFrame) -> dict[str, int | None]:
    """Curated USDA links are written as descriptions; resolve each to exactly one fdc_id."""
    out = {}
    for ing, desc in links.items():
        if desc is None:
            out[ing] = None
            continue
        hit = food[food.description == desc]
        if len(hit) != 1:
            hit = food[food.description.str.startswith(desc, na=False)]
        if len(hit) != 1:
            raise ValueError(f"usda_links.csv: {ing!r} -> {desc!r} matches {len(hit)} foods")
        out[ing] = int(hit.fdc_id.iloc[0])
    return out


def main() -> None:
    paths.PROCESSED.mkdir(parents=True, exist_ok=True)
    (paths.ROOT / "reports").mkdir(exist_ok=True)

    print("Loading and normalising recipes ...")
    rec = load_recipes()
    n_raw = len(rec)

    # ---------------- vocabulary ----------------
    doc_freq = Counter(i for ings in rec.ingredients for i in ings)
    vocab = sorted((k for k, n in doc_freq.items() if n >= VOCAB_MIN_RECIPES),
                   key=lambda k: (-doc_freq[k], k))
    vocab_set = set(vocab)
    print(f"  {n_raw} recipes, {len(doc_freq)} canonical ingredients, {len(vocab)} in vocabulary")

    # ---------------- flavor links ----------------
    print("Linking to Ahn et al. flavor network ...")
    ii, cc, ic = load_ahn()
    aidx = build_ahn_index(ii, ic)
    ahn_links, ahn_notes = read_curation(paths.CURATION / "ahn_links.csv", "ahn_ingredient")
    unknown = [k for k in ahn_links if k not in vocab_set]
    if unknown:
        raise ValueError(f"ahn_links.csv has ingredients outside the vocabulary: {unknown}")

    # ---------------- nutrient links ----------------
    print("Linking to USDA SR Legacy ...")
    food, cats, nut = load_usda()
    usda_desc_links, usda_notes = read_curation(paths.CURATION / "usda_links.csv", "usda_description")
    unknown = [k for k in usda_desc_links if k not in vocab_set]
    if unknown:
        raise ValueError(f"usda_links.csv has ingredients outside the vocabulary: {unknown}")
    usda_links = resolve_usda_descriptions(usda_desc_links, food)
    uidx = build_usda_index(food, cats)
    inv = invert_usda_index(uidx)
    all_foods = {f.fdc_id: f for f in build_usda_index(food, cats.assign(description=""))}
    cat_name = dict(zip(cats.id, cats.description))
    food_cat = dict(zip(food.fdc_id, food.food_category_id.map(cat_name)))
    food_desc = dict(zip(food.fdc_id, food.description))

    rows = []
    for ing in vocab:
        a, a_method = match_ahn(ing, aidx, ahn_links)
        u, u_method, u_score = match_usda(ing, inv, all_foods, usda_links)
        # Partial automatic matches proved unreliable in the audit; they are kept only as a
        # candidate for reference and never used as the nutrient link.
        candidate = None
        if u_method == "auto_low":
            candidate, u, u_method = u, None, "rejected_low_confidence"
        rows.append({
            "ingredient": ing,
            "n_recipes": doc_freq[ing],
            "ahn_ingredient": a,
            "ahn_category": aidx.categories.get(a) if a else None,
            "ahn_n_compounds": aidx.n_compounds.get(a, 0) if a else 0,
            "ahn_link_method": a_method,
            "ahn_note": ahn_notes.get(ing, ""),
            "usda_fdc_id": u,
            "usda_description": food_desc.get(u) if u else None,
            "usda_category": food_cat.get(u) if u else None,
            "usda_link_method": u_method,
            "usda_match_score": round(u_score, 3) if u_score == u_score else None,
            "usda_note": usda_notes.get(ing, ""),
            "usda_rejected_candidate": food_desc.get(candidate) if candidate else None,
        })
    ing_df = pd.DataFrame(rows)
    ing_df["usda_fdc_id"] = ing_df["usda_fdc_id"].astype("Int64")
    ing_df = ing_df.merge(nut, left_on="usda_fdc_id", right_index=True, how="left")
    # Estimated trans fat for health scoring (the reported column trans_fat_g is left untouched).
    fat_trans = nut[["fat_g", "trans_fat_g"]].join(
        food.set_index("fdc_id")[["food_category_id", "description"]])
    fat_trans["category"] = fat_trans.food_category_id.map(cat_name)
    ratios, overall_ratio = category_trans_ratios(fat_trans)
    ing_df["trans_fat_g_est"], ing_df["trans_fat_est_method"] = estimate_trans_fat(
        ing_df, ratios, overall_ratio)
    ing_df.insert(1, "rank", np.arange(1, len(ing_df) + 1))
    def _s(x):
        return x if isinstance(x, str) else None

    fg = [food_group(r.ingredient, _s(r.usda_category), _s(r.usda_description), _s(r.ahn_category))
          for r in ing_df.itertuples()]
    ing_df.insert(3, "food_group", [g for g, _ in fg])
    ing_df.insert(4, "food_group_source", [src for _, src in fg])

    # ---------------- ingredient -> compound table ----------------
    name_to_id = dict(zip(ii.name, ii.id))
    linked = ing_df.loc[ing_df.ahn_ingredient.notna(), ["ingredient", "ahn_ingredient"]].copy()
    linked["ingredient_id"] = linked.ahn_ingredient.map(name_to_id)
    ing_comp = (linked.merge(ic, on="ingredient_id")
                .merge(cc, on="compound_id")
                [["ingredient", "ahn_ingredient", "compound_id", "compound_name", "cas_number"]]
                .sort_values(["ingredient", "compound_id"]).reset_index(drop=True))

    # ---------------- recipes ----------------
    rec["n_ingredients_canonical"] = rec.ingredients.map(len)
    rec["ingredients_in_vocab"] = rec.ingredients.map(lambda l: [i for i in l if i in vocab_set])
    has_usda = set(ing_df.loc[ing_df.usda_fdc_id.notna(), "ingredient"])
    has_ahn = set(ing_df.loc[ing_df.ahn_ingredient.notna(), "ingredient"])
    rec["frac_ingredients_usda_linked"] = rec.ingredients.map(
        lambda l: sum(i in has_usda for i in l) / len(l) if l else np.nan)
    rec["frac_ingredients_flavor_linked"] = rec.ingredients.map(
        lambda l: sum(i in has_ahn for i in l) / len(l) if l else np.nan)
    # Food.com contains a few implausible per-serving calorie values (data-entry errors).
    rec["calories_outlier"] = rec.calories > rec.calories.quantile(0.999)

    out_cols = ["recipe_id", "name", "minutes", "n_steps", "submitted", "cuisine", "cuisine_tags",
                "ingredients_raw", "ingredients", "ingredients_in_vocab", "n_ingredients",
                "n_ingredients_canonical", *NUTRITION_COLS, "calories_outlier",
                "frac_ingredients_usda_linked", "frac_ingredients_flavor_linked", "tags"]
    rec_out = rec[out_cols].rename(columns={"n_ingredients": "n_ingredients_raw"})

    # ---------------- aliases ----------------
    raw_counts = Counter(i for l in rec.ingredients_raw for i in l)
    alias = pd.DataFrame(
        [(r, n, json.dumps(canonicalize(r))) for r, n in raw_counts.most_common()],
        columns=["raw_ingredient", "n_occurrences", "canonical_ingredients"])

    # ---------------- write ----------------
    rec_out.to_parquet(paths.PROCESSED / "recipes.parquet", index=False)
    ing_df.to_csv(paths.PROCESSED / "ingredients.csv", index=False)
    ing_comp.to_csv(paths.PROCESSED / "ingredient_compounds.csv", index=False)
    cc.to_csv(paths.PROCESSED / "compounds.csv", index=False)
    alias.to_csv(paths.PROCESSED / "ingredient_aliases.csv", index=False)

    write_report(rec_out, ing_df, ing_comp, doc_freq, vocab, n_raw)
    print("Done.")


def write_report(rec, ing, ing_comp, doc_freq, vocab, n_raw) -> None:
    occ_total = sum(doc_freq.values())
    occ_vocab = sum(doc_freq[v] for v in vocab)

    def occ_share(mask):
        return ing.loc[mask, "n_recipes"].sum() / occ_vocab

    def table(series):
        return "\n".join(f"| {k} | {v} |" for k, v in series.items())

    flavor = ing.ahn_ingredient.notna()
    usda = ing.usda_fdc_id.notna()
    a_meth = ing.ahn_link_method.value_counts()
    u_meth = ing.usda_link_method.value_counts()
    top400 = ing["rank"] <= 400
    top800 = ing["rank"] <= 800
    nut_cols = list(NUTRIENTS.values())

    def audit_line(fname: str, label: str) -> str:
        p = paths.ROOT / "reports" / fname
        if not p.exists():
            return f"- {label}: not audited"
        a = pd.read_csv(p)
        ok = (a.verdict == "correct").sum()
        return (f"- Manual audit of a random sample of {label} links outside the curated head: "
                f"**{ok}/{len(a)} correct ({ok / len(a):.0%})** - see `reports/{fname}`. "
                "Every link judged wrong was then fixed by hand.")
    missing_nut = ing.loc[usda, nut_cols].isna().mean().round(4)

    lines = [
        "# FlavorSwap - Deliverable 1 dataset report",
        "",
        "Generated by `src/build_dataset.py`.",
        "",
        "## Recipes",
        f"- Recipes: **{n_raw:,}** (Food.com RAW_recipes.csv)",
        f"- Canonical ingredients (all): **{len(doc_freq):,}**",
        f"- Vocabulary (ingredients in >= {VOCAB_MIN_RECIPES} recipes): **{len(vocab):,}**, covering "
        f"**{occ_vocab / occ_total:.2%}** of all ingredient occurrences",
        f"- Recipes with every canonical ingredient in the vocabulary: "
        f"**{(rec.ingredients.map(len) == rec.ingredients_in_vocab.map(len)).mean():.2%}**",
        f"- Median canonical ingredients per recipe: {rec.n_ingredients_canonical.median():.0f}",
        f"- Recipes flagged as calorie outliers (> 99.9th percentile, "
        f"{rec.calories.quantile(0.999):.0f} kcal/serving): {int(rec.calories_outlier.sum())}",
        "",
        "### Cuisine labels",
        "| cuisine | recipes |", "|---|---|",
        table(rec.cuisine.value_counts()),
        "",
        "### Ingredient food groups",
        "Source priority: USDA category > Ahn category > name keywords (column `food_group_source`).",
        "",
        "| food group | ingredients |", "|---|---|", table(ing.food_group.value_counts()),
        "",
        "| source | ingredients |", "|---|---|", table(ing.food_group_source.value_counts()),
        "",
        "## Flavor-compound links (Ahn et al. 2011)",
        f"- Vocabulary ingredients linked: **{flavor.sum():,} / {len(ing):,}** "
        f"({flavor.mean():.1%}); occurrence-weighted: **{occ_share(flavor):.1%}**",
        f"- Top-400 ingredients linked (occurrence-weighted): "
        f"**{ing.loc[top400 & flavor, 'n_recipes'].sum() / ing.loc[top400, 'n_recipes'].sum():.1%}**",
        f"- Ingredient-compound pairs: {len(ing_comp):,}; distinct compounds reached: "
        f"{ing_comp.compound_id.nunique():,}",
        f"- Median compounds per linked ingredient: {ing.loc[flavor, 'ahn_n_compounds'].median():.0f}",
        "",
        "| link method | ingredients |", "|---|---|", table(a_meth),
        "",
        "`manual` / `manual_none` = hand-checked (data/curation/ahn_links.csv); `exact` = same name; "
        "`head_noun` = name after dropping safe modifiers (e.g. *red onion* -> *onion*).",
        "",
        "## Nutrient links (USDA FoodData Central, SR Legacy)",
        f"- Vocabulary ingredients linked: **{usda.sum():,} / {len(ing):,}** ({usda.mean():.1%}); "
        f"occurrence-weighted: **{occ_share(usda):.1%}**",
        f"- Hand-checked (`manual`/`manual_none`) among the 800 most frequent ingredients: "
        f"**{ing.loc[top800, 'usda_link_method'].str.startswith('manual').sum()} / {int(top800.sum())}**; "
        f"these 800 ingredients account for **{ing.loc[top800, 'n_recipes'].sum() / occ_vocab:.1%}** "
        "of vocabulary occurrences",
        "",
        "| link method | ingredients |", "|---|---|", table(u_meth),
        "",
        "`manual` / `manual_none` = hand-checked (data/curation/usda_links.csv); `auto_high` = automatic match "
        "with every ingredient word present; `rejected_low_confidence` = partial automatic match, "
        "**not used** (shown in `usda_rejected_candidate` for reference only).",
        "",
        audit_line("usda_auto_link_audit.csv", "USDA `auto_high`"),
        "- In addition, every automatic USDA link for a single-word ingredient name (the riskiest "
        "case: *cookie*, *fish*, *seed*, ...) was reviewed and corrected by hand "
        "(note `fixed after single-word review` in usda_links.csv).",
        audit_line("ahn_auto_link_audit.csv", "Ahn `exact`/`head_noun`"),
        "",
        "Share of linked ingredients missing each nutrient (missing values are left empty, never "
        "imputed):",
        "", "| nutrient | missing |", "|---|---|", table(missing_nut),
        "",
        "Trans fat is often unreported in SR Legacy. For health scoring only, a separate column "
        "`trans_fat_g_est` is derived (see `src/flavorswap/nutrition.py`): reported value; else "
        "*unknown* for margarine/shortening/spread/topping-type products and baked goods, snacks and "
        "sweets (possible industrial trans fat); else total fat x the median trans/fat ratio of SR "
        "Legacy foods in the same category.",
        "", "| trans_fat_est_method | ingredients |", "|---|---|",
        table(ing.trans_fat_est_method.value_counts()),
        "",
    ]
    (paths.ROOT / "reports" / "dataset_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
