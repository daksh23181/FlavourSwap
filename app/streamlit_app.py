"""Deliverable 4 - FlavorSwap web app.

Enter an ingredient and get ranked substitutes - either *similar* swaps or *healthier* swaps -
each with an explanation: how alike the recipe contexts are, which aroma compounds are shared, how
the nutrients change per 100 g, and whether Food.com reviewers have made the swap themselves.

Run from the project root:   streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from flavorswap import paths  # noqa: E402
from flavorswap.recommender import (  # noqa: E402
    PCT_DISPLAY_MIN, Recommender, UnknownIngredient,
)

st.set_page_config(page_title="FlavorSwap", page_icon=":material/swap_horiz:", layout="wide",
                   initial_sidebar_state="collapsed")

QUICK_PICKS = ["butter", "sour cream", "sugar", "heavy cream", "ground beef", "mayonnaise",
               "all purpose flour", "white rice"]
# Option values double as URL values (?mode=Healthier), so they are short, readable labels.
MODES = {"Similar": "Similar swaps", "Healthier": "Healthier swaps"}
# (column, label, unit, decimals, lower_is_better, "same" threshold)
NUTRIENTS = [
    ("energy_kcal", "Energy", "kcal", 0, True, 1.0),
    ("sat_fat_g", "Sat. fat", "g", 1, True, 0.1),
    ("sugar_g", "Sugars", "g", 1, True, 0.1),
    ("sodium_mg", "Sodium", "mg", 0, True, 1.0),
    ("trans_fat_g_est", "Trans fat", "g", 1, True, 0.1),
    ("fiber_g", "Fiber", "g", 1, False, 0.1),
]
MEANINGFUL_FLAVOR_OVERLAP = 0.2   # below this, shared compounds are not cited as a reason
MIN_CONTEXT_REASON = 0.15         # below this cosine, recipe context is not cited as a reason
EXAMPLE_NAME_MAX = 32             # compound names longer than this are not used as examples
HEALTH_HELP = ("Per 100 g, lower is healthier. Sum of % Daily Values for energy (2000 kcal), saturated "
               "fat (20 g), free sugars (50 g), sodium (2300 mg) and trans fat (2.2 g, WHO limit), minus "
               "fiber (28 g). Like Nutri-Score, it does not penalise total fat and credits fiber; like the "
               "WHO sugar guideline, it does not count the natural sugar in whole fruit, vegetables or "
               "plain milk.")

# Banner styling. Colours are translucent tints of the brand green so they work on the light and the
# dark theme alike; text inherits the theme's text colour.
STYLE = """
<style>
.fs-hero { padding: 30px 34px 28px; border-radius: 22px; margin: 0 0 6px;
  background: linear-gradient(135deg, rgba(15,123,85,0.16) 0%, rgba(15,123,85,0.05) 55%, rgba(15,123,85,0.02) 100%);
  border: 1px solid rgba(15,123,85,0.20); }
.fs-hero.compact { padding: 16px 22px; border-radius: 16px; }
.fs-brand { display: flex; align-items: center; gap: 10px; font-weight: 700; font-size: 1.05rem; }
.fs-logo { width: 34px; height: 34px; border-radius: 10px; background: #0f7b55; color: #ffffff;
  display: inline-flex; align-items: center; justify-content: center; font-size: 1.15rem; font-weight: 700; }
.fs-tag { margin-left: auto; font-size: 0.8rem; font-weight: 600; padding: 4px 10px; border-radius: 999px;
  background: rgba(15,123,85,0.12); white-space: nowrap; }
.fs-title { font-size: clamp(1.9rem, 3.4vw, 2.8rem); line-height: 1.12; font-weight: 800;
  letter-spacing: -0.02em; margin: 16px 0 8px; }
.fs-sub { font-size: 1.1rem; opacity: 0.8; max-width: 780px; margin: 0; }
.fs-hero.compact .fs-sub { font-size: 0.95rem; margin: 0 0 0 44px; }
@media (max-width: 640px) { .fs-tag { display: none; } .fs-hero { padding: 22px 20px; }
  .fs-hero.compact .fs-sub { margin-left: 0; margin-top: 6px; } }
</style>
"""


# ============================================================================================
# data
# ============================================================================================
@st.cache_resource(show_spinner="Loading the FlavorSwap model...")
def load_model() -> Recommender:
    return Recommender.load()


@st.cache_data(show_spinner=False)
def load_review_support() -> dict[tuple[str, str], int]:
    """(source, target) -> number of Food.com users who reported making that swap."""
    p = pd.read_csv(paths.PROCESSED / "substitution_counts.csv", usecols=["source", "target", "n_users"])
    return {(s, t): int(n) for s, t, n in zip(p.source, p.target, p.n_users)}


@st.cache_data(show_spinner=False)
def load_config() -> dict:
    return json.loads((paths.MODELS / "hybrid_config.json").read_text())


@st.cache_data(show_spinner=False)
def n_recipes() -> int:
    """Number of Food.com recipes the embeddings were learned from (recipes with at least one
    vocabulary ingredient), as recorded by src/train_embeddings.py."""
    return int(json.loads((paths.MODELS / "ingredient_w2v_config.json").read_text())["corpus_recipes"])


def nice(name: str) -> str:
    return name[:1].upper() + name[1:] if name else name


def plural(n: int, word: str) -> str:
    return f"{n:,} {word}" + ("" if n == 1 else "s")


def is_missing(x) -> bool:
    return x is None or (isinstance(x, float) and math.isnan(x))


# ============================================================================================
# state callbacks (run before the script re-renders, so they may change widget values)
# ============================================================================================
def choose(name: str) -> None:
    st.session_state.ingredient = name
    st.session_state.typed = None


def on_ingredient_change() -> None:
    """Normalise free text ("Large Eggs") to the vocabulary name so the URL stays clean."""
    text = st.session_state.get("ingredient")
    st.session_state.typed = None
    if not text:
        return
    try:
        resolved = load_model().resolve(text)
    except UnknownIngredient:
        return
    if resolved != text:
        st.session_state.typed = text
        st.session_state.ingredient = resolved


def on_quick_pick() -> None:
    pick = st.session_state.get("quick_pick")
    if pick:
        choose(pick)
    st.session_state.quick_pick = None


def reset_weights(defaults: tuple[float, float]) -> None:
    st.session_state.w_flav, st.session_state.w_nutr = defaults


# ============================================================================================
# presentation helpers
# ============================================================================================
def context_word(cos: float) -> str:
    """Plain-language strength of the recipe-context (embedding) similarity."""
    return "very similar" if cos >= 0.6 else "similar" if cos >= 0.35 else "related" if cos >= 0.15 else "different"


def nutrient_changes(src: pd.Series, tgt: pd.Series) -> tuple[list[tuple[str, str]], bool]:
    """Notable per-100 g changes as (label, colour) chips, and whether any nutrient could be compared.
    Each chip carries an arrow and a sign, so meaning never relies on colour alone."""
    chips, compared = [], False
    for c, lab, unit, dec, lower_better, same in NUTRIENTS:
        a, b = src[c], tgt[c]
        if is_missing(a) or is_missing(b):
            continue
        compared = True
        d = float(b) - float(a)
        if abs(d) < same:
            continue
        good = d < 0 if lower_better else d > 0
        chips.append((f"{lab} {'↓' if d < 0 else '↑'} {abs(d):,.{dec}f} {unit}", "green" if good else "red"))
    return chips, compared


def health_badge(rec: Recommender, i: int, j: int, min_reduction: float) -> tuple[str, str, str]:
    """(label, colour, icon) comparing the health score of j with i, using the model's own rule."""
    hs, ht = rec.health[i], rec.health[j]
    if math.isnan(hs) or math.isnan(ht):
        return "No nutrition comparison", "gray", ":material/help:"
    need = rec.required_improvement(i, min_reduction)
    d = ht - hs
    amount = f"{100 * d / hs:+.0f}%" if hs >= PCT_DISPLAY_MIN else f"{d:+.2f}"
    if d <= -need:
        return f"Healthier ({amount})", "green", ":material/favorite:"
    if d >= need:
        return f"Less healthy ({amount})", "orange", ":material/warning:"
    return "Similar healthiness", "gray", ":material/drag_handle:"


def why_sentence(rec: Recommender, i: int, j: int, row: pd.Series) -> str:
    """Why the substitute fits: recipe context and shared aroma compounds, each cited only when it is a
    real reason; otherwise say what the ranking relied on."""
    cos = float(row.embedding_cosine)
    parts = [f"used in {context_word(cos)} recipes"] if cos >= MIN_CONTEXT_REASON else []
    n, overlap = row.shared_compounds, row.flavor_similarity
    if not is_missing(n) and n > 0 and not is_missing(overlap) and overlap >= MEANINGFUL_FLAVOR_OVERLAP:
        # the recommender's list, rarest compounds first (names may contain commas, so never split text)
        examples = [c for c in rec.shared_compounds(i, j, 10)[1] if len(c) <= EXAMPLE_NAME_MAX][:2]
        parts.append(f"shares {plural(int(n), 'aroma compound')}" + (f", e.g. {' and '.join(examples)}" if examples else ""))
    if not parts:
        return ("Rarely used in the same recipes - ranked here mainly for its nutrient profile and because cooks "
                "do report swaps between these food groups.")
    text = " and ".join(parts)
    return text[:1].upper() + text[1:] + "."


# ============================================================================================
# sections
# ============================================================================================
def sidebar(cfg: dict) -> tuple[int, float, float]:
    w = cfg["weights"]
    defaults = (float(w["flav"]), float(w["nutr"]))
    st.session_state.setdefault("w_flav", defaults[0])
    st.session_state.setdefault("w_nutr", defaults[1])
    with st.sidebar:
        st.header("Settings")
        st.caption("Optional. The defaults were tuned on substitutions that Food.com reviewers reported.")
        k = st.slider("Number of suggestions", 5, 20, 10, key="k")
        with st.expander("Ranking weights", icon=":material/tune:"):
            st.caption("Raise a weight to favour that evidence more.")
            flav = st.slider("Flavor-compound overlap", 0.0, 1.0, step=0.05, key="w_flav")
            nutr = st.slider("Nutrient-profile similarity", 0.0, 1.5, step=0.05, key="w_nutr")
            custom = (flav, nutr) != defaults
            st.button("Reset to tuned defaults", on_click=reset_weights, args=(defaults,),
                      disabled=not custom, icon=":material/restart_alt:", width="stretch")
            if custom:
                st.caption(f"Custom weights. Evaluated defaults: flavor {defaults[0]:g}, "
                           f"nutrition {defaults[1]:g}.")
        st.divider()
        st.caption("Computational Gastronomy mini project · Group 27")
    return k, flav, nutr


def banner(compact: bool) -> None:
    """Brand banner: large on the start page, a slim strip once results are shown."""
    st.html(STYLE)
    brand = ('<div class="fs-brand"><span class="fs-logo">⇄</span><span>FlavorSwap</span>'
             '<span class="fs-tag">Computational Gastronomy · Group 27</span></div>')
    if compact:
        st.html(f'<div class="fs-hero compact">{brand}<p class="fs-sub">Substitutes that cook like it, '
                'taste like it, or are better for you.</p></div>')
    else:
        st.html(f'<div class="fs-hero">{brand}<div class="fs-title">Swap any ingredient. Keep the flavor.</div>'
                '<p class="fs-sub">Search an ingredient and get substitutes that cook like it, taste like it, '
                'or are better for you - ranked and explained with recipe data, flavor chemistry and '
                'nutrition.</p></div>')


def search(rec: Recommender) -> str | None:
    left, right = st.columns([3, 2], vertical_alignment="bottom")
    with left:
        st.selectbox("Ingredient", options=rec.names, index=None, format_func=nice, key="ingredient",
                     placeholder=f"Search {len(rec.names):,} ingredients, e.g. butter, sour cream, rice",
                     accept_new_options=True, on_change=on_ingredient_change, bind="query-params")
    with right:
        st.segmented_control("Kind of swap", options=list(MODES), key="mode", default="Similar", required=True,
                             bind="query-params", width="stretch",
                             help="Similar: the closest replacement. Healthier: plausible replacements that are "
                                  "clearly healthier per 100 g.")
    row = st.container(horizontal=True, vertical_alignment="center", gap="small")
    row.caption("Try:", width="content")
    row.pills("Popular ingredients", options=QUICK_PICKS, format_func=nice, key="quick_pick",
              on_change=on_quick_pick, label_visibility="collapsed", wrap=True)
    return st.session_state.get("ingredient")


def landing(rec: Recommender, cfg: dict) -> None:
    st.space("small")
    test = cfg["test"]["default"]
    stats = st.columns(4)
    stats[0].metric("Ingredients", f"{len(rec.names):,}", border=True)
    stats[1].metric("Recipes learned from", f"{n_recipes():,}", border=True)
    stats[2].metric("Aroma compounds", f"{len(rec.compound_names):,}", border=True)
    stats[3].metric("Known swaps in the top 10", f"{test['Hit@10']:.0%}", border=True,
                 help="Hit@10 on held-out ingredients: share of substitutions reported by Food.com reviewers "
                      f"that appear in FlavorSwap's top 10 (out of {len(rec.names) - 1:,} candidates).")
    st.space("small")
    st.subheader("How it works", anchor=False)
    cols = st.columns(3)
    cards = [
        (":material/menu_book:", "Cooks like it",
         f"Ingredients used in the same kinds of dishes, learned from {n_recipes():,} Food.com recipes."),
        (":material/science:", "Tastes like it",
         "Shared aroma compounds, from the flavor network of Ahn et al. (2011)."),
        (":material/nutrition:", "Better for you",
         "USDA nutrition per 100 g finds swaps that are clearly healthier."),
    ]
    for col, (icon, title, text) in zip(cols, cards):
        with col.container(border=True):
            st.markdown(f"#### :green[{icon}] {title}")
            st.markdown(text)


def source_card(rec: Recommender, name: str) -> None:
    i = rec.index[name]
    row = rec.ing.iloc[i]
    with st.container(border=True):
        top = st.container(horizontal=True, vertical_alignment="center", gap="small")
        top.markdown(f"## {nice(name)}")
        top.badge(row.food_group, color="gray")
        top.badge(f"in {plural(int(row.n_recipes), 'recipe')}", icon=":material/menu_book:", color="blue")
        if rec.flav_has[i]:
            top.badge(plural(int(rec.flav_bin[i].sum()), "aroma compound"), icon=":material/science:",
                      color="violet")
        if is_missing(row.energy_kcal):
            st.caption("No reliable USDA nutrition data is linked to this ingredient, so nutrient changes "
                       "and healthier swaps are not available for it.")
            return
        st.caption("Nutrition per 100 g (USDA FoodData Central)")
        tiles = st.container(horizontal=True, wrap=True, gap="small")
        for c, label, unit, dec, _, _ in NUTRIENTS:
            v = row[c]
            tiles.metric(label, "n/a" if is_missing(v) else f"{v:,.{dec}f} {unit}", border=True, width="content")
        h = rec.health[i]
        tiles.metric("Health score", "n/a" if math.isnan(h) else f"{h:.2f}", help=HEALTH_HELP, border=True,
                     width="content")


def suggestion_card(rec: Recommender, source: str, row: pd.Series, support: dict, min_reduction: float) -> None:
    i, j = rec.index[source], rec.index[row.ingredient]
    label, colour, icon = health_badge(rec, i, j, min_reduction)
    reviewers = support.get((source, row.ingredient), 0)
    rank = int(row["rank"])
    with st.container(border=True):
        st.markdown(f"#### {rank} · {nice(row.ingredient)}")
        badges = st.container(horizontal=True, gap="small")
        if rank == 1:
            badges.badge("Top pick", icon=":material/star:", color="primary")
        badges.badge(row.food_group, color="gray")
        badges.badge(label, icon=icon, color=colour)
        if reviewers:
            badges.badge(f"{plural(reviewers, 'cook')} made this swap", icon=":material/forum:", color="blue",
                         help="Food.com reviewers who reported making exactly this swap.")
        st.markdown(why_sentence(rec, i, j, row))

        cos = float(row.embedding_cosine)
        bars = st.columns(2, gap="medium")
        bars[0].progress(min(max(cos, 0.0), 1.0), text=f"Recipe context: {context_word(cos)}")
        if is_missing(row.flavor_similarity):
            bars[1].caption(":material/science: No flavor-compound data for this pair.")
        else:
            n = int(row.shared_compounds)
            bars[1].progress(float(row.flavor_similarity),
                             text=f"Flavor: {plural(n, 'shared compound')}" if n else "Flavor: no shared compounds")

        chips, compared = nutrient_changes(rec.ing.iloc[i], rec.ing.iloc[j])
        if not compared:
            st.caption("No nutrient comparison (USDA data missing).")
        elif not chips:
            st.caption(f"Per 100 g: nutrition about the same as {source}.")
        else:
            line = st.container(horizontal=True, gap="small", vertical_alignment="center")
            line.caption(f"Per 100 g vs. {source}:", width="content")
            for text, c in chips:
                line.badge(text, color=c)
        st.button(f"Find swaps for {row.ingredient}", key=f"go-{row.ingredient}", on_click=choose,
                  args=(row.ingredient,), icon=":material/arrow_forward:", icon_position="right", type="tertiary")


def compare_table(rec: Recommender, source: str, df: pd.DataFrame, support: dict, min_reduction: float) -> None:
    i = rec.index[source]
    src = rec.ing.iloc[i]
    rows = []
    for r in df.itertuples(index=False):
        j = rec.index[r.ingredient]
        tgt = rec.ing.iloc[j]
        entry = {"Rank": int(r.rank), "Substitute": nice(r.ingredient), "Food group": r.food_group,
                 "Recipe context": max(float(r.embedding_cosine), 0.0),
                 "Flavor overlap": None if is_missing(r.flavor_similarity) else float(r.flavor_similarity),
                 "Shared compounds": None if is_missing(r.shared_compounds) else int(r.shared_compounds)}
        for c, lab, unit, dec, _, _ in NUTRIENTS:
            a, b = src[c], tgt[c]
            entry[f"Δ {lab} ({unit})"] = None if is_missing(a) or is_missing(b) else round(float(b - a), dec)
        entry["Health"] = health_badge(rec, i, j, min_reduction)[0]
        entry["Reviewer reports"] = support.get((source, r.ingredient), 0)
        rows.append(entry)
    table = pd.DataFrame(rows)
    cfg = {
        "Rank": st.column_config.NumberColumn(width="small"),
        "Recipe context": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f",
                                                          help="Cosine similarity of Word2Vec ingredient vectors"),
        "Flavor overlap": st.column_config.ProgressColumn(min_value=0.0, max_value=1.0, format="%.2f",
                                                          help="Ochiai overlap of flavor-compound sets"),
        "Reviewer reports": st.column_config.NumberColumn(help="Food.com reviewers who reported this swap"),
    }
    st.dataframe(table, hide_index=True, column_config=cfg, width="stretch")
    st.caption("Δ columns: substitute minus original, per 100 g. Empty cells: data not available.")


def empty_healthier_message(rec: Recommender, name: str) -> None:
    i = rec.index[name]
    if rec.whole_plant[i]:
        st.info(f"**{nice(name)}** is a whole fruit, vegetable, legume or nut - already one of the foods "
                "dietary guidelines recommend, so FlavorSwap does not rank it against others for health. "
                "Try *Similar* swaps for alternatives.", icon=":material/eco:")
    elif rec.little_to_limit(i):
        st.info(f"**{nice(name)}** has almost no calories, saturated fat, sugar or sodium to cut, so there "
                "is no meaningfully healthier swap. Try *Similar* swaps for alternatives.",
                icon=":material/eco:")
    elif math.isnan(rec.health[i]):
        st.info(f"There is no reliable USDA nutrition data for **{nice(name).lower()}**, so FlavorSwap "
                "cannot claim that any swap is healthier. Try *Similar* swaps instead.",
                icon=":material/info:")
    else:
        st.info(f"None of the plausible substitutes for **{nice(name).lower()}** is clearly healthier "
                "per 100 g (comparable form, at least 10% better health score). Try *Similar* swaps.",
                icon=":material/info:")


def results(rec: Recommender, source: str, mode: str, k: int, flav: float, nutr: float, support: dict) -> None:
    weights = replace(rec.weights, flav=flav, nutr=nutr)
    df = rec.healthier(source, k, weights) if mode == "Healthier" else rec.similar(source, k, weights)
    st.subheader(f"{MODES[mode]} for {nice(source).lower()}", anchor=False)
    if df.empty:
        empty_healthier_message(rec, source)
        return
    if mode == "Healthier":
        st.caption("Plausible substitutes that are clearly healthier per 100 g, best replacement first. "
                   "Assumes an equal-weight swap; sugars are total sugars (the health score counts only free sugars).")
    else:
        st.caption("Best replacement first, judged on recipe context, flavor chemistry, nutrient profile, how "
                   "common the ingredient is and how often people swap between these food groups.")
    cards_tab, table_tab = st.tabs(["Cards", "Table"])
    with cards_tab:
        for start in range(0, len(df), 2):
            cols = st.columns(2)
            for col, (_, row) in zip(cols, df.iloc[start:start + 2].iterrows()):
                with col:
                    suggestion_card(rec, source, row, support, weights.min_reduction)
    with table_tab:
        compare_table(rec, source, df, support, weights.min_reduction)


def how_it_works(cfg: dict) -> None:
    test = cfg["test"]["default"]
    with st.expander("How FlavorSwap works", icon=":material/help:"):
        st.markdown(f"""
**Scoring.** Every one of the 3,583 other ingredients is scored on five signals, each standardised:

- **Recipe context** - cosine similarity of Word2Vec vectors learned from the ingredient lists of
  {n_recipes():,} Food.com recipes (ingredients used in the same kinds of dishes end up close together);
- **Flavor-compound overlap** - shared aroma compounds from the flavor network of Ahn et al. (2011);
- **Nutrient-profile similarity** - per 100 g, from USDA FoodData Central;
- **Popularity** - ingredients people actually keep at home are preferred;
- **Food-group transitions** - how often reviewers swap between the two food groups.

**Evaluation.** Weights were tuned on 30% of ingredients and tested on the other 70%, against
2,895 substitutions that Food.com reviewers reported. A reported substitute appears in the top 10
for **{test['Hit@10']:.0%}** of held-out pairs and in first place for **{test['Hit@1']:.0%}**.

**Healthier swaps** are drawn only from plausible substitutes (similar by content, from a food
group people actually swap into, no spices for main ingredients), compared per 100 g only when the
foods are in a comparable form, and must improve the health score by at least 10%. {HEALTH_HELP}

**Keep in mind.** Suggestions are for an ingredient in general, not for a specific recipe, and
nutrient changes assume you swap equal weights. Taste and texture still need a cook's judgement.
""")


def footer() -> None:
    st.space("small")
    st.caption("Data: Food.com recipes and reviews (Majumder et al. 2019) · flavor compounds (Ahn et al. 2011) · "
               "USDA FoodData Central SR Legacy. Computational Gastronomy mini project, Group 27.")


def main() -> None:
    rec = load_model()
    support = load_review_support()
    cfg = load_config()
    k, flav, nutr = sidebar(cfg)
    top = st.empty()          # the banner goes above the search box, but its size depends on the search
    text = search(rec)
    with top.container():
        banner(compact=bool(text))
    if not text:
        landing(rec, cfg)
        how_it_works(cfg)
        footer()
        return
    try:
        source = rec.resolve(text)
    except UnknownIngredient as e:
        with st.container(border=True):
            st.error(f"FlavorSwap does not know **{e.text}** yet.", icon=":material/search_off:")
            if e.suggestions:
                st.markdown("Did you mean:")
                row = st.container(horizontal=True, gap="small")
                # most commonly used ingredients first (vocabulary is ordered by recipe count)
                for s in sorted(e.suggestions, key=rec.index.get):
                    row.button(nice(s), key=f"suggest-{s}", on_click=choose, args=(s,))
        how_it_works(cfg)
        footer()
        return
    typed = st.session_state.get("typed") or (text if text != source else None)
    if typed and typed != source:
        st.caption(f"Showing **{nice(source).lower()}** for “{typed}”.")
    source_card(rec, source)
    mode = st.session_state.get("mode") or "Similar"
    results(rec, source, mode, k, flav, nutr, support)
    how_it_works(cfg)
    footer()


main()
