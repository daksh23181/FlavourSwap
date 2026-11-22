# FlavorSwap: Flavor- and Nutrition-Aware Ingredient Substitution Recommender

Computational Gastronomy mini project - Group 27 (2023181, 2023142).

FlavorSwap suggests ingredient substitutes by combining recipe co-occurrence (ingredient
embeddings), shared flavor compounds and nutrition data, and recommends both *similar* and
*healthier* swaps.

| Deliverable | Status | Main outputs |
|---|---|---|
| 1. Curated dataset linking recipe ingredients to flavor compounds and nutrients | done | `data/processed/`, `reports/dataset_report.md` |
| 2. Word2Vec ingredient embeddings + interactive t-SNE/UMAP map | done | `models/`, `outputs/ingredient_map.html`, `reports/embedding_report.md` |
| 3. Hybrid substitution model (similar + healthier), Hit@k evaluation | done | `src/flavorswap/recommender.py`, `models/hybrid_config.json`, `data/processed/substitutions.csv`, `reports/substitution_report.md` |
| 4. Streamlit web app with ranked, explained substitutes | done | `app/streamlit_app.py` |
| 5. Recipe-level impact analysis across cuisines | done | `outputs/impact_dashboard.html`, `reports/impact_report.md`, `reports/figures/`, `reports/quantity_validation.md` |

## Data sources

| Source | What we use | Location after download |
|---|---|---|
| Food.com Recipes and Interactions (Kaggle, Majumder et al. 2019) | `RAW_recipes.csv`: 231,637 recipes with ingredient lists, tags and per-serving nutrition; `RAW_interactions.csv`: 1.13M reviews (mined for known substitutions) | `data/raw/foodcom/` |
| Food.com Recipes with Search Terms and Tags (Kaggle, same recipe ids) | `recipes_w_search_terms.csv`: the raw ingredient lines with quantities and units, servings and serving size (Deliverable 5) | `data/raw/foodcom/` |
| Ahn et al. 2011, *Flavor network and the principles of food pairing* (Sci. Rep. 1:196) | ingredient -> flavor-compound table (1,530 ingredients, 1,107 compounds) | `data/raw/flavor_network/` |
| USDA FoodData Central, SR Legacy (2018) | 7,793 foods with nutrients per 100 g, and household-measure gram weights (`food_portion.csv`, Deliverable 5) | `data/raw/usda_sr_legacy/` |

The proposal lists FooDB as an alternative flavor source; the Ahn et al. data was used because it is
an ingredient-level compound table and FooDB's bulk download (~1 GB) is not needed for that.
`data/raw/MANIFEST.txt` records SHA-256 checksums of every raw file used.

## How to run

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows   (source .venv/bin/activate on macOS/Linux)
pip install -r requirements.txt

python src/download_data.py        # ~500 MB download, no Kaggle account needed
python src/build_dataset.py        # Deliverable 1   (~1 min)
python src/train_embeddings.py     # Deliverable 2a  (~8 min, trains a 4-model grid)
python src/make_embedding_map.py   # Deliverable 2b  (~2 min)
python src/mine_substitutions.py   # Deliverable 3 ground truth (~20 s)
python src/evaluate_substitutions.py   # Deliverable 3 tuning + evaluation (~7 min)
python src/precompute_substitutions.py # Deliverable 3 table for every ingredient (~40 s)
python src/build_recipe_quantities.py  # Deliverable 5 amounts in grams + validation (~4 min)
python src/impact_analysis.py          # Deliverable 5 analysis, figures, dashboard (~3 min)
python src/impact_dashboard.py         # (optional) rebuild only the dashboard (~20 s)

python tests/test_mine_substitutions.py   # regression tests
python tests/test_quantities.py
python tests/test_recommender.py
python tests/test_app.py

streamlit run app/streamlit_app.py     # Deliverable 4: opens the web app in the browser
```

Every step is deterministic (fixed seeds, single-threaded Word2Vec, deterministic hash).

## Deliverable 1 - curated dataset

Pipeline (`src/build_dataset.py`):

1. **Ingredient normalisation** (`src/flavorswap/normalize.py`). Food.com stores ~15k distinct
   ingredient strings. Surface variants and true synonyms are merged ("eggs"/"large eggs" -> *egg*,
   "lemon, juice of" -> *lemon juice*, "scallion" -> *green onion*, "salt & pepper" -> *salt* +
   *black pepper*); kitchen supplies listed as ingredients (foil, skewers, ...) are dropped.
   Variants that differ nutritionally or are themselves plausible substitutes are deliberately kept
   apart (unsalted vs salted butter, skim vs whole milk, chicken stock vs broth, garlic powder vs
   garlic). Every raw -> canonical mapping is exported in `ingredient_aliases.csv`.
2. **Vocabulary**: canonical ingredients used in >= 20 recipes (covers 98% of ingredient occurrences).
3. **Flavor links** to Ahn et al.: hand-checked table `data/curation/ahn_links.csv`, then exact
   name match, then a head-noun fallback that may drop only "safe" modifiers (*red onion* -> *onion*,
   never *roasted red pepper* -> black *pepper*). Items without aroma compounds (salt, sugar,
   water, baking soda, ...) are explicitly marked as such.
4. **Nutrient links** to USDA SR Legacy: every one of the 800 most frequent ingredients (~91% of
   occurrences) is linked by hand (`data/curation/usda_links.csv`, written as USDA descriptions and
   resolved to exactly one `fdc_id`; proxies are labelled). Rarer ingredients use an automatic
   matcher; only complete, high-scoring matches are kept, partial ones are rejected.
5. **Audits**: random samples of the automatic links were judged by hand
   (`reports/*_auto_link_audit.csv`); every wrong link found was fixed in the curation tables.

Outputs in `data/processed/`:

| File | Content |
|---|---|
| `recipes.parquet` | one row per recipe: raw and canonical ingredients, per-serving nutrition (calories + %DV), cuisine label, link coverage |
| `ingredients.csv` | one row per vocabulary ingredient: frequency, food group, Ahn link + method, USDA link + method, nutrients per 100 g (energy, protein, fat, saturated fat, trans fat (reported + flagged estimate), carbohydrate, sugar, fiber, sodium, cholesterol, water) |
| `ingredient_compounds.csv` | long table ingredient -> flavor compound (id, name, CAS) |
| `compounds.csv` | the 1,107 Ahn et al. compounds |
| `ingredient_aliases.csv` | raw Food.com string -> canonical ingredient(s) |

Coverage and quality numbers: `reports/dataset_report.md`.

Known limitations, stated rather than hidden:
- Ahn et al. compound profiles are uneven (e.g. *olive oil* 3 compounds, *black tea* 239) and
  composite products (sauces, spice blends) have no profile.
- The `RAW_recipes` ingredient lists carry no quantities; recipe nutrition is the site's own
  per-serving estimate (Deliverable 5 recovers amounts from the raw ingredient lines of the second
  Food.com dump). Missing USDA nutrient values are left empty, never imputed.
- Cuisine labels come from Food.com tags; ~62% of recipes carry no cuisine tag ("Unspecified").

## Deliverable 2 - ingredient embeddings and similarity map

`src/train_embeddings.py` trains Word2Vec on recipes-as-sentences (window larger than any
recipe, no window shrinking, so ingredient order does not matter). A small grid (CBOW / skip-gram x
two down-sampling rates) is compared on **food-group coherence@10**, an intrinsic criterion that
does not use substitution pairs, so Deliverable 3's Hit@k evaluation stays unbiased.

`src/make_embedding_map.py` projects the vectors with UMAP and t-SNE (cosine) and writes
`outputs/ingredient_map.html` - a self-contained page (works offline) where you can switch
projection, highlight a food group, and search/click an ingredient to see its 10 nearest
neighbours (computed in the full embedding space) with their nutrition. Open it with
`?q=butter` appended to the URL to start on an ingredient.

Results: `reports/embedding_report.md`.

## Deliverable 3 - hybrid substitution model

**Ground truth.** `src/mine_substitutions.py` mines 1.13M Food.com reviews for substitutions people
report ("used greek yogurt instead of sour cream", "substituted honey for the sugar"). A mention is
kept only if the replaced ingredient is an ingredient of the reviewed recipe, the substitute is a known
ingredient directly attached to the cue phrase, and there is no negation or ambiguous coordination.
A manual audit of 98 random mentions found 97 correct. Pairs reported by >= 2 users form the
evaluation set: **2,895 directional pairs over 755 source ingredients**. The embeddings never saw
review text.

**Model** (`src/flavorswap/recommender.py`). Every candidate is scored by five standardised signals:
embedding cosine (Deliverable 2), flavor-compound overlap (Ahn et al.), nutrient-profile similarity
(USDA), a popularity prior and a food-group transition prior learned from non-test reviews. Weights
are tuned on 30% of source ingredients (validation) and evaluated once on the other 70% (test).

| test split (529 source ingredients, 2,108 pairs) | Hit@1 | Hit@10 | MRR |
|---|---|---|---|
| Random | 0.000 | 0.003 | 0.002 |
| Same food group + popularity | 0.053 | 0.258 | 0.117 |
| Embedding only (Deliverable 2) | 0.065 | 0.319 | 0.144 |
| **Hybrid (default)** | **0.146** | **0.504** | **0.261** |

The hybrid improves MRR over embeddings alone by +0.105 to +0.134 (95% paired bootstrap CI). Flavor
overlap is a weak predictor on its own; with a small weight (0.1, IDF-weighted) it is part of the
accuracy-optimal model on validation (see the report's *Flavor trade-off*).

**Healthier mode.** Ranks only candidates that are plausible (among the most similar by content, from
a food group people actually swap into, no spices for main ingredients), comparable per 100 g (no
dry-vs-cooked comparisons) and healthier by a health score aligned with nutrient-profiling systems
such as Nutri-Score: %DV of energy, saturated fat, free sugars, sodium and trans fat, minus fiber.
Whole fruit, vegetables, legumes and nuts, and ingredients with almost nothing to limit (vinegar,
lemon juice, wine, ice), get no health swaps. On test pairs where users made a genuinely healthier
swap, the healthier mode finds it in the top 10 for 71% of pairs (similar mode: 54%). Its top-5
suggestions cut saturated fat by 2.0 g and sodium by 612 mg per 100 g on average.

`data/processed/substitutions.csv` holds the top-10 similar and healthier substitutes, with
explanation fields, for all 3,584 ingredients (used by the app).

## Deliverable 4 - FlavorSwap web app

```bash
streamlit run app/streamlit_app.py
```

- **Three pages** (menu at the top): **Find swaps** (this deliverable, the start page), **Ingredient
  map** (Deliverable 2, `outputs/ingredient_map.html`) and **Recipe impact** (Deliverable 5,
  `outputs/impact_dashboard.html`); the two generated pages are embedded and follow the app's
  light/dark theme, so the deployed app shows every interactive deliverable.
- **Start page**: a banner, the search box, one-tap popular ingredients, key numbers (3,584
  ingredients, 231,089 recipes the embeddings learned from, 1,107 aroma compounds, 50% of known
  swaps in the top 10) and a
  three-step "how it works".
- **Search** any of the 3,584 ingredients (type-ahead), or type free text: "Large Eggs", "garlic
  cloves" and small typos ("chiken breast") are resolved; unknown words get "did you mean" buttons.
- **Similar / Healthier** switch ("Kind of swap").
- **Ingredient card**: food group, recipe count, aroma compounds, and nutrition per 100 g as tiles,
  including the health score (hover the ? for its definition).
- **Ranked substitute cards**: the first one is marked *Top pick*; badges show the food group, the
  health verdict and how many Food.com cooks reported exactly this swap; a one-line *why* (recipe
  context and shared aroma compounds, only when they are real reasons); two evidence bars (recipe
  context, flavor overlap); and only the nutrient changes that matter, as chips with arrows (green =
  better, red = worse). "Find swaps for ..." explores further.
- **Table** tab with a sortable table of the same results (all signals and nutrient differences).
- **Settings** (sidebar, collapsed by default): number of suggestions and optional ranking weights;
  the evaluated defaults are restored with one click.
- **Shareable links**: the ingredient and mode live in the URL, e.g.
  `http://localhost:8501/?ingredient=butter&mode=Healthier`.
- Light and dark themes (`.streamlit/config.toml`, switch in the top-right menu); layout adapts down
  to phone width. Developer items such as *Deploy* are hidden for presenting.

Testing: `tests/test_app.py` drives the app headlessly (start page, both modes, top pick, card
navigation, quick picks, weights and reset, missing-nutrition handling, every quick pick in both modes). Free-text
and unknown-ingredient input cannot be typed through Streamlit's test runner, so they were checked in
a real browser against the running app (`?ingredient=Large%20Eggs`, `?ingredient=mozarella`).


### Deploying the app (Streamlit Community Cloud, free)

The repository contains everything the app needs at runtime (~10 MB: `models/`, the small tables in
`data/processed/` and `data/raw/flavor_network/ingr_comp.tsv`); the raw downloads and large
intermediate files are excluded by `.gitignore` and are rebuilt by the pipeline. Files with Food.com
review text or user ids are not published; the app reads the counts-only
`data/processed/substitution_counts.csv`.

1. Push this repository to GitHub.
2. Sign in at <https://share.streamlit.io> with GitHub, click **Create app** -> **Deploy a public app
   from GitHub**, choose the repository, branch `main` and main file path `app/streamlit_app.py`.
3. Under **Advanced settings** choose **Python 3.12** (numpy 2.5 and scipy 1.18 need >= 3.12), then
   **Deploy**. Dependencies come from `app/requirements.txt` (app only; the pipeline uses
   `requirements.txt`). The theme and toolbar settings come from `.streamlit/config.toml`.

## Deliverable 5 - recipe-level impact of healthier swaps across cuisines

**Question.** If cooks used FlavorSwap's healthier suggestions in real recipes, how much would
calories, fat, saturated fat, sugars and sodium per serving change, how much flavor would be kept,
and does this differ between cuisines?

**Ingredient amounts** (`src/build_recipe_quantities.py`, `src/flavorswap/quantities.py`). Food.com's
raw ingredient lines ("1 1/2 cups flour", "2 (15 ounce) cans chicken broth", "1 stick butter") are
parsed and converted to grams with USDA household measures (cup, tbsp, "1 large", "1 clove", ...).
Recipes are kept only if every quantified main ingredient converts and the reconstructed calories
per serving are within 20% of Food.com's own value: **48,145 recipes** (median calorie error 2.4%;
details in `reports/quantity_validation.md`).

**Swaps** (`src/impact_analysis.py`). Each ingredient gets the first suggestion of the healthier mode
(Deliverable 3), swapped at equal mass; herbs, spices and salt are never swapped. Three scenarios:
the *best single swap* per recipe (the most realistic), *all swaps* (upper bound), and a real-world
check that uses only healthier swaps *Food.com reviewers actually reported*. Flavor kept = share of the
recipe's aroma compounds (Ahn et al.) still present after the swap. 95% bootstrap CIs; Kruskal-Wallis
tests between cuisines (cuisines with >= 75 recipes).

| best single swap, 48,145 recipes | mean change per serving (95% CI) | median |
|---|---|---|
| Calories | -6.1% [-6.2, -6.0] (-30 kcal) | -3.3% |
| Total fat | -9.5% [-9.7, -9.3] | 0.0% |
| Saturated fat | -14.8% [-15.1, -14.4] | -6.8% |
| Sugars | -1.7% [-1.9, -1.4] | 0.0% |
| Sodium | -10.2% [-10.4, -10.0] | 0.0% |
| Aroma compounds kept | 95.3% | |

- **One swap matters.** A single well-chosen swap cuts saturated fat per serving by about 15% on
  average while keeping ~95% of the recipe's aroma compounds. Swapping everything possible roughly
  doubles the effect (calories -11%, saturated fat -25%, sodium -29%) but keeps only 90% of the aroma.
- **Cuisines differ significantly** (Kruskal-Wallis p < 1e-20 for every outcome, small effect sizes).
  Saturated-fat cuts are largest for Spanish, North African, Middle Eastern and Indian recipes (-24 to
  -30%; mostly olive oil -> canola oil, and oil and yogurt swaps in Indian recipes); sodium cuts are
  largest for British/Irish, Japanese, German and Chinese recipes (butter -> unsalted butter, soy
  sauce and broth -> low-sodium versions). Total sugars rise slightly for Italian (tomato paste for
  sauce), Middle Eastern and Indian recipes (non-fat yogurt has a little more milk sugar), and
  Japanese recipes show a large but very uncertain increase (99 recipes; a few marinades).
- **Real-world check.** Using only swaps that reviewers reported gives similar calorie cuts (-7.3%)
  and larger saturated-fat cuts (-34%; the most common are butter -> cooking spray and sugar -> dates),
  but total fat goes *up* slightly (spray and oils are almost pure fat, compared at equal mass) and
  flavor retention drops to 88%: the model's swaps are not more aggressive than what people do.
- **Plausibility audit.** The 100 most frequent swaps (86% of recipes) were judged by hand
  (`reports/impact_swap_audit.csv`): 90% of those recipes received a sensible kitchen substitute.
  The implausible ones (e.g. brown sugar -> molasses, baking powder -> cream of tartar,
  buttermilk -> cooking apple) are listed in the report; restricted to plausible swaps, the
  results are essentially unchanged (calories -6.5%, saturated fat -16.6%).

Outputs:
- `outputs/impact_dashboard.html` - **presentation dashboard** (one file, opens offline in any
  browser). It follows the story of the talk: headline result (hero figure + stat tiles), the
  method in four steps, ten real recipes before/after their swap, a per-cuisine chart with 95% CIs
  for any outcome, a cuisine x outcome overview table, the most common swaps (questionable ones
  marked), and the reality check against reviewers' own swaps. A scenario switch (best single swap /
  reviewers' own swaps / every possible swap) updates the numbers; hover any mark for details.
  Light and dark mode (toggle button, remembered; or add `?theme=light` to the URL).
- `reports/impact_report.md` - full results per cuisine and scenario, statistics, audit, limitations.
- `reports/figures/impact_by_cuisine_{single,all,reviewer}.png` - static versions of the dashboard.
- `data/processed/recipe_swap_impact.parquet` (one row per recipe and scenario) and
  `recipe_swaps.parquet` (one row per applied swap).

Sugars here are total sugars per serving (the health score itself counts only free sugars).
Limitations: equal-mass swaps (cooks may adjust amounts; texture and baking chemistry are not
modelled), cooking losses are ignored, the validated sample favours recipes with well-specified
amounts, and cuisine labels are Food.com user tags.

## Repository layout

```
src/
  download_data.py        raw data download + checksums
  build_dataset.py        Deliverable 1
  train_embeddings.py     Deliverable 2 (model)
  make_embedding_map.py   Deliverable 2 (interactive map)
  mine_substitutions.py   Deliverable 3 ground truth from reviews
  evaluate_substitutions.py  Deliverable 3 tuning + Hit@k evaluation
  precompute_substitutions.py  Deliverable 3 table for every ingredient
  build_recipe_quantities.py   Deliverable 5 ingredient amounts in grams + validation
  impact_analysis.py      Deliverable 5 analysis + report
  impact_figures.py       Deliverable 5 static figures
  impact_dashboard.py     Deliverable 5 presentation dashboard
  flavorswap/             shared code: paths, normalisation, matching, food groups, nutrition,
                          recommender, quantities (ingredient-line parsing)
app/streamlit_app.py      web app entry point (page menu); pages in app/views/, shared look in app/ui.py
.streamlit/config.toml    app theme (light + dark)
tests/                    regression tests (miner, quantities, recommender invariants, app)
data/
  curation/               hand-checked link tables (the human knowledge in the pipeline)
  raw/                    downloaded data (not edited)
  processed/              Deliverable 1 outputs + embedding neighbours/coordinates
models/                   Word2Vec model and vectors
outputs/                  interactive map (D2) and impact dashboard (D5)
reports/                  generated reports and audits
```
