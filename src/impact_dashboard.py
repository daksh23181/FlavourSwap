"""Deliverable 5 - presentation dashboard: outputs/impact_dashboard.html.

A single self-contained page (no internet needed, no plotting library) that tells the impact story
in the order of a talk:

  1. headline       - what one healthier swap per recipe does, on average (hero figure + stat tiles)
  2. method         - how the numbers were made, in four steps
  3. example        - real recipes before / after their swap (curated, hand-audited swaps)
  4. cuisines       - one outcome at a time, every cuisine with its 95% CI (dot + whisker chart)
  5. overview       - cuisine x outcome table coloured by direction (doubles as the table view)
  6. swaps          - the most frequently applied swaps, questionable ones (hand audit) marked
  7. reality check  - FlavorSwap's swaps vs. swaps Food.com reviewers reported making

A scenario switch (best single swap / reviewer-reported swap / all swaps) drives sections 1, 4, 5, 6.
Light and dark mode (follows the OS; `?theme=light|dark` or the toggle button forces one).

Built by `src/impact_analysis.py`; can be rebuilt on its own from the saved outputs:
    python src/impact_dashboard.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flavorswap import paths  # noqa: E402

# (column stem, label used in impact_analysis tables, unit, decimals for absolute values)
OUTCOMES = [("energy_kcal", "Calories", "kcal", 0), ("fat_g", "Total fat", "g", 1),
            ("sat_fat_g", "Saturated fat", "g", 1), ("sugar_g", "Sugars", "g", 1),
            ("sodium_mg", "Sodium", "mg", 0)]
SCENARIOS = {
    "best single swap": ("Best single swap",
                         "Each recipe gets the one FlavorSwap swap that improves it most — what a cook would most likely do."),
    "reviewer-reported swap": ("Reviewers' own swaps",
                               "Only healthier swaps that at least two Food.com reviewers reported making — a real-world check."),
    "all swaps": ("Every possible swap",
                  "Every ingredient with a healthier substitute is swapped — the upper bound."),
}
# Example recipes for the "one recipe, one swap" section: popular recipes from different cuisines whose
# best single swap was judged plausible in the hand audit (reports/impact_swap_audit.csv), chosen to show
# different kinds of swaps, including one trade-off (less saturated fat, more sodium).
EXAMPLES = [
    (50144, "Crock Pot Potato Chowder"),
    (48760, "Szechuan Noodles with Spicy Beef Sauce"),
    (34688, "Potato Kielbasa Skillet"),
    (101085, "Gajar Halva (Indian Carrot Pudding)"),
    (90822, "Greek Lemon Potatoes"),
    (204109, "Homemade Flour Tortillas"),
    (140936, "Persian Eggs Poached in Tomato Sauce"),
    (135116, "Potatoes a l'Alsacienne"),
    (24088, "Strawberry Julius"),
    (106618, "Baked Stuffed Pasta Shells"),
]
AUDIT_FILE = paths.ROOT / "reports" / "impact_swap_audit.csv"


def _r(x, d: int = 1):
    """Rounded float for JSON (None for missing)."""
    if x is None:
        return None
    x = float(x)
    return round(x, d) if np.isfinite(x) else None


def _scenario(t: pd.DataFrame, s: pd.DataFrame, audit: pd.DataFrame) -> dict:
    rows = []
    for _, row in t.iterrows():
        rows.append({
            "c": row.cuisine, "n": int(row.recipes), "w": _r(100 * row["with a swap"]),
            "m": {k: _r(row[f"{lab} %"]) for k, lab, _, _ in OUTCOMES},
            "lo": {k: _r(row[f"{lab} % CI"][0]) for k, lab, _, _ in OUTCOMES},
            "hi": {k: _r(row[f"{lab} % CI"][1]) for k, lab, _, _ in OUTCOMES},
            "abs": {k: _r(row[f"{lab} abs"], d) for k, lab, _, d in OUTCOMES},
            "aroma": _r(100 * row["Aroma retained"]),
            "aroma_lo": _r(100 * row["Aroma retained CI"][0]), "aroma_hi": _r(100 * row["Aroma retained CI"][1]),
        })
    flags = {(a, b): p for a, b, p in zip(audit.ingredient, audit.substitute, audit.plausible)}
    top = (s.groupby(["ingredient", "substitute"])
           .agg(n=("recipe_id", "size"), kcal=("d_energy_kcal", "mean"), sat=("d_sat_fat_g", "mean"),
                sodium=("d_sodium_mg", "mean"), sugar=("d_sugar_g", "mean"))
           .sort_values("n", ascending=False).head(10).reset_index())
    swaps = [{"a": r.ingredient, "b": r.substitute, "n": int(r.n), "kcal": _r(r.kcal, 0), "sat": _r(r.sat, 1),
              "sodium": _r(r.sodium, 0), "sugar": _r(r.sugar, 1), "flag": flags.get((r.ingredient, r.substitute))}
             for r in top.itertuples()]
    return {"rows": rows, "swaps": swaps}


def _clean_line(line: str) -> str:
    return re.sub(r"\s+", " ", str(line)).strip()


def _examples(impact: pd.DataFrame, swaps: pd.DataFrame) -> list[dict]:
    ids = [rid for rid, _ in EXAMPLES]
    v = impact[(impact.scenario == "best single swap") & impact.recipe_id.isin(ids)].set_index("recipe_id")
    sw = swaps[(swaps.scenario == "best single swap") & swaps.recipe_id.isin(ids)].set_index("recipe_id")
    lines = pq.read_table(paths.PROCESSED / "recipe_ingredient_amounts.parquet",
                          columns=["recipe_id", "line_no", "raw_line", "ingredient"],
                          filters=[("recipe_id", "in", ids)]).to_pandas().sort_values(["recipe_id", "line_no"])
    inter = pd.read_csv(paths.RAW / "foodcom" / "RAW_interactions.csv", usecols=["recipe_id", "rating"])
    inter = inter[inter.recipe_id.isin(ids)]
    out = []
    for rid, title in EXAMPLES:
        if rid not in v.index or rid not in sw.index:
            print(f"  example {rid} ({title}) has no best single swap in this run - skipped")
            continue
        row, s = v.loc[rid], sw.loc[rid]
        lr = lines[lines.recipe_id == rid]
        ings = list(dict.fromkeys(i for i in lr.ingredient if isinstance(i, str)))
        raw = [_clean_line(x) for x in lr.raw_line[lr.ingredient == s.ingredient]]
        rated = inter.rating[(inter.recipe_id == rid) & (inter.rating > 0)]
        out.append({
            "id": int(rid), "title": title, "cuisine": row.cuisine, "servings": int(s.servings),
            "reviews": int((inter.recipe_id == rid).sum()), "rating": _r(rated.mean()) if len(rated) else None,
            "a": s.ingredient, "b": s.substitute, "g": _r(s.g, 0), "raw": raw, "ings": ings,
            "before": {k: _r(row[f"{k}_before"], d) for k, _, _, d in OUTCOMES},
            "after": {k: _r(row[f"{k}_before"] + row[f"d_{k}"], d) for k, _, _, d in OUTCOMES},
            "aroma": _r(100 * row.aroma_retained), "compounds": int(row.aroma_compounds_before),
        })
    return out


def _meta(impact: pd.DataFrame, swaps: pd.DataFrame, audit: pd.DataFrame, tables: dict) -> dict:
    summ = pd.read_parquet(paths.PROCESSED / "recipe_amounts_summary.parquet", columns=["validated", "kcal_ratio"])
    n_foodcom = len(pd.read_csv(paths.RAW_RECIPES, usecols=["id"]))
    s1 = swaps[swaps.scenario == "best single swap"][["recipe_id", "ingredient", "substitute"]]
    m = s1.merge(audit[["ingredient", "substitute", "plausible"]], on=["ingredient", "substitute"], how="left")
    judged = m[m.plausible.notna()]
    single = tables["best single swap"]
    return {
        "foodcom": n_foodcom, "recipes": int(impact.recipe_id.nunique()),
        "kcal_err": _r(100 * (summ.kcal_ratio[summ.validated] - 1).abs().median()),
        "cuisines": int((single.cuisine != "All recipes").sum()),
        "audit_pairs": len(audit), "audit_cover": _r(100 * len(judged) / len(m)),
        "audit_ok": _r(100 * judged.plausible.eq("yes").mean()),
    }


def build(tables: dict[str, pd.DataFrame], impact: pd.DataFrame, swaps: pd.DataFrame, out: Path) -> None:
    audit = pd.read_csv(AUDIT_FILE) if AUDIT_FILE.exists() else \
        pd.DataFrame(columns=["ingredient", "substitute", "plausible"])
    payload = {
        "outcomes": [{"k": k, "label": lab, "unit": u, "d": d} for k, lab, u, d in OUTCOMES],
        "scenarios": [{"key": k, "label": SCENARIOS[k][0], "note": SCENARIOS[k][1]} for k in SCENARIOS],
        "scen": {k: _scenario(tables[k], swaps[swaps.scenario == k], audit) for k in SCENARIOS},
        "examples": _examples(impact, swaps),
        "meta": _meta(impact, swaps, audit, tables),
    }
    html = TEMPLATE.replace("/*DATA_JSON*/", json.dumps(payload).replace("</", "<\\/"))
    out.parent.mkdir(exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"  wrote {out.relative_to(paths.ROOT)} ({len(html) / 1024:.0f} KB)")


def main() -> None:
    import impact_analysis as ia
    impact = pd.read_parquet(paths.PROCESSED / "recipe_swap_impact.parquet")
    swaps = pd.read_parquet(paths.PROCESSED / "recipe_swaps.parquet")
    rng = np.random.default_rng(ia.SEED)
    tables = {s: ia.cuisine_table(impact, s, rng) for s in ia.SCENARIOS}
    build(tables, impact, swaps, paths.OUTPUTS / "impact_dashboard.html")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Healthier Swaps Impact</title>
<style>
:root { color-scheme: light;
  --page:#f9f9f7; --surface:#fcfcfb; --surface-2:#f2f1ed; --ink:#0b0b0b; --ink-2:#52514e; --muted:#898781;
  --grid:#e1e0d9; --axis:#c3c2b7; --border:rgba(11,11,11,0.10); --wash:rgba(42,120,214,0.07);
  --s1:#2a78d6; --s2:#eb6834; --neg:#2a78d6; --pos:#e34948; --mid:#f0efec; --before:#898781;
  --good:#006300; --bad:#d03b3b; --warn:#fab219; --chip:#ecebe6; --chip-on:#0b0b0b; --chip-on-ink:#fcfcfb; }
@media (prefers-color-scheme: dark) { :root:where(:not([data-theme="light"])) { color-scheme: dark;
  --page:#0d0d0d; --surface:#1a1a19; --surface-2:#222220; --ink:#f0efec; --ink-2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10); --wash:rgba(57,135,229,0.12);
  --s1:#3987e5; --s2:#d95926; --neg:#3987e5; --pos:#e66767; --mid:#383835; --before:#898781;
  --good:#0ca30c; --bad:#e66767; --warn:#fab219; --chip:#2a2a28; --chip-on:#f0efec; --chip-on-ink:#1a1a19; } }
:root[data-theme="dark"] { color-scheme: dark;
  --page:#0d0d0d; --surface:#1a1a19; --surface-2:#222220; --ink:#f0efec; --ink-2:#c3c2b7; --muted:#898781;
  --grid:#2c2c2a; --axis:#383835; --border:rgba(255,255,255,0.10); --wash:rgba(57,135,229,0.12);
  --s1:#3987e5; --s2:#d95926; --neg:#3987e5; --pos:#e66767; --mid:#383835; --before:#898781;
  --good:#0ca30c; --bad:#e66767; --warn:#fab219; --chip:#2a2a28; --chip-on:#f0efec; --chip-on-ink:#1a1a19; }
* { box-sizing:border-box; }
html { scroll-behavior:smooth; }
@media (prefers-reduced-motion: reduce) { html { scroll-behavior:auto; } * { transition:none !important; } }
body { margin:0; background:var(--page); color:var(--ink); font:16px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif; }
.wrap { max-width:1180px; margin:0 auto; padding:0 28px; }
.num { font-variant-numeric:tabular-nums; }
button { font:inherit; color:inherit; }
:focus-visible { outline:2px solid var(--s1); outline-offset:2px; }

/* header */
header.top { padding:40px 0 20px; }
.toprow { display:flex; justify-content:space-between; align-items:flex-start; gap:16px; }
.eyebrow { font-size:13px; letter-spacing:.06em; text-transform:uppercase; color:var(--ink-2); font-weight:600; }
h1 { font-size:clamp(32px,4.6vw,52px); line-height:1.08; margin:10px 0 14px; font-weight:700; letter-spacing:-.02em; }
.lede { font-size:clamp(17px,1.6vw,20px); color:var(--ink-2); max-width:820px; margin:0; }
.lede b { color:var(--ink); }
.theme { border:1px solid var(--border); background:var(--surface); border-radius:999px; padding:7px 14px; cursor:pointer;
  font-size:14px; color:var(--ink-2); white-space:nowrap; }

/* scenario bar */
.bar { position:sticky; top:0; z-index:5; background:color-mix(in srgb, var(--page) 92%, transparent);
  backdrop-filter:blur(8px); border-bottom:1px solid var(--border); }
.bar .wrap { display:flex; align-items:center; gap:14px; flex-wrap:wrap; padding-top:10px; padding-bottom:10px; }
.bar .lbl { font-size:13px; font-weight:600; color:var(--ink-2); }
.seg { display:inline-flex; background:var(--chip); border-radius:10px; padding:3px; gap:2px; }
.seg button { border:0; background:transparent; padding:7px 14px; border-radius:8px; cursor:pointer; font-size:14px;
  font-weight:600; color:var(--ink-2); }
.seg button[aria-pressed="true"] { background:var(--surface); color:var(--ink); box-shadow:0 1px 2px rgba(0,0,0,.12); }
.bar .note { font-size:14px; color:var(--ink-2); flex:1 1 320px; }

/* sections */
section { padding:44px 0 8px; }
.kicker { font-size:13px; font-weight:700; letter-spacing:.06em; text-transform:uppercase; color:var(--s1); }
h2 { font-size:clamp(24px,2.6vw,32px); line-height:1.2; margin:6px 0 8px; letter-spacing:-.01em; }
.sub { color:var(--ink-2); margin:0 0 20px; max-width:860px; }
.card { background:var(--surface); border:1px solid var(--border); border-radius:16px; padding:22px; }
.takeaway { font-size:17px; margin:0 0 14px; }
.takeaway b { font-weight:700; }

/* headline */
.headline { display:grid; grid-template-columns:minmax(0,1.1fr) minmax(0,2fr); gap:16px; }
.hero { display:flex; flex-direction:column; justify-content:center; }
.hero .big { font-size:clamp(56px,7vw,88px); font-weight:700; line-height:1; letter-spacing:-.03em; }
.hero .what { font-size:20px; font-weight:600; margin-top:10px; }
.hero .ctx { color:var(--ink-2); margin-top:6px; font-size:15px; }
.tiles { display:grid; grid-template-columns:repeat(3,minmax(0,1fr)); gap:16px; }
.tile { background:var(--surface); border:1px solid var(--border); border-radius:16px; padding:18px 18px 16px; }
.tile .label { font-size:14px; color:var(--ink-2); font-weight:600; }
.tile .value { font-size:36px; font-weight:700; letter-spacing:-.02em; margin-top:4px; line-height:1.1; }
.tile .dir { font-size:13px; font-weight:600; margin-top:6px; display:flex; align-items:center; gap:6px; }
.tile .dir .ic { font-size:11px; }
.tile .meta { font-size:13px; color:var(--muted); margin-top:2px; }
.good { color:var(--good); } .bad { color:var(--bad); }

/* method */
.steps { display:grid; grid-template-columns:repeat(4,minmax(0,1fr)); gap:16px; counter-reset:step; }
.step { background:var(--surface); border:1px solid var(--border); border-radius:16px; padding:20px; position:relative; }
.step .n { width:28px; height:28px; border-radius:50%; background:var(--s1); color:#fff; font-weight:700; font-size:14px;
  display:flex; align-items:center; justify-content:center; }
.step .fig { font-size:28px; font-weight:700; margin-top:12px; letter-spacing:-.01em; }
.step .txt { color:var(--ink-2); font-size:15px; margin-top:4px; }

/* example */
.chips { display:flex; flex-wrap:wrap; gap:8px; margin-bottom:16px; }
.chip { border:0; background:var(--chip); color:var(--ink); border-radius:999px; padding:7px 14px; cursor:pointer; font-size:14px; }
.chip[aria-pressed="true"] { background:var(--chip-on); color:var(--chip-on-ink); font-weight:600; }
.ex { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1.25fr); gap:16px; }
.ex h3 { font-size:26px; margin:0; line-height:1.2; letter-spacing:-.01em; }
.ex .facts { color:var(--ink-2); font-size:14px; margin:8px 0 18px; display:flex; flex-wrap:wrap; gap:6px 14px; }
.swapbox { background:var(--surface-2); border-radius:12px; padding:16px; }
.swapbox .cap { font-size:12px; font-weight:700; letter-spacing:.06em; text-transform:uppercase; color:var(--ink-2); }
.swapline { display:flex; align-items:center; gap:12px; flex-wrap:wrap; margin-top:8px; font-size:20px; font-weight:700; }
.swapline .from { text-decoration:line-through; text-decoration-thickness:2px; color:var(--ink-2); font-weight:600; }
.swapline .arrow { color:var(--s1); }
.swapbox .raw { font-size:14px; color:var(--ink-2); margin-top:8px; }
.ings { display:flex; flex-wrap:wrap; gap:6px; margin-top:16px; }
.ing { font-size:13px; border:1px solid var(--border); border-radius:999px; padding:3px 10px; color:var(--ink-2); background:var(--surface); }
.ing.swapped { border-color:var(--s1); color:var(--ink); font-weight:600; background:var(--wash); }
.nav { display:flex; gap:8px; margin-top:16px; }
.nav button { border:1px solid var(--border); background:var(--surface); border-radius:10px; padding:6px 14px; cursor:pointer; font-size:14px; }
.legend { display:flex; gap:18px; flex-wrap:wrap; font-size:13px; color:var(--ink-2); margin-bottom:10px; align-items:center; }
.key { display:inline-flex; align-items:center; gap:6px; }
.key .dot { width:10px; height:10px; border-radius:50%; display:inline-block; }
.key .sq { width:12px; height:12px; border-radius:3px; display:inline-block; }
.db { display:grid; grid-template-columns:118px minmax(0,1fr) 170px; align-items:center; gap:12px; padding:9px 0;
  border-bottom:1px solid var(--grid); }
.db:last-child { border-bottom:0; }
.db .nm { font-weight:600; font-size:15px; }
.db .vals { text-align:right; font-size:14px; }
.db .vals .chg { display:block; font-size:13px; font-weight:600; }
.track { position:relative; height:24px; }
.track .rail { position:absolute; left:0; right:0; top:11px; height:2px; background:var(--grid); border-radius:1px; }
.track .link { position:absolute; top:11px; height:2px; background:var(--s1); opacity:.45; }
.track .pt { position:absolute; top:6px; width:12px; height:12px; margin-left:-6px; border-radius:50%;
  box-shadow:0 0 0 2px var(--surface); }
.aroma-row { display:flex; justify-content:space-between; align-items:baseline; gap:12px; padding-top:12px; font-size:15px; }

/* cuisine chart */
.toolbar { display:flex; gap:8px; flex-wrap:wrap; margin-bottom:14px; }
svg text { font-family:system-ui,-apple-system,"Segoe UI",sans-serif; }
.hit { fill:transparent; cursor:default; }
.hit:hover, .hit:focus { fill:var(--wash); outline:none; }

/* heat table */
.tablewrap { overflow-x:auto; }
table { border-collapse:separate; border-spacing:2px; width:100%; font-size:14px; }
th { font-weight:600; color:var(--ink-2); text-align:right; padding:6px 8px; font-size:13px; vertical-align:bottom; }
th:first-child, td:first-child { text-align:left; }
td { padding:7px 8px; text-align:right; border-radius:6px; white-space:nowrap; }
td.cell { min-width:78px; }
tr.all td { font-weight:700; }
tr.all td:first-child { border-bottom:0; }
.scale { display:flex; align-items:center; gap:10px; font-size:13px; color:var(--ink-2); margin-top:12px; flex-wrap:wrap; }
.scale .grad { width:220px; height:10px; border-radius:5px;
  background:linear-gradient(90deg, var(--neg), var(--mid), var(--pos)); }

/* swaps */
.swaprow { display:grid; grid-template-columns:minmax(150px,1.3fr) minmax(0,2fr) minmax(170px,1.2fr); gap:14px; align-items:center;
  padding:8px 0; border-bottom:1px solid var(--grid); }
.swaprow:last-child { border-bottom:0; }
.swaprow .pair { font-weight:600; font-size:15px; }
.swaprow .pair .ar { color:var(--s1); padding:0 4px; }
.swaprow .flag { display:inline-flex; align-items:center; gap:5px; font-size:12px; font-weight:600; color:var(--ink-2);
  margin-left:6px; white-space:nowrap; }
.swaprow .flag .wd { width:8px; height:8px; border-radius:50%; background:var(--warn); }
.hbar { display:flex; align-items:center; gap:8px; }
.hbar .b { height:16px; background:var(--s1); border-radius:0 4px 4px 0; min-width:2px; }
.hbar .v { font-size:14px; color:var(--ink); }
.swaprow .eff { font-size:13px; color:var(--ink-2); text-align:right; }

/* reality */
.two { display:grid; grid-template-columns:minmax(0,1.6fr) minmax(0,1fr); gap:16px; align-items:start; }
.points { margin:0; padding-left:20px; color:var(--ink-2); }
.points li { margin-bottom:10px; }
.points b { color:var(--ink); }

/* tooltip */
#tip { position:fixed; z-index:20; pointer-events:none; background:var(--surface); color:var(--ink); border:1px solid var(--border);
  border-radius:10px; padding:10px 12px; font-size:13px; box-shadow:0 6px 24px rgba(0,0,0,.16); opacity:0; transition:opacity .08s;
  max-width:280px; }
#tip .t { font-weight:700; margin-bottom:4px; }
#tip .r { display:flex; justify-content:space-between; gap:16px; }
#tip .r .k { color:var(--ink-2); }
#tip .r .v { font-weight:700; font-variant-numeric:tabular-nums; }

footer { padding:40px 0 56px; color:var(--ink-2); font-size:14px; }
footer h3 { font-size:15px; color:var(--ink); margin:0 0 8px; }
footer ul { margin:0; padding-left:18px; }
footer li { margin-bottom:6px; }
.fgrid { display:grid; grid-template-columns:minmax(0,1.4fr) minmax(0,1fr); gap:28px; border-top:1px solid var(--border); padding-top:24px; }

@media (max-width:980px) {
  .headline, .ex, .two, .fgrid { grid-template-columns:minmax(0,1fr); }
  .steps { grid-template-columns:repeat(2,minmax(0,1fr)); }
}
@media (max-width:640px) {
  .wrap { padding:0 16px; }
  .tiles { grid-template-columns:repeat(2,minmax(0,1fr)); }
  .tile .value { font-size:28px; }
  .steps { grid-template-columns:minmax(0,1fr); }
  .db { grid-template-columns:92px minmax(0,1fr); }
  .db .vals { grid-column:1 / -1; text-align:left; display:flex; gap:10px; }
  .db .vals .chg { display:inline; }
  .swaprow { grid-template-columns:minmax(0,1fr); gap:4px; }
  .swaprow .eff { text-align:left; }
  .seg { display:flex; width:100%; }
  .seg button { flex:1 1 0; padding:7px 6px; font-size:13px; }
  .card { padding:16px; }
}
</style>
</head>
<body>
<header class="top">
  <div class="wrap">
    <div class="toprow">
      <div class="eyebrow">FlavorSwap &middot; Deliverable 5 &middot; Recipe-level impact</div>
      <button class="theme" id="theme" type="button" aria-label="Toggle dark mode">Dark mode</button>
    </div>
    <h1>Healthier swaps, same flavor</h1>
    <p class="lede" id="lede"></p>
  </div>
</header>

<div class="bar" role="region" aria-label="Scenario">
  <div class="wrap">
    <span class="lbl">Scenario</span>
    <div class="seg" id="scen" role="group" aria-label="Scenario"></div>
    <span class="note" id="scen-note"></span>
  </div>
</div>

<main class="wrap">
  <section id="headline" aria-labelledby="h-headline">
    <div class="kicker">The result</div>
    <h2 id="h-headline">What healthier swaps do to a recipe</h2>
    <p class="sub">Average change per serving across all validated recipes, with 95% confidence intervals.</p>
    <div class="headline">
      <div class="card hero" id="hero"></div>
      <div class="tiles" id="tiles"></div>
    </div>
  </section>

  <section id="method" aria-labelledby="h-method">
    <div class="kicker">How we measured it</div>
    <h2 id="h-method">From recipe text to nutrients and flavor</h2>
    <div class="steps" id="steps"></div>
  </section>

  <section id="example" aria-labelledby="h-example">
    <div class="kicker">One recipe, one swap</div>
    <h2 id="h-example">Real recipes, before and after</h2>
    <p class="sub">Popular Food.com recipes with their best single swap (each checked by hand). Values are per serving.</p>
    <div class="chips" id="ex-chips" role="group" aria-label="Example recipe"></div>
    <div class="ex">
      <div class="card" id="ex-left"></div>
      <div class="card" id="ex-right"></div>
    </div>
  </section>

  <section id="cuisines" aria-labelledby="h-cuisines">
    <div class="kicker">Across cuisines</div>
    <h2 id="h-cuisines">Which cuisines gain the most?</h2>
    <p class="sub" id="cu-sub"></p>
    <div class="card">
      <div class="toolbar" id="metric" role="group" aria-label="Outcome"></div>
      <p class="takeaway" id="cu-take"></p>
      <div id="cu-chart"></div>
    </div>
  </section>

  <section id="overview" aria-labelledby="h-overview">
    <div class="kicker">At a glance</div>
    <h2 id="h-overview">Every cuisine, every outcome</h2>
    <p class="sub">Mean change per serving. Blue = lower than the original recipe, red = higher; stronger colour = bigger change.</p>
    <div class="card">
      <div class="tablewrap"><table id="heat"></table></div>
      <div class="scale"><span>lower</span><span class="grad" aria-hidden="true"></span><span>higher</span>
        <span>&middot; colour saturates at &plusmn;30%</span></div>
    </div>
  </section>

  <section id="swaps" aria-labelledby="h-swaps">
    <div class="kicker">What gets swapped</div>
    <h2 id="h-swaps">The most common swaps</h2>
    <p class="sub" id="sw-sub"></p>
    <div class="card" id="sw-list"></div>
  </section>

  <section id="reality" aria-labelledby="h-reality">
    <div class="kicker">Reality check</div>
    <h2 id="h-reality">FlavorSwap vs. what home cooks actually did</h2>
    <p class="sub">Average change per serving (%) for FlavorSwap's best single swap and for the best healthier swap that
      Food.com reviewers reported making, over the same recipes. Aroma kept is in percentage points below 100%.</p>
    <div class="two">
      <div class="card">
        <div class="legend"><span class="key"><span class="sq" style="background:var(--s1)"></span>FlavorSwap (best single swap)</span>
          <span class="key"><span class="sq" style="background:var(--s2)"></span>Reviewers' own swaps</span></div>
        <div id="re-chart"></div>
      </div>
      <div class="card"><ul class="points" id="re-points"></ul></div>
    </div>
  </section>
</main>

<footer class="wrap">
  <div class="fgrid">
    <div>
      <h3>How to read these numbers</h3>
      <ul id="caveats"></ul>
    </div>
    <div>
      <h3>Data</h3>
      <ul>
        <li>Recipes and reviews: Food.com (Majumder et al., 2019)</li>
        <li>Nutrients and household measures: USDA FoodData Central, SR Legacy</li>
        <li>Flavor compounds: Ahn et al., <i>Flavor network and the principles of food pairing</i>, Sci. Rep. 2011</li>
        <li>Full tables and statistics: <code>reports/impact_report.md</code></li>
      </ul>
    </div>
  </div>
</footer>
<div id="tip" role="tooltip"></div>

<script>
const DATA = /*DATA_JSON*/;
const OUT = DATA.outcomes;
const M = DATA.meta;
const ALL = "All recipes";
let scen = "best single swap", metric = "sat_fat_g", exIdx = 0;

// ---------- helpers
const minus = "−";
function signed(v, d = 1) {
  if (v === null || v === undefined) return "–";
  const r = Number(v.toFixed(d));
  if (r === 0) return (0).toFixed(d);
  return (r > 0 ? "+" : minus) + Math.abs(r).toFixed(d);
}
const fmt = (v, d = 0) => v === null || v === undefined ? "–" : Number(v).toLocaleString("en-US", {minimumFractionDigits: d, maximumFractionDigits: d});
const pct = (v, d = 1) => signed(v, d) + "%";
function el(tag, cls, text) { const e = document.createElement(tag); if (cls) e.className = cls; if (text !== undefined) e.textContent = text; return e; }
function svgEl(tag, attrs) { const e = document.createElementNS("http://www.w3.org/2000/svg", tag); for (const k in attrs) e.setAttribute(k, attrs[k]); return e; }
const label = k => k === "aroma" ? "Aroma kept" : OUT.find(o => o.k === k).label;
const S = () => DATA.scen[scen];
const allRow = () => S().rows.find(r => r.c === ALL);
const cuisineRows = () => S().rows.filter(r => r.c !== ALL);
function niceTicks(lo, hi, n) {
  const raw = (hi - lo) / n, mag = Math.pow(10, Math.floor(Math.log10(raw))), f = raw / mag;
  const step = (f < 1.5 ? 1 : f < 3 ? 2 : f < 7 ? 5 : 10) * mag, out = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + 1e-9; v += step) out.push(Number(v.toFixed(10)));
  return out;
}

// ---------- tooltip
const tip = document.getElementById("tip");
function showTip(target, title, rows, evt) {
  tip.replaceChildren(el("div", "t", title));
  for (const [k, v] of rows) { const r = el("div", "r"); r.append(el("span", "k", k), el("span", "v", v)); tip.append(r); }
  tip.style.opacity = 1;
  const b = target.getBoundingClientRect();
  let x = evt && evt.clientX !== undefined ? evt.clientX + 14 : b.right + 8;
  let y = evt && evt.clientY !== undefined ? evt.clientY + 14 : b.top;
  const w = tip.offsetWidth, h = tip.offsetHeight;
  if (x + w > innerWidth - 8) x = Math.max(8, (evt && evt.clientX !== undefined ? evt.clientX : b.left) - w - 14);
  if (y + h > innerHeight - 8) y = Math.max(8, innerHeight - h - 8);
  tip.style.left = x + "px"; tip.style.top = y + "px";
}
const hideTip = () => { tip.style.opacity = 0; };
function hover(node, title, rowsFn) {
  node.setAttribute("tabindex", "0");
  node.addEventListener("pointermove", e => showTip(node, title, rowsFn(), e));
  node.addEventListener("pointerleave", hideTip);
  node.addEventListener("focus", () => showTip(node, title, rowsFn()));
  node.addEventListener("blur", hideTip);
}

// ---------- header + method + caveats (static)
document.getElementById("lede").replaceChildren(
  "We applied FlavorSwap's healthier suggestions to ", el("b", null, fmt(M.recipes) + " real Food.com recipes"),
  " and measured what happens to calories, fat, saturated fat, sugars and sodium per serving — and how much of each recipe's flavor survives.");

const steps = [
  [fmt(M.foodcom), "Food.com recipes. Ingredient lines such as “1 1/2 cups flour” are converted to grams with USDA household measures."],
  [fmt(M.recipes), "recipes kept whose rebuilt calories match Food.com's own figure within 20% (median error " + M.kcal_err + "%)."],
  ["1 swap", "FlavorSwap's top healthier suggestion, at the same weight. Herbs, spices and salt are never swapped."],
  ["6 outcomes", "Calories, fat, saturated fat, sugars and sodium per serving (USDA), plus the share of aroma compounds kept (Ahn et al.)."],
];
const stepsEl = document.getElementById("steps");
steps.forEach(([fig, txt], i) => { const s = el("div", "step"); s.append(el("div", "n", String(i + 1)), el("div", "fig", fig), el("div", "txt", txt)); stepsEl.append(s); });

const caveats = [
  "Swaps are compared at equal weight (e.g. 30 g butter → 30 g olive oil); cooks may adjust amounts, and texture or baking chemistry is not modelled.",
  "Percentages are averages over recipes; a recipe with very little sugar or sodium can show a large percentage change, which widens the intervals of small cuisines.",
  "Hand audit: the " + M.audit_pairs + " most frequent swaps cover " + M.audit_cover + "% of recipes; " + M.audit_ok + "% of those recipes received a sensible kitchen substitute.",
  "Aroma kept counts which flavor compounds are still present, not their amounts. Cooking losses are ignored, as in Food.com's own nutrition figures.",
  "Cuisines are Food.com user tags; " + M.cuisines + " cuisines with at least 75 validated recipes are compared.",
];
document.getElementById("caveats").replaceChildren(...caveats.map(t => el("li", null, t)));

// ---------- scenario switch
const segEl = document.getElementById("scen");
DATA.scenarios.forEach(s => {
  const b = el("button", null, s.label); b.type = "button"; b.dataset.key = s.key;
  b.onclick = () => { scen = s.key; renderScenario(); };
  segEl.append(b);
});

// ---------- headline
function dirNode(v, k) {
  const d = el("div", "dir");
  if (k === "aroma") { d.append(el("span", null, "of the recipe's aroma compounds kept")); d.style.color = "var(--ink-2)"; return d; }
  const r = Number(v.toFixed(1));
  if (r < 0) { d.classList.add("good"); d.append(el("span", "ic", "▼"), el("span", null, "lower")); }
  else if (r > 0) { d.classList.add("bad"); d.append(el("span", "ic", "▲"), el("span", null, "higher")); }
  else { d.style.color = "var(--ink-2)"; d.append(el("span", null, "no change on average")); }
  return d;
}
function renderHeadline() {
  const a = allRow();
  const hero = document.getElementById("hero");
  hero.replaceChildren(el("div", "big", pct(a.m.sat_fat_g)), el("div", "what", "saturated fat per serving"),
    el("div", "ctx", fmt(a.n) + " recipes · " + fmt(a.w, 1) + "% had a healthier swap available · "
      + fmt(a.aroma, 1) + "% of aroma compounds kept"));
  const tiles = document.getElementById("tiles");
  tiles.replaceChildren();
  for (const k of ["energy_kcal", "fat_g", "sugar_g", "sodium_mg", "aroma"]) {
    const t = el("div", "tile");
    if (k === "aroma") {
      t.append(el("div", "label", "Aroma kept"), el("div", "value", fmt(a.aroma, 1) + "%"), dirNode(0, "aroma"),
        el("div", "meta", "95% CI " + fmt(a.aroma_lo, 1) + " to " + fmt(a.aroma_hi, 1) + "%"));
    } else {
      const o = OUT.find(x => x.k === k);
      t.append(el("div", "label", o.label), el("div", "value", pct(a.m[k])), dirNode(a.m[k], k),
        el("div", "meta", signed(a.abs[k], o.d) + " " + o.unit + " per serving · 95% CI " + signed(a.lo[k]) + " to " + signed(a.hi[k]) + "%"));
    }
    tiles.append(t);
  }
  // Saturated fat is the hero: one tile slot left -> show the number of recipes with a swap
  const t = el("div", "tile");
  t.append(el("div", "label", "Recipes analysed"), el("div", "value", fmt(a.n)),
    el("div", "dir", cuisineRows().length + " cuisines compared"), el("div", "meta", "validated against Food.com calories"));
  t.querySelector(".dir").style.color = "var(--ink-2)";
  tiles.append(t);
}

// ---------- example recipes
const exChips = document.getElementById("ex-chips");
DATA.examples.forEach((x, i) => {
  const b = el("button", "chip", x.title); b.type = "button";
  b.onclick = () => { exIdx = i; renderExample(); };
  exChips.append(b);
});
const nice = s => s.charAt(0).toUpperCase() + s.slice(1);
function renderExample() {
  const x = DATA.examples[exIdx];
  [...exChips.children].forEach((b, i) => b.setAttribute("aria-pressed", String(i === exIdx)));
  const L = document.getElementById("ex-left");
  const facts = el("div", "facts");
  facts.append(el("span", null, x.cuisine), el("span", null, x.servings + " servings"));
  if (x.rating !== null) facts.append(el("span", null, "★ " + x.rating.toFixed(1) + " (" + fmt(x.reviews) + " reviews)"));
  const box = el("div", "swapbox");
  const line = el("div", "swapline");
  line.append(el("span", "from", nice(x.a)), el("span", "arrow", "→"), el("span", null, nice(x.b)));
  box.append(el("div", "cap", "The swap"), line,
    el("div", "raw", "Recipe line: “" + x.raw.join("; ") + "” → the same weight of " + x.b + " (" + fmt(x.g) + " g per serving)."));
  const ings = el("div", "ings");
  for (const n of x.ings) ings.append(el("span", n === x.a ? "ing swapped" : "ing", n === x.a ? x.b + " (was " + n + ")" : n));
  const nav = el("div", "nav");
  const prev = el("button", null, "← Previous"), next = el("button", null, "Next →");
  prev.type = next.type = "button";
  prev.onclick = () => { exIdx = (exIdx + DATA.examples.length - 1) % DATA.examples.length; renderExample(); };
  next.onclick = () => { exIdx = (exIdx + 1) % DATA.examples.length; renderExample(); };
  nav.append(prev, next);
  L.replaceChildren(el("h3", null, x.title), facts, box, ings, nav);

  const R = document.getElementById("ex-right");
  const leg = el("div", "legend");
  const k1 = el("span", "key"), d1 = el("span", "dot"); d1.style.background = "var(--before)"; k1.append(d1, "Original recipe");
  const k2 = el("span", "key"), d2 = el("span", "dot"); d2.style.background = "var(--s1)"; k2.append(d2, "With the swap");
  leg.append(k1, k2);
  R.replaceChildren(leg);
  for (const o of OUT) {
    const b = x.before[o.k], a = x.after[o.k];
    const row = el("div", "db");
    const max = Math.max(b, a) * 1.12 || 1;
    const tr = el("div", "track");
    const rail = el("div", "rail"), link = el("div", "link");
    const lo = Math.min(a, b) / max * 100, hi = Math.max(a, b) / max * 100;
    link.style.left = lo + "%"; link.style.width = (hi - lo) + "%";
    const pb = el("div", "pt"), pa = el("div", "pt");
    pb.style.left = (b / max * 100) + "%"; pb.style.background = "var(--before)";
    pa.style.left = (a / max * 100) + "%"; pa.style.background = "var(--s1)";
    tr.append(rail, link, pb, pa);
    const vals = el("div", "vals num");
    vals.append(fmt(b, o.d) + " → " + fmt(a, o.d) + " " + o.unit);
    const ch = b > 0 ? (a - b) / b * 100 : 0;
    const chg = el("span", "chg");
    const r = Number(ch.toFixed(0));
    chg.textContent = r === 0 ? "no change" : (r < 0 ? "▼ " : "▲ ") + signed(ch, 0) + "%";
    chg.className = "chg " + (r < 0 ? "good" : r > 0 ? "bad" : "");
    if (r === 0) chg.style.color = "var(--ink-2)";
    vals.append(chg);
    row.append(el("div", "nm", o.label), tr, vals);
    hover(row, o.label + " per serving", () => [["Original", fmt(b, o.d) + " " + o.unit], ["With swap", fmt(a, o.d) + " " + o.unit], ["Change", r === 0 ? "0%" : signed(ch, 0) + "%"]]);
    R.append(row);
  }
  const ar = el("div", "aroma-row");
  const lft = el("span"); lft.append(el("b", null, "Aroma compounds kept"));
  ar.append(lft, el("span", "num", fmt(x.aroma, 0) + "% of " + x.compounds));
  R.append(ar);
}

// ---------- cuisine chart
const metricEl = document.getElementById("metric");
for (const k of ["energy_kcal", "fat_g", "sat_fat_g", "sugar_g", "sodium_mg", "aroma"]) {
  const b = el("button", "chip", label(k)); b.type = "button"; b.dataset.k = k;
  b.onclick = () => { metric = k; renderCuisines(); };
  metricEl.append(b);
}
const val = (r, k) => k === "aroma" ? {v: r.aroma, lo: r.aroma_lo, hi: r.aroma_hi} : {v: r.m[k], lo: r.lo[k], hi: r.hi[k]};
function renderCuisines() {
  [...metricEl.children].forEach(b => b.setAttribute("aria-pressed", String(b.dataset.k === metric)));
  const isAroma = metric === "aroma";
  document.getElementById("cu-sub").textContent = (isAroma
    ? "Share of each recipe's aroma compounds still present after the swap."
    : "Mean change in " + label(metric).toLowerCase() + " per serving.") + " Dots are means, lines are 95% confidence intervals; the vertical line is the average over all recipes.";
  const rows = cuisineRows().slice().sort((p, q) => isAroma ? val(q, metric).v - val(p, metric).v : val(p, metric).v - val(q, metric).v);
  const a = val(allRow(), metric);
  const best = rows[0], worst = rows[rows.length - 1], bv = val(best, metric), wv = val(worst, metric);
  const take = document.getElementById("cu-take");
  const unit = v => isAroma ? fmt(v, 1) + "%" : pct(v);
  take.replaceChildren();
  if (isAroma) {
    take.append("Most flavor kept: ", el("b", null, best.c + " (" + unit(bv.v) + ")"), ". Least: ", el("b", null, worst.c + " (" + unit(wv.v) + ")"),
      ". Average " + unit(a.v) + ".");
  } else {
    take.append(bv.v < 0 ? "Biggest cut: " : "Smallest increase: ", el("b", null, best.c + " (" + unit(bv.v) + ")"), ". ",
      wv.v > 0 ? "Largest increase: " : "Smallest cut: ", el("b", null, worst.c + " (" + unit(wv.v) + ")"),
      (wv.hi - wv.lo > 20 ? ", very uncertain (" + fmt(worst.n) + " recipes)" : "") + ". Average " + unit(a.v) + ".");
  }
  const box = document.getElementById("cu-chart");
  const W = Math.max(box.clientWidth, 300), narrow = W < 560;
  const padL = narrow ? 118 : 170, padR = narrow ? 58 : 76, top = 30, rowH = narrow ? 30 : 32, bottom = 34;
  const H = top + rows.length * rowH + bottom;
  let lo = Math.min(...rows.map(r => val(r, metric).lo)), hi = Math.max(...rows.map(r => val(r, metric).hi));
  const ref = isAroma ? 100 : 0;
  lo = Math.min(lo, ref, a.v); hi = Math.max(hi, ref, a.v);
  const pad = (hi - lo) * 0.04; lo -= pad; hi += pad;
  const x = v => padL + (v - lo) / (hi - lo) * (W - padL - padR);
  const svg = svgEl("svg", {width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img",
    "aria-label": label(metric) + " by cuisine, mean and 95% confidence interval"});
  for (const t of niceTicks(lo, hi, narrow ? 4 : 6)) {
    svg.append(svgEl("line", {x1: x(t), x2: x(t), y1: top - 6, y2: H - bottom, style: "stroke:var(--grid);stroke-width:1"}));
    const tx = svgEl("text", {x: x(t), y: H - bottom + 20, "text-anchor": "middle", style: "fill:var(--muted);font-size:12px"});
    tx.textContent = isAroma ? fmt(t, 0) + "%" : signed(t, Math.abs(t) < 10 && t % 1 ? 1 : 0) + "%"; svg.append(tx);
  }
  svg.append(svgEl("line", {x1: x(ref), x2: x(ref), y1: top - 6, y2: H - bottom, style: "stroke:var(--axis);stroke-width:1.5"}));
  svg.append(svgEl("line", {x1: x(a.v), x2: x(a.v), y1: top - 6, y2: H - bottom, style: "stroke:var(--ink-2);stroke-width:1.5"}));
  const at = svgEl("text", {x: x(a.v), y: top - 12, "text-anchor": "middle", style: "fill:var(--ink-2);font-size:12px;font-weight:600"});
  at.textContent = "All recipes " + unit(a.v); svg.append(at);
  rows.forEach((r, i) => {
    const y = top + i * rowH + rowH / 2, v = val(r, metric);
    const g = svgEl("g", {});
    const hit = svgEl("rect", {x: 0, y: y - rowH / 2, width: W, height: rowH, rx: 6, class: "hit"});
    g.append(hit);
    const t = svgEl("text", {x: padL - 12, y: y + 4.5, "text-anchor": "end", style: "fill:var(--ink);font-size:" + (narrow ? 13 : 14) + "px"});
    t.textContent = r.c; g.append(t);
    g.append(svgEl("line", {x1: x(v.lo), x2: x(v.hi), y1: y, y2: y, style: "stroke:var(--s1);stroke-width:2;stroke-linecap:round"}));
    g.append(svgEl("circle", {cx: x(v.v), cy: y, r: 5.5, style: "fill:var(--s1);stroke:var(--surface);stroke-width:2"}));
    if (i === 0 || i === rows.length - 1) {
      const lt = svgEl("text", {x: x(v.hi) + 8, y: y + 4.5, style: "fill:var(--ink);font-size:13px;font-weight:600"});
      lt.textContent = unit(v.v); g.append(lt);
    }
    hover(hit, r.c, () => [[label(metric), unit(v.v)], ["95% CI", (isAroma ? fmt(v.lo, 1) + "%" : pct(v.lo)) + " to " + (isAroma ? fmt(v.hi, 1) + "%" : pct(v.hi))], ["Recipes", fmt(r.n)]]);
    svg.append(g);
  });
  box.replaceChildren(svg);
}

// ---------- heat table
function cellColor(v) {
  const t = Math.min(Math.abs(v) / 30, 1), mix = Math.round(t * 85);
  return {bg: `color-mix(in oklab, var(${v < 0 ? "--neg" : "--pos"}) ${mix}%, var(--mid))`, ink: mix >= 55 ? "#ffffff" : "var(--ink)"};
}
function renderHeat() {
  const tb = document.getElementById("heat");
  const head = el("tr");
  head.append(el("th", null, "Cuisine"), el("th", null, "Recipes"));
  for (const o of OUT) head.append(el("th", null, o.label));
  head.append(el("th", null, "Aroma kept"));
  const thead = el("thead"); thead.append(head);
  const body = el("tbody");
  const rows = [allRow(), ...cuisineRows().slice().sort((p, q) => q.n - p.n)];
  for (const r of rows) {
    const tr = el("tr", r.c === ALL ? "all" : "");
    tr.append(el("td", null, r.c), el("td", "num", fmt(r.n)));
    for (const o of OUT) {
      const v = r.m[o.k], c = cellColor(v), td = el("td", "cell num", pct(v));
      td.style.background = c.bg; td.style.color = c.ink;
      hover(td, r.c + " · " + o.label, () => [["Mean change", pct(v)], ["95% CI", pct(r.lo[o.k]) + " to " + pct(r.hi[o.k])],
        ["Per serving", signed(r.abs[o.k], o.d) + " " + o.unit], ["Recipes", fmt(r.n)]]);
      tr.append(td);
    }
    tr.append(el("td", "num", fmt(r.aroma, 1) + "%"));
    body.append(tr);
  }
  tb.replaceChildren(thead, body);
}

// ---------- swaps
function renderSwaps() {
  const list = S().swaps, maxN = list[0].n;
  document.getElementById("sw-sub").textContent = "Number of recipes in which each swap was applied (" +
    DATA.scenarios.find(s => s.key === scen).label.toLowerCase() + "), with its average effect per serving.";
  const box = document.getElementById("sw-list");
  box.replaceChildren();
  for (const s of list) {
    const row = el("div", "swaprow");
    const pair = el("div", "pair");
    pair.append(s.a, el("span", "ar", "→"), s.b);
    if (s.flag === "no") { const f = el("span", "flag"); f.append(el("span", "wd"), "questionable swap"); pair.append(f); }
    const bar = el("div", "hbar"), b = el("div", "b");
    b.style.width = (s.n / maxN * 78) + "%";
    bar.append(b, el("span", "v num", fmt(s.n)));
    const parts = [];
    if (Math.abs(s.kcal) >= 1) parts.push(signed(s.kcal, 0) + " kcal");
    if (Math.abs(s.sat) >= 0.1) parts.push(signed(s.sat, 1) + " g sat. fat");
    if (Math.abs(s.sodium) >= 5) parts.push(signed(s.sodium, 0) + " mg sodium");
    if (Math.abs(s.sugar) >= 0.5) parts.push(signed(s.sugar, 1) + " g sugars");
    row.append(pair, bar, el("div", "eff num", parts.length ? parts.join(" · ") : "small changes"));
    hover(row, s.a + " → " + s.b, () => [["Recipes", fmt(s.n)], ["Calories", signed(s.kcal, 0) + " kcal"], ["Saturated fat", signed(s.sat, 1) + " g"],
      ["Sodium", signed(s.sodium, 0) + " mg"], ["Sugars", signed(s.sugar, 1) + " g"]]);
    box.append(row);
  }
  const note = el("p", "sub");
  note.style.margin = "12px 0 0"; note.style.fontSize = "14px";
  note.textContent = "Marked swaps were judged questionable in our hand audit of the model's " + M.audit_pairs +
    " most common swaps (" + M.audit_ok + "% of audited recipes got a sensible substitute).";
  box.append(note);
}

// ---------- reality check (fixed: model best single swap vs reviewer-reported swap)
function renderReality() {
  const A = DATA.scen["best single swap"].rows.find(r => r.c === ALL), B = DATA.scen["reviewer-reported swap"].rows.find(r => r.c === ALL);
  const items = OUT.map(o => ({name: o.label, a: A.m[o.k], b: B.m[o.k], u: "%"}));
  items.push({name: "Aroma kept", a: A.aroma - 100, b: B.aroma - 100, u: " pts"});
  const box = document.getElementById("re-chart");
  const W = Math.max(box.clientWidth, 300), narrow = W < 520;
  // narrow screens: category names sit above their bars, so the plot can use the full width
  const padL = narrow ? 8 : 130, padR = narrow ? 8 : 20, barH = 15, gap = 2, rowH = narrow ? 66 : 52, top = 6, bottom = 30;
  const H = top + items.length * rowH + bottom;
  let lo = Math.min(0, ...items.map(i => Math.min(i.a, i.b))), hi = Math.max(0, ...items.map(i => Math.max(i.a, i.b)));
  const span = hi - lo; lo -= span * (narrow ? 0.24 : 0.17); hi += span * (narrow ? 0.2 : 0.15);   // room for the value labels
  const x = v => padL + (v - lo) / (hi - lo) * (W - padL - padR);
  const svg = svgEl("svg", {width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: "img", "aria-label": "FlavorSwap vs reviewers' swaps, mean change per serving"});
  for (const t of niceTicks(lo, hi, narrow ? 4 : 6)) {
    svg.append(svgEl("line", {x1: x(t), x2: x(t), y1: top, y2: H - bottom, style: "stroke:var(--grid);stroke-width:1"}));
    const tx = svgEl("text", {x: x(t), y: H - bottom + 20, "text-anchor": "middle", style: "fill:var(--muted);font-size:12px"});
    tx.textContent = signed(t, 0); svg.append(tx);
  }
  svg.append(svgEl("line", {x1: x(0), x2: x(0), y1: top, y2: H - bottom, style: "stroke:var(--axis);stroke-width:1.5"}));
  items.forEach((it, i) => {
    const y0 = top + i * rowH + (rowH - 2 * barH - gap) / 2 + (narrow ? 8 : 0);
    const name = narrow
      ? svgEl("text", {x: padL, y: y0 - 7, "text-anchor": "start", style: "fill:var(--ink);font-size:13px;font-weight:600"})
      : svgEl("text", {x: padL - 12, y: y0 + barH + 5, "text-anchor": "end", style: "fill:var(--ink);font-size:14px"});
    name.textContent = it.name; svg.append(name);
    [[it.a, "--s1", "FlavorSwap"], [it.b, "--s2", "Reviewers"]].forEach(([v, c, who], j) => {
      const y = y0 + j * (barH + gap), x0 = x(Math.min(0, v)), w = Math.max(Math.abs(x(v) - x(0)), 1);
      const neg = v < 0, rr = Math.min(4, w / 2);
      // bar with a 4px rounded data end, square at the zero baseline
      const px = x0, d = neg
        ? `M${px + w},${y} H${px + rr} Q${px},${y} ${px},${y + rr} V${y + barH - rr} Q${px},${y + barH} ${px + rr},${y + barH} H${px + w} Z`
        : `M${px},${y} H${px + w - rr} Q${px + w},${y} ${px + w},${y + rr} V${y + barH - rr} Q${px + w},${y + barH} ${px + w - rr},${y + barH} H${px} Z`;
      const bar = svgEl("path", {d, style: `fill:var(${c})`});
      svg.append(bar);
      const lt = svgEl("text", {x: neg ? x(v) - 6 : x(v) + 6, y: y + barH - 1.5, "text-anchor": neg ? "end" : "start",
        style: "fill:var(--ink-2);font-size:12px"});
      lt.textContent = signed(v) + it.u; svg.append(lt);
      const hit = svgEl("rect", {x: padL, y: y - 1, width: W - padL - padR, height: barH + 2, class: "hit"});
      hover(hit, it.name + " · " + who, () => [[it.u === "%" ? "Mean change" : "Change in aroma kept", signed(v) + it.u]]);
      svg.append(hit);
    });
  });
  box.replaceChildren(svg);
  const pts = document.getElementById("re-points");
  const sa = A.m.sat_fat_g, sb = B.m.sat_fat_g, la = 100 - A.aroma, lb = 100 - B.aroma, fa = A.m.fat_g, fb = B.m.fat_g;
  const li = (...parts) => { const l = el("li"); l.append(...parts); return l; };
  pts.replaceChildren(
    li("Reviewers' own swaps cut saturated fat ", el("b", null, Math.abs(sb) > Math.abs(sa) ? "harder" : "less"), " (" + pct(sb) + " vs " + pct(sa) + ")."),
    li("But they lose ", el("b", null, (lb > la ? "more" : "less") + " flavor"), ": " + fmt(lb, 1) + "% of aroma compounds vs " + fmt(la, 1) + "% for FlavorSwap."),
    li("Total fat: " + pct(fb) + " vs " + pct(fa) + (fb > 0 ? " — reviewers often replace butter with cooking spray or oil, which is almost pure fat at the same weight." : ".")),
  );
  if (la < lb && Math.abs(sa) <= Math.abs(sb))
    pts.append(li("FlavorSwap's swaps are ", el("b", null, "no more aggressive"), " than what home cooks already do, and keep more of the recipe's flavor."));
}

// ---------- render
function renderScenario() {
  [...segEl.children].forEach(b => b.setAttribute("aria-pressed", String(b.dataset.key === scen)));
  document.getElementById("scen-note").textContent = DATA.scenarios.find(s => s.key === scen).note;
  renderHeadline(); renderCuisines(); renderHeat(); renderSwaps();
}
const themeBtn = document.getElementById("theme");
function isDark() { const t = document.documentElement.dataset.theme; return t ? t === "dark" : matchMedia("(prefers-color-scheme: dark)").matches; }
function syncThemeBtn() { themeBtn.textContent = isDark() ? "Light mode" : "Dark mode"; }
themeBtn.onclick = () => {
  const t = isDark() ? "light" : "dark";
  document.documentElement.dataset.theme = t; syncThemeBtn();
  try { localStorage.setItem("flavorswap-theme", t); } catch (e) { /* storage unavailable: choice lasts this visit */ }
};
// theme: ?theme=light|dark in the URL, else the viewer's last choice, else the OS setting
const qt = new URLSearchParams(location.search).get("theme");
let savedTheme = null;
try { savedTheme = localStorage.getItem("flavorswap-theme"); } catch (e) { savedTheme = null; }
const startTheme = qt === "light" || qt === "dark" ? qt : savedTheme === "light" || savedTheme === "dark" ? savedTheme : null;
if (startTheme) document.documentElement.dataset.theme = startTheme;
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", syncThemeBtn);
syncThemeBtn();
renderScenario(); renderExample(); renderReality();
let rt; addEventListener("resize", () => { clearTimeout(rt); rt = setTimeout(() => { renderCuisines(); renderReality(); }, 120); });
</script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
