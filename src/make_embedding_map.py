"""Deliverable 2 (part 2) - interactive t-SNE / UMAP map of ingredient similarity.

Projects the Word2Vec ingredient vectors to 2-D with both UMAP and t-SNE (cosine metric) and writes
a single self-contained HTML page (plotly.js inlined, works offline):

  * switch between the UMAP and t-SNE projections;
  * highlight one food group (everything else stays muted);
  * search or click an ingredient to highlight it and its 10 nearest neighbours in the full
    100-d embedding space (lines + labels), with a side table of cosine similarity, food group,
    recipe count and nutrition per 100 g.

Colour follows the data-viz method used for this project: a scatter compares all pairs of
colours, so at most three categorical slots are used (group / selected / neighbours).

Outputs:
    outputs/ingredient_map.html
    data/processed/embedding_coords.csv   (ingredient, umap_x, umap_y, tsne_x, tsne_y)

Run:  python src/make_embedding_map.py
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import umap
from gensim.models import KeyedVectors
from plotly.offline import get_plotlyjs
from sklearn.manifold import TSNE

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flavorswap import paths  # noqa: E402
from flavorswap.food_groups import GROUPS  # noqa: E402

SEED = 42
N_NEIGHBORS_SHOWN = 10


def project(vectors: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    print("  UMAP ...")
    um = umap.UMAP(n_neighbors=15, min_dist=0.1, metric="cosine", random_state=SEED).fit_transform(vectors)
    print("  t-SNE ...")
    ts = TSNE(n_components=2, perplexity=30, metric="cosine", init="pca", random_state=SEED,
              max_iter=1500).fit_transform(vectors)
    return um, ts


def num(x, nd=1):
    return None if x is None or (isinstance(x, float) and math.isnan(x)) else round(float(x), nd)


def main() -> None:
    paths.OUTPUTS.mkdir(exist_ok=True)
    kv = KeyedVectors.load(str(paths.MODELS / "ingredient_vectors.kv"))
    ing = pd.read_csv(paths.PROCESSED / "ingredients.csv").set_index("ingredient")
    words = [w for w in kv.index_to_key if w in ing.index]
    missing = set(kv.index_to_key) - set(words)
    assert not missing, f"embedding words missing from ingredients.csv: {sorted(missing)[:10]}"

    vecs = kv.get_normed_vectors()[[kv.key_to_index[w] for w in words]]
    print(f"Projecting {len(words)} ingredient vectors ...")
    um, ts = project(vecs)
    coords = pd.DataFrame({"ingredient": words, "umap_x": um[:, 0], "umap_y": um[:, 1],
                           "tsne_x": ts[:, 0], "tsne_y": ts[:, 1]})
    coords.to_csv(paths.PROCESSED / "embedding_coords.csv", index=False)

    # nearest neighbours in the full embedding space (not in 2-D)
    sims = vecs @ vecs.T
    np.fill_diagonal(sims, -np.inf)
    top = np.argsort(-sims, axis=1)[:, :N_NEIGHBORS_SHOWN]

    pts = []
    for i, w in enumerate(words):
        r = ing.loc[w]
        pts.append({
            "n": w,
            "g": GROUPS.index(r.food_group),
            "c": int(r.n_recipes),
            "u": [round(float(um[i, 0]), 3), round(float(um[i, 1]), 3)],
            "t": [round(float(ts[i, 0]), 3), round(float(ts[i, 1]), 3)],
            "k": [num(r.energy_kcal, 0), num(r.fat_g), num(r.sugar_g), num(r.sodium_mg, 0),
                  num(r.protein_g)],
            "f": int(r.ahn_n_compounds),
            "nb": [[int(j), round(float(sims[i, j]), 3)] for j in top[i]],
        })
    payload = {"groups": GROUPS, "points": pts}

    html = TEMPLATE.replace("/*PLOTLY_JS*/", get_plotlyjs()).replace(
        "/*DATA_JSON*/", json.dumps(payload, separators=(",", ":")).replace("</", "<\\/"))
    out = paths.OUTPUTS / "ingredient_map.html"
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out.relative_to(paths.ROOT)} ({out.stat().st_size / 1e6:.1f} MB)")


TEMPLATE = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Ingredient Similarity Map</title>
<style>
:root {
  color-scheme: light;
  --page: #f9f9f7; --surface-1: #fcfcfb; --text-primary: #0b0b0b; --text-secondary: #52514e;
  --text-muted: #898781; --grid: #e1e0d9; --axis: #c3c2b7; --border: rgba(11,11,11,0.10);
  --muted-point: #c3c2b7;
  --series-1: #2a78d6; --series-2: #eb6834; --series-3: #1baf7a;
}
@media (prefers-color-scheme: dark) {
  :root:where(:not([data-theme="light"])) {
    color-scheme: dark;
    --page: #0d0d0d; --surface-1: #1a1a19; --text-primary: #f0efec; --text-secondary: #c3c2b7;
    --text-muted: #898781; --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
    --muted-point: #4a4a46;
    --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70;
  }
}
:root[data-theme="dark"] {
  color-scheme: dark;
  --page: #0d0d0d; --surface-1: #1a1a19; --text-primary: #f0efec; --text-secondary: #c3c2b7;
  --text-muted: #898781; --grid: #2c2c2a; --axis: #383835; --border: rgba(255,255,255,0.10);
  --muted-point: #4a4a46;
  --series-1: #3987e5; --series-2: #d95926; --series-3: #199e70;
}
* { box-sizing: border-box; }
body { margin: 0; background: var(--page); color: var(--text-primary);
  font: 14px/1.45 system-ui, -apple-system, "Segoe UI", sans-serif; }
header { padding: 20px 24px 8px; max-width: 1400px; margin: 0 auto; }
h1 { font-size: 20px; margin: 0 0 4px; font-weight: 600; }
header p { margin: 0; color: var(--text-secondary); max-width: 900px; }
.controls { display: flex; flex-wrap: wrap; gap: 12px 20px; align-items: end;
  padding: 12px 24px; max-width: 1400px; margin: 0 auto; }
.control { display: flex; flex-direction: column; gap: 4px; }
.control label, .control .lbl { font-size: 12px; color: var(--text-secondary); }
select, input[type=search] { font: inherit; color: var(--text-primary); background: var(--surface-1);
  border: 1px solid var(--border); border-radius: 6px; padding: 6px 8px; min-width: 220px; }
.seg { display: inline-flex; border: 1px solid var(--border); border-radius: 6px; overflow: hidden; }
.seg button { font: inherit; border: 0; background: var(--surface-1); color: var(--text-secondary);
  padding: 6px 14px; cursor: pointer; }
.seg button[aria-pressed=true] { background: var(--text-primary); color: var(--surface-1); }
button.clear { font: inherit; background: none; border: 1px solid var(--border); border-radius: 6px;
  color: var(--text-secondary); padding: 6px 10px; cursor: pointer; }
main > * { min-width: 0; }
main { display: flex; gap: 16px; padding: 0 24px 16px; max-width: 1400px; margin: 0 auto; }
.card { background: var(--surface-1); border: 1px solid var(--border); border-radius: 10px; }
#plot { flex: 1; min-width: 0; height: 760px; }
aside { width: 340px; flex: none; padding: 16px; height: 760px; overflow: auto; }
aside h2 { font-size: 16px; margin: 0 0 2px; font-weight: 600; }
aside .sub { color: var(--text-secondary); margin: 0 0 12px; }
.legend { display: flex; flex-direction: column; gap: 6px; margin-bottom: 16px; }
.legend span { display: inline-flex; align-items: center; gap: 8px; color: var(--text-secondary); }
.dot { width: 10px; height: 10px; border-radius: 50%; flex: none; }
table { width: 100%; border-collapse: collapse; font-size: 13px; }
th { text-align: left; font-weight: 600; color: var(--text-secondary); border-bottom: 1px solid var(--grid);
  padding: 4px 4px; }
td { padding: 4px 4px; border-bottom: 1px solid var(--grid); }
td.num { text-align: right; font-variant-numeric: tabular-nums; }
tbody tr { cursor: pointer; }
tbody tr:hover { background: var(--page); }
.kv { display: grid; grid-template-columns: auto 1fr; gap: 2px 12px; margin: 0 0 14px; }
.kv dt { color: var(--text-secondary); }
.kv dd { margin: 0; font-variant-numeric: tabular-nums; }
footer { max-width: 1400px; margin: 0 auto; padding: 0 24px 24px; color: var(--text-muted); font-size: 12px; }
@media (max-width: 900px) {
  main { flex-direction: column; padding: 0 16px 16px; }
  header, .controls, footer { padding-left: 16px; padding-right: 16px; }
  #plot { flex: none; width: 100%; height: 520px; }
  aside { width: 100%; height: auto; }
  select, input[type=search] { min-width: 0; width: 100%; }
  .control { flex: 1 1 100%; }
}
</style>
</head>
<body>
<header>
  <h1>Ingredient Similarity Map</h1>
  <p>Each point is one of the most common Food.com ingredients, placed by its Word2Vec embedding
  (learned from which ingredients appear together in 231k recipes). Ingredients used in the same kinds of
  recipes sit close together. Point size grows with how many recipes use the ingredient.</p>
</header>
<div class="controls">
  <div class="control"><span class="lbl">Projection</span>
    <div class="seg" role="group" aria-label="Projection">
      <button id="btn-u" aria-pressed="true">UMAP</button><button id="btn-t" aria-pressed="false">t-SNE</button>
    </div></div>
  <div class="control"><label for="grp">Highlight food group</label><select id="grp"></select></div>
  <div class="control"><label for="q">Find an ingredient</label>
    <input id="q" type="search" list="names" placeholder="e.g. butter, cilantro, tofu" autocomplete="off">
    <datalist id="names"></datalist></div>
  <div class="control"><span class="lbl">&nbsp;</span><button class="clear" id="clr">Clear selection</button></div>
</div>
<main>
  <div id="plot" class="card" role="img" aria-label="Scatter map of ingredient embeddings"></div>
  <aside class="card" id="panel"></aside>
</main>
<footer>Neighbours and similarities are computed in the full 100-dimensional embedding space (cosine); the 2-D
layout is only an approximation of it. Nutrition is per 100 g from USDA FoodData Central SR Legacy (blank when no
reliable link exists). Flavor compounds: Ahn et al. (2011) flavor network.</footer>
<script>/*PLOTLY_JS*/</script>
<script>
const THEME = new URLSearchParams(location.search).get("theme");
if (THEME === "light" || THEME === "dark") document.documentElement.dataset.theme = THEME;
const DATA = /*DATA_JSON*/;
const P = DATA.points, G = DATA.groups;
const state = { proj: "u", group: -1, sel: -1 };
const css = n => getComputedStyle(document.documentElement).getPropertyValue(n).trim();
const fmt = (v, u) => v === null ? "–" : v.toLocaleString() + (u || "");
const size = c => 8 + 2.5 * Math.log10(Math.max(c, 20) / 20);

function hoverText(p) {
  const k = p.k;
  return `<b>${p.n}</b><br>${G[p.g]} · ${p.c.toLocaleString()} recipes` +
    (k[0] !== null ? `<br>${k[0]} kcal · fat ${fmt(k[1], " g")} · sugar ${fmt(k[2], " g")} · sodium ${fmt(k[3], " mg")} /100 g` : "") +
    (p.f ? `<br>${p.f} flavor compounds` : "") +
    `<br>nearest: ${p.nb.slice(0, 3).map(([j]) => P[j].n).join(", ")}`;
}
const HOVER = P.map(hoverText);
const xy = (idx) => ({ x: idx.map(i => P[i][state.proj][0]), y: idx.map(i => P[i][state.proj][1]) });
const ALL = P.map((_, i) => i);

function traces() {
  const surf = css("--surface-1");
  const ring = { color: surf, width: 1 };
  const t = [];
  // 0: every ingredient, muted
  t.push({ type: "scatter", mode: "markers", ...xy(ALL), text: HOVER, hoverinfo: "text",
    marker: { size: P.map(p => size(p.c)), color: css("--muted-point"), opacity: 0.75, line: ring },
    name: "All ingredients", showlegend: false });
  // 1: highlighted group
  const gi = state.group < 0 ? [] : ALL.filter(i => P[i].g === state.group);
  t.push({ type: "scatter", mode: "markers", ...xy(gi), text: gi.map(i => HOVER[i]), hoverinfo: "text",
    marker: { size: gi.map(i => size(P[i].c)), color: css("--series-1"), opacity: 0.9, line: ring },
    showlegend: false });
  // 2: lines selected -> neighbours, 3: neighbours, 4: selected
  const s = state.sel;
  const nb = s < 0 ? [] : P[s].nb.map(([j]) => j);
  const lx = [], ly = [];
  if (s >= 0) for (const j of nb) {
    lx.push(P[s][state.proj][0], P[j][state.proj][0], null);
    ly.push(P[s][state.proj][1], P[j][state.proj][1], null);
  }
  t.push({ type: "scatter", mode: "lines", x: lx, y: ly, hoverinfo: "skip",
    line: { color: css("--series-3"), width: 1 }, opacity: 0.6, showlegend: false });
  t.push({ type: "scatter", mode: "markers", ...xy(nb),
    hovertext: nb.map(j => HOVER[j]), hoverinfo: "text",
    marker: { size: 10, color: css("--series-3"), line: { color: surf, width: 2 } }, showlegend: false });
  t.push({ type: "scatter", mode: "markers+text", ...xy(s < 0 ? [] : [s]), text: s < 0 ? [] : [P[s].n],
    hovertext: s < 0 ? [] : [HOVER[s]], hoverinfo: "text", textposition: "bottom center",
    textfont: { color: css("--text-primary"), size: 13 },
    marker: { size: 14, color: css("--series-2"), line: { color: surf, width: 2 } }, showlegend: false });
  return t;
}

function layout() {
  return {
    paper_bgcolor: css("--surface-1"), plot_bgcolor: css("--surface-1"),
    margin: { l: 16, r: 16, t: 16, b: 16 }, hovermode: "closest", hoverdistance: 24,
    dragmode: "pan", uirevision: state.proj,
    xaxis: { visible: false }, yaxis: { visible: false, scaleanchor: "x" },
    font: { family: 'system-ui, -apple-system, "Segoe UI", sans-serif', color: css("--text-primary") },
    hoverlabel: { bgcolor: css("--surface-1"), bordercolor: css("--axis"),
      font: { color: css("--text-primary"), size: 12 } },
  };
}

function draw() {
  Plotly.react("plot", traces(), layout(), { responsive: true, displaylogo: false, scrollZoom: true,
    modeBarButtonsToRemove: ["select2d", "lasso2d", "toggleSpikelines"] });
  panel();
}

function panel() {
  const el = document.getElementById("panel");
  const legend = `<div class="legend">
    <span><i class="dot" style="background:var(--muted-point)"></i>All ingredients</span>
    ${state.group >= 0 ? `<span><i class="dot" style="background:var(--series-1)"></i>${G[state.group]}</span>` : ""}
    ${state.sel >= 0 ? `<span><i class="dot" style="background:var(--series-2)"></i>Selected ingredient</span>
      <span><i class="dot" style="background:var(--series-3)"></i>Its ${P[state.sel].nb.length} nearest neighbours</span>` : ""}
  </div>`;
  if (state.sel < 0) {
    const counts = G.map((g, gi) => [g, P.filter(p => p.g === gi).length]).filter(r => r[1]);
    el.innerHTML = `<h2>${P.length.toLocaleString()} ingredients</h2>
      <p class="sub">Search or click a point to see its nearest neighbours.</p>${legend}
      <table><thead><tr><th>Food group</th><th class="num">Ingredients</th></tr></thead><tbody>
      ${counts.map(([g, n]) => `<tr data-g="${G.indexOf(g)}"><td>${g}</td><td class="num">${n}</td></tr>`).join("")}
      </tbody></table>`;
    el.querySelectorAll("tr[data-g]").forEach(tr => tr.onclick = () => {
      state.group = +tr.dataset.g; document.getElementById("grp").value = state.group; draw(); });
    return;
  }
  const p = P[state.sel], k = p.k;
  el.innerHTML = `<h2>${p.n}</h2><p class="sub">${G[p.g]} · used in ${p.c.toLocaleString()} recipes</p>${legend}
    <dl class="kv">
      <dt>Energy</dt><dd>${fmt(k[0], " kcal")}</dd><dt>Fat</dt><dd>${fmt(k[1], " g")}</dd>
      <dt>Sugar</dt><dd>${fmt(k[2], " g")}</dd><dt>Sodium</dt><dd>${fmt(k[3], " mg")}</dd>
      <dt>Protein</dt><dd>${fmt(k[4], " g")}</dd><dt>Flavor compounds</dt><dd>${p.f || "–"}</dd>
    </dl>
    <table><thead><tr><th>Nearest neighbour</th><th class="num">Cosine</th><th class="num">kcal/100 g</th></tr></thead>
    <tbody>${p.nb.map(([j, s]) => `<tr data-i="${j}"><td>${P[j].n}<br><small style="color:var(--text-muted)">${G[P[j].g]}</small></td>
      <td class="num">${s.toFixed(2)}</td><td class="num">${fmt(P[j].k[0])}</td></tr>`).join("")}</tbody></table>`;
  el.querySelectorAll("tr[data-i]").forEach(tr => tr.onclick = () => select(+tr.dataset.i));
}

const byName = new Map(P.map((p, i) => [p.n, i]));
function select(i) { state.sel = i; document.getElementById("q").value = i < 0 ? "" : P[i].n; draw(); }

// controls
const grp = document.getElementById("grp");
grp.innerHTML = `<option value="-1">None</option>` + G.map((g, i) => `<option value="${i}">${g}</option>`).join("");
grp.onchange = () => { state.group = +grp.value; draw(); };
document.getElementById("names").innerHTML =
  [...P].sort((a, b) => b.c - a.c).map(p => `<option value="${p.n}">`).join("");
const q = document.getElementById("q");
q.addEventListener("change", () => { const i = byName.get(q.value.trim().toLowerCase()); if (i !== undefined) select(i); });
q.addEventListener("keydown", e => { if (e.key === "Enter") { const i = byName.get(q.value.trim().toLowerCase()); if (i !== undefined) select(i); } });
document.getElementById("clr").onclick = () => select(-1);
const setProj = pr => { state.proj = pr;
  document.getElementById("btn-u").setAttribute("aria-pressed", pr === "u");
  document.getElementById("btn-t").setAttribute("aria-pressed", pr === "t"); draw(); };
document.getElementById("btn-u").onclick = () => setProj("u");
document.getElementById("btn-t").onclick = () => setProj("t");

draw();
document.getElementById("plot").on("plotly_click", ev => {
  const pt = ev.points[0];
  if (pt.curveNumber === 0) select(pt.pointIndex);
  else if (pt.curveNumber === 1) select(ALL.filter(i => P[i].g === state.group)[pt.pointIndex]);
  else if (pt.curveNumber === 3) select(P[state.sel].nb[pt.pointIndex][0]);
});
matchMedia("(prefers-color-scheme: dark)").addEventListener("change", draw);
new MutationObserver(draw).observe(document.documentElement, { attributes: true, attributeFilter: ["data-theme"] });
const params = new URLSearchParams(location.search);
const g0 = G.indexOf(params.get("group"));
if (g0 >= 0) { state.group = g0; grp.value = g0; draw(); }
const initial = params.get("q");
if (initial && byName.has(initial)) select(byName.get(initial));
</script>
</body>
</html>
"""

if __name__ == "__main__":
    main()
