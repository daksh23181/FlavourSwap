"""Hybrid ingredient-substitution recommender (Deliverable 3, reused by the Deliverable 4 app).

For a source ingredient s and every candidate c in the vocabulary, five signals are computed:

  emb   cosine similarity of Word2Vec ingredient vectors (recipe-context interchangeability)
  flav  flavor-compound overlap from Ahn et al. (2011): Ochiai cosine on compound sets, plain or
        IDF-weighted (compounds shared by many ingredients count less)
  nutr  similarity of nutrient profiles per 100 g (USDA SR Legacy): negative RMS distance of
        log-scaled, standardised nutrient values
  pop   log number of recipes using c (a prior towards ingredients people actually have)
  grp   food-group transition prior log P(group(c) given group(s)), estimated from review-reported
        substitutions of non-test source ingredients (models/group_transitions.csv); people swap
        dairy -> fats often, grains -> vegetables rarely

Each signal is standardised across the candidates of the query (z-score); a missing value
(no flavor profile / no nutrient link) is set to 0, i.e. neutral. The hybrid score is a weighted
sum; weights are tuned on validation substitution pairs (src/evaluate_substitutions.py) and stored
in models/hybrid_config.json.

"Healthier" mode makes no suggestions for whole fruit, vegetables, legumes and nuts (already the
foods guidelines recommend; see flavorswap.nutrition.whole_plant_food) or for ingredients with almost
nothing to limit (less than MIN_ABS_IMPROVEMENT of energy, saturated fat, free sugars, sodium and trans
fat per 100 g: vinegar, lemon juice, wine, water, ice - cutting these nutrients cannot reach the
required improvement, so a "healthier" swap would only add fiber, e.g. ice -> guava). Otherwise it
only considers candidates that are
  * among the `health_pool` most similar candidates by content (hybrid score without the popularity
    prior), so a healthier option is still a sensible swap,
  * from a food group that receives at least `min_group_prob` of the source group's reported swaps
    (no grains -> herbs & spices "swaps" such as rice -> garlic),
  * not a herb/spice unless the source is one (seasonings are used by the gram, so a per-100 g
    health comparison with a main ingredient is meaningless; salt -> herbs is still allowed),
  * comparable per 100 g: water content within MAX_WATER_DIFF percentage points (no dry-vs-cooked
    comparisons such as raw rice vs cooked rice),
  * healthier by at least max(min_reduction x |score|, MIN_ABS_IMPROVEMENT) in health score
    (flavorswap.nutrition: %DV of energy, saturated fat, sugars, sodium and trans fat minus fiber),
and re-ranks them by hybrid score + lambda x relative improvement. Comparisons are per 100 g (equal-
mass swap); an ingredient without a reliable USDA link is never called healthier.
"""

from __future__ import annotations

import difflib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from scipy import sparse

from . import paths
from .normalize import canonicalize
from .nutrition import health_scores, limit_scores, whole_plant_food

NUTR_FEATURES = ["energy_kcal", "protein_g", "fat_g", "carbs_g", "sugar_g", "fiber_g", "sodium_mg"]
MIN_SHARED_NUTR_FEATURES = 5
MAX_WATER_DIFF = 50.0      # percentage points; beyond this a per-100 g comparison is meaningless
MIN_ABS_IMPROVEMENT = 0.10  # health-score units (%DV): ~200 kcal, 230 mg sodium or 2.8 g fiber /100 g
PCT_DISPLAY_MIN = 0.5       # percentage changes are only meaningful for clearly positive scores
SEASONING_GROUP = "Herbs & Spices"
TABLE_COLUMNS = ["rank", "ingredient", "food_group", "n_recipes", "score", "embedding_cosine",
                 "flavor_similarity", "shared_compounds", "example_shared_compounds", "nutrient_similarity",
                 "health_score", "health_delta", "health_change_pct", "delta_energy_kcal", "delta_fat_g",
                 "delta_sat_fat_g", "delta_sugar_g", "delta_sodium_mg"]


class UnknownIngredient(KeyError):
    """Raised when user text cannot be resolved to a vocabulary ingredient; carries suggestions."""

    def __init__(self, text: str, suggestions: list[str]):
        super().__init__(text)
        self.text = text
        self.suggestions = suggestions

    def __str__(self) -> str:
        hint = f" Did you mean: {', '.join(self.suggestions)}?" if self.suggestions else ""
        return f"Unknown ingredient {self.text!r}.{hint}"


@dataclass
class HybridWeights:
    emb: float = 1.0
    flav: float = 0.0
    nutr: float = 0.0
    pop: float = 0.0
    grp: float = 0.0
    flavor_variant: str = "ochiai"     # "ochiai" | "idf"
    health_lambda: float = 0.0
    min_reduction: float = 0.10
    health_pool: int = 30              # healthier options come from this many most similar candidates
    min_group_prob: float = 0.02       # healthier options only from plausible food-group transitions

    @classmethod
    def load(cls, path: Path | None = None) -> "HybridWeights":
        """Tuned weights from models/hybrid_config.json (written by evaluate_substitutions.py)."""
        cfg = json.loads((path or paths.MODELS / "hybrid_config.json").read_text())
        return cls(**cfg["weights"])

    def as_dict(self) -> dict:
        return dict(self.__dict__)


@dataclass
class Recommender:
    names: list[str]
    index: dict[str, int]
    ing: pd.DataFrame
    emb: np.ndarray                       # (N, d) L2-normalised
    flav_bin: sparse.csr_matrix           # (N, C) binary compound membership
    flav_idf: sparse.csr_matrix           # (N, C) IDF-weighted, rows L2-normalised
    flav_has: np.ndarray                  # (N,) bool
    nutr: np.ndarray                      # (N, F) standardised log nutrients (NaN = missing)
    health: np.ndarray                    # (N,) %DV health score (NaN = unknown)
    limit: np.ndarray                     # (N,) nutrients-to-limit part of the health score (no fiber)
    water: np.ndarray                    # (N,) water g/100 g (NaN = unknown)
    whole_plant: np.ndarray               # (N,) bool: unprocessed fruit/vegetable/legume/nut
    log_pop: np.ndarray                   # (N,)
    group_idx: np.ndarray                 # (N,) food-group index
    group_logp: np.ndarray                # (G, G) log P(target group given source group); 0 if absent
    compound_names: np.ndarray            # (C,)
    compound_idf: np.ndarray              # (C,)
    weights: HybridWeights = field(default_factory=HybridWeights)

    # ------------------------------------------------------------------ construction
    @classmethod
    def load(cls, weights: HybridWeights | None = None) -> "Recommender":
        """Load all artefacts. Without explicit weights, the tuned ones are used if available."""
        if weights is None and (paths.MODELS / "hybrid_config.json").exists():
            weights = HybridWeights.load()
        ing = pd.read_csv(paths.PROCESSED / "ingredients.csv")
        kv = KeyedVectors.load(str(paths.MODELS / "ingredient_vectors.kv"))
        names = ing.ingredient.tolist()
        missing = [n for n in names if n not in kv.key_to_index]
        if missing:
            raise ValueError(f"{len(missing)} vocabulary ingredients have no embedding, e.g. {missing[:5]}")
        emb = kv.get_normed_vectors()[[kv.key_to_index[n] for n in names]].astype(np.float32)
        index = {n: i for i, n in enumerate(names)}

        # ---- flavor compounds: IDF over all Ahn et al. ingredients (the source data)
        raw_ic = pd.read_csv(paths.AHN_INGR_COMP, sep="\t")
        raw_ic.columns = ["ahn_id", "compound_id"]
        n_ahn = raw_ic.ahn_id.nunique()
        df_c = raw_ic.groupby("compound_id").ahn_id.nunique()
        comp = pd.read_csv(paths.PROCESSED / "compounds.csv")
        comp_ids = comp.compound_id.to_numpy()
        col = {c: j for j, c in enumerate(comp_ids)}
        idf = np.array([math.log(n_ahn / df_c.get(c, n_ahn)) + 1e-9 for c in comp_ids])
        ic = pd.read_csv(paths.PROCESSED / "ingredient_compounds.csv")
        ic = ic[ic.ingredient.isin(index)]
        rows = ic.ingredient.map(index).to_numpy()
        cols = ic.compound_id.map(col).to_numpy()
        fb = sparse.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)),
                               shape=(len(names), len(comp_ids)))
        fi = fb.multiply(idf[None, :]).tocsr()
        norms = np.sqrt(np.asarray(fi.multiply(fi).sum(axis=1)).ravel())
        norms[norms == 0] = 1.0
        fi = sparse.diags(1.0 / norms) @ fi
        flav_has = np.asarray(fb.sum(axis=1)).ravel() > 0

        # ---- nutrients: log1p then standardise over ingredients that have the value
        X = np.log1p(ing[NUTR_FEATURES].astype(float).clip(lower=0).to_numpy())
        mu, sd = np.nanmean(X, axis=0), np.nanstd(X, axis=0)
        nutr = (X - mu) / sd

        # ---- food-group transition prior (written by evaluate_substitutions.py)
        groups = sorted(ing.food_group.unique())
        gpos = {g: k for k, g in enumerate(groups)}
        group_idx = ing.food_group.map(gpos).to_numpy()
        group_logp = np.zeros((len(groups), len(groups)))
        tpath = paths.MODELS / "group_transitions.csv"
        if tpath.exists():
            t = pd.read_csv(tpath, index_col=0)
            group_logp = t.reindex(index=groups, columns=groups).fillna(t.to_numpy().min()).to_numpy()

        return cls(names=names, index=index, ing=ing, emb=emb, flav_bin=fb, flav_idf=fi.tocsr(),
                   flav_has=flav_has, nutr=nutr, health=health_scores(ing), limit=limit_scores(ing),
                   water=ing.water_g.to_numpy(dtype=float),
                   whole_plant=whole_plant_food(ing).to_numpy(),
                   log_pop=np.log(ing.n_recipes.to_numpy(dtype=float)),
                   group_idx=group_idx, group_logp=group_logp,
                   compound_names=comp.compound_name.to_numpy(), compound_idf=idf,
                   weights=weights or HybridWeights())

    # ------------------------------------------------------------------ raw signals
    def emb_sim(self, i: int) -> np.ndarray:
        return self.emb @ self.emb[i]

    def flav_sim(self, i: int, variant: str) -> np.ndarray:
        """Ochiai (binary cosine) or IDF-weighted cosine; NaN where either side lacks a profile."""
        out = np.full(len(self.names), np.nan)
        if not self.flav_has[i]:
            return out
        if variant == "idf":
            sims = np.asarray((self.flav_idf @ self.flav_idf[i].T).todense()).ravel()
        else:
            shared = np.asarray((self.flav_bin @ self.flav_bin[i].T).todense()).ravel()
            sizes = np.asarray(self.flav_bin.sum(axis=1)).ravel()
            with np.errstate(divide="ignore", invalid="ignore"):
                sims = shared / np.sqrt(sizes * sizes[i])
        out[self.flav_has] = sims[self.flav_has]
        return out

    def nutr_sim(self, i: int) -> np.ndarray:
        """Negative RMS distance over nutrient features known for both (NaN if < 5 shared)."""
        d = self.nutr - self.nutr[i]
        both = ~np.isnan(d)
        n = both.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            rms = np.sqrt(np.nansum(d * d, axis=1) / n)
        rms[n < MIN_SHARED_NUTR_FEATURES] = np.nan
        return -rms

    def signals(self, i: int) -> dict[str, np.ndarray]:
        return {"emb": self.emb_sim(i), "flav_ochiai": self.flav_sim(i, "ochiai"),
                "flav_idf": self.flav_sim(i, "idf"), "nutr": self.nutr_sim(i), "pop": self.log_pop,
                "grp": self.group_logp[self.group_idx[i], self.group_idx]}

    @staticmethod
    def zscore(x: np.ndarray, exclude: int) -> np.ndarray:
        """Standardise across candidates (excluding the query itself); missing -> 0 (neutral)."""
        m = ~np.isnan(x)
        m[exclude] = False
        z = np.zeros_like(x, dtype=float)
        if m.sum() > 1:
            sd = x[m].std()
            if sd > 0:
                z[m] = (x[m] - x[m].mean()) / sd
        return z

    def standardized(self, i: int) -> dict[str, np.ndarray]:
        return {k: self.zscore(v, i) for k, v in self.signals(i).items()}

    def combine(self, z: dict[str, np.ndarray], w: HybridWeights) -> np.ndarray:
        flav = z["flav_idf"] if w.flavor_variant == "idf" else z["flav_ochiai"]
        return (w.emb * z["emb"] + w.flav * flav + w.nutr * z["nutr"] + w.pop * z["pop"]
                + w.grp * z["grp"])

    def comparable(self, i: int) -> np.ndarray:
        """Per-100 g comparison is meaningful (similar water content: not dry vs. cooked)."""
        with np.errstate(invalid="ignore"):
            return np.abs(self.water - self.water[i]) <= MAX_WATER_DIFF

    def required_improvement(self, i: int, min_reduction: float) -> float:
        return max(min_reduction * abs(self.health[i]), MIN_ABS_IMPROVEMENT)

    def little_to_limit(self, i: int) -> bool:
        """The source has less than MIN_ABS_IMPROVEMENT of nutrients to limit per 100 g."""
        return bool(np.isfinite(self.limit[i]) and self.limit[i] < MIN_ABS_IMPROVEMENT)

    def healthier_mask(self, i: int, min_reduction: float) -> np.ndarray:
        """Candidates that are comparable per 100 g and healthier by at least
        max(min_reduction x |score|, MIN_ABS_IMPROVEMENT). Empty for whole plant foods and for sources
        with almost nothing to limit."""
        hs = self.health[i]
        if not np.isfinite(hs) or self.whole_plant[i] or self.little_to_limit(i):
            return np.zeros(len(self.names), dtype=bool)
        need = self.required_improvement(i, min_reduction)
        with np.errstate(invalid="ignore"):
            return np.isfinite(self.health) & (hs - self.health >= need) & self.comparable(i)

    def seasoning_ok(self, i: int) -> np.ndarray:
        """Herbs/spices are only offered as healthier swaps for other herbs/spices."""
        groups = self.ing.food_group.to_numpy()
        if groups[i] == SEASONING_GROUP:
            return np.ones(len(self.names), dtype=bool)
        return groups != SEASONING_GROUP

    def group_ok(self, i: int, min_prob: float) -> np.ndarray:
        """Candidates whose food group receives >= min_prob of the source group's reported swaps."""
        return np.exp(self.group_logp[self.group_idx[i], self.group_idx]) >= min_prob

    @staticmethod
    def pool_mask(score: np.ndarray, i: int, pool: int) -> np.ndarray:
        """The `pool` highest-scoring candidates (the query itself excluded)."""
        sc = score.copy()
        sc[i] = -np.inf
        mask = np.zeros(len(sc), dtype=bool)
        mask[np.argsort(-sc, kind="stable")[:pool]] = True
        return mask

    def health_gain(self, i: int) -> np.ndarray:
        """Improvement of the health score relative to the source's magnitude; NaN when unknown."""
        hs = self.health[i]
        with np.errstate(invalid="ignore", divide="ignore"):
            return (hs - self.health) / max(abs(hs), MIN_ABS_IMPROVEMENT)

    # ------------------------------------------------------------------ public API
    def resolve(self, text: str) -> str:
        """Map free user text to a vocabulary ingredient.

        1. exact name; 2. the dataset's own normaliser (case, plurals, "garlic cloves" -> garlic,
        "fresh basil leaves" -> basil, ...); 3. a close spelling match for typos ("sour creme").
        Raises UnknownIngredient with up to 5 suggestions otherwise."""
        raw = " ".join(str(text).split())
        if raw in self.index:
            return raw
        for c in canonicalize(raw):
            if c in self.index:
                return c
        key = raw.lower()
        close = difflib.get_close_matches(key, self.names, n=5, cutoff=0.6)
        if close and difflib.SequenceMatcher(None, key, close[0]).ratio() >= 0.85:
            return close[0]
        contains = [n for n in self.names if key and key in n][:5]
        raise UnknownIngredient(raw, close or contains)

    def _rank(self, source: str, k: int, healthier: bool, weights: HybridWeights | None):
        source = self.resolve(source)
        w = weights or self.weights
        i = self.index[source]
        z = self.standardized(i)
        score = self.combine(z, w)
        valid = np.ones(len(self.names), dtype=bool)
        valid[i] = False
        if healthier:
            content = score - w.pop * z["pop"]      # plausibility pool ignores popularity
            valid &= (self.healthier_mask(i, w.min_reduction) & self.group_ok(i, w.min_group_prob)
                      & self.seasoning_ok(i) & self.pool_mask(content, i, w.health_pool))
            score = score + w.health_lambda * np.nan_to_num(self.health_gain(i))
        cand = np.where(valid)[0]
        order = cand[np.argsort(-score[cand], kind="stable")][:k]
        return i, order, score

    def similar(self, source: str, k: int = 10, weights: HybridWeights | None = None) -> pd.DataFrame:
        i, order, score = self._rank(source, k, False, weights)
        return self._table(i, order, score)

    def healthier(self, source: str, k: int = 10, weights: HybridWeights | None = None) -> pd.DataFrame:
        i, order, score = self._rank(source, k, True, weights)
        return self._table(i, order, score)

    # ------------------------------------------------------------------ explanations
    def shared_compounds(self, i: int, j: int, top: int = 5) -> tuple[int, list[str]]:
        a = set(self.flav_bin[i].indices)
        b = set(self.flav_bin[j].indices)
        common = sorted(a & b, key=lambda c: -self.compound_idf[c])
        return len(common), [str(self.compound_names[c]).replace("_", " ") for c in common[:top]]

    def _table(self, i: int, order: np.ndarray, score: np.ndarray) -> pd.DataFrame:
        sig = self.signals(i)
        rows = []
        src = self.ing.iloc[i]
        for rank, j in enumerate(order, 1):
            c = self.ing.iloc[j]
            n_shared, top_shared = self.shared_compounds(i, j) if self.flav_has[i] and self.flav_has[j] else (None, [])
            row = {
                "rank": rank, "ingredient": self.names[j], "food_group": c.food_group,
                "n_recipes": int(c.n_recipes), "score": round(float(score[j]), 3),
                "embedding_cosine": round(float(sig["emb"][j]), 3),
                "flavor_similarity": None if np.isnan(sig["flav_ochiai"][j]) else round(float(sig["flav_ochiai"][j]), 3),
                "shared_compounds": n_shared, "example_shared_compounds": ", ".join(top_shared),
                "nutrient_similarity": None if np.isnan(sig["nutr"][j]) else round(float(sig["nutr"][j]), 3),
                "health_score": None if np.isnan(self.health[j]) else round(float(self.health[j]), 3),
                "health_delta": None if np.isnan(self.health[j]) or np.isnan(self.health[i])
                else round(float(self.health[j] - self.health[i]), 3),
                "health_change_pct": None if np.isnan(self.health[j]) or np.isnan(self.health[i])
                or self.health[i] < PCT_DISPLAY_MIN
                else round(100 * float((self.health[j] - self.health[i]) / self.health[i]), 1),
            }
            for col in ["energy_kcal", "fat_g", "sat_fat_g", "sugar_g", "sodium_mg"]:
                a, b = src[col], c[col]
                row[f"delta_{col}"] = None if pd.isna(a) or pd.isna(b) else round(float(b - a), 2)
            rows.append(row)
        return pd.DataFrame(rows, columns=TABLE_COLUMNS)

    def explain(self, source: str, target: str) -> str:
        """One-paragraph, human-readable explanation of why `target` can replace `source`."""
        i, j = self.index[self.resolve(source)], self.index[self.resolve(target)]
        sig = self.signals(i)
        parts = [f"Used in similar recipes (embedding cosine {sig['emb'][j]:.2f})."]
        if self.flav_has[i] and self.flav_has[j]:
            n, top = self.shared_compounds(i, j, 3)
            noun = "flavor compound" if n == 1 else "flavor compounds"
            parts.append(f"Shares {n} {noun}" + (f" (e.g. {', '.join(top)})." if top else "."))
        else:
            parts.append("Flavor-compound data not available for this pair.")
        hs, ht = self.health[i], self.health[j]
        if np.isfinite(hs) and np.isfinite(ht):
            if hs >= PCT_DISPLAY_MIN:
                parts.append(f"Health score per 100 g changes by {100 * (ht - hs) / hs:+.0f}%.")
            else:
                parts.append(f"Health score per 100 g changes by {ht - hs:+.2f} (%DV units).")
        return " ".join(parts)
