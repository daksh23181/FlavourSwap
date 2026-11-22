"""Deliverable 3 - tune and evaluate the hybrid substitution model on known substitution pairs.

Ground truth: substitution pairs mined from Food.com reviews (src/mine_substitutions.py), kept when
reported by >= 2 different users. Pairs are directional (source -> target).

Protocol
  * Source ingredients are split once (seed 42): 30% validation, 70% test. All weights are tuned on
    validation sources only; the test sources are scored once at the end.
  * For a source s, every other vocabulary ingredient is a candidate. Ranking is "filtered": when
    scoring target t, the *other* known targets of s are removed from the candidate list, so a
    model is never penalised for ranking another correct substitute higher. Ties count half.
  * Metrics: pair-level Hit@k (share of pairs whose target ranks <= k) and MRR; 95% CIs from 1,000
    bootstrap resamples of test source ingredients; paired bootstrap for model differences.

Outputs
  models/hybrid_config.json             tuned weights (loaded by the recommender / app)
  data/processed/substitution_split.csv source -> validation/test
  reports/substitution_report.md        full results
  reports/substitution_eval_pairs.csv   per-pair ranks of every method on the test split

Run:  python src/evaluate_substitutions.py
"""

from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from flavorswap import paths  # noqa: E402
from flavorswap.recommender import HybridWeights, Recommender  # noqa: E402

GT_MIN_USERS = 2
VAL_FRAC = 0.30
SEED = 42
KS = [1, 3, 5, 10, 20]
N_BOOT = 1000
GRID = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
GRP_GRID = [0.0, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0]
LAMBDAS = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]
MIN_REDUCTION = 0.10
POOLS = [10, 20, 30, 50, 100, 3583]   # healthier options drawn from the top-N similar candidates
POOL_TOLERANCE = 0.05                 # smallest pool within 5% of the best validation MRR
MIN_GROUP_PROB = 0.02                 # healthier options only from food groups with >= 2% of swaps
HIGH_CONF_USERS = 5
FLAVOR_CURVE = [0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 0.75, 1.0, 1.5, 2.0]
FLAVOR_TOLERANCE = 0.01          # relative validation-MRR loss accepted for flavor retention
ACC = "Hybrid, accuracy-optimal"
DEFAULT = "Hybrid, flavor-aware (default)"
PROBES = ["butter", "sour cream", "sugar", "heavy cream", "mayonnaise", "ground beef",
          "all purpose flour", "vegetable oil", "egg", "cream cheese", "white rice", "bacon"]


# ============================================================================================
# ranking
# ============================================================================================
def filtered_ranks(scores: np.ndarray, src_idx: np.ndarray, targets: list[np.ndarray],
                   allowed: np.ndarray | None = None) -> list[np.ndarray]:
    """scores: (n_src, N). For each source row, rank of each of its targets among the allowed
    candidates (all but the source and its other targets). Ties count half. Returns a list of
    rank arrays aligned with `targets`."""
    out = []
    N = scores.shape[1]
    for r, (i, T) in enumerate(zip(src_idx, targets)):
        sc = scores[r]
        mask = np.ones(N, dtype=bool) if allowed is None else allowed[r].copy()
        mask[i] = False
        mask[T] = False
        base = np.sort(sc[mask])
        t_sc = sc[T]
        right = np.searchsorted(base, t_sc, side="right")
        left = np.searchsorted(base, t_sc, side="left")
        greater = len(base) - right
        equal = right - left
        out.append(1.0 + greater + 0.5 * equal)
    return out


def flavor_retention(b: "Block", scores: np.ndarray, rec: Recommender, k: int = 10) -> float:
    """Mean flavor-compound similarity (Ochiai) between a source and its top-k suggestions, over
    sources that have a flavor profile (suggestions without a profile are skipped)."""
    vals = []
    for r, i in enumerate(b.src_idx):
        if not rec.flav_has[i]:
            continue
        sc = scores[r].copy()
        sc[i] = -np.inf
        top = np.argpartition(-sc, k)[:k]
        f = b.raw["flav_ochiai"][r, top]
        f = f[~np.isnan(f)]
        if len(f):
            vals.append(f.mean())
    return float(np.mean(vals))


def metrics(ranks: np.ndarray) -> dict[str, float]:
    m = {f"Hit@{k}": float(np.mean(ranks <= k)) for k in KS}
    m["MRR"] = float(np.mean(1.0 / ranks))
    m["median_rank"] = float(np.median(ranks))
    return m


def random_baseline(n_cands: np.ndarray) -> dict[str, float]:
    """Exact expectation for a uniformly random ranking of n candidates (target included)."""
    m = {f"Hit@{k}": float(np.mean(np.minimum(k, n_cands) / n_cands)) for k in KS}
    harmonic = np.array([np.sum(1.0 / np.arange(1, n + 1)) for n in n_cands])
    m["MRR"] = float(np.mean(harmonic / n_cands))
    m["median_rank"] = float(np.median((n_cands + 1) / 2))
    return m


def bootstrap_ci(rank_lists: list[np.ndarray], fn, rng: np.random.Generator) -> tuple[float, float]:
    """Percentile CI of fn(concatenated ranks) when resampling sources with replacement."""
    n = len(rank_lists)
    vals = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, n, n)
        vals.append(fn(np.concatenate([rank_lists[j] for j in pick])))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def paired_diff_ci(a: list[np.ndarray], b: list[np.ndarray], fn, rng) -> tuple[float, float, float]:
    """Mean difference fn(a) - fn(b) with a paired bootstrap over sources; also P(diff <= 0)."""
    n = len(a)
    diffs = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, n, n)
        diffs.append(fn(np.concatenate([a[j] for j in pick])) - fn(np.concatenate([b[j] for j in pick])))
    diffs = np.array(diffs)
    return float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5)), float(np.mean(diffs <= 0))


# ============================================================================================
# data
# ============================================================================================
class Block:
    """Precomputed signals for a set of source ingredients."""

    def __init__(self, rec: Recommender, sources: list[str], targets: dict[str, list[str]]):
        self.sources = sources
        self.src_idx = np.array([rec.index[s] for s in sources])
        self.targets = [np.array([rec.index[t] for t in targets[s]]) for s in sources]
        self.target_names = [targets[s] for s in sources]
        n, N = len(sources), len(rec.names)
        self.raw = {k: np.zeros((n, N), dtype=np.float64)
                    for k in ["emb", "flav_ochiai", "flav_idf", "nutr", "pop", "grp"]}
        self.z = {k: np.zeros((n, N), dtype=np.float64) for k in self.raw}
        for r, i in enumerate(self.src_idx):
            sig = rec.signals(i)
            for k, v in sig.items():
                self.raw[k][r] = v
                self.z[k][r] = rec.zscore(v, i)
        self.gain = np.vstack([rec.health_gain(i) for i in self.src_idx])
        # ground-truth definition of a healthy swap (health + comparability only) ...
        self.healthier = np.vstack([rec.healthier_mask(i, MIN_REDUCTION) for i in self.src_idx])
        # ... and the model's extra plausibility filter (a blocked true swap counts as a miss)
        self.group_ok = np.vstack([rec.group_ok(i, MIN_GROUP_PROB) & rec.seasoning_ok(i) for i in self.src_idx])
        groups = rec.ing.food_group.to_numpy()
        self.same_group = np.vstack([groups == groups[i] for i in self.src_idx]).astype(float)

    def hybrid(self, w: HybridWeights) -> np.ndarray:
        flav = self.z["flav_idf"] if w.flavor_variant == "idf" else self.z["flav_ochiai"]
        return (w.emb * self.z["emb"] + w.flav * flav + w.nutr * self.z["nutr"] + w.pop * self.z["pop"]
                + w.grp * self.z["grp"])

    def single(self, key: str) -> np.ndarray:
        x = self.raw[key].copy()
        x[np.isnan(x)] = -np.inf     # no data -> ranked last
        return x


def fit_group_transitions(rec: Recommender, split: dict[str, str]) -> pd.DataFrame:
    """log P(target food group given source food group) from all mined review mentions whose source
    is NOT a test source (add-one smoothing). Saved to models/group_transitions.csv."""
    men = pd.read_csv(paths.PROCESSED / "substitution_mentions.csv")
    men = men[men.source.map(split).fillna("train") != "test"]
    grp = dict(zip(rec.ing.ingredient, rec.ing.food_group))
    groups = sorted(rec.ing.food_group.unique())
    counts = pd.crosstab(men.source.map(grp), men.target.map(grp)).reindex(
        index=groups, columns=groups, fill_value=0)
    probs = (counts + 1).div((counts + 1).sum(axis=1), axis=0)
    logp = np.log(probs)
    logp.to_csv(paths.MODELS / "group_transitions.csv")
    return logp


def load_ground_truth(rec: Recommender) -> pd.DataFrame:
    pairs = pd.read_csv(paths.PROCESSED / "substitution_pairs.csv")
    gt = pairs[pairs.n_users >= GT_MIN_USERS].copy()
    assert gt.source.isin(rec.index).all() and gt.target.isin(rec.index).all()
    gt["trivial"] = [set(s.split()) <= set(t.split()) or set(t.split()) <= set(s.split())
                     for s, t in zip(gt.source, gt.target)]
    return gt.reset_index(drop=True)


def split_sources(gt: pd.DataFrame) -> dict[str, str]:
    sources = np.array(sorted(gt.source.unique()))
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(sources))
    n_val = int(round(VAL_FRAC * len(sources)))
    return {s: ("validation" if k < n_val else "test") for k, s in zip(range(len(sources)), sources[perm])}


# ============================================================================================
# main
# ============================================================================================
def main() -> None:
    t0 = time.time()
    rng = np.random.default_rng(SEED)
    rec = Recommender.load(HybridWeights())
    gt = load_ground_truth(rec)
    split = split_sources(gt)
    fit_group_transitions(rec, split)
    rec = Recommender.load(HybridWeights())   # reload with the fitted transition prior
    gt["split"] = gt.source.map(split)
    pd.DataFrame(sorted(split.items()), columns=["source", "split"]).to_csv(
        paths.PROCESSED / "substitution_split.csv", index=False)
    print(f"Ground truth: {len(gt)} pairs, {gt.source.nunique()} sources "
          f"(val {sum(v == 'validation' for v in split.values())}, test {sum(v == 'test' for v in split.values())})")

    def block_for(df: pd.DataFrame) -> Block:
        tg = df.groupby("source").target.apply(list).to_dict()
        return Block(rec, sorted(tg), tg)

    val_b = block_for(gt[gt.split == "validation"])
    test_b = block_for(gt[gt.split == "test"])
    print(f"Signals precomputed ({time.time() - t0:.0f}s)")

    # ---------------------------------------------------------------- tuning (validation)
    grid_rows = []
    for variant, wf, wn, wp, wg in itertools.product(["ochiai", "idf"], GRID, GRID, GRID, GRP_GRID):
        w = HybridWeights(emb=1.0, flav=wf, nutr=wn, pop=wp, grp=wg, flavor_variant=variant)
        ranks = np.concatenate(filtered_ranks(val_b.hybrid(w), val_b.src_idx, val_b.targets))
        m = metrics(ranks)
        grid_rows.append({"flavor_variant": variant, "w_flav": wf, "w_nutr": wn, "w_pop": wp, "w_grp": wg, **m})
    grid = pd.DataFrame(grid_rows)
    grid = grid.sort_values(["MRR", "Hit@10"], ascending=False).reset_index(drop=True)
    print(f"Grid search: {len(grid)} configs ({time.time() - t0:.0f}s)")

    def best(cond) -> pd.Series:
        g = grid[cond(grid)]
        return g.iloc[0]

    def to_weights(row) -> HybridWeights:
        return HybridWeights(emb=1.0, flav=float(row.w_flav), nutr=float(row.w_nutr), pop=float(row.w_pop),
                             grp=float(row.w_grp), flavor_variant=row.flavor_variant)

    variants = {
        ACC: lambda g: g.index >= 0,
        "Hybrid without popularity prior": lambda g: g.w_pop == 0,
        "Hybrid without group-transition prior": lambda g: g.w_grp == 0,
        "Hybrid without flavor": lambda g: g.w_flav == 0,
        "Hybrid without nutrition": lambda g: g.w_nutr == 0,
        "Embedding + flavor": lambda g: (g.w_nutr == 0) & (g.w_pop == 0) & (g.w_grp == 0),
        "Embedding + nutrition": lambda g: (g.w_flav == 0) & (g.w_pop == 0) & (g.w_grp == 0),
    }
    tuned = {name: best(cond) for name, cond in variants.items()}
    acc_w = to_weights(tuned[ACC])

    # Flavor trade-off (validation). Accuracy is nearly flat in the flavor weight, so the default
    # model takes the LARGEST flavor weight whose validation MRR stays within FLAVOR_TOLERANCE of the
    # accuracy-optimal model (nutrition/popularity weights kept), maximising flavor retention at no
    # meaningful accuracy cost.
    best_mrr = float(tuned[ACC].MRR)
    curve_rows = []
    for wf in FLAVOR_CURVE:
        cands = []
        for variant in ["ochiai", "idf"]:
            w = HybridWeights(emb=1.0, flav=wf, nutr=acc_w.nutr, pop=acc_w.pop, grp=acc_w.grp,
                              flavor_variant=variant)
            m = metrics(np.concatenate(filtered_ranks(val_b.hybrid(w), val_b.src_idx, val_b.targets)))
            cands.append((m["MRR"], variant, m))
        mrr_v, variant, m = max(cands, key=lambda x: x[0])
        w = HybridWeights(emb=1.0, flav=wf, nutr=acc_w.nutr, pop=acc_w.pop, grp=acc_w.grp,
                          flavor_variant=variant)
        curve_rows.append({"w_flav": wf, "flavor_variant": variant, "val MRR": m["MRR"], "val Hit@10": m["Hit@10"],
                           "relative MRR": m["MRR"] / best_mrr,
                           "top-10 flavor similarity": flavor_retention(val_b, val_b.hybrid(w), rec)})
    curve = pd.DataFrame(curve_rows)
    ok = curve[curve["relative MRR"] >= 1 - FLAVOR_TOLERANCE]
    pick = ok.sort_values("w_flav").iloc[-1]
    best_w = HybridWeights(emb=1.0, flav=float(pick.w_flav), nutr=acc_w.nutr, pop=acc_w.pop, grp=acc_w.grp,
                           flavor_variant=pick.flavor_variant)

    # healthier-mode lambda (validation healthy pairs)
    def healthy_targets(b: Block) -> tuple[np.ndarray, list[np.ndarray], np.ndarray]:
        keep_rows, keep_t = [], []
        for r, T in enumerate(b.targets):
            ok = T[b.healthier[r, T]]
            if len(ok):
                keep_rows.append(r)
                keep_t.append(ok)
        return np.array(keep_rows, dtype=int), keep_t, b.src_idx[keep_rows]

    def healthier_ranks(b: Block, w: HybridWeights, lam: float, pool: int):
        """Healthier mode: candidates must be healthier, comparable, and within the top-`pool`
        similar candidates; a target outside that set can never be retrieved (rank = inf)."""
        rows, T, src = healthy_targets(b)
        sim = b.hybrid(w)[rows]
        content = sim - w.pop * b.z["pop"][rows]     # plausibility pool ignores popularity
        allowed = b.healthier[rows] & b.group_ok[rows]
        for r in range(len(rows)):
            allowed[r] &= Recommender.pool_mask(content[r], src[r], pool)
        sc = sim + lam * np.nan_to_num(b.gain[rows])
        ranks = filtered_ranks(sc, src, T, allowed=allowed)
        return [np.where(allowed[r, T[r]], ranks[r], np.inf) for r in range(len(rows))]

    lam_rows = []
    for pool, lam in itertools.product(POOLS, LAMBDAS):
        r = np.concatenate(healthier_ranks(val_b, best_w, lam, pool))
        lam_rows.append({"pool": pool, "lambda": lam, **metrics(r)})
    lam_df = pd.DataFrame(lam_rows).sort_values(["MRR", "Hit@10"], ascending=False)
    per_pool = lam_df.sort_values(["MRR", "Hit@10"], ascending=False).groupby("pool").head(1).sort_values("pool")
    best_pool_mrr = per_pool.MRR.max()
    chosen = per_pool[per_pool.MRR >= (1 - POOL_TOLERANCE) * best_pool_mrr].iloc[0]
    best_w.health_pool = int(chosen["pool"])
    best_w.health_lambda = float(chosen["lambda"])
    best_w.min_reduction = MIN_REDUCTION
    best_w.min_group_prob = MIN_GROUP_PROB

    # ---------------------------------------------------------------- test evaluation
    methods: dict[str, list[np.ndarray]] = {}
    methods["Popularity"] = filtered_ranks(np.tile(rec.log_pop, (len(test_b.sources), 1)),
                                           test_b.src_idx, test_b.targets)
    methods["Same food group + popularity"] = filtered_ranks(
        10.0 * test_b.same_group + np.tile(rec.log_pop, (len(test_b.sources), 1)), test_b.src_idx, test_b.targets)
    methods["Flavor only (Ochiai)"] = filtered_ranks(test_b.single("flav_ochiai"), test_b.src_idx, test_b.targets)
    methods["Flavor only (IDF)"] = filtered_ranks(test_b.single("flav_idf"), test_b.src_idx, test_b.targets)
    methods["Nutrition only"] = filtered_ranks(test_b.single("nutr"), test_b.src_idx, test_b.targets)
    methods["Embedding only (Deliverable 2)"] = filtered_ranks(test_b.single("emb"), test_b.src_idx, test_b.targets)
    for name in ["Embedding + flavor", "Embedding + nutrition", "Hybrid without flavor", "Hybrid without nutrition",
                 "Hybrid without popularity prior", "Hybrid without group-transition prior", ACC]:
        methods[name] = filtered_ranks(test_b.hybrid(to_weights(tuned[name])), test_b.src_idx, test_b.targets)
    methods[DEFAULT] = filtered_ranks(test_b.hybrid(best_w), test_b.src_idx, test_b.targets)
    same_model = acc_w.flav == best_w.flav and acc_w.flavor_variant == best_w.flavor_variant
    if same_model:          # the flavor-aware rule kept the accuracy-optimal weights: one model
        del methods[ACC]
    flavor_models = [("Embedding only (Deliverable 2)", HybridWeights(emb=1.0))] + \
        ([] if same_model else [(ACC, acc_w)]) + [(DEFAULT, best_w)]
    test_flavor = pd.DataFrame([
        {"model": name, "w_flav": w.flav, "top-10 flavor similarity": flavor_retention(test_b, test_b.hybrid(w), rec)}
        for name, w in flavor_models])

    n_cands = np.concatenate([np.full(len(T), len(rec.names) - 1 - (len(T) - 1)) for T in test_b.targets])
    results = [{"method": "Random (expected)", **random_baseline(n_cands), "MRR 95% CI": "",
                "Hit@10 95% CI": ""}]
    for name, rl in methods.items():
        all_r = np.concatenate(rl)
        lo, hi = bootstrap_ci(rl, lambda x: float(np.mean(1 / x)), rng)
        lo10, hi10 = bootstrap_ci(rl, lambda x: float(np.mean(x <= 10)), rng)
        results.append({"method": name, **metrics(all_r), "MRR 95% CI": f"[{lo:.3f}, {hi:.3f}]",
                        "Hit@10 95% CI": f"[{lo10:.3f}, {hi10:.3f}]"})
    res = pd.DataFrame(results)

    hyb = methods[DEFAULT]
    emb = methods["Embedding only (Deliverable 2)"]
    grp = methods["Same food group + popularity"]
    d_mrr = paired_diff_ci(hyb, emb, lambda x: float(np.mean(1 / x)), rng)
    d_h10 = paired_diff_ci(hyb, emb, lambda x: float(np.mean(x <= 10)), rng)
    d_grp = paired_diff_ci(hyb, grp, lambda x: float(np.mean(1 / x)), rng)
    d_acc = None if same_model else paired_diff_ci(hyb, methods[ACC], lambda x: float(np.mean(1 / x)), rng)

    # per-pair table + subsets
    pair_rows = []
    for r, (s, T) in enumerate(zip(test_b.sources, test_b.target_names)):
        for k, t in enumerate(T):
            row = {"source": s, "target": t}
            for name, rl in methods.items():
                row[name] = rl[r][k]
            pair_rows.append(row)
    per_pair = pd.DataFrame(pair_rows).merge(gt[["source", "target", "n_users", "trivial"]], on=["source", "target"])
    per_pair.to_csv(paths.ROOT / "reports" / "substitution_eval_pairs.csv", index=False)

    def subset_table(mask: pd.Series) -> pd.DataFrame:
        sub = per_pair[mask]
        rows = []
        for name in [n for n in ["Same food group + popularity", "Embedding only (Deliverable 2)", ACC, DEFAULT]
                     if n in methods]:
            rows.append({"method": name, "pairs": len(sub), **metrics(sub[name].to_numpy())})
        return pd.DataFrame(rows)

    nontrivial = subset_table(~per_pair.trivial)
    highconf = subset_table(per_pair.n_users >= HIGH_CONF_USERS)

    # healthier mode on test healthy pairs: similar-mode vs healthier-mode ranking
    rows_h, T_h, src_h = healthy_targets(test_b)
    sim_scores = test_b.hybrid(best_w)[rows_h]
    sim_r = filtered_ranks(sim_scores, src_h, T_h)
    hl_r = healthier_ranks(test_b, best_w, best_w.health_lambda, best_w.health_pool)
    healthy_res = pd.DataFrame([
        {"mode": "Similar mode (all candidates)", "pairs": int(sum(map(len, T_h))), **metrics(np.concatenate(sim_r))},
        {"mode": "Healthier mode (healthier candidates only)", "pairs": int(sum(map(len, T_h))),
         **metrics(np.concatenate(hl_r))},
    ])

    # nutrient outcomes of healthier recommendations for every test source
    outcome = healthier_outcomes(rec, test_b.sources, best_w)

    # ---------------------------------------------------------------- save config + report
    def test_metrics(name: str) -> dict:
        row = res[res.method == name].iloc[0]
        return {k: round(float(v), 4) for k, v in row.items() if k.startswith(("Hit", "MRR")) and "CI" not in k}

    cfg = {
        "weights": best_w.as_dict(),
        "accuracy_optimal_weights": {**acc_w.as_dict(), "health_lambda": best_w.health_lambda,
                                     "health_pool": best_w.health_pool},
        "selection": f"largest flavor weight with validation MRR >= {1 - FLAVOR_TOLERANCE:.2f} x best",
        "ground_truth": {"min_users": GT_MIN_USERS, "pairs": int(len(gt)), "sources": int(gt.source.nunique())},
        "split": {"seed": SEED, "validation_fraction": VAL_FRAC},
        "validation_MRR": {"accuracy_optimal": round(best_mrr, 4), "default": round(float(pick["val MRR"]), 4)},
        "test": {"default": test_metrics(DEFAULT),
                 "accuracy_optimal": test_metrics(DEFAULT if same_model else ACC)},
    }
    (paths.MODELS / "hybrid_config.json").write_text(json.dumps(cfg, indent=2))
    rec.weights = best_w
    write_report(gt, split, grid, tuned, per_pool, res, d_mrr, d_h10, d_grp, d_acc, nontrivial, highconf,
                 healthy_res, outcome, rec, curve, test_flavor, acc_w)
    print(res[["method", "Hit@1", "Hit@5", "Hit@10", "MRR"]].round(3).to_string(index=False))
    print(f"Done ({time.time() - t0:.0f}s)")


def healthier_outcomes(rec: Recommender, sources: list[str], w: HybridWeights, k: int = 5) -> pd.DataFrame:
    """Average per-100 g change of the top-k healthier suggestions vs. the source, and how much
    similarity is retained compared with the top-k similar suggestions."""
    rows = []
    for s in sources:
        i = rec.index[s]
        if not np.isfinite(rec.health[i]):
            continue
        h = rec.healthier(s, k, w)
        if h.empty:
            continue
        sim = rec.similar(s, k, w)
        src = rec.ing.iloc[i]
        row = {"source": s, "n_healthier": len(h),
               "health_change_pct": pd.to_numeric(h.health_change_pct, errors="coerce").mean(),
               "health_delta": pd.to_numeric(h.health_delta, errors="coerce").mean(),
               "emb_cos_healthier": h.embedding_cosine.mean(), "emb_cos_similar": sim.embedding_cosine.mean(),
               "flav_healthier": pd.to_numeric(h.flavor_similarity, errors="coerce").mean(),
               "flav_similar": pd.to_numeric(sim.flavor_similarity, errors="coerce").mean()}
        for col in ["energy_kcal", "fat_g", "sat_fat_g", "sugar_g", "sodium_mg"]:
            base = src[col]
            d = pd.to_numeric(h[f"delta_{col}"], errors="coerce")
            row[f"delta_{col}"] = d.mean()
            row[f"src_{col}"] = base
        rows.append(row)
    return pd.DataFrame(rows)


def fmt(df: pd.DataFrame, digits: int = 3) -> str:
    return df.round(digits).to_markdown(index=False)


def write_report(gt, split, grid, tuned, lam_df, res, d_mrr, d_h10, d_grp, d_acc, nontrivial, highconf,
                 healthy_res, outcome, rec: Recommender, curve, test_flavor, acc_w) -> None:
    w = rec.weights
    audit = pd.read_csv(paths.ROOT / "reports" / "substitution_mining_audit.csv")
    audit = audit[audit.still_mined]
    n_val = sum(v == "validation" for v in split.values())
    n_test = len(split) - n_val
    gt_test = gt[gt.split == "test"]
    tuned_tbl = pd.DataFrame([{"variant": k, "flavor": v.flavor_variant, "w_flav": v.w_flav, "w_nutr": v.w_nutr,
                               "w_pop": v.w_pop, "w_grp": v.w_grp, "val MRR": v.MRR, "val Hit@10": v["Hit@10"]} for k, v in tuned.items()])
    res_tbl = res[["method", "Hit@1", "Hit@3", "Hit@5", "Hit@10", "Hit@20", "MRR", "MRR 95% CI",
                   "Hit@10 95% CI", "median_rank"]]
    mean_out = outcome[[c for c in outcome.columns if c.startswith(("delta_", "health_", "emb_cos", "flav_"))]].mean()

    lines = [
        "# FlavorSwap - Deliverable 3: hybrid substitution model",
        "",
        "Generated by `src/evaluate_substitutions.py`. Ground truth from `src/mine_substitutions.py`.",
        "",
        "## Model",
        "For a source ingredient, every other vocabulary ingredient is scored with five signals, each "
        "standardised across candidates (missing data = neutral):",
        "",
        "| signal | data | definition |",
        "|---|---|---|",
        "| embedding | Deliverable 2 Word2Vec (recipe co-occurrence) | cosine similarity |",
        "| flavor | Ahn et al. (2011) compound sets | Ochiai cosine, plain or IDF-weighted (tuned) |",
        "| nutrition | USDA SR Legacy, per 100 g | negative RMS distance of standardised log nutrients "
        "(energy, protein, fat, carbs, sugars, fiber, sodium) |",
        "| popularity | Food.com | log number of recipes using the candidate |",
        "| group transition | Food.com reviews (non-test sources only) | log P(candidate food group given "
        "source food group), add-one smoothed |",
        "",
        f"Default weights: embedding **{w.emb}**, flavor **{w.flav}** ({w.flavor_variant}), "
        f"nutrition **{w.nutr}**, popularity **{w.pop}**, group transition **{w.grp}**. The "
        + (f"accuracy-optimal weights on validation already include flavor weight {acc_w.flav}; the "
           "flavor-aware rule (largest flavor weight within "
           f"{FLAVOR_TOLERANCE:.0%} of the best validation MRR) keeps it unchanged (see *Flavor trade-off*)."
           if acc_w.flav == w.flav else
           f"accuracy-optimal weights on validation are identical except flavor = {acc_w.flav}; accuracy "
           "is flat in the flavor weight, so the default takes the largest flavor weight within "
           f"{FLAVOR_TOLERANCE:.0%} of the best validation MRR (see *Flavor trade-off*)."),
        "",
        "**Healthier mode** makes no suggestions for whole fruit, vegetables, legumes and nuts (the foods "
        "dietary guidelines already recommend; ranking them against each other by small fiber or sugar "
        "differences is not meaningful health advice) or for ingredients with almost nothing to limit "
        "(< 0.10 %DV units of energy, saturated fat, free sugars, sodium and trans fat per 100 g, e.g. "
        "vinegar, lemon juice, wine, ice: cutting these cannot reach the required improvement, so a "
        "'healthier' swap would only add fiber, such as ice -> guava) and otherwise considers only "
        "candidates that are (a) "
        "among the "
        f"**{w.health_pool}** most similar candidates by content (hybrid score without the popularity "
        f"prior), (b) from a food group that receives at least {MIN_GROUP_PROB:.0%} of the source "
        "group's reported swaps (fixed a priori; keeps implausible cross-group suggestions such as "
        "rice -> garlic out), (c) not a herb/spice unless the source is one (seasonings are used by the "
        "gram, so per-100 g comparisons with main ingredients are meaningless; salt -> herbs stays "
        "allowed), (d) comparable per 100 g (water content within 50 percentage points, so no "
        "dry-vs-cooked comparisons) and (e) healthier by at least "
        f"max({int(MIN_REDUCTION * 100)}% of the source's score, 0.10 units); it adds lambda x relative "
        f"improvement (lambda = **{w.health_lambda}**). Pool size and lambda are tuned on validation: the "
        f"smallest pool within {POOL_TOLERANCE:.0%} of the best validation MRR.",
        "",
        "**Health score** per 100 g (lower = healthier), in %Daily-Value units: energy/2000 kcal + "
        "saturated fat/20 g + free sugars/50 g + sodium/2300 mg + trans fat/2.2 g (WHO limit) - fiber/28 g. "
        "Following the WHO definition of free sugars, sugars inside intact fruit and vegetables and the milk "
        "sugar of plain dairy are not penalised (juices, syrups and sweetened products are). "
        "As in nutrient-profiling systems such as Nutri-Score, total fat is not penalised (unsaturated "
        "oils and nuts are not punished) and fiber is credited (whole grains beat refined grains). An "
        "ingredient without reliable nutrient data is never called healthier. Comparisons are per 100 g, "
        "i.e. assume an equal-mass swap.",
        "",
        "## Ground truth: substitutions reported in Food.com reviews",
        f"- Mined from 1,132,198 reviews with fixed substitution phrasings; the replaced ingredient must "
        "be an ingredient of the reviewed recipe (see `src/mine_substitutions.py`, tests in "
        "`tests/test_mine_substitutions.py`).",
        f"- Manual audit of {len(audit)} randomly sampled mentions: **{(audit.verdict == 'correct').sum()}/"
        f"{len(audit)} correct** (`reports/substitution_mining_audit.csv`).",
        f"- Evaluation pairs: reported by >= {GT_MIN_USERS} different users -> **{len(gt):,} pairs** over "
        f"**{gt.source.nunique()} source ingredients** ({gt.trivial.mean():.1%} are name variants such as "
        "*garlic -> garlic powder*).",
        f"- Split by source ingredient: {n_val} validation sources, {n_test} test sources "
        f"({len(gt_test):,} test pairs). The Word2Vec model never saw review text, and its "
        "hyper-parameters were chosen without substitution pairs (Deliverable 2).",
        "",
        "## Results on the test split",
        f"Candidates per query: {len(rec.names) - 1:,} ingredients. Filtered ranking; 95% bootstrap CIs over "
        "test source ingredients.",
        "",
        fmt(res_tbl),
        "",
        f"- Default hybrid vs. embedding only, paired bootstrap: MRR difference 95% CI [{d_mrr[0]:+.3f}, "
        f"{d_mrr[1]:+.3f}] (bootstrap share with no improvement = {d_mrr[2]:.3f}); Hit@10 difference "
        f"[{d_h10[0]:+.3f}, {d_h10[1]:+.3f}] (share = {d_h10[2]:.3f}).",
        f"- Default hybrid vs. same-food-group + popularity: MRR difference [{d_grp[0]:+.3f}, {d_grp[1]:+.3f}] "
        f"(share = {d_grp[2]:.3f}).",
        ("- The flavor-aware default and the accuracy-optimal hybrid are the same model (the rule kept "
         "the accuracy-optimal flavor weight), so they are reported as one row." if d_acc is None else
         f"- Default (flavor-aware) vs. accuracy-optimal hybrid: MRR difference [{d_acc[0]:+.3f}, "
         f"{d_acc[1]:+.3f}]."),
        "",
        "## Flavor trade-off",
        "Flavor overlap alone is a weak predictor of which swaps reviewers make (see table above): "
        "reviewers often report convenience swaps (\"it's what I had\"), which do not reward flavor "
        "fidelity. Because the project explicitly aims to preserve flavor, the default model takes the "
        f"largest flavor weight whose validation MRR is within {FLAVOR_TOLERANCE:.0%} of the best. This "
        "rule was added after an earlier evaluation run (before the group-transition prior existed) "
        "selected a flavor weight of 0 in a flat region; it uses validation data only.",
        "",
        "Validation curve (nutrition and popularity weights fixed at their tuned values):",
        "",
        fmt(curve),
        "",
        "Mean flavor-compound similarity between a source and its top-10 suggestions (test sources with a "
        "flavor profile):",
        "",
        fmt(test_flavor),
        "",
        "### Subsets (test split)",
        "Excluding name-variant pairs (e.g. *onion -> onion powder*):",
        "",
        fmt(nontrivial[["method", "pairs", "Hit@1", "Hit@5", "Hit@10", "MRR"]]),
        "",
        f"High-confidence pairs (reported by >= {HIGH_CONF_USERS} users):",
        "",
        fmt(highconf[["method", "pairs", "Hit@1", "Hit@5", "Hit@10", "MRR"]]),
        "",
        "## Healthier mode",
        "Test pairs where users swapped to a comparable, healthier ingredient (same criterion (d)-(e) as "
        "the model):",
        "",
        fmt(healthy_res[["mode", "pairs", "Hit@1", "Hit@3", "Hit@5", "Hit@10", "MRR"]]),
        "",
        "The healthier mode ranks only healthier candidates, so on these pairs part of its advantage comes "
        "from that filter by construction; the comparison shows that the healthy swaps people actually "
        "make are found near the top. A true healthy swap excluded by the pool or the food-group filter "
        "counts as a miss.",
        "",
        f"Average outcome of the top-5 healthier suggestions over {len(outcome)} test source ingredients "
        "(per 100 g, equal-mass swap):",
        "",
        "| quantity | mean |",
        "|---|---|",
        f"| health score change | {mean_out['health_delta']:+.3f} %DV units "
        f"({mean_out['health_change_pct']:+.1f}% over sources with score >= 0.5) |",
        f"| energy change | {mean_out['delta_energy_kcal']:+.1f} kcal |",
        f"| fat change | {mean_out['delta_fat_g']:+.1f} g |",
        f"| saturated fat change | {mean_out['delta_sat_fat_g']:+.2f} g |",
        f"| sugars change | {mean_out['delta_sugar_g']:+.1f} g |",
        f"| sodium change | {mean_out['delta_sodium_mg']:+.0f} mg |",
        f"| embedding cosine: healthier vs. similar top-5 | {mean_out['emb_cos_healthier']:.3f} vs. "
        f"{mean_out['emb_cos_similar']:.3f} |",
        f"| flavor similarity: healthier vs. similar top-5 | {mean_out['flav_healthier']:.3f} vs. "
        f"{mean_out['flav_similar']:.3f} |",
        "",
        "## Tuning (validation split only)",
        "Each variant's remaining weights are re-tuned on validation (so ablations are fair); "
        f"grid {GRID} for flavor, nutrition and popularity, {GRP_GRID} for the group prior; embedding "
        "weight fixed at 1.",
        "",
        fmt(tuned_tbl),
        "",
        "Top 10 configurations of the full grid:",
        "",
        fmt(grid.head(10)[["flavor_variant", "w_flav", "w_nutr", "w_pop", "w_grp", "Hit@1", "Hit@5", "Hit@10",
                           "MRR"]]),
        "",
        "Healthier mode, best lambda per pool size (validation healthy pairs):",
        "",
        fmt(lam_df[["pool", "lambda", "Hit@1", "Hit@5", "Hit@10", "MRR"]]),
        "",
        "## Example recommendations (tuned model)",
        "",
    ]
    for p in PROBES:
        if p not in rec.index:
            continue
        sim = rec.similar(p, 5)
        hl = rec.healthier(p, 5)
        lines.append(f"**{p}** - similar: " + ", ".join(sim.ingredient))
        lines.append("")
        if hl.empty:
            lines.append("  healthier: none (no nutrient data or no healthier candidate)")
        else:
            lines.append("  healthier: " + ", ".join(
                f"{a} ({b:+.0f}%)" if pd.notna(b) else f"{a} ({d:+.2f})"
                for a, b, d in zip(hl.ingredient, hl.health_change_pct, hl.health_delta)))
        lines.append("")
    lines += [
        "## Limitations",
        "- Ground truth comes from what reviewers chose to write; substitutes that work but are rarely "
        "mentioned count as misses, so absolute Hit@k values are conservative.",
        "- Context-free: the model scores a swap for an ingredient in general, not for a specific recipe "
        "(e.g. butter -> oil suits a saute better than a pie crust). Some healthier suggestions are "
        "ingredients of the same healthy-cooking context rather than true swaps (e.g. brown rice -> whole "
        "wheat flour); the app shows the evidence for every suggestion so users can judge.",
        "- Health comparisons are per 100 g (equal-mass swap) and do not consider micronutrients or fat "
        "quality beyond saturated/trans fat (e.g. omega-3 in oily fish); trans fat is partly estimated "
        "(see `reports/dataset_report.md`).",
        "",
    ]
    (paths.ROOT / "reports" / "substitution_report.md").write_text("\n".join(lines), encoding="utf-8")


if __name__ == "__main__":
    main()
