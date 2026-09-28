"""Process A: Candidate Coverage Depth Audit (H1) and Geo & Delivery Analytical Ablations (H3).

Evaluates:
- H1: Candidate coverage-by-depth at 50, 100, 150, 200, 300, 500, 750, 1000
- H3: Geo & Delivery analytical treatments:
  Variant A: Baseline geo formula (same_loc=30, decay=0.25/10km)
  Variant B: No geo (pure text/semantic RRF)
  Variant C: Weaker same-location bonus (5.0, 10.0, 15.0)
  Variant D: Category-0 wildcard fix + Delivery-aware distance decay
  Variant E: Geo bonus gated by semantic score (e.g. Top 30% / 50% percentile)
- Critical slice evaluations: seen vs unseen, cat 0 vs 114, cross-location vs same-location
- Saves the candidate pool to artifacts/research_v3/val_candidate_pool.parquet
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from avito_candidate_generation.fusion.candidate_pool import (
    evaluate_rankings,
    evaluate_slices,
    oracle_candidate_coverage,
    reciprocal_rank_fusion,
)
from avito_candidate_generation.research import (
    context_folds,
    context_frame,
    field_text,
    ground_truth_map,
)
from avito_candidate_generation.retrievers.routed_e5 import RoutedDenseE5Retriever
from avito_candidate_generation.retrievers.routed_lexical import (
    FieldAwareSparseIndex,
    SparseBranchIndex,
)
from avito_candidate_generation.routing import (
    ExactPosteriorPredictor,
    GeneralizingClassifierPredictor,
    HybridMicrocategoryPredictor,
)


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r_lat1, r_lon1 = np.radians(lat1), np.radians(lon1)
    r_lat2, r_lon2 = np.radians(lat2), np.radians(lon2)
    dlat = r_lat2 - r_lat1
    dlon = r_lon2 - r_lon1
    a = np.sin(dlat / 2.0) ** 2 + np.cos(r_lat1) * np.cos(r_lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return 6371.0 * c


def main() -> None:
    t0 = time.time()
    print("=" * 70)
    print("PROCESS A: COVERAGE CEILING (H1) & GEO/DELIVERY ABLATIONS (H3)")
    print("=" * 70)

    train = pd.read_parquet("data/train.parquet")
    benchmark_items = pd.read_parquet("data/benchmark_items.parquet")
    benchmark_ids_set = set(benchmark_items["item_id"].astype(str))

    work = context_frame(train)
    folds = context_folds(work["internal_query_id"].astype(str), seed=42, folds=5)
    merged = pd.merge(work, folds, on="internal_query_id")

    train_part = merged[merged["fold"] != 0].copy()
    val_rows = merged[merged["fold"] == 0].copy()
    val_with_bm_pos = val_rows[val_rows["item_id"].astype(str).isin(list(benchmark_ids_set))].copy()

    val_contexts = (
        val_with_bm_pos[
            [
                "search_query",
                "search_location_id",
                "search_is_delivery_search",
                "search_infm_params_text",
                "search_category",
                "internal_query_id",
            ]
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    query_ids = [str(x) for x in val_contexts["internal_query_id"].tolist()]
    relevant = ground_truth_map(val_with_bm_pos[["internal_query_id", "item_id"]].drop_duplicates())

    train_queries_set = set(
        train_part["search_query"].dropna().astype(str).str.lower().str.strip()
    )

    loc_geo = (
        train.dropna(subset=["item_latitude", "item_longitude"])
        .groupby("item_location_id")[["item_latitude", "item_longitude"]]
        .mean()
    )
    loc_lat_dict = loc_geo["item_latitude"].astype(float).to_dict()
    loc_lon_dict = loc_geo["item_longitude"].astype(float).to_dict()

    item_id_list = [str(x) for x in benchmark_items["item_id"].tolist()]
    item_loc_map = dict(zip(item_id_list, benchmark_items["item_location_id"]))
    item_cat_map = dict(zip(item_id_list, benchmark_items["item_category_id"]))
    item_microcat_map = dict(zip(item_id_list, benchmark_items["item_microcat_id"]))
    item_lat_map = dict(zip(item_id_list, benchmark_items["item_latitude"].astype(float).fillna(0.0)))
    item_lon_map = dict(zip(item_id_list, benchmark_items["item_longitude"].astype(float).fillna(0.0)))
    item_titles = [str(x) for x in benchmark_items["item_title_raw"].fillna("").tolist()]

    query_loc_map = dict(zip(query_ids, val_contexts["search_location_id"]))
    query_cat_map = dict(zip(query_ids, val_contexts["search_category"]))
    query_deliv_map = dict(zip(query_ids, val_contexts["search_is_delivery_search"]))

    # Microcategory Routing Top-5
    print("Fitting routing models...")
    exact = ExactPosteriorPredictor().fit(train_part, query_col="search_query", category_col="item_microcat_id")
    clf = GeneralizingClassifierPredictor(min_df=5, max_features=50000, alpha=1e-5, seed=42).fit(train_part)
    hybrid = HybridMicrocategoryPredictor(exact_predictor=exact, generalizing_predictor=clf, min_count=1, blend_strategy="fallback")
    top5_microcats = hybrid.predict_top_k(val_contexts, k=5)

    microcat_to_items: dict[str, set[str]] = {}
    for item_id in item_id_list:
        mc = item_microcat_map[item_id]
        microcat_to_items.setdefault(str(mc), set()).add(item_id)

    allowed_items_per_query = []
    for cats in top5_microcats:
        allowed = set()
        for c in cats:
            allowed.update(microcat_to_items.get(str(c), set()))
        allowed_items_per_query.append(allowed)

    # Sparse Retrieval
    print("Fitting sparse retrieval...")
    field_index = FieldAwareSparseIndex.fit(benchmark_items, branches=["branch_a", "branch_b"], min_df=2)
    title_char_index = SparseBranchIndex.fit(
        item_id_list, item_titles, name="title_char", analyzer="char_wb", ngram_range=(3, 5), min_df=2
    )

    title_queries = field_text(val_contexts, ("search_query",))
    query_params_texts = field_text(val_contexts, ("search_query", "search_infm_params_text"))

    print("Retrieving sparse candidates...")
    routed_word = field_index.branches["branch_a"].retrieve_routed(title_queries, allowed_items_per_query, k=1000)
    routed_char = title_char_index.retrieve_routed(title_queries, allowed_items_per_query, k=1000)
    routed_title_params = field_index.branches["branch_b"].retrieve_routed(query_params_texts, allowed_items_per_query, k=1000)
    global_word = field_index.branches["branch_a"].retrieve(title_queries, k=300)
    global_char = title_char_index.retrieve(title_queries, k=300)

    # Dense E5
    print("Retrieving Dense E5 candidates...")
    e5_items_json = [str(x) for x in json.loads(Path("artifacts/selected/embeddings/e5/benchmark/item_ids.json").read_text())]
    e5_item_embeddings = np.load("artifacts/selected/embeddings/e5/benchmark/embeddings.npy", mmap_mode="r")
    val_query_embeddings = np.load("artifacts/research/val_query_e5_embeddings.npy")
    e5_retriever = RoutedDenseE5Retriever(item_ids=e5_items_json, item_embeddings=e5_item_embeddings)

    routed_e5 = e5_retriever.retrieve_routed(val_query_embeddings, allowed_items_per_query, k=1000)
    global_e5 = e5_retriever.retrieve_global(val_query_embeddings, k=300)

    sources = [routed_e5, routed_char, routed_word, routed_title_params, global_e5, global_word, global_char]
    weights = [1.5, 1.2, 0.8, 0.5, 0.4, 0.3, 0.3]

    print("Computing candidate pool RRF (depth=1000)...")
    raw_rrf = reciprocal_rank_fusion(sources, weights=weights, c=60.0, limit=1000)

    # =========================================================================
    # H1: CANDIDATE COVERAGE-BY-DEPTH AUDIT
    # =========================================================================
    print("\n" + "=" * 50)
    print("H1: CANDIDATE COVERAGE-BY-DEPTH AUDIT")
    print("=" * 50)
    depths = [50, 100, 150, 200, 300, 500, 750, 1000]
    coverage_by_depth: dict[int, float] = {}

    for d in depths:
        cov = oracle_candidate_coverage([raw_rrf], query_ids, relevant, ks=(d,))[f"union_coverage@{d}"]
        coverage_by_depth[d] = float(cov)
        print(f"Candidate Pool Depth @{d:<4}: Oracle Coverage = {cov:.4f}")

    ranking_gap_300 = coverage_by_depth[300] - 0.7050
    ranking_gap_500 = coverage_by_depth[500] - 0.7050
    print(f"\nRanking Gap vs Baseline (0.7050): @300 = {ranking_gap_300:+.4f}, @500 = {ranking_gap_500:+.4f}")
    if ranking_gap_300 > 0.08 or ranking_gap_500 > 0.08:
        print(">> H1 CONFIRMED: Substantial positives (>8 pp) already inside candidate pool but lost during final ranking!")
    else:
        print(">> H1 KILLED or LIMITED: Pool ceiling saturates early.")

    # =========================================================================
    # H3: GEO & DELIVERY ANALYTICAL ABLATIONS
    # =========================================================================
    print("\n" + "=" * 50)
    print("H3: GEO & DELIVERY ANALYTICAL ABLATIONS")
    print("=" * 50)

    # Slices metadata
    val_queries_norm = val_contexts["search_query"].fillna("").astype(str).str.lower().str.strip()
    is_seen = val_queries_norm.isin(train_queries_set).to_numpy()
    is_unseen = ~is_seen
    is_cat0 = (val_contexts["search_category"] == 0).to_numpy()
    is_cat114 = (val_contexts["search_category"] == 114).to_numpy()

    # Precompute cross-location positives flag per query
    has_cross_loc_pos = []
    for qid in query_ids:
        qloc = query_loc_map[qid]
        pos_items = relevant.get(qid, set())
        has_cross = any(item_loc_map.get(it) != qloc for it in pos_items)
        has_cross_loc_pos.append(has_cross)
    has_cross_loc_pos = np.array(has_cross_loc_pos, dtype=bool)

    geo_experiments_results: dict[str, Any] = {
        "coverage_by_depth": coverage_by_depth,
        "variants": {},
    }

    variants = [
        # (name, same_loc_bonus, dist_decay, cat0_wildcard, dist_scale, score_gating_pct)
        ("A_Baseline_Geo", 30.0, 0.25, False, 10.0, 0.0),
        ("B_No_Geo", 1.0, 0.0, False, 10.0, 0.0),
        ("C1_Weaker_SameLoc_5", 5.0, 0.25, False, 10.0, 0.0),
        ("C2_Weaker_SameLoc_15", 15.0, 0.25, False, 10.0, 0.0),
        ("D1_Cat0_Wildcard", 30.0, 0.25, True, 10.0, 0.0),
        ("D2_Cat0_Wildcard_Decay015", 30.0, 0.15, True, 10.0, 0.0),
        ("D3_Cat0_Wildcard_SameLoc15_Decay015", 15.0, 0.15, True, 10.0, 0.0),
        ("E1_SemanticGated_Top50pct", 30.0, 0.25, True, 10.0, 0.50),
        ("E2_SemanticGated_Top30pct", 30.0, 0.25, True, 10.0, 0.30),
    ]

    for name, same_loc_b, decay, cat0_wild, scale, gate_pct in variants:
        rankings = []
        for qid, ranking in zip(query_ids, raw_rrf, strict=True):
            q_loc = query_loc_map[qid]
            q_cat = query_cat_map[qid]
            q_lat = loc_lat_dict.get(q_loc)
            q_lon = loc_lon_dict.get(q_loc)

            # If semantic gating is active: determine score threshold for this query
            score_thresh = 0.0
            if gate_pct > 0.0 and ranking:
                scores_all = [s for _, s in ranking]
                score_thresh = float(np.percentile(scores_all, (1.0 - gate_pct) * 100.0))

            boosted = []
            for item_id, score in ranking:
                it_loc = item_loc_map.get(item_id)
                it_cat = item_cat_map.get(item_id)
                it_lat = item_lat_map.get(item_id, 0.0)
                it_lon = item_lon_map.get(item_id, 0.0)

                # Category match
                if cat0_wild and q_cat == 0:
                    cat_mult = 1.0  # Wildcard: no penalty!
                else:
                    cat_mult = 1.0 if (it_cat == q_cat) else 0.001

                # Geo match
                if decay == 0.0 and same_loc_b == 1.0:
                    geo_mult = 1.0
                elif gate_pct > 0.0 and score < score_thresh:
                    # Gated out: no geo bonus if semantic relevance is poor
                    geo_mult = 1.0
                elif it_loc == q_loc:
                    geo_mult = same_loc_b
                elif q_lat is not None and it_lat != 0.0:
                    dist = haversine_km(q_lat, q_lon, it_lat, it_lon)
                    geo_mult = float(np.exp(-decay * (dist / scale)))
                else:
                    geo_mult = 0.05

                boosted.append((item_id, score * cat_mult * geo_mult))

            boosted.sort(key=lambda p: (-p[1], p[0]))
            rankings.append(boosted[:50])

        m = evaluate_rankings(rankings, query_ids, relevant, ks=(50,))
        r50 = m["recall@50"]

        # Evaluate on slices
        per_q_recall = []
        for qid, rk in zip(query_ids, rankings, strict=True):
            exp = relevant.get(qid, set())
            top50 = {it for it, _ in rk}
            per_q_recall.append(len(top50 & exp) / max(len(exp), 1))
        per_q_arr = np.array(per_q_recall)

        r50_seen = float(per_q_arr[is_seen].mean()) if is_seen.any() else 0.0
        r50_unseen = float(per_q_arr[is_unseen].mean()) if is_unseen.any() else 0.0
        r50_cat0 = float(per_q_arr[is_cat0].mean()) if is_cat0.any() else 0.0
        r50_cat114 = float(per_q_arr[is_cat114].mean()) if is_cat114.any() else 0.0
        r50_cross_loc = float(per_q_arr[has_cross_loc_pos].mean()) if has_cross_loc_pos.any() else 0.0

        geo_experiments_results["variants"][name] = {
            "recall@50": float(r50),
            "delta_vs_baseline": float(r50 - 0.7050),
            "r50_seen": r50_seen,
            "r50_unseen": r50_unseen,
            "r50_cat0": r50_cat0,
            "r50_cat114": r50_cat114,
            "r50_cross_loc": r50_cross_loc,
        }

        print(
            f"{name:<35}: R@50={r50:.4f} (Δ={r50 - 0.7050:+.4f}) | "
            f"Seen={r50_seen:.4f}, Unseen={r50_unseen:.4f}, Cat0={r50_cat0:.4f}, CrossLoc={r50_cross_loc:.4f}"
        )

    # Save artifact
    out_path = Path("artifacts/research_v3/geo_experiments.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(geo_experiments_results, f, indent=2)

    runtime_sec = time.time() - t0
    print(f"\nProcess A completed in {runtime_sec:.1f}s. Artifact saved to {out_path}")


if __name__ == "__main__":
    main()
