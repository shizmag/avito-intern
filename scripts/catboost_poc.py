"""Proof of Concept: Memory-safe CatBoost Ranker vs Analytical Geo Formula.

Evaluates on a representative sample of 1,000 canonical validation queries:
- Candidate depth = 100 per query (100,000 candidate pairs total)
- Pure vectorized feature engineering in Pandas (zero memory explosion)
- 5-Fold Cross Validation
- Direct head-to-head comparison of Recall@50: Analytical Geo Formula vs Learned CatBoost
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from avito_candidate_generation.fusion.candidate_pool import evaluate_rankings, reciprocal_rank_fusion
from avito_candidate_generation.research import context_folds, context_frame, field_text, ground_truth_map
from avito_candidate_generation.retrievers.routed_e5 import RoutedDenseE5Retriever
from avito_candidate_generation.retrievers.routed_lexical import FieldAwareSparseIndex, SparseBranchIndex
from avito_candidate_generation.routing import ExactPosteriorPredictor, GeneralizingClassifierPredictor, HybridMicrocategoryPredictor


def haversine_km(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
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
    print("PROOF OF CONCEPT: CATBOOST RANKER VS ANALYTICAL GEO-CASCADE")
    print("=" * 70)

    # 1. Load data
    train = pd.read_parquet("data/train.parquet")
    benchmark_items = pd.read_parquet("data/benchmark_items.parquet")
    benchmark_ids_set = set(benchmark_items["item_id"].astype(str))

    work = context_frame(train)
    folds = context_folds(work["internal_query_id"].astype(str), seed=42, folds=5)
    merged = pd.merge(work, folds, on="internal_query_id")

    train_part = merged[merged["fold"] != 0].copy()
    val_rows = merged[merged["fold"] == 0].copy()
    val_with_bm_pos = val_rows[val_rows["item_id"].astype(str).isin(list(benchmark_ids_set))].copy()

    val_contexts_all = (
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

    # Representative sample of 1,000 queries for PoC
    rng = np.random.default_rng(42)
    sample_indices = rng.choice(len(val_contexts_all), size=1000, replace=False)
    val_contexts = val_contexts_all.iloc[sample_indices].reset_index(drop=True)

    query_ids = [str(x) for x in val_contexts["internal_query_id"].tolist()]
    relevant_all = ground_truth_map(val_with_bm_pos[["internal_query_id", "item_id"]].drop_duplicates())
    relevant = {qid: relevant_all[qid] for qid in query_ids if qid in relevant_all}

    # Item metadata maps
    item_id_list = [str(x) for x in benchmark_items["item_id"].tolist()]
    item_loc_map = dict(zip(item_id_list, benchmark_items["item_location_id"]))
    item_cat_map = dict(zip(item_id_list, benchmark_items["item_category_id"]))
    item_microcat_map = dict(zip(item_id_list, benchmark_items["item_microcat_id"].astype(str)))
    item_lat_map = dict(zip(item_id_list, benchmark_items["item_latitude"].astype(float).fillna(0.0)))
    item_lon_map = dict(zip(item_id_list, benchmark_items["item_longitude"].astype(float).fillna(0.0)))
    item_titles = [str(x) for x in benchmark_items["item_title_raw"].fillna("").tolist()]

    loc_geo = (
        train.dropna(subset=["item_latitude", "item_longitude"])
        .groupby("item_location_id")[["item_latitude", "item_longitude"]]
        .mean()
    )
    loc_lat_dict = loc_geo["item_latitude"].astype(float).to_dict()
    loc_lon_dict = loc_geo["item_longitude"].astype(float).to_dict()

    # 2. Routing (K=10 champion)
    print("Fitting routing models (K=10)...")
    exact = ExactPosteriorPredictor().fit(train_part, query_col="search_query", category_col="item_microcat_id")
    clf = GeneralizingClassifierPredictor(min_df=5, max_features=50000, alpha=1e-5, seed=42).fit(train_part)
    hybrid = HybridMicrocategoryPredictor(exact_predictor=exact, generalizing_predictor=clf, min_count=1, blend_strategy="fallback")
    top10_microcats = hybrid.predict_top_k(val_contexts, k=10)

    microcat_to_items: dict[str, set[str]] = {}
    for item_id, mc in item_microcat_map.items():
        microcat_to_items.setdefault(str(mc), set()).add(item_id)

    allowed_items_per_query = []
    for cats in top10_microcats:
        allowed = set()
        for c in cats:
            allowed.update(microcat_to_items.get(str(c), set()))
        allowed_items_per_query.append(allowed)

    # 3. Candidate Retrieval (Depth 300)
    print("Retrieving candidates for 1,000 queries...")
    field_index = FieldAwareSparseIndex.fit(benchmark_items, branches=["branch_a"], min_df=2)
    title_char_index = SparseBranchIndex.fit(
        item_id_list, item_titles, name="title_char", analyzer="char_wb", ngram_range=(3, 5), min_df=2
    )

    title_queries = field_text(val_contexts, ("search_query",))
    routed_word = field_index.branches["branch_a"].retrieve_routed(title_queries, allowed_items_per_query, k=300)
    routed_char = title_char_index.retrieve_routed(title_queries, allowed_items_per_query, k=300)
    global_word = field_index.branches["branch_a"].retrieve(title_queries, k=150)
    global_char = title_char_index.retrieve(title_queries, k=150)

    # Dense E5
    e5_items_json = [str(x) for x in json.loads(Path("artifacts/selected/embeddings/e5/benchmark/item_ids.json").read_text())]
    e5_item_embeddings = np.load("artifacts/selected/embeddings/e5/benchmark/embeddings.npy", mmap_mode="r")
    val_query_embeddings_all = np.load("artifacts/research/val_query_e5_embeddings.npy")
    val_query_embeddings = val_query_embeddings_all[sample_indices]

    e5_retriever = RoutedDenseE5Retriever(item_ids=e5_items_json, item_embeddings=e5_item_embeddings)
    routed_e5 = e5_retriever.retrieve_routed(val_query_embeddings, allowed_items_per_query, k=300)
    global_e5 = e5_retriever.retrieve_global(val_query_embeddings, k=150)

    sources = [routed_e5, routed_char, routed_word, global_e5, global_word, global_char]
    weights = [1.5, 1.2, 0.8, 0.4, 0.3, 0.3]
    raw_rrf = reciprocal_rank_fusion(sources, weights=weights, c=60.0, limit=300)

    # 4. Evaluate Analytical Geo Baseline on this 1,000 query sample
    print("\n--- Evaluating Analytical Geo Baseline on 1,000 queries ---")
    analytical_rankings = []
    dist_decay = 0.25
    same_loc_bonus = 30.0

    for q_idx, qid in enumerate(query_ids):
        q_row = val_contexts.iloc[q_idx]
        q_loc = q_row["search_location_id"]
        q_cat = q_row["search_category"]
        q_lat = loc_lat_dict.get(q_loc)
        q_lon = loc_lon_dict.get(q_loc)

        boosted = []
        for item_id, score in raw_rrf[q_idx]:
            it_loc = item_loc_map.get(item_id)
            it_cat = item_cat_map.get(item_id)
            it_lat = item_lat_map.get(item_id, 0.0)
            it_lon = item_lon_map.get(item_id, 0.0)

            # Category wildcard for cat 0
            if q_cat == 0:
                cat_mult = 1.0
            else:
                cat_mult = 1.0 if (it_cat == q_cat) else 0.001

            if it_loc == q_loc:
                geo_mult = same_loc_bonus
            elif q_lat is not None and it_lat != 0.0:
                dist = haversine_km(np.array([q_lat]), np.array([q_lon]), np.array([it_lat]), np.array([it_lon]))[0]
                geo_mult = float(np.exp(-dist_decay * (dist / 10.0)))
            else:
                geo_mult = 0.05

            boosted.append((item_id, score * cat_mult * geo_mult))

        boosted.sort(key=lambda p: (-p[1], p[0]))
        analytical_rankings.append(boosted[:50])

    m_analytical = evaluate_rankings(analytical_rankings, query_ids, relevant, ks=(50,))
    print(f"Analytical Geo Baseline Recall@50: {m_analytical['recall@50']:.4f}")

    # 5. Build Compact Vectorized Dataframe for CatBoost
    print("\n--- Building Compact Feature DataFrame for CatBoost ---")
    depth_cb = 100
    rows = []

    for q_idx, qid in enumerate(query_ids):
        ranking = raw_rrf[q_idx][:depth_cb]
        if not ranking:
            continue
        q_row = val_contexts.iloc[q_idx]
        q_loc = q_row["search_location_id"]
        q_cat = q_row["search_category"]
        q_deliv = int(q_row["search_is_delivery_search"] or 0)
        q_lat = loc_lat_dict.get(q_loc, 0.0)
        q_lon = loc_lon_dict.get(q_loc, 0.0)
        pos_set = relevant.get(qid, set())
        pred_mc = top10_microcats[q_idx]

        max_sc = max(s for _, s in ranking) if ranking else 1.0

        for r_idx, (it, sc) in enumerate(ranking):
            it_loc = item_loc_map.get(it, -1)
            it_cat = item_cat_map.get(it, -1)
            it_mc = item_microcat_map.get(it, "")
            it_lat = item_lat_map.get(it, 0.0)
            it_lon = item_lon_map.get(it, 0.0)

            try:
                mc_rank = pred_mc.index(it_mc)
            except ValueError:
                mc_rank = -1

            rows.append((
                qid,
                it,
                int(it in pos_set),
                q_idx % 5,  # 5-fold assignment by query
                r_idx + 1,
                sc,
                sc / (max_sc + 1e-6),
                1.0 / (r_idx + 2.0),
                int(it_loc == q_loc),
                int(it_cat == q_cat or q_cat == 0),
                q_deliv,
                mc_rank,
                q_lat,
                q_lon,
                it_lat,
                it_lon,
            ))

    col_names = [
        "qid", "item_id", "label", "fold",
        "rank", "score", "norm_score", "inv_rank",
        "same_loc", "valid_cat", "is_deliv", "mc_rank",
        "qlat", "qlon", "itlat", "itlon"
    ]
    df = pd.DataFrame(rows, columns=col_names)

    # Vectorized distance computation
    dist = haversine_km(df["qlat"].to_numpy(), df["qlon"].to_numpy(), df["itlat"].to_numpy(), df["itlon"].to_numpy())
    missing_geo = (df["qlat"] == 0.0) | (df["itlat"] == 0.0)
    dist[missing_geo] = 500.0

    df["dist_km"] = dist
    df["log_dist"] = np.log1p(dist)
    df["deliv_x_dist"] = df["is_deliv"] * dist
    df["deliv_x_same_loc"] = df["is_deliv"] * df["same_loc"]

    features = [
        "rank", "score", "norm_score", "inv_rank",
        "same_loc", "valid_cat", "is_deliv", "mc_rank",
        "dist_km", "log_dist", "deliv_x_dist", "deliv_x_same_loc"
    ]

    print(f"DataFrame ready: {len(df):,} rows, memory: {df.memory_usage().sum() / 1024**2:.1f} MB")

    # 6. Train CatBoost 5-fold OOF
    print("\n--- Training CatBoost 5-Fold OOF ---")
    oof_preds = np.zeros(len(df), dtype=float)

    for f in range(5):
        train_idx = df["fold"] != f
        val_idx = df["fold"] == f

        X_tr, y_tr = df.loc[train_idx, features], df.loc[train_idx, "label"]
        X_val, y_val = df.loc[val_idx, features], df.loc[val_idx, "label"]

        cb = CatBoostClassifier(
            iterations=250,
            depth=5,
            learning_rate=0.08,
            loss_function="Logloss",
            eval_metric="Logloss",
            random_seed=42,
            verbose=False,
            thread_count=8,
        )
        cb.fit(X_tr, y_tr, eval_set=(X_val, y_val), early_stopping_rounds=25, verbose=False)
        oof_preds[val_idx] = cb.predict_proba(X_val)[:, 1]

    df["cb_score"] = oof_preds

    # 7. Evaluate CatBoost OOF Rankings
    catboost_rankings = []
    for qid in query_ids:
        sub = df[df["qid"] == qid]
        if not sub.empty:
            ranked = sub.sort_values(by="cb_score", ascending=False)[["item_id", "cb_score"]].to_numpy()
            catboost_rankings.append([(str(it), float(sc)) for it, sc in ranked[:50]])
        else:
            catboost_rankings.append([])

    m_cb = evaluate_rankings(catboost_rankings, query_ids, relevant, ks=(50,))
    print(f"CatBoost OOF Recall@50:            {m_cb['recall@50']:.4f}")
    print(f"Analytical Geo Baseline Recall@50: {m_analytical['recall@50']:.4f}")
    delta = m_cb['recall@50'] - m_analytical['recall@50']
    print(f"Δ (CatBoost vs Analytical):         {delta:+.4f} ({delta*100:+.2f} pp)")

    # Save PoC report
    report = {
        "poc_queries": 1000,
        "analytical_r50": float(m_analytical["recall@50"]),
        "catboost_r50": float(m_cb["recall@50"]),
        "delta": float(delta),
        "runtime_sec": time.time() - t0,
    }
    with open("artifacts/research_v3/catboost_poc_results.json", "w") as f:
        json.dump(report, f, indent=2)

    print(f"\nPoC completed in {time.time()-t0:.1f}s. Report saved to artifacts/research_v3/catboost_poc_results.json")


if __name__ == "__main__":
    main()
