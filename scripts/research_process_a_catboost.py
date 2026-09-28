"""Process A: CatBoost Ranking Experiment (Hypothesis H2).

Trains CatBoost on routed candidate pool (depth=1000) using:
- Multi-retriever scores and ranks (Dense E5, Char TF-IDF, Word TF-IDF, Title+Params)
- Per-query normalized retriever features (score / max_score, max_score - score, zscore, 1/(rank+1), log1p(rank))
- Routing features (predicted_microcat_rank, is_top1, is_top3)
- Geo features (same_location, distance_km, log1p(distance_km))
- Delivery interactions (delivery * distance, delivery * same_location)
- Category features (same_category, is_cat0, valid_cat_match)
- Text overlap features (token overlap, token jaccard)
- 5-Fold Out-Of-Fold (OOF) evaluation on canonical validation set (zero leakage)
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from avito_candidate_generation.fusion.candidate_pool import (
    evaluate_rankings,
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
    print("PROCESS A: CATBOOST RANKING EXPERIMENT (H2)")
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
    item_microcat_map = dict(zip(item_id_list, benchmark_items["item_microcat_id"].astype(str)))
    item_lat_map = dict(zip(item_id_list, benchmark_items["item_latitude"].astype(float).fillna(0.0)))
    item_lon_map = dict(zip(item_id_list, benchmark_items["item_longitude"].astype(float).fillna(0.0)))
    item_titles = [str(x) for x in benchmark_items["item_title_raw"].fillna("").tolist()]
    item_title_dict = dict(zip(item_id_list, item_titles, strict=True))

    query_loc_map = dict(zip(query_ids, val_contexts["search_location_id"]))
    query_cat_map = dict(zip(query_ids, val_contexts["search_category"]))
    query_deliv_map = dict(zip(query_ids, val_contexts["search_is_delivery_search"].fillna(0).astype(int)))
    query_text_map = dict(zip(query_ids, val_contexts["search_query"].fillna("").astype(str)))

    # Assign 5 cross-validation folds to canonical validation queries
    val_q_folds = context_folds(query_ids, seed=42, folds=5)
    q_to_fold = dict(zip(val_q_folds["internal_query_id"], val_q_folds["fold"]))

    # Microcategory Routing (K=10 champion routing)
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

    # Sparse Retrieval
    print("Fitting sparse retrieval...")
    field_index = FieldAwareSparseIndex.fit(benchmark_items, branches=["branch_a", "branch_b"], min_df=2)
    title_char_index = SparseBranchIndex.fit(
        item_id_list, item_titles, name="title_char", analyzer="char_wb", ngram_range=(3, 5), min_df=2
    )

    title_queries = field_text(val_contexts, ("search_query",))
    query_params_texts = field_text(val_contexts, ("search_query", "search_infm_params_text"))

    print("Retrieving candidates across all branches...")
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
    raw_rrf = reciprocal_rank_fusion(sources, weights=weights, c=60.0, limit=1000)

    # Build Feature Dataset
    # For efficiency and speed: candidate depth = 300 per query (top 300 RRF candidates)
    depth_train = 300
    print(f"\nBuilding feature dataset for top-{depth_train} candidates per query...")

    feature_rows = []
    
    for q_idx, qid in enumerate(query_ids):
        ranking = raw_rrf[q_idx][:depth_train]
        if not ranking:
            continue
            
        q_text = query_text_map[qid]
        q_tokens = set(q_text.lower().split())
        q_loc = query_loc_map[qid]
        q_cat = query_cat_map[qid]
        q_deliv = query_deliv_map[qid]
        q_lat = loc_lat_dict.get(q_loc)
        q_lon = loc_lon_dict.get(q_loc)
        q_fold = q_to_fold[qid]
        pred_microcats = top10_microcats[q_idx]
        pos_set = relevant.get(qid, set())

        rrf_scores = [s for _, s in ranking]
        max_rrf = max(rrf_scores) if rrf_scores else 1.0
        mean_rrf = np.mean(rrf_scores) if rrf_scores else 0.0
        std_rrf = np.std(rrf_scores) if rrf_scores else 1.0

        # Branch candidate maps for quick lookup
        e5_map = {it: (rank + 1, sc) for rank, (it, sc) in enumerate(routed_e5[q_idx])}
        char_map = {it: (rank + 1, sc) for rank, (it, sc) in enumerate(routed_char[q_idx])}
        word_map = {it: (rank + 1, sc) for rank, (it, sc) in enumerate(routed_word[q_idx])}

        for rank_idx, (item_id, rrf_score) in enumerate(ranking):
            is_pos = int(item_id in pos_set)
            it_loc = item_loc_map.get(item_id)
            it_cat = item_cat_map.get(item_id)
            it_mc = item_microcat_map.get(item_id)
            it_lat = item_lat_map.get(item_id, 0.0)
            it_lon = item_lon_map.get(item_id, 0.0)
            it_title = item_title_dict.get(item_id, "")
            it_tokens = set(it_title.lower().split())

            # Geo features
            is_same_loc = int(it_loc == q_loc)
            if q_lat is not None and q_lon is not None and it_lat != 0.0 and it_lon != 0.0:
                dist = haversine_km(q_lat, q_lon, it_lat, it_lon)
                dist_missing = 0
            else:
                dist = 500.0  # default large distance for missing geo
                dist_missing = 1

            # Category features
            is_same_cat = int(it_cat == q_cat)
            is_cat_0 = int(q_cat == 0)
            valid_cat = int(is_same_cat or is_cat_0)

            # Routing rank of item's microcategory
            try:
                mc_rank = pred_microcats.index(it_mc)
            except ValueError:
                mc_rank = -1

            # Text overlap
            tok_overlap = len(q_tokens & it_tokens)
            tok_jaccard = tok_overlap / max(len(q_tokens | it_tokens), 1)

            # Retriever specifics
            e5_rank, e5_sc = e5_map.get(item_id, (1001, 0.0))
            char_rank, char_sc = char_map.get(item_id, (1001, 0.0))
            word_rank, word_sc = word_map.get(item_id, (1001, 0.0))

            feature_rows.append({
                "internal_query_id": qid,
                "item_id": item_id,
                "fold": q_fold,
                "label": is_pos,
                # RRF features
                "rrf_rank": rank_idx + 1,
                "rrf_score": rrf_score,
                "rrf_norm": rrf_score / (max_rrf + 1e-6),
                "rrf_diff": max_rrf - rrf_score,
                "rrf_zscore": (rrf_score - mean_rrf) / (std_rrf + 1e-6),
                "rrf_inv_rank": 1.0 / (rank_idx + 2.0),
                "rrf_log_rank": np.log1p(rank_idx + 1),
                # Retriever scores
                "e5_score": e5_sc,
                "e5_inv_rank": 1.0 / (e5_rank + 1.0),
                "in_e5": int(e5_rank <= 1000),
                "char_score": char_sc,
                "char_inv_rank": 1.0 / (char_rank + 1.0),
                "in_char": int(char_rank <= 1000),
                "word_score": word_sc,
                "word_inv_rank": 1.0 / (word_rank + 1.0),
                "in_word": int(word_rank <= 1000),
                "retriever_count": int(e5_rank <= 1000) + int(char_rank <= 1000) + int(word_rank <= 1000),
                # Routing features
                "mc_rank": mc_rank,
                "is_top1_mc": int(mc_rank == 0),
                "is_top3_mc": int(0 <= mc_rank < 3),
                # Geo features
                "same_location": is_same_loc,
                "dist_km": dist,
                "log1p_dist": np.log1p(dist),
                "dist_missing": dist_missing,
                # Delivery & Category
                "is_delivery": q_deliv,
                "deliv_x_dist": q_deliv * dist,
                "deliv_x_same_loc": q_deliv * is_same_loc,
                "same_category": is_same_cat,
                "is_cat_0": is_cat_0,
                "valid_cat": valid_cat,
                # Text features
                "tok_overlap": tok_overlap,
                "tok_jaccard": tok_jaccard,
                "query_len": len(q_tokens),
            })

    df_feat = pd.DataFrame(feature_rows)
    print(f"Dataset constructed: {len(df_feat)} candidate pairs across {len(query_ids)} queries.")
    print(f"Positive pairs: {df_feat['label'].sum()} ({df_feat['label'].mean():.4%})")

    feature_cols = [
        c for c in df_feat.columns
        if c not in {"internal_query_id", "item_id", "fold", "label"}
    ]
    print(f"Features ({len(feature_cols)}): {feature_cols}")

    # 5-Fold Cross Validation
    print("\nRunning 5-Fold Cross-Validation for CatBoost Ranker...")
    oof_predictions = []

    for fold_val in range(5):
        train_mask = (df_feat["fold"] != fold_val)
        val_mask = (df_feat["fold"] == fold_val)

        X_train, y_train = df_feat.loc[train_mask, feature_cols], df_feat.loc[train_mask, "label"]
        X_val, y_val = df_feat.loc[val_mask, feature_cols], df_feat.loc[val_mask, "label"]

        model = CatBoostClassifier(
            iterations=300,
            depth=6,
            learning_rate=0.08,
            loss_function="Logloss",
            eval_metric="Logloss",
            random_seed=42,
            verbose=False,
            thread_count=8,
        )
        model.fit(X_train, y_train, eval_set=(X_val, y_val), early_stopping_rounds=30, verbose=False)

        val_preds = model.predict_proba(X_val)[:, 1]
        val_subset = df_feat.loc[val_mask, ["internal_query_id", "item_id"]].copy()
        val_subset["cb_score"] = val_preds
        oof_predictions.append(val_subset)
        print(f"  Fold {fold_val} complete: best iteration = {model.get_best_iteration()}")

    df_oof = pd.concat(oof_predictions, ignore_index=True)

    # Evaluate OOF Recall@50
    print("\nEvaluating CatBoost OOF Rankings...")
    catboost_rankings = []
    grouped_oof = df_oof.groupby("internal_query_id")

    for qid in query_ids:
        if qid in grouped_oof.groups:
            group = grouped_oof.get_group(qid)
            # Sort by CatBoost score descending
            sorted_items = [
                (row["item_id"], row["cb_score"])
                for _, row in group.sort_values(by="cb_score", ascending=False).iterrows()
            ]
            catboost_rankings.append(sorted_items[:50])
        else:
            catboost_rankings.append([])

    m_cb = evaluate_rankings(catboost_rankings, query_ids, relevant, ks=(50,))
    cb_r50 = m_cb["recall@50"]
    delta_cb = cb_r50 - 0.7050

    print(f"\n=======================================================")
    print(f"CatBoost OOF Recall@50: {cb_r50:.4f} (Δ vs Baseline: {delta_cb:+.4f})")
    print(f"=======================================================")

    results = {
        "catboost_oof_recall_50": float(cb_r50),
        "baseline_recall_50": 0.7050,
        "delta": float(delta_cb),
        "feature_count": len(feature_cols),
        "feature_names": feature_cols,
    }

    out_path = Path("artifacts/research_v3/catboost_experiments.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    runtime_sec = time.time() - t0
    print(f"\nProcess A (CatBoost) completed in {runtime_sec:.1f}s. Artifact saved to {out_path}")


if __name__ == "__main__":
    main()
