"""Analyze distribution shift and generalization gap between canonical validation and benchmark queries."""

import numpy as np
import pandas as pd
from avito_candidate_generation.research import context_frame, context_folds, ground_truth_map

def main():
    print("=" * 70)
    print("ANALYSIS OF OFFLINE VS BENCHMARK / OFFICIAL DISTRIBUTION SHIFT")
    print("=" * 70)

    train = pd.read_parquet("data/train.parquet")
    benchmark_items = pd.read_parquet("data/benchmark_items.parquet")
    benchmark_queries = pd.read_parquet("data/benchmark_queries.parquet")
    benchmark_ids_set = set(benchmark_items["item_id"].astype(str))

    work = context_frame(train)
    folds = context_folds(work["internal_query_id"].astype(str), seed=42, folds=5)
    merged = pd.merge(work, folds, on="internal_query_id")

    train_part = merged[merged["fold"] != 0].copy()
    val_rows = merged[merged["fold"] == 0].copy()
    val_with_bm_pos = val_rows[val_rows["item_id"].astype(str).isin(list(benchmark_ids_set))].copy()

    val_contexts = val_with_bm_pos[["search_query", "search_location_id", "search_is_delivery_search", "search_infm_params_text", "search_category", "internal_query_id"]].drop_duplicates().reset_index(drop=True)

    print(f"Total benchmark queries: {len(benchmark_queries)}")
    print(f"Total canonical validation queries: {len(val_contexts)}")

    # 1. Query text seen vs unseen
    train_queries_set = set(train_part["search_query"].dropna().astype(str).str.lower().str.strip())
    
    val_seen = val_contexts["search_query"].dropna().astype(str).str.lower().str.strip().isin(train_queries_set)
    bm_seen = benchmark_queries["search_query"].dropna().astype(str).str.lower().str.strip().isin(train_queries_set)

    print("\n--- 1. Query Text Novelty ---")
    print(f"Canonical Val seen queries: {val_seen.mean():.4f} ({val_seen.sum()} / {len(val_seen)})")
    print(f"Benchmark seen queries:     {bm_seen.mean():.4f} ({bm_seen.sum()} / {len(bm_seen)})")
    print(f"Benchmark UNSEEN fraction:  {(~bm_seen).mean():.4f} (Val unseen: {(~val_seen).mean():.4f})")

    # 2. Delivery search fraction
    print("\n--- 2. Delivery Search Fraction ---")
    val_delivery = val_contexts["search_is_delivery_search"].fillna(0).astype(int)
    bm_delivery = benchmark_queries["search_is_delivery_search"].fillna(0).astype(int)
    print(f"Canonical Val delivery fraction: {val_delivery.mean():.4f} ({val_delivery.sum()} / {len(val_delivery)})")
    print(f"Benchmark delivery fraction:     {bm_delivery.mean():.4f} ({bm_delivery.sum()} / {len(bm_delivery)})")

    # 3. Filters / params presence
    print("\n--- 3. Filters / Params Text Presence ---")
    val_params = val_contexts["search_infm_params_text"].fillna("").str.strip().ne("")
    bm_params = benchmark_queries["search_infm_params_text"].fillna("").str.strip().ne("")
    print(f"Canonical Val params present: {val_params.mean():.4f} ({val_params.sum()} / {len(val_params)})")
    print(f"Benchmark params present:     {bm_params.mean():.4f} ({bm_params.sum()} / {len(bm_params)})")

    # 4. Search Category distribution
    print("\n--- 4. Search Category Distribution ---")
    val_cat_dist = val_contexts["search_category"].value_counts(normalize=True).to_dict()
    bm_cat_dist = benchmark_queries["search_category"].value_counts(normalize=True).to_dict()
    all_cats = sorted(set(list(val_cat_dist.keys()) + list(bm_cat_dist.keys())))
    for c in all_cats:
        v_pct = val_cat_dist.get(c, 0.0) * 100
        b_pct = bm_cat_dist.get(c, 0.0) * 100
        print(f"  Category {c:<6}: Val = {v_pct:5.2f}%, BM = {b_pct:5.2f}% (diff: {b_pct - v_pct:+5.2f} pp)")

    # 5. Search Location Seen in Train
    print("\n--- 5. Search Location Seen in Train ---")
    train_locations = set(train_part["search_location_id"].unique())
    val_loc_seen = val_contexts["search_location_id"].isin(train_locations)
    bm_loc_seen = benchmark_queries["search_location_id"].isin(train_locations)
    print(f"Canonical Val location seen: {val_loc_seen.mean():.4f}")
    print(f"Benchmark location seen:     {bm_loc_seen.mean():.4f}")

    # 6. Query length
    print("\n--- 6. Query Length (words) ---")
    val_q_len = val_contexts["search_query"].fillna("").str.split().str.len()
    bm_q_len = benchmark_queries["search_query"].fillna("").str.split().str.len()
    print(f"Canonical Val word len: mean={val_q_len.mean():.2f}, median={val_q_len.median():.1f}, <=2 words: {(val_q_len <= 2).mean():.4f}")
    print(f"Benchmark word len:     mean={bm_q_len.mean():.2f}, median={bm_q_len.median():.1f}, <=2 words: {(bm_q_len <= 2).mean():.4f}")

    # 7. Ground truth analysis on canonical validation: Same location vs Delivery interaction
    print("\n--- 7. Ground Truth Analysis on Canonical Validation ---")
    item_id_list = [str(x) for x in benchmark_items["item_id"].tolist()]
    item_loc_map = dict(zip(item_id_list, benchmark_items["item_location_id"]))
    item_cat_map = dict(zip(item_id_list, benchmark_items["item_category_id"]))
    query_ids = [str(x) for x in val_contexts["internal_query_id"].tolist()]
    query_loc_map = dict(zip(query_ids, val_contexts["search_location_id"]))
    query_deliv_map = dict(zip(query_ids, val_contexts["search_is_delivery_search"]))
    relevant = ground_truth_map(val_with_bm_pos[["internal_query_id", "item_id"]].drop_duplicates())

    same_loc_all = []
    same_loc_deliv = []
    same_loc_nondeliv = []

    for qid, pos_items in relevant.items():
        qloc = query_loc_map[qid]
        qdeliv = query_deliv_map[qid]
        for it in pos_items:
            is_same = (item_loc_map.get(it) == qloc)
            same_loc_all.append(is_same)
            if qdeliv == 1:
                same_loc_deliv.append(is_same)
            else:
                same_loc_nondeliv.append(is_same)

    print(f"Overall positives same location:      {np.mean(same_loc_all):.4f} ({sum(same_loc_all)} / {len(same_loc_all)})")
    print(f"Non-delivery positives same location: {np.mean(same_loc_nondeliv):.4f} ({sum(same_loc_nondeliv)} / {len(same_loc_nondeliv)})")
    print(f"Delivery positives same location:     {np.mean(same_loc_deliv):.4f} ({sum(same_loc_deliv)} / {len(same_loc_deliv)})")
    print(f"Cross-location positive fraction in Delivery:     {1.0 - np.mean(same_loc_deliv):.4f}")
    print(f"Cross-location positive fraction in Non-Delivery: {1.0 - np.mean(same_loc_nondeliv):.4f}")

if __name__ == "__main__":
    main()
