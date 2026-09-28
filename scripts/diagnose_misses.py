"""Diagnose misses in Geo-aware candidate generation."""

import gc
import json
import numpy as np
import pandas as pd
from avito_candidate_generation.research import context_frame, context_folds, ground_truth_map

def main():
    train = pd.read_parquet("data/train.parquet")
    benchmark_items = pd.read_parquet("data/benchmark_items.parquet")
    benchmark_ids_set = set(benchmark_items["item_id"].astype(str))

    work = context_frame(train)
    folds = context_folds(work["internal_query_id"].astype(str), seed=42, folds=5)
    merged = pd.merge(work, folds, on="internal_query_id")

    val_rows = merged[merged["fold"] == 0]
    val_with_bm_pos = val_rows[val_rows["item_id"].astype(str).isin(list(benchmark_ids_set))].copy()

    val_contexts = val_with_bm_pos[["search_query", "search_location_id", "search_is_delivery_search", "search_infm_params_text", "search_category", "internal_query_id"]].drop_duplicates().reset_index(drop=True)
    query_ids = [str(x) for x in val_contexts["internal_query_id"].tolist()]
    relevant = ground_truth_map(val_with_bm_pos[["internal_query_id", "item_id"]].drop_duplicates())

    item_id_list = [str(x) for x in benchmark_items["item_id"].tolist()]
    item_loc_map = dict(zip(item_id_list, benchmark_items["item_location_id"]))
    item_cat_map = dict(zip(item_id_list, benchmark_items["item_category_id"]))
    item_microcat_map = dict(zip(item_id_list, benchmark_items["item_microcat_id"]))

    query_loc_map = dict(zip(query_ids, val_contexts["search_location_id"]))
    query_cat_map = dict(zip(query_ids, val_contexts["search_category"]))

    # Check for all true positives in val:
    same_loc = []
    same_cat = []
    for qid, pos_items in relevant.items():
        qloc = query_loc_map[qid]
        qcat = query_cat_map[qid]
        for it in pos_items:
            same_loc.append(item_loc_map.get(it) == qloc)
            same_cat.append(item_cat_map.get(it) == qcat)

    print("Val GT pairs count:", len(same_loc))
    print("Same location fraction:", np.mean(same_loc))
    print("Same category fraction:", np.mean(same_cat))

if __name__ == "__main__":
    main()
