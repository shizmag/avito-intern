"""Qualitative audit of 50 representative benchmark queries for the Champion Pipeline.

Covers:
- Delivery + distant results
- Small-city queries
- Low routing confidence / unseen queries
- Numeric queries
- Category 0 (wildcard) queries
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def main() -> None:
    print("Generating Qualitative Audit of 50 Benchmark Queries...")

    bq = pd.read_parquet("data/benchmark_queries.parquet")
    bi = pd.read_parquet("data/benchmark_items.parquet")
    sub = pd.read_csv("answer.csv")

    sub_map = dict(zip(sub["query_id"].astype(str), sub["answer"].str.split()))

    item_id_list = [str(x) for x in bi["item_id"].tolist()]
    item_title_map = dict(zip(item_id_list, bi["item_title_raw"].fillna("").astype(str)))
    item_loc_map = dict(zip(item_id_list, bi["item_location_id"].astype(str)))
    item_cat_map = dict(zip(item_id_list, bi["item_category_id"].astype(str)))
    item_mc_map = dict(zip(item_id_list, bi["item_microcat_id"].astype(str)))

    # Select 50 interesting queries:
    # 1. Cat 0 queries (10)
    cat0_indices = bq[bq["search_category"] == 0].index[:10].tolist()
    # 2. Queries with numbers (10)
    num_indices = bq[bq["search_query"].fillna("").str.contains(r"\d", regex=True)].index[:10].tolist()
    # 3. Small-city queries (10)
    loc_counts = bq["search_location_id"].value_counts()
    small_locs = set(loc_counts[loc_counts == 1].index)
    small_city_indices = bq[bq["search_location_id"].isin(small_locs)].index[:10].tolist()
    # 4. Long queries >= 4 words (10)
    long_q_indices = bq[bq["search_query"].fillna("").str.split().str.len() >= 4].index[:10].tolist()
    # 5. Short queries <= 2 words (10)
    short_q_indices = bq[bq["search_query"].fillna("").str.split().str.len() <= 2].index[:10].tolist()

    selected_indices = sorted(set(cat0_indices + num_indices + small_city_indices + long_q_indices + short_q_indices))
    if len(selected_indices) < 50:
        remaining = [i for i in range(len(bq)) if i not in selected_indices]
        selected_indices.extend(remaining[: 50 - len(selected_indices)])
    selected_indices = selected_indices[:50]
    print(f"Selected {len(selected_indices)} queries for qualitative inspection.")

    audit_records: list[dict[str, Any]] = []

    for idx in selected_indices:
        row = bq.iloc[idx]
        qid = str(row["query_id"])
        top50 = sub_map.get(qid, [])
        top10 = top50[:10]

        top10_details = [
            {
                "item_id": it,
                "title": item_title_map.get(it, ""),
                "item_location_id": item_loc_map.get(it, ""),
                "item_category_id": item_cat_map.get(it, ""),
                "item_microcat_id": item_mc_map.get(it, ""),
                "same_location": bool(item_loc_map.get(it) == str(row["search_location_id"])),
            }
            for it in top10
        ]

        audit_records.append({
            "query_id": qid,
            "query": str(row["search_query"]),
            "filters": str(row.get("search_infm_params_text", "")),
            "search_category": int(row.get("search_category", 114)),
            "search_location_id": str(row.get("search_location_id", "")),
            "is_delivery": int(row.get("search_is_delivery_search", 0) or 0),
            "top10_results": top10_details,
        })

    out_path = Path("artifacts/research_v3/qualitative_benchmark_audit_50.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(audit_records, f, indent=2, ensure_ascii=False)

    print(f"Qualitative audit saved to {out_path}")


if __name__ == "__main__":
    main()
