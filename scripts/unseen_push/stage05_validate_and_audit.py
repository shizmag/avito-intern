"""Stage 05: Independent Validation, SHA-256 Checksum, Comparison vs Baseline, and Qualitative Unseen Audit.

1. Strict Contract Validation (2,452 queries, 50 items/query, no duplicates, all valid items).
2. SHA-256 checksum calculation.
3. Comparative analysis vs official baseline (0.646469):
   - Overlap mean & median
   - Changed top-1, changed >= 10, changed >= 25
   - Broken down into Seen vs Unseen queries
4. Qualitative Unseen Audit on 50 unseen benchmark queries.
5. Saves artifacts/unseen_push/final/manifest.json and qualitative_unseen_audit_50.json.
"""

from __future__ import annotations

import csv
import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def compute_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    t0 = time.time()
    print("=" * 80)
    print("STAGE 05: INDEPENDENT VALIDATION, CHECKSUM, AND COMPARATIVE AUDIT")
    print("=" * 80)

    sub_path = Path("answer.csv")
    final_dir = Path("artifacts/unseen_push/final")
    queries_path = Path("data/benchmark_queries.parquet")
    items_path = Path("data/benchmark_items.parquet")
    train_path = Path("data/train.parquet")
    baseline_sub_path = Path("artifacts/submissions/official_0646469/answer.csv")

    df_queries = pd.read_parquet(queries_path)
    df_items = pd.read_parquet(items_path)
    train = pd.read_parquet(train_path)

    expected_query_ids = [str(x) for x in df_queries["query_id"].tolist()]
    expected_query_set = set(expected_query_ids)
    valid_item_set = {str(x) for x in df_items["item_id"].tolist()}

    # 1. Independent Contract Validation
    print("\n[1/4] Running independent submission contract validation...")
    if not sub_path.is_file():
        raise FileNotFoundError(f"{sub_path} does not exist")

    with sub_path.open("r", encoding="utf-8") as f:
        header_line = f.readline().rstrip("\r\n")
        if header_line != "query_id,answer":
            raise ValueError(f"Invalid header line: {header_line!r}, expected 'query_id,answer'")

    seen_query_ids: list[str] = []
    query_answer_counts: list[int] = []
    has_duplicate_items = False
    has_unknown_items = False
    new_answers_map: dict[str, list[str]] = {}

    with sub_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        _ = next(reader)
        for row_idx, row in enumerate(reader):
            if len(row) != 2:
                raise ValueError(f"Row {row_idx + 2} has {len(row)} columns, expected 2")
            qid, ans_str = row
            qid_str = str(qid)
            seen_query_ids.append(qid_str)
            items = ans_str.strip().split()
            query_answer_counts.append(len(items))
            new_answers_map[qid_str] = items

            if len(items) != len(set(items)):
                has_duplicate_items = True

            for it in items:
                if it not in valid_item_set:
                    has_unknown_items = True

    is_queries_match = (set(seen_query_ids) == expected_query_set) and (len(seen_query_ids) == len(expected_query_ids))
    all_50_items = all(c == 50 for c in query_answer_counts)
    sha256 = compute_sha256(sub_path)

    val_res = {
        "rows_count": len(seen_query_ids),
        "expected_queries_count": len(expected_query_ids),
        "all_queries_present_once": is_queries_match,
        "all_answers_exactly_50_items": all_50_items,
        "min_items_per_query": min(query_answer_counts) if query_answer_counts else 0,
        "max_items_per_query": max(query_answer_counts) if query_answer_counts else 0,
        "no_duplicate_items": not has_duplicate_items,
        "no_unknown_items": not has_unknown_items,
        "sha256": sha256,
        "status": "PASS" if (is_queries_match and all_50_items and not has_duplicate_items and not has_unknown_items) else "FAIL",
    }

    print("Contract Validation Results:")
    for k, v in val_res.items():
        print(f"  {k}: {v}")

    if val_res["status"] != "PASS":
        raise ValueError(f"Submission contract validation FAILED: {val_res}")

    # 2. Identification of Seen vs Unseen Benchmark Queries
    train_queries_set = set(
        train["search_query"].dropna().astype(str).str.lower().str.replace("ё", "е").str.strip()
    )
    bm_q_norm = (
        df_queries["search_query"].fillna("").astype(str).str.lower().str.replace("ё", "е").str.strip()
    )
    is_bm_seen = bm_q_norm.isin(train_queries_set).to_numpy()
    is_bm_unseen = ~is_bm_seen

    n_bm_seen = int(is_bm_seen.sum())
    n_bm_unseen = int(is_bm_unseen.sum())
    print(f"\n[2/4] Benchmark Query Distribution: {len(df_queries)} total (Seen: {n_bm_seen}, UNSEEN: {n_bm_unseen} — {n_bm_unseen/len(df_queries):.1%})")

    # 3. Comparative Analysis against Frozen Official Baseline (0.646469)
    print("\n[3/4] Comparing new submission against frozen official baseline (0.646469)...")
    base_df = pd.read_csv(baseline_sub_path, dtype=str).set_index("query_id")
    base_answers_map = {str(qid): row["answer"].split() for qid, row in base_df.iterrows()}

    overlaps_all: list[int] = []
    top1_changed_all: list[bool] = []
    diff_ge10_all: list[bool] = []
    diff_ge25_all: list[bool] = []

    for qid in expected_query_ids:
        new_items = new_answers_map.get(qid, [])
        base_items = base_answers_map.get(qid, [])
        ov = len(set(new_items) & set(base_items))
        overlaps_all.append(ov)
        top1_changed_all.append(new_items[0] != base_items[0] if (new_items and base_items) else True)
        diff_ge10_all.append(ov <= 40)
        diff_ge25_all.append(ov <= 25)

    arr_ov = np.array(overlaps_all)
    arr_t1 = np.array(top1_changed_all)
    arr_d10 = np.array(diff_ge10_all)
    arr_d25 = np.array(diff_ge25_all)

    comp_overall = {
        "mean_overlap_top50": float(arr_ov.mean()),
        "median_overlap_top50": float(np.median(arr_ov)),
        "fraction_top1_changed": float(arr_t1.mean()),
        "fraction_changed_ge10": float(arr_d10.mean()),
        "fraction_changed_ge25": float(arr_d25.mean()),
    }
    comp_seen = {
        "mean_overlap_top50": float(arr_ov[is_bm_seen].mean()),
        "median_overlap_top50": float(np.median(arr_ov[is_bm_seen])),
        "fraction_top1_changed": float(arr_t1[is_bm_seen].mean()),
        "fraction_changed_ge10": float(arr_d10[is_bm_seen].mean()),
        "fraction_changed_ge25": float(arr_d25[is_bm_seen].mean()),
    }
    comp_unseen = {
        "mean_overlap_top50": float(arr_ov[is_bm_unseen].mean()),
        "median_overlap_top50": float(np.median(arr_ov[is_bm_unseen])),
        "fraction_top1_changed": float(arr_t1[is_bm_unseen].mean()),
        "fraction_changed_ge10": float(arr_d10[is_bm_unseen].mean()),
        "fraction_changed_ge25": float(arr_d25[is_bm_unseen].mean()),
    }

    print("Overall Comparison:")
    for k, v in comp_overall.items():
        print(f"  {k}: {v:.4f}")
    print("\nSEEN Queries Comparison:")
    for k, v in comp_seen.items():
        print(f"  {k}: {v:.4f}")
    print("\nUNSEEN Queries Comparison (Critical Sanity Check):")
    for k, v in comp_unseen.items():
        print(f"  {k}: {v:.4f}")

    # 4. Qualitative Unseen Audit on 50 Benchmark Queries
    print("\n[4/4] Generating Qualitative Unseen Audit for 50 Unseen Benchmark Queries...")
    with open("artifacts/unseen_push/benchmark/top10_microcats.json", "r", encoding="utf-8") as f:
        top10_microcats_list = json.load(f)

    item_title_map = dict(zip(df_items["item_id"].astype(str), df_items["item_title_raw"].fillna("")))
    item_loc_map = dict(zip(df_items["item_id"].astype(str), df_items["item_location_id"]))
    item_microcat_map = dict(zip(df_items["item_id"].astype(str), df_items["item_microcat_id"].astype(str)))

    unseen_idx = np.where(is_bm_unseen)[0]
    audit_indices = unseen_idx[:50]
    audit_records: list[dict[str, Any]] = []

    for idx in audit_indices:
        row = df_queries.iloc[idx]
        qid = str(row["query_id"])
        new_top10 = new_answers_map[qid][:10]
        old_top10 = base_answers_map.get(qid, [])[:10]
        overlap_10 = len(set(new_top10) & set(old_top10))

        audit_records.append({
            "query_id": qid,
            "search_query": row["search_query"],
            "search_infm_params_text": row["search_infm_params_text"],
            "search_category": int(row["search_category"]),
            "search_location_id": int(row["search_location_id"]),
            "predicted_top10_microcats": top10_microcats_list[idx],
            "top10_overlap_with_baseline": overlap_10,
            "new_top5_items": [
                {
                    "item_id": it,
                    "title": item_title_map.get(it, ""),
                    "location_id": item_loc_map.get(it, -1),
                    "microcat_id": item_microcat_map.get(it, ""),
                }
                for it in new_top10[:5]
            ],
            "old_top5_items": [
                {
                    "item_id": it,
                    "title": item_title_map.get(it, ""),
                    "location_id": item_loc_map.get(it, -1),
                    "microcat_id": item_microcat_map.get(it, ""),
                }
                for it in old_top10[:5]
            ],
        })

    with (final_dir / "qualitative_unseen_audit_50.json").open("w", encoding="utf-8") as f:
        json.dump(audit_records, f, indent=2, ensure_ascii=False)

    # 5. Manifest
    manifest = {
        "pipeline": "champion_unseen_push_a_plus_b",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "output_path": str(sub_path),
        "sha256": sha256,
        "validation_contract": val_res,
        "comparison_overall": comp_overall,
        "comparison_seen": comp_seen,
        "comparison_unseen": comp_unseen,
        "qualitative_audit_queries_count": len(audit_records),
        "runtime_sec": time.time() - t0,
    }
    with (final_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nAll artifacts verified and saved in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
