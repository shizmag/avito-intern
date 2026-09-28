"""Generate final production benchmark submission with Geo-Aware Microcategory-Routed Pipeline.

Architecture:
1. Microcategory Routing: Hybrid (Exact Posterior + TF-IDF SGDClassifier) -> Top-5 Microcategories.
2. Candidate Retrieval (depth=1000):
   - Routed Dense E5 (k=1000)
   - Routed Char TF-IDF (k=1000)
   - Routed Word TF-IDF (k=1000)
   - Routed Title+Params Char (k=1000)
   - Global Dense E5 Fallback (k=300)
   - Global Word TF-IDF Fallback (k=300)
   - Global Char TF-IDF Fallback (k=300)
3. Fusion: Weighted RRF (c=60.0).
4. Local-First Hyperlocal Cascade Ranking:
   - Tier 1: Same location (item_location_id == search_location_id) & same category (item_category_id == search_category).
   - Tier 2: Nearby locations (dist <= 50 km) & same category.
   - Tier 3: Broader region & same category.
   - Tier 4: Global fallback.
   - Produces exactly 50 unique item IDs per query.
5. Verification, Checksum, Backup, and Qualitative Audit.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import shutil
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from avito_candidate_generation.fusion.candidate_pool import reciprocal_rank_fusion
from avito_candidate_generation.research import field_text
from avito_candidate_generation.retrievers.routed_e5 import (
    E5QueryEncoder,
    RoutedDenseE5Retriever,
)
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


def compute_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_submission_file(
    submission_path: str | Path,
    expected_queries_path: str | Path,
    expected_items_path: str | Path,
) -> dict[str, Any]:
    """Independent submission validator following competition contract."""
    sub_path = Path(submission_path)
    if not sub_path.is_file():
        raise FileNotFoundError(f"submission file {submission_path} does not exist")

    with sub_path.open("r", encoding="utf-8") as f:
        header_line = f.readline().rstrip("\r\n")
        if header_line != "query_id,answer":
            raise ValueError(f"invalid header line: {header_line!r}, expected 'query_id,answer'")

    df_queries = pd.read_parquet(expected_queries_path)
    expected_query_ids = [str(x) for x in df_queries["query_id"].tolist()]
    expected_query_set = set(expected_query_ids)

    df_items = pd.read_parquet(expected_items_path)
    valid_item_set = {str(x) for x in df_items["item_id"].tolist()}

    seen_query_ids: list[str] = []
    rows_count = 0
    query_answer_counts: list[int] = []
    has_duplicate_items = False
    has_unknown_items = False

    with sub_path.open("r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        if header != ["query_id", "answer"]:
            raise ValueError(f"unexpected CSV columns: {header}")
        for row in reader:
            rows_count += 1
            if len(row) != 2:
                raise ValueError(f"row {rows_count} does not have exactly 2 columns: {row}")
            qid, ans = row[0], row[1]
            seen_query_ids.append(qid)
            item_tokens = ans.strip().split()
            count = len(item_tokens)
            query_answer_counts.append(count)
            if len(set(item_tokens)) != count:
                has_duplicate_items = True
            for it in item_tokens:
                if it not in valid_item_set:
                    has_unknown_items = True

    if rows_count != len(expected_query_ids):
        raise ValueError(
            f"row count mismatch: got {rows_count}, expected {len(expected_query_ids)}"
        )
    if len(set(seen_query_ids)) != len(seen_query_ids):
        raise ValueError("duplicate query_ids detected in submission")
    if set(seen_query_ids) != expected_query_set:
        raise ValueError("query_ids in submission do not match benchmark_queries")
    if has_duplicate_items:
        raise ValueError("duplicate item_ids found in query answer")
    if has_unknown_items:
        raise ValueError("unknown item_ids found in query answer")

    return {
        "status": "PASS",
        "rows": rows_count,
        "queries_with_exactly_50": sum(1 for c in query_answer_counts if c == 50),
        "queries_with_less_than_50": sum(1 for c in query_answer_counts if c < 50),
        "min_items": min(query_answer_counts) if query_answer_counts else 0,
        "max_items": max(query_answer_counts) if query_answer_counts else 0,
        "sha256": compute_sha256(sub_path),
    }


def compare_submissions(
    new_sub_path: str | Path,
    old_sub_path: str | Path,
) -> dict[str, Any]:
    new_df = pd.read_csv(new_sub_path, dtype=str).set_index("query_id")
    old_df = pd.read_csv(old_sub_path, dtype=str).set_index("query_id")

    overlaps: list[float] = []
    top1_changed: list[bool] = []
    substantially_changed: list[bool] = []

    for qid in new_df.index:
        if qid not in old_df.index:
            continue
        new_items = new_df.loc[qid, "answer"].split()
        old_items = old_df.loc[qid, "answer"].split()
        overlap_count = len(set(new_items) & set(old_items))
        overlaps.append(overlap_count)
        top1_changed.append(new_items[0] != old_items[0] if (new_items and old_items) else True)
        substantially_changed.append(overlap_count < 25)

    arr = np.array(overlaps, dtype=np.float64)
    return {
        "queries_compared": len(overlaps),
        "mean_overlap_top50": float(np.mean(arr)),
        "median_overlap_top50": float(np.median(arr)),
        "p10_overlap": float(np.percentile(arr, 10)),
        "p90_overlap": float(np.percentile(arr, 90)),
        "fraction_top1_changed": float(np.mean(top1_changed)),
        "fraction_queries_with_gt_25_changed_items": float(np.mean(substantially_changed)),
    }


def run_submission_generation() -> dict[str, Any]:
    t_start = time.time()
    print("=" * 80)
    print("FULL PRODUCTION BENCHMARK INFERENCE (GEO-CASCADE ARCHITECTURE)")
    print("=" * 80)

    # 1. Load benchmark & training datasets
    print("\n[1/6] Loading benchmark queries, items, and train coordinates...")
    benchmark_queries = pd.read_parquet("data/benchmark_queries.parquet")
    benchmark_items = pd.read_parquet("data/benchmark_items.parquet")
    train = pd.read_parquet("data/train.parquet")

    n_benchmark_queries = len(benchmark_queries)
    query_id_list = [str(x) for x in benchmark_queries["query_id"].tolist()]
    query_text_list = [str(x) for x in benchmark_queries["search_query"].tolist()]
    query_loc_map = dict(zip(query_id_list, benchmark_queries["search_location_id"]))
    query_cat_map = dict(zip(query_id_list, benchmark_queries["search_category"]))

    item_id_list = [str(x) for x in benchmark_items["item_id"].tolist()]
    item_loc_map = dict(zip(item_id_list, benchmark_items["item_location_id"]))
    item_cat_map = dict(zip(item_id_list, benchmark_items["item_category_id"]))
    item_microcat_map = dict(zip(item_id_list, benchmark_items["item_microcat_id"]))
    item_titles = [str(x) for x in benchmark_items["item_title_raw"].fillna("").tolist()]
    item_lat_map = dict(zip(item_id_list, benchmark_items["item_latitude"].astype(float).fillna(0.0)))
    item_lon_map = dict(zip(item_id_list, benchmark_items["item_longitude"].astype(float).fillna(0.0)))

    # Location coordinates from train
    loc_lat_dict: dict[Any, float] = {}
    loc_lon_dict: dict[Any, float] = {}
    loc_counts: dict[Any, int] = {}
    train_locs = train["item_location_id"].tolist()
    train_lats = train["item_latitude"].tolist()
    train_lons = train["item_longitude"].tolist()
    for loc, lat, lon in zip(train_locs, train_lats, train_lons, strict=True):
        if pd.notna(lat) and pd.notna(lon):
            loc_lat_dict[loc] = loc_lat_dict.get(loc, 0.0) + float(lat)
            loc_lon_dict[loc] = loc_lon_dict.get(loc, 0.0) + float(lon)
            loc_counts[loc] = loc_counts.get(loc, 0) + 1
    for loc, count in loc_counts.items():
        loc_lat_dict[loc] /= count
        loc_lon_dict[loc] /= count

    # 2. Fit routing on train data
    print("\n[2/6] Fitting Microcategory Routing model (Exact + Generalizing Classifier)...")
    t0 = time.time()
    exact = ExactPosteriorPredictor()
    exact.fit(train, query_col="search_query", category_col="item_microcat_id")

    clf = GeneralizingClassifierPredictor(min_df=5, max_features=50000, alpha=1e-5, seed=42)
    clf.fit(train)

    hybrid = HybridMicrocategoryPredictor(
        exact_predictor=exact,
        generalizing_predictor=clf,
        min_count=1,
        blend_strategy="fallback",
    )
    top5_benchmark_microcats = hybrid.predict_top_k(benchmark_queries, k=5)
    print(f"Routing model fit and predicted Top-5 in {time.time() - t0:.1f}s")

    # Free train data memory!
    del train
    gc.collect()

    microcat_to_items: dict[str, set[str]] = {}
    for item_id in item_id_list:
        mc = item_microcat_map[item_id]
        microcat_to_items.setdefault(str(mc), set()).add(item_id)

    allowed_items_per_query: list[set[str]] = []
    for cats in top5_benchmark_microcats:
        allowed: set[str] = set()
        for c in cats:
            allowed.update(microcat_to_items.get(str(c), set()))
        allowed_items_per_query.append(allowed)

    # 3. Fit Lexical Index & Retrieve candidates (k=1000)
    print("\n[3/6] Fitting Lexical Index and retrieving candidates (k=1000)...")
    t0 = time.time()
    field_index = FieldAwareSparseIndex.fit(
        benchmark_items,
        branch_b_analyzer="char_wb",
        branch_b_ngram_range=(3, 5),
        min_df=2,
    )
    title_char_index = SparseBranchIndex.fit(
        item_id_list,
        item_titles,
        name="title_char",
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
    )
    query_title_texts = field_text(benchmark_queries, ("search_query",))
    query_title_params_texts = field_text(
        benchmark_queries, ("search_query", "search_infm_params_text")
    )

    routed_word = field_index.branches["branch_a"].retrieve_routed(
        query_title_texts, allowed_items_per_query, k=1000
    )
    routed_char = title_char_index.retrieve_routed(
        query_title_texts, allowed_items_per_query, k=1000
    )
    routed_title_params = field_index.branches["branch_b"].retrieve_routed(
        query_title_params_texts, allowed_items_per_query, k=1000
    )
    global_word = field_index.branches["branch_a"].retrieve(query_title_texts, k=300)
    global_char = title_char_index.retrieve(query_title_texts, k=300)
    print(f"Lexical retrieval complete in {time.time() - t0:.1f}s")

    del field_index, title_char_index
    gc.collect()

    # 4. Dense E5 Retrieval (k=1000)
    print("\n[4/6] Dense E5 Retrieval (Loading embeddings & Search)...")
    t0 = time.time()
    query_emb_file = Path("artifacts/research/benchmark_query_e5_embeddings.npy")
    if query_emb_file.is_file():
        print(f"Loading precomputed benchmark query embeddings from {query_emb_file}...")
        benchmark_query_embeddings = np.load(query_emb_file)
    else:
        print("Encoding benchmark queries with E5...")
        encoder = E5QueryEncoder()
        benchmark_query_embeddings = encoder.encode(query_text_list, batch_size=64)

    e5_items_json = [
        str(x)
        for x in json.loads(
            Path("artifacts/selected/embeddings/e5/benchmark/item_ids.json").read_text()
        )
    ]
    e5_item_embeddings = np.load(
        "artifacts/selected/embeddings/e5/benchmark/embeddings.npy", mmap_mode="r"
    )
    e5_retriever = RoutedDenseE5Retriever(
        item_ids=e5_items_json, item_embeddings=e5_item_embeddings
    )
    routed_e5 = e5_retriever.retrieve_routed(
        benchmark_query_embeddings, allowed_items_per_query, k=1000
    )
    global_e5 = e5_retriever.retrieve_global(benchmark_query_embeddings, k=300)
    print(f"Dense E5 retrieval complete in {time.time() - t0:.1f}s")

    # 5. Fusion & Local-First Geo-Spatial Cascade Ranking
    print("\n[5/6] Candidate Fusion & Local-First Hyperlocal Cascade Ranking...")
    t0 = time.time()
    sources = [
        routed_e5,
        routed_char,
        routed_word,
        routed_title_params,
        global_e5,
        global_word,
        global_char,
    ]
    weights = [1.5, 1.2, 0.8, 0.5, 0.4, 0.3, 0.3]
    raw_rrf = reciprocal_rank_fusion(sources, weights=weights, c=60.0, limit=1000)

    # Geo + Category score weighting (Validated on canonical benchmark validation)
    final_top50_rankings: list[list[str]] = []
    dist_decay = 0.25
    same_loc_bonus = 30.0

    for qid, ranking in zip(query_id_list, raw_rrf, strict=True):
        q_loc = query_loc_map[qid]
        q_cat = query_cat_map[qid]
        q_lat = loc_lat_dict.get(q_loc)
        q_lon = loc_lon_dict.get(q_loc)

        boosted: list[tuple[str, float]] = []
        for item_id, score in ranking:
            it_loc = item_loc_map.get(item_id)
            it_cat = item_cat_map.get(item_id)
            it_lat = item_lat_map.get(item_id, 0.0)
            it_lon = item_lon_map.get(item_id, 0.0)

            # Category match multiplier (clicks have 99.98% same category)
            cat_mult = 1.0 if (it_cat == q_cat) else 0.001

            # Geo match multiplier (clicks have 84.16% same location, 97.28% within 30km)
            if it_loc == q_loc:
                geo_mult = same_loc_bonus
            elif q_lat is not None and q_lon is not None and it_lat != 0.0:
                dist = haversine_km(q_lat, q_lon, it_lat, it_lon)
                geo_mult = float(np.exp(-dist_decay * (dist / 10.0)))
            else:
                geo_mult = 0.05

            final_sc = score * cat_mult * geo_mult
            boosted.append((item_id, final_sc))

        boosted.sort(key=lambda p: (-p[1], p[0]))
        top50 = [it for it, _ in boosted[:50]]
        if len(top50) < 50:
            seen_items = set(top50)
            for it in item_id_list:
                if it not in seen_items:
                    top50.append(it)
                    seen_items.add(it)
                    if len(top50) == 50:
                        break
        final_top50_rankings.append(top50)

    print(f"Cascade ranking complete in {time.time() - t0:.1f}s")

    # 6. Write and Validate answer.csv
    print("\n[6/6] Writing answer.csv and running independent validation...")
    final_output_path = Path("answer.csv")
    with final_output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "answer"])
        for qid, items in zip(query_id_list, final_top50_rankings, strict=True):
            writer.writerow([qid, " ".join(items)])

    # Save versioned copy
    versioned_dir = Path("artifacts/submissions/geo_cascade_v1")
    versioned_dir.mkdir(parents=True, exist_ok=True)
    versioned_path = versioned_dir / "answer.csv"
    shutil.copyfile(final_output_path, versioned_path)
    print(f"Versioned submission copy saved to {versioned_path}")

    val_res = validate_submission_file(
        final_output_path,
        "data/benchmark_queries.parquet",
        "data/benchmark_items.parquet",
    )
    print("\nIndependent Validation Result:")
    for k, v in val_res.items():
        print(f"  {k}: {v}")

    # Compare against old failed submission (0.303828)
    old_sub_path = Path("artifacts/submissions/old_0303828/answer.csv")
    if old_sub_path.is_file():
        overlap_res = compare_submissions(final_output_path, old_sub_path)
        print("\nComparison against old failed submission (0.303828):")
        for k, v in overlap_res.items():
            print(f"  {k}: {v}")
        with open("artifacts/research/submission_comparison_geo_cascade.json", "w", encoding="utf-8") as f:
            json.dump(overlap_res, f, indent=2)

    # Qualitative Benchmark Audit (~30 queries)
    print("\nRunning Qualitative Benchmark Audit (30 sample queries)...")
    rng = np.random.default_rng(42)
    audit_indices = rng.choice(n_benchmark_queries, size=30, replace=False)
    audit_records: list[dict[str, Any]] = []

    item_title_dict = dict(zip(item_id_list, item_titles, strict=True))

    for idx in sorted(audit_indices):
        qid = query_id_list[idx]
        q_row = benchmark_queries.iloc[idx]
        predicted_cats = top5_benchmark_microcats[idx]
        top10_items = final_top50_rankings[idx][:10]
        top10_details = [
            {
                "item_id": it_id,
                "title": item_title_dict.get(it_id, ""),
                "item_location_id": item_loc_map.get(it_id, ""),
                "item_category_id": item_cat_map.get(it_id, ""),
                "same_location": item_loc_map.get(it_id) == q_row["search_location_id"],
            }
            for it_id in top10_items
        ]
        audit_records.append({
            "query_id": qid,
            "query": str(q_row["search_query"]),
            "filters": str(q_row.get("search_infm_params_text", "")),
            "category": str(q_row.get("search_category", "")),
            "location_id": str(q_row.get("search_location_id", "")),
            "predicted_top5_microcategories": predicted_cats,
            "top10_results": top10_details,
        })

    with open("artifacts/research/qualitative_benchmark_audit_geo_cascade.json", "w", encoding="utf-8") as f:
        json.dump(audit_records, f, indent=2, ensure_ascii=False)

    total_time = time.time() - t_start
    print(f"\nAll tasks complete in {total_time:.1f}s (~{total_time/60.0:.1f} minutes).")
    return val_res


if __name__ == "__main__":
    run_submission_generation()
