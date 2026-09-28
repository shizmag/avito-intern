"""Generate benchmark submission with frozen microcategory routed pipeline.

Architecture:
1. Microcategory Routing: Hybrid (Exact Posterior + TF-IDF SGDClassifier) -> Top-5 Microcategories
2. Routed Candidates:
   - Routed Dense E5 (k=500)
   - Routed Char TF-IDF (k=500)
   - Routed Word TF-IDF (k=500)
   - Routed Title+Params Char (k=500)
3. Global Fallback Candidates:
   - Global Dense E5 (k=200)
   - Global Word TF-IDF (k=200)
4. Fusion: Weighted RRF (c=60.0) -> Top-50 Unique Items
5. Submission Validation & Overlap Analysis
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import time
from pathlib import Path
from typing import Any, cast

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
    """Independent validator according to competition requirements."""
    sub_path = Path(submission_path)
    if not sub_path.is_file():
        raise FileNotFoundError(f"submission file {submission_path} does not exist")

    # Check raw header without pandas
    with sub_path.open("r", encoding="utf-8") as f:
        header_line = f.readline().rstrip("\r\n")
        if header_line != "query_id,answer":
            raise ValueError(f"invalid header line: {header_line!r}, expected 'query_id,answer'")

    df_queries = pd.read_parquet(expected_queries_path)
    expected_query_ids = [str(x) for x in df_queries["query_id"].tolist()]
    expected_query_set = set(expected_query_ids)

    df_items = pd.read_parquet(expected_items_path)
    valid_item_set = set(str(x) for x in df_items["item_id"].tolist())

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
    """Compare new submission vs old failed submission."""
    new_df = pd.read_csv(new_sub_path, dtype=str).set_index("query_id")
    old_df = pd.read_csv(old_sub_path, dtype=str).set_index("query_id")

    overlaps: list[float] = []
    top1_changed: list[bool] = []
    substantially_changed: list[bool] = []  # > 25 changed items (i.e. overlap < 25)

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


def main() -> None:
    t_start = time.time()
    print("=" * 80)
    print("STARTING BENCHMARK INFERENCE & SUBMISSION GENERATION")
    print("=" * 80)

    # 1. Load benchmark data & train data
    print("\n[1/6] Loading benchmark and training datasets...")
    benchmark_queries = pd.read_parquet("data/benchmark_queries.parquet")
    benchmark_items = pd.read_parquet("data/benchmark_items.parquet")
    train = pd.read_parquet("data/train.parquet")

    n_benchmark_queries = len(benchmark_queries)
    print(
        f"Benchmark queries: {n_benchmark_queries:,}, Benchmark items: {len(benchmark_items):,}, Train rows: {len(train):,}"
    )

    query_id_list = [str(x) for x in benchmark_queries["query_id"].tolist()]
    query_text_list = [str(x) for x in benchmark_queries["search_query"].tolist()]
    item_id_list = [str(x) for x in benchmark_items["item_id"].tolist()]
    item_microcat_list = [str(x) for x in benchmark_items["item_microcat_id"].tolist()]

    # 2. Fit frozen routing models on all allowed train data
    print("\n[2/6] Fitting frozen Microcategory Routing model (Exact + Generalizing Classifier)...")
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
    # Predict top-5 microcategories for all 2,452 benchmark queries
    top5_benchmark_microcats = hybrid.predict_top_k(benchmark_queries, k=5)
    print(f"Routing model fit and predicted Top-5 in {time.time() - t0:.1f}s")

    # Map microcategory -> item IDs
    microcat_to_items: dict[str, set[str]] = {}
    for item_id, microcat in zip(item_id_list, item_microcat_list, strict=True):
        microcat_to_items.setdefault(microcat, set()).add(item_id)

    allowed_items_per_query: list[set[str]] = []
    for cats in top5_benchmark_microcats:
        allowed: set[str] = set()
        for c in cats:
            allowed.update(microcat_to_items.get(c, set()))
        allowed_items_per_query.append(allowed)

    mean_bm_corpus = np.mean([len(s) for s in allowed_items_per_query])
    print(f"Mean routed corpus size for benchmark queries: {mean_bm_corpus:.1f} items")

    # 3. Fit Lexical Index on full benchmark items
    print("\n[3/6] Fitting Lexical Index (FieldAwareSparseIndex + TitleChar)...")
    t0 = time.time()
    field_index = FieldAwareSparseIndex.fit(
        benchmark_items,
        branch_b_analyzer="char_wb",
        branch_b_ngram_range=(3, 5),
        min_df=2,
    )
    item_titles = [str(x) for x in benchmark_items["item_title_raw"].fillna("").tolist()]
    title_char_index = SparseBranchIndex.fit(
        item_id_list,
        item_titles,
        name="title_char",
        analyzer="char_wb",
        ngram_range=(3, 5),
        min_df=2,
    )
    print(f"Lexical indices fitted in {time.time() - t0:.1f}s")

    # Run Lexical Retrievers for benchmark queries
    print("Retrieving lexical candidates for benchmark queries...")
    query_title_texts = field_text(benchmark_queries, ("search_query",))
    query_title_params_texts = field_text(
        benchmark_queries, ("search_query", "search_infm_params_text")
    )

    t0 = time.time()
    routed_word = field_index.branches["branch_a"].retrieve_routed(
        query_title_texts, allowed_items_per_query, k=500
    )
    routed_char = title_char_index.retrieve_routed(
        query_title_texts, allowed_items_per_query, k=500
    )
    routed_title_params = field_index.branches["branch_b"].retrieve_routed(
        query_title_params_texts, allowed_items_per_query, k=500
    )
    global_word = field_index.branches["branch_a"].retrieve(query_title_texts, k=200)
    print(f"Lexical retrieval complete in {time.time() - t0:.1f}s")

    # 4. Dense E5 Retrieval for benchmark queries
    print("\n[4/6] Dense E5 Retrieval (Encoding benchmark queries & search)...")
    t0 = time.time()
    encoder = E5QueryEncoder()
    benchmark_query_embeddings = encoder.encode(query_text_list, batch_size=64)
    print(f"Encoded {n_benchmark_queries} benchmark queries in {time.time() - t0:.1f}s")

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

    t0 = time.time()
    routed_e5 = e5_retriever.retrieve_routed(
        benchmark_query_embeddings, allowed_items_per_query, k=500
    )
    global_e5 = e5_retriever.retrieve_global(benchmark_query_embeddings, k=200)
    print(f"Dense E5 retrieval complete in {time.time() - t0:.1f}s")

    # 5. Fusion: Weighted RRF (c=60.0)
    print("\n[5/6] Final Ranking & Candidate Fusion via Weighted RRF...")
    # Best validated weights:
    # routed_e5: 1.2, routed_char: 1.0, routed_word: 0.8, routed_title_params: 0.5, global_e5: 0.4, global_word: 0.3
    sources = [
        routed_e5,
        routed_char,
        routed_word,
        routed_title_params,
        global_e5,
        global_word,
    ]
    weights = [1.2, 1.0, 0.8, 0.5, 0.4, 0.3]

    t0 = time.time()
    final_top50_rankings = reciprocal_rank_fusion(
        sources,
        weights=weights,
        c=60.0,
        limit=50,
    )
    print(f"Fusion complete in {time.time() - t0:.1f}s")

    # 6. Build and write answer.csv
    print("\n[6/6] Formatting and writing answer.csv...")
    final_output_path = Path("answer.csv")
    with final_output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "answer"])
        for qid, ranking in zip(query_id_list, final_top50_rankings, strict=True):
            items_str = " ".join(item_id for item_id, _ in ranking[:50])
            writer.writerow([qid, items_str])

    # Save versioned copy
    versioned_dir = Path("artifacts/submissions/routed_recall_v2")
    versioned_dir.mkdir(parents=True, exist_ok=True)
    versioned_path = versioned_dir / "answer.csv"
    shutil.copyfile(final_output_path, versioned_path)
    print(f"Copied versioned submission to {versioned_path}")

    # Independent validation
    print("\nRunning independent submission validation...")
    val_result = validate_submission_file(
        final_output_path,
        "data/benchmark_queries.parquet",
        "data/benchmark_items.parquet",
    )
    print("Independent Validation Result:")
    for k, v in val_result.items():
        print(f"  {k}: {v}")

    # Compare with old failed submission
    old_sub_path = Path("artifacts/submissions/old_0303828/answer.csv")
    if old_sub_path.is_file():
        print("\nComparing new submission against old failed submission (0.303828)...")
        overlap_res = compare_submissions(final_output_path, old_sub_path)
        for k, v in overlap_res.items():
            print(f"  {k}: {v}")
        with open("artifacts/research/submission_comparison.json", "w", encoding="utf-8") as f:
            json.dump(overlap_res, f, indent=2)

    # Qualitative Benchmark Audit (~30 queries)
    print("\nPerforming Qualitative Benchmark Audit (30 sample queries)...")
    rng = np.random.default_rng(42)
    audit_indices = rng.choice(n_benchmark_queries, size=30, replace=False)
    audit_records: list[dict[str, Any]] = []

    item_title_dict = dict(zip(item_id_list, item_titles, strict=True))
    item_microcat_dict = dict(zip(item_id_list, item_microcat_list, strict=True))

    for idx in sorted(audit_indices):
        qid = query_id_list[idx]
        q_row = benchmark_queries.iloc[idx]
        predicted_cats = top5_benchmark_microcats[idx]
        top10_recs = final_top50_rankings[idx][:10]
        top10_details = [
            {
                "item_id": it_id,
                "score": round(sc, 5),
                "title": item_title_dict.get(it_id, ""),
                "item_microcat_id": item_microcat_dict.get(it_id, ""),
            }
            for it_id, sc in top10_recs
        ]
        audit_records.append({
            "query_id": qid,
            "query": str(q_row["search_query"]),
            "filters": str(q_row.get("search_infm_params_text", "")),
            "category": str(q_row.get("search_category", "")),
            "predicted_top5_microcategories": predicted_cats,
            "top10_results": top10_details,
        })

    with open("artifacts/research/qualitative_benchmark_audit.json", "w", encoding="utf-8") as f:
        json.dump(audit_records, f, indent=2, ensure_ascii=False)
    print(f"Qualitative audit saved to artifacts/research/qualitative_benchmark_audit.json")

    # Manifest file
    manifest = {
        "pipeline_name": "microcategory_routed_candidate_generation_v2",
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "microcategory_routing": {
            "model": "Hybrid (ExactPosterior + TF-IDF SGDClassifier)",
            "k": 5,
            "validation_containment": 0.9285,
        },
        "retrieval_branches": {
            "routed_dense_e5": {"model": "intfloat/multilingual-e5-base", "k": 500, "weight": 1.2},
            "routed_char_tfidf": {"analyzer": "char_wb", "ngrams": [3, 5], "k": 500, "weight": 1.0},
            "routed_word_tfidf": {"analyzer": "word", "ngrams": [1, 2], "k": 500, "weight": 0.8},
            "routed_title_params_char": {"analyzer": "char_wb", "ngrams": [3, 5], "k": 500, "weight": 0.5},
            "global_e5_fallback": {"k": 200, "weight": 0.4},
            "global_word_fallback": {"k": 200, "weight": 0.3},
        },
        "historical_intent": "REJECTED (negligible incremental gain)",
        "two_tower": "EXCLUDED",
        "fusion": {"method": "reciprocal_rank_fusion", "c": 60.0, "top_k": 50},
        "submission": {
            "path": "answer.csv",
            "versioned_path": "artifacts/submissions/routed_recall_v2/answer.csv",
            "sha256": val_result["sha256"],
            "rows": val_result["rows"],
        },
    }
    with open("artifacts/research/selected_manifest.json", "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print(f"\nAll submission tasks complete in {time.time() - t_start:.1f}s.")


if __name__ == "__main__":
    main()
