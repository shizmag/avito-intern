"""Stage 02: Sparse Candidate Retrieval for Benchmark Queries.

Fits FieldAwareSparseIndex and SparseBranchIndex.
Retrieves depth=1000 candidates for:
- Routed Word TF-IDF
- Routed Char TF-IDF
- Routed Title+Params TF-IDF
- Global Word TF-IDF (k=300)
- Global Char TF-IDF (k=300)
Saves:
- artifacts/unseen_push/benchmark/sparse_candidates.pkl
"""

from __future__ import annotations

import gc
import pickle
import time
from pathlib import Path

import pandas as pd

from avito_candidate_generation.research import field_text
from avito_candidate_generation.retrievers.routed_lexical import (
    FieldAwareSparseIndex,
    SparseBranchIndex,
)


def main() -> None:
    t0 = time.time()
    print("=" * 80)
    print("STAGE 02: BENCHMARK SPARSE RETRIEVAL (DEPTH 1000)")
    print("=" * 80)

    in_dir = Path("artifacts/unseen_push/benchmark")
    with (in_dir / "allowed_items.pkl").open("rb") as f:
        allowed_items_per_query: list[set[str]] = pickle.load(f)

    benchmark_queries = pd.read_parquet("data/benchmark_queries.parquet")
    benchmark_items = pd.read_parquet("data/benchmark_items.parquet")

    item_id_list = [str(x) for x in benchmark_items["item_id"].tolist()]
    item_titles = [str(x) for x in benchmark_items["item_title_raw"].fillna("").tolist()]

    print("Fitting FieldAwareSparseIndex (word & char branches)...")
    field_index = FieldAwareSparseIndex.fit(
        benchmark_items,
        branch_b_analyzer="char_wb",
        branch_b_ngram_range=(3, 5),
        min_df=2,
    )

    print("Fitting SparseBranchIndex (title char branch)...")
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

    print("Retrieving routed and global sparse candidates...")
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

    sparse_dict = {
        "routed_word": routed_word,
        "routed_char": routed_char,
        "routed_title_params": routed_title_params,
        "global_word": global_word,
        "global_char": global_char,
    }

    out_file = in_dir / "sparse_candidates.pkl"
    with out_file.open("wb") as f:
        pickle.dump(sparse_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Stage 02 complete in {time.time() - t0:.1f}s. Saved to {out_file}")


if __name__ == "__main__":
    main()
