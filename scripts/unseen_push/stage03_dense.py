"""Stage 03: Dense Candidate Retrieval for Benchmark Queries.

Pure dense process lifetime:
Loads precomputed Fine-Tuned E5 and Generic E5 embeddings.
Retrieves depth=1000 candidates for:
- Routed Fine-Tuned E5
- Routed Generic E5
- Global Fine-Tuned E5 (k=300)
- Global Generic E5 (k=300)
Saves:
- artifacts/unseen_push/benchmark/dense_candidates.pkl
"""

from __future__ import annotations

import gc
import json
import pickle
import time
from pathlib import Path

import numpy as np

from avito_candidate_generation.retrievers.routed_e5 import RoutedDenseE5Retriever


def main() -> None:
    t0 = time.time()
    print("=" * 80)
    print("STAGE 03: BENCHMARK DENSE RETRIEVAL (PURE PROCESS LIFETIME)")
    print("=" * 80)

    in_dir = Path("artifacts/unseen_push/benchmark")
    dense_dir = Path("artifacts/unseen_push/dense")

    with (in_dir / "allowed_items.pkl").open("rb") as f:
        allowed_items_per_query: list[set[str]] = pickle.load(f)

    # 1. Load Precomputed Query Embeddings
    print("Loading precomputed benchmark query embeddings...")
    ft_query_emb_file = dense_dir / "benchmark_query_ft_e5_embeddings.npy"
    if not ft_query_emb_file.is_file():
        raise FileNotFoundError(f"Missing {ft_query_emb_file}")
    bm_query_emb_ft = np.load(ft_query_emb_file)

    gen_query_emb_file = Path("artifacts/research/benchmark_query_e5_embeddings.npy")
    if not gen_query_emb_file.is_file():
        raise FileNotFoundError(f"Missing {gen_query_emb_file}")
    bm_query_emb_gen = np.load(gen_query_emb_file)

    print(f"Loaded FT embeddings shape: {bm_query_emb_ft.shape}, Generic embeddings shape: {bm_query_emb_gen.shape}")

    # 2. Load Item Embeddings
    print("Loading benchmark item embeddings...")
    e5_items_json = [
        str(x)
        for x in json.loads(
            Path("artifacts/selected/embeddings/e5/benchmark/item_ids.json").read_text()
        )
    ]
    e5_item_embeddings = np.load(
        "artifacts/selected/embeddings/e5/benchmark/embeddings.npy", mmap_mode="r"
    )

    retriever = RoutedDenseE5Retriever(
        item_ids=e5_items_json, item_embeddings=e5_item_embeddings
    )

    # 3. Retrieve Dense Candidates
    print("Retrieving Fine-Tuned E5 candidates (routed depth=1000, global=300)...")
    routed_ft_e5 = retriever.retrieve_routed(bm_query_emb_ft, allowed_items_per_query, k=1000)
    global_ft_e5 = retriever.retrieve_global(bm_query_emb_ft, k=300)

    print("Retrieving Generic E5 candidates (routed depth=1000, global=300)...")
    routed_gen_e5 = retriever.retrieve_routed(bm_query_emb_gen, allowed_items_per_query, k=1000)
    global_gen_e5 = retriever.retrieve_global(bm_query_emb_gen, k=300)

    dense_dict = {
        "routed_ft_e5": routed_ft_e5,
        "routed_gen_e5": routed_gen_e5,
        "global_ft_e5": global_ft_e5,
        "global_gen_e5": global_gen_e5,
    }

    out_file = in_dir / "dense_candidates.pkl"
    with out_file.open("wb") as f:
        pickle.dump(dense_dict, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Stage 03 complete in {time.time() - t0:.1f}s. Saved to {out_file}")


if __name__ == "__main__":
    main()
