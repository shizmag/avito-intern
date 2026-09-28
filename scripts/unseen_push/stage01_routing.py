"""Stage 01: Microcategory Routing for Benchmark Queries.

Fits High-Capacity Multi-Label Generalizing Router on train.parquet.
Predicts Top-10 microcategories for all 2,452 benchmark queries.
Saves:
- artifacts/unseen_push/benchmark/top10_microcats.json
- artifacts/unseen_push/benchmark/allowed_items.pkl
"""

from __future__ import annotations

import gc
import json
import pickle
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from avito_candidate_generation.routing import ExactPosteriorPredictor


def main() -> None:
    t0 = time.time()
    print("=" * 80)
    print("STAGE 01: BENCHMARK MICROCATEGORY ROUTING (K=10)")
    print("=" * 80)

    out_dir = Path("artifacts/unseen_push/benchmark")
    out_dir.mkdir(parents=True, exist_ok=True)

    train = pd.read_parquet("data/train.parquet")
    benchmark_queries = pd.read_parquet("data/benchmark_queries.parquet")
    benchmark_items = pd.read_parquet("data/benchmark_items.parquet")

    item_id_list = [str(x) for x in benchmark_items["item_id"].tolist()]
    item_microcat_map = dict(
        zip(item_id_list, benchmark_items["item_microcat_id"].astype(str))
    )
    microcat_to_items: dict[str, set[str]] = {}
    for it, mc in item_microcat_map.items():
        microcat_to_items.setdefault(mc, set()).add(it)

    # 1. Exact Posterior
    print("Fitting ExactPosteriorPredictor...")
    exact = ExactPosteriorPredictor().fit(
        train, query_col="search_query", category_col="item_microcat_id"
    )

    # 2. Generalizing Multi-Label SGD
    print("Aggregating train contexts and fitting High-Capacity Generalizing Router...")
    ctx_mc = (
        train.groupby(["search_query", "search_infm_params_text", "search_category", "item_microcat_id"])
        .size()
        .reset_index(name="count")
    )
    ctx_totals = ctx_mc.groupby(["search_query", "search_infm_params_text", "search_category"])["count"].transform("sum")
    ctx_mc["sample_weight"] = np.clip(np.log1p(ctx_mc["count"]) / np.log1p(ctx_totals), 0.1, 1.0)

    mc_train_q = ctx_mc["search_query"].fillna("").astype(str).tolist()
    mc_train_p = ctx_mc["search_infm_params_text"].fillna("").astype(str).tolist()
    mc_train_c = ctx_mc["search_category"].fillna("").astype(str).tolist()
    mc_train_words = [f"{q} {p} cat_{c}".strip() for q, p, c in zip(mc_train_q, mc_train_p, mc_train_c, strict=True)]
    mc_train_chars = [f" {q} ".replace("\t", " ") for q in mc_train_q]
    mc_train_targets = ctx_mc["item_microcat_id"].astype(str).tolist()
    mc_sample_weights = ctx_mc["sample_weight"].to_numpy()

    bm_q = benchmark_queries["search_query"].fillna("").astype(str).tolist()
    bm_p = benchmark_queries["search_infm_params_text"].fillna("").astype(str).tolist()
    bm_c = benchmark_queries["search_category"].fillna("").astype(str).tolist()
    bm_words = [f"{q} {p} cat_{c}".strip() for q, p, c in zip(bm_q, bm_p, bm_c, strict=True)]
    bm_chars = [f" {q} ".replace("\t", " ") for q in bm_q]

    vec_w = TfidfVectorizer(ngram_range=(1, 2), analyzer="word", min_df=2, max_features=120000, sublinear_tf=True)
    vec_c = TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", min_df=3, max_features=120000, sublinear_tf=True)
    X_tr = hstack([vec_w.fit_transform(mc_train_words), vec_c.fit_transform(mc_train_chars)], format="csr")
    X_bm = hstack([vec_w.transform(bm_words), vec_c.transform(bm_chars)], format="csr")

    clf = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=35, tol=1e-4, random_state=42)
    clf.fit(X_tr, mc_train_targets, sample_weight=mc_sample_weights)

    preds_raw = clf.predict_proba(X_bm)
    classes = list(clf.classes_)
    top10_gen = []
    for row in preds_raw:
        top_idx = np.argsort(row)[::-1][:10]
        top10_gen.append([classes[i] for i in top_idx])

    top10_benchmark: list[list[str]] = []
    seen_in_train = 0
    for q, gen_cats in zip(bm_q, top10_gen, strict=True):
        if exact.is_seen(q):
            seen_in_train += 1
            top10_benchmark.append(exact.predict_top_k([q], k=10)[0])
        else:
            top10_benchmark.append(gen_cats)

    print(f"Benchmark queries: {len(bm_q)}, Seen in train: {seen_in_train}, Unseen: {len(bm_q) - seen_in_train}")

    # Build allowed items sets
    allowed_items_per_query: list[set[str]] = []
    corpus_sizes = []
    for cats in top10_benchmark:
        allowed: set[str] = set()
        for c in cats:
            allowed.update(microcat_to_items.get(c, set()))
        allowed_items_per_query.append(allowed)
        corpus_sizes.append(len(allowed))

    c_arr = np.array(corpus_sizes)
    print(f"Corpus size per query: mean={c_arr.mean():.1f}, median={np.median(c_arr):.0f}, p95={np.percentile(c_arr, 95):.0f}")

    # Save artifacts
    with (out_dir / "top10_microcats.json").open("w", encoding="utf-8") as f:
        json.dump(top10_benchmark, f)

    with (out_dir / "allowed_items.pkl").open("wb") as f:
        pickle.dump(allowed_items_per_query, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"Stage 01 complete in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
