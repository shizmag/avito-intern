"""Step 00: Build Comprehensive Candidate Cache for Fast Hyperparameter Optimization.

Generates candidate pools for all 5347 validation queries:
1. Global branches (up to k=1000):
   - global_ft_e5
   - global_gen_e5
   - global_word
   - global_char
2. Routed branches (up to k=1500) for routing policies:
   - k10 (Baseline Champion)
   - var_a (Adaptive Top1 Prob: K=5 / 10 / 15)
   - var_b (Adaptive Top1 Prob: K=8 / 10 / 12)
   - k12 (Fixed K=12)
   - k15 (Fixed K=15)
   Each policy contains:
   - routed_ft_e5
   - routed_gen_e5
   - routed_char
   - routed_word
   - routed_title_params

Saves to:
artifacts/fast_opt/val_candidate_cache.pkl
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

from avito_candidate_generation.research import field_text
from avito_candidate_generation.retrievers.routed_e5 import RoutedDenseE5Retriever
from avito_candidate_generation.retrievers.routed_lexical import (
    FieldAwareSparseIndex,
    SparseBranchIndex,
)
from avito_candidate_generation.routing import ExactPosteriorPredictor


def main() -> None:
    t0 = time.time()
    print("=" * 80)
    print("STEP 00: BUILDING COMPREHENSIVE VALIDATION CANDIDATE CACHE")
    print("=" * 80)

    val_dir = Path("artifacts/unseen_push/validation")
    dense_dir = Path("artifacts/unseen_push/dense")
    out_dir = Path("artifacts/fast_opt")
    out_dir.mkdir(parents=True, exist_ok=True)

    val_contexts = pd.read_parquet(val_dir / "val_contexts.parquet")
    train_part = pd.read_parquet(val_dir / "train_part.parquet")
    benchmark_items = pd.read_parquet("data/benchmark_items.parquet")

    query_ids = [str(x) for x in val_contexts["internal_query_id"].tolist()]
    is_seen = val_contexts["is_seen"].to_numpy()
    is_unseen = val_contexts["is_unseen"].to_numpy()

    item_id_list = [str(x) for x in benchmark_items["item_id"].tolist()]
    item_titles = [
        str(x) for x in benchmark_items["item_title_raw"].fillna("").tolist()
    ]
    item_microcat_map = dict(
        zip(item_id_list, benchmark_items["item_microcat_id"].astype(str))
    )
    microcat_to_items: dict[str, set[str]] = {}
    for it, mc in item_microcat_map.items():
        microcat_to_items.setdefault(mc, set()).add(it)

    # 1. Routing Model Training
    print("\n[1/4] Training Router and computing Routing Policies...")
    exact = ExactPosteriorPredictor().fit(
        train_part, query_col="search_query", category_col="item_microcat_id"
    )

    ctx_mc = (
        train_part.groupby(
            ["search_query", "search_infm_params_text", "search_category", "item_microcat_id"]
        )
        .size()
        .reset_index(name="count")
    )
    ctx_totals = ctx_mc.groupby(
        ["search_query", "search_infm_params_text", "search_category"]
    )["count"].transform("sum")
    ctx_mc["sample_weight"] = np.clip(
        np.log1p(ctx_mc["count"]) / np.log1p(ctx_totals), 0.1, 1.0
    )

    mc_train_q = ctx_mc["search_query"].fillna("").astype(str).tolist()
    mc_train_p = ctx_mc["search_infm_params_text"].fillna("").astype(str).tolist()
    mc_train_c = ctx_mc["search_category"].fillna("").astype(str).tolist()
    mc_train_words = [
        f"{q} {p} cat_{c}".strip()
        for q, p, c in zip(mc_train_q, mc_train_p, mc_train_c, strict=True)
    ]
    mc_train_chars = [f" {q} ".replace("\t", " ") for q in mc_train_q]
    mc_train_targets = ctx_mc["item_microcat_id"].astype(str).tolist()
    mc_sample_weights = ctx_mc["sample_weight"].to_numpy()

    val_q = val_contexts["search_query"].fillna("").astype(str).tolist()
    val_p = val_contexts["search_infm_params_text"].fillna("").astype(str).tolist()
    val_c = val_contexts["search_category"].fillna("").astype(str).tolist()
    val_words = [
        f"{q} {p} cat_{c}".strip()
        for q, p, c in zip(val_q, val_p, val_c, strict=True)
    ]
    val_chars = [f" {q} ".replace("\t", " ") for q in val_q]

    vec_w = TfidfVectorizer(
        ngram_range=(1, 2), analyzer="word", min_df=2, max_features=120000, sublinear_tf=True
    )
    vec_c = TfidfVectorizer(
        ngram_range=(3, 5), analyzer="char_wb", min_df=3, max_features=120000, sublinear_tf=True
    )
    X_tr = hstack([vec_w.fit_transform(mc_train_words), vec_c.fit_transform(mc_train_chars)], format="csr")
    X_vl = hstack([vec_w.transform(val_words), vec_c.transform(val_chars)], format="csr")

    clf_new = SGDClassifier(
        loss="log_loss", penalty="l2", alpha=5e-6, max_iter=35, tol=1e-4, random_state=42
    )
    clf_new.fit(X_tr, mc_train_targets, sample_weight=mc_sample_weights)

    preds_raw = clf_new.predict_proba(X_vl)
    classes_new = list(clf_new.classes_)

    top20_gen = []
    top1_probs = []
    for row in preds_raw:
        sorted_indices = np.argsort(row)[::-1]
        top20_gen.append([classes_new[i] for i in sorted_indices[:20]])
        top1_probs.append(float(row[sorted_indices[0]]))
    top1_probs = np.array(top1_probs)

    hybrid_top20 = []
    for q, gen_cats in zip(val_q, top20_gen, strict=True):
        if exact.is_seen(q):
            hybrid_top20.append(exact.predict_top_k([q], k=20)[0])
        else:
            hybrid_top20.append(gen_cats)

    p_33, p_66 = np.percentile(top1_probs[is_unseen], [33.3, 66.7])
    print(f"Top1 prob quantiles: 33%={p_33:.4f}, 66%={p_66:.4f}")

    def get_allowed(k_list: list[int]) -> list[set[str]]:
        allowed = []
        for k, cats in zip(k_list, hybrid_top20, strict=True):
            s = set()
            for c in cats[:k]:
                s.update(microcat_to_items.get(c, set()))
            allowed.append(s)
        return allowed

    # Routing policies
    k_k10 = [10] * len(val_q)
    k_k12 = [12] * len(val_q)
    k_k15 = [15] * len(val_q)

    k_var_a = []
    k_var_b = []
    for idx, p in enumerate(top1_probs):
        if not is_unseen[idx]:
            k_var_a.append(10)
            k_var_b.append(10)
        else:
            if p >= p_66:
                k_var_a.append(5)
                k_var_b.append(8)
            elif p >= p_33:
                k_var_a.append(10)
                k_var_b.append(10)
            else:
                k_var_a.append(15)
                k_var_b.append(12)

    policies = {
        "k10": get_allowed(k_k10),
        "k12": get_allowed(k_k12),
        "k15": get_allowed(k_k15),
        "var_a": get_allowed(k_var_a),
        "var_b": get_allowed(k_var_b),
    }

    del train_part, ctx_mc, X_tr, X_vl, preds_raw
    gc.collect()

    # 2. Sparse Indexing and Retrieval
    print("\n[2/4] Fitting Sparse Indices...")
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

    query_title_texts = field_text(val_contexts, ("search_query",))
    query_title_params_texts = field_text(
        val_contexts, ("search_query", "search_infm_params_text")
    )

    print("Retrieving global sparse candidates (depth=1000)...")
    global_word = field_index.branches["branch_a"].retrieve(query_title_texts, k=1000)
    global_char = title_char_index.retrieve(query_title_texts, k=1000)

    routed_sparse: dict[str, dict[str, list[list[tuple[str, float]]]]] = {}
    for pol_name, allowed_sets in policies.items():
        print(f"Retrieving routed sparse candidates for policy '{pol_name}' (depth=1500)...")
        r_w = field_index.branches["branch_a"].retrieve_routed(query_title_texts, allowed_sets, k=1500)
        r_c = title_char_index.retrieve_routed(query_title_texts, allowed_sets, k=1500)
        r_tp = field_index.branches["branch_b"].retrieve_routed(query_title_params_texts, allowed_sets, k=1500)
        routed_sparse[pol_name] = {
            "routed_word": r_w,
            "routed_char": r_c,
            "routed_title_params": r_tp,
        }

    del field_index, title_char_index
    gc.collect()

    # 3. Dense Retrieval
    print("\n[3/4] Dense Retrieval (FT-E5 and Generic E5)...")
    val_query_embeddings_generic = np.load("artifacts/research/val_query_e5_embeddings.npy")
    val_query_embeddings_ft = np.load(dense_dir / "val_query_ft_e5_embeddings.npy")

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

    print("Retrieving global dense candidates (depth=1000)...")
    global_ft_e5 = e5_retriever.retrieve_global(val_query_embeddings_ft, k=1000)
    global_gen_e5 = e5_retriever.retrieve_global(val_query_embeddings_generic, k=1000)

    routed_dense: dict[str, dict[str, list[list[tuple[str, float]]]]] = {}
    for pol_name, allowed_sets in policies.items():
        print(f"Retrieving routed dense candidates for policy '{pol_name}' (depth=1500)...")
        r_ft = e5_retriever.retrieve_routed(val_query_embeddings_ft, allowed_sets, k=1500)
        r_gen = e5_retriever.retrieve_routed(val_query_embeddings_generic, allowed_sets, k=1500)
        routed_dense[pol_name] = {
            "routed_ft_e5": r_ft,
            "routed_gen_e5": r_gen,
        }

    del e5_retriever
    gc.collect()

    # 4. Save Complete Candidate Cache
    print("\n[4/4] Assembling and saving cache...")
    cache_data = {
        "global_branches": {
            "global_ft_e5": global_ft_e5,
            "global_gen_e5": global_gen_e5,
            "global_word": global_word,
            "global_char": global_char,
        },
        "policies": {},
    }

    for pol_name in policies:
        cache_data["policies"][pol_name] = {
            "routed_ft_e5": routed_dense[pol_name]["routed_ft_e5"],
            "routed_gen_e5": routed_dense[pol_name]["routed_gen_e5"],
            "routed_char": routed_sparse[pol_name]["routed_char"],
            "routed_word": routed_sparse[pol_name]["routed_word"],
            "routed_title_params": routed_sparse[pol_name]["routed_title_params"],
        }

    cache_file = out_dir / "val_candidate_cache.pkl"
    with cache_file.open("wb") as f:
        pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"\nSaved complete candidate cache to {cache_file} ({cache_file.stat().st_size / (1024 * 1024):.1f} MB)")
    print(f"Total step time: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
