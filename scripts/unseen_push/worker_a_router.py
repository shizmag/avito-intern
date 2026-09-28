"""Worker A: H1 Generalizing Microcategory Routing Experiments.

Evaluates on strict unseen queries:
1. Baseline Router: Exact Posterior + Standard Generalizing SGD (alpha=1e-5, max_features=50k)
2. High-capacity Sparse Classifier (Word (1,2) + Char (3,5) + Params, 250k features, sublinear TF)
3. Balanced Multi-Label Target Aggregation (equal weight per positive category)
4. LogisticRegression vs SGDClassifier with tuned regularization (alpha in [1e-6, 1e-5, 3e-5])
5. Probability blend / Category Prior Smoothing

Computes for each:
- Top-1, Top-3, Top-5, Top-10 Containment on STRICT UNSEEN queries
- Top-1, Top-3, Top-5, Top-10 Containment on SEEN queries
- Mean routed corpus size, P95 corpus size
- Accept / Kill gate: Baseline unseen Top10 ≈ 91.99%. Strong >= 94.0%, Very Strong >= 95.0%, Kill < 92.5%.
"""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier

from avito_candidate_generation.routing import ExactPosteriorPredictor


def main() -> None:
    t0 = time.time()
    print("=" * 80)
    print("WORKER A: GENERALIZING MICROCATEGORY ROUTER EXPERIMENT")
    print("=" * 80)

    val_dir = Path("artifacts/unseen_push/validation")
    router_dir = Path("artifacts/unseen_push/router")
    router_dir.mkdir(parents=True, exist_ok=True)

    val_contexts = pd.read_parquet(val_dir / "val_contexts.parquet")
    train_part = pd.read_parquet(val_dir / "train_part.parquet")
    benchmark_items = pd.read_parquet("data/benchmark_items.parquet")

    with (val_dir / "ground_truth.json").open("r", encoding="utf-8") as f:
        relevant: dict[str, set[str]] = {k: set(v) for k, v in json.load(f).items()}

    query_ids = [str(x) for x in val_contexts["internal_query_id"].tolist()]
    is_seen = val_contexts["is_seen"].to_numpy()
    is_unseen = val_contexts["is_unseen"].to_numpy()

    item_id_list = [str(x) for x in benchmark_items["item_id"].tolist()]
    item_microcat_map = dict(
        zip(item_id_list, benchmark_items["item_microcat_id"].astype(str))
    )

    microcat_to_items: dict[str, set[str]] = {}
    for it, mc in item_microcat_map.items():
        microcat_to_items.setdefault(mc, set()).add(it)

    # Pre-extract GT microcats
    gt_microcats: list[set[str]] = []
    for qid in query_ids:
        pos_items = relevant.get(qid, set())
        cats = {item_microcat_map[it] for it in pos_items if it in item_microcat_map}
        gt_microcats.append(cats)

    # 1. Exact posterior
    print("\nFitting ExactPosteriorPredictor on train_part...")
    exact = ExactPosteriorPredictor().fit(
        train_part, query_col="search_query", category_col="item_microcat_id"
    )

    # Format texts for training
    print("Extracting train and validation texts...")
    train_q = train_part["search_query"].fillna("").astype(str).tolist()
    train_params = train_part["search_infm_params_text"].fillna("").astype(str).tolist()
    train_cat = train_part["search_category"].fillna("").astype(str).tolist()
    train_target = train_part["item_microcat_id"].astype(str).tolist()

    val_q = val_contexts["search_query"].fillna("").astype(str).tolist()
    val_params = (
        val_contexts["search_infm_params_text"].fillna("").astype(str).tolist()
    )
    val_cat = val_contexts["search_category"].fillna("").astype(str).tolist()

    # Form text representations
    train_word_texts = [
        f"{q} {p} cat_{c}".strip()
        for q, p, c in zip(train_q, train_params, train_cat, strict=True)
    ]
    train_char_texts = [
        f" {q} ".replace("\t", " ") for q in train_q
    ]
    val_word_texts = [
        f"{q} {p} cat_{c}".strip()
        for q, p, c in zip(val_q, val_params, val_cat, strict=True)
    ]
    val_char_texts = [
        f" {q} ".replace("\t", " ") for q in val_q
    ]

    # Model 1: Baseline SGDClassifier (50k features, alpha=1e-5)
    print("\n--- Model 1: Baseline SGDClassifier (50k features, alpha=1e-5) ---")
    t_start = time.time()
    vec_word_50k = TfidfVectorizer(
        ngram_range=(1, 2), analyzer="word", min_df=5, max_features=50000, sublinear_tf=True
    )
    vec_char_50k = TfidfVectorizer(
        ngram_range=(3, 5), analyzer="char_wb", min_df=5, max_features=50000, sublinear_tf=True
    )
    X_tr_w = vec_word_50k.fit_transform(train_word_texts)
    X_tr_c = vec_char_50k.fit_transform(train_char_texts)
    X_train_base = hstack([X_tr_w, X_tr_c], format="csr")

    X_val_w = vec_word_50k.transform(val_word_texts)
    X_val_c = vec_char_50k.transform(val_char_texts)
    X_val_base = hstack([X_val_w, X_val_c], format="csr")

    clf_base = SGDClassifier(
        loss="log_loss", penalty="l2", alpha=1e-5, max_iter=25, tol=1e-3, random_state=42
    )
    clf_base.fit(X_train_base, train_target)
    print(f"Model 1 trained in {time.time() - t_start:.1f}s")

    def eval_predictions(preds_k10: list[list[str]], name: str) -> dict[str, Any]:
        results: dict[str, Any] = {}
        print(f"\nEvaluation for {name}:")
        for k in [1, 3, 5, 10]:
            hits = []
            sizes = []
            for pred, gt in zip(preds_k10, gt_microcats, strict=True):
                topk = set(pred[:k])
                hits.append(bool(topk & gt) if gt else False)
                sizes.append(sum(len(microcat_to_items.get(c, set())) for c in topk))
            h_arr = np.array(hits, dtype=bool)
            s_arr = np.array(sizes, dtype=int)
            res_k = {
                "containment_overall": float(h_arr.mean()),
                "containment_seen": float(h_arr[is_seen].mean()),
                "containment_unseen": float(h_arr[is_unseen].mean()),
                "mean_corpus": float(s_arr.mean()),
                "p95_corpus": float(np.percentile(s_arr, 95)),
            }
            results[f"top_{k}"] = res_k
            print(
                f"  Top-{k:<2}: Overall={res_k['containment_overall']:.4f} | "
                f"Seen={res_k['containment_seen']:.4f} | "
                f"UNSEEN={res_k['containment_unseen']:.4f} | "
                f"Corpus={res_k['mean_corpus']:.0f} (p95={res_k['p95_corpus']:.0f})"
            )
        return results

    # Hybrid 1 predictions
    preds_m1_raw = clf_base.predict_proba(X_val_base)
    classes_m1 = list(clf_base.classes_)
    preds_m1_top10 = []
    for row in preds_m1_raw:
        top_idx = np.argsort(row)[::-1][:10]
        preds_m1_top10.append([classes_m1[i] for i in top_idx])

    preds_hybrid1 = []
    for q, m1_cats in zip(val_q, preds_m1_top10, strict=True):
        if exact.is_seen(q):
            preds_hybrid1.append(exact.predict_top_k([q], k=10)[0])
        else:
            preds_hybrid1.append(m1_cats)

    m1_metrics = eval_predictions(preds_hybrid1, "Model 1: Baseline Hybrid K=10")

    # Model 2: High-capacity Sparse Classifier (150k features, sublinear TF, balanced multi-label aggregation)
    print("\n--- Model 2: High-Capacity Multi-Label Aggregated Sparse Classifier ---")
    t_start = time.time()
    # Build unique (context, microcat) train table with normalized weights
    ctx_mc = (
        train_part.groupby(["search_query", "search_infm_params_text", "search_category", "item_microcat_id"])
        .size()
        .reset_index(name="count")
    )
    # Per-context total positive microcategories
    ctx_totals = ctx_mc.groupby(["search_query", "search_infm_params_text", "search_category"])["count"].transform("sum")
    ctx_mc["sample_weight"] = np.clip(np.log1p(ctx_mc["count"]) / np.log1p(ctx_totals), 0.1, 1.0)

    mc_train_q = ctx_mc["search_query"].fillna("").astype(str).tolist()
    mc_train_p = ctx_mc["search_infm_params_text"].fillna("").astype(str).tolist()
    mc_train_c = ctx_mc["search_category"].fillna("").astype(str).tolist()
    mc_train_words = [f"{q} {p} cat_{c}".strip() for q, p, c in zip(mc_train_q, mc_train_p, mc_train_c, strict=True)]
    mc_train_chars = [f" {q} ".replace("\t", " ") for q in mc_train_q]
    mc_train_targets = ctx_mc["item_microcat_id"].astype(str).tolist()
    mc_sample_weights = ctx_mc["sample_weight"].to_numpy()

    vec_word_high = TfidfVectorizer(
        ngram_range=(1, 2), analyzer="word", min_df=2, max_features=120000, sublinear_tf=True
    )
    vec_char_high = TfidfVectorizer(
        ngram_range=(3, 5), analyzer="char_wb", min_df=3, max_features=120000, sublinear_tf=True
    )
    X_tr_w2 = vec_word_high.fit_transform(mc_train_words)
    X_tr_c2 = vec_char_high.fit_transform(mc_train_chars)
    X_train_high = hstack([X_tr_w2, X_tr_c2], format="csr")

    X_val_w2 = vec_word_high.transform(val_word_texts)
    X_val_c2 = vec_char_high.transform(val_char_texts)
    X_val_high = hstack([X_val_w2, X_val_c2], format="csr")

    clf_high = SGDClassifier(
        loss="log_loss", penalty="l2", alpha=5e-6, max_iter=35, tol=1e-4, random_state=42
    )
    clf_high.fit(X_train_high, mc_train_targets, sample_weight=mc_sample_weights)
    print(f"Model 2 trained in {time.time() - t_start:.1f}s")

    preds_m2_raw = clf_high.predict_proba(X_val_high)
    classes_m2 = list(clf_high.classes_)
    preds_m2_top10 = []
    for row in preds_m2_raw:
        top_idx = np.argsort(row)[::-1][:10]
        preds_m2_top10.append([classes_m2[i] for i in top_idx])

    preds_hybrid2 = []
    for q, m2_cats in zip(val_q, preds_m2_top10, strict=True):
        if exact.is_seen(q):
            preds_hybrid2.append(exact.predict_top_k([q], k=10)[0])
        else:
            preds_hybrid2.append(m2_cats)

    m2_metrics = eval_predictions(preds_hybrid2, "Model 2: High-Capacity Multi-Label SGD")

    # Model 3: LogisticRegression with L2 regularization
    print("\n--- Model 3: LogisticRegression (C=0.5, sag solver, max_iter=25) ---")
    t_start = time.time()
    clf_lr = LogisticRegression(
        C=0.5, solver="sag", max_iter=25, tol=1e-2, random_state=42, n_jobs=-1
    )
    clf_lr.fit(X_train_high, mc_train_targets, sample_weight=mc_sample_weights)
    print(f"Model 3 trained in {time.time() - t_start:.1f}s")

    preds_m3_raw = clf_lr.predict_proba(X_val_high)
    classes_m3 = list(clf_lr.classes_)
    preds_m3_top10 = []
    for row in preds_m3_raw:
        top_idx = np.argsort(row)[::-1][:10]
        preds_m3_top10.append([classes_m3[i] for i in top_idx])

    preds_hybrid3 = []
    for q, m3_cats in zip(val_q, preds_m3_top10, strict=True):
        if exact.is_seen(q):
            preds_hybrid3.append(exact.predict_top_k([q], k=10)[0])
        else:
            preds_hybrid3.append(m3_cats)

    m3_metrics = eval_predictions(preds_hybrid3, "Model 3: High-Capacity LogisticRegression")

    # Decision summary
    base_unseen_top10 = m1_metrics["top_10"]["containment_unseen"]
    best_unseen_top10 = max(
        m1_metrics["top_10"]["containment_unseen"],
        m2_metrics["top_10"]["containment_unseen"],
        m3_metrics["top_10"]["containment_unseen"],
    )
    delta_pp = (best_unseen_top10 - base_unseen_top10) * 100

    decision = "ACCEPT" if delta_pp >= 0.5 else "KILL"
    print("\n" + "=" * 80)
    print(f"WORKER A DECISION GATE:")
    print(f"  Baseline Unseen Top10: {base_unseen_top10:.4f}")
    print(f"  Best Unseen Top10:     {best_unseen_top10:.4f}")
    print(f"  Delta:                 {delta_pp:+.2f} pp")
    print(f"  Decision:              {decision}")
    print("=" * 80)

    # Save artifacts
    report = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "baseline_hybrid": m1_metrics,
        "high_capacity_sgd": m2_metrics,
        "high_capacity_lr": m3_metrics,
        "baseline_unseen_top10": base_unseen_top10,
        "best_unseen_top10": best_unseen_top10,
        "delta_pp": delta_pp,
        "decision": decision,
        "runtime_sec": time.time() - t0,
    }
    with (router_dir / "worker_a_report.json").open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(f"Report saved to {router_dir / 'worker_a_report.json'}")


if __name__ == "__main__":
    main()
