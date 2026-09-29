"""Probe H3 Adaptive Routing containment and corpus size on validation set."""

from __future__ import annotations

import gc
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from scipy.stats import entropy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier

from avito_candidate_generation.routing import ExactPosteriorPredictor


def main() -> None:
    t0 = time.time()
    val_dir = Path("artifacts/unseen_push/validation")
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

    gt_microcats: list[set[str]] = []
    for qid in query_ids:
        pos_items = relevant.get(qid, set())
        cats = {item_microcat_map[it] for it in pos_items if it in item_microcat_map}
        gt_microcats.append(cats)

    # 1. Exact posterior
    exact = ExactPosteriorPredictor().fit(
        train_part, query_col="search_query", category_col="item_microcat_id"
    )

    # 2. Train Generalizing Router
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

    # Extract top 20 candidates per query and confidence signals
    top20_gen = []
    top1_probs = []
    margins = []
    entropies = []

    for row in preds_raw:
        sorted_indices = np.argsort(row)[::-1]
        top20_gen.append([classes_new[i] for i in sorted_indices[:20]])
        p1 = float(row[sorted_indices[0]])
        p2 = float(row[sorted_indices[1]])
        top1_probs.append(p1)
        margins.append(p1 - p2)
        entropies.append(float(entropy(row + 1e-12)))

    top1_probs = np.array(top1_probs)
    margins = np.array(margins)
    entropies = np.array(entropies)

    # For seen queries, exact posterior is used.
    # Hybrid top-20:
    hybrid_top20 = []
    for q, gen_cats in zip(val_q, top20_gen, strict=True):
        if exact.is_seen(q):
            hybrid_top20.append(exact.predict_top_k([q], k=20)[0])
        else:
            hybrid_top20.append(gen_cats)

    def evaluate_policy(policy_name: str, k_per_query: list[int] | np.ndarray) -> dict[str, Any]:
        hits_seen, hits_unseen = [], []
        corpus_sizes_unseen = []
        for q_idx, (k, cats, gt) in enumerate(zip(k_per_query, hybrid_top20, gt_microcats, strict=True)):
            chosen_cats = set(cats[:k])
            hit = bool(chosen_cats & gt) if gt else False
            if is_unseen[q_idx]:
                hits_unseen.append(hit)
                corpus_sizes_unseen.append(sum(len(microcat_to_items.get(c, set())) for c in chosen_cats))
            else:
                hits_seen.append(hit)

        unseen_containment = float(np.mean(hits_unseen))
        seen_containment = float(np.mean(hits_seen))
        mean_corpus_unseen = float(np.mean(corpus_sizes_unseen))
        p95_corpus_unseen = float(np.percentile(corpus_sizes_unseen, 95))
        mean_k_unseen = float(np.mean([k_per_query[i] for i in np.where(is_unseen)[0]]))

        print(
            f"{policy_name:<35} | Mean K: {mean_k_unseen:.1f} | "
            f"Unseen Cont: {unseen_containment * 100:.2f}% | "
            f"Seen Cont: {seen_containment * 100:.2f}% | "
            f"Mean Corpus: {mean_corpus_unseen:.0f} (p95: {p95_corpus_unseen:.0f})"
        )
        return {
            "policy": policy_name,
            "mean_k_unseen": mean_k_unseen,
            "unseen_containment": unseen_containment,
            "seen_containment": seen_containment,
            "mean_corpus_unseen": mean_corpus_unseen,
            "p95_corpus_unseen": p95_corpus_unseen,
        }

    print("\n" + "=" * 95)
    print("ROUTING POLICIES CONTAINMENT EVALUATION ON UNSEEN VALIDATION")
    print("=" * 95)

    # Fixed Ks
    evaluate_policy("Fixed K=5", [5] * len(val_q))
    evaluate_policy("Fixed K=8", [8] * len(val_q))
    evaluate_policy("Fixed K=10 (Baseline Champion)", [10] * len(val_q))
    evaluate_policy("Fixed K=12", [12] * len(val_q))
    evaluate_policy("Fixed K=15", [15] * len(val_q))

    # Adaptive policies by Top1 Prob
    p_33, p_66 = np.percentile(top1_probs[is_unseen], [33.3, 66.7])
    print(f"\nTop1 prob quantiles on unseen: 33%={p_33:.4f}, 66%={p_66:.4f}")

    def make_adaptive_k(signal: np.ndarray, q_low: float, q_high: float, k_high: int, k_med: int, k_low: int, higher_is_confident: bool = True) -> list[int]:
        ks = []
        for i, val in enumerate(signal):
            if higher_is_confident:
                if val >= q_high:
                    ks.append(k_high)
                elif val >= q_low:
                    ks.append(k_med)
                else:
                    ks.append(k_low)
            else:
                if val <= q_low:
                    ks.append(k_high)
                elif val <= q_high:
                    ks.append(k_med)
                else:
                    ks.append(k_low)
        return ks

    # Variant A: High K5, Med K10, Low K15
    k_var_a_prob = make_adaptive_k(top1_probs, p_33, p_66, k_high=5, k_med=10, k_low=15)
    evaluate_policy("Variant A (Top1 Prob 5/10/15)", k_var_a_prob)

    # Variant B: High K8, Med K10, Low K12
    k_var_b_prob = make_adaptive_k(top1_probs, p_33, p_66, k_high=8, k_med=10, k_low=12)
    evaluate_policy("Variant B (Top1 Prob 8/10/12)", k_var_b_prob)

    # Variant A on margin
    m_33, m_66 = np.percentile(margins[is_unseen], [33.3, 66.7])
    k_var_a_marg = make_adaptive_k(margins, m_33, m_66, k_high=5, k_med=10, k_low=15)
    evaluate_policy("Variant A (Margin 5/10/15)", k_var_a_marg)
    k_var_b_marg = make_adaptive_k(margins, m_33, m_66, k_high=8, k_med=10, k_low=12)
    evaluate_policy("Variant B (Margin 8/10/12)", k_var_b_marg)

    # Variant A on entropy (lower entropy = higher confidence)
    e_33, e_66 = np.percentile(entropies[is_unseen], [33.3, 66.7])
    k_var_a_ent = make_adaptive_k(entropies, e_33, e_66, k_high=5, k_med=10, k_low=15, higher_is_confident=False)
    evaluate_policy("Variant A (Entropy 5/10/15)", k_var_a_ent)
    k_var_b_ent = make_adaptive_k(entropies, e_33, e_66, k_high=8, k_med=10, k_low=12, higher_is_confident=False)
    evaluate_policy("Variant B (Entropy 8/10/12)", k_var_b_ent)

    # Save probe results
    out = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "runtime_sec": time.time() - t0,
    }
    with open("artifacts/fast_opt/probe_adaptive_routing.json", "w") as f:
        json.dump(out, f, indent=2)


if __name__ == "__main__":
    main()
