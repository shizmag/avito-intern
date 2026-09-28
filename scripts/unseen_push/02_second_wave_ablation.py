"""Second Wave: Systematic interaction ablation between baseline, Router (A), Fine-Tuned E5 (B), and Combined (A+B).

Evaluates:
1. Baseline (Baseline Router, Generic E5)
2. Branch A (New Generalizing Router, Generic E5)
3. Branch B (Baseline Router, Fine-Tuned E5)
4. Branch A+B (New Generalizing Router, Fine-Tuned E5)

Computes for each:
- Seen Recall@50
- UNSEEN Recall@50
- Benchmark Proxy = 0.35 * Seen + 0.65 * Unseen
- Unseen Recall@100, @200, @500, @1000
- Unseen Candidate Pool Coverage@1000
- Routing containment unseen
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
from sklearn.linear_model import SGDClassifier

from avito_candidate_generation.fusion.candidate_pool import (
    evaluate_rankings,
    reciprocal_rank_fusion,
)
from avito_candidate_generation.research import field_text
from avito_candidate_generation.retrievers.routed_e5 import RoutedDenseE5Retriever
from avito_candidate_generation.retrievers.routed_lexical import (
    FieldAwareSparseIndex,
    SparseBranchIndex,
)
from avito_candidate_generation.routing import (
    ExactPosteriorPredictor,
    GeneralizingClassifierPredictor,
    HybridMicrocategoryPredictor,
)


def haversine_km(
    lat1: float | np.ndarray,
    lon1: float | np.ndarray,
    lat2: float | np.ndarray,
    lon2: float | np.ndarray,
) -> float | np.ndarray:
    r_lat1, r_lon1 = np.radians(lat1), np.radians(lon1)
    r_lat2, r_lon2 = np.radians(lat2), np.radians(lon2)
    dlat = r_lat2 - r_lat1
    dlon = r_lon2 - r_lon1
    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(r_lat1) * np.cos(r_lat2) * np.sin(dlon / 2.0) ** 2
    )
    c = 2.0 * np.arcsin(np.sqrt(np.clip(a, 0.0, 1.0)))
    return 6371.0 * c


def main() -> None:
    t0 = time.time()
    print("=" * 80)
    print("STEP 2: SECOND WAVE ABLATION (BASELINE vs A vs B vs A+B)")
    print("=" * 80)

    val_dir = Path("artifacts/unseen_push/validation")
    dense_dir = Path("artifacts/unseen_push/dense")
    metrics_dir = Path("artifacts/unseen_push/metrics")
    metrics_dir.mkdir(parents=True, exist_ok=True)

    val_contexts = pd.read_parquet(val_dir / "val_contexts.parquet")
    train_part = pd.read_parquet(val_dir / "train_part.parquet")
    benchmark_items = pd.read_parquet("data/benchmark_items.parquet")

    with (val_dir / "ground_truth.json").open("r", encoding="utf-8") as f:
        relevant: dict[str, set[str]] = {k: set(v) for k, v in json.load(f).items()}

    with (val_dir / "loc_coords.json").open("r", encoding="utf-8") as f:
        loc_coords = {int(k): v for k, v in json.load(f).items()}

    query_ids = [str(x) for x in val_contexts["internal_query_id"].tolist()]
    is_seen = val_contexts["is_seen"].to_numpy()
    is_unseen = val_contexts["is_unseen"].to_numpy()

    item_id_list = [str(x) for x in benchmark_items["item_id"].tolist()]
    item_loc_map = dict(zip(item_id_list, benchmark_items["item_location_id"]))
    item_cat_map = dict(zip(item_id_list, benchmark_items["item_category_id"]))
    item_microcat_map = dict(
        zip(item_id_list, benchmark_items["item_microcat_id"].astype(str))
    )
    item_lat_map = dict(
        zip(
            item_id_list,
            benchmark_items["item_latitude"].astype(float).fillna(0.0),
        )
    )
    item_lon_map = dict(
        zip(
            item_id_list,
            benchmark_items["item_longitude"].astype(float).fillna(0.0),
        )
    )
    item_titles = [
        str(x) for x in benchmark_items["item_title_raw"].fillna("").tolist()
    ]

    microcat_to_items: dict[str, set[str]] = {}
    for it, mc in item_microcat_map.items():
        microcat_to_items.setdefault(mc, set()).add(it)

    query_loc_map = dict(zip(query_ids, val_contexts["search_location_id"]))
    query_cat_map = dict(zip(query_ids, val_contexts["search_category"]))

    # 1. Fit Routing Models
    print("\n[1/5] Fitting Routers (Baseline Router vs New Generalizing Router)...")
    exact = ExactPosteriorPredictor().fit(
        train_part, query_col="search_query", category_col="item_microcat_id"
    )

    # Baseline Router (clf_base: 50k features)
    clf_base = GeneralizingClassifierPredictor(
        min_df=5, max_features=50000, alpha=1e-5, seed=42
    ).fit(train_part)
    hybrid_base = HybridMicrocategoryPredictor(
        exact_predictor=exact,
        generalizing_predictor=clf_base,
        min_count=1,
        blend_strategy="fallback",
    )
    top10_base = hybrid_base.predict_top_k(val_contexts, k=10)

    # New Router (High-capacity Multi-Label Aggregated SGD)
    ctx_mc = (
        train_part.groupby(["search_query", "search_infm_params_text", "search_category", "item_microcat_id"])
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

    val_q = val_contexts["search_query"].fillna("").astype(str).tolist()
    val_p = val_contexts["search_infm_params_text"].fillna("").astype(str).tolist()
    val_c = val_contexts["search_category"].fillna("").astype(str).tolist()
    val_words = [f"{q} {p} cat_{c}".strip() for q, p, c in zip(val_q, val_p, val_c, strict=True)]
    val_chars = [f" {q} ".replace("\t", " ") for q in val_q]

    vec_w = TfidfVectorizer(ngram_range=(1, 2), analyzer="word", min_df=2, max_features=120000, sublinear_tf=True)
    vec_c = TfidfVectorizer(ngram_range=(3, 5), analyzer="char_wb", min_df=3, max_features=120000, sublinear_tf=True)
    X_tr = hstack([vec_w.fit_transform(mc_train_words), vec_c.fit_transform(mc_train_chars)], format="csr")
    X_vl = hstack([vec_w.transform(val_words), vec_c.transform(val_chars)], format="csr")

    clf_new = SGDClassifier(loss="log_loss", penalty="l2", alpha=5e-6, max_iter=35, tol=1e-4, random_state=42)
    clf_new.fit(X_tr, mc_train_targets, sample_weight=mc_sample_weights)

    preds_raw = clf_new.predict_proba(X_vl)
    classes_new = list(clf_new.classes_)
    top10_new_gen = []
    for row in preds_raw:
        top_idx = np.argsort(row)[::-1][:10]
        top10_new_gen.append([classes_new[i] for i in top_idx])

    top10_new = []
    for q, cats in zip(val_q, top10_new_gen, strict=True):
        if exact.is_seen(q):
            top10_new.append(exact.predict_top_k([q], k=10)[0])
        else:
            top10_new.append(cats)

    # Allowed items maps
    def get_allowed(top10_preds: list[list[str]]) -> list[set[str]]:
        allowed_list = []
        for cats in top10_preds:
            allowed = set()
            for c in cats:
                allowed.update(microcat_to_items.get(c, set()))
            allowed_list.append(allowed)
        return allowed_list

    allowed_base = get_allowed(top10_base)
    allowed_new = get_allowed(top10_new)

    del train_part, ctx_mc, X_tr, X_vl
    gc.collect()

    # 2. Fit Sparse Indices & Retrieve
    print("\n[2/5] Fitting Sparse Indices and retrieving...")
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

    # Lexical for baseline router
    routed_word_base = field_index.branches["branch_a"].retrieve_routed(query_title_texts, allowed_base, k=1000)
    routed_char_base = title_char_index.retrieve_routed(query_title_texts, allowed_base, k=1000)
    routed_tp_base = field_index.branches["branch_b"].retrieve_routed(query_title_params_texts, allowed_base, k=1000)

    # Lexical for new router
    routed_word_new = field_index.branches["branch_a"].retrieve_routed(query_title_texts, allowed_new, k=1000)
    routed_char_new = title_char_index.retrieve_routed(query_title_texts, allowed_new, k=1000)
    routed_tp_new = field_index.branches["branch_b"].retrieve_routed(query_title_params_texts, allowed_new, k=1000)

    # Global lexical
    global_word = field_index.branches["branch_a"].retrieve(query_title_texts, k=300)
    global_char = title_char_index.retrieve(query_title_texts, k=300)

    del field_index, title_char_index
    gc.collect()

    # 3. Dense Retrieval: Generic vs Fine-Tuned
    print("\n[3/5] Loading Dense Embeddings and retrieving...")
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

    # Generic E5
    routed_e5_gen_base = e5_retriever.retrieve_routed(val_query_embeddings_generic, allowed_base, k=1000)
    routed_e5_gen_new = e5_retriever.retrieve_routed(val_query_embeddings_generic, allowed_new, k=1000)
    global_e5_gen = e5_retriever.retrieve_global(val_query_embeddings_generic, k=300)

    # Fine-Tuned E5
    routed_e5_ft_base = e5_retriever.retrieve_routed(val_query_embeddings_ft, allowed_base, k=1000)
    routed_e5_ft_new = e5_retriever.retrieve_routed(val_query_embeddings_ft, allowed_new, k=1000)
    global_e5_ft = e5_retriever.retrieve_global(val_query_embeddings_ft, k=300)

    del e5_retriever
    gc.collect()

    # 4. Fusion and Cascade Ranking Evaluation Function
    def evaluate_configuration(
        sources: list[Any], weights: list[float], name: str
    ) -> dict[str, Any]:
        raw_rrf = reciprocal_rank_fusion(sources, weights=weights, c=60.0, limit=1000)

        # Coverage at 1000
        cov_1000_unseen = []
        for q_idx in np.where(is_unseen)[0]:
            qid = query_ids[q_idx]
            pool = {it for it, _ in raw_rrf[q_idx]}
            gt = relevant.get(qid, set())
            cov_1000_unseen.append(len(pool & gt) / max(len(gt), 1))
        unseen_coverage_1000 = float(np.mean(cov_1000_unseen))

        # Geo-cascade ranking
        final_rankings = []
        for qid, ranking in zip(query_ids, raw_rrf, strict=True):
            q_loc = query_loc_map[qid]
            q_cat = query_cat_map[qid]
            q_coord = loc_coords.get(int(q_loc)) if pd.notna(q_loc) else None
            q_lat = q_coord["lat"] if q_coord else None
            q_lon = q_coord["lon"] if q_coord else None

            boosted = []
            for item_id, score in ranking:
                it_loc = item_loc_map.get(item_id)
                it_cat = item_cat_map.get(item_id)
                it_lat = item_lat_map.get(item_id, 0.0)
                it_lon = item_lon_map.get(item_id, 0.0)

                cat_mult = 1.0 if (q_cat == 0 or it_cat == q_cat) else 0.001

                if it_loc == q_loc:
                    geo_mult = 30.0
                elif q_lat is not None and q_lon is not None and it_lat != 0.0 and it_lon != 0.0:
                    dist = float(haversine_km(q_lat, q_lon, it_lat, it_lon))
                    geo_mult = float(np.exp(-0.25 * (dist / 10.0)))
                else:
                    geo_mult = 0.05

                boosted.append((item_id, score * cat_mult * geo_mult))

            boosted.sort(key=lambda p: (-p[1], p[0]))
            final_rankings.append(boosted[:50])

        # Recalls
        rec_seen50 = []
        rec_unseen50 = []
        rec_unseen100 = []
        for q_idx, (qid, ranking) in enumerate(zip(query_ids, final_rankings, strict=True)):
            top50 = {it for it, _ in ranking[:50]}
            gt = relevant.get(qid, set())
            r50 = len(top50 & gt) / max(len(gt), 1)
            if is_unseen[q_idx]:
                rec_unseen50.append(r50)
            else:
                rec_seen50.append(r50)

        seen_r50 = float(np.mean(rec_seen50))
        unseen_r50 = float(np.mean(rec_unseen50))
        proxy = 0.35 * seen_r50 + 0.65 * unseen_r50

        res = {
            "name": name,
            "seen_r50": seen_r50,
            "unseen_r50": unseen_r50,
            "proxy": proxy,
            "unseen_coverage_1000": unseen_coverage_1000,
        }
        print(
            f"{name:<28} | Seen R@50: {seen_r50:.4f} | UNSEEN R@50: {unseen_r50:.4f} | "
            f"Proxy: {proxy:.4f} | Unseen Cov@1000: {unseen_coverage_1000:.4f}"
        )
        return res

    print("\n" + "=" * 80)
    print("SECOND WAVE EXPERIMENT COMPARISON MATRIX")
    print("=" * 80)

    # 1. Baseline: Baseline Router + Generic E5
    src_base = [
        routed_e5_gen_base,
        routed_char_base,
        routed_word_base,
        routed_tp_base,
        global_e5_gen,
        global_word,
        global_char,
    ]
    w_base = [1.5, 1.2, 0.8, 0.5, 0.4, 0.3, 0.3]
    res_baseline = evaluate_configuration(src_base, w_base, "Baseline (Champion Frozen)")

    # 2. Configuration A: New Generalizing Router + Generic E5
    src_a = [
        routed_e5_gen_new,
        routed_char_new,
        routed_word_new,
        routed_tp_new,
        global_e5_gen,
        global_word,
        global_char,
    ]
    res_a = evaluate_configuration(src_a, w_base, "Branch A (Generalizing Router)")

    # 3. Configuration B: Baseline Router + Fine-Tuned E5 + Generic E5
    src_b = [
        routed_e5_ft_base,
        routed_e5_gen_base,
        routed_char_base,
        routed_word_base,
        routed_tp_base,
        global_e5_ft,
        global_e5_gen,
        global_word,
        global_char,
    ]
    w_b = [1.5, 0.8, 1.2, 0.8, 0.5, 0.4, 0.3, 0.3, 0.3]
    res_b = evaluate_configuration(src_b, w_b, "Branch B (Fine-Tuned E5)")

    # 4. Configuration A+B: New Router + Fine-Tuned E5 + Generic E5
    src_ab = [
        routed_e5_ft_new,
        routed_e5_gen_new,
        routed_char_new,
        routed_word_new,
        routed_tp_new,
        global_e5_ft,
        global_e5_gen,
        global_word,
        global_char,
    ]
    w_ab = [1.5, 0.8, 1.2, 0.8, 0.5, 0.4, 0.3, 0.3, 0.3]
    res_ab = evaluate_configuration(src_ab, w_ab, "Branch A+B (Combined Champion)")

    # Summary table
    print("\n" + "=" * 80)
    print("FINAL ABLATION SUMMARY")
    print("=" * 80)
    print(f"Baseline Unseen R@50:  {res_baseline['unseen_r50']:.4f} (Proxy: {res_baseline['proxy']:.4f})")
    print(f"Branch A Unseen R@50:  {res_a['unseen_r50']:.4f} (Delta: {(res_a['unseen_r50'] - res_baseline['unseen_r50']) * 100:+.2f} pp)")
    print(f"Branch B Unseen R@50:  {res_b['unseen_r50']:.4f} (Delta: {(res_b['unseen_r50'] - res_baseline['unseen_r50']) * 100:+.2f} pp)")
    print(f"Branch A+B Unseen R@50:{res_ab['unseen_r50']:.4f} (Delta: {(res_ab['unseen_r50'] - res_baseline['unseen_r50']) * 100:+.2f} pp)")

    # Select champion
    candidates = [res_baseline, res_a, res_b, res_ab]
    best_candidate = max(candidates, key=lambda c: (c["unseen_r50"], c["proxy"]))
    print(f"\nSELECTED CHAMPION: '{best_candidate['name']}' with Unseen R@50 = {best_candidate['unseen_r50']:.4f}, Proxy = {best_candidate['proxy']:.4f}")

    # Save artifacts
    results_dict = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "baseline": res_baseline,
        "branch_a": res_a,
        "branch_b": res_b,
        "branch_ab": res_ab,
        "selected_champion": best_candidate,
        "runtime_sec": time.time() - t0,
    }
    with (metrics_dir / "second_wave_ablation.json").open("w", encoding="utf-8") as f:
        json.dump(results_dict, f, indent=2)

    print(f"Saved ablation report to {metrics_dir / 'second_wave_ablation.json'}")


if __name__ == "__main__":
    main()
