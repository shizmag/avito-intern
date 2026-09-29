"""Worker D: Hypotheses H4 (RRF constant c), H5 (Generic E5 ablation), H6 (Branch Overlap Analysis).

Evaluates:
- H4: RRF constant c in [20, 40, 60, 80, 100]
- H5: Generic E5 ablation:
  1. Full dense (FT-E5 + Generic E5)
  2. FT-E5 only dense (Generic E5 weight = 0, depth = 0)
- H6: Overlap analysis between retriever branches on top-50:
  - FT-E5 vs Generic E5
  - FT-E5 vs Char TF-IDF
  - Generic E5 vs Char TF-IDF
  - Mean Jaccard / Overlap count on Seen vs Unseen queries

Saves:
artifacts/fast_opt/results_h4_h5_h6.json
"""

from __future__ import annotations

import json
import pickle
import time
from pathlib import Path
from typing import Any

import numpy as np

from eval_helper import FastEvaluator


def main() -> None:
    t0 = time.time()
    print("=" * 80)
    print("WORKER D: HYPOTHESES H4, H5, H6 - RRF C, ABLATIONS & OVERLAP ANALYSIS")
    print("=" * 80)

    cache_file = Path("artifacts/fast_opt/val_candidate_cache.pkl")
    if not cache_file.is_file():
        raise FileNotFoundError(f"Missing cache file {cache_file}")

    print("Loading candidate cache...")
    with cache_file.open("rb") as f:
        cache = pickle.load(f)

    evaluator = FastEvaluator()
    p_k10 = cache["policies"]["k10"]
    g = cache["global_branches"]

    def slice_source(src: list[list[tuple[str, float]]], k: int) -> list[list[tuple[str, float]]]:
        return [row[:k] for row in src]

    base_sources = [
        slice_source(p_k10["routed_ft_e5"], 1000),
        slice_source(p_k10["routed_gen_e5"], 1000),
        slice_source(p_k10["routed_char"], 1000),
        slice_source(p_k10["routed_word"], 1000),
        slice_source(p_k10["routed_title_params"], 1000),
        slice_source(g["global_ft_e5"], 300),
        slice_source(g["global_gen_e5"], 300),
        slice_source(g["global_word"], 300),
        slice_source(g["global_char"], 300),
    ]
    base_weights = [1.5, 0.8, 1.2, 0.8, 0.5, 0.4, 0.3, 0.3, 0.3]

    # Baseline evaluation
    base_res = evaluator.evaluate_pipeline(base_sources, base_weights, c=60.0)
    print(
        f"BASELINE (c=60, with Gen-E5) | Seen: {base_res['seen_r50']:.4f} | "
        f"UNSEEN: {base_res['unseen_r50']:.4f} | Proxy: {base_res['proxy']:.4f}"
    )

    # 1. H4: RRF Constant c
    print("\n--- Part 1: H4 - RRF Constant c Evaluation ---")
    c_results = []
    for c_val in [20.0, 40.0, 60.0, 80.0, 100.0]:
        res = evaluator.evaluate_pipeline(base_sources, base_weights, c=c_val)
        delta = (res["unseen_r50"] - base_res["unseen_r50"]) * 100.0
        c_entry = {
            "c": c_val,
            "seen_r50": res["seen_r50"],
            "unseen_r50": res["unseen_r50"],
            "proxy": res["proxy"],
            "delta_unseen_pp": delta,
        }
        c_results.append(c_entry)
        print(
            f"  c={c_val:<5.1f} | Unseen R@50: {res['unseen_r50']:.4f} ({delta:+6.2f} pp) | "
            f"Seen R@50: {res['seen_r50']:.4f} | Proxy: {res['proxy']:.4f}"
        )
    best_c = max(c_results, key=lambda x: (x["unseen_r50"], x["proxy"]))
    print(f"Best RRF constant: c={best_c['c']} (Delta: {best_c['delta_unseen_pp']:+6.2f} pp)")

    # 2. H5: Generic E5 Ablation
    print("\n--- Part 2: H5 - Generic E5 Ablation ---")
    # Ablation: Zero out Generic E5 (routed and global)
    no_gen_weights = [1.5, 0.0, 1.2, 0.8, 0.5, 0.4, 0.0, 0.3, 0.3]
    res_no_gen = evaluator.evaluate_pipeline(base_sources, no_gen_weights, c=60.0)
    delta_no_gen = (res_no_gen["unseen_r50"] - base_res["unseen_r50"]) * 100.0
    print(
        f"  FT-E5 Only (No Generic E5)  | Unseen R@50: {res_no_gen['unseen_r50']:.4f} ({delta_no_gen:+6.2f} pp) | "
        f"Seen R@50: {res_no_gen['seen_r50']:.4f} | Proxy: {res_no_gen['proxy']:.4f}"
    )

    # Also test scaled down Generic E5 weights
    scaled_gen_weights = [1.5, 0.4, 1.2, 0.8, 0.5, 0.4, 0.15, 0.3, 0.3]
    res_scaled_gen = evaluator.evaluate_pipeline(base_sources, scaled_gen_weights, c=60.0)
    delta_scaled = (res_scaled_gen["unseen_r50"] - base_res["unseen_r50"]) * 100.0
    print(
        f"  Half Generic E5 (w=0.4/0.15)| Unseen R@50: {res_scaled_gen['unseen_r50']:.4f} ({delta_scaled:+6.2f} pp) | "
        f"Seen R@50: {res_scaled_gen['seen_r50']:.4f} | Proxy: {res_scaled_gen['proxy']:.4f}"
    )

    h5_summary = {
        "with_generic_e5": base_res,
        "without_generic_e5": res_no_gen,
        "half_generic_e5": res_scaled_gen,
        "delta_without_pp": delta_no_gen,
        "delta_half_pp": delta_scaled,
    }

    # 3. H6: Branch Overlap Analysis on Top-50
    print("\n--- Part 3: H6 - Branch Overlap Analysis (Top-50 Candidates) ---")
    routed_ft = [set(it for it, _ in row[:50]) for row in p_k10["routed_ft_e5"]]
    routed_gen = [set(it for it, _ in row[:50]) for row in p_k10["routed_gen_e5"]]
    routed_chr = [set(it for it, _ in row[:50]) for row in p_k10["routed_char"]]

    is_unseen = evaluator.is_unseen
    unseen_idx = evaluator.unseen_indices
    seen_idx = evaluator.seen_indices

    def compute_overlap(sets_a: list[set[str]], sets_b: list[set[str]]) -> dict[str, float]:
        overlaps = [len(a & b) for a, b in zip(sets_a, sets_b, strict=True)]
        jaccards = [len(a & b) / max(len(a | b), 1) for a, b in zip(sets_a, sets_b, strict=True)]
        arr_ov = np.array(overlaps)
        arr_jc = np.array(jaccards)
        return {
            "mean_overlap_overall": float(arr_ov.mean()),
            "mean_overlap_seen": float(arr_ov[seen_idx].mean()),
            "mean_overlap_unseen": float(arr_ov[unseen_idx].mean()),
            "mean_jaccard_unseen": float(arr_jc[unseen_idx].mean()),
        }

    ov_ft_gen = compute_overlap(routed_ft, routed_gen)
    ov_ft_chr = compute_overlap(routed_ft, routed_chr)
    ov_gen_chr = compute_overlap(routed_gen, routed_chr)

    print(f"  FT-E5 vs Generic E5 | Unseen Overlap: {ov_ft_gen['mean_overlap_unseen']:.1f}/50 (Jaccard: {ov_ft_gen['mean_jaccard_unseen']:.3f}) | Seen Overlap: {ov_ft_gen['mean_overlap_seen']:.1f}/50")
    print(f"  FT-E5 vs Char TFIDF | Unseen Overlap: {ov_ft_chr['mean_overlap_unseen']:.1f}/50 (Jaccard: {ov_ft_chr['mean_jaccard_unseen']:.3f}) | Seen Overlap: {ov_ft_chr['mean_overlap_seen']:.1f}/50")
    print(f"  Gen E5 vs Char TFIDF| Unseen Overlap: {ov_gen_chr['mean_overlap_unseen']:.1f}/50 (Jaccard: {ov_gen_chr['mean_jaccard_unseen']:.3f}) | Seen Overlap: {ov_gen_chr['mean_overlap_seen']:.1f}/50")

    out_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "h4_rrf_c": {
            "results": c_results,
            "best_c": best_c,
        },
        "h5_generic_e5_ablation": h5_summary,
        "h6_overlap_analysis": {
            "ft_vs_generic_e5": ov_ft_gen,
            "ft_vs_char": ov_ft_chr,
            "gen_vs_char": ov_gen_chr,
        },
        "runtime_sec": time.time() - t0,
    }

    out_file = Path("artifacts/fast_opt/results_h4_h5_h6.json")
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)

    print(f"\nSaved H4, H5, H6 results to {out_file} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
