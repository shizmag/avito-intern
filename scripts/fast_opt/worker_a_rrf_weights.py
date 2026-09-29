"""Worker A: Hypothesis H1 - RRF Weights Optimization.

Tests optimization of RRF source weights following domain fine-tuned E5 introduction.
Systematic coordinate and staged grid search:
- FT-E5 weight: [1.2, 1.5, 1.8, 2.1, 2.5, 3.0]
- Generic E5 weight: [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
- Char weight: [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
- Global FT-E5 weight: [0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
- Fallback weights: [0.1, 0.2, 0.3, 0.4]

Computes:
- Seen Recall@50
- Strict Unseen Recall@50
- Benchmark Proxy = 0.35 * Seen + 0.65 * Unseen
- Unseen Candidate Pool Coverage@1000
- Paired bootstrap 95% CI for top 3 configs vs baseline

Saves:
artifacts/fast_opt/results_h1_rrf_weights.json
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
    print("WORKER A: HYPOTHESIS H1 - RRF WEIGHTS OPTIMIZATION")
    print("=" * 80)

    cache_file = Path("artifacts/fast_opt/val_candidate_cache.pkl")
    if not cache_file.is_file():
        raise FileNotFoundError(f"Missing cache file {cache_file}")

    print("Loading candidate cache...")
    with cache_file.open("rb") as f:
        cache = pickle.load(f)

    evaluator = FastEvaluator()

    # Slice sources to baseline depth: routed k=1000, global k=300
    p_k10 = cache["policies"]["k10"]
    g = cache["global_branches"]

    def slice_source(src: list[list[tuple[str, float]]], k: int) -> list[list[tuple[str, float]]]:
        return [row[:k] for row in src]

    base_sources = [
        slice_source(p_k10["routed_ft_e5"], 1000),       # idx 0: routed FT-E5
        slice_source(p_k10["routed_gen_e5"], 1000),      # idx 1: routed Gen E5
        slice_source(p_k10["routed_char"], 1000),        # idx 2: routed Char
        slice_source(p_k10["routed_word"], 1000),        # idx 3: routed Word
        slice_source(p_k10["routed_title_params"], 1000),# idx 4: routed Title+Params
        slice_source(g["global_ft_e5"], 300),            # idx 5: global FT-E5
        slice_source(g["global_gen_e5"], 300),           # idx 6: global Gen E5
        slice_source(g["global_word"], 300),             # idx 7: global Word
        slice_source(g["global_char"], 300),             # idx 8: global Char
    ]

    source_names = [
        "routed_ft_e5",
        "routed_gen_e5",
        "routed_char",
        "routed_word",
        "routed_title_params",
        "global_ft_e5",
        "global_gen_e5",
        "global_word",
        "global_char",
    ]

    base_weights = [1.5, 0.8, 1.2, 0.8, 0.5, 0.4, 0.3, 0.3, 0.3]

    print("\nEvaluating Baseline Champion configuration...")
    base_res = evaluator.evaluate_pipeline(base_sources, base_weights, c=60.0, return_per_query=True)
    print(
        f"BASELINE | Seen R@50: {base_res['seen_r50']:.4f} | "
        f"UNSEEN R@50: {base_res['unseen_r50']:.4f} | "
        f"Proxy: {base_res['proxy']:.4f} | "
        f"Unseen Cov@1000: {base_res['unseen_coverage_pool']:.4f}"
    )

    tested_configs: list[dict[str, Any]] = []

    def test_weight_vector(w: list[float], tag: str) -> dict[str, Any]:
        res = evaluator.evaluate_pipeline(base_sources, w, c=60.0, return_per_query=False)
        delta_unseen = res["unseen_r50"] - base_res["unseen_r50"]
        delta_proxy = res["proxy"] - base_res["proxy"]
        entry = {
            "tag": tag,
            "weights": [round(x, 3) for x in w],
            "seen_r50": res["seen_r50"],
            "unseen_r50": res["unseen_r50"],
            "proxy": res["proxy"],
            "delta_unseen_pp": delta_unseen * 100.0,
            "delta_proxy_pp": delta_proxy * 100.0,
            "unseen_coverage_pool": res["unseen_coverage_pool"],
        }
        tested_configs.append(entry)
        return entry

    # 1. Coordinate Search: FT-E5 weight
    print("\n--- Phase 1: Routed FT-E5 Weight Variation ---")
    current_w = list(base_weights)
    for w_ft in [1.0, 1.2, 1.5, 1.8, 2.1, 2.5, 3.0]:
        cand_w = list(current_w)
        cand_w[0] = w_ft
        e = test_weight_vector(cand_w, f"FT-E5 weight {w_ft}")
        print(f"  FT-E5 {w_ft:<4} -> Unseen: {e['unseen_r50']:.4f} ({e['delta_unseen_pp']:+6.2f} pp) | Proxy: {e['proxy']:.4f}")

    best_ft = max(tested_configs, key=lambda x: (x["unseen_r50"], x["proxy"]))
    current_w = list(best_ft["weights"])
    print(f"Best FT-E5 weight: {current_w[0]}")

    # 2. Coordinate Search: Generic E5 weight
    print("\n--- Phase 2: Routed Generic E5 Weight Variation ---")
    for w_gen in [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]:
        cand_w = list(current_w)
        cand_w[1] = w_gen
        e = test_weight_vector(cand_w, f"Generic E5 weight {w_gen}")
        print(f"  Gen-E5 {w_gen:<4} -> Unseen: {e['unseen_r50']:.4f} ({e['delta_unseen_pp']:+6.2f} pp) | Proxy: {e['proxy']:.4f}")

    best_gen = max(tested_configs, key=lambda x: (x["unseen_r50"], x["proxy"]))
    current_w = list(best_gen["weights"])
    print(f"Best Generic E5 weight: {current_w[1]}")

    # 3. Coordinate Search: Char TF-IDF weight
    print("\n--- Phase 3: Routed Char Weight Variation ---")
    for w_char in [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8]:
        cand_w = list(current_w)
        cand_w[2] = w_char
        e = test_weight_vector(cand_w, f"Char weight {w_char}")
        print(f"  Char {w_char:<4} -> Unseen: {e['unseen_r50']:.4f} ({e['delta_unseen_pp']:+6.2f} pp) | Proxy: {e['proxy']:.4f}")

    best_char = max(tested_configs, key=lambda x: (x["unseen_r50"], x["proxy"]))
    current_w = list(best_char["weights"])
    print(f"Best Char weight: {current_w[2]}")

    # 4. Coordinate Search: Global FT-E5 weight
    print("\n--- Phase 4: Global FT-E5 Weight Variation ---")
    for w_g_ft in [0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.5]:
        cand_w = list(current_w)
        cand_w[5] = w_g_ft
        e = test_weight_vector(cand_w, f"Global FT-E5 weight {w_g_ft}")
        print(f"  Global FT-E5 {w_g_ft:<4} -> Unseen: {e['unseen_r50']:.4f} ({e['delta_unseen_pp']:+6.2f} pp) | Proxy: {e['proxy']:.4f}")

    best_g_ft = max(tested_configs, key=lambda x: (x["unseen_r50"], x["proxy"]))
    current_w = list(best_g_ft["weights"])
    print(f"Best Global FT-E5 weight: {current_w[5]}")

    # 5. Coordinate Search: Other Global Fallback weights
    print("\n--- Phase 5: Global Fallback Weights Variation ---")
    for w_fb in [0.1, 0.2, 0.3, 0.4, 0.5]:
        cand_w = list(current_w)
        cand_w[6] = w_fb  # global gen e5
        cand_w[7] = w_fb  # global word
        cand_w[8] = w_fb  # global char
        e = test_weight_vector(cand_w, f"Global fallbacks weight {w_fb}")
        print(f"  Fallbacks {w_fb:<4} -> Unseen: {e['unseen_r50']:.4f} ({e['delta_unseen_pp']:+6.2f} pp) | Proxy: {e['proxy']:.4f}")

    best_fb = max(tested_configs, key=lambda x: (x["unseen_r50"], x["proxy"]))
    current_w = list(best_fb["weights"])

    # 6. Joint Fine Grid around best coordinate
    print("\n--- Phase 6: Joint Grid Refinement ---")
    w0_vals = [current_w[0] - 0.3, current_w[0], current_w[0] + 0.3]
    w1_vals = [max(0.0, current_w[1] - 0.2), current_w[1], current_w[1] + 0.2]
    w2_vals = [current_w[2] - 0.2, current_w[2], current_w[2] + 0.2]
    w5_vals = [current_w[5] - 0.2, current_w[5], current_w[5] + 0.2]

    for w0 in w0_vals:
        for w1 in w1_vals:
            for w2 in w2_vals:
                for w5 in w5_vals:
                    cand_w = list(current_w)
                    cand_w[0] = round(w0, 3)
                    cand_w[1] = round(w1, 3)
                    cand_w[2] = round(w2, 3)
                    cand_w[5] = round(w5, 3)
                    test_weight_vector(cand_w, f"Grid w0={w0:.1f},w1={w1:.1f},w2={w2:.1f},w5={w5:.1f}")

    # Sort all configs by (unseen_r50, proxy)
    sorted_configs = sorted(tested_configs, key=lambda x: (x["unseen_r50"], x["proxy"]), reverse=True)
    top_3 = sorted_configs[:3]

    print("\n" + "=" * 80)
    print("TOP 3 FUSION WEIGHT CONFIGURATIONS (WITH BOOTSTRAP CI)")
    print("=" * 80)
    base_unseen_scores = base_res["unseen_r50_per_query"]

    for rank, cfg in enumerate(top_3, 1):
        # Run per query for bootstrap
        res_detailed = evaluator.evaluate_pipeline(
            base_sources, cfg["weights"], c=60.0, return_per_query=True
        )
        ci_res = FastEvaluator.bootstrap_ci(
            base_unseen_scores, res_detailed["unseen_r50_per_query"], n_boot=1000
        )
        cfg["bootstrap_ci"] = ci_res
        print(
            f"#{rank} {cfg['tag']:<45} | Weights: {cfg['weights']} | "
            f"Unseen R@50: {cfg['unseen_r50']:.4f} ({cfg['delta_unseen_pp']:+6.2f} pp) | "
            f"Proxy: {cfg['proxy']:.4f} | 95% CI: [{ci_res['ci_95_lower']*100:+.2f}, {ci_res['ci_95_upper']*100:+.2f}] pp (p={ci_res['p_value_one_sided']:.4f})"
        )

    best_cfg = top_3[0]
    out_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "baseline": {
            "weights": base_weights,
            "seen_r50": base_res["seen_r50"],
            "unseen_r50": base_res["unseen_r50"],
            "proxy": base_res["proxy"],
            "unseen_coverage_pool": base_res["unseen_coverage_pool"],
        },
        "best_config": best_cfg,
        "top_3": top_3,
        "total_configs_evaluated": len(tested_configs),
        "runtime_sec": time.time() - t0,
    }

    out_file = Path("artifacts/fast_opt/results_h1_rrf_weights.json")
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)

    print(f"\nSaved H1 results to {out_file} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
