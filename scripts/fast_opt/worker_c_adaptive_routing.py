"""Worker C: Hypothesis H3 - Adaptive Microcategory Routing Policy.

Evaluates end-to-end candidate retrieval, fusion, and geo-cascade ranking across routing policies:
- Policy 1: Fixed K=10 (Baseline Champion)
- Policy 2: Variant A (Adaptive Top1 Prob: K=5 / 10 / 15)
- Policy 3: Variant B (Adaptive Top1 Prob: K=8 / 10 / 12)
- Policy 4: Fixed K=12
- Policy 5: Fixed K=15

Metrics computed:
- Unseen routing containment
- Unseen candidate pool coverage@1000
- Seen R@50
- Strict Unseen R@50
- Benchmark Proxy (0.35 * Seen + 0.65 * Unseen)
- Delta Unseen R@50 vs baseline K=10
- Paired bootstrap CI vs baseline

Saves:
artifacts/fast_opt/results_h3_adaptive_routing.json
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
    print("WORKER C: HYPOTHESIS H3 - ADAPTIVE ROUTING EVALUATION")
    print("=" * 80)

    cache_file = Path("artifacts/fast_opt/val_candidate_cache.pkl")
    if not cache_file.is_file():
        raise FileNotFoundError(f"Missing cache file {cache_file}")

    print("Loading candidate cache...")
    with cache_file.open("rb") as f:
        cache = pickle.load(f)

    evaluator = FastEvaluator()
    g = cache["global_branches"]

    base_weights = [1.5, 0.8, 1.2, 0.8, 0.5, 0.4, 0.3, 0.3, 0.3]

    def slice_source(src: list[list[tuple[str, float]]], k: int) -> list[list[tuple[str, float]]]:
        return [row[:k] for row in src]

    # Evaluate each routing policy
    policies = ["k10", "var_a", "var_b", "k12", "k15"]
    policy_labels = {
        "k10": "Fixed K=10 (Baseline Champion)",
        "var_a": "Variant A (Adaptive 5 / 10 / 15)",
        "var_b": "Variant B (Adaptive 8 / 10 / 12)",
        "k12": "Fixed K=12",
        "k15": "Fixed K=15",
    }

    results: dict[str, Any] = {}
    base_unseen_scores: np.ndarray | None = None

    for pol in policies:
        p_data = cache["policies"][pol]
        sources = [
            slice_source(p_data["routed_ft_e5"], 1000),
            slice_source(p_data["routed_gen_e5"], 1000),
            slice_source(p_data["routed_char"], 1000),
            slice_source(p_data["routed_word"], 1000),
            slice_source(p_data["routed_title_params"], 1000),
            slice_source(g["global_ft_e5"], 300),
            slice_source(g["global_gen_e5"], 300),
            slice_source(g["global_word"], 300),
            slice_source(g["global_char"], 300),
        ]

        res = evaluator.evaluate_pipeline(sources, base_weights, c=60.0, return_per_query=True)

        if pol == "k10":
            base_unseen_scores = res["unseen_r50_per_query"]
            delta_unseen = 0.0
            ci_res = {"observed_delta": 0.0, "ci_95_lower": 0.0, "ci_95_upper": 0.0, "p_value_one_sided": 1.0}
        else:
            assert base_unseen_scores is not None
            delta_unseen = res["unseen_r50"] - results["k10"]["unseen_r50"]
            ci_res = FastEvaluator.bootstrap_ci(base_unseen_scores, res["unseen_r50_per_query"])

        pol_res = {
            "policy": pol,
            "label": policy_labels[pol],
            "seen_r50": res["seen_r50"],
            "unseen_r50": res["unseen_r50"],
            "proxy": res["proxy"],
            "unseen_coverage_pool": res["unseen_coverage_pool"],
            "delta_unseen_pp": delta_unseen * 100.0,
            "bootstrap_ci": ci_res,
        }
        results[pol] = pol_res

        print(
            f"{policy_labels[pol]:<35} | Unseen R@50: {res['unseen_r50']:.4f} ({delta_unseen * 100:+6.2f} pp) | "
            f"Seen R@50: {res['seen_r50']:.4f} | Proxy: {res['proxy']:.4f} | "
            f"Unseen Pool Cov: {res['unseen_coverage_pool']:.4f}"
        )
        if pol != "k10":
            print(f"    95% CI: [{ci_res['ci_95_lower']*100:+.2f}, {ci_res['ci_95_upper']*100:+.2f}] pp (p={ci_res['p_value_one_sided']:.4f})")

    # Select best policy
    best_pol = max(results.values(), key=lambda x: (x["unseen_r50"], x["proxy"]))
    print("\n" + "=" * 80)
    print(f"BEST ROUTING POLICY: {best_pol['label']} (Delta: {best_pol['delta_unseen_pp']:+6.2f} pp)")
    print("=" * 80)

    out_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "policies_evaluated": results,
        "best_policy": best_pol,
        "runtime_sec": time.time() - t0,
    }

    out_file = Path("artifacts/fast_opt/results_h3_adaptive_routing.json")
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)

    print(f"\nSaved H3 results to {out_file} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
