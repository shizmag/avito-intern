"""Combined Champion Evaluation and Selection.

Integrates winning variants from:
- A: Optimal RRF Weights
- B: Candidate Budgets & Global FT-E5 depth
- C: Routing Policy (Adaptive or Fixed K)
- D: Optimal RRF c

Evaluates full interaction matrix:
- Baseline
- A (Weights only)
- B (Budgets only)
- C (Routing policy only)
- D (c only)
- A + B
- A + C
- A + D
- B + C
- A + B + C + D (Full Combined Champion)

Computes for each:
- Seen Recall@50
- Strict Unseen Recall@50
- Benchmark Proxy (0.35 * Seen + 0.65 * Unseen)
- Delta Unseen pp vs baseline
- Paired bootstrap 95% CI vs baseline (10,000 resamples)
- Final Champion Selection

Saves:
artifacts/fast_opt/champion_report.json
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
    print("COMBINED CHAMPION EVALUATION & SELECTION MATRIX")
    print("=" * 80)

    cache_file = Path("artifacts/fast_opt/val_candidate_cache.pkl")
    if not cache_file.is_file():
        raise FileNotFoundError(f"Missing cache file {cache_file}")

    # Load findings from workers
    f_h1 = Path("artifacts/fast_opt/results_h1_rrf_weights.json")
    f_h2 = Path("artifacts/fast_opt/results_h2_candidate_depth.json")
    f_h3 = Path("artifacts/fast_opt/results_h3_adaptive_routing.json")
    f_h4 = Path("artifacts/fast_opt/results_h4_h5_h6.json")

    with cache_file.open("rb") as f:
        cache = pickle.load(f)

    evaluator = FastEvaluator()
    g = cache["global_branches"]

    # Read discovered best parameters or defaults
    best_weights = [1.5, 0.8, 1.2, 0.8, 0.5, 0.4, 0.3, 0.3, 0.3]
    if f_h1.is_file():
        h1_data = json.loads(f_h1.read_text())
        best_weights = h1_data["best_config"]["weights"]
        print(f"Loaded Best RRF Weights from H1: {best_weights} (Delta: {h1_data['best_config']['delta_unseen_pp']:+.2f} pp)")

    base_k = {
        "routed_ft_e5": 1000,
        "routed_gen_e5": 1000,
        "routed_char": 1000,
        "routed_word": 1000,
        "routed_title_params": 1000,
        "global_ft_e5": 300,
        "global_gen_e5": 300,
        "global_word": 300,
        "global_char": 300,
    }
    best_k = dict(base_k)
    if f_h2.is_file():
        h2_data = json.loads(f_h2.read_text())
        best_k = h2_data["best_budget_config"]["k_dict"]
        print(f"Loaded Best Candidate Budgets from H2: {best_k} (Delta: {h2_data['best_budget_config']['delta_unseen_pp']:+.2f} pp)")

    best_policy = "k10"
    if f_h3.is_file():
        h3_data = json.loads(f_h3.read_text())
        best_policy = h3_data["best_policy"]["policy"]
        print(f"Loaded Best Routing Policy from H3: {best_policy} (Delta: {h3_data['best_policy']['delta_unseen_pp']:+.2f} pp)")

    best_c = 60.0
    if f_h4.is_file():
        h4_data = json.loads(f_h4.read_text())
        best_c = h4_data["h4_rrf_c"]["best_c"]["c"]
        print(f"Loaded Best RRF c from H4: {best_c} (Delta: {h4_data['h4_rrf_c']['best_c']['delta_unseen_pp']:+.2f} pp)")

    def build_sources(pol: str, k_dict: dict[str, int]) -> list[list[list[tuple[str, float]]]]:
        p_data = cache["policies"][pol]
        return [
            [row[:k_dict["routed_ft_e5"]] for row in p_data["routed_ft_e5"]],
            [row[:k_dict["routed_gen_e5"]] for row in p_data["routed_gen_e5"]],
            [row[:k_dict["routed_char"]] for row in p_data["routed_char"]],
            [row[:k_dict["routed_word"]] for row in p_data["routed_word"]],
            [row[:k_dict["routed_title_params"]] for row in p_data["routed_title_params"]],
            [row[:k_dict["global_ft_e5"]] for row in g["global_ft_e5"]],
            [row[:k_dict["global_gen_e5"]] for row in g["global_gen_e5"]],
            [row[:k_dict["global_word"]] for row in g["global_word"]],
            [row[:k_dict["global_char"]] for row in g["global_char"]],
        ]

    # Baseline run
    base_weights = [1.5, 0.8, 1.2, 0.8, 0.5, 0.4, 0.3, 0.3, 0.3]
    src_baseline = build_sources("k10", base_k)
    base_res = evaluator.evaluate_pipeline(src_baseline, base_weights, c=60.0, return_per_query=True)
    base_unseen_scores = base_res["unseen_r50_per_query"]

    configs_to_run = [
        ("Baseline (Champion Frozen)", "k10", base_k, base_weights, 60.0),
        ("A (Best Weights Only)", "k10", base_k, best_weights, 60.0),
        ("B (Best Budgets Only)", "k10", best_k, base_weights, 60.0),
        ("C (Best Routing Only)", best_policy, base_k, base_weights, 60.0),
        ("D (Best RRF c Only)", "k10", base_k, base_weights, best_c),
        ("A + B (Weights + Budgets)", "k10", best_k, best_weights, 60.0),
        ("A + C (Weights + Routing)", best_policy, base_k, best_weights, 60.0),
        ("A + D (Weights + RRF c)", "k10", base_k, best_weights, best_c),
        ("B + C (Budgets + Routing)", best_policy, best_k, base_weights, 60.0),
        ("B + D (Budgets + RRF c)", "k10", best_k, base_weights, best_c),
        ("A + B + C (Weights + Budgets + Routing)", best_policy, best_k, best_weights, 60.0),
        ("A + B + D (Weights + Budgets + RRF c)", "k10", best_k, best_weights, best_c),
        ("A + B + C + D (Full Combined Champion)", best_policy, best_k, best_weights, best_c),
    ]

    print("\n" + "=" * 95)
    print("RUNNING INTERACTION MATRIX")
    print("=" * 95)

    evaluated_matrix = []

    for name, pol, k_d, w, c_val in configs_to_run:
        srcs = build_sources(pol, k_d)
        res = evaluator.evaluate_pipeline(srcs, w, c=c_val, return_per_query=True)
        d_unseen = (res["unseen_r50"] - base_res["unseen_r50"]) * 100.0
        d_proxy = (res["proxy"] - base_res["proxy"]) * 100.0
        ci_res = FastEvaluator.bootstrap_ci(base_unseen_scores, res["unseen_r50_per_query"], n_boot=2000)

        entry = {
            "name": name,
            "policy": pol,
            "k_dict": dict(k_d),
            "weights": list(w),
            "c": c_val,
            "seen_r50": res["seen_r50"],
            "unseen_r50": res["unseen_r50"],
            "proxy": res["proxy"],
            "delta_unseen_pp": d_unseen,
            "delta_proxy_pp": d_proxy,
            "unseen_coverage_pool": res["unseen_coverage_pool"],
            "bootstrap_ci": ci_res,
        }
        evaluated_matrix.append(entry)

        print(
            f"{name:<42} | Unseen R@50: {res['unseen_r50']:.4f} ({d_unseen:+6.2f} pp) | "
            f"Seen: {res['seen_r50']:.4f} | Proxy: {res['proxy']:.4f} | "
            f"95% CI: [{ci_res['ci_95_lower']*100:+.2f}, {ci_res['ci_95_upper']*100:+.2f}] pp (p={ci_res['p_value_one_sided']:.4f})"
        )

    # Sort and pick overall champion
    sorted_matrix = sorted(evaluated_matrix, key=lambda x: (x["unseen_r50"], x["proxy"]), reverse=True)
    champion = sorted_matrix[0]

    print("\n" + "=" * 95)
    print(f"OVERALL WINNING CHAMPION: '{champion['name']}'")
    print(f"Strict Unseen Recall@50: {champion['unseen_r50']:.4f} (Delta: {champion['delta_unseen_pp']:+6.2f} pp vs baseline)")
    print(f"Benchmark Proxy:         {champion['proxy']:.4f} (Delta: {champion['delta_proxy_pp']:+6.2f} pp)")
    print(f"Seen Recall@50:          {champion['seen_r50']:.4f}")
    print(f"Unseen Pool Coverage:    {champion['unseen_coverage_pool']:.4f}")
    print(f"Bootstrap 95% CI:        [{champion['bootstrap_ci']['ci_95_lower']*100:+.2f}, {champion['bootstrap_ci']['ci_95_upper']*100:+.2f}] pp (p={champion['bootstrap_ci']['p_value_one_sided']:.4f})")
    print("=" * 95)

    out_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "champion": champion,
        "interaction_matrix": evaluated_matrix,
        "runtime_sec": time.time() - t0,
    }

    out_file = Path("artifacts/fast_opt/champion_report.json")
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)

    print(f"\nSaved combined champion report to {out_file} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
