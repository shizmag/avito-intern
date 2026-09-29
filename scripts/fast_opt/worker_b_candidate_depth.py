"""Worker B: Hypothesis H2 - Candidate Depth & Budget Allocation.

Tests:
1. Standalone positive coverage and incremental positive coverage over union for each branch on unseen queries.
2. Global FT-E5 depth study: k in [100, 300, 500, 750, 1000]
3. Routed FT-E5 depth: k in [500, 750, 1000, 1500]
4. Routed Generic E5 depth: k in [0, 300, 500, 750, 1000]
5. Global Char depth: k in [100, 300, 500]
6. Joint candidate budget redistribution.

Computes:
- Standalone unseen coverage
- Incremental unseen coverage
- Seen R@50, Unseen R@50, Proxy
- Total candidate pool depth

Saves:
artifacts/fast_opt/results_h2_candidate_depth.json
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
    print("WORKER B: HYPOTHESIS H2 - CANDIDATE DEPTH & BUDGET ALLOCATION")
    print("=" * 80)

    cache_file = Path("artifacts/fast_opt/val_candidate_cache.pkl")
    if not cache_file.is_file():
        raise FileNotFoundError(f"Missing cache file {cache_file}")

    print("Loading candidate cache...")
    with cache_file.open("rb") as f:
        cache = pickle.load(f)

    evaluator = FastEvaluator()
    unseen_idx = evaluator.unseen_indices
    relevant = evaluator.relevant
    qids = evaluator.query_ids

    p_k10 = cache["policies"]["k10"]
    g = cache["global_branches"]

    branches = {
        "routed_ft_e5": p_k10["routed_ft_e5"],
        "routed_gen_e5": p_k10["routed_gen_e5"],
        "routed_char": p_k10["routed_char"],
        "routed_word": p_k10["routed_word"],
        "routed_title_params": p_k10["routed_title_params"],
        "global_ft_e5": g["global_ft_e5"],
        "global_gen_e5": g["global_gen_e5"],
        "global_word": g["global_word"],
        "global_char": g["global_char"],
    }

    # 1. Standalone and Incremental Coverage Analysis
    print("\n--- Part 1: Standalone & Incremental Coverage Analysis on Strict Unseen ---")

    def calc_coverage(candidate_sets: list[set[str]]) -> float:
        covs = []
        for q_idx in unseen_idx:
            qid = qids[q_idx]
            gt = relevant.get(qid, set())
            cands = candidate_sets[q_idx]
            covs.append(len(cands & gt) / max(len(gt), 1))
        return float(np.mean(covs))

    standalone_cov = {}
    for name, cand_list in branches.items():
        k_eval = 1000 if "routed" in name else 300
        c_sets = [{it for it, _ in row[:k_eval]} for row in cand_list]
        cov = calc_coverage(c_sets)
        standalone_cov[name] = cov
        print(f"  {name:<22} (k={k_eval:<4}) | Standalone Unseen Coverage: {cov * 100:.2f}%")

    # Incremental coverage over union of all other branches
    incremental_cov = {}
    for target_name in branches:
        # Union without target
        union_without = [set() for _ in range(len(qids))]
        for other_name, cand_list in branches.items():
            if other_name == target_name:
                continue
            k_other = 1000 if "routed" in other_name else 300
            for q_idx in range(len(qids)):
                union_without[q_idx].update(it for it, _ in cand_list[q_idx][:k_other])

        cov_without = calc_coverage(union_without)

        # Union with target
        k_target = 1000 if "routed" in target_name else 300
        union_with = [set(s) for s in union_without]
        for q_idx in range(len(qids)):
            union_with[q_idx].update(it for it, _ in branches[target_name][q_idx][:k_target])
        cov_with = calc_coverage(union_with)

        inc = cov_with - cov_without
        incremental_cov[target_name] = {
            "cov_with": cov_with,
            "cov_without": cov_without,
            "incremental_pp": inc * 100.0,
        }
        print(f"  {target_name:<22} | Inc over union: {inc * 100:+.2f} pp ({cov_without*100:.2f}% -> {cov_with*100:.2f}%)")

    # 2. Global FT-E5 depth study
    print("\n--- Part 2: Dedicated Global FT-E5 Budget Study ---")
    base_weights = [1.5, 0.8, 1.2, 0.8, 0.5, 0.4, 0.3, 0.3, 0.3]

    def run_budget_eval(k_dict: dict[str, int], weights: list[float], tag: str) -> dict[str, Any]:
        srcs = [
            [row[:k_dict["routed_ft_e5"]] for row in p_k10["routed_ft_e5"]],
            [row[:k_dict["routed_gen_e5"]] for row in p_k10["routed_gen_e5"]],
            [row[:k_dict["routed_char"]] for row in p_k10["routed_char"]],
            [row[:k_dict["routed_word"]] for row in p_k10["routed_word"]],
            [row[:k_dict["routed_title_params"]] for row in p_k10["routed_title_params"]],
            [row[:k_dict["global_ft_e5"]] for row in g["global_ft_e5"]],
            [row[:k_dict["global_gen_e5"]] for row in g["global_gen_e5"]],
            [row[:k_dict["global_word"]] for row in g["global_word"]],
            [row[:k_dict["global_char"]] for row in g["global_char"]],
        ]
        res = evaluator.evaluate_pipeline(srcs, weights, c=60.0)
        entry = {
            "tag": tag,
            "k_dict": dict(k_dict),
            "seen_r50": res["seen_r50"],
            "unseen_r50": res["unseen_r50"],
            "proxy": res["proxy"],
            "unseen_coverage_pool": res["unseen_coverage_pool"],
        }
        return entry

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

    base_eval = run_budget_eval(base_k, base_weights, "Baseline Depths (Routed 1000, Global 300)")
    print(
        f"BASELINE DEPTHS | Unseen R@50: {base_eval['unseen_r50']:.4f} | "
        f"Proxy: {base_eval['proxy']:.4f} | Unseen Cov: {base_eval['unseen_coverage_pool']:.4f}"
    )

    g_ft_results = []
    for k_g_ft in [100, 300, 500, 750, 1000]:
        k_d = dict(base_k)
        k_d["global_ft_e5"] = k_g_ft
        res = run_budget_eval(k_d, base_weights, f"Global FT-E5 k={k_g_ft}")
        d_unseen = (res["unseen_r50"] - base_eval["unseen_r50"]) * 100.0
        d_cov = (res["unseen_coverage_pool"] - base_eval["unseen_coverage_pool"]) * 100.0
        res["delta_unseen_pp"] = d_unseen
        res["delta_cov_pp"] = d_cov
        g_ft_results.append(res)
        print(
            f"  Global FT-E5 k={k_g_ft:<4} | Unseen R@50: {res['unseen_r50']:.4f} ({d_unseen:+6.2f} pp) | "
            f"Proxy: {res['proxy']:.4f} | Unseen Pool Cov: {res['unseen_coverage_pool']:.4f} ({d_cov:+6.2f} pp)"
        )

    # 3. Routed FT-E5 and Routed Gen E5 Budget Studies
    print("\n--- Part 3: Dense Branches Budget Grid ---")
    budget_grid_results = []
    for k_r_ft in [500, 750, 1000, 1500]:
        for k_r_gen in [0, 300, 500, 750, 1000]:
            for k_g_ft in [300, 500, 750, 1000]:
                k_d = dict(base_k)
                k_d["routed_ft_e5"] = k_r_ft
                k_d["routed_gen_e5"] = k_r_gen
                k_d["global_ft_e5"] = k_g_ft
                tag = f"R_FT={k_r_ft}, R_Gen={k_r_gen}, G_FT={k_g_ft}"
                res = run_budget_eval(k_d, base_weights, tag)
                res["delta_unseen_pp"] = (res["unseen_r50"] - base_eval["unseen_r50"]) * 100.0
                res["delta_proxy_pp"] = (res["proxy"] - base_eval["proxy"]) * 100.0
                budget_grid_results.append(res)

    sorted_budgets = sorted(budget_grid_results, key=lambda x: (x["unseen_r50"], x["proxy"]), reverse=True)
    best_budget = sorted_budgets[0]

    print("\n" + "=" * 80)
    print("TOP 3 CANDIDATE BUDGET CONFIGURATIONS")
    print("=" * 80)
    for rank, b in enumerate(sorted_budgets[:3], 1):
        print(
            f"#{rank} {b['tag']:<40} | Unseen R@50: {b['unseen_r50']:.4f} ({b['delta_unseen_pp']:+6.2f} pp) | "
            f"Proxy: {b['proxy']:.4f} | Unseen Cov: {b['unseen_coverage_pool']:.4f}"
        )

    out_payload = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "standalone_coverage": standalone_cov,
        "incremental_coverage": incremental_cov,
        "global_ft_e5_study": g_ft_results,
        "best_budget_config": best_budget,
        "top_3_budgets": sorted_budgets[:3],
        "runtime_sec": time.time() - t0,
    }

    out_file = Path("artifacts/fast_opt/results_h2_candidate_depth.json")
    with out_file.open("w", encoding="utf-8") as f:
        json.dump(out_payload, f, indent=2)

    print(f"\nSaved H2 results to {out_file} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
