"""Shared fast evaluation engine for fusion, geo-cascade ranking, and metrics."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from avito_candidate_generation.fusion.candidate_pool import reciprocal_rank_fusion


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


class FastEvaluator:
    def __init__(self, val_dir: str | Path = "artifacts/unseen_push/validation"):
        self.val_dir = Path(val_dir)
        self.val_contexts = pd.read_parquet(self.val_dir / "val_contexts.parquet")
        benchmark_items = pd.read_parquet("data/benchmark_items.parquet")

        with (self.val_dir / "ground_truth.json").open("r", encoding="utf-8") as f:
            self.relevant: dict[str, set[str]] = {k: set(v) for k, v in json.load(f).items()}

        with (self.val_dir / "loc_coords.json").open("r", encoding="utf-8") as f:
            self.loc_coords = {int(k): v for k, v in json.load(f).items()}

        self.query_ids = [str(x) for x in self.val_contexts["internal_query_id"].tolist()]
        self.is_seen = self.val_contexts["is_seen"].to_numpy()
        self.is_unseen = self.val_contexts["is_unseen"].to_numpy()
        self.unseen_indices = np.where(self.is_unseen)[0]
        self.seen_indices = np.where(self.is_seen)[0]

        item_id_list = [str(x) for x in benchmark_items["item_id"].tolist()]
        self.item_loc_map = dict(zip(item_id_list, benchmark_items["item_location_id"]))
        self.item_cat_map = dict(zip(item_id_list, benchmark_items["item_category_id"]))
        self.item_lat_map = dict(
            zip(
                item_id_list,
                benchmark_items["item_latitude"].astype(float).fillna(0.0),
            )
        )
        self.item_lon_map = dict(
            zip(
                item_id_list,
                benchmark_items["item_longitude"].astype(float).fillna(0.0),
            )
        )

        self.query_loc_map = dict(zip(self.query_ids, self.val_contexts["search_location_id"]))
        self.query_cat_map = dict(zip(self.query_ids, self.val_contexts["search_category"]))

        # Precompute query location coords
        self.query_lats: list[float | None] = []
        self.query_lons: list[float | None] = []
        for qid in self.query_ids:
            q_loc = self.query_loc_map[qid]
            q_coord = self.loc_coords.get(int(q_loc)) if pd.notna(q_loc) else None
            self.query_lats.append(q_coord["lat"] if q_coord else None)
            self.query_lons.append(q_coord["lon"] if q_coord else None)

    def evaluate_pipeline(
        self,
        sources: list[list[list[tuple[str, float]]]],
        weights: list[float],
        c: float = 60.0,
        pool_depth: int = 1000,
        return_per_query: bool = False,
    ) -> dict[str, Any]:
        """Runs RRF fusion + geo-cascade ranking + recall metrics calculation."""
        raw_rrf = reciprocal_rank_fusion(sources, weights=weights, c=c, limit=pool_depth)

        # Coverage at pool_depth on unseen
        unseen_cov_list = []
        for q_idx in self.unseen_indices:
            qid = self.query_ids[q_idx]
            pool = {it for it, _ in raw_rrf[q_idx]}
            gt = self.relevant.get(qid, set())
            unseen_cov_list.append(len(pool & gt) / max(len(gt), 1))
        unseen_coverage_pool = float(np.mean(unseen_cov_list))

        # Geo-cascade ranking
        per_query_r50: list[float] = []
        rec_seen50: list[float] = []
        rec_unseen50: list[float] = []

        for q_idx, (qid, ranking) in enumerate(zip(self.query_ids, raw_rrf, strict=True)):
            q_loc = self.query_loc_map[qid]
            q_cat = self.query_cat_map[qid]
            q_lat = self.query_lats[q_idx]
            q_lon = self.query_lons[q_idx]

            boosted: list[tuple[str, float]] = []
            for item_id, score in ranking:
                it_loc = self.item_loc_map.get(item_id)
                it_cat = self.item_cat_map.get(item_id)
                it_lat = self.item_lat_map.get(item_id, 0.0)
                it_lon = self.item_lon_map.get(item_id, 0.0)

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
            top50 = {it for it, _ in boosted[:50]}
            gt = self.relevant.get(qid, set())
            r50 = len(top50 & gt) / max(len(gt), 1)

            per_query_r50.append(r50)
            if self.is_unseen[q_idx]:
                rec_unseen50.append(r50)
            else:
                rec_seen50.append(r50)

        seen_r50 = float(np.mean(rec_seen50))
        unseen_r50 = float(np.mean(rec_unseen50))
        proxy = 0.35 * seen_r50 + 0.65 * unseen_r50

        res = {
            "seen_r50": seen_r50,
            "unseen_r50": unseen_r50,
            "proxy": proxy,
            "unseen_coverage_pool": unseen_coverage_pool,
        }
        if return_per_query:
            res["per_query_r50"] = np.array(per_query_r50, dtype=np.float64)
            res["unseen_r50_per_query"] = np.array(rec_unseen50, dtype=np.float64)
            res["seen_r50_per_query"] = np.array(rec_seen50, dtype=np.float64)
        return res

    @staticmethod
    def bootstrap_ci(
        baseline_scores: np.ndarray,
        candidate_scores: np.ndarray,
        n_boot: int = 1000,
        seed: int = 42,
    ) -> dict[str, float]:
        """Paired bootstrap confidence interval for difference (candidate - baseline)."""
        rng = np.random.default_rng(seed)
        n = len(baseline_scores)
        diffs = candidate_scores - baseline_scores
        observed_delta = float(np.mean(diffs))

        indices = rng.integers(0, n, size=(n_boot, n))
        boot_deltas = np.mean(diffs[indices], axis=1)

        ci_lower = float(np.percentile(boot_deltas, 2.5))
        ci_upper = float(np.percentile(boot_deltas, 97.5))
        p_val = float(np.mean(boot_deltas <= 0))

        return {
            "observed_delta": observed_delta,
            "ci_95_lower": ci_lower,
            "ci_95_upper": ci_upper,
            "p_value_one_sided": p_val,
        }
