"""Stage 04: Candidate Fusion, Geo-Cascade Ranking, and Final answer.csv Generation.

Fuses sparse and dense candidate branches with Weighted RRF (c=60.0, limit=1000).
Applies Hyperlocal Geo-Cascade Ranking with Category 0 Wildcard Fix:
- Category 0 (wildcard all categories): no category penalty
- Other categories: valid category priority
- Hyperlocal matching: same location bonus = 30.0, nearby exponential decay = 0.25
- Strictly produces 50 unique items per query
Writes:
- artifacts/unseen_push/final/answer.csv
- answer.csv (root)
"""

from __future__ import annotations

import csv
import gc
import pickle
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd

from avito_candidate_generation.fusion.candidate_pool import reciprocal_rank_fusion


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
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
    print("STAGE 04: CANDIDATE FUSION AND LOCAL-FIRST GEO CASCADE RANKING")
    print("=" * 80)

    in_dir = Path("artifacts/unseen_push/benchmark")
    final_dir = Path("artifacts/unseen_push/final")
    final_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load candidates
    print("Loading sparse and dense candidate pools...")
    with (in_dir / "sparse_candidates.pkl").open("rb") as f:
        sparse_dict = pickle.load(f)

    with (in_dir / "dense_candidates.pkl").open("rb") as f:
        dense_dict = pickle.load(f)

    # 2. Candidate Fusion with Champion Weights
    print("Performing Champion Weighted RRF (c=60.0, limit=1000)...")
    sources = [
        dense_dict["routed_ft_e5"],
        dense_dict["routed_gen_e5"],
        sparse_dict["routed_char"],
        sparse_dict["routed_word"],
        sparse_dict["routed_title_params"],
        dense_dict["global_ft_e5"],
        dense_dict["global_gen_e5"],
        sparse_dict["global_word"],
        sparse_dict["global_char"],
    ]
    weights = [1.5, 0.8, 1.2, 0.8, 0.5, 0.4, 0.3, 0.3, 0.3]
    raw_rrf = reciprocal_rank_fusion(sources, weights=weights, c=60.0, limit=1000)

    del sparse_dict, dense_dict, sources
    gc.collect()

    # 3. Load metadata for geo-cascade
    print("Loading metadata and location coordinates...")
    benchmark_queries = pd.read_parquet("data/benchmark_queries.parquet")
    benchmark_items = pd.read_parquet("data/benchmark_items.parquet")
    train = pd.read_parquet("data/train.parquet")

    query_id_list = [str(x) for x in benchmark_queries["query_id"].tolist()]
    query_loc_map = dict(zip(query_id_list, benchmark_queries["search_location_id"]))
    query_cat_map = dict(zip(query_id_list, benchmark_queries["search_category"]))

    item_id_list = [str(x) for x in benchmark_items["item_id"].tolist()]
    item_loc_map = dict(zip(item_id_list, benchmark_items["item_location_id"]))
    item_cat_map = dict(zip(item_id_list, benchmark_items["item_category_id"]))
    item_lat_map: dict[str, float] = {
        str(it): float(lat) if pd.notna(lat) else 0.0
        for it, lat in zip(item_id_list, benchmark_items["item_latitude"].tolist(), strict=True)
    }
    item_lon_map: dict[str, float] = {
        str(it): float(lon) if pd.notna(lon) else 0.0
        for it, lon in zip(item_id_list, benchmark_items["item_longitude"].tolist(), strict=True)
    }

    # Location mean coordinates
    loc_geo = (
        train.dropna(subset=["item_latitude", "item_longitude"])
        .groupby("item_location_id")[["item_latitude", "item_longitude"]]
        .mean()
    )
    loc_lat_dict = loc_geo["item_latitude"].astype(float).to_dict()
    loc_lon_dict = loc_geo["item_longitude"].astype(float).to_dict()

    del train, loc_geo
    gc.collect()

    # 4. Local-First Hyperlocal Cascade Ranking
    print("Applying Local-First Hyperlocal Cascade Ranking...")
    same_loc_bonus = 30.0
    dist_decay = 0.25

    final_top50_rankings: list[list[str]] = []

    for qid, ranking in zip(query_id_list, raw_rrf, strict=True):
        q_loc = query_loc_map[qid]
        q_cat = query_cat_map[qid]
        q_lat = loc_lat_dict.get(q_loc)
        q_lon = loc_lon_dict.get(q_loc)

        boosted: list[tuple[str, float]] = []
        for item_id, score in ranking:
            it_loc = item_loc_map.get(item_id)
            it_cat = item_cat_map.get(item_id)
            it_lat = item_lat_map.get(item_id, 0.0)
            it_lon = item_lon_map.get(item_id, 0.0)

            # Wildcard category 0 fix
            if q_cat == 0:
                cat_mult = 1.0
            else:
                cat_mult = 1.0 if (it_cat == q_cat) else 0.001

            # Geo match multiplier
            if it_loc == q_loc:
                geo_mult = same_loc_bonus
            elif (
                q_lat is not None
                and q_lon is not None
                and it_lat != 0.0
                and it_lon != 0.0
            ):
                dist = haversine_km(q_lat, q_lon, it_lat, it_lon)
                geo_mult = float(np.exp(-dist_decay * (dist / 10.0)))
            else:
                geo_mult = 0.05

            boosted.append((item_id, score * cat_mult * geo_mult))

        boosted.sort(key=lambda p: (-p[1], p[0]))
        top50 = [it for it, _ in boosted[:50]]

        # Ensure strictly 50 unique items
        if len(top50) < 50:
            seen_items = set(top50)
            for it in item_id_list:
                if it not in seen_items:
                    top50.append(it)
                    seen_items.add(it)
                    if len(top50) == 50:
                        break
        final_top50_rankings.append(top50)

    # 5. Write final answer.csv atomically
    output_path = final_dir / "answer.csv"
    tmp_path = output_path.with_suffix(".csv.tmp")

    print(f"Writing {output_path}...")
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["query_id", "answer"])
        for qid, items in zip(query_id_list, final_top50_rankings, strict=True):
            writer.writerow([qid, " ".join(items)])

    tmp_path.replace(output_path)

    # Copy to root answer.csv
    root_answer = Path("answer.csv")
    shutil.copyfile(output_path, root_answer)
    print(f"Copied final submission to {root_answer}")

    print(f"Stage 04 complete in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
