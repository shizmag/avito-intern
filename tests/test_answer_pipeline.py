"""Unit tests for answer_pipeline module and independent validation contract."""

from pathlib import Path

import numpy as np
import pytest

from avito_candidate_generation.answer_pipeline import (
    compare_with_baseline,
    compute_sha256,
    haversine_km,
    validate_submission_contract,
)


def test_haversine_km_accuracy():
    # Moscow to Saint Petersburg is ~634 km
    moscow_lat, moscow_lon = 55.7558, 37.6173
    spb_lat, spb_lon = 59.9343, 30.3351
    d = haversine_km(moscow_lat, moscow_lon, spb_lat, spb_lon)
    assert 620.0 < d < 650.0

    # Same location distance is 0.0
    d_same = haversine_km(moscow_lat, moscow_lon, moscow_lat, moscow_lon)
    assert np.isclose(d_same, 0.0)


def test_compute_sha256(tmp_path: Path):
    test_file = tmp_path / "sample.txt"
    test_file.write_text("test_avito_answer")
    checksum = compute_sha256(test_file)
    assert len(checksum) == 64
    assert isinstance(checksum, str)


def test_validate_submission_contract_on_real_submission():
    sub_path = Path("answer.csv")
    if sub_path.is_file():
        report = validate_submission_contract(
            submission_path=sub_path,
            expected_queries_path="data/benchmark_queries.parquet",
            expected_items_path="data/benchmark_items.parquet",
        )
        assert report["status"] == "PASS"
        assert report["rows_count"] == 2452
        assert report["all_answers_exactly_50_items"] is True
        assert report["no_duplicate_items"] is True
        assert report["no_unknown_items"] is True


def test_validate_submission_contract_invalid_header(tmp_path: Path):
    bad_file = tmp_path / "bad.csv"
    bad_file.write_text("invalid_header,col2\n1,2 3\n")
    with pytest.raises(ValueError, match="invalid header line"):
        validate_submission_contract(bad_file)


def test_compare_with_baseline_real():
    sub_path = Path("answer.csv")
    base_path = Path("artifacts/submissions/official_0646469/answer.csv")
    if sub_path.is_file() and base_path.is_file():
        comp = compare_with_baseline(sub_path, base_path)
        assert comp["queries_compared"] == 2452
        assert "mean_overlap_top50" in comp
        assert 0.0 <= comp["fraction_top1_changed"] <= 1.0
