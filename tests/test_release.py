import pandas as pd
import pytest
from avito_candidate_generation.experiments import ExperimentRegistry, ExperimentResult
from avito_candidate_generation.inference import rank_predictions
from avito_candidate_generation.pipeline import FinalPipeline
from avito_candidate_generation.submission import validate_submission, write_submission


def test_registry_rejects_incompatible_and_tie_breaks():
    registry = ExperimentRegistry()
    registry.add(ExperimentResult("b", "c", "d", "p", {"recall@50": 0.5}))
    registry.add(ExperimentResult("a", "c", "d", "p", {"recall@50": 0.5}))
    assert registry.best().run_id == "a"
    with pytest.raises(ValueError):
        registry.add(ExperimentResult("x", "c", "other", "p", {"recall@50": 0.9}))


def test_inference_tie_break():
    frame = pd.DataFrame(
        {"query_id": ["q", "q"], "item_id": ["b", "a"], "final_score": [1.0, 1.0]}
    )
    assert rank_predictions(frame).iloc[0].item_id == "a"


def test_bundle_load(tmp_path):
    pipeline = FinalPipeline(tmp_path, {})
    path = pipeline.save_manifest([])
    assert FinalPipeline.load(tmp_path).manifest["status"] == "PASS"
    assert path.exists()


def test_bm25_no_match_uses_stable_zero_score_fallback() -> None:
    from avito_candidate_generation.retrievers.bm25 import BM25Index

    items = pd.DataFrame({"item_id": ["b", "a"], "text": ["red", "blue"]})
    queries = pd.DataFrame({"internal_query_id": ["q"], "text": ["missing"]})
    result = BM25Index.fit(items).retrieve(queries, k=2)
    assert result["item_id"].tolist() == ["a", "b"]
    assert result["score"].tolist() == [0.0, 0.0]


def test_streaming_recall_matches_query_mean() -> None:
    from avito_candidate_generation.validation import stream_recall_at_k

    queries = pd.DataFrame({"internal_query_id": ["q1", "q2"]})
    truth = pd.DataFrame({"internal_query_id": ["q1", "q2"], "item_id": ["a", "b"]})

    def retrieve(batch: pd.DataFrame, k: int) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "internal_query_id": batch["internal_query_id"].tolist(),
                "item_id": [
                    "a" if q == "q1" else "x" for q in batch["internal_query_id"]
                ],
                "rank": [1] * len(batch),
            }
        )

    assert stream_recall_at_k(queries, truth, retrieve, ks=(1,), batch_size=1) == {
        "recall@1": 0.5
    }


def test_model_preflight_rejects_missing_local_artifact(tmp_path) -> None:
    from avito_candidate_generation.provisioning import _has_required_model_files

    assert not _has_required_model_files(tmp_path)


def test_submission_roundtrip(tmp_path):
    queries = tmp_path / "queries.parquet"
    items = tmp_path / "items.parquet"
    predictions = tmp_path / "predictions.parquet"
    output = tmp_path / "answer.csv"
    pd.DataFrame({"query_id": ["0" * 16, "1" * 16]}).to_parquet(queries)
    pd.DataFrame({"item_id": ["A" * 16]}).to_parquet(items)
    pd.DataFrame(
        {"query_id": ["0" * 16], "item_id": ["A" * 16], "rank": [1]}
    ).to_parquet(predictions)
    write_submission(predictions, queries, items, output)
    assert validate_submission(output, queries, items).passed
    assert output.read_text().splitlines()[0] == "query_id,answer"
