"""Comparable experiment registry and deterministic architecture selection."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path


@dataclass(frozen=True)
class ExperimentResult:
    run_id: str
    config_hash: str
    data_fingerprint: str
    corpus_fingerprint: str
    metrics: dict[str, float]
    status: str = "PASS"
    comparable_group: str = "default"


class ExperimentRegistry:
    def __init__(self) -> None:
        self.results: list[ExperimentResult] = []

    def add(self, result: ExperimentResult) -> None:
        if self.results:
            first = self.results[0]
            current = (
                result.data_fingerprint,
                result.corpus_fingerprint,
                result.comparable_group,
            )
            baseline = (
                first.data_fingerprint,
                first.corpus_fingerprint,
                first.comparable_group,
            )
            if current != baseline:
                raise ValueError("incompatible experiment fingerprints")
        self.results.append(result)

    def best(self, metric: str = "recall@50") -> ExperimentResult:
        if not self.results:
            raise ValueError("registry is empty")
        return sorted(
            self.results,
            key=lambda r: (-r.metrics.get(metric, float("-inf")), r.run_id),
        )[0]

    def write(self, path: str | Path) -> Path:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = (
            json.dumps([asdict(result) for result in self.results], indent=2) + "\n"
        )
        target.write_text(payload, encoding="utf-8")
        return target
