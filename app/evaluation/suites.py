from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from app.evaluation import corpus
from app.evaluation.models import BenchmarkSuiteManifest

BENCHMARK_ROOT = Path(__file__).resolve().parents[2] / "benchmarks"

_CORPUS_FACTORIES = {
    "dirty_customer_records": corpus.dirty_customer_records,
    "relationship_tables": corpus.relationship_tables,
    "semantic_questions": corpus.semantic_questions,
    "loop_outcomes": corpus.loop_outcomes,
}


def available_suites() -> tuple[str, ...]:
    return tuple(sorted(path.stem for path in (BENCHMARK_ROOT / "suites").glob("*.json")))


def load_suite_manifest(name: str) -> BenchmarkSuiteManifest:
    if re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) is None:
        raise ValueError("Benchmark suite name is invalid.")
    path = BENCHMARK_ROOT / "suites" / f"{name}.json"
    if not path.is_file():
        raise FileNotFoundError(f"Unknown benchmark suite: {name}")
    payload = _hydrate(json.loads(path.read_text(encoding="utf-8")))
    checksum = corpus.corpus_checksum()
    declared = str(payload.get("corpus_checksum") or "auto")
    if declared not in {"auto", checksum}:
        raise RuntimeError("Benchmark corpus checksum does not match the suite manifest.")
    payload["corpus_checksum"] = checksum
    return BenchmarkSuiteManifest.model_validate(payload)


def _hydrate(value: Any) -> Any:
    if isinstance(value, dict):
        if set(value) == {"$corpus"}:
            name = str(value["$corpus"])
            factory = _CORPUS_FACTORIES.get(name)
            if factory is None:
                raise ValueError(f"Unknown benchmark corpus reference: {name}")
            return factory()
        return {key: _hydrate(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_hydrate(item) for item in value]
    return value
