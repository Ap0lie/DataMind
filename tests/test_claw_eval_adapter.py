from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

ROOT = Path(__file__).resolve().parents[1]


def _load(name: str, relative_path: str):
    path = ROOT / relative_path
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_adapter_extracts_scenario_and_removes_internal_marker() -> None:
    adapter = _load("datamind_claw_adapter", "deploy/claw-eval/adapter.py")

    scenario_id, question = adapter._extract_scenario(
        {
            "messages": [
                {"role": "system", "content": "system"},
                {
                    "role": "user",
                    "content": "[DATAMIND_SCENARIO=DM004]\n分析多表收入",
                },
            ]
        }
    )

    assert scenario_id == "DM004"
    assert question == "分析多表收入"
    assert "DATAMIND_SCENARIO" not in question


def test_adapter_redacts_keys_tokens_and_provider_secrets() -> None:
    adapter = _load("datamind_claw_adapter_redaction", "deploy/claw-eval/adapter.py")

    redacted = adapter._redact(
        {
            "api_key": "secret",
            "message": "token sk-abcdefghijklmnopqrstuvwxyz and abcdefghijklmnopqrst.BhUkXYtW8FdCeERd",
        }
    )

    assert redacted["api_key"] == "[REDACTED]"
    assert "sk-" not in redacted["message"]
    assert "BhUkXYtW8FdCeERd" not in redacted["message"]


def test_adapter_strips_group_only_relationship_fields_for_analysis_api() -> None:
    adapter = _load("datamind_claw_adapter_relationships", "deploy/claw-eval/adapter.py")

    normalized = adapter._analysis_relationships(
        [
            {
                "left_dataset_id": "left",
                "right_dataset_id": "right",
                "left_column": "order_id",
                "right_column": "order_id",
                "join_type": "left",
                "relationship_type": "many_to_one",
                "confidence": 1.0,
                "source": "rules",
            }
        ]
    )

    assert normalized == [
        {
            "left_dataset_id": "left",
            "right_dataset_id": "right",
            "left_column": "order_id",
            "right_column": "order_id",
            "join_type": "left",
        }
    ]


def test_adapter_sets_bounded_model_retry_defaults(monkeypatch) -> None:
    adapter = _load("datamind_claw_adapter_retries", "deploy/claw-eval/adapter.py")
    monkeypatch.delenv("DATAMIND_LLM_TRANSIENT_RETRIES", raising=False)
    monkeypatch.delenv("DATAMIND_LLM_RETRY_BACKOFF_SECONDS", raising=False)

    environment = adapter._datamind_environment()

    assert environment["DATAMIND_LLM_TRANSIENT_RETRIES"] == "4"
    assert environment["DATAMIND_LLM_RETRY_BACKOFF_SECONDS"] == "2"


def test_fixture_sampling_is_stable_and_independent_of_input_order() -> None:
    builder = _load("datamind_claw_fixture_builder", "deploy/claw-eval/build_fixtures.py")
    first = pd.Series(["o3", "o1", "o4", "o2", "o1"])
    second = pd.Series(list(reversed(first.tolist())))

    assert builder._stable_ids(first, 3) == builder._stable_ids(second, 3)
    assert len(builder._stable_ids(first, 3)) == 3


def test_summary_reports_datamind_release_gate(tmp_path: Path) -> None:
    summary = _load("datamind_claw_summary", "deploy/claw-eval/summarize_results.py")
    trace = tmp_path / "successful.jsonl"
    trace.write_text(
        "\n".join(
            [
                '{"type":"audit_snapshot","audit_data":{"runs":[{"result":{"loop_terminal_reason":"model_finished"}}]}}',
                '{"type":"grading_result","judge_calls":{"communication":{"score":1}}}',
            ]
        ),
        encoding="utf-8",
    )
    results = [
        {
            "task_id": task_id,
            "avg_score": 0.85,
            "trials": [
                {
                    "task_score": 0.85,
                    "passed": True,
                    "completion": 0.82,
                    "robustness": 1.0,
                    "communication": 0.8,
                    "safety": 1.0,
                    "trace": str(trace),
                }
                for _ in range(3)
            ],
        }
        for task_id in ("DM001", "DM002", "DM003", "DM004", "DM005", "DM006")
    ]

    rendered = summary.render(
        results, Path("batch_results.json"), judge_model="kimi-k3"
    )

    assert "套件平均分：**0.850**" in rendered
    assert "关键任务 pass^3：**通过**" in rendered
    assert "建议发布门槛：**通过**" in rendered
    assert "DataMind 在 Kimi K2.6 配置下" in rendered


def test_summary_rejects_provider_error_and_missing_judge(tmp_path: Path) -> None:
    summary = _load("datamind_claw_summary_diagnostics", "deploy/claw-eval/summarize_results.py")
    failed_trace = tmp_path / "provider-error.jsonl"
    failed_trace.write_text(
        '{"type":"audit_snapshot","audit_data":{"runs":[{"status":"failed","error":"RuntimeError: DataMind agent loop provider failed; refusing fallback"}]}}\n'
        '{"type":"grading_result","judge_calls":null}\n',
        encoding="utf-8",
    )
    successful_trace = tmp_path / "successful.jsonl"
    successful_trace.write_text(
        '{"type":"audit_snapshot","audit_data":{"runs":[{"result":{"loop_terminal_reason":"model_finished"}}]}}\n'
        '{"type":"grading_result","judge_calls":{"communication":{"score":1}}}\n',
        encoding="utf-8",
    )
    results = [
        {
            "task_id": task_id,
            "avg_score": 0.90,
            "trials": [
                {
                    "task_score": 0.90,
                    "passed": True,
                    "completion": 0.90,
                    "robustness": 1.0,
                    "communication": 0.8,
                    "safety": 1.0,
                    "trace": str(
                        failed_trace if task_id == "DM001" else successful_trace
                    ),
                }
            ],
        }
        for task_id in ("DM001", "DM002", "DM003", "DM004", "DM005", "DM006")
    ]

    rendered = summary.render(
        results, Path("batch_results.json"), judge_model="kimi-k3"
    )

    assert "评测有效性：**受污染**" in rendered
    assert "Agent provider_error：**1/6 trials**" in rendered
    assert "kimi-k3 裁判结果缺失：**1** 次（DM001#1）" in rendered
    assert "建议发布门槛：**未通过**" in rendered


@pytest.mark.integration
def test_adapter_lifespan_starts_isolated_datamind_and_exposes_control_endpoints() -> None:
    adapter = _load("datamind_claw_adapter_integration", "deploy/claw-eval/adapter.py")

    with TestClient(adapter.app) as client:
        health = client.get("/health")
        assert health.status_code == 200
        assert health.json() == {"status": "ok", "datamind": True, "fixtures": True}

        reset = client.post("/reset")
        assert reset.status_code == 200
        assert reset.json()["namespace"].startswith("claw-")

        audit = client.get("/audit")
        assert audit.status_code == 200
        assert audit.json() == {"calls": [], "runs": []}

        invalid = client.post(
            "/v1/chat/completions",
            json={"model": "datamind-core", "messages": [{"role": "user", "content": "hi"}]},
        )
        assert invalid.status_code == 400
