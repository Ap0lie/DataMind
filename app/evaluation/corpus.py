from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path
from typing import Any

CORPUS_ROOT = Path(__file__).resolve().parents[2] / "benchmarks" / "corpus"


def corpus_checksum() -> str:
    manifest = json.loads((CORPUS_ROOT / "manifest.json").read_text(encoding="utf-8"))
    payload = {
        "manifest": manifest,
        "dirty_customer_records": dirty_customer_records(),
        "relationship_tables": relationship_tables(),
        "semantic_questions": semantic_questions(),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def dirty_customer_records() -> list[dict[str, Any]]:
    return [
        {" customer_id ": " C001 ", "amount": "10", "订单日期": "2026-01-01"},
        {" customer_id ": " C002 ", "amount": "20", "订单日期": "2026-01-02"},
        {" customer_id ": " C002 ", "amount": "20", "订单日期": "2026-01-02"},
        {" customer_id ": "   ", "amount": None, "订单日期": "not-a-date"},
        {" customer_id ": None, "amount": None, "订单日期": None},
    ]


def relationship_tables() -> dict[str, list[dict[str, Any]]]:
    return {
        "orders": [
            {"order_id": "O1", "customer_id": "C1", "amount": 10},
            {"order_id": "O2", "customer_id": "C2", "amount": 20},
            {"order_id": "O3", "customer_id": "C1", "amount": 30},
        ],
        "customers": [
            {"customer_id": "C1", "segment": "A"},
            {"customer_id": "C2", "segment": "B"},
        ],
        "tags": [
            {"order_id": "O1", "tag": "priority"},
            {"order_id": "O1", "tag": "gift"},
            {"order_id": "O2", "tag": "priority"},
        ],
    }


def semantic_questions() -> list[dict[str, str]]:
    metric_terms = ("销售额", "营收", "收入", "GMV")
    dimension_terms = ("地区", "区域", "大区", "地域")
    suffixes = ("", "趋势", "合计", "表现", "同比")
    questions = [
        {"question": f"按地区查看{term}{suffix}", "expected": "sales", "type": "metric"}
        for term in metric_terms
        for suffix in suffixes
    ]
    questions.extend(
        {"question": f"按{term}分析销售额{suffix}", "expected": "region", "type": "dimension"}
        for term in dimension_terms
        for suffix in suffixes
    )
    return (questions * 3)[:100]


def loop_outcomes() -> list[dict[str, Any]]:
    return [
        {
            "case_id": f"loop-{index}",
            "selected_tool": "execute_safe_sql",
            "expected_tools": ["execute_safe_sql", "execute_semantic_query"],
            "legal_call": index < 96,
            "recoverable_error": index < 20,
            "recovered": index < 18,
            "tool_calls": 2 if index < 20 else 1,
            "duplicate_successful_actions": 0,
        }
        for index in range(100)
    ]


def generated_rows(size: int, *, seed: int = 20260716) -> list[dict[str, Any]]:
    randomizer = random.Random(seed)
    regions = ("华东", "华南", "华北", "西部")
    return [
        {
            "row_id": index,
            "region": regions[index % len(regions)],
            "amount": round(randomizer.uniform(1.0, 1000.0), 2),
            "comment": f" order {index} ",
        }
        for index in range(size)
    ]


def external_corpus_status(root: Path | None) -> dict[str, Any]:
    if root is None:
        return {"available": False, "reason": "BENCHMARK_DATA_ROOT is not configured."}
    manifest_path = root / "benchmark-manifest.json"
    if not manifest_path.exists():
        return {"available": False, "reason": "External benchmark manifest is missing."}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    failures = []
    for item in manifest.get("files") or []:
        path = root / str(item["path"])
        if not path.is_file():
            failures.append(f"missing:{item['path']}")
            continue
        with path.open("rb") as handle:
            digest = hashlib.file_digest(handle, "sha256").hexdigest()
        if digest != item.get("sha256"):
            failures.append(f"checksum:{item['path']}")
    return {
        "available": not failures,
        "version": manifest.get("version"),
        "failures": failures,
    }
