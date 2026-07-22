from __future__ import annotations

import json
import math
import sqlite3
from collections import Counter
from pathlib import Path
from statistics import median
from typing import Any


def aggregate_sqlite_history(path: Path) -> dict[str, Any]:
    uri = f"file:{path.resolve().as_posix()}?mode=ro"
    connection = sqlite3.connect(uri, uri=True)
    connection.row_factory = sqlite3.Row
    try:
        tables = {
            row[0]
            for row in connection.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        return {
            "database": path.name,
            "analysis_jobs": _job_summary(connection, "analysis_jobs", tables),
            "cleaning_jobs": _job_summary(connection, "cleaning_jobs", tables),
            "assistant_runs": _job_summary(connection, "assistant_runs", tables),
            "analysis_events": _event_summary(connection, tables),
            "privacy": "Aggregate status, timing, repair and fallback fields only; no prompts, rows, or report content.",
        }
    finally:
        connection.close()


def _job_summary(
    connection: sqlite3.Connection, table: str, tables: set[str]
) -> dict[str, Any]:
    if table not in tables:
        return {"available": False}
    columns = {row[1] for row in connection.execute(f"PRAGMA table_info({table})")}
    start = "started_at" if "started_at" in columns else "created_at"
    rows = connection.execute(
        f"SELECT status, {start} AS started, completed_at FROM {table}"
    ).fetchall()
    durations = [
        float(row[0])
        for row in connection.execute(
            f"SELECT (julianday(completed_at)-julianday({start}))*86400 "
            f"FROM {table} WHERE completed_at IS NOT NULL"
        ).fetchall()
        if row[0] is not None and float(row[0]) >= 0
    ]
    statuses = Counter(str(row["status"]) for row in rows)
    return {
        "available": True,
        "count": len(rows),
        "statuses": dict(sorted(statuses.items())),
        "duration_seconds": {
            "samples": len(durations),
            "median": round(median(durations), 3) if durations else None,
            "p95": round(_percentile(durations, 0.95), 3) if durations else None,
            "max": round(max(durations), 3) if durations else None,
        },
    }


def _event_summary(connection: sqlite3.Connection, tables: set[str]) -> dict[str, Any]:
    if "analysis_job_events" not in tables:
        return {"available": False}
    rows = connection.execute(
        "SELECT duration_ms, token_usage, status, event_type FROM analysis_job_events"
    ).fetchall()
    durations = [float(row["duration_ms"]) for row in rows if row["duration_ms"] is not None]
    token_total = 0
    for row in rows:
        try:
            usage = json.loads(str(row["token_usage"] or "{}"))
            token_total += int((usage or {}).get("total_tokens") or 0)
        except (TypeError, ValueError, json.JSONDecodeError):
            continue
    token_available = token_total > 0
    return {
        "available": True,
        "count": len(rows),
        "failed": sum(str(row["status"]) == "failed" for row in rows),
        "duration_ms": {
            "samples": len(durations),
            "median": round(median(durations), 3) if durations else None,
            "p95": round(_percentile(durations, 0.95), 3) if durations else None,
        },
        "token_usage_status": "available" if token_available else "metric_unavailable",
        "total_tokens": token_total if token_available else None,
    }


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))]
