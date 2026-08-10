from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from itertools import islice
from math import isnan
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from uuid import UUID

import pandas as pd

from app.core.settings import get_settings
from app.schemas.analysis import ChartResponse, PythonAnalysisResponse

SUPPORTED_CHART_TYPES = {
    "bar",
    "line",
    "pie",
    "histogram",
    "box_plot",
    "correlation_heatmap",
}
MAX_CHART_ROWS = 500
MAX_HISTOGRAM_BINS = 30
MAX_CHART_COLUMNS = 8
MAX_STAT_DICT_KEYS = 80
MAX_STAT_LIST_ITEMS = 80
MAX_STRING_CHARS = 500


class GeneratedPythonSafetyError(ValueError):
    """Raised when generated Python code violates the Python Agent output contract."""


SAFE_IMPORT_ROOTS = {
    "collections",
    "datetime",
    "itertools",
    "json",
    "math",
    "numpy",
    "pandas",
    "re",
    "statistics",
}
DANGEROUS_IMPORT_ROOTS = {
    "builtins",
    "ctypes",
    "http",
    "importlib",
    "os",
    "pathlib",
    "requests",
    "shutil",
    "socket",
    "subprocess",
    "sys",
    "urllib",
}
DANGEROUS_CALLS = {
    "__import__",
    "compile",
    "eval",
    "exec",
    "globals",
    "input",
    "locals",
    "open",
}
EXECUTION_TIMEOUT_SECONDS = 8
OUTPUT_LIMIT_BYTES = 200_000


@dataclass(frozen=True)
class PythonExecutionPolicy:
    timeout_seconds: float = EXECUTION_TIMEOUT_SECONDS
    output_limit_bytes: int = OUTPUT_LIMIT_BYTES


def run_generated_python_analysis(
    code: str,
    dataframe: pd.DataFrame,
    *,
    policy: PythonExecutionPolicy | None = None,
) -> PythonAnalysisResponse:
    """Execute generated analysis code in an isolated subprocess."""

    execution_policy = policy or PythonExecutionPolicy()
    tree = ast.parse(code, mode="exec")
    _validate_generated_code(tree)
    payload = {
        "code": code,
        "records": _jsonable(dataframe.to_dict(orient="records")),
    }
    settings = get_settings()
    if settings.python_runner_url:
        return _run_remote_python_analysis(payload, policy=execution_policy)
    if settings.environment.lower() == "production":
        raise GeneratedPythonSafetyError(
            "Production Python execution requires the container Runner."
        )
    env = {
        "PYTHONIOENCODING": "utf-8",
        "PYTHONPATH": os.environ.get("PYTHONPATH", ""),
    }
    try:
        with tempfile.TemporaryDirectory(prefix="datamind-python-agent-") as sandbox_dir:
            completed = subprocess.run(
                [sys.executable, "-I", "-c", _WORKER_SOURCE],
                input=json.dumps(payload, ensure_ascii=False, default=str),
                capture_output=True,
                text=True,
                timeout=execution_policy.timeout_seconds,
                env=env,
                cwd=sandbox_dir,
                check=False,
            )
    except subprocess.TimeoutExpired as exc:
        raise GeneratedPythonSafetyError("Generated Python timed out.") from exc

    stdout = completed.stdout or ""
    stderr = completed.stderr or ""
    if (
        len(stdout.encode("utf-8")) > execution_policy.output_limit_bytes
        or len(stderr.encode("utf-8")) > execution_policy.output_limit_bytes
    ):
        raise GeneratedPythonSafetyError("Generated Python output exceeded the size limit.")
    if completed.returncode != 0:
        message = (stderr or stdout or "Generated Python subprocess failed.").strip()
        raise GeneratedPythonSafetyError(message[:1000])
    try:
        result = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise GeneratedPythonSafetyError("Generated Python did not return valid JSON.") from exc
    if not isinstance(result, dict):
        raise GeneratedPythonSafetyError("analyze(df) must return a dict.")
    return _python_result_from_payload(result)


def _run_remote_python_analysis(
    payload: dict[str, Any],
    *,
    policy: PythonExecutionPolicy,
) -> PythonAnalysisResponse:
    settings = get_settings()
    url = f"{str(settings.python_runner_url).rstrip('/')}/execute"
    headers = {"Content-Type": "application/json"}
    if settings.python_runner_shared_secret:
        headers["X-Runner-Token"] = settings.python_runner_shared_secret.get_secret_value()
    request = Request(
        url,
        data=json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8"),
        headers=headers,
        method="POST",
    )
    try:
        with urlopen(request, timeout=settings.python_runner_timeout_seconds) as response:
            raw = response.read(policy.output_limit_bytes + 1)
    except HTTPError as exc:
        message = exc.read(1000).decode("utf-8", errors="replace")
        raise GeneratedPythonSafetyError(f"Python Runner error {exc.code}: {message}") from exc
    except (URLError, TimeoutError) as exc:
        raise GeneratedPythonSafetyError(f"Python Runner unavailable: {exc}") from exc
    if len(raw) > policy.output_limit_bytes:
        raise GeneratedPythonSafetyError("Generated Python output exceeded the size limit.")
    try:
        envelope = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise GeneratedPythonSafetyError("Python Runner returned invalid JSON.") from exc
    result = envelope.get("result") if isinstance(envelope, dict) else None
    if not isinstance(result, dict):
        detail = envelope.get("detail") if isinstance(envelope, dict) else None
        raise GeneratedPythonSafetyError(str(detail or "Python Runner returned no result."))
    return _python_result_from_payload(result)


def _validate_generated_code(tree: ast.AST) -> None:
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                _validate_import(alias.name)
        elif isinstance(node, ast.ImportFrom):
            _validate_import(node.module or "")
        elif isinstance(node, ast.Call):
            name = _call_name(node.func)
            if name in DANGEROUS_CALLS:
                raise GeneratedPythonSafetyError(f"Generated code cannot call {name}().")
            if name in {"to_csv", "to_excel", "to_json", "to_parquet", "to_pickle"}:
                raise GeneratedPythonSafetyError(f"Generated code cannot write files with {name}().")


def _validate_import(module_name: str) -> None:
    root = module_name.split(".", 1)[0]
    if root in DANGEROUS_IMPORT_ROOTS or root not in SAFE_IMPORT_ROOTS:
        raise GeneratedPythonSafetyError(f"Generated code cannot import {module_name}.")


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ""


_WORKER_SOURCE = textwrap.dedent(
    """
    import builtins
    import json
    from itertools import islice

    import pandas as pd

    payload = json.loads(input())
    namespace = {
        "__builtins__": {
            "abs": builtins.abs,
            "all": builtins.all,
            "any": builtins.any,
            "bool": builtins.bool,
            "dict": builtins.dict,
            "enumerate": builtins.enumerate,
            "float": builtins.float,
            "int": builtins.int,
            "len": builtins.len,
            "list": builtins.list,
            "max": builtins.max,
            "min": builtins.min,
            "pow": builtins.pow,
            "range": builtins.range,
            "round": builtins.round,
            "set": builtins.set,
            "sorted": builtins.sorted,
            "str": builtins.str,
            "sum": builtins.sum,
            "tuple": builtins.tuple,
            "zip": builtins.zip,
            "__import__": builtins.__import__,
        },
        "pd": pd,
        "pandas": pd,
        "to_numeric": pd.to_numeric,
    }
    def _compact_result(value):
        if not isinstance(value, dict):
            return value
        return {
            "statistics": _compact_json(value.get("statistics"), 2),
            "insights": _compact_list(value.get("insights"), 8, 1),
            "charts": _compact_charts(value.get("charts")),
        }

    def _compact_charts(value):
        if not isinstance(value, (list, tuple)):
            return []
        charts = []
        for item in islice(value, 6):
            if not isinstance(item, dict):
                continue
            chart_type = str(item.get("chart_type") or "")
            data = _compact_chart_data(chart_type, item.get("data"), item.get("spec"))
            if not data:
                continue
            charts.append({
                "title": _truncate(item.get("title") or "Python Agent 图表"),
                "chart_type": chart_type,
                "spec": _compact_json(item.get("spec"), 2) if isinstance(item.get("spec"), dict) else {},
                "data": data,
            })
        return charts

    def _compact_chart_data(chart_type, value, spec):
        if isinstance(value, pd.DataFrame):
            value = value.to_dict(orient="records")
        if isinstance(value, pd.Series):
            value = value.to_frame().to_dict(orient="records")
        if not isinstance(value, (list, tuple)):
            return []
        rows = [row for row in value if isinstance(row, dict)]
        if chart_type == "histogram" and len(rows) > 30:
            return _histogram_bins(rows, spec)
        if chart_type == "box_plot" and len(rows) > 120:
            summary = _box_summary(rows, spec)
            if summary:
                return summary
        return [_compact_row(row) for row in rows[:500]]

    def _histogram_bins(rows, spec):
        field = None
        if isinstance(spec, dict):
            field = spec.get("x") or spec.get("value") or spec.get("y")
        if not field and rows:
            for key, value in rows[0].items():
                if _to_float(value) is not None:
                    field = key
                    break
        values = [_to_float(row.get(field)) for row in rows] if field else []
        values = [value for value in values if value is not None]
        if not values:
            return [_compact_row(row) for row in rows[:500]]
        low, high = min(values), max(values)
        if low == high:
            return [{"bin_start": low, "bin_end": high, "count": len(values)}]
        bin_count = min(30, max(1, len(values)))
        width = (high - low) / bin_count
        counts = [0] * bin_count
        for value in values:
            index = min(int((value - low) / width), bin_count - 1)
            counts[index] += 1
        return [
            {"bin_start": low + index * width, "bin_end": low + (index + 1) * width, "count": count}
            for index, count in enumerate(counts)
        ]

    def _box_summary(rows, spec):
        x_field = spec.get("x") if isinstance(spec, dict) else None
        y_field = spec.get("y") if isinstance(spec, dict) else None
        if not y_field and rows:
            for key, value in rows[0].items():
                if _to_float(value) is not None:
                    y_field = key
                    break
        groups = {}
        for row in rows:
            value = _to_float(row.get(y_field)) if y_field else None
            if value is None:
                continue
            group = str(row.get(x_field)) if x_field else "all"
            groups.setdefault(group, []).append(value)
        summary = []
        for group, values in islice(groups.items(), 30):
            values.sort()
            if values:
                item = {
                    "group": group,
                    "min": values[0],
                    "q1": _quantile(values, 0.25),
                    "median": _quantile(values, 0.5),
                    "q3": _quantile(values, 0.75),
                    "max": values[-1],
                    "count": len(values),
                }
                summary.append(item)
        return summary

    def _quantile(values, q):
        if not values:
            return None
        position = (len(values) - 1) * q
        lower = int(position)
        upper = min(lower + 1, len(values) - 1)
        weight = position - lower
        return values[lower] * (1 - weight) + values[upper] * weight

    def _to_float(value):
        try:
            number = float(value)
        except (TypeError, ValueError):
            return None
        if number != number:
            return None
        return number

    def _compact_row(row):
        compact = {}
        for key, value in islice(row.items(), 8):
            compact[str(key)] = _compact_json(value, 1)
        return compact

    def _compact_json(value, depth):
        if depth <= 0:
            if isinstance(value, (dict, list, tuple)):
                return str(type(value).__name__)
            return _truncate(value)
        if isinstance(value, dict):
            return {
                str(key): _compact_json(item, depth - 1)
                for key, item in islice(value.items(), 80)
            }
        if isinstance(value, (list, tuple)):
            return [_compact_json(item, depth - 1) for item in islice(value, 80)]
        return _truncate(value)

    def _compact_list(value, max_items, depth):
        if not isinstance(value, (list, tuple)):
            return []
        return [_compact_json(item, depth) for item in islice(value, max_items)]

    def _truncate(value):
        text = str(value) if isinstance(value, str) else value
        if isinstance(text, str) and len(text) > 500:
            return text[:500] + "..."
        return text

    exec(compile(payload["code"], "<datamind-python-agent>", "exec"), namespace, namespace)
    analyze = namespace.get("analyze")
    if not callable(analyze):
        raise ValueError("Generated code must define analyze(df).")
    df = pd.DataFrame.from_records(payload.get("records") or [])
    result = _compact_result(analyze(df.copy(deep=True)))
    print(json.dumps(result, ensure_ascii=False, default=str))
    """
)


def _python_result_from_payload(payload: dict[str, Any]) -> PythonAnalysisResponse:
    statistics = payload.get("statistics")
    insights = payload.get("insights")
    charts_payload = payload.get("charts")
    return PythonAnalysisResponse(
        statistics=_compact_json(statistics, depth=2) if isinstance(statistics, dict) else {},
        insights=tuple(_clean_insights(insights)),
        charts=tuple(_clean_charts(charts_payload)),
    )


def _clean_insights(value: Any) -> list[str]:
    if not isinstance(value, list | tuple):
        return []
    insights: list[str] = []
    for item in value[:8]:
        text = str(item).strip()
        if text:
            insights.append(text)
    return insights


def _clean_charts(value: Any) -> list[ChartResponse]:
    if not isinstance(value, list | tuple):
        return []
    charts: list[ChartResponse] = []
    for item in value[:6]:
        if not isinstance(item, dict):
            continue
        chart_type = str(item.get("chart_type") or "").strip()
        if chart_type not in SUPPORTED_CHART_TYPES:
            continue
        spec = item.get("spec")
        data = _chart_data(item.get("data"), chart_type=chart_type, spec=spec if isinstance(spec, dict) else {})
        if not data:
            continue
        charts.append(
            ChartResponse(
                title=str(item.get("title") or "Python Agent 图表"),
                chart_type=chart_type,
                spec=_compact_json(spec, depth=2) if isinstance(spec, dict) else {},
                data=tuple(data),
            )
        )
    return charts


def _chart_data(value: Any, *, chart_type: str = "", spec: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    if isinstance(value, pd.DataFrame):
        value = value.to_dict(orient="records")
    if isinstance(value, pd.Series):
        value = value.to_frame().to_dict(orient="records")
    if not isinstance(value, list | tuple):
        return []
    rows = [row for row in value if isinstance(row, dict)]
    if chart_type == "histogram" and len(rows) > MAX_HISTOGRAM_BINS:
        return _histogram_bins(rows, spec or {})
    if chart_type == "box_plot" and len(rows) > 120:
        summary = _box_summary(rows, spec or {})
        if summary:
            return summary
    return [_compact_row(row) for row in rows[:MAX_CHART_ROWS]]


def _histogram_bins(rows: list[dict[str, Any]], spec: dict[str, Any]) -> list[dict[str, Any]]:
    field = str(spec.get("x") or spec.get("value") or spec.get("y") or "")
    if not field and rows:
        for key, value in rows[0].items():
            if _to_float(value) is not None:
                field = str(key)
                break
    values = [_to_float(row.get(field)) for row in rows] if field else []
    numeric_values = [value for value in values if value is not None]
    if not numeric_values:
        return [_compact_row(row) for row in rows[:MAX_CHART_ROWS]]
    low = min(numeric_values)
    high = max(numeric_values)
    if low == high:
        return [{"bin_start": low, "bin_end": high, "count": len(numeric_values)}]
    bin_count = min(MAX_HISTOGRAM_BINS, max(1, len(numeric_values)))
    width = (high - low) / bin_count
    counts = [0] * bin_count
    for value in numeric_values:
        index = min(int((value - low) / width), bin_count - 1)
        counts[index] += 1
    return [
        {
            "bin_start": low + index * width,
            "bin_end": low + (index + 1) * width,
            "count": count,
        }
        for index, count in enumerate(counts)
    ]


def _box_summary(rows: list[dict[str, Any]], spec: dict[str, Any]) -> list[dict[str, Any]]:
    x_field = str(spec.get("x") or "")
    y_field = str(spec.get("y") or "")
    if not y_field and rows:
        for key, value in rows[0].items():
            if _to_float(value) is not None:
                y_field = str(key)
                break
    groups: dict[str, list[float]] = {}
    for row in rows:
        value = _to_float(row.get(y_field)) if y_field else None
        if value is None:
            continue
        group = str(row.get(x_field)) if x_field else "all"
        groups.setdefault(group, []).append(value)
    summary: list[dict[str, Any]] = []
    for group, values in islice(groups.items(), MAX_HISTOGRAM_BINS):
        values.sort()
        if values:
            summary.append(
                {
                    "group": group,
                    "min": values[0],
                    "q1": _quantile(values, 0.25),
                    "median": _quantile(values, 0.5),
                    "q3": _quantile(values, 0.75),
                    "max": values[-1],
                    "count": len(values),
                }
            )
    return summary


def _quantile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    position = (len(values) - 1) * q
    lower = int(position)
    upper = min(lower + 1, len(values) - 1)
    weight = position - lower
    return values[lower] * (1 - weight) + values[upper] * weight


def _to_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if isnan(number):
        return None
    return number


def _compact_row(row: dict[str, Any]) -> dict[str, Any]:
    return {
        str(key): _compact_json(value, depth=1)
        for key, value in islice(row.items(), MAX_CHART_COLUMNS)
    }


def _compact_json(value: Any, *, depth: int) -> Any:
    if depth <= 0:
        if isinstance(value, dict):
            return "dict"
        if isinstance(value, list | tuple):
            return "list"
        return _json_scalar(value)
    if isinstance(value, dict):
        return {
            str(key): _compact_json(item, depth=depth - 1)
            for key, item in islice(value.items(), MAX_STAT_DICT_KEYS)
        }
    if isinstance(value, list | tuple):
        return [
            _compact_json(item, depth=depth - 1)
            for item in islice(value, MAX_STAT_LIST_ITEMS)
        ]
    return _json_scalar(value)


def _json_scalar(value: Any) -> Any:
    value = _jsonable(value)
    if isinstance(value, str) and len(value) > MAX_STRING_CHARS:
        return f"{value[:MAX_STRING_CHARS]}..."
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(item) for item in value]
    if isinstance(value, UUID):
        return str(value)
    if hasattr(value, "item"):
        return _jsonable(value.item())
    if isinstance(value, float) and isnan(value):
        return None
    return value
