from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
import tempfile
import textwrap
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

from app.core.settings import get_settings


class GeneratedCleaningSandboxError(ValueError):
    """Raised when generated cleaning code is unsafe or fails in isolation."""


@dataclass(frozen=True)
class CleaningExecutionPolicy:
    timeout_seconds: float = 12.0
    output_limit_bytes: int = 4_000_000


_FORBIDDEN_NODES = (
    ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.ClassDef,
    ast.Delete, ast.Try, ast.With, ast.AsyncFunctionDef, ast.Await, ast.While,
)
_FORBIDDEN_CALLS = {
    "__import__", "compile", "eval", "exec", "getattr", "globals", "input",
    "locals", "open", "setattr", "vars", "connect", "glob", "listdir",
    "mkdir", "popen", "query", "read", "read_csv", "read_excel", "read_parquet",
    "remove", "request", "rmdir", "system", "to_csv", "to_excel", "to_parquet",
    "to_pickle", "to_sql", "unlink", "urlopen", "walk", "write",
}


def validate_generated_cleaning_code(code: str) -> None:
    tree = ast.parse(code, mode="exec")
    definitions = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "clean_dataset"
    ]
    if len(definitions) != 1:
        raise GeneratedCleaningSandboxError(
            "Generated code must define exactly one clean_dataset(df)."
        )
    for node in ast.walk(tree):
        if isinstance(node, _FORBIDDEN_NODES):
            raise GeneratedCleaningSandboxError(f"Forbidden Python construct: {type(node).__name__}.")
        if isinstance(node, (ast.Name, ast.Attribute)):
            name = node.id if isinstance(node, ast.Name) else node.attr
            if name.startswith("_") or name in _FORBIDDEN_CALLS:
                raise GeneratedCleaningSandboxError(f"Forbidden Python access: {name}.")
        if isinstance(node, ast.Call):
            name = node.func.id if isinstance(node.func, ast.Name) else (
                node.func.attr if isinstance(node.func, ast.Attribute) else ""
            )
            if name in _FORBIDDEN_CALLS:
                raise GeneratedCleaningSandboxError(f"Forbidden function call: {name}.")


def run_generated_cleaning_analysis(
    code: str,
    dataframe: pd.DataFrame,
    *,
    policy: CleaningExecutionPolicy | None = None,
) -> pd.DataFrame:
    execution_policy = policy or CleaningExecutionPolicy()
    validate_generated_cleaning_code(code)
    payload = {
        "execution_kind": "cleaning",
        "code": code,
        "records": _jsonable(dataframe.to_dict(orient="records")),
    }
    settings = get_settings()
    if settings.python_runner_url:
        result = _run_remote(payload, execution_policy)
    else:
        env = {"PYTHONIOENCODING": "utf-8", "PYTHONPATH": os.environ.get("PYTHONPATH", "")}
        try:
            with tempfile.TemporaryDirectory(prefix="datamind-cleaning-agent-") as folder:
                completed = subprocess.run(
                    [sys.executable, "-I", "-c", CLEANING_WORKER_SOURCE],
                    input=json.dumps(payload, ensure_ascii=False, default=str),
                    capture_output=True, text=True, timeout=execution_policy.timeout_seconds,
                    env=env, cwd=folder, check=False,
                )
        except subprocess.TimeoutExpired as exc:
            raise GeneratedCleaningSandboxError("Generated cleaning timed out.") from exc
        output = completed.stdout or ""
        error = completed.stderr or ""
        if len(output.encode()) > execution_policy.output_limit_bytes or len(error.encode()) > 200_000:
            raise GeneratedCleaningSandboxError("Generated cleaning output exceeded the size limit.")
        if completed.returncode != 0:
            raise GeneratedCleaningSandboxError((error or output or "Cleaning subprocess failed.")[:1200])
        try:
            result = json.loads(output)
        except json.JSONDecodeError as exc:
            raise GeneratedCleaningSandboxError("Cleaning subprocess returned invalid JSON.") from exc
    records = result.get("records") if isinstance(result, dict) else None
    if not isinstance(records, list) or not all(isinstance(item, dict) for item in records):
        raise GeneratedCleaningSandboxError("clean_dataset(df) did not return valid records.")
    return pd.DataFrame(records)


def _run_remote(payload: dict[str, Any], policy: CleaningExecutionPolicy) -> dict[str, Any]:
    settings = get_settings()
    headers = {"Content-Type": "application/json"}
    if settings.python_runner_shared_secret:
        headers["X-Runner-Token"] = settings.python_runner_shared_secret.get_secret_value()
    request = Request(
        f"{str(settings.python_runner_url).rstrip('/')}/execute",
        data=json.dumps(payload, ensure_ascii=False, default=str).encode(),
        headers=headers, method="POST",
    )
    try:
        with urlopen(request, timeout=settings.python_runner_timeout_seconds) as response:
            raw = response.read(policy.output_limit_bytes + 1)
    except HTTPError as exc:
        detail = exc.read(1200).decode(errors="replace")
        raise GeneratedCleaningSandboxError(f"Python Runner error {exc.code}: {detail}") from exc
    except (URLError, TimeoutError) as exc:
        raise GeneratedCleaningSandboxError(f"Python Runner unavailable: {exc}") from exc
    if len(raw) > policy.output_limit_bytes:
        raise GeneratedCleaningSandboxError("Generated cleaning output exceeded the size limit.")
    envelope = json.loads(raw.decode())
    result = envelope.get("result") if isinstance(envelope, dict) else None
    if not isinstance(result, dict):
        raise GeneratedCleaningSandboxError(str(envelope.get("detail") if isinstance(envelope, dict) else "Runner returned no result."))
    return result


def _jsonable(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=str))


CLEANING_WORKER_SOURCE = textwrap.dedent(
    """
    import builtins
    import json
    import pandas as pd

    payload = json.loads(input())
    namespace = {
        "__builtins__": {
            "abs": builtins.abs, "all": builtins.all, "any": builtins.any,
            "bool": builtins.bool, "dict": builtins.dict, "enumerate": builtins.enumerate,
            "float": builtins.float, "int": builtins.int, "isinstance": builtins.isinstance,
            "len": builtins.len, "list": builtins.list, "max": builtins.max,
            "min": builtins.min, "range": builtins.range, "round": builtins.round,
            "set": builtins.set, "sorted": builtins.sorted, "str": builtins.str,
            "sum": builtins.sum, "tuple": builtins.tuple, "zip": builtins.zip,
        },
        "pd": pd,
    }
    exec(compile(payload["code"], "<datamind-cleaning-agent>", "exec"), namespace, namespace)
    function = namespace.get("clean_dataset")
    if not callable(function):
        raise ValueError("Generated code must define clean_dataset(df).")
    frame = function(pd.DataFrame(payload.get("records") or []).copy(deep=True))
    if not isinstance(frame, pd.DataFrame):
        raise TypeError("clean_dataset(df) must return a pandas DataFrame.")
    frame = frame.where(pd.notna(frame), None)
    print(json.dumps({"records": frame.to_dict(orient="records")}, ensure_ascii=False, default=str))
    """
)
