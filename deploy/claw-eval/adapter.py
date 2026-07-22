from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from copy import deepcopy
from pathlib import Path
from typing import Any

import httpx
import uvicorn
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse

SCENARIO_RE = re.compile(r"\[DATAMIND_SCENARIO=(DM\d{3})\]", re.IGNORECASE)
PROJECT_ROOT = Path(os.environ.get("DATAMIND_EVAL_PROJECT_ROOT", "D:/datamind")).resolve()
SOURCE_DIR = Path(
    os.environ.get("DATAMIND_EVAL_SOURCE_DIR", str(Path.home() / "Downloads"))
).resolve()
RUNTIME_DIR = Path(
    os.environ.get("DATAMIND_EVAL_RUNTIME_DIR", str(PROJECT_ROOT / "artifacts/claw-eval"))
).resolve()
FIXTURE_DIR = RUNTIME_DIR / "fixtures"
MANIFEST_PATH = FIXTURE_DIR / "scenarios.json"
STORE_DIR = RUNTIME_DIR / "store"
ADAPTER_PORT = int(os.environ.get("PORT", "9320"))
DATAMIND_PORT = int(os.environ.get("DATAMIND_EVAL_API_PORT", "9310"))
DATAMIND_URL = f"http://127.0.0.1:{DATAMIND_PORT}/api/v1"
ANALYSIS_TIMEOUT_SECONDS = float(os.environ.get("DATAMIND_EVAL_TIMEOUT_SECONDS", "900"))

_state_lock = threading.RLock()
_run_lock = threading.Lock()
_datamind_process: subprocess.Popen[bytes] | None = None
_trial_user = f"claw-{uuid.uuid4().hex}"
_audit: dict[str, Any] = {"calls": [], "runs": []}


def _resolve_datamind_python() -> Path:
    configured = os.environ.get("DATAMIND_EVAL_PYTHON")
    if configured:
        path = Path(configured).resolve()
        if path.exists():
            return path
        raise RuntimeError(f"DATAMIND_EVAL_PYTHON does not exist: {path}")

    conda = shutil.which("conda")
    if conda:
        try:
            completed = subprocess.run(
                [conda, "info", "--envs", "--json"],
                check=True,
                capture_output=True,
                text=True,
                timeout=30,
            )
            envs = json.loads(completed.stdout).get("envs", [])
            for env in envs:
                candidate = Path(env)
                if candidate.name.lower() == "datamind-py312":
                    python = candidate / ("python.exe" if os.name == "nt" else "bin/python")
                    if python.exists():
                        return python
        except Exception:
            pass

    if PROJECT_ROOT.exists():
        try:
            completed = subprocess.run(
                [sys.executable, "-c", "import app, pandas, uvicorn"],
                cwd=PROJECT_ROOT,
                capture_output=True,
                timeout=20,
            )
            if completed.returncode == 0:
                return Path(sys.executable)
        except Exception:
            pass
    raise RuntimeError(
        "Cannot locate the DataMind Python environment. Set DATAMIND_EVAL_PYTHON "
        "to the python executable for the datamind-py312 environment."
    )


def _ensure_fixtures(python: Path) -> None:
    if MANIFEST_PATH.exists() and os.environ.get("DATAMIND_EVAL_REBUILD_FIXTURES") != "1":
        return
    command = [
        str(python),
        str(PROJECT_ROOT / "deploy/claw-eval/build_fixtures.py"),
        "--source-dir",
        str(SOURCE_DIR),
        "--output-dir",
        str(FIXTURE_DIR),
    ]
    if os.environ.get("DATAMIND_EVAL_REBUILD_FIXTURES") == "1":
        command.append("--force")
    subprocess.run(command, cwd=PROJECT_ROOT, check=True, timeout=300)


def _datamind_environment() -> dict[str, str]:
    environment = dict(os.environ)
    provider = os.environ.get("DATAMIND_EVAL_LLM_PROVIDER", "kimi").strip().lower()
    model = "kimi-k2.6" if provider == "kimi" else "mock-model"
    environment.update(
        {
            "DATAMIND_DATASET_STORE_PATH": str(STORE_DIR),
            "DATAMIND_AUTH_MODE": "legacy",
            "DATAMIND_EXECUTION_BACKEND": "local",
            "DATAMIND_ENVIRONMENT": "development",
            "DATAMIND_RATE_LIMITS_ENABLED": "false",
            "DATAMIND_EVAL_EXPECTED_PROVIDER": provider,
            "DATAMIND_LLM_ALLOW_PROVIDER_FALLBACK": "false",
            "DATAMIND_LLM_PROVIDER": provider,
            "DATAMIND_DEFAULT_LLM_PROVIDER": provider,
            "DATAMIND_CLEANING_LLM_PROVIDER": provider,
            "DATAMIND_PLANNER_LLM_PROVIDER": provider,
            "DATAMIND_SQL_LLM_PROVIDER": provider,
            "DATAMIND_PYTHON_LLM_PROVIDER": provider,
            "DATAMIND_REFLECTION_LLM_PROVIDER": provider,
            "DATAMIND_REPORT_LLM_PROVIDER": provider,
            "DATAMIND_REVIEW_LLM_PROVIDER": provider,
            "DATAMIND_MULTIMODAL_LLM_PROVIDER": provider,
            "DATAMIND_AGENT_LOOP_PROVIDER": provider,
            "DATAMIND_AGENT_LOOP_MODEL": model,
            "DATAMIND_KIMI_MODEL": "kimi-k2.6",
            "DATAMIND_KIMI_BASE_URL": os.environ.get(
                "DATAMIND_KIMI_BASE_URL", "https://api.moonshot.cn/v1"
            ),
            "DATAMIND_LLM_TIMEOUT_SECONDS": os.environ.get(
                "DATAMIND_LLM_TIMEOUT_SECONDS", "180"
            ),
            "DATAMIND_LLM_TRANSIENT_RETRIES": os.environ.get(
                "DATAMIND_LLM_TRANSIENT_RETRIES", "4"
            ),
            "DATAMIND_LLM_RETRY_BACKOFF_SECONDS": os.environ.get(
                "DATAMIND_LLM_RETRY_BACKOFF_SECONDS", "2"
            ),
            "DATAMIND_AGENT_LOOP_TIMEOUT_SECONDS": os.environ.get(
                "DATAMIND_AGENT_LOOP_TIMEOUT_SECONDS", "600"
            ),
            "DATAMIND_AGENT_LOOP_DEFAULT_MODE": "loop",
            "DATAMIND_AGENT_LOOP_ENABLED": "true",
        }
    )
    return environment


def _start_datamind(python: Path) -> subprocess.Popen[bytes]:
    STORE_DIR.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [
            str(python),
            str(PROJECT_ROOT / "deploy/claw-eval/datamind_server.py"),
            "--parent-pid",
            str(os.getpid()),
            "--port",
            str(DATAMIND_PORT),
        ],
        cwd=PROJECT_ROOT,
        env=_datamind_environment(),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
    )
    deadline = time.monotonic() + 120
    while time.monotonic() < deadline:
        if process.poll() is not None:
            stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
            raise RuntimeError(f"DataMind API exited during startup: {stderr[-1000:]}")
        try:
            response = httpx.get(f"{DATAMIND_URL}/health", timeout=2, trust_env=False)
            if response.status_code < 500:
                return process
        except Exception:
            pass
        time.sleep(0.5)
    process.terminate()
    raise RuntimeError("DataMind API did not become healthy within 120 seconds.")


def _stop_datamind() -> None:
    global _datamind_process
    process = _datamind_process
    _datamind_process = None
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    global _datamind_process
    python = await asyncio.to_thread(_resolve_datamind_python)
    await asyncio.to_thread(_ensure_fixtures, python)
    _datamind_process = await asyncio.to_thread(_start_datamind, python)
    try:
        yield
    finally:
        await asyncio.to_thread(_stop_datamind)


app = FastAPI(title="DataMind claw-eval adapter", version="1.0", lifespan=lifespan)


def _message_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(
            str(item.get("text") or "")
            for item in content
            if isinstance(item, dict) and item.get("type") == "text"
        )
    return ""


def _extract_scenario(payload: dict[str, Any]) -> tuple[str, str]:
    for message in reversed(payload.get("messages") or []):
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        text = _message_text(message.get("content"))
        match = SCENARIO_RE.search(text)
        if match:
            return match.group(1).upper(), SCENARIO_RE.sub("", text).strip()
    raise ValueError("The user prompt is missing [DATAMIND_SCENARIO=DMxxx].")


def _manifest() -> dict[str, Any]:
    return json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))


def _request(
    client: httpx.Client,
    method: str,
    path: str,
    *,
    user: str,
    **kwargs: Any,
) -> dict[str, Any]:
    response = client.request(
        method,
        f"{DATAMIND_URL}{path}",
        headers={"X-DataMind-User": user},
        **kwargs,
    )
    try:
        response_body: Any = response.json()
    except Exception:
        response_body = {"text": response.text[:1000]}
    with _state_lock:
        _audit["calls"].append(
            {
                "endpoint": path,
                "method": method,
                "response_status": response.status_code,
                "response_body": _redact(response_body),
            }
        )
    if response.status_code >= 400:
        raise RuntimeError(f"DataMind {method} {path} failed: {response.status_code} {response_body}")
    if not isinstance(response_body, dict):
        raise RuntimeError(f"DataMind {method} {path} returned a non-object response.")
    return response_body


def _import_files(
    client: httpx.Client,
    scenario: dict[str, Any],
    generated: dict[str, Any],
    *,
    user: str,
) -> dict[str, str]:
    dataset_ids: dict[str, str] = {}
    for logical_name in scenario["files"]:
        info = generated[logical_name]
        path = FIXTURE_DIR / "data" / info["path"]
        with path.open("rb") as handle:
            response = _request(
                client,
                "POST",
                "/store/files/import",
                user=user,
                files={"file": (path.name, handle, "text/csv")},
                data={"dataset_name": logical_name},
            )
        dataset_ids[logical_name] = str(response["dataset"]["dataset_id"])
    return dataset_ids


def _relationships(
    scenario: dict[str, Any], dataset_ids: dict[str, str]
) -> list[dict[str, Any]]:
    return [
        {
            "left_dataset_id": dataset_ids[item["left"]],
            "right_dataset_id": dataset_ids[item["right"]],
            "left_column": item["left_column"],
            "right_column": item["right_column"],
            "join_type": "left",
            "relationship_type": "many_to_one",
            "source": "user",
            "confidence": 1.0,
            "reason": "Deterministic evaluation relationship",
        }
        for item in scenario.get("relationships", [])
    ]


def _analysis_relationships(values: list[dict[str, Any]]) -> list[dict[str, Any]]:
    allowed = {
        "left_dataset_id",
        "right_dataset_id",
        "left_column",
        "right_column",
        "join_type",
        "left_value_mode",
        "right_value_mode",
        "left_delimiter",
        "right_delimiter",
    }
    return [{key: value for key, value in item.items() if key in allowed} for item in values]


def _compact_result(result: dict[str, Any]) -> dict[str, Any]:
    allowed = {
        "dataset_id",
        "dataset_group_id",
        "report_id",
        "question",
        "plan",
        "planner_metadata",
        "multi_dataset_context",
        "profile",
        "analysis_framework",
        "sql_result",
        "python_result",
        "rounds",
        "final_insights",
        "validation_issues",
        "structured_report",
        "report_markdown",
        "agent_mode",
        "loop_summary",
        "loop_terminal_reason",
        "report_strategy",
        "report_revision_count",
        "report_terminal_reason",
        "workflow_trace",
    }
    return _redact({key: value for key, value in result.items() if key in allowed})


def _redact(value: Any) -> Any:
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            if any(token in key.lower() for token in ("api_key", "authorization", "secret")):
                redacted[key] = "[REDACTED]"
            else:
                redacted[key] = _redact(item)
        return redacted
    if isinstance(value, list):
        return [_redact(item) for item in value]
    if isinstance(value, str):
        result = re.sub(r"sk-[A-Za-z0-9_-]{8,}", "[REDACTED]", value)
        result = re.sub(
            r"\b[A-Za-z0-9]{20,}\.[A-Za-z0-9_-]{8,}\b", "[REDACTED]", result
        )
        return result
    return value


def _token_usage(job: dict[str, Any]) -> dict[str, int]:
    prompt = completion = total = 0
    for event in job.get("events") or []:
        usage = event.get("token_usage") or {}
        prompt += int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
        completion += int(usage.get("completion_tokens") or usage.get("output_tokens") or 0)
        total += int(usage.get("total_tokens") or 0)
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total or prompt + completion,
    }


def _run_scenario(scenario_id: str, question: str) -> dict[str, Any]:
    started = time.monotonic()
    with _run_lock:
        manifest = _manifest()
        scenario = manifest["scenarios"].get(scenario_id)
        if not scenario:
            raise RuntimeError(f"Unknown DataMind scenario: {scenario_id}")
        effective_question = question or str(scenario["question"])
        user = _trial_user
        run_record: dict[str, Any] = {
            "scenario_id": scenario_id,
            "user_namespace": user,
            "status": "running",
            "question": effective_question,
        }
        with _state_lock:
            _audit["runs"].append(run_record)
        try:
            with httpx.Client(timeout=180, trust_env=False) as client:
                dataset_ids = _import_files(
                    client, scenario, manifest["generated"], user=user
                )
                primary = scenario["primary"]
                primary_id = dataset_ids[primary]
                additional_ids = [
                    value for key, value in dataset_ids.items() if key != primary
                ]
                group_id: str | None = None
                relationship_plan: list[dict[str, Any]] = []
                relationship_audit: dict[str, Any] = {}
                if additional_ids:
                    group = _request(
                        client,
                        "POST",
                        "/store/dataset-groups",
                        user=user,
                        json={
                            "name": f"{scenario_id}-{user}",
                            "dataset_ids": list(dataset_ids.values()),
                            "description": "Isolated DataMind claw-eval dataset group",
                            "metadata": {"scenario_id": scenario_id},
                        },
                    )
                    group_id = str(group["group_id"])
                    if scenario.get("auto_configure_relationships"):
                        configured = _request(
                            client,
                            "POST",
                            f"/store/dataset-groups/{group_id}/relationships/auto-configure",
                            user=user,
                        )
                        relationship_plan = list(configured.get("saved_relationships") or [])
                        relationship_audit = {
                            "candidates": configured.get("candidates") or [],
                            "saved_relationships": relationship_plan,
                            "unresolved_dataset_ids": configured.get("unresolved_dataset_ids") or [],
                            "validation_issues": configured.get("validation_issues") or [],
                        }
                        if configured.get("primary_dataset_id"):
                            primary_id = str(configured["primary_dataset_id"])
                            additional_ids = [
                                value for value in dataset_ids.values() if value != primary_id
                            ]
                    else:
                        relationship_plan = _relationships(scenario, dataset_ids)
                        _request(
                            client,
                            "PATCH",
                            f"/store/dataset-groups/{group_id}/relationships",
                            user=user,
                            json={"relationships": relationship_plan},
                        )

                analysis_relationships = _analysis_relationships(relationship_plan)
                job = _request(
                    client,
                    "POST",
                    "/analysis/jobs",
                    user=user,
                    json={
                        "dataset_id": primary_id,
                        "dataset_group_id": group_id,
                        "additional_dataset_ids": additional_ids,
                        "join_plan": analysis_relationships,
                        "relationship_plan": analysis_relationships,
                        "question": effective_question,
                        "agent_mode": "loop",
                        "confirmed_low_confidence": True,
                    },
                )
                job_id = str(job["job_id"])
                deadline = time.monotonic() + ANALYSIS_TIMEOUT_SECONDS
                while job.get("status") in {"queued", "running", "cancel_requested"}:
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"DataMind analysis exceeded {ANALYSIS_TIMEOUT_SECONDS:.0f}s"
                        )
                    time.sleep(1)
                    job = _request(
                        client, "GET", f"/analysis/jobs/{job_id}", user=user
                    )
                if job.get("status") != "completed":
                    raise RuntimeError(
                        f"DataMind job {job_id} ended as {job.get('status')}: {job.get('error')}"
                    )
                result = _request(
                    client, "GET", f"/analysis/jobs/{job_id}/result", user=user
                )
                if (
                    os.environ.get("DATAMIND_EVAL_LLM_PROVIDER", "kimi").strip().lower()
                    != "mock"
                    and str(result.get("loop_terminal_reason") or "") == "provider_error"
                ):
                    raise RuntimeError(
                        "DataMind agent loop provider failed; refusing to score a "
                        "deterministic fallback as a Kimi-driven evaluation."
                    )
                compact = _compact_result(result)
                answer = str(compact.get("report_markdown") or "").strip()
                if not answer:
                    answer = json.dumps(
                        compact.get("structured_report") or compact,
                        ensure_ascii=False,
                        default=str,
                    )
                usage = _token_usage(job)
                run_record.update(
                    {
                        "status": "completed",
                        "job_id": job_id,
                        "dataset_ids": dataset_ids,
                        "relationships": _redact(relationship_audit or relationship_plan),
                        "result": compact,
                        "token_usage": usage,
                        "wall_time_s": round(time.monotonic() - started, 3),
                    }
                )
                return {"answer": answer[:120_000], "usage": usage}
        except Exception as exc:
            run_record.update(
                {
                    "status": "failed",
                    "error": f"{type(exc).__name__}: {_redact(str(exc))}",
                    "wall_time_s": round(time.monotonic() - started, 3),
                }
            )
            raise


def _chunk(
    completion_id: str,
    *,
    delta: dict[str, Any] | None = None,
    finish_reason: str | None = None,
    usage: dict[str, int] | None = None,
) -> str:
    payload: dict[str, Any] = {
        "id": completion_id,
        "object": "chat.completion.chunk",
        "created": int(time.time()),
        "model": "datamind-core",
        "choices": [
            {
                "index": 0,
                "delta": delta or {},
                "finish_reason": finish_reason,
            }
        ],
    }
    if usage is not None:
        payload["usage"] = usage
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _stream_completion(scenario_id: str, question: str) -> AsyncIterator[str]:
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    yield _chunk(completion_id, delta={"role": "assistant", "content": ""})
    task = asyncio.create_task(asyncio.to_thread(_run_scenario, scenario_id, question))
    while not task.done():
        done, _ = await asyncio.wait({task}, timeout=10)
        if not done:
            yield _chunk(completion_id, delta={"content": ""})
    try:
        result = await task
        yield _chunk(completion_id, delta={"content": result["answer"]})
        yield _chunk(
            completion_id,
            finish_reason="stop",
            usage=result["usage"],
        )
    except Exception as exc:
        message = f"DataMind evaluation failed: {type(exc).__name__}: {_redact(str(exc))}"
        yield _chunk(completion_id, delta={"content": message})
        yield _chunk(completion_id, finish_reason="stop", usage={"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})
    yield "data: [DONE]\n\n"


@app.get("/health")
@app.post("/health")
def health() -> dict[str, Any]:
    process_ready = _datamind_process is not None and _datamind_process.poll() is None
    return {
        "status": "ok" if process_ready and MANIFEST_PATH.exists() else "starting",
        "datamind": process_ready,
        "fixtures": MANIFEST_PATH.exists(),
    }


@app.post("/reset")
def reset() -> dict[str, Any]:
    global _trial_user, _audit
    with _state_lock:
        _trial_user = f"claw-{uuid.uuid4().hex}"
        _audit = {"calls": [], "runs": []}
    return {"status": "ok", "namespace": _trial_user}


@app.get("/audit")
def audit() -> dict[str, Any]:
    with _state_lock:
        return _redact(deepcopy(_audit))


@app.post("/v1/chat/completions")
async def chat_completions(payload: dict[str, Any]) -> Any:
    try:
        scenario_id, question = _extract_scenario(payload)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    if payload.get("stream"):
        return StreamingResponse(
            _stream_completion(scenario_id, question), media_type="text/event-stream"
        )
    try:
        result = await asyncio.to_thread(_run_scenario, scenario_id, question)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=_redact(str(exc))) from exc
    return {
        "id": f"chatcmpl-{uuid.uuid4().hex}",
        "object": "chat.completion",
        "created": int(time.time()),
        "model": "datamind-core",
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result["answer"]},
                "finish_reason": "stop",
            }
        ],
        "usage": result["usage"],
    }


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=ADAPTER_PORT, log_level="warning")
