from __future__ import annotations

import ast
import json
import logging
import tempfile
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel, ConfigDict, Field

from app.analysis.cleaning_sandbox import validate_generated_cleaning_code
from app.analysis.python_sandbox import OUTPUT_LIMIT_BYTES, _validate_generated_code
from app.core.settings import get_settings

logger = logging.getLogger(__name__)


class RunnerRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    code: str = Field(min_length=1, max_length=100_000)
    records: list[dict[str, Any]] = Field(default_factory=list, max_length=200_000)
    execution_kind: str = Field(default="analysis", pattern="^(analysis|cleaning)$")


class RunnerResponse(BaseModel):
    result: dict[str, Any]


def create_runner_app() -> FastAPI:
    app = FastAPI(title="DataMind Python Runner", docs_url=None, redoc_url=None)

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.post("/execute", response_model=RunnerResponse)
    def execute(
        payload: RunnerRequest,
        x_runner_token: str | None = Header(default=None),
    ) -> RunnerResponse:
        settings = get_settings()
        expected = (
            settings.python_runner_shared_secret.get_secret_value()
            if settings.python_runner_shared_secret
            else None
        )
        if expected and x_runner_token != expected:
            raise HTTPException(status_code=401, detail="Invalid Runner token.")
        try:
            if payload.execution_kind == "cleaning":
                validate_generated_cleaning_code(payload.code)
            else:
                _validate_generated_code(ast.parse(payload.code, mode="exec"))
            result = _execute_in_container(payload.model_dump(mode="json"))
        except Exception as exc:
            raise HTTPException(status_code=422, detail=str(exc)[:1000]) from exc
        return RunnerResponse(result=result)

    return app


def _execute_in_container(payload: dict[str, Any]) -> dict[str, Any]:
    import docker
    from requests.exceptions import Timeout as RequestsTimeout

    settings = get_settings()
    client = docker.from_env()
    container: Any | None = None
    temp_root = Path(settings.python_runner_temp_path)
    temp_root.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(prefix="job-", dir=temp_root) as folder:
            payload_path = Path(folder) / "payload.json"
            payload_path.write_text(
                json.dumps(payload, ensure_ascii=False, default=str),
                encoding="utf-8",
            )
            # The Runner is root, while the sandbox intentionally runs as uid 10001.
            # Allow traversal and read-only access without making the job directory listable.
            Path(folder).chmod(0o711)
            payload_path.chmod(0o444)
            relative_payload = payload_path.relative_to(temp_root).as_posix()
            container = client.containers.create(
                settings.python_sandbox_image,
                command=[f"/runner-input/{relative_payload}"],
                network_disabled=True,
                read_only=True,
                mem_limit="512m",
                nano_cpus=1_000_000_000,
                pids_limit=64,
                cap_drop=["ALL"],
                security_opt=["no-new-privileges"],
                tmpfs={"/tmp": "rw,noexec,nosuid,size=64m"},
                volumes={
                    settings.python_runner_volume_name: {
                        "bind": "/runner-input",
                        "mode": "ro",
                    }
                },
            )
            container.start()
            try:
                wait_result = container.wait(
                    timeout=settings.python_runner_container_timeout_seconds
                )
            except RequestsTimeout as exc:
                raise RuntimeError(
                    "Generated Python timed out in the container sandbox."
                ) from exc
            output = container.logs(stdout=True, stderr=False)
            status_code = int((wait_result or {}).get("StatusCode", 1))
            if status_code != 0:
                message = output.decode("utf-8", errors="replace").strip()
                raise RuntimeError(
                    (message or "Python sandbox container failed.")[:1000]
                )
    finally:
        if container is not None:
            try:
                container.reload()
                if container.status in {"created", "running", "restarting", "paused"}:
                    container.kill()
            except Exception:
                logger.warning("Unable to stop Python sandbox container.", exc_info=True)
            try:
                container.remove(force=True)
            except Exception:
                logger.warning("Unable to remove Python sandbox container.", exc_info=True)
        client.close()
    if len(output) > OUTPUT_LIMIT_BYTES:
        raise RuntimeError("Generated Python output exceeded the size limit.")
    result = json.loads(output.decode("utf-8"))
    if not isinstance(result, dict):
        raise RuntimeError("Sandbox function must return a dict.")
    return result


app = create_runner_app()
