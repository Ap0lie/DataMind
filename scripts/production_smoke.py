from __future__ import annotations

import argparse
import json
import math
import time
import urllib.error
import urllib.request
from datetime import UTC, datetime
from http.cookiejar import CookieJar
from pathlib import Path
from threading import Thread
from typing import Any


class SmokeClient:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")
        self.cookies = CookieJar()
        self.opener = urllib.request.build_opener(urllib.request.HTTPCookieProcessor(self.cookies))
        self.csrf_token = ""

    def request(
        self,
        method: str,
        path: str,
        payload: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        body = json.dumps(payload).encode("utf-8") if payload is not None else None
        headers = {"Accept": "application/json", "Origin": "http://127.0.0.1:5173"}
        cookie_header = "; ".join(f"{cookie.name}={cookie.value}" for cookie in self.cookies)
        if cookie_header:
            # The production profile correctly marks the session Secure. This CLI smoke
            # runs over the Compose HTTP port, so it explicitly replays the cookie.
            headers["Cookie"] = cookie_header
        if body is not None:
            headers["Content-Type"] = "application/json"
        if self.csrf_token and method not in {"GET", "HEAD", "OPTIONS"}:
            headers["X-CSRF-Token"] = self.csrf_token
        request = urllib.request.Request(
            f"{self.base_url}{path}",
            data=body,
            headers=headers,
            method=method,
        )
        try:
            with self.opener.open(request, timeout=20) as response:
                content = response.read()
                return json.loads(content) if content else {}
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"{method} {path} failed with HTTP {exc.code}: {detail}") from exc

    def stream_events(self, path: str, *, timeout: float) -> list[dict[str, Any]]:
        headers = {"Accept": "text/event-stream", "Origin": "http://127.0.0.1:5173"}
        cookie_header = "; ".join(f"{cookie.name}={cookie.value}" for cookie in self.cookies)
        if cookie_header:
            headers["Cookie"] = cookie_header
        request = urllib.request.Request(f"{self.base_url}{path}", headers=headers, method="GET")
        events: list[dict[str, Any]] = []
        with self.opener.open(request, timeout=timeout) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8", errors="replace").strip()
                if not line.startswith("data: "):
                    continue
                payload = json.loads(line[6:])
                if "sequence" not in payload:
                    break
                received_at = datetime.now(UTC)
                created_at = _parse_datetime(payload.get("created_at"))
                payload["delivery_latency_ms"] = max(
                    0.0, (received_at - created_at).total_seconds() * 1000.0
                )
                events.append(payload)
        return events


def wait_for_ready(client: SmokeClient, timeout: float) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    last_error = "service did not respond"
    while time.monotonic() < deadline:
        try:
            payload = client.request("GET", "/health/ready")
            if payload.get("status") in {"ready", "degraded", "ok"}:
                return payload
            last_error = json.dumps(payload, ensure_ascii=False)
        except (OSError, RuntimeError) as exc:
            last_error = str(exc)
        time.sleep(2)
    raise RuntimeError(f"DataMind readiness timeout: {last_error}")


def wait_for_job(
    client: SmokeClient,
    path: str,
    *,
    timeout: float,
) -> dict[str, Any]:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = client.request("GET", path)
        if job.get("status") in {"completed", "failed", "canceled", "interrupted"}:
            return job
        time.sleep(1)
    raise RuntimeError(f"Job did not finish in {timeout:.0f}s: {path}")


def run(base_url: str, timeout: float) -> dict[str, Any]:
    smoke_started = time.perf_counter()
    client = SmokeClient(base_url)
    ready = wait_for_ready(client, timeout)
    print("ready", json.dumps(ready, ensure_ascii=False, sort_keys=True))

    login = client.request(
        "POST",
        "/auth/login",
        {"username": "production-smoke", "password": "production-smoke-password"},
    )
    client.csrf_token = str(login["csrf_token"])
    assert login["user_id"] == "production_smoke"

    dataset = client.request(
        "POST",
        "/store/datasets",
        {"name": "production-smoke.csv", "source_type": "csv", "source_metadata": {"smoke": True}},
    )
    dataset_id = str(dataset["dataset_id"])
    appended = client.request(
        "POST",
        f"/store/datasets/{dataset_id}/raw-records",
        {
            "records": [
                {"region": "North", "sales": 120.0, "month": "2026-01"},
                {"region": "South", "sales": 180.0, "month": "2026-01"},
                {"region": "North", "sales": 160.0, "month": "2026-02"},
            ]
        },
    )
    assert appended["inserted"] == 3

    cleaning_started = time.perf_counter()
    cleaning = client.request(
        "POST",
        f"/store/datasets/{dataset_id}/cleaning-jobs",
        {"cleaning_strategy": "rules", "requirement": "Trim text and preserve valid rows."},
    )
    cleaning_id = str(cleaning["job_id"])
    cleaning = wait_for_job(
        client,
        f"/store/datasets/{dataset_id}/cleaning-jobs/{cleaning_id}",
        timeout=timeout,
    )
    if cleaning.get("status") != "completed" or not cleaning.get("cleaning_run_id"):
        raise RuntimeError(f"Cleaning failed: {json.dumps(cleaning, ensure_ascii=False)}")
    cleaning_duration = time.perf_counter() - cleaning_started

    analysis_started = time.perf_counter()
    analysis = client.request(
        "POST",
        "/analysis/jobs",
        {
            "dataset_id": dataset_id,
            "question": "按地区汇总销售额并给出简明结论",
            "agent_mode": "auto",
            "prompt_overrides": {"report": "输出简洁标准报告，保留数值证据。"},
        },
    )
    job_id = str(analysis["job_id"])
    stream_result: dict[str, Any] = {"events": [], "error": None}

    def collect_events() -> None:
        try:
            stream_result["events"] = client.stream_events(
                f"/analysis/jobs/{job_id}/events", timeout=timeout
            )
        except Exception as exc:  # pragma: no cover - exercised by production smoke.
            stream_result["error"] = f"{type(exc).__name__}: {exc}"

    stream_thread = Thread(target=collect_events, name="production-smoke-sse", daemon=True)
    stream_thread.start()
    analysis = wait_for_job(client, f"/analysis/jobs/{job_id}", timeout=timeout)
    if analysis.get("status") != "completed" or not analysis.get("report_id"):
        raise RuntimeError(f"Analysis failed: {json.dumps(analysis, ensure_ascii=False)}")
    result = client.request("GET", f"/analysis/jobs/{job_id}/result")
    assert result["agent_mode"] == "loop", result
    assert result["report_id"] == analysis["report_id"], result
    assert result.get("structured_report"), result
    stream_thread.join(timeout=min(timeout, 30.0))
    if stream_thread.is_alive():
        raise RuntimeError("Analysis SSE stream did not terminate after the job completed.")
    if stream_result["error"]:
        raise RuntimeError(f"Analysis SSE failed: {stream_result['error']}")
    event_latencies = [
        float(event["delivery_latency_ms"]) for event in stream_result["events"]
    ]
    metrics = {
        "success": True,
        "cleaning_duration_seconds": round(cleaning_duration, 3),
        "analysis_duration_seconds": round(time.perf_counter() - analysis_started, 3),
        "total_duration_seconds": round(time.perf_counter() - smoke_started, 3),
        "sse_event_count": len(event_latencies),
        "sse_delivery_p95_ms": round(_percentile(event_latencies, 0.95), 3)
        if event_latencies
        else None,
    }
    print(
        "production smoke passed",
        json.dumps(
            {
                "dataset_id": dataset_id,
                "cleaning_job_id": cleaning_id,
                "analysis_job_id": job_id,
                "report_id": analysis["report_id"],
            },
            ensure_ascii=False,
        ),
    )
    return metrics


def _parse_datetime(value: Any) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    return parsed if parsed.tzinfo is not None else parsed.replace(tzinfo=UTC)


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(len(ordered) * fraction) - 1))]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the real DataMind production-stack smoke flow.")
    parser.add_argument("--base-url", default="http://127.0.0.1:8010/api/v1")
    parser.add_argument("--timeout", type=float, default=300)
    parser.add_argument("--benchmark-output", type=Path)
    args = parser.parse_args()
    metrics = run(args.base_url, args.timeout)
    if args.benchmark_output:
        args.benchmark_output.parent.mkdir(parents=True, exist_ok=True)
        args.benchmark_output.write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
