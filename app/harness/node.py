from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from app.observability import tracer

logger = logging.getLogger("datamind.harness.node")

NodeHandler = Callable[[dict[str, Any]], Mapping[str, Any]]
NodeEventCallback = Callable[[Any, dict[str, Any]], None]


@dataclass(frozen=True)
class NodeHarnessPolicy:
    transient_retries: int = 1
    backoff_seconds: float = 0.2


class NodeExecutionHarness:
    """Reliability boundary around LangGraph nodes without becoming a scheduler."""

    def __init__(
        self,
        policy: NodeHarnessPolicy | None = None,
        event_callback: NodeEventCallback | None = None,
    ) -> None:
        self._policy = policy or NodeHarnessPolicy()
        self._event_callback = event_callback

    def wrap(self, node_name: str, handler: Callable[[Any], Any]) -> Callable[[Any], Any]:
        def run(state: Any) -> Any:
            started = time.monotonic()
            attempt = 0
            while True:
                attempt += 1
                with tracer("datamind.workflow").start_as_current_span(
                    f"workflow.node.{node_name}"
                ) as span:
                    span.set_attribute("datamind.node", node_name)
                    span.set_attribute("datamind.attempt", attempt)
                    try:
                        output = handler(state)
                        if not isinstance(output, Mapping):
                            raise TypeError(
                                f"LangGraph node '{node_name}' returned "
                                f"{type(output).__name__}; expected a mapping."
                            )
                        self._record(node_name, "succeeded", attempt, started)
                        self._emit(
                            state,
                            node_name=node_name,
                            status="completed",
                            attempt=attempt,
                            started=started,
                            output=output,
                        )
                        return output
                    except Exception as exc:
                        span.record_exception(exc)
                        retry = attempt <= self._policy.transient_retries and _is_transient(exc)
                        self._record(
                            node_name,
                            "retrying" if retry else "failed",
                            attempt,
                            started,
                            error=exc,
                        )
                        self._emit(
                            state,
                            node_name=node_name,
                            status="retrying" if retry else "failed",
                            attempt=attempt,
                            started=started,
                            error=exc,
                        )
                        if not retry:
                            raise
                        if self._policy.backoff_seconds > 0:
                            time.sleep(self._policy.backoff_seconds * attempt)

        return run

    def _emit(
        self,
        state: Any,
        *,
        node_name: str,
        status: str,
        attempt: int,
        started: float,
        output: Mapping[str, Any] | None = None,
        error: Exception | None = None,
    ) -> None:
        if self._event_callback is None:
            return
        payload = {
            "node": node_name,
            "status": status,
            "attempt": attempt,
            "duration_ms": round((time.monotonic() - started) * 1000, 3),
            "provider": output.get("model_router_provider") if output else None,
            "model": output.get("model_router_model") if output else None,
            "error_code": type(error).__name__ if error else None,
            "message": str(error)[:1000] if error else f"{node_name} completed.",
        }
        self._event_callback(state, payload)

    @staticmethod
    def _record(
        node_name: str,
        status: str,
        attempt: int,
        started: float,
        *,
        error: Exception | None = None,
    ) -> None:
        payload = {
            "node": node_name,
            "status": status,
            "attempt": attempt,
            "duration_ms": round((time.monotonic() - started) * 1000, 3),
        }
        if error is not None:
            payload["error_type"] = type(error).__name__
            payload["error"] = str(error)[:1000]
        logger.info("node_execution %s", json.dumps(payload, ensure_ascii=False))


def _is_transient(exc: Exception) -> bool:
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "timed out",
            "timeout",
            "connection reset",
            "temporarily unavailable",
            "too many requests",
            "http 429",
            "http 500",
            "http 502",
            "http 503",
            "http 504",
        )
    )
