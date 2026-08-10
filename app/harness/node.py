from __future__ import annotations

import json
import logging
import time
from collections.abc import Callable, Mapping
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Any

from app.observability import tracer

logger = logging.getLogger("datamind.harness.node")

NodeHandler = Callable[[dict[str, Any]], Mapping[str, Any]]
NodeEventCallback = Callable[[Any, dict[str, Any]], None]
_NODE_DEADLINE: ContextVar[float | None] = ContextVar(
    "datamind_node_deadline", default=None
)


class NodeExecutionTimeout(TimeoutError):
    """Raised when a node crosses its cooperative wall-clock deadline."""


@dataclass(frozen=True)
class NodeHarnessPolicy:
    transient_retries: int = 1
    backoff_seconds: float = 0.2
    timeout_seconds: float | None = None


def remaining_node_timeout(default: float | None = None) -> float | None:
    """Return the current node's remaining budget, optionally capped by a default."""

    deadline = _NODE_DEADLINE.get()
    if deadline is None:
        return default
    remaining = max(0.0, deadline - time.monotonic())
    return remaining if default is None else min(default, remaining)


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
            deadline = (
                started + self._policy.timeout_seconds
                if self._policy.timeout_seconds is not None
                else None
            )
            deadline_token = _NODE_DEADLINE.set(deadline)
            attempt = 0
            try:
                while True:
                    attempt += 1
                    with tracer("datamind.workflow").start_as_current_span(
                        f"workflow.node.{node_name}"
                    ) as span:
                        span.set_attribute("datamind.node", node_name)
                        span.set_attribute("datamind.attempt", attempt)
                        try:
                            _raise_if_expired(node_name, deadline)
                            output = handler(state)
                            _raise_if_expired(node_name, deadline)
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
                            expired = deadline is not None and time.monotonic() >= deadline
                            retry = (
                                not expired
                                and not isinstance(exc, NodeExecutionTimeout)
                                and attempt <= self._policy.transient_retries
                                and _is_transient(exc)
                            )
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
                            delay = self._policy.backoff_seconds * attempt
                            if deadline is not None:
                                delay = min(delay, max(0.0, deadline - time.monotonic()))
                            if delay > 0:
                                time.sleep(delay)
            finally:
                _NODE_DEADLINE.reset(deadline_token)

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


def _raise_if_expired(node_name: str, deadline: float | None) -> None:
    if deadline is not None and time.monotonic() >= deadline:
        raise NodeExecutionTimeout(f"LangGraph node '{node_name}' exceeded its deadline.")
