from __future__ import annotations

from threading import Lock
from typing import Any

from app.core.settings import Settings, get_settings

_configured = False
_lock = Lock()


def configure_observability(
    service_name: str,
    settings: Settings | None = None,
) -> None:
    global _configured
    resolved = settings or get_settings()
    if _configured or not resolved.otel_exporter_otlp_endpoint:
        return
    with _lock:
        if _configured:
            return
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor

        provider = TracerProvider(
            resource=Resource.create(
                {
                    "service.name": service_name,
                    "deployment.environment": resolved.environment,
                }
            )
        )
        exporter = OTLPSpanExporter(endpoint=resolved.otel_exporter_otlp_endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        _configured = True


def tracer(name: str) -> Any:
    try:
        from opentelemetry import trace

        return trace.get_tracer(name)
    except ImportError:
        return _NoopTracer()


class _NoopSpan:
    def __enter__(self) -> _NoopSpan:
        return self

    def __exit__(self, *_args: Any) -> None:
        return None

    def set_attribute(self, *_args: Any) -> None:
        return None

    def record_exception(self, *_args: Any) -> None:
        return None


class _NoopTracer:
    def start_as_current_span(self, *_args: Any, **_kwargs: Any) -> _NoopSpan:
        return _NoopSpan()
