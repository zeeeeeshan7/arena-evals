"""The only module that touches Phoenix/OpenTelemetry. Spans go to a running `phoenix serve`, whose
PHOENIX_WORKING_DIR holds the SQLite trace DB that CI uploads as an artifact."""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_TRACER = None
_PROVIDER = None


def init(working_dir: Path | str | None = None, endpoint: str | None = None) -> None:
    """Register an OTLP exporter to Phoenix and instrument the Anthropic SDK. No-op if already initialised."""
    global _TRACER, _PROVIDER
    if _TRACER is not None:
        return
    from openinference.instrumentation.anthropic import AnthropicInstrumentor
    from phoenix.otel import register

    if working_dir:
        os.environ["PHOENIX_WORKING_DIR"] = str(working_dir)  # read by `phoenix serve`, not by the exporter
    endpoint = endpoint or os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006") + "/v1/traces"
    _PROVIDER = register(project_name="arena-evals", endpoint=endpoint, batch=True, verbose=False)
    AnthropicInstrumentor().instrument(tracer_provider=_PROVIDER)
    _TRACER = _PROVIDER.get_tracer("arena_evals")


def enabled() -> bool:
    return _TRACER is not None


@contextmanager
def sample_span(name: str = "arena.sample", **attrs) -> Iterator[str]:
    """Open a span with arena.<key> attributes; yields the 32-hex trace ID ("" when tracing is off)."""
    if _TRACER is None:
        yield ""
        return
    clean = {f"arena.{k}": (v if isinstance(v, (str, bool, int, float)) else str(v)) for k, v in attrs.items()
             if v is not None}
    with _TRACER.start_as_current_span(name, attributes=clean) as span:
        yield format(span.get_span_context().trace_id, "032x")


def flush() -> None:
    if _PROVIDER is not None:
        _PROVIDER.force_flush()
