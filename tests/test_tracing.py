import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from arena_evals import tracing


@pytest.fixture
def spans(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tracing, "_TRACER", provider.get_tracer("test"))
    return exporter


def test_sample_span_off_yields_empty_id(monkeypatch):
    monkeypatch.setattr(tracing, "_TRACER", None)
    with tracing.sample_span(task_id="t") as tid:
        assert tid == ""


def test_sample_span_sets_arena_attributes(spans):
    with tracing.sample_span(run_id="r1", task_id="gate-0001", repeat=0, model="m", skipped=None) as tid:
        assert len(tid) == 32
    (span,) = spans.get_finished_spans()
    assert dict(span.attributes) == {"arena.run_id": "r1", "arena.task_id": "gate-0001", "arena.repeat": 0,
                                     "arena.model": "m"}
