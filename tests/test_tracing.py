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


def test_agent_spans_carry_hashes_and_trace_id_reaches_results(spans, tmp_path):
    from arena_evals import config, datasets, run
    from test_e2e_mock import GOOD, TASKS, run_variant

    cfg = config.load()
    ds = datasets.write_jsonl(tmp_path / "e2e.jsonl", TASKS[:1])
    out = run_variant(cfg, ds, tmp_path, GOOD, "traced")
    rows = run.read_results(out.results_path)
    sample_spans = [s for s in spans.get_finished_spans() if s.name == "arena.sample"]
    assert len(sample_spans) == 2
    attrs = dict(sample_spans[0].attributes)
    for key in ("arena.run_id", "arena.dataset_hash", "arena.prompt_hash", "arena.rubric_hash", "arena.model",
                "arena.task_id", "arena.repeat"):
        assert key in attrs, key
    ids = {format(s.context.trace_id, "032x") for s in sample_spans}
    assert {r["trace_id"] for r in rows} == ids
