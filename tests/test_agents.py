import asyncio
from types import SimpleNamespace

import pytest

from arena_evals.agents import CostCapExceeded, CostMeter, calculate, get_doc, search_docs, set_meter


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def unlimited_meter():
    set_meter(CostMeter(float("inf")))
    yield
    set_meter(CostMeter(float("inf")))


def test_search_docs_format():
    out = run(search_docs()(query="Pro tier price per seat"))
    first = out.splitlines()[0]
    doc_id, title, snippet = first.split(" | ", 2)
    assert doc_id == "PRC-001" and title == "Fleet Console Pricing (2024)"
    assert len(out.splitlines()) == 5


def test_search_docs_empty_result_is_tool_output_not_exception():
    assert run(search_docs()(query="")) == "no results for query: "
    assert run(search_docs()(query="xylophone quasar")) == "no results for query: xylophone quasar"


def test_get_doc_known_and_unknown():
    assert run(get_doc()(id="PRC-006")).startswith("---\nid: PRC-006\n")
    assert run(get_doc()(id="HR-999")) == "error: no document with id HR-999"


def test_calculate_tool_returns_errors_as_output():
    assert run(calculate()(expr="179 * 25")) == "4475"
    assert run(calculate()(expr="__import__('os')")) == "error: unsupported element: Call"


def test_cost_meter_charges_and_caps():
    m = CostMeter(0.01, {"m": {"input": 1.0, "output": 5.0}, "default": {"input": 15.0, "output": 75.0}})
    assert m.charge(SimpleNamespace(input_tokens=1000, output_tokens=1000), "m") == pytest.approx(0.006)
    m.check()
    m.charge(SimpleNamespace(input_tokens=1000, output_tokens=0), "unknown-model")  # default price: 0.015
    with pytest.raises(CostCapExceeded, match=r"cost cap hit \(\$0.02 of \$0.01\)"):
        m.check()


def test_tools_refuse_after_cap():
    set_meter(CostMeter(0.0))
    with pytest.raises(CostCapExceeded):
        run(get_doc()(id="HR-001"))
