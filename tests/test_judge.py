import asyncio

import pytest
from inspect_ai.model import ModelOutput, get_model

from arena_evals import agents, config
from arena_evals.agents import CostCapExceeded, CostMeter
from arena_evals.scorers import RUBRIC_DIR, judge_answer, parse_verdict

M = "mockllm/model"


def model(replies):
    it = iter(replies)
    return get_model(M, custom_outputs=lambda *a: ModelOutput.from_content(M, next(it)), memoize=False)


def ask(m):
    return asyncio.run(judge_answer(m, RUBRIC_DIR, "Q?", "ref", "ans", {"HR-001": "doc text"}, cache=False))


@pytest.fixture(autouse=True)
def meter():
    agents.set_meter(CostMeter(float("inf")))
    yield
    agents.set_meter(CostMeter(float("inf")))


def test_parse_verdict_accepts_fenced_json_and_rejects_bad_schema():
    v = parse_verdict('```json\n{"correct": true, "faithful": false, "reason": "r"}\n```')
    assert (v.correct, v.faithful, v.reason) == (True, False, "r")
    for bad in ('{"correct": "yes", "faithful": true, "reason": "r"}',   # strict bool
                '{"correct": true, "faithful": true}',                    # missing reason
                '{"correct": true, "faithful": true, "reason": "r", "score": 5}',
                "no json here"):
        with pytest.raises(ValueError):
            parse_verdict(bad)


def test_retry_once_then_valid():
    v, _ = ask(model(["nope", '{"correct": false, "faithful": true, "reason": "wrong value"}']))
    assert v.correct is False and v.reason == "wrong value"


def test_two_invalid_replies_is_judge_error():
    v, _ = ask(model(["nope", "still nope", '{"correct": true, "faithful": true, "reason": "never reached"}']))
    assert v is None


def test_judge_respects_cost_cap():
    agents.set_meter(CostMeter(0.0))
    with pytest.raises(CostCapExceeded):
        ask(model(['{"correct": true, "faithful": true, "reason": "r"}']))


def test_rubric_hash_covers_both_files():
    h = config.rubric_hash(RUBRIC_DIR)
    assert h == config.sha256_bytes((RUBRIC_DIR / "system.md").read_bytes(), (RUBRIC_DIR / "rubric.md").read_bytes())
