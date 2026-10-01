import pytest

from arena_evals.scorers import citation_pr, first_number, match_answer, normalize_answer, task_success

IDS = {"HR-001", "HR-009", "PRC-006"}


def test_normalize_and_exact():
    assert normalize_answer("  Dev  Okafor. ") == "dev okafor"
    assert match_answer("dev okafor", "Dev Okafor", "exact")
    assert not match_answer("Dev Okafor, CTO", "Dev Okafor", "exact")


@pytest.mark.parametrize("answer, ref, ok", [
    ("$4,475", "4,475 USD", True),
    ("4475.00 USD per month", "4475 USD", True),
    ("4,497 USD", "4475 USD", True),      # within 0.5%
    ("4,500 USD", "4475 USD", False),     # 0.56% off
    ("12%", "12%", True),
    ("0.004", "0", True),                 # absolute floor 0.01
    ("about twenty", "20", False),
    ("-3 degrees", "-3", True),
])
def test_numeric(answer, ref, ok):
    assert match_answer(answer, ref, "numeric") is ok


def test_first_number():
    assert first_number("1,340 alarms") == 1340.0
    assert first_number("£12.5") == 12.5
    assert first_number("none") is None


def test_free_text_never_matches_deterministically():
    assert match_answer("An expired TLS certificate", "An expired TLS certificate", "free_text") is False


def test_citation_precision_recall():
    assert citation_pr(["HR-009"], ["HR-009"], IDS) == (1.0, 1.0)
    assert citation_pr(["HR-009", "HR-001"], ["HR-009"], IDS) == (0.5, 1.0)
    assert citation_pr(["HR-999"], ["HR-009"], IDS) == (0.0, 0.0)       # not in corpus -> wrong
    assert citation_pr([], ["HR-009"], IDS) == (None, 0.0)              # empty citations
    assert citation_pr(["HR-999"], [], IDS) == (0.0, None)              # unanswerable + hallucinated citation
    assert citation_pr([], [], IDS) == (None, None)


def S(**kw):
    base = {"answer_match": True, "abstention": False, "citation_recall": 1.0, "judge": None, "judge_error": False}
    return base | kw


LOOKUP = {"type": "lookup", "answer_kind": "exact"}
FREE = {"type": "multi_hop", "answer_kind": "free_text"}
CONFLICT = {"type": "conflicting", "answer_kind": "exact"}
UNANS = {"type": "unanswerable", "answer_kind": "abstain"}


@pytest.mark.parametrize("task, scores, status, expected", [
    (LOOKUP, S(), "ok", True),
    (LOOKUP, S(answer_match=False), "ok", False),
    (LOOKUP, S(citation_recall=0.0), "ok", False),                  # right answer, no gold citation
    (LOOKUP, S(citation_recall=None), "ok", False),
    (LOOKUP, S(abstention=True), "ok", False),                      # abstained on answerable
    (CONFLICT, S(), "ok", True),
    (CONFLICT, S(answer_match=False), "ok", False),                 # used superseded value
    (UNANS, S(answer_match=False, abstention=True, citation_recall=None), "ok", True),
    (UNANS, S(answer_match=False, abstention=False, citation_recall=None), "ok", False),
    (FREE, S(answer_match=False, judge={"correct": True, "faithful": True}), "ok", True),
    (FREE, S(answer_match=False, judge={"correct": True, "faithful": False}), "ok", False),
    (FREE, S(answer_match=False, judge_error=True), "ok", None),    # judge error -> missing
    (FREE, S(answer_match=False, judge_error=True), "agent_error", False),
    (LOOKUP, S(), "agent_error", False),
    (LOOKUP, S(), "timeout", False),
    (LOOKUP, S(), "limit", False),
    (LOOKUP, S(), "format_error", False),
])
def test_task_success_table(task, scores, status, expected):
    assert task_success(task, scores, status) is expected
