"""Deterministic scorers, the per-task success rule, and (M4) the LLM judge. Changing this file changes scorer_hash."""
from __future__ import annotations

import re

from inspect_ai.model import ChatMessageTool
from inspect_ai.scorer import Score, Target, scorer
from inspect_ai.solver import TaskState

from arena_evals.agents import _index, parse_final

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


# ---------------------------------------------------------------- pure helpers
def normalize_answer(s: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", "", s.casefold()).split())


def first_number(s: str) -> float | None:
    m = _NUM.search(s.replace("$", "").replace("€", "").replace("£", ""))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def match_answer(answer: str, reference: str, answer_kind: str) -> bool:
    if answer_kind == "exact":
        return normalize_answer(answer) == normalize_answer(reference)
    if answer_kind == "numeric":
        a, r = first_number(answer), first_number(reference)
        return a is not None and r is not None and abs(a - r) <= max(0.005 * abs(r), 0.01)
    return False


def citation_pr(cited: list[str], gold: list[str], corpus_ids: set[str]) -> tuple[float | None, float | None]:
    """precision = |cited & gold| / |cited| (None if nothing cited); recall = |cited & gold| / |gold| (None if no gold).
    Cited IDs that are not in the corpus stay in the denominator, so they count as wrong."""
    cited_set, gold_set = set(cited), set(gold)
    hit = len(cited_set & gold_set & corpus_ids)
    precision = hit / len(cited_set) if cited_set else None
    recall = hit / len(gold_set) if gold_set else None
    return precision, recall


def fetched_doc_ids(messages) -> set[str]:
    return {str(tc.arguments.get("id", "")).strip() for m in messages if m.role == "assistant" and m.tool_calls
            for tc in m.tool_calls if tc.function == "get_doc"}


def tool_call_count(messages) -> int:
    return sum(1 for m in messages if isinstance(m, ChatMessageTool))


def task_success(task: dict, sample_scores: dict, status: str) -> bool | None:
    """Binary success for one repeat (spec 1.5). None = judge error or unjudged free text (treated as missing)."""
    if status != "ok":
        return False
    if task["type"] == "unanswerable":
        return bool(sample_scores["abstention"])
    if sample_scores["abstention"]:
        return False
    if task["answer_kind"] == "free_text":
        j = sample_scores.get("judge")
        if sample_scores.get("judge_error") or j is None:
            return None
        correct = bool(j["correct"] and j["faithful"])
    else:
        correct = bool(sample_scores["answer_match"])
    return correct and (sample_scores["citation_recall"] or 0) > 0


# ---------------------------------------------------------------- Inspect scorers
@scorer(metrics=[])
def answer_match():
    async def score(state: TaskState, target: Target) -> Score:
        task, f = state.metadata, parse_final(state.output.completion)
        ok = bool(f and task["reference"] and match_answer(f.answer, task["reference"], task["answer_kind"]))
        return Score(value=ok, answer=f.answer if f else None)

    return score


@scorer(metrics=[])
def citation():
    async def score(state: TaskState, target: Target) -> Score:
        f = parse_final(state.output.completion)
        p, r = citation_pr(f.citations if f else [], state.metadata["gold_doc_ids"], set(_index().docs))
        return Score(value={"precision": p, "recall": r})

    return score


@scorer(metrics=[])
def gold_retrieval():
    async def score(state: TaskState, target: Target) -> Score:
        return Score(value=bool(fetched_doc_ids(state.messages) & set(state.metadata["gold_doc_ids"])))

    return score


@scorer(metrics=[])
def abstention():
    async def score(state: TaskState, target: Target) -> Score:
        f = parse_final(state.output.completion)
        return Score(value=bool(f and f.abstain))

    return score
