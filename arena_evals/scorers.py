"""Deterministic scorers, the per-task success rule, and (M4) the LLM judge. Changing this file changes scorer_hash."""
from __future__ import annotations

import re
from pathlib import Path

from inspect_ai.model import (CachePolicy, ChatMessageAssistant, ChatMessageSystem, ChatMessageTool,
                              ChatMessageUser, GenerateConfig, get_model)
from inspect_ai.scorer import Score, Target, scorer
from inspect_ai.solver import TaskState
from pydantic import BaseModel, ConfigDict, ValidationError

from arena_evals import agents, tracing
from arena_evals.agents import _index, parse_final, usage_cost
from arena_evals.config import ROOT

RUBRIC_DIR = ROOT / "prompts" / "judge"

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


# ---------------------------------------------------------------- LLM judge (M4)
class JudgeVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    correct: bool
    faithful: bool
    reason: str


def judge_messages(rubric_dir: Path, task_input: str, reference: str | None, answer: str,
                   cited_docs: dict[str, str]) -> list:
    system = (rubric_dir / "system.md").read_text(encoding="utf-8")
    rubric = (rubric_dir / "rubric.md").read_text(encoding="utf-8")
    docs = "\n\n".join(f"<doc id=\"{i}\">\n{t}\n</doc>" for i, t in cited_docs.items()) or "(no valid cited documents)"
    user = (f"{rubric}\n\n## Question\n{task_input}\n\n## Reference answer\n{reference}\n\n"
            f"## Assistant answer\n{answer}\n\n## Documents the assistant cited\n{docs}\n")
    return [ChatMessageSystem(content=system), ChatMessageUser(content=user)]


def parse_verdict(text: str) -> JudgeVerdict:
    """Extract the outermost {...} and validate it. Raises ValueError (ValidationError is a ValueError)."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in judge output")
    return JudgeVerdict.model_validate_json(text[start:end + 1])


async def judge_answer(model, rubric_dir: Path, task_input: str, reference: str | None, answer: str,
                       cited_docs: dict[str, str], cache: bool = True, prices: dict | None = None
                       ) -> tuple[JudgeVerdict | None, float]:
    """Two attempts: on invalid output, retry once with the validation error appended. (None, cost) = judge error."""
    messages = judge_messages(rubric_dir, task_input, reference, answer, cited_docs)
    cost = 0.0
    for _ in range(2):
        agents.METER.check()
        out = await model.generate(messages, cache=CachePolicy(expiry=None) if cache else False)
        if out.usage:
            cost += usage_cost(out.usage, str(model), prices or agents.METER.prices)
        try:
            return parse_verdict(out.completion), cost
        except ValueError as e:
            messages = messages + [ChatMessageAssistant(content=out.completion),
                                   ChatMessageUser(content=f"Your reply failed validation: {e}\n"
                                                           "Reply with only the JSON object.")]
    return None, cost


@scorer(metrics=[])
def judge(model, rubric_dir: str = str(RUBRIC_DIR), temperature: float = 0.0, seed: int = 0, cache: bool = True):
    """Pointwise judge, only for free_text answers that did not abstain. Value keys: correct, faithful, error, cost_usd."""

    async def score(state: TaskState, target: Target) -> Score:
        task, f = state.metadata, parse_final(state.output.completion)
        if task["answer_kind"] != "free_text" or f is None or f.abstain:
            return Score(value={"judged": False, "correct": None, "faithful": None, "error": False, "cost_usd": 0.0})
        m = model if not isinstance(model, str) else get_model(
            model, config=GenerateConfig(temperature=temperature, seed=seed))
        docs = {c: _index().docs[c].text for c in f.citations if c in _index().docs}
        with tracing.sample_span("arena.judge", task_id=str(state.sample_id), repeat=state.epoch - 1, model=str(m)):
            verdict, cost = await judge_answer(m, Path(rubric_dir), task["input"], task["reference"], f.answer, docs, cache)
        if verdict is None:
            return Score(value={"judged": True, "correct": None, "faithful": None, "error": True, "cost_usd": cost})
        return Score(value={"judged": True, "correct": verdict.correct, "faithful": verdict.faithful, "error": False,
                            "cost_usd": cost}, explanation=verdict.reason)

    return score
