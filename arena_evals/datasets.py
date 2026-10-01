"""Task JSONL schema, hashing, gate-mix + near-duplicate checks, LLM drafting, spot-check CLI."""
from __future__ import annotations

import asyncio
import json
import random
import re
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from arena_evals import corpus
from arena_evals.config import ROOT, sha256_files
from arena_evals.stats.agreement import wilson

TYPES = ("lookup", "multi_hop", "arithmetic", "conflicting", "unanswerable")
SHARES = {"lookup": 0.30, "multi_hop": 0.25, "arithmetic": 0.15, "conflicting": 0.15, "unanswerable": 0.15}
SPLIT_SIZES = {"gate": 300, "calibration": 120, "dev": 100}
PREFIX = {"gate": "gate", "calibration": "cal", "dev": "dev"}


class TaskRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    input: str
    reference: str | None
    gold_doc_ids: list[str]
    type: Literal["lookup", "multi_hop", "arithmetic", "conflicting", "unanswerable"]
    answer_kind: Literal["exact", "numeric", "free_text", "abstain"]
    tags: list[str]
    split: Literal["gate", "calibration", "dev"]

    @model_validator(mode="after")
    def _rules(self) -> "TaskRecord":
        if (self.type == "unanswerable") != (self.answer_kind == "abstain"):
            raise ValueError("type unanswerable <=> answer_kind abstain")
        if self.type == "unanswerable":
            if self.reference is not None or self.gold_doc_ids:
                raise ValueError("unanswerable tasks need reference null and gold_doc_ids []")
        else:
            if not self.reference or not self.gold_doc_ids:
                raise ValueError("answerable tasks need a reference and at least one gold doc")
        if self.type == "arithmetic" and self.answer_kind != "numeric":
            raise ValueError("arithmetic tasks are numeric")
        if self.type == "conflicting" and len(self.gold_doc_ids) != 1:
            raise ValueError("conflicting tasks cite exactly the currently effective doc")
        return self


def load(path: Path | str) -> list[TaskRecord]:
    records, seen = [], set()
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = TaskRecord.model_validate_json(line)
        except ValidationError as e:
            raise ValueError(f"{path}:{n}: {e}") from None
        if r.id in seen:
            raise ValueError(f"{path}:{n}: duplicate id {r.id}")
        seen.add(r.id)
        records.append(r)
    return records


def write_jsonl(path: Path | str, rows: list) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [r.model_dump_json() if isinstance(r, BaseModel) else json.dumps(r) for r in rows]
    path.write_text("".join(l + "\n" for l in lines), encoding="utf-8", newline="\n")
    return path


def dataset_hash(path: Path | str) -> str:
    return sha256_files(Path(path))


def mix_counts(split: str) -> dict[str, int]:
    return {t: round(SHARES[t] * SPLIT_SIZES[split]) for t in TYPES}


def check_mix(tasks: list[TaskRecord], split: str = "gate", tol: int = 1) -> list[str]:
    want = mix_counts(split)
    have = {t: sum(1 for r in tasks if r.type == t) for t in TYPES}
    return [f"{split}: {t} has {have[t]} tasks, want {want[t]} +/- {tol}" for t in TYPES if abs(have[t] - want[t]) > tol]


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", "", text.lower()).split())


def _shingles(text: str, n: int = 5) -> set[str]:
    t = normalize(text)
    return {t[i:i + n] for i in range(max(len(t) - n + 1, 1))}


def near_duplicates(gate: list[TaskRecord], others: list[TaskRecord],
                    threshold: float = 0.8) -> list[tuple[str, str, float]]:
    """(gate_id, other_id, jaccard) for every pair whose character 5-gram Jaccard >= threshold. O(n*m)."""
    other_sh = [(o.id, _shingles(o.input)) for o in others]
    out = []
    for g in gate:
        gs = _shingles(g.input)
        for oid, os_ in other_sh:
            j = len(gs & os_) / len(gs | os_)
            if j >= threshold:
                out.append((g.id, oid, j))
    return out


def check(root: Path = ROOT) -> int:
    """`datasets check`: schema, gold IDs, gate mix, cross-split IDs, near-duplicates. Exit code 0/1."""
    errors: list[str] = []
    docs = corpus.load(root / "corpus" / "docs")
    splits: dict[str, list[TaskRecord]] = {}
    for s in SPLIT_SIZES:
        p = root / "datasets" / f"{s}.jsonl"
        if not p.exists():
            errors.append(f"missing {p}")
            continue
        try:
            splits[s] = load(p)
        except ValueError as e:
            errors.append(str(e))
            continue
        errors += [f"{t.id}: split field is {t.split}, file is {s}" for t in splits[s] if t.split != s]
        errors += [f"{t.id}: unknown gold doc {g}" for t in splits[s] for g in t.gold_doc_ids if g not in docs]
    if "gate" in splits:
        errors += check_mix(splits["gate"], "gate")
    ids = [t.id for ts in splits.values() for t in ts]
    errors += [f"id used in more than one split: {i}" for i in sorted({i for i in ids if ids.count(i) > 1})]
    others = splits.get("dev", []) + splits.get("calibration", [])
    for g, o, j in near_duplicates(splits.get("gate", []), others):
        errors.append(f"near-duplicate: {g} ~ {o} (jaccard {j:.2f})")
    for e in errors:
        print(f"ERROR {e}")
    print("datasets check: " + ("FAIL" if errors else "OK") + f" ({sum(len(v) for v in splits.values())} tasks)")
    return 1 if errors else 0


# ---------------------------------------------------------------- drafting (LLM)
TYPE_GUIDE = {
    "lookup": "a single fact stated in one document. answer_kind is exact (short span such as a name or date), "
              "numeric (a quantity), or free_text (a sentence-length answer).",
    "multi_hop": "needs facts from two different documents combined, e.g. follow a 'Related documents' reference. "
                 "answer_kind is usually free_text; exact or numeric are allowed.",
    "arithmetic": "needs a calculation over numbers stated in the documents. answer_kind is numeric; reference is "
                  "the computed number with its unit, e.g. \"4475 USD\".",
    "conflicting": "asks for a fact that two documents state differently (one supersedes the other). reference is "
                   "the value from the currently effective document and gold_doc_ids is ONLY that document. "
                   "answer_kind is exact or numeric.",
    "unanswerable": "sounds plausible for this company but the documents do not answer it. answer_kind is abstain, "
                    "reference is null, gold_doc_ids is [].",
}
DRAFT_PROMPT = """You write evaluation questions for an assistant that answers from the Halcyon Robotics corpus below.
The corpus as-of date is {as_of}: a document is in force if its effective_date is not after that date.

Write {n} distinct {type} questions. A {type} question {guide}
Each question must be answerable (or clearly not answerable) from the corpus alone, phrased as an employee would ask it,
and different in wording and target fact from the other questions.

Return ONLY a JSON array. Each element: {{"input": str, "reference": str or null, "gold_doc_ids": [doc ids],
"answer_kind": "exact" | "numeric" | "free_text" | "abstain"}}

CORPUS:
{corpus}
"""


def _effective_ids(docs: dict[str, corpus.Doc], as_of: str) -> set[str]:
    superseded = {d.supersedes for d in docs.values() if d.supersedes and d.effective_date <= as_of}
    return {i for i, d in docs.items() if i not in superseded and d.effective_date <= as_of}


def _parse_drafts(text: str) -> list[dict]:
    try:
        items = json.loads(text[text.index("["): text.rindex("]") + 1])
    except ValueError:
        return []
    return [i for i in items if isinstance(i, dict)]


def draft(split: str, corpus_docs: dict[str, corpus.Doc], model, as_of: str = "2026-09-01",
          batch: int = 15, max_batches_per_type: int = 12, on_progress: Callable[[str], None] | None = None,
          checkpoint: Path | None = None) -> list[TaskRecord]:
    """Ask the drafting model for ceil(1.1 x mix) tasks per type; keep only schema-valid, corpus-consistent ones.
    `checkpoint` is rewritten after every batch and, if it already exists, resumed from, so an interrupted run
    does not lose paid calls. `on_progress(line)` is called after each batch, once the checkpoint is current."""
    return asyncio.run(_draft(split, corpus_docs, model, as_of, batch, max_batches_per_type, on_progress, checkpoint))


async def _draft(split, corpus_docs, model, as_of, batch, max_batches_per_type, on_progress=None,
                 checkpoint=None) -> list[TaskRecord]:
    from inspect_ai.model import GenerateConfig, get_model

    m = get_model(model) if isinstance(model, str) else model
    corpus_text = "\n\n".join(d.text for d in corpus_docs.values())
    effective = _effective_ids(corpus_docs, as_of)
    out: list[TaskRecord] = load(checkpoint) if checkpoint and Path(checkpoint).exists() else []
    if any(r.split != split for r in out):
        raise ValueError(f"checkpoint {checkpoint} holds drafts for another split; delete it to start over")
    seen: set[str] = {normalize(r.input) for r in out}
    for t in TYPES:
        want = -(-11 * mix_counts(split)[t] // 10)  # ceil(1.1 x mix), in integers to dodge float error
        got = sum(r.type == t for r in out)
        for _ in range(max_batches_per_type):
            if got >= want:
                break
            prompt = DRAFT_PROMPT.format(as_of=as_of, n=min(batch, want - got), type=t, guide=TYPE_GUIDE[t],
                                         corpus=corpus_text)
            resp = await m.generate(prompt, config=GenerateConfig(temperature=1.0))
            for item in _parse_drafts(resp.completion):
                if got >= want:
                    break
                gold = item.get("gold_doc_ids") or []
                if any(g not in corpus_docs for g in gold):
                    continue
                if t == "conflicting" and (len(gold) != 1 or gold[0] not in effective):
                    continue
                key = normalize(str(item.get("input", "")))
                if not key or key in seen:
                    continue
                tags = [t] + sorted({corpus_docs[g].kind for g in gold})
                try:
                    rec = TaskRecord(id=f"{PREFIX[split]}-{len(out) + 1:04d}", input=item["input"],
                                     reference=item.get("reference"), gold_doc_ids=gold, type=t,
                                     answer_kind=item.get("answer_kind"), tags=tags, split=split)
                except (ValidationError, KeyError):
                    continue
                seen.add(key)
                out.append(rec)
                got += 1
            if checkpoint:
                write_jsonl(checkpoint, out)
            if on_progress:
                on_progress(f"{t} {got}/{want} ({len(out)} drafts saved)")
        if got < want:
            raise RuntimeError(f"drafting produced only {got}/{want} valid {t} tasks; re-run or raise max_batches")
    return out


# ---------------------------------------------------------------- spot-check (HUMAN)
def spotcheck(split: str, frac: float = 0.2, seed: int = 20260930, ask: Callable[[str], str] = input,
              root: Path = ROOT) -> Path:
    """Review a seeded sample of drafts; write datasets/spotcheck.jsonl and the final datasets/<split>.jsonl."""
    drafts = load(root / "datasets" / "drafts" / f"{split}.jsonl")
    docs = corpus.load(root / "corpus" / "docs")
    sample = random.Random(seed).sample(drafts, round(frac * len(drafts)))
    verdicts, fixed = [], {}
    for i, t in enumerate(sample, 1):
        print(f"\n[{i}/{len(sample)}] {t.id} ({t.type}, {t.answer_kind})\nQ: {t.input}\nREF: {t.reference}\nGOLD: {t.gold_doc_ids}")
        for g in t.gold_doc_ids:
            print(f"--- {g}\n{docs[g].body}")
        v = ""
        while v not in ("o", "f", "d"):
            v = ask("[o]k / [f]ix / [d]rop > ").strip().lower()[:1]
        note = ""
        if v == "f":
            ref = ask("correct reference (enter = keep): ").strip()
            gold = ask("correct gold doc ids, comma separated (enter = keep): ").strip()
            note = ask("note: ").strip()
            upd = {}
            if ref:
                upd["reference"] = ref
            if gold:
                upd["gold_doc_ids"] = [g.strip() for g in gold.split(",")]
            fixed[t.id] = TaskRecord.model_validate(t.model_dump() | upd)
        verdicts.append({"split": split, "task_id": t.id, "verdict": {"o": "ok", "f": "fixed", "d": "dropped"}[v],
                         "note": note})
    sc_path = root / "datasets" / "spotcheck.jsonl"
    kept = [json.loads(l) for l in sc_path.read_text(encoding="utf-8").splitlines()] if sc_path.exists() else []
    write_jsonl(sc_path, [r for r in kept if r["split"] != split] + verdicts)
    dropped = {v["task_id"] for v in verdicts if v["verdict"] == "dropped"}
    final: list[TaskRecord] = []
    for t_ in TYPES:
        pool = [fixed.get(r.id, r) for r in drafts if r.type == t_ and r.id not in dropped]
        need = mix_counts(split)[t_]
        if len(pool) < need:
            raise RuntimeError(f"{split}: only {len(pool)} {t_} tasks left after drops, need {need}; draft more")
        final += pool[:need]
    return write_jsonl(root / "datasets" / f"{split}.jsonl", final)


def label_noise_floor(split: str, root: Path = ROOT) -> dict | None:
    """(fixed + dropped) / spot-checked sample, with a 95% Wilson interval. None if never spot-checked."""
    p = root / "datasets" / "spotcheck.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r["split"] == split]
    if not rows:
        return None
    ci = wilson(sum(r["verdict"] != "ok" for r in rows), len(rows))
    return {"point": ci.point, "ci95": [ci.lo, ci.hi], "n": len(rows)}
