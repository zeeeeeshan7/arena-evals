"""Task JSONL schema, hashing, gate-mix + near-duplicate checks, LLM drafting, spot-check CLI."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from arena_evals import corpus
from arena_evals.config import ROOT, sha256_files

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
