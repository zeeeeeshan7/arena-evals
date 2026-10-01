import shutil
from pathlib import Path

import pytest

from arena_evals import datasets
from arena_evals.config import ROOT
from arena_evals.datasets import TaskRecord


def rec(i: int, type_: str = "lookup", split: str = "gate", text: str | None = None) -> TaskRecord:
    if type_ == "unanswerable":
        return TaskRecord(id=f"{split}-{i:04d}", input=text or f"unanswerable question number {i}", reference=None,
                          gold_doc_ids=[], type=type_, answer_kind="abstain", tags=[type_], split=split)
    kind = "numeric" if type_ == "arithmetic" else "exact"
    return TaskRecord(id=f"{split}-{i:04d}", input=text or f"question {type_} number {i} about item {i * 7}",
                      reference="42", gold_doc_ids=["HR-001"], type=type_, answer_kind=kind, tags=[type_], split=split)


def gate_set(counts: dict[str, int]) -> list[TaskRecord]:
    out, i = [], 0
    for t, n in counts.items():
        for _ in range(n):
            i += 1
            out.append(rec(i, t))
    return out


def test_schema_rejects_unknown_and_missing_fields():
    good = rec(1).model_dump()
    TaskRecord.model_validate(good)
    with pytest.raises(ValueError):
        TaskRecord.model_validate(good | {"extra": 1})
    with pytest.raises(ValueError):
        TaskRecord.model_validate({k: v for k, v in good.items() if k != "gold_doc_ids"})


def test_schema_type_rules():
    base = rec(1).model_dump()
    with pytest.raises(ValueError):  # unanswerable must abstain with null reference and no gold
        TaskRecord.model_validate(base | {"type": "unanswerable"})
    with pytest.raises(ValueError):
        TaskRecord.model_validate(base | {"type": "arithmetic", "answer_kind": "exact"})
    with pytest.raises(ValueError):
        TaskRecord.model_validate(base | {"type": "conflicting", "gold_doc_ids": ["HR-002", "HR-009"]})
    TaskRecord.model_validate(rec(2, "unanswerable").model_dump())


def test_load_rejects_duplicate_ids_and_reports_line(tmp_path: Path):
    p = datasets.write_jsonl(tmp_path / "x.jsonl", [rec(1), rec(1)])
    with pytest.raises(ValueError, match="x.jsonl:2: duplicate id"):
        datasets.load(p)


def test_dataset_hash_changes_with_content(tmp_path: Path):
    p = datasets.write_jsonl(tmp_path / "a.jsonl", [rec(1)])
    h1 = datasets.dataset_hash(p)
    assert h1.startswith("sha256:") and h1 == datasets.dataset_hash(p)
    datasets.write_jsonl(p, [rec(2)])
    assert datasets.dataset_hash(p) != h1


def test_gate_mix_within_one_task():
    assert datasets.mix_counts("gate") == {"lookup": 90, "multi_hop": 75, "arithmetic": 45,
                                           "conflicting": 45, "unanswerable": 45}
    assert datasets.check_mix(gate_set(datasets.mix_counts("gate"))) == []
    assert datasets.check_mix(gate_set({"lookup": 91, "multi_hop": 74, "arithmetic": 45,
                                        "conflicting": 45, "unanswerable": 45})) == []
    errs = datasets.check_mix(gate_set({"lookup": 92, "multi_hop": 73, "arithmetic": 45,
                                        "conflicting": 45, "unanswerable": 45}))
    assert len(errs) == 2


def test_near_duplicate_planted_and_clean():
    g = [rec(1, text="What is the monthly price of the Pro tier for 25 seats?")]
    dup = [rec(9, split="dev", text="what is the monthly price of the pro tier for 25 seats")]
    far = [rec(8, split="dev", text="Who leads the Customer Support organization?")]
    hits = datasets.near_duplicates(g, dup + far)
    assert [(a, b) for a, b, _ in hits] == [("gate-0001", "dev-0009")]
    assert hits[0][2] == pytest.approx(1.0)
    assert datasets.near_duplicates(g, far) == []


def _fake_repo(tmp_path: Path) -> Path:
    shutil.copytree(ROOT / "corpus", tmp_path / "corpus")
    (tmp_path / "datasets").mkdir()
    return tmp_path


def test_check_fails_on_planted_near_duplicate(tmp_path: Path, capsys):
    root = _fake_repo(tmp_path)
    gate = gate_set(datasets.mix_counts("gate"))
    datasets.write_jsonl(root / "datasets" / "gate.jsonl", gate)
    datasets.write_jsonl(root / "datasets" / "calibration.jsonl", [rec(1, split="calibration", text="unrelated wording entirely here")])
    datasets.write_jsonl(root / "datasets" / "dev.jsonl", [rec(1, split="dev", text=gate[0].input + "?")])
    assert datasets.check(root) == 1
    assert "near-duplicate: gate-0001 ~ dev-0001" in capsys.readouterr().out


def test_split_sizes_can_be_overridden_for_a_cheaper_demo_scale():
    assert datasets.parse_split_sizes(None) == {"gate": 300, "calibration": 120, "dev": 100}
    assert datasets.parse_split_sizes("gate=150") == {"gate": 150, "calibration": 120, "dev": 100}
    assert datasets.parse_split_sizes("gate=150, calibration=60") == {"gate": 150, "calibration": 60, "dev": 100}
    with pytest.raises(ValueError, match="unknown split"):
        datasets.parse_split_sizes("test=5")
    with pytest.raises(ValueError, match="positive integer"):
        datasets.parse_split_sizes("gate=abc")
