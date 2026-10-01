import json
import shutil
from collections import Counter
from pathlib import Path

import pytest
from inspect_ai.model import ModelOutput, get_model

from arena_evals import calibration, config, datasets
from arena_evals.config import ROOT
from arena_evals.datasets import TaskRecord

M = "mockllm/model"


def test_stratified_sample_proportional_with_minimum():
    items = [{"id": i, "cell": "big"} for i in range(170)] + [{"id": 1000 + i, "cell": "small"} for i in range(8)] \
        + [{"id": 2000 + i, "cell": "tiny"} for i in range(3)]
    out = calibration.stratified_sample(items, lambda e: (e["cell"],), n=100, min_per_cell=5, seed=1)
    counts = Counter(e["cell"] for e in out)
    assert sum(counts.values()) == 100
    assert counts["tiny"] == 3 and counts["small"] == 5 and counts["big"] == 92
    assert out == calibration.stratified_sample(items, lambda e: (e["cell"],), 100, 5, 1)  # seeded
    assert len({e["id"] for e in out}) == 100


def test_report_known_table():
    key = [{"example_id": f"e{i}", "judge_pass": p} for i, p in enumerate([1] * 47 + [0] * 41 + [1] * 5 + [0] * 7)]
    labels = [{"example_id": f"e{i}", "human_label": h}
              for i, h in enumerate(["pass"] * 47 + ["fail"] * 41 + ["fail"] * 5 + ["pass"] * 7)]
    rep = calibration.report(labels, key, n_resamples=2000, seed=0)
    assert rep["n"] == 100 and rep["confusion"] == [[41, 5], [7, 47]]
    assert rep["agreement"]["raw"]["point"] == pytest.approx(0.88)
    assert set(rep["agreement"]) == {"raw", "kappa", "ac1", "pabak", "precision", "recall"}


def test_label_cli_is_blind_resumable_and_appends(tmp_path: Path, capsys):
    to_label = datasets.write_jsonl(tmp_path / "to_label.jsonl", [
        {"example_id": f"cal-000{i}-baseline-r0", "task_id": f"cal-000{i}", "variant": "baseline", "repeat": 0,
         "type": "lookup", "input": "q", "reference": "r", "answer": "a", "citations": ["HR-001"]} for i in (1, 2, 3)])
    answers = iter(["p", "", "f", "wrong date", "q"])
    labels = calibration.label_cli(to_label, labeler="zeesh", ask=lambda _: next(answers))
    assert "judge" not in capsys.readouterr().out.lower()
    rows = calibration.read_jsonl(labels)
    assert [(r["example_id"], r["human_label"]) for r in rows] == [("cal-0001-baseline-r0", "pass"),
                                                                    ("cal-0002-baseline-r0", "fail")]
    answers = iter(["p", ""])
    calibration.label_cli(to_label, labeler="zeesh", ask=lambda _: next(answers))  # resumes at #3
    assert len(calibration.read_jsonl(labels)) == 3


def _tmp_root(tmp_path: Path) -> Path:
    for d in ("configs", "prompts", "corpus"):
        shutil.copytree(ROOT / d, tmp_path / d)
    (tmp_path / "arena_evals").mkdir()
    shutil.copy(ROOT / "arena_evals" / "scorers.py", tmp_path / "arena_evals" / "scorers.py")
    return tmp_path


def test_export_runs_four_variants_and_writes_blind_file(tmp_path: Path):
    root = _tmp_root(tmp_path)
    tasks = [TaskRecord(id=f"cal-{i:04d}", input=f"Explain incident number {i}", reference="ref",
                        gold_doc_ids=["INC-001"], type="lookup" if i % 2 else "multi_hop", answer_kind="free_text",
                        tags=["lookup"], split="calibration") for i in range(1, 16)]
    tasks.append(TaskRecord(id="cal-0099", input="How many PTO days?", reference="20 days", gold_doc_ids=["HR-001"],
                            type="lookup", answer_kind="exact", tags=["lookup"], split="calibration"))
    datasets.write_jsonl(root / "datasets" / "calibration.jsonl", tasks)
    cfg = config.load(root)
    cfg.cert["n_labels"] = 30

    def agent(messages, tools, tool_choice, c):
        q = next(m.text for m in messages if m.role == "user")
        good = "GOOD" if int(q.split()[-1]) % 3 else "BAD"
        return ModelOutput.from_content(M, f'x\nFINAL: {{"answer": "{good} answer", "citations": ["INC-001"], "abstain": false}}')

    def judge(messages, tools, tool_choice, c):
        ok = "GOOD answer" in messages[-1].text
        return ModelOutput.from_content(M, json.dumps({"correct": ok, "faithful": True, "reason": "r"}))

    out = calibration.export(cfg, model=get_model(M, custom_outputs=agent, memoize=False),
                             judge_model=get_model(M, custom_outputs=judge, memoize=False))
    blind = calibration.read_jsonl(out)
    key = calibration.read_jsonl(root / "calibration" / "key.jsonl")
    assert len(blind) == 30 and len(key) == 30
    assert all("judge_pass" not in e and "judge" not in e for e in blind)
    assert {e["variant"] for e in blind} <= set(calibration.VARIANTS)
    assert all(not e["task_id"] == "cal-0099" for e in blind)  # only free_text tasks
    cells = Counter((e["type"], k["judge_pass"]) for e, k in zip(blind, key))
    assert all(v >= 5 for v in cells.values())
