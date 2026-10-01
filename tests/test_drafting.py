import json
from pathlib import Path

import pytest
from inspect_ai.model import ModelOutput, get_model

from arena_evals import corpus, datasets
from test_datasets import _fake_repo, rec


def test_draft_with_scripted_model_keeps_only_valid_tasks(monkeypatch):
    docs = corpus.load()

    def replies():
        n = 0
        while True:
            n += 1
            items = [
                {"input": f"Question variant {n}-{j} about PTO days", "reference": "20 days",
                 "gold_doc_ids": ["HR-001"], "answer_kind": "exact"} for j in range(15)
            ] + [{"input": "bad gold", "reference": "x", "gold_doc_ids": ["NOPE-1"], "answer_kind": "exact"}]
            yield ModelOutput.from_content("mockllm/model", json.dumps(items))

    monkeypatch.setattr(datasets, "TYPES", ("lookup",))  # script one type only
    model = get_model("mockllm/model", custom_outputs=replies(), memoize=False)
    out = datasets.draft("dev", docs, model)
    assert len(out) == 33  # ceil(1.1 * 30)
    assert all(r.gold_doc_ids == ["HR-001"] and r.tags == ["lookup", "hr"] for r in out)
    assert out[0].id == "dev-0001"


def test_spotcheck_writes_final_split_and_noise_floor(tmp_path: Path):
    root = _fake_repo(tmp_path)
    drafts = []
    i = 0
    for t, n in datasets.mix_counts("dev").items():
        for _ in range(n + 2):
            i += 1
            drafts.append(rec(i, t, split="dev", text=f"draft question {i} of type {t}"))
    datasets.write_jsonl(root / "datasets" / "drafts" / "dev.jsonl", drafts)
    answers = iter(["d", "f", "43", "", "reference was wrong"] + ["o"] * 1000)
    datasets.spotcheck("dev", frac=0.2, seed=1, ask=lambda _: next(answers), root=root)
    final = datasets.load(root / "datasets" / "dev.jsonl")
    assert len(final) == 100 and datasets.check_mix(final, "dev") == []
    sc = [json.loads(l) for l in (root / "datasets" / "spotcheck.jsonl").read_text().splitlines()]
    assert [r["verdict"] for r in sc[:2]] == ["dropped", "fixed"]
    assert sc[1]["task_id"] not in {r.id for r in final} or \
        next(r for r in final if r.id == sc[1]["task_id"]).reference == "43"
    assert sc[0]["task_id"] not in {r.id for r in final}
    floor = datasets.label_noise_floor("dev", root)
    assert floor["n"] == 22 and floor["point"] == pytest.approx(2 / 22)
    assert floor["ci95"][0] < floor["point"] < floor["ci95"][1]


def _scripted(calls: list, tag: str = "a"):
    def replies():
        while True:
            calls.append(1)
            n = len(calls)
            items = [{"input": f"Question variant {tag}{n}-{j} about PTO days", "reference": "20 days",
                      "gold_doc_ids": ["HR-001"], "answer_kind": "exact"} for j in range(15)]
            yield ModelOutput.from_content("mockllm/model", json.dumps(items))
    return get_model("mockllm/model", custom_outputs=replies(), memoize=False)


def test_draft_reports_progress_and_checkpoints_after_every_batch(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(datasets, "TYPES", ("lookup",))
    lines, saved = [], []
    ckpt = tmp_path / "dev.partial.jsonl"

    def progress(msg):
        lines.append(msg)
        saved.append(len(datasets.load(ckpt)))      # the checkpoint on disk is already current when we report

    out = datasets.draft("dev", corpus.load(), _scripted([]), on_progress=progress, checkpoint=ckpt)
    assert len(out) == 33 and saved == [15, 30, 33]
    assert lines[0].startswith("lookup 15/33") and lines[-1].startswith("lookup 33/33")


def test_draft_resumes_from_checkpoint_without_repaying(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(datasets, "TYPES", ("lookup",))
    ckpt = tmp_path / "dev.partial.jsonl"
    first = []
    with pytest.raises(RuntimeError, match="interrupted"):          # simulate Ctrl+C after batch 2
        datasets.draft("dev", corpus.load(), _scripted(first), checkpoint=ckpt,
                       on_progress=lambda m: (_ for _ in ()).throw(RuntimeError("interrupted"))
                       if m.startswith("lookup 30") else None)
    assert len(datasets.load(ckpt)) == 30
    calls = []
    out = datasets.draft("dev", corpus.load(), _scripted(calls, "b"), checkpoint=ckpt)
    assert len(calls) == 1 and len(out) == 33                       # one more batch, not three
    assert [r.id for r in out][:2] == ["dev-0001", "dev-0002"] and out[-1].id == "dev-0033"


def test_draft_refuses_a_checkpoint_from_another_split(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(datasets, "TYPES", ("lookup",))
    ckpt = tmp_path / "x.partial.jsonl"
    datasets.write_jsonl(ckpt, [rec(1, "lookup", split="calibration", text="other split question")])
    with pytest.raises(ValueError, match="another split"):
        datasets.draft("dev", corpus.load(), _scripted([]), checkpoint=ckpt)
