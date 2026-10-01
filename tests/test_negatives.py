"""Calibration negatives: known-bad copies of real answers, failing the rubric by construction."""
import json
import random
import shutil
from pathlib import Path

import pytest
from inspect_ai.model import ModelOutput, get_model

from arena_evals import bias, calibration, config, corpus, datasets
from arena_evals.config import ROOT

DOCS = corpus.load()
REAL = {"example_id": "cal-0068-baseline-r0", "task_id": "cal-0068", "variant": "baseline", "repeat": 0,
        "type": "multi_hop", "input": "What is the annual learning budget per employee, and how many times per year "
        "are performance reviews held?", "reference": "1,500 USD per year; twice a year, in March and September.",
        "answer": "The annual learning budget is 1,500 USD per employee; performance reviews are held twice a year "
        "(in March and September).", "citations": ["HR-006", "HR-008"]}


def test_wrong_number_changes_a_required_number():
    n = calibration.corrupt("wrong_number", REAL, DOCS, random.Random(0))
    assert n["answer"] != REAL["answer"] and "1,500" not in n["answer"]
    assert n["citations"] == REAL["citations"] and n["synthetic"] == "wrong_number"
    assert n["example_id"] == "cal-0068-baseline-r0~wrong_number"


def test_dropped_fact_removes_the_second_required_fact():
    n = calibration.corrupt("dropped_fact", REAL, DOCS, random.Random(0))
    assert "1,500" in n["answer"] and "twice" not in n["answer"]
    assert calibration.corrupt("dropped_fact", REAL | {"reference": "one single fact"}, DOCS, random.Random(0)) is None


def test_unsupported_claim_adds_a_sentence_from_an_uncited_document():
    n = calibration.corrupt("unsupported_claim", REAL, DOCS, random.Random(0))
    extra = n["answer"][len(REAL["answer"]):].strip()
    assert n["answer"].startswith(REAL["answer"]) and extra
    cited_text = " ".join(DOCS[c].text for c in REAL["citations"])
    assert extra not in cited_text and any(extra in d.text for i, d in DOCS.items() if i not in REAL["citations"])


def test_wrong_citation_cites_a_document_that_does_not_hold_the_facts():
    n = calibration.corrupt("wrong_citation", REAL, DOCS, random.Random(0))
    assert n["answer"] == REAL["answer"] and len(n["citations"]) == 1 and n["citations"][0] not in REAL["citations"]
    held = set(bias.facts_signature(DOCS[n["citations"][0]].body))
    assert not held & set(bias.facts_signature(REAL["answer"]))


def test_corrupt_is_deterministic_for_a_seed():
    a = calibration.corrupt("unsupported_claim", REAL, DOCS, random.Random(7))
    b = calibration.corrupt("unsupported_claim", REAL, DOCS, random.Random(7))
    assert a == b


@pytest.fixture
def repo(tmp_path):
    for d in ("configs", "prompts", "corpus"):
        shutil.copytree(ROOT / d, tmp_path / d, ignore=shutil.ignore_patterns("__pycache__"))
    real = [REAL | {"example_id": f"cal-00{i:02d}-baseline-r0", "task_id": f"cal-00{i:02d}"} for i in range(12)]
    datasets.write_jsonl(tmp_path / "calibration" / "to_label.jsonl", real)
    datasets.write_jsonl(tmp_path / "calibration" / "key.jsonl",
                         [{"example_id": e["example_id"], "judge_pass": True} for e in real])
    datasets.write_jsonl(tmp_path / "calibration" / "labels.jsonl", [
        {"example_id": e["example_id"], "task_id": e["task_id"], "variant": "baseline", "human_label": "pass",
         "labeler": "zeesh", "note": ""} for e in real[:2]])
    return config.load(tmp_path)


def scripted_judge(verdicts):
    def gen():
        for v in verdicts:
            yield ModelOutput.from_content("mockllm/model", json.dumps(v))
        while True:
            yield ModelOutput.from_content("mockllm/model",
                                           json.dumps({"correct": False, "faithful": False, "reason": "bad"}))
    return get_model("mockllm/model", custom_outputs=gen(), memoize=False)


def test_augment_appends_examples_keys_and_construction_labels_and_is_idempotent(repo):
    p = calibration.augment(repo, judge_model=scripted_judge([]), per_kind=3, seed=1)
    rows = calibration.read_jsonl(p)
    new = [r for r in rows if "synthetic" in r]
    assert len(rows) == 12 + 12 and len(new) == 12 and {r["synthetic"] for r in new} == set(calibration.KINDS)
    key = {k["example_id"]: k for k in calibration.read_jsonl(repo.root / "calibration" / "key.jsonl")}
    assert all(key[r["example_id"]]["judge_pass"] is False for r in new)      # the scripted judge rejects them
    labels = calibration.read_jsonl(repo.root / "calibration" / "labels.jsonl")
    cons = [l for l in labels if l["labeler"] == "construction"]
    assert len(cons) == 12 and all(l["human_label"] == "fail" for l in cons)
    assert {l["note"] for l in cons} == set(calibration.KINDS)
    assert len([l for l in labels if l["labeler"] == "zeesh"]) == 2           # existing human labels untouched
    calibration.augment(repo, judge_model=scripted_judge([]), per_kind=3, seed=1)   # second run adds nothing
    assert len(calibration.read_jsonl(p)) == 24
