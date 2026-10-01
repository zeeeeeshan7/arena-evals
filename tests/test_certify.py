import json
import shutil

import pytest
from inspect_ai.model import ModelOutput, get_model

from arena_evals import ci, config, datasets
from arena_evals.config import ROOT

M = "mockllm/model"


def _repo(tmp_path):
    for d in ("configs", "prompts", "corpus"):
        shutil.copytree(ROOT / d, tmp_path / d)
    return tmp_path


def _examples(n=40):
    ex, labels = [], []
    for i in range(n):
        v = ("baseline", "concise", "verbose", "regressed")[i % 4]
        good = i % 5 != 0
        ex.append({"example_id": f"cal-{i:04d}-{v}-r0", "task_id": f"cal-{i:04d}", "variant": v, "repeat": 0,
                   "type": "lookup", "input": f"q{i}", "reference": "ref",
                   "answer": ("GOOD " if good else "BAD ") + "the outage lasted 47 minutes per INC-001." * (1 + i % 3),
                   "citations": ["INC-001"]})
        labels.append({"example_id": ex[-1]["example_id"], "task_id": ex[-1]["task_id"], "variant": v,
                       "human_label": "pass" if good else "fail", "labeler": "t", "note": ""})
    return ex, labels


def _judge(mode: str):
    """agree: passes GOOD answers (matches the human). contrarian: the opposite. length: also passes long answers."""
    def reply(messages, tools, tool_choice, c):
        ans = messages[-1].text.split("## Assistant answer\n", 1)[1].split("\n\n## Documents", 1)[0]
        good = ans.startswith("GOOD")
        ok = {"agree": good, "contrarian": not good, "length": good or len(ans) > 150}[mode]
        return ModelOutput.from_content(M, json.dumps({"correct": ok, "faithful": True, "reason": "r"}))
    return get_model(M, custom_outputs=reply, memoize=False)


@pytest.fixture
def cfg(tmp_path):
    root = _repo(tmp_path)
    ex, labels = _examples()
    datasets.write_jsonl(root / "calibration" / "to_label.jsonl", ex)
    datasets.write_jsonl(root / "calibration" / "labels.jsonl", labels)
    c = config.load(root)
    c.eval["n_resamples"], c.eval["cache"] = 1000, False
    return c


def test_certify_agreeing_judge_writes_valid_cert_then_goes_stale(cfg):
    assert ci.certify(cfg, perturber_model=_perturber(), judge_model=_judge("agree")) == 0
    cert = json.loads(ci.cert_path(cfg).read_text())
    assert cert["certified"] is True and cert["n_labels"] == 40
    assert cert["rubric_hash"] == config.rubric_hash(cfg.rubric_dir)
    assert cert["agreement"]["kappa"]["point"] == pytest.approx(1.0)
    assert cert["agreement"]["confusion"] == [[8, 0], [0, 32]]
    assert ci.cert_status(cfg) == (True, "certified")
    (cfg.root / "calibration" / "labels.jsonl").write_text("")
    assert ci.cert_status(cfg) == (False, "stale cert: labels_hash changed")


def test_certify_rejects_disagreeing_judge(cfg):
    assert ci.certify(cfg, perturber_model=_perturber(), judge_model=_judge("contrarian")) == 1
    assert json.loads(ci.cert_path(cfg).read_text())["certified"] is False
    assert ci.cert_status(cfg) == (False, "judge failed certification")


def test_cert_status_without_cert(cfg):
    assert ci.cert_status(cfg)[0] is False


def _perturber():
    """Compression stand-in: keeps the first sentence only (drops repeated facts, so most are discarded)."""
    def reply(messages, tools, tool_choice, c):
        ans = messages[-1].text.split("ANSWER:\n", 1)[1]
        return ModelOutput.from_content(M, ans.split(".")[0] + ".")
    return get_model(M, custom_outputs=reply, memoize=False)


def test_certify_fails_a_length_biased_judge_on_the_verbosity_slope(cfg):
    assert ci.certify(cfg, perturber_model=_perturber(), judge_model=_judge("length")) == 1
    cert = json.loads(ci.cert_path(cfg).read_text())
    assert cert["certified"] is False
    assert cert["agreement"]["kappa"]["point"] == pytest.approx(1.0)       # agreement alone would have passed
    assert cert["bias"]["verbosity_slope_pts"]["pass"] is False
    assert cert["bias"]["verbosity_slope_pts"]["ci95"][1] > 2.0
    assert set(cert["bias"]) == {"verbosity_slope_pts", "length_partial_corr", "self_preference_warning"}


def test_certify_agreeing_judge_passes_bias_tests(cfg):
    assert ci.certify(cfg, perturber_model=_perturber(), judge_model=_judge("agree")) == 0
    bias_ = json.loads(ci.cert_path(cfg).read_text())["bias"]
    assert bias_["verbosity_slope_pts"]["pass"] and bias_["length_partial_corr"]["pass"]
