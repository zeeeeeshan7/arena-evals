import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from arena_evals import ci, config, datasets, report, run
from arena_evals.agents import CostCapExceeded
from arena_evals.config import ROOT
from test_e2e_mock import BAD, GOOD, TASKS, scripted_agent


def row(task, success, judge_error=False, trace="t"):
    return {"task_id": task, "success": success, "scores": {"judge_error": judge_error}, "trace_id": trace}


def test_judge_error_frac():
    assert ci.judge_error_frac([row("a", True)] * 49 + [row("b", None, True)]) == pytest.approx(0.02)
    assert ci.judge_error_frac([]) == 0.0


def test_cache_key_changes_with_every_input():
    cfg = config.load()
    k = ci.cache_key("base", cfg, "sha256:d", "sha256:r", "sha256:s")
    assert k == ci.cache_key("base", cfg, "sha256:d", "sha256:r", "sha256:s")
    assert k != ci.cache_key("base", cfg, "sha256:d", "sha256:r", "sha256:s2")   # scorer change -> miss
    assert k != ci.cache_key("base", cfg, "sha256:d2", "sha256:r", "sha256:s")
    assert k != ci.cache_key("base", cfg, "sha256:d", "sha256:r2", "sha256:s")
    assert k != ci.cache_key("base2", cfg, "sha256:d", "sha256:r", "sha256:s")
    cfg.models["agent"]["temperature"] = 0.1
    assert k != ci.cache_key("base", cfg, "sha256:d", "sha256:r", "sha256:s")


def test_rerun_action():
    prior = {"head_sha": "h1", "verdict": "block", "run_id": "r"}
    assert ci.rerun_action(prior, "h1", "") == "replay"
    assert ci.rerun_action(prior, "h1", "   ") == "replay"
    assert ci.rerun_action(prior, "h1", "flaky judge, see #12") == "evaluate"
    assert ci.rerun_action(prior, "h2", "") == "evaluate"                        # new commit
    assert ci.rerun_action(prior | {"verdict": "pass"}, "h1", "") == "evaluate"
    assert ci.rerun_action(None, "h1", "") == "evaluate"


def test_worst_lists_only_regressions_sorted():
    base = [row("a", True), row("b", True), row("c", False)]
    cand = [row("a", False, trace="ta"), row("b", True), row("c", True)]
    ids, d, _ = run.paired_deltas(base, cand)
    assert ci.worst(ids, d, base, cand) == [{"task_id": "a", "base": 1.0, "cand": 0.0, "d_pts": -100.0,
                                             "trace_id": "ta"}]


# ---------------------------------------------------------------- full gate() against a fake GitHub
class FakeGitHub:
    def __init__(self, head="h1", changed=("prompts/baseline.md",)):
        self.head, self.changed, self.bodies, self.statuses = head, list(changed), [], []
        self.statuses_desc = []

    def pr(self, n):
        return {"head": {"sha": self.head, "repo": {"full_name": "org/arena"}},
                "base": {"sha": "b0", "repo": {"full_name": "org/arena"}}}

    def changed_files(self, n):
        return self.changed

    def comments(self, n):
        return [{"id": 1, "body": self.bodies[-1], "user": {"login": "github-actions[bot]"}}] if self.bodies else []

    def upsert_comment(self, n, body):
        self.bodies.append(body)

    def set_status(self, sha, state, description, context, target_url=""):
        self.statuses.append((sha, state, context))
        self.statuses_desc.append(description)


@pytest.fixture
def repo(tmp_path, monkeypatch):
    for d in ("configs", "prompts", "corpus"):
        shutil.copytree(ROOT / d, tmp_path / d)
    (tmp_path / "arena_evals").mkdir()
    for f in ci.SCORING_FILES:
        shutil.copy(ROOT / f, tmp_path / f)
    datasets.write_jsonl(tmp_path / "datasets" / "gate.jsonl", TASKS)
    datasets.write_jsonl(tmp_path / "calibration" / "labels.jsonl", [{"example_id": "x"}])
    cfg = config.load(tmp_path)
    cfg.eval["k"] = 2
    ci.cert_path(cfg).write_text(json.dumps({"certified": True, **ci.current_cert_inputs(cfg)}))
    orig = run.generate
    calls = {"cand": 0, "base": 0, "cand_script": BAD}

    def fake_base(base_sha, cfg_, ds, log_dir, max_usd, cache):
        calls["base"] += 1
        return orig("gate", "baseline", cfg_.prompts_dir, ds, cfg_, model=scripted_agent(GOOD), log_dir=log_dir,
                    cache=False)

    def fake_cand(*a, **k):
        calls["cand"] += 1
        return orig(*a, **(k | {"model": scripted_agent(calls["cand_script"]), "cache": False}))

    monkeypatch.setattr(ci, "generate_baseline", fake_base)
    monkeypatch.setattr(run, "generate", fake_cand)
    return cfg, calls


def test_gate_blocks_then_replays_then_reruns_with_reason(repo):
    cfg, calls = repo
    gh = FakeGitHub()
    assert ci.gate(7, cfg, gh=gh) == 1
    body = gh.bodies[-1]
    assert "Merge blocked." in body and report.parse_marker(body)["verdict"] == "block"
    assert gh.statuses[-1] == ("h1", "failure", "arena-eval/gate")
    assert (cfg.root / ".arena-out" / "candidate" / "results.jsonl").exists()
    assert calls == {"cand": 1, "base": 1, "cand_script": BAD}

    assert ci.gate(7, cfg, gh=gh) == 1                       # same SHA, no reason: replay, no evaluation
    assert calls["cand"] == 1 and report.REPLAY_NOTE in gh.bodies[-1]

    assert ci.gate(7, cfg, gh=gh, rerun_reason="judge outage") == 1   # reason: evaluates, cache bypassed
    assert calls["cand"] == 2 and calls["base"] == 2
    assert "| Rerun reason | judge outage |" in gh.bodies[-1]


def test_gate_passes_on_noop_and_reuses_cached_baseline(repo):
    cfg, calls = repo
    calls["cand_script"] = GOOD
    assert ci.gate(8, cfg, gh=FakeGitHub(head="h2")) == 0
    shutil.rmtree(cfg.root / ".arena-out")
    gh = FakeGitHub(head="h3")
    assert ci.gate(8, cfg, gh=gh) == 0
    assert calls["base"] == 1                                # second run: baseline cache hit
    assert (cfg.root / ".arena-out" / "baseline" / "results.jsonl").exists()   # artifact still has both sides
    assert "cache hit" in gh.bodies[-1] and gh.statuses[-1][1] == "success"


def test_gate_refuses_stale_cert(repo):
    cfg, _ = repo
    (cfg.rubric_dir / "rubric.md").write_text("changed rubric")
    gh = FakeGitHub()
    assert ci.gate(9, cfg, gh=gh) == 2
    assert "judge uncertified (stale cert: rubric_hash changed)" in gh.bodies[-1]


def test_gate_cost_cap_is_error_with_no_score(repo, monkeypatch):
    cfg, _ = repo

    def boom(*a, **k):
        raise CostCapExceeded("cost cap hit ($20.01 of $20.00)")

    monkeypatch.setattr(ci, "generate_baseline", boom)
    gh = FakeGitHub()
    assert ci.gate(10, cfg, gh=gh) == 2
    assert "**No verdict:** cost cap hit ($20.01 of $20.00)" in gh.bodies[-1]
    assert "dropped task success" not in gh.bodies[-1]


def _git_repo(tmp_path: Path, with_harness: bool = True) -> tuple[Path, str]:
    import subprocess
    for d in ("configs", "prompts", "corpus") + (("arena_evals",) if with_harness else ()):
        shutil.copytree(ROOT / d, tmp_path / d, ignore=shutil.ignore_patterns("__pycache__"))
    models = tmp_path / "configs" / "models.yaml"
    models.write_text(models.read_text().replace("anthropic/deepseek-flash\n  temperature",
                                                 "mockllm/model\n  temperature", 1))
    datasets.write_jsonl(tmp_path / "datasets" / "gate.jsonl", TASKS[:2])
    git = lambda *a: subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True, text=True).stdout
    git("init", "-q")
    git("add", "-A")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base")
    return tmp_path, git("rev-parse", "HEAD").strip()


def test_generate_baseline_runs_base_worktree_via_cli(tmp_path):
    root, sha = _git_repo(tmp_path)
    cfg = config.load(root)
    log = ci.generate_baseline(sha, cfg, root / "datasets" / "gate.jsonl", tmp_path / "logs", 1.0, cache=False)
    assert log.exists() and log.suffix == ".eval"
    assert (root / ".arena-base" / "arena_evals" / "__main__.py").exists()


def test_generate_baseline_errors_when_base_lacks_harness(tmp_path):
    root, sha = _git_repo(tmp_path, with_harness=False)
    cfg = config.load(root)
    with pytest.raises(RuntimeError, match="does not contain the eval harness"):
        ci.generate_baseline(sha, cfg, root / "datasets" / "gate.jsonl", tmp_path / "logs", 1.0, cache=False)


def test_aa_counts_false_blocks(repo):
    cfg, calls = repo
    calls["cand_script"] = GOOD
    assert ci.aa(2, cfg, split="gate") == 0                  # identical scripted agent: never blocks
    out = json.loads((cfg.root / ".arena-out" / "aa" / "aa.json").read_text())
    assert out == {"runs": 2, "verdicts": ["pass", "pass"], "false_block_rate": 0.0}
    assert calls["cand"] == 4                                # 2 runs x 2 sides, cache bypassed
