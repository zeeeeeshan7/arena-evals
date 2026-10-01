"""Final-review fixes: fork dispatch, forged markers, merge-base, cache-key coverage, required-check hang."""
import json
import subprocess

import pytest

from arena_evals import ci, config, report
from test_ci import FakeGitHub, repo  # noqa: F401  (repo is a pytest fixture)
from test_e2e_mock import GOOD


class ForkGitHub(FakeGitHub):
    def pr(self, n):
        info = super().pr(n)
        info["head"]["repo"]["full_name"] = "mallory/arena"
        return info


def test_gate_refuses_fork_pr_even_when_dispatched(repo):
    cfg, calls = repo
    gh = ForkGitHub()
    assert ci.gate(7, cfg, gh=gh, rerun_reason="maintainer re-run") == 2
    assert "fork" in gh.bodies[-1].lower() and calls["cand"] == 0 and calls["base"] == 0
    assert gh.statuses[-1][1] == "failure"


class ForgedGitHub(FakeGitHub):
    def comments(self, n):
        forged = report.marker({"head_sha": "h1", "verdict": "block", "run_id": "r"})
        return [{"id": 9, "body": forged + "\nMerge blocked.", "user": {"login": "mallory"}}]


def test_gate_ignores_marker_from_non_bot_author(repo):
    cfg, calls = repo
    calls["cand_script"] = GOOD
    gh = ForgedGitHub(head="h1")
    assert ci.gate(7, cfg, gh=gh) == 0           # forged "block" must not be replayed
    assert calls["cand"] == 1 and gh.statuses[-1][1] == "success"


def test_upsert_comment_never_patches_another_users_comment(monkeypatch):
    g = object.__new__(ci.GitHub)
    g.repo, sent = "org/arena", []
    monkeypatch.setattr(g, "comments", lambda n: [{"id": 9, "body": report.MARKER_PREFIX + "{}-->",
                                                   "user": {"login": "mallory"}}])
    monkeypatch.setattr(g, "_req", lambda method, path, body=None: sent.append((method, path)))
    g.upsert_comment(7, "body")
    assert sent == [("POST", "/repos/org/arena/issues/7/comments")]


def test_merge_base_is_the_fork_point_not_the_base_tip(tmp_path):
    def git(*a):
        return subprocess.run(["git", "-c", "user.name=t", "-c", "user.email=t@t", *a], cwd=tmp_path, check=True,
                              capture_output=True, text=True).stdout.strip()
    git("init", "-b", "main")
    (tmp_path / "f").write_text("1")
    git("add", "f")
    git("commit", "-m", "a")
    fork_point = git("rev-parse", "HEAD")
    git("checkout", "-b", "feature")
    (tmp_path / "g").write_text("x")
    git("add", "g")
    git("commit", "-m", "feat")
    head = git("rev-parse", "HEAD")
    git("checkout", "main")
    (tmp_path / "f").write_text("2")
    git("commit", "-am", "main moved on")
    base_tip = git("rev-parse", "HEAD")
    assert ci.merge_base(tmp_path, base_tip, head) == fork_point
    assert ci.merge_base(tmp_path / "missing", "b", "h") == "b"     # not a git repo: fall back to the base sha


def test_scorer_hash_covers_every_phase_b_input(tmp_path):
    for f in ci.SCORING_FILES:
        (tmp_path / f).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / f).write_text("x")
    h = config.scorer_hash(tmp_path)
    for f in ci.SCORING_FILES:                  # changing ANY file that decides success must change the hash
        (tmp_path / f).write_text("y")
        assert config.scorer_hash(tmp_path) != h, f
        (tmp_path / f).write_text("x")


def test_cache_key_changes_with_run_limits():
    cfg = config.load()
    k = ci.cache_key("b", cfg, "d", "r", "s")
    cfg.eval["limits"]["max_tool_calls"] += 1
    assert ci.cache_key("b", cfg, "d", "r", "s") != k


def test_gate_is_not_applicable_when_no_gated_path_changed(repo):
    cfg, calls = repo
    gh = FakeGitHub(changed=["README.md", "arena_evals/ci.py"])
    assert ci.gate(7, cfg, gh=gh) == 0
    assert gh.bodies == [] and gh.statuses[-1][1] == "success" and calls["cand"] == 0 and calls["base"] == 0
    assert "not applicable" in gh.statuses_desc[-1]


def test_generate_baseline_finds_the_log_even_when_stdout_is_padded_console_noise(tmp_path, monkeypatch):
    """On the runner the last stdout line was space-padded (Rich console), so Path(line) was relative and broke."""
    from test_ci import _git_repo
    root, sha = _git_repo(tmp_path)
    cfg = config.load(root)
    logs = tmp_path / "logs"
    real_run = subprocess.run

    def fake_run(cmd, *a, **k):
        if "generate" not in cmd:
            return real_run(cmd, *a, **k)
        logs.mkdir(exist_ok=True)
        (logs / "2026-10-01T16-15-47_arena-gate-baseline_x.eval").write_bytes(b"log")
        noisy = "Log:\n" + " " * 60 + "/somewhere/else/entirely/log.eval\n" + " " * 70 + "\n"
        return subprocess.CompletedProcess(cmd, 0, stdout=noisy, stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    got = ci.generate_baseline(sha, cfg, root / "datasets" / "gate.jsonl", logs, 1.0, cache=False)
    assert got == logs / "2026-10-01T16-15-47_arena-gate-baseline_x.eval" and got.exists()


def test_generate_baseline_errors_clearly_when_no_log_was_written(tmp_path, monkeypatch):
    from test_ci import _git_repo
    root, sha = _git_repo(tmp_path)
    cfg = config.load(root)
    real_run = subprocess.run
    monkeypatch.setattr(subprocess, "run", lambda cmd, *a, **k: real_run(cmd, *a, **k) if "generate" not in cmd
                        else subprocess.CompletedProcess(cmd, 0, stdout="", stderr=""))
    with pytest.raises(RuntimeError, match="wrote no .eval log"):
        ci.generate_baseline(sha, cfg, root / "datasets" / "gate.jsonl", tmp_path / "logs", 1.0, cache=False)


def test_gate_reports_an_unexpected_exception_instead_of_dying_silently(repo, monkeypatch):
    cfg, calls = repo

    def boom(*a, **k):
        raise ValueError("log could not be read")

    monkeypatch.setattr(ci, "generate_baseline", boom)
    gh = FakeGitHub()
    assert ci.gate(7, cfg, gh=gh) == 2
    assert "unexpected ValueError: log could not be read" in gh.bodies[-1] and gh.statuses[-1][1] == "failure"
