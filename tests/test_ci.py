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
