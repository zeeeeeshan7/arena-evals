import numpy as np

from arena_evals import report
from arena_evals.stats.bootstrap import paired_bootstrap

CERT = {"certified": True, "n_labels": 100, "agreement": {"kappa": {"point": 0.76, "ci95": [0.63, 0.88]}}}
META = {"verdict": "block", "head_sha": "abc123", "run_id": "gh-1-1", "eps_pts": 2.0, "artifact_url": "https://x/runs/1",
        "base_rate": {"point": 0.82, "ci95": [0.78, 0.86]}, "cand_rate": {"point": 0.75, "ci95": [0.70, 0.79]},
        "judge_errors": {"baseline": 0, "candidate": 1}, "dataset_hash": "sha256:d", "rubric_hash": "sha256:r",
        "baseline_cache": "miss", "baseline_rerun_on_head": True, "dropped_tasks": []}
WORST = [{"task_id": "gate-0042", "base": 1.0, "cand": 0.0, "d_pts": -100.0, "trace_id": "5f1c"}]


def paired():
    return paired_bootstrap(np.random.default_rng(0).normal(-0.073, 0.45, 300), seed=1)


def test_render_block_has_required_fields_and_marker():
    body = report.render(paired(), [], WORST, CERT, {"spent": 12.3, "cap": 20.0}, META)
    assert report.parse_marker(body) == {"head_sha": "abc123", "verdict": "block", "run_id": "gh-1-1"}
    for s in ("dropped task success by", "one-sided 97.5% upper bound", "n=300", "Merge blocked.",
              "Minimum detectable effect", "gate-0042", "`5f1c`", "https://x/runs/1", "certified (kappa 0.76",
              "$12.30 of $20.00", "sha256:d", "sha256:r", "baseline re-run on this PR's dataset/rubric",
              "Judge errors | baseline 0, candidate 1", "phoenix serve", "Baseline success | 82.0% (95% CI"):
        assert s in body, s


def test_render_error_has_no_verdict_numbers():
    body = report.render(None, [], [], {}, {"spent": 20.1, "cap": 20.0},
                         {"verdict": "error", "error": "cost cap hit ($20.10 of $20.00)", "run_id": "r"})
    assert "**No verdict:** cost cap hit ($20.10 of $20.00)" in body
    assert report.parse_marker(body)["verdict"] == "error"


def test_render_truncates_huge_per_tag_table_but_keeps_marker():
    tags = [{"tag": f"tag-{i:05d}-" + "x" * 40, "n": 3, "delta_pts": -1.0, "ci95_pts": [-5.0, 3.0], "flagged": False}
            for i in range(5000)]
    body = report.render(paired(), tags, WORST, CERT, {"spent": 1.0, "cap": 20.0}, META)
    assert len(body) <= 65_536
    assert "more tags omitted" in body
    assert report.parse_marker(body)["verdict"] == "block"
    assert body.rstrip().endswith("http://localhost:6006.")


def test_replay_note_is_idempotent():
    once = report.replay_note("body")
    assert report.replay_note(once) == once and report.REPLAY_NOTE in once


def test_parse_marker_absent():
    assert report.parse_marker("just a human comment") is None
    assert report.parse_marker(None) is None
