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


# ---------------------------------------------------------------- visual comment
def test_bar_never_shows_a_full_bar_for_less_than_100_percent():
    assert report._bar(1.0, 20) == "█" * 20
    assert report._bar(0.982, 20) == "█" * 19 + "░"
    assert report._bar(0.0, 10) == "░" * 10 and report._bar(-0.2, 10) == "░" * 10 and len(report._bar(0.5, 7)) == 7


def test_strip_places_point_ci_zero_and_threshold_on_one_axis():
    line, axis = report._strip(lo=-19.5, hi=-10.5, point=-14.8, eps=2.0)
    assert len(line) == len(axis)
    assert line.index("●") > line.index("━") and line.rindex("━") > line.index("●")
    assert line.count("│") == 1 and line.count("┊") == 1 and line.index("┊") < line.index("│")
    assert line.index("┊") - line.index("│") == -2          # one char per point at this scale: threshold is 2 left of zero
    assert "0" in axis and "-10" in axis


def test_strip_stays_readable_for_huge_ranges():
    line, axis = report._strip(lo=-400.0, hi=-300.0, point=-350.0, eps=2.0)
    assert len(line) == len(axis) <= 80 and "●" in line


def test_tag_chart_sorts_by_size_marks_flags_and_caps_rows():
    tags = [{"tag": f"t{i}", "n": 5, "delta_pts": -float(i), "ci95_pts": [-i - 1.0, 1.0], "flagged": i % 2 == 0}
            for i in range(12)]
    rows = report._tag_chart(tags, max_rows=8)
    assert len(rows) == 8 and rows[0].startswith("t11") and "-11.0 pts" in rows[0]
    assert rows[0].rstrip().endswith("pts") and rows[1].startswith("t10") and rows[1].rstrip().endswith("●")
    assert report._tag_chart([], 8) == []


def test_render_leads_with_icon_bars_and_strip_and_keeps_the_tables():
    body = report.render(paired(), [{"tag": "lookup", "n": 45, "delta_pts": -10.4, "ci95_pts": [-17.0, -4.4],
                                     "flagged": True}], WORST, CERT, {"spent": 1.0, "cap": 4.0}, META)
    assert "## 🚫 Arena eval gate: BLOCK" in body
    assert "baseline   " in body and "candidate  " in body and "█" in body and "●" in body
    assert "| Baseline success | 82.0% (95% CI" in body and "### Worst 5 regressions" in body
    assert "lookup" in body and body.index("█") < body.index("### Worst 5 regressions")


def test_render_icons_for_every_verdict():
    pr = paired()
    for verdict, icon in (("pass", "✅"), ("warn", "⚠️"), ("block", "🚫")):
        assert f"## {icon} Arena eval gate: {verdict.upper()}" in report.render(
            pr, [], [], CERT, {"spent": 1.0, "cap": 4.0}, META | {"verdict": verdict})
    err = report.render(None, [], [], {}, {"spent": 0.1, "cap": 4.0}, {"verdict": "error", "error": "x", "run_id": "r"})
    assert "## ❗ Arena eval gate: ERROR" in err
