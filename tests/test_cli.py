import pytest

from arena_evals.__main__ import main


def test_help_lists_every_command(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    for cmd in ("corpus", "datasets", "generate", "run", "score", "compare", "simulate", "calibrate", "certify",
                "cache-key", "gate", "aa"):
        assert cmd in out


def test_compare_prints_delta_ci_and_verdict(tmp_path, capsys):
    from arena_evals import datasets

    def rows(successes):
        return [{"task_id": f"t{i}", "repeat": 0, "success": s, "scores": {"judge_error": False}}
                for i, s in enumerate(successes)]
    base = datasets.write_jsonl(tmp_path / "b.jsonl", rows([True] * 100))
    cand = datasets.write_jsonl(tmp_path / "c.jsonl", rows([True] * 80 + [False] * 20))
    assert main(["compare", "--base", str(base), "--cand", str(cand)]) == 0
    out = capsys.readouterr().out
    assert "delta -20.00 pts" in out and "verdict: block" in out
