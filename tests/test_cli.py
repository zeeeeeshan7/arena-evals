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
