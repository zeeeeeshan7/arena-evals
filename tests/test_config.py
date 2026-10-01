from pathlib import Path

import pytest

from arena_evals import config


def test_load_repo_configs():
    cfg = config.load()
    assert cfg.eval["k"] == 3 and cfg.eval["n_resamples"] == 10_000
    assert cfg.gate["eps_pts"] == 2.0 and cfg.gate["upper_q"] == 0.975
    assert cfg.cert["kappa_ci_lower_min"] == 0.4 and cfg.cert["kappa_point_min"] == 0.6
    assert cfg.models["agent"]["model"] != cfg.models["judge"]["model"]
    assert set(cfg.resolved()) == {"eval", "gate", "models", "cert"}


def test_load_rejects_missing_keys(tmp_path: Path):
    (tmp_path / "configs").mkdir()
    for name in ("eval", "gate", "models", "cert"):
        (tmp_path / "configs" / f"{name}.yaml").write_text("{}\n")
    with pytest.raises(ValueError, match="missing 'k'"):
        config.load(tmp_path)


def test_hashes(tmp_path: Path):
    (tmp_path / "a.md").write_bytes(b"alpha")
    (tmp_path / "b.md").write_bytes(b"beta")
    h = config.sha256_files(tmp_path / "a.md", tmp_path / "b.md")
    assert h == config.sha256_bytes(b"alphabeta")
    assert h == "sha256:a4c4aeb92c20500f364b12b3771ef3a11193e2cf04d0f28956a829749993b39f"
    assert h != config.sha256_files(tmp_path / "b.md", tmp_path / "a.md")
    (tmp_path / "system.md").write_bytes(b"s")
    (tmp_path / "rubric.md").write_bytes(b"r")
    assert config.rubric_hash(tmp_path) == config.sha256_bytes(b"sr")
    (tmp_path / "baseline.md").write_bytes(b"p")
    (tmp_path / "_output_contract.md").write_bytes(b"c")
    assert config.prompt_hash(tmp_path, "baseline") == config.sha256_bytes(b"pc")
