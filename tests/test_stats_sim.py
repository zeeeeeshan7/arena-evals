import pytest

from arena_evals import config
from arena_evals.stats.simulate import simulate_coverage, simulate_gate

SD = config.load().eval["sim"]["sd"]


@pytest.mark.slow
def test_power_at_minus_8_pts():
    assert simulate_gate(-8.0, SD, 300, trials=1000, n_resamples=2000, eps_pts=2.0, seed=1) >= 0.80


@pytest.mark.slow
def test_false_block_at_zero():
    assert simulate_gate(0.0, SD, 300, trials=1000, n_resamples=2000, eps_pts=2.0, seed=2) <= 0.05


@pytest.mark.slow
def test_ci_coverage():
    assert 0.935 <= simulate_coverage(-5.0, SD, 300, trials=1000, seed=3) <= 0.965


def test_simulation_smoke():
    rate = simulate_gate(-30.0, 0.45, 300, trials=20, n_resamples=500, seed=0)
    assert rate == 1.0
