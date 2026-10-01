import pytest

from arena_evals.stats.agreement import wilson


def test_wilson_known_value():
    ci = wilson(6, 60)
    assert ci.point == pytest.approx(0.1)
    assert ci.lo == pytest.approx(0.0466, abs=1e-3)
    assert ci.hi == pytest.approx(0.2012, abs=1e-3)
    with pytest.raises(ValueError):
        wilson(0, 0)
