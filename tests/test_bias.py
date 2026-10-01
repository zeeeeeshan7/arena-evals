import math

import numpy as np
import pytest

from arena_evals import bias


def test_pad_reaches_target_length_and_is_seeded():
    a = bias.pad("The cap is 220 USD per night.", 0.5, seed=3)
    assert len(a) >= 1.5 * len("The cap is 220 USD per night.")
    assert a == bias.pad("The cap is 220 USD per night.", 0.5, seed=3)
    assert bias.facts_signature(a) == bias.facts_signature("The cap is 220 USD per night.")
    padded = bias.pad("x" * 100, 0.5, 0)
    assert bias.length_x(padded, "x" * 100) == pytest.approx(math.log(len(padded) / 100) / math.log(1.5))
    assert bias.length_x(padded, "x" * 100) >= 1.0


def test_facts_signature_detects_changed_numbers_and_ids():
    assert bias.facts_signature("16 weeks per HR-009, 1,340 alarms") == ["1340", "16", "HR-009"]
    assert bias.facts_signature("16 weeks") != bias.facts_signature("12 weeks")


def test_slope_ci_detects_length_bias_and_flat_judge():
    groups = np.repeat(np.arange(100), 3)
    x = np.tile([0.0, 1.0, 1.71], 100)
    flat = np.tile([100.0, 100.0, 100.0], 100)
    s = bias.slope_ci(x, flat, groups, n_resamples=500, seed=0)
    assert (s.point, s.lo, s.hi) == (0.0, 0.0, 0.0)
    rng = np.random.default_rng(0)
    biased = (rng.random(300) < 0.5 + 0.1 * x).astype(float) * 100   # +10 pts per +50% length
    s = bias.slope_ci(x, biased, groups, n_resamples=500, seed=0)
    assert s.lo > 2.0


def test_partial_corr_controls_for_human_label():
    rng = np.random.default_rng(1)
    human = rng.integers(0, 2, 400)
    loglen = rng.normal(5, 1, 400) + human            # longer answers are genuinely better
    judge = human.copy()                             # judge tracks the human label only
    c = bias.partial_corr_ci(judge, loglen, human, n_resamples=500, seed=0)
    assert (c.point, c.lo, c.hi) == (0.0, 0.0, 0.0)             # perfect judge: zero residual variance -> 0
    judge2 = ((loglen + rng.normal(0, 0.5, 400)) > 5.5).astype(int)   # judge rewards length
    c2 = bias.partial_corr_ci(judge2, loglen, human, n_resamples=500, seed=0)
    assert c2.lo > 0
