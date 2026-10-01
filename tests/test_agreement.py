import math

import numpy as np
import pytest

from arena_evals.stats.agreement import (agreement_report, cohen_kappa, confusion, gwet_ac1, pabak,
                                         precision_recall, raw_agreement, wilson)

# 100 examples: human pass 54, judge pass 52; TP 47, TN 41, FP 5, FN 7
TRUTH = [1] * 47 + [0] * 41 + [0] * 5 + [1] * 7
PRED = [1] * 47 + [0] * 41 + [1] * 5 + [0] * 7


def test_point_metrics_on_known_table():
    assert raw_agreement(PRED, TRUTH) == pytest.approx(0.88)
    # pe = 0.52*0.54 + 0.48*0.46 = 0.5016; kappa = (0.88-0.5016)/(1-0.5016)
    assert cohen_kappa(PRED, TRUTH) == pytest.approx((0.88 - 0.5016) / 0.4984)
    assert pabak(PRED, TRUTH) == pytest.approx(0.76)
    # pi = 0.53; pe = 2*0.53*0.47 = 0.4982
    assert gwet_ac1(PRED, TRUTH) == pytest.approx((0.88 - 0.4982) / 0.5018)
    assert precision_recall(PRED, TRUTH) == pytest.approx((47 / 52, 47 / 54))
    assert confusion(PRED, TRUTH) == [[41, 5], [7, 47]]


def test_kappa_degenerate_all_same_is_nan_ac1_is_defined():
    assert math.isnan(cohen_kappa([1] * 10, [1] * 10))
    assert gwet_ac1([1] * 10, [1] * 10) == 1.0


def test_agreement_report_has_cis_and_is_seeded():
    a = agreement_report(PRED, TRUTH, n_resamples=2000, seed=5)
    b = agreement_report(PRED, TRUTH, n_resamples=2000, seed=5)
    assert a == b
    for name in ("raw", "kappa", "ac1", "pabak", "precision", "recall"):
        assert a[name].lo <= a[name].point <= a[name].hi


def test_agreement_report_rejects_mismatch():
    with pytest.raises(ValueError):
        agreement_report([1, 0], [1])


def test_wilson_known_value():
    ci = wilson(6, 60)
    assert ci.point == pytest.approx(0.1)
    assert ci.lo == pytest.approx(0.0466, abs=1e-3)
    assert ci.hi == pytest.approx(0.2012, abs=1e-3)
    with pytest.raises(ValueError):
        wilson(0, 0)
