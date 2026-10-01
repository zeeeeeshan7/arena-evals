import numpy as np
import pytest

from arena_evals.stats.bootstrap import (PairedResult, benjamini_hochberg, bootstrap_ci, gate_decision, mde,
                                         paired_bootstrap, per_tag)


def _result(delta: float, boot: list[float]) -> PairedResult:
    return PairedResult(delta=delta, ci95=(0.0, 0.0), upper_975=0.0, p_neg=0.0, mde=0.0, n=300, sd=0.45,
                        boot=np.asarray(boot, dtype=float))


def test_bootstrap_ci_known_data_is_seeded_and_brackets_mean():
    x = np.array([0.0, 1.0] * 150)
    a = bootstrap_ci(x, n_resamples=10_000, seed=7)
    b = bootstrap_ci(x, n_resamples=10_000, seed=7)
    assert a == b
    assert a.point == pytest.approx(0.5)
    # SE of a 0/1 mean at p=0.5, n=300 is 0.0289; 95% half-width ~0.057
    assert a.lo == pytest.approx(0.443, abs=0.01)
    assert a.hi == pytest.approx(0.557, abs=0.01)


def test_mde_matches_prd():
    assert round(mde(0.45, 300) * 100, 1) == 7.3


def test_paired_bootstrap_fields():
    rng = np.random.default_rng(0)
    d = rng.normal(-0.08, 0.45, 300)
    r = paired_bootstrap(d, n_resamples=10_000, seed=1)
    assert r.n == 300
    assert r.ci95[0] < r.delta < r.ci95[1]
    assert r.upper_975 == pytest.approx(r.ci95[1])
    assert 0.0 <= r.p_neg <= 1.0
    assert r.as_dict()["delta_pts"] == pytest.approx(r.delta * 100)


def test_gate_block_at_exactly_minus_two_pts():
    assert gate_decision(_result(-0.02, [-0.03] * 1000)) == "block"


def test_gate_warn_when_upper_bound_exactly_zero():
    boot = [-0.05] * 900 + [0.0] * 100          # 97.5th percentile is exactly 0.0
    assert gate_decision(_result(-0.03, boot)) == "warn"


def test_gate_pass_when_drop_smaller_than_eps():
    assert gate_decision(_result(-0.019, [-0.03] * 1000)) == "pass"


def test_gate_block_via_bootstrap_on_big_drop():
    rng = np.random.default_rng(3)
    r = paired_bootstrap(rng.normal(-0.12, 0.45, 300), seed=3)
    assert gate_decision(r) == "block"


def test_all_ties_zero_variance_passes_without_nan():
    r = paired_bootstrap(np.zeros(300), seed=0)
    assert (r.delta, r.sd, r.mde, r.upper_975, r.p_neg) == (0.0, 0.0, 0.0, 0.0, 0.0)
    assert gate_decision(r) == "pass"


def test_constant_negative_deltas_block():
    r = paired_bootstrap(np.full(300, -0.05), seed=0)
    assert gate_decision(r) == "block"


def test_empty_deltas_raise():
    with pytest.raises(ValueError):
        paired_bootstrap(np.array([]))


def test_benjamini_hochberg():
    assert benjamini_hochberg([0.001, 0.02, 0.04, 0.5], q=0.05) == [True, True, False, False]
    assert benjamini_hochberg([]) == []


def test_per_tag_flags_only_drops():
    rows = per_tag({"pricing": np.full(40, -0.3), "hr": np.full(40, 0.3)}, n_resamples=2000, seed=0)
    by = {r["tag"]: r for r in rows}
    assert by["pricing"]["flagged"] is True
    assert by["hr"]["flagged"] is False
