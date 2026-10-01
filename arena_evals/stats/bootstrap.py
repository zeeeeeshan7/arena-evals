"""Percentile bootstrap, paired comparison, gate rule, MDE. Units: fractions (0.02 = 2 pts) unless a name ends in _pts."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, Sequence

import numpy as np
from scipy.stats import norm

from arena_evals.stats import CI


def _boot_means(x: np.ndarray, n_resamples: int, seed: int) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.size == 0:
        raise ValueError("cannot bootstrap an empty array")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_resamples, x.size))
    return x[idx].mean(axis=1)


def bootstrap_ci(x: np.ndarray, n_resamples: int = 10_000, seed: int = 0, alpha: float = 0.05) -> CI:
    """Two-sided (1 - alpha) percentile bootstrap CI of the mean."""
    means = _boot_means(x, n_resamples, seed)
    lo, hi = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return CI(float(np.mean(x)), float(lo), float(hi))


def mde(sd: float, n: int, alpha_one_sided: float = 0.025, power: float = 0.80) -> float:
    """Minimum detectable effect (fraction) for a one-sided test at the given power."""
    if n <= 0:
        return float("nan")
    return float((norm.ppf(1 - alpha_one_sided) + norm.ppf(power)) * sd / np.sqrt(n))


@dataclass(frozen=True)
class PairedResult:
    delta: float                    # mean of d (fraction)
    ci95: tuple[float, float]       # two-sided 95% percentile CI, display only
    upper_975: float                # one-sided 97.5% upper bound = 97.5th percentile of resampled means
    p_neg: float                    # P(delta < 0) over resamples
    mde: float                      # 80%-power MDE (fraction) from observed SD and n
    n: int
    sd: float
    boot: np.ndarray = field(repr=False, compare=False)

    def as_dict(self) -> dict:
        return {"delta_pts": self.delta * 100, "ci95_pts": [self.ci95[0] * 100, self.ci95[1] * 100],
                "upper_975_pts": self.upper_975 * 100, "p_neg": self.p_neg, "mde_pts": self.mde * 100,
                "n": self.n, "sd": self.sd}


def paired_bootstrap(d: np.ndarray, n_resamples: int = 10_000, seed: int = 0) -> PairedResult:
    """d = per-task (candidate - baseline), each already averaged over repeats."""
    d = np.asarray(d, dtype=float)
    boot = _boot_means(d, n_resamples, seed)
    lo, hi, upper = np.quantile(boot, [0.025, 0.975, 0.975])
    sd = float(np.std(d, ddof=1)) if d.size > 1 else 0.0
    return PairedResult(delta=float(d.mean()), ci95=(float(lo), float(hi)), upper_975=float(upper),
                        p_neg=float(np.mean(boot < 0)), mde=mde(sd, d.size), n=int(d.size), sd=sd, boot=boot)


def gate_decision(r: PairedResult, eps_pts: float = 2.0, upper_q: float = 0.975) -> Literal["pass", "warn", "block"]:
    """block iff delta <= -eps AND one-sided upper bound < 0; warn iff delta <= -eps AND upper >= 0; else pass."""
    upper = float(np.quantile(r.boot, upper_q))
    if r.delta * 100 <= -eps_pts + 1e-9:  # tolerance so an exact -2.0 pts counts as <= -eps despite float error
        return "block" if upper < 0 else "warn"
    return "pass"


def benjamini_hochberg(pvals: Sequence[float], q: float = 0.05) -> list[bool]:
    """BH step-up: True where the null is rejected at FDR q."""
    p = np.asarray(pvals, dtype=float)
    m = p.size
    if m == 0:
        return []
    order = np.argsort(p)
    passed = p[order] <= q * np.arange(1, m + 1) / m
    k = int(np.max(np.nonzero(passed)[0])) + 1 if passed.any() else 0
    reject = np.zeros(m, dtype=bool)
    reject[order[:k]] = True
    return reject.tolist()


def per_tag(d_by_tag: dict[str, np.ndarray], n_resamples: int = 10_000, seed: int = 0, q: float = 0.05) -> list[dict]:
    """Advisory per-tag deltas (never gate). p = one-sided bootstrap P(mean >= 0); flagged = BH-rejected drop."""
    rows = []
    for tag in sorted(d_by_tag):
        d = np.asarray(d_by_tag[tag], dtype=float)
        boot = _boot_means(d, n_resamples, seed)
        rows.append({"tag": tag, "n": int(d.size), "delta_pts": float(d.mean()) * 100,
                     "ci95_pts": [float(np.quantile(boot, 0.025)) * 100, float(np.quantile(boot, 0.975)) * 100],
                     "p_one_sided": float(np.mean(boot >= 0))})
    for row, rej in zip(rows, benjamini_hochberg([r["p_one_sided"] for r in rows], q)):
        row["flagged"] = bool(rej and row["delta_pts"] < 0)
    return rows
