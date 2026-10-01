"""Monte Carlo power and CI-coverage checks for the gate (normal per-task diffs, as in the PRD)."""
from __future__ import annotations

import numpy as np

from arena_evals.stats.bootstrap import gate_decision, paired_bootstrap


def simulate_gate(true_delta_pts: float, sd: float, n: int, trials: int, n_resamples: int = 2000,
                  eps_pts: float = 2.0, seed: int = 0) -> float:
    """Fraction of trials the gate blocks when the true mean paired delta is true_delta_pts."""
    rng = np.random.default_rng(seed)
    blocks = 0
    for _ in range(trials):
        d = rng.normal(true_delta_pts / 100, sd, n)
        r = paired_bootstrap(d, n_resamples=n_resamples, seed=int(rng.integers(2**31)))
        blocks += gate_decision(r, eps_pts=eps_pts) == "block"
    return blocks / trials


def simulate_coverage(true_delta_pts: float, sd: float, n: int, trials: int, seed: int = 0,
                      n_resamples: int = 2000) -> float:
    """Fraction of trials whose two-sided 95% percentile CI contains the true delta."""
    rng = np.random.default_rng(seed)
    truth = true_delta_pts / 100
    hits = 0
    for _ in range(trials):
        r = paired_bootstrap(rng.normal(truth, sd, n), n_resamples=n_resamples, seed=int(rng.integers(2**31)))
        hits += r.ci95[0] <= truth <= r.ci95[1]
    return hits / trials
