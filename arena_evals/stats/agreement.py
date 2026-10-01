"""Judge-vs-human agreement on binary labels (1 = pass). This task adds only the Wilson interval; Task 18 adds the rest."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from arena_evals.stats import CI


def wilson(k: int, n: int, conf: float = 0.95) -> CI:
    if n <= 0:
        raise ValueError("wilson needs n > 0")
    z = norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return CI(p, float(centre - half), float(centre + half))
