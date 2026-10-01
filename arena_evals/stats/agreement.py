"""Judge-vs-human agreement on binary labels (1 = pass). Degenerate cases return nan, never raise."""
from __future__ import annotations

import numpy as np
from scipy.stats import norm

from arena_evals.stats import CI


def _arr(x) -> np.ndarray:
    return np.asarray(x, dtype=int)


def raw_agreement(a, b) -> float:
    return float(np.mean(_arr(a) == _arr(b)))


def cohen_kappa(a, b) -> float:
    a, b = _arr(a), _arr(b)
    po = np.mean(a == b)
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float("nan") if pe == 1 else float((po - pe) / (1 - pe))


def gwet_ac1(a, b) -> float:
    a, b = _arr(a), _arr(b)
    po = np.mean(a == b)
    pi = (a.mean() + b.mean()) / 2
    pe = 2 * pi * (1 - pi)
    return float((po - pe) / (1 - pe))


def pabak(a, b) -> float:
    return 2 * raw_agreement(a, b) - 1


def precision_recall(pred, truth) -> tuple[float, float]:
    pred, truth = _arr(pred), _arr(truth)
    tp = int(np.sum((pred == 1) & (truth == 1)))
    fp = int(np.sum((pred == 1) & (truth == 0)))
    fn = int(np.sum((pred == 0) & (truth == 1)))
    precision = tp / (tp + fp) if tp + fp else float("nan")
    recall = tp / (tp + fn) if tp + fn else float("nan")
    return precision, recall


def confusion(pred, truth) -> list[list[int]]:
    """Rows = human (fail, pass); cols = judge (fail, pass): [[TN, FP], [FN, TP]]."""
    pred, truth = _arr(pred), _arr(truth)
    return [[int(np.sum((truth == t) & (pred == p))) for p in (0, 1)] for t in (0, 1)]


_METRICS = {
    "raw": raw_agreement,
    "kappa": cohen_kappa,
    "ac1": gwet_ac1,
    "pabak": pabak,
    "precision": lambda p, t: precision_recall(p, t)[0],
    "recall": lambda p, t: precision_recall(p, t)[1],
}


def agreement_report(pred, truth, n_resamples: int = 10_000, seed: int = 0) -> dict[str, CI]:
    """Each metric with a seeded percentile bootstrap 95% CI over examples (nan resamples ignored)."""
    pred, truth = _arr(pred), _arr(truth)
    if pred.size == 0 or pred.size != truth.size:
        raise ValueError("pred and truth must be non-empty and equal length")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, pred.size, size=(n_resamples, pred.size))
    out = {}
    for name, fn in _METRICS.items():
        vals = np.array([fn(pred[i], truth[i]) for i in idx])
        lo, hi = np.nanquantile(vals, [0.025, 0.975]) if np.isfinite(vals).any() else (float("nan"),) * 2
        out[name] = CI(float(fn(pred, truth)), float(lo), float(hi))
    return out


def wilson(k: int, n: int, conf: float = 0.95) -> CI:
    if n <= 0:
        raise ValueError("wilson needs n > 0")
    z = norm.ppf(1 - (1 - conf) / 2)
    p = k / n
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return CI(p, float(centre - half), float(centre + half))
