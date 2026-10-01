"""Judge bias checks: self-preference warning (M4) and verbosity tests (M6)."""
from __future__ import annotations

import math
import random
import re

import numpy as np

from arena_evals.config import Config
from arena_evals.stats import CI


def model_family(model_id: str) -> str:
    """'anthropic/claude-sonnet-4-5-20250929' -> 'claude'."""
    return model_id.split("/")[-1].split("-")[0].lower()


def self_preference(cfg: Config) -> bool:
    """True (warn) when agent and judge models share a family prefix."""
    return model_family(cfg.models["agent"]["model"]) == model_family(cfg.models["judge"]["model"])


# ---------------------------------------------------------------- verbosity (M6)
FILLER = [
    "This answer is based on the documents available to me.",
    "Please let me know if you need anything else.",
    "I hope this information is helpful to you.",
    "The details above summarize what the documentation says.",
    "Feel free to ask a follow-up question at any time.",
    "This response has been written to be as clear as possible.",
    "Thank you for your question about this topic.",
    "As always, it is worth keeping this information in mind.",
]
_FACT = re.compile(r"[A-Z]{2,3}-\d{3}|\d[\d,]*(?:\.\d+)?")
COMPRESS_PROMPT = ("Rewrite the answer below so it is at least 30% shorter. Keep every fact, number, name, date and "
                   "document ID exactly as written. Output only the rewritten answer.\n\nANSWER:\n{answer}")


def pad(text: str, frac: float, seed: int) -> str:
    """Append content-free filler sentences until the text is >= (1 + frac) x its original length."""
    rng = random.Random(seed)
    out, target = text, len(text) * (1 + frac)
    while len(out) < target:
        out += " " + rng.choice(FILLER)
    return out


def facts_signature(text: str) -> list[str]:
    """Numbers and doc IDs in order: the deterministic extraction a compression must not change."""
    return sorted(m.replace(",", "") for m in _FACT.findall(text))


def length_x(perturbed: str, original: str) -> float:
    """Length change in units of +50%: log(len_p / len_o) / log(1.5)."""
    return math.log(len(perturbed) / len(original)) / math.log(1.5)


def _slope(x: np.ndarray, y: np.ndarray) -> float:
    vx = np.var(x)
    return float(np.cov(x, y, bias=True)[0, 1] / vx) if vx > 0 else float("nan")


def slope_ci(x, y, groups, n_resamples: int = 10_000, seed: int = 0) -> CI:
    """OLS slope of y on x with a cluster bootstrap over groups (one group = one labeled example)."""
    x, y, groups = np.asarray(x, float), np.asarray(y, float), np.asarray(groups)
    uniq = np.unique(groups)
    members = [np.nonzero(groups == g)[0] for g in uniq]
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_resamples):
        idx = np.concatenate([members[i] for i in rng.integers(0, len(uniq), len(uniq))])
        boots.append(_slope(x[idx], y[idx]))
    boots = np.asarray(boots)
    lo, hi = np.nanquantile(boots, [0.025, 0.975]) if np.isfinite(boots).any() else (float("nan"),) * 2
    return CI(_slope(x, y), float(lo), float(hi))


def _partial_corr(judge_pass: np.ndarray, loglen: np.ndarray, human: np.ndarray) -> float:
    def resid(v):
        r = v.astype(float).copy()
        for h in (0, 1):
            m = human == h
            if m.any():
                r[m] -= r[m].mean()
        return r
    a, b = resid(judge_pass), resid(loglen)
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    # zero residual variance (e.g. judge agrees with the human on every example): no length association to measure
    return float((a * b).sum() / denom) if denom > 0 else 0.0


def partial_corr_ci(judge_pass, loglen, human, n_resamples: int = 10_000, seed: int = 0) -> CI:
    """Correlation of judge pass with log length, controlling for the human label (residualize both on it)."""
    jp, ll, hu = (np.asarray(v, float) for v in (judge_pass, loglen, human))
    rng = np.random.default_rng(seed)
    boots = np.array([_partial_corr(jp[i], ll[i], hu[i]) for i in rng.integers(0, jp.size, (n_resamples, jp.size))])
    lo, hi = np.nanquantile(boots, [0.025, 0.975]) if np.isfinite(boots).any() else (float("nan"),) * 2
    return CI(_partial_corr(jp, ll, hu), float(lo), float(hi))


def verbosity(labels: list[dict], perturbed: list[dict], cfg: Config) -> dict:
    """labels: labeled examples with answer, variant, human_label, judge_pass (re-judged original).
    perturbed: rows {example_id, x, judge_pass} for padded/compressed re-judgements (see ci.certify)."""
    n_res, seed = cfg.eval["n_resamples"], cfg.eval["seed"]
    rows = [{"example_id": e["example_id"], "x": 0.0, "judge_pass": e["judge_pass"]} for e in labels
            if e.get("judge_pass") is not None]
    rows += [r for r in perturbed if r.get("judge_pass") is not None]
    slope = slope_ci([r["x"] for r in rows], [100.0 * r["judge_pass"] for r in rows],
                     [r["example_id"] for r in rows], n_res, seed)
    real = [e for e in labels if e["variant"] in ("concise", "verbose") and e.get("judge_pass") is not None]
    pc = partial_corr_ci([e["judge_pass"] for e in real], [math.log(max(len(e["answer"]), 1)) for e in real],
                         [e["human_label"] == "pass" for e in real], n_res, seed)
    return {"verbosity_slope_pts": {**slope.as_dict(), "pass": bool(slope.hi <= cfg.cert["verbosity_slope_max_pts"]),
                                    "n_rows": len(rows)},
            "length_partial_corr": {**pc.as_dict(), "pass": bool(pc.lo <= 0), "n": len(real)},
            "self_preference_warning": self_preference(cfg)}
