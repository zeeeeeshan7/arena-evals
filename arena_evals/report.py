"""Render the ONE gate PR comment (markdown) and read back its hidden marker. Pure: no I/O."""
from __future__ import annotations

import json
import math
import re

MARKER_PREFIX = "<!-- arena-eval:v1 "
_MARKER = re.compile(r"<!-- arena-eval:v1 (\{.*?\}) -->")
MAX_CHARS = 65_000          # GitHub rejects comment bodies over 65,536 characters
REPLAY_NOTE = "> Re-run on the same head SHA without a `rerun_reason`: the previous **block** stands."


def marker(meta: dict) -> str:
    return MARKER_PREFIX + json.dumps({"head_sha": meta.get("head_sha"), "verdict": meta.get("verdict"),
                                       "run_id": meta.get("run_id")}) + " -->"


def parse_marker(body: str) -> dict | None:
    m = _MARKER.search(body or "")
    return json.loads(m.group(1)) if m else None


def replay_note(prior_body: str) -> str:
    return prior_body if REPLAY_NOTE in prior_body else prior_body.rstrip() + "\n\n" + REPLAY_NOTE + "\n"


ICON = {"block": "🚫", "warn": "⚠️", "pass": "✅", "error": "❗"}


def _bar(frac: float, width: int = 20) -> str:
    """Filled blocks for frac of width; floored so anything under 100% never shows a full bar."""
    n = max(0, min(width, math.floor(frac * width)))
    return "█" * n + "░" * (width - n)


def _strip(lo: float, hi: float, point: float, eps: float) -> tuple[str, str]:
    """One-line axis (in pts) with the CI as ━, the point as ●, zero as │ and the -eps gate threshold as ┊,
    plus its label line. One character per point, coarser only when the range would not fit in 70 columns."""
    a = math.floor(min(lo, -eps, 0.0) / 5) * 5 - 5
    b = math.ceil(max(hi, 0.0) / 5) * 5 + 5
    step = max(1, math.ceil((b - a) / 70))
    idx = lambda x: round((x - a) / step)                       # noqa: E731
    width = idx(b) + 1
    line = [" "] * width
    for i in range(idx(lo), idx(hi) + 1):
        line[i] = "━"
    line[idx(-eps)], line[idx(0.0)], line[idx(point)] = "┊", "│", "●"
    axis = [" "] * width
    first = math.ceil(a / (10 * step)) * 10 * step              # ticks on multiples of 10 so 0 is always labelled
    for tick in range(first, int(b) + 1, 10 * step):
        text, i = f"{tick:g}", idx(tick)
        for j, ch in enumerate(text):
            if i + j < width:
                axis[i + j] = ch
    return "".join(line), "".join(axis)


def _tag_chart(per_tag: list[dict], max_rows: int = 8, width: int = 12) -> list[str]:
    """Bars for the largest per-tag changes (advisory); a trailing ● marks a Benjamini-Hochberg flag."""
    top = sorted(per_tag, key=lambda t: -abs(t["delta_pts"]))[:max_rows]
    if not top:
        return []
    big = max(abs(t["delta_pts"]) for t in top) or 1.0
    name = max(len(t["tag"]) for t in top)
    return [f"{t['tag']:<{name}}  {'▼' if t['delta_pts'] < 0 else '▲'} {_bar(abs(t['delta_pts']) / big, width)} "
            f"{t['delta_pts']:+.1f} pts" + (" ●" if t["flagged"] else "") for t in top]


def _pct(ci: dict) -> str:
    return f"{ci['point'] * 100:.1f}% (95% CI {ci['ci95'][0] * 100:.1f} to {ci['ci95'][1] * 100:.1f})"


def _headline(p: dict, verdict: str, eps: float) -> str:
    d = p["delta_pts"]
    moved = f"dropped task success by {-d:.1f} pts" if d < 0 else (
        f"raised task success by {d:.1f} pts" if d > 0 else "did not change task success")
    tail = {"block": "**Merge blocked.**",
            "warn": f"Warning: the drop is at least {eps:.1f} pts but not statistically clear. Not blocked.",
            "pass": f"No statistically clear drop of {eps:.1f} pts or more. Pass."}[verdict]
    return (f"**This prompt change {moved}** (two-sided 95% CI {p['ci95_pts'][0]:+.1f} to {p['ci95_pts'][1]:+.1f} pts; "
            f"one-sided 97.5% upper bound {p['upper_975_pts']:+.1f} pts, n={p['n']}). {tail}")


def _cert_line(cert: dict, meta: dict) -> str:
    if not cert:
        return "not certified"
    k = cert.get("agreement", {}).get("kappa")
    s = "certified" if cert.get("certified") else "NOT certified"
    if k:
        s += f" (kappa {k['point']:.2f}, 95% CI {k['ci95'][0]:.2f} to {k['ci95'][1]:.2f}, n={cert.get('n_labels')})"
    if meta.get("recertified"):
        s += "; auto-recertified for this PR: commit the `judge.cert.json` from the artifact"
    if meta.get("self_preference_warning"):
        s += "; **self-preference warning**: agent and judge are the same model family"
    return s


def render(paired, per_tag: list[dict], worst5: list[dict], cert: dict, cost: dict, meta: dict) -> str:
    """paired: PairedResult or None (error outcome). meta carries verdict, hashes, cache/rerun info, artifact URL."""
    verdict = meta["verdict"]
    art = meta.get("artifact_url", "")
    head = [marker(meta), f"## {ICON.get(verdict, '')} Arena eval gate: {verdict.upper()}", ""]
    if verdict == "error" or paired is None:
        head += [f"**No verdict:** {meta.get('error', 'unknown error')}", ""]
    else:
        p = paired.as_dict()
        je = meta.get("judge_errors", {})
        b, c = meta["base_rate"], meta["cand_rate"]
        line, axis = _strip(p["ci95_pts"][0], p["ci95_pts"][1], p["delta_pts"], meta["eps_pts"])
        head += [_headline(p, verdict, meta["eps_pts"]), "", "```text",
                 "Task success (95% CI)",
                 f"  baseline   {_bar(b['point'])} {b['point'] * 100:5.1f}%   [{b['ci95'][0] * 100:.1f} to {b['ci95'][1] * 100:.1f}]",
                 f"  candidate  {_bar(c['point'])} {c['point'] * 100:5.1f}%   [{c['ci95'][0] * 100:.1f} to {c['ci95'][1] * 100:.1f}]",
                 "", f"Paired change in pts, with 95% CI   (● estimate, ━ CI, │ no change, ┊ -{meta['eps_pts']:g} pt gate threshold)",
                 "  " + line, "  " + axis, "```", ""]
        if per_tag:
            head += ["```text", "Where it moved (advisory; ● = flagged after Benjamini-Hochberg q=0.05)",
                     *("  " + r for r in _tag_chart(per_tag)), "```", ""]
        head += ["| | |", "|---|---|",
                 f"| Baseline success | {_pct(meta['base_rate'])} |",
                 f"| Candidate success | {_pct(meta['cand_rate'])} |",
                 f"| P(delta < 0) | {p['p_neg']:.3f} |",
                 f"| Minimum detectable effect (80% power) | {p['mde_pts']:.1f} pts at n={p['n']}, SD={p['sd']:.2f} |",
                 f"| Tasks compared | {p['n']} ({len(meta.get('dropped_tasks', []))} dropped: no valid repeat) |",
                 f"| Judge errors | baseline {je.get('baseline', 0)}, candidate {je.get('candidate', 0)} |"]
    head += [f"| Judge | {_cert_line(cert, meta)} |" if paired is not None else f"Judge: {_cert_line(cert, meta)}"]
    rows = [f"| Cost | ${cost['spent']:.2f} of ${cost['cap']:.2f} cap |",
            f"| Baseline | cache {meta.get('baseline_cache', 'n/a')}"
            + ("; **baseline re-run on this PR's dataset/rubric**" if meta.get("baseline_rerun_on_head") else "") + " |",
            f"| Dataset hash | `{meta.get('dataset_hash', '')}` |",
            f"| Rubric hash | `{meta.get('rubric_hash', '')}` |"]
    if meta.get("rerun_reason"):
        rows.append(f"| Rerun reason | {meta['rerun_reason']} |")
    if paired is None:
        rows = ["", "| | |", "|---|---|"] + rows
    head += rows
    if worst5:
        head += ["", "### Worst 5 regressions", "", "| Task | Baseline | Candidate | Delta | Trace |", "|---|---|---|---|---|"]
        head += [f"| {w['task_id']} | {w['base'] * 100:.0f}% | {w['cand'] * 100:.0f}% | {w['d_pts']:+.0f} pts | "
                 f"`{w['trace_id'] or 'n/a'}`" + (f" ([artifact]({art}))" if art else "") + " |" for w in worst5]
    foot = ["", f"Traces: download the `arena-eval-{meta.get('run_id', '')}` artifact"
            + (f" from [this run]({art})" if art else "")
            + ", unzip it, then run `PHOENIX_WORKING_DIR=<unzipped>/phoenix phoenix serve` and open http://localhost:6006."]
    tag_head = ["", f"### Per-tag deltas (advisory only, never gates; Benjamini-Hochberg q=0.05)", "",
                "| Tag | n | Delta | 95% CI | Flagged |", "|---|---|---|---|---|"] if per_tag else []
    tag_rows = [f"| {t['tag']} | {t['n']} | {t['delta_pts']:+.1f} pts | {t['ci95_pts'][0]:+.1f} to "
                f"{t['ci95_pts'][1]:+.1f} | {'yes' if t['flagged'] else ''} |" for t in per_tag]
    budget = MAX_CHARS - sum(len(x) + 1 for x in head + tag_head + foot) - 100
    kept = []
    for r in tag_rows:
        if budget - (len(r) + 1) < 0:
            break
        kept.append(r)
        budget -= len(r) + 1
    if len(kept) < len(tag_rows):
        kept.append(f"| ({len(tag_rows) - len(kept)} more tags omitted: comment size limit) | | | | |")
    return "\n".join(head + tag_head + kept + foot) + "\n"
