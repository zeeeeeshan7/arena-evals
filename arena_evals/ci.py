"""Judge certification and the CI gate orchestration (baseline cache, rerun policy, GitHub API)."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from inspect_ai.model import GenerateConfig, get_model

from arena_evals import agents, bias, calibration, corpus
from arena_evals.config import Config, rubric_hash, sha256_files
from arena_evals.scorers import judge_answer


# ---------------------------------------------------------------- certification (M5, M6)
def cert_path(cfg: Config) -> Path:
    return cfg.root / "configs" / "judge.cert.json"


def labels_path(cfg: Config) -> Path:
    return cfg.root / "calibration" / "labels.jsonl"


def current_cert_inputs(cfg: Config) -> dict:
    lp = labels_path(cfg)
    return {"rubric_hash": rubric_hash(cfg.rubric_dir), "judge_model": cfg.models["judge"]["model"],
            "judge_temperature": cfg.models["judge"]["temperature"],
            "labels_hash": sha256_files(lp) if lp.exists() else None}


def cert_status(cfg: Config) -> tuple[bool, str]:
    """(usable, reason). Stale if rubric, judge model/temperature, or labels changed since certification."""
    p = cert_path(cfg)
    if not p.exists():
        return False, "no cert (run `python -m arena_evals certify`)"
    cert = json.loads(p.read_text(encoding="utf-8"))
    stale = [k for k, v in current_cert_inputs(cfg).items() if cert.get(k) != v]
    if stale:
        return False, "stale cert: " + ", ".join(stale) + " changed"
    if not cert.get("certified"):
        return False, "judge failed certification"
    return True, "certified"


async def _certify_judgements(cfg: Config, labeled: list[dict], judge_model, perturber_model
                              ) -> tuple[list[bool | None], list[dict]]:
    """Re-judge every labeled answer, plus padded (+50%, +100%) and compressed versions of it."""
    jc = cfg.models["judge"]
    jm = judge_model or get_model(jc["model"], config=GenerateConfig(temperature=jc["temperature"], seed=jc["seed"]))
    pm = perturber_model or get_model(cfg.models["perturber"]["model"])
    sem = asyncio.Semaphore(cfg.eval["max_connections"])
    docs = corpus.load(cfg.root / "corpus" / "docs")
    seed = cfg.eval["seed"]

    async def judge_text(e: dict, text: str) -> bool | None:
        async with sem:
            v, _ = await judge_answer(jm, cfg.rubric_dir, e["input"], e["reference"], text,
                                      {c: docs[c].text for c in e["citations"] if c in docs}, cfg.eval["cache"])
        return None if v is None else bool(v.correct and v.faithful)

    async def compress(e: dict) -> str | None:
        async with sem:
            agents.METER.check()
            out = await pm.generate(bias.COMPRESS_PROMPT.format(answer=e["answer"]))
        c = out.completion.strip()
        keep = c and len(c) <= 0.7 * len(e["answer"]) and bias.facts_signature(c) == bias.facts_signature(e["answer"])
        return c if keep else None

    rejudged = list(await asyncio.gather(*(judge_text(e, e["answer"]) for e in labeled)))
    usable = [e for e in labeled if e["answer"]]
    variants = [(e, bias.pad(e["answer"], frac, seed + i)) for i, e in enumerate(usable) for frac in (0.5, 1.0)]
    comps = await asyncio.gather(*(compress(e) for e in usable))
    variants += [(e, c) for e, c in zip(usable, comps) if c]
    passes = await asyncio.gather(*(judge_text(e, t) for e, t in variants))
    perturbed = [{"example_id": e["example_id"], "x": bias.length_x(t, e["answer"]), "judge_pass": p}
                 for (e, t), p in zip(variants, passes)]
    return rejudged, perturbed


def certify(cfg: Config, *, judge_model=None, perturber_model=None) -> int:
    """Re-judge labeled examples with the current judge + rubric, run bias tests, write configs/judge.cert.json."""
    examples = {e["example_id"]: e for e in calibration.read_jsonl(cfg.root / "calibration" / "to_label.jsonl")}
    labels = calibration.read_jsonl(labels_path(cfg))
    labeled = [examples[l["example_id"]] | {"human_label": l["human_label"]} for l in labels
               if l["example_id"] in examples]
    if not labeled:
        print("certify: no labeled examples in calibration/labels.jsonl")
        return 1
    rejudged, perturbed = asyncio.run(_certify_judgements(cfg, labeled, judge_model, perturber_model))
    for e, jp in zip(labeled, rejudged):
        e["judge_pass"] = jp
    rep = calibration.report(labels, [{"example_id": e["example_id"], "judge_pass": e["judge_pass"]} for e in labeled],
                             cfg.eval["n_resamples"], cfg.eval["seed"])
    b = bias.verbosity(labeled, perturbed, cfg)
    k, t = rep["agreement"]["kappa"], cfg.cert
    agree_ok = bool(k["ci95"][0] >= t["kappa_ci_lower_min"] and k["point"] >= t["kappa_point_min"])
    certified = agree_ok and b["verbosity_slope_pts"]["pass"] and b["length_partial_corr"]["pass"]
    cert = {"certified": certified, "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            **current_cert_inputs(cfg), "n_labels": rep["n"],
            "judge_errors": sum(j is None for j in rejudged),
            "thresholds": {"kappa_ci_lower_min": t["kappa_ci_lower_min"], "kappa_point_min": t["kappa_point_min"],
                           "verbosity_slope_max_pts": t["verbosity_slope_max_pts"]},
            "agreement": rep["agreement"] | {"confusion": rep["confusion"]}, "bias": b}
    cert_path(cfg).write_text(json.dumps(cert, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"kappa {k['point']:.3f} (95% CI {k['ci95'][0]:.3f} to {k['ci95'][1]:.3f}), agreement ok: {agree_ok}")
    print(f"verbosity slope {b['verbosity_slope_pts']['point']:.2f} pts per +50% length "
          f"(95% CI {b['verbosity_slope_pts']['ci95'][0]:.2f} to {b['verbosity_slope_pts']['ci95'][1]:.2f}), "
          f"pass: {b['verbosity_slope_pts']['pass']}")
    print(f"length partial corr {b['length_partial_corr']['point']:.3f} (95% CI {b['length_partial_corr']['ci95'][0]:.3f}"
          f" to {b['length_partial_corr']['ci95'][1]:.3f}), pass: {b['length_partial_corr']['pass']}")
    print(f"certified: {certified} -> {cert_path(cfg)}")
    return 0 if certified else 1
