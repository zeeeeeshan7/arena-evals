"""Judge certification and the CI gate orchestration (baseline cache, rerun policy, GitHub API)."""
from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from inspect_ai.model import GenerateConfig, get_model

from arena_evals import calibration, corpus
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


async def _rejudge(cfg: Config, labeled: list[dict], judge_model) -> list[bool | None]:
    """Re-judge every labeled answer with the current judge + rubric. None = judge error."""
    jc = cfg.models["judge"]
    jm = judge_model or get_model(jc["model"], config=GenerateConfig(temperature=jc["temperature"], seed=jc["seed"]))
    sem = asyncio.Semaphore(cfg.eval["max_connections"])
    docs = corpus.load(cfg.root / "corpus" / "docs")

    async def judge_text(e: dict, text: str) -> bool | None:
        async with sem:
            v, _ = await judge_answer(jm, cfg.rubric_dir, e["input"], e["reference"], text,
                                      {c: docs[c].text for c in e["citations"] if c in docs}, cfg.eval["cache"])
        return None if v is None else bool(v.correct and v.faithful)

    return list(await asyncio.gather(*(judge_text(e, e["answer"]) for e in labeled)))


def certify(cfg: Config, *, judge_model=None) -> int:
    """Re-judge labeled examples with the current judge + rubric; write configs/judge.cert.json. Exit 0 iff certified."""
    examples = {e["example_id"]: e for e in calibration.read_jsonl(cfg.root / "calibration" / "to_label.jsonl")}
    labels = calibration.read_jsonl(labels_path(cfg))
    labeled = [examples[l["example_id"]] | {"human_label": l["human_label"]} for l in labels
               if l["example_id"] in examples]
    if not labeled:
        print("certify: no labeled examples in calibration/labels.jsonl")
        return 1
    rejudged = asyncio.run(_rejudge(cfg, labeled, judge_model))
    rep = calibration.report(labels, [{"example_id": e["example_id"], "judge_pass": jp} for e, jp in zip(labeled, rejudged)],
                             cfg.eval["n_resamples"], cfg.eval["seed"])
    k, t = rep["agreement"]["kappa"], cfg.cert
    certified = bool(k["ci95"][0] >= t["kappa_ci_lower_min"] and k["point"] >= t["kappa_point_min"])
    cert = {"certified": certified, "created_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            **current_cert_inputs(cfg), "n_labels": rep["n"],
            "judge_errors": sum(j is None for j in rejudged),
            "thresholds": {"kappa_ci_lower_min": t["kappa_ci_lower_min"], "kappa_point_min": t["kappa_point_min"]},
            "agreement": rep["agreement"] | {"confusion": rep["confusion"]}}
    cert_path(cfg).write_text(json.dumps(cert, indent=2) + "\n", encoding="utf-8", newline="\n")
    print(f"kappa {k['point']:.3f} (95% CI {k['ci95'][0]:.3f} to {k['ci95'][1]:.3f})")
    print(f"certified: {certified} -> {cert_path(cfg)}")
    return 0 if certified else 1
