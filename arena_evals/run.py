"""Phase A (generate: agent -> .eval log) and phase B (score: scorers -> results.jsonl + manifest.json)."""
from __future__ import annotations

import importlib.metadata
import json
import os
import secrets
import subprocess
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from inspect_ai import Task, eval as inspect_eval, score as inspect_score
from inspect_ai.dataset import MemoryDataset, Sample
from inspect_ai.log import read_eval_log
from inspect_ai.model import GenerateConfig

from arena_evals import datasets
from arena_evals.agents import CostCapExceeded, arena_agent, parse_final, usage_cost
from arena_evals.config import Config, prompt_hash, scorer_hash
from arena_evals.scorers import abstention, answer_match, citation, gold_retrieval, task_success, tool_call_count


@dataclass
class RunOutput:
    results_path: Path
    manifest: dict


def new_run_id() -> str:
    if os.environ.get("GITHUB_RUN_ID"):
        return f"gh-{os.environ['GITHUB_RUN_ID']}-{os.environ.get('GITHUB_RUN_ATTEMPT', '1')}"
    return f"r-{datetime.now(timezone.utc):%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"


def git_sha(cwd: Path) -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=cwd, capture_output=True, text=True,
                              check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def generate(split: str, variant: str, prompts_dir: Path, dataset: Path, cfg: Config, *, model=None,
             log_dir: Path | None = None, run_id: str | None = None, epochs: int | None = None,
             cache: bool | None = None, limit: int | None = None) -> Path:
    """Phase A. Runs the agent on every task x epochs; returns the .eval log path. Raises CostCapExceeded."""
    prompts_dir, dataset = Path(prompts_dir), Path(dataset)
    records = datasets.load(dataset)
    run_id = run_id or new_run_id()
    cache = cfg.eval["cache"] if cache is None else cache
    lim, agent = cfg.eval["limits"], cfg.models["agent"]
    meta = {"run_id": run_id, "split": split, "variant": variant, "git_sha": git_sha(prompts_dir),
            "dataset_hash": datasets.dataset_hash(dataset), "prompt_hash": prompt_hash(prompts_dir, variant),
            "k": epochs or cfg.eval["k"], "cache": cache}
    task = Task(
        dataset=MemoryDataset([Sample(input=r.input, target=r.reference or "", id=r.id, metadata=r.model_dump())
                               for r in records]),
        solver=arena_agent(variant, str(prompts_dir), cache),
        epochs=meta["k"], message_limit=lim["message_limit"], token_limit=lim["token_limit"],
        time_limit=lim["time_limit"], name=f"arena-{split}-{variant}",
        config=GenerateConfig(temperature=agent["temperature"], seed=agent["seed"]),
    )
    log = inspect_eval(task, model=model or agent["model"], log_dir=str(log_dir or cfg.root / "logs"),
                       fail_on_error=False, max_connections=cfg.eval["max_connections"], limit=limit,
                       metadata=meta)[0]
    for s in log.samples or []:
        if s.error and "CostCapExceeded" in s.error.message:
            raise CostCapExceeded(s.error.message)
    if log.status == "error":
        raise RuntimeError(f"eval failed: {log.error.message if log.error else 'unknown error'}")
    return Path(log.location)


def sample_status(sample, max_tool_calls: int) -> str:
    if sample.error:
        return "agent_error"
    if sample.limit:
        return "timeout" if sample.limit.type in ("time", "working") else "limit"
    if tool_call_count(sample.messages) > max_tool_calls:
        return "limit"
    if parse_final(sample.output.completion if sample.output else "") is None:
        return "format_error"
    return "ok"


def score(log: Path, dataset: Path, cfg: Config, *, out_dir: Path | None = None,
          cache: bool | None = None) -> RunOutput:
    """Phase B. Scores a phase-A log with THIS checkout's scorers; writes results.jsonl + manifest.json.
    The LLM judge (which is what `cache` controls) arrives in Task 16; until then free_text samples have
    judge=None and success=None."""
    ev = read_eval_log(str(log))
    records = {r.id: r for r in datasets.load(dataset)}
    prices = cfg.models["prices"]
    scored = inspect_score(ev, [answer_match(), citation(), gold_retrieval(), abstention()], display="none")
    meta = scored.eval.metadata or {}
    run_id = meta.get("run_id") or new_run_id()
    out_dir = Path(out_dir or cfg.root / "runs" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in scored.samples:
        task = records[str(s.id)]
        sc = {k: v.value for k, v in (s.scores or {}).items()}
        status = sample_status(s, cfg.eval["limits"]["max_tool_calls"])
        scores = {"answer_match": sc["answer_match"], "gold_retrieval": sc["gold_retrieval"],
                  "citation_precision": sc["citation"]["precision"], "citation_recall": sc["citation"]["recall"],
                  "abstention": sc["abstention"], "judge": None, "judge_error": False}
        f = parse_final(s.output.completion if s.output else "")
        rows.append({
            "run_id": run_id, "variant": meta.get("variant"), "task_id": task.id, "repeat": s.epoch - 1,
            "status": status, "answer": f.answer if f else None, "citations": f.citations if f else [],
            "abstain": f.abstain if f else False, "scores": scores,
            "success": task_success(task.model_dump(), scores, status),
            "latency_s": s.total_time, "tokens_in": sum(u.input_tokens for u in (s.model_usage or {}).values()),
            "tokens_out": sum(u.output_tokens for u in (s.model_usage or {}).values()),
            "cost_usd": sum(usage_cost(u, name, prices) for name, u in (s.model_usage or {}).items()),
            "trace_id": (s.metadata or {}).get("trace_id", ""),
        })
    results_path = datasets.write_jsonl(out_dir / "results.jsonl", rows)
    agent = cfg.models["agent"]
    manifest = {
        "run_id": run_id, "git_sha": meta.get("git_sha", "unknown"), "variant": meta.get("variant"),
        "split": meta.get("split"), "dataset_hash": datasets.dataset_hash(dataset),
        "generated_dataset_hash": meta.get("dataset_hash"), "prompt_hash": meta.get("prompt_hash"),
        "rubric_hash": None, "scorer_hash": scorer_hash(cfg.root),
        "agent_model": str(scored.eval.model), "judge_model": None,
        "agent_temperature": agent["temperature"], "judge_temperature": None,
        "k": meta.get("k"), "seeds": {"agent": agent["seed"], "bootstrap": cfg.eval["seed"]},
        "inspect_version": importlib.metadata.version("inspect-ai"), "limits": cfg.eval["limits"],
        "cache": meta.get("cache"), "n_samples": len(rows), "cost_usd": sum(r["cost_usd"] for r in rows),
        "label_noise_floor": datasets.label_noise_floor(meta.get("split") or "", cfg.root),
        "log": str(log), "config": cfg.resolved(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8", newline="\n")
    return RunOutput(results_path, manifest)


def read_results(path: Path) -> list[dict]:
    return [json.loads(l) for l in Path(path).read_text(encoding="utf-8").splitlines() if l.strip()]


def task_means(rows: list[dict]) -> dict[str, float | None]:
    """Mean success per task over its valid repeats (success None = judge error = missing). None if no valid repeat.
    A task with fewer than k rows (a missing repeat) is averaged over the rows it has."""
    acc: dict[str, list[float]] = {}
    for r in rows:
        acc.setdefault(r["task_id"], [])
        if r["success"] is not None:
            acc[r["task_id"]].append(float(r["success"]))
    return {t: (float(np.mean(v)) if v else None) for t, v in acc.items()}


def paired_deltas(base_rows: list[dict], cand_rows: list[dict]) -> tuple[list[str], np.ndarray, list[str]]:
    """(task_ids, d = cand - base, dropped task_ids). Tasks without a valid repeat on either side are dropped."""
    b, c = task_means(base_rows), task_means(cand_rows)
    ids = sorted(set(b) | set(c))
    keep = [t for t in ids if b.get(t) is not None and c.get(t) is not None]
    return keep, np.array([c[t] - b[t] for t in keep]), [t for t in ids if t not in keep]
