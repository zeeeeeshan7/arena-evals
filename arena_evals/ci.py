"""Judge certification and the CI gate orchestration (baseline cache, rerun policy, GitHub API)."""
from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from inspect_ai.model import GenerateConfig, get_model

from arena_evals import agents, bias, calibration, corpus, datasets, report, run
from arena_evals.agents import CostCapExceeded, CostMeter
from arena_evals.config import (SCORING_FILES, Config, model_args, rubric_hash, scorer_hash, sha256_bytes,
                                sha256_files, temperature_sent)
from arena_evals.scorers import judge_answer
from arena_evals.stats.bootstrap import bootstrap_ci, gate_decision, paired_bootstrap, per_tag


# ---------------------------------------------------------------- certification (M5, M6)
def cert_path(cfg: Config) -> Path:
    return cfg.root / "configs" / "judge.cert.json"


def labels_path(cfg: Config) -> Path:
    return cfg.root / "calibration" / "labels.jsonl"


def current_cert_inputs(cfg: Config) -> dict:
    lp = labels_path(cfg)
    return {"rubric_hash": rubric_hash(cfg.rubric_dir), "judge_model": cfg.models["judge"]["model"],
            "judge_temperature": cfg.models["judge"]["temperature"],
            "judge_temperature_via_extra_body": not temperature_sent(cfg.models["judge"]["model"]),
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
    jm = judge_model or get_model(
        jc["model"], config=GenerateConfig(temperature=jc["temperature"], seed=jc["seed"]),
        **model_args(jc["model"], jc["temperature"]))
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


# ---------------------------------------------------------------- gate pieces (M7)
EXIT = {"pass": 0, "warn": 0, "block": 1, "error": 2}


class GitHub:
    """Minimal GitHub REST client over urllib (GITHUB_TOKEN, GITHUB_REPOSITORY, GITHUB_API_URL)."""

    def __init__(self, repo: str | None = None, token: str | None = None, api: str | None = None):
        self.repo = repo or os.environ["GITHUB_REPOSITORY"]
        self.token = token or os.environ["GITHUB_TOKEN"]
        self.api = (api or os.environ.get("GITHUB_API_URL", "https://api.github.com")).rstrip("/")

    def _req(self, method: str, path: str, body: dict | None = None):
        req = urllib.request.Request(f"{self.api}{path}", method=method,
                                     data=json.dumps(body).encode() if body is not None else None,
                                     headers={"Authorization": f"Bearer {self.token}",
                                              "Accept": "application/vnd.github+json",
                                              "X-GitHub-Api-Version": "2022-11-28"})
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = resp.read()
        return json.loads(data) if data else None

    def pr(self, n: int) -> dict:
        return self._req("GET", f"/repos/{self.repo}/pulls/{n}")

    def changed_files(self, n: int) -> list[str]:
        out, page = [], 1
        while True:
            batch = self._req("GET", f"/repos/{self.repo}/pulls/{n}/files?per_page=100&page={page}")
            out += [f["filename"] for f in batch]
            if len(batch) < 100:
                return out
            page += 1

    def comments(self, n: int) -> list[dict]:
        out, page = [], 1
        while True:
            batch = self._req("GET", f"/repos/{self.repo}/issues/{n}/comments?per_page=100&page={page}")
            out += batch
            if len(batch) < 100:
                return out
            page += 1

    def upsert_comment(self, n: int, body: str) -> None:
        mine = own_gate_comments(self.comments(n))
        if mine:
            self._req("PATCH", f"/repos/{self.repo}/issues/comments/{mine[-1]['id']}", {"body": body})
        else:
            self._req("POST", f"/repos/{self.repo}/issues/{n}/comments", {"body": body})

    def set_status(self, sha: str, state: str, description: str, context: str, target_url: str = "") -> None:
        body = {"state": state, "description": description[:140], "context": context}
        if target_url:
            body["target_url"] = target_url
        self._req("POST", f"/repos/{self.repo}/statuses/{sha}", body)


BOT_LOGIN = "github-actions[bot]"
GATED_PREFIXES = ("prompts/", "agents/", "configs/", "datasets/")
GATED_FILES = ("arena_evals/agents.py",)


def own_gate_comments(comments: list[dict]) -> list[dict]:
    """Gate comments written by the Actions bot. A marker typed by anyone else is untrusted input."""
    return [c for c in comments if report.MARKER_PREFIX in (c.get("body") or "")
            and (c.get("user") or {}).get("login") == BOT_LOGIN]


def merge_base(root: Path, base_sha: str, head_sha: str) -> str:
    """The fork point: the base tip may contain commits the PR never saw, which would read as a regression."""
    try:
        return subprocess.run(["git", "merge-base", base_sha, head_sha], cwd=root, check=True, capture_output=True,
                              text=True).stdout.strip() or base_sha
    except (subprocess.CalledProcessError, FileNotFoundError, NotADirectoryError):
        return base_sha


def is_gated(changed: list[str]) -> bool:
    return any(f.startswith(GATED_PREFIXES) or f in GATED_FILES for f in changed)


def cache_key(base_sha: str, cfg: Config, dataset_hash: str, rubric_hash_: str, scorer_hash_: str) -> str:
    a, j = cfg.models["agent"], cfg.models["judge"]
    parts = [base_sha, a["model"], j["model"], a["temperature"], j["temperature"], cfg.eval["limits"],
             {"agent": a["seed"], "judge": j["seed"], "bootstrap": cfg.eval["seed"]}, dataset_hash, rubric_hash_,
             scorer_hash_]
    return sha256_bytes(json.dumps(parts, sort_keys=True).encode()).removeprefix("sha256:")[:32]


def judge_error_frac(rows: list[dict]) -> float:
    return sum(r["scores"]["judge_error"] for r in rows) / len(rows) if rows else 0.0


def worst(task_ids: list[str], d: np.ndarray, base_rows: list[dict], cand_rows: list[dict], n: int = 5) -> list[dict]:
    b, c = run.task_means(base_rows), run.task_means(cand_rows)
    trace = {}
    for r in cand_rows:
        if r["success"] is not True and r["trace_id"]:
            trace.setdefault(r["task_id"], r["trace_id"])
    order = sorted(range(len(task_ids)), key=lambda i: (d[i], task_ids[i]))[:n]
    return [{"task_id": task_ids[i], "base": b[task_ids[i]], "cand": c[task_ids[i]], "d_pts": float(d[i]) * 100,
             "trace_id": trace.get(task_ids[i], "")} for i in order if d[i] < 0]


def tag_deltas(task_ids: list[str], d: np.ndarray, records: dict[str, datasets.TaskRecord]) -> dict[str, np.ndarray]:
    out: dict[str, list[float]] = {}
    for t, v in zip(task_ids, d):
        for tag in records[t].tags:
            out.setdefault(tag, []).append(float(v))
    return {k: np.array(v) for k, v in out.items()}


def rerun_action(prior: dict | None, head_sha: str, rerun_reason: str) -> str:
    """'replay' = re-post the prior block without evaluating; 'evaluate' otherwise."""
    if prior and prior.get("head_sha") == head_sha and prior.get("verdict") == "block" and not rerun_reason.strip():
        return "replay"
    return "evaluate"


def artifact_url() -> str:
    if os.environ.get("GITHUB_RUN_ID"):
        return (f"{os.environ.get('GITHUB_SERVER_URL', 'https://github.com')}/{os.environ['GITHUB_REPOSITORY']}"
                f"/actions/runs/{os.environ['GITHUB_RUN_ID']}")
    return ""


# ---------------------------------------------------------------- gate orchestration (M7)
def generate_baseline(base_sha: str, cfg: Config, dataset: Path, log_dir: Path, max_usd: float, cache: bool) -> Path:
    """Phase A from a worktree of the base commit (its agent + prompts) against the HEAD dataset file."""
    wt = cfg.root / ".arena-base"
    if wt.exists():
        subprocess.run(["git", "worktree", "remove", "--force", str(wt)], cwd=cfg.root, check=False)
        shutil.rmtree(wt, ignore_errors=True)
    subprocess.run(["git", "worktree", "add", "--detach", str(wt), base_sha], cwd=cfg.root, check=True,
                   capture_output=True)
    if not (wt / "arena_evals" / "__main__.py").exists():
        raise RuntimeError(f"base commit {base_sha[:12]} does not contain the eval harness")
    cmd = [sys.executable, "-m", "arena_evals", "generate", "--split", cfg.gate["split"], "--variant",
           cfg.gate["variant"], "--dataset", str(dataset), "--log-dir", str(log_dir), "--max-usd", f"{max_usd:.4f}"]
    p = subprocess.run(cmd + ([] if cache else ["--no-cache"]), cwd=wt, capture_output=True, text=True)
    if p.returncode == 3:
        raise CostCapExceeded(p.stderr.strip().splitlines()[-1] if p.stderr.strip() else "cost cap hit in baseline")
    if p.returncode != 0:
        raise RuntimeError(f"baseline generate failed (exit {p.returncode}): {p.stderr[-2000:]}")
    # Do not parse stdout: on a CI runner the console output is padded and wrapped. The log is in the directory
    # we told `generate` to write to.
    logs = sorted(Path(log_dir).glob("*.eval"), key=lambda f: f.stat().st_mtime)
    if not logs:
        raise RuntimeError(f"baseline generate wrote no .eval log in {log_dir}")
    return logs[-1]


def gate(pr: int, cfg: Config, *, gh: GitHub | None = None, rerun_reason: str = "") -> int:
    gh = gh or GitHub()
    info = gh.pr(pr)
    head_sha, base_sha = info["head"]["sha"], info["base"]["sha"]
    ctx, split, variant = cfg.gate["status_context"], cfg.gate["split"], cfg.gate["variant"]
    out = cfg.root / ".arena-out"
    out.mkdir(exist_ok=True)
    run_id = run.new_run_id()
    ds = cfg.root / "datasets" / f"{split}.jsonl"
    cap = cfg.eval["cost"]["max_usd_per_gate"]
    meter = agents.set_meter(CostMeter(cap, cfg.models["prices"]))
    cert: dict = {}
    meta = {"head_sha": head_sha, "run_id": run_id, "artifact_url": artifact_url(), "rerun_reason": rerun_reason,
            "eps_pts": cfg.gate["eps_pts"], "dataset_hash": datasets.dataset_hash(ds),
            "rubric_hash": rubric_hash(cfg.rubric_dir), "self_preference_warning": bias.self_preference(cfg)}

    def finish(verdict: str, body: str) -> int:
        (out / "comment.md").write_text(body, encoding="utf-8", newline="\n")
        (out / "gate.json").write_text(json.dumps(meta | {"verdict": verdict}, indent=2, default=str), encoding="utf-8")
        gh.upsert_comment(pr, body)
        state = "success" if verdict in ("pass", "warn") else "failure"
        gh.set_status(head_sha, state, f"{verdict}: {meta.get('headline', meta.get('error', ''))}", ctx,
                      meta["artifact_url"])
        print(f"gate verdict: {verdict}")
        return EXIT[verdict]

    def fail(reason: str) -> int:
        meta["error"] = reason
        return finish("error", report.render(None, [], [], cert, {"spent": meter.spent, "cap": cap},
                                             meta | {"verdict": "error"}))

    # 0. never run a fork's code with this repo's secrets, not even on a maintainer's workflow_dispatch
    if info["head"]["repo"]["full_name"] != info["base"]["repo"]["full_name"]:
        return fail("fork PR: the gate does not run untrusted code with secrets; push the branch to this repo")
    changed = gh.changed_files(pr)
    if not rerun_reason.strip() and not is_gated(changed):
        gh.set_status(head_sha, "success", "not applicable: no gated path changed", ctx)
        print("gate verdict: not applicable")
        return 0
    base_sha = merge_base(cfg.root, base_sha, head_sha)
    meta["merge_base"] = base_sha

    # 1. cert check (auto-recertify only when this PR touched the judge)
    ok, why = cert_status(cfg)
    cert = json.loads(cert_path(cfg).read_text(encoding="utf-8")) if cert_path(cfg).exists() else {}
    if not ok:
        if any(f.startswith("prompts/judge/") or f == "configs/models.yaml" for f in changed):
            ok = certify(cfg) == 0
            cert = json.loads(cert_path(cfg).read_text(encoding="utf-8"))
            shutil.copy(cert_path(cfg), out / "judge.cert.json")
            meta["recertified"] = True
        if not ok:
            return fail(f"judge uncertified ({why})")

    # 2. rerun policy
    prior_body = next((c["body"] for c in reversed(own_gate_comments(gh.comments(pr)))), "")
    if rerun_action(report.parse_marker(prior_body), head_sha, rerun_reason) == "replay":
        meta["headline"] = "prior block stands (re-run without a reason)"
        return finish("block", report.replay_note(prior_body))
    cache = bool(cfg.eval["cache"]) and not rerun_reason.strip()

    try:
        # 3. baseline: cache hit, or base worktree phase A + head phase B
        key = cache_key(base_sha, cfg, meta["dataset_hash"], meta["rubric_hash"], scorer_hash(cfg.root))
        bdir = cfg.root / ".arena-cache" / "baseline" / key
        meta["baseline_cache"] = "hit" if cache and (bdir / "results.jsonl").exists() else "miss"
        meta["baseline_rerun_on_head"] = any(f.startswith(("datasets/", "prompts/judge/")) for f in changed)
        if meta["baseline_cache"] == "hit":
            base_rows = run.read_results(bdir / "results.jsonl")
            base_manifest = json.loads((bdir / "manifest.json").read_text(encoding="utf-8"))
            shutil.copytree(bdir, out / "baseline", dirs_exist_ok=True)  # artifact carries both sides on a hit too
        else:
            log = generate_baseline(base_sha, cfg, ds, out / "logs-base", cap - meter.spent, cache)
            b_out = run.score(log, ds, cfg, out_dir=out / "baseline", cache=cache)
            base_rows, base_manifest = run.read_results(b_out.results_path), b_out.manifest
            meter.spent = max(meter.spent, base_manifest["cost_usd"])
            if cache:
                bdir.mkdir(parents=True, exist_ok=True)
                shutil.copy(b_out.results_path, bdir / "results.jsonl")
                shutil.copy(out / "baseline" / "manifest.json", bdir / "manifest.json")
        # 4. candidate, after a pre-flight cost check
        need = base_manifest["cost_usd"] * cfg.eval["cost"]["preflight_factor"]
        if need > cap - meter.spent:
            return fail(f"cost pre-flight: candidate needs ~${need:.2f}, ${cap - meter.spent:.2f} of ${cap:.2f} left")
        log = run.generate(split, variant, cfg.prompts_dir, ds, cfg, run_id=run_id, cache=cache,
                           log_dir=out / "logs-cand")
        c_out = run.score(log, ds, cfg, out_dir=out / "candidate", cache=cache)
        cand_rows = run.read_results(c_out.results_path)
    except (CostCapExceeded, RuntimeError) as e:
        return fail(str(e))
    except subprocess.CalledProcessError as e:
        return fail(f"git failed: {' '.join(map(str, e.cmd))}: {(e.stderr or b'')[-500:]!r}")
    except Exception as e:  # noqa: BLE001  never leave the PR silent: report it as an error and set the status
        return fail(f"unexpected {type(e).__name__}: {e}")

    # 5. compare
    meta["judge_errors"] = {"baseline": sum(r["scores"]["judge_error"] for r in base_rows),
                            "candidate": sum(r["scores"]["judge_error"] for r in cand_rows)}
    if max(judge_error_frac(base_rows), judge_error_frac(cand_rows)) > cfg.gate["judge_error_max_frac"]:
        return fail(f"judge errors above {cfg.gate['judge_error_max_frac']:.0%} of samples: {meta['judge_errors']}")
    ids, d, dropped = run.paired_deltas(base_rows, cand_rows)
    if len(ids) == 0:
        return fail("no task has a valid repeat on both sides")
    n_res, seed = cfg.eval["n_resamples"], cfg.eval["seed"]
    paired = paired_bootstrap(d, n_res, seed)
    verdict = gate_decision(paired, cfg.gate["eps_pts"], cfg.gate["upper_q"])
    records = {r.id: r for r in datasets.load(ds)}
    bm, cm = run.task_means(base_rows), run.task_means(cand_rows)
    meta |= {"verdict": verdict, "dropped_tasks": dropped,
             "base_rate": bootstrap_ci(np.array([bm[t] for t in ids]), n_res, seed).as_dict(),
             "cand_rate": bootstrap_ci(np.array([cm[t] for t in ids]), n_res, seed).as_dict(),
             "paired": paired.as_dict()}
    meta["headline"] = f"{paired.delta * 100:+.1f} pts (97.5% upper {paired.upper_975 * 100:+.1f}, n={paired.n})"
    tags = per_tag(tag_deltas(ids, d, records), n_res, seed, cfg.gate["bh_q"])
    body = report.render(paired, tags, worst(ids, d, base_rows, cand_rows), cert,
                         {"spent": meter.spent, "cap": cap}, meta)
    return finish(verdict, body)


# ---------------------------------------------------------------- A/A flake test (M7)
def aa(runs: int, cfg: Config, split: str = "gate") -> int:
    """A/A flake test: head vs itself, response cache bypassed. Pass iff false-block rate <= 5%."""
    ds = cfg.root / "datasets" / f"{split}.jsonl"
    out = cfg.root / ".arena-out" / "aa"
    agents.set_meter(CostMeter(cfg.eval["cost"]["max_usd_per_gate"] * runs, cfg.models["prices"]))
    verdicts = []
    for i in range(runs):
        sides = []
        for side in ("a", "b"):
            log = run.generate(split, cfg.gate["variant"], cfg.prompts_dir, ds, cfg, cache=False,
                               log_dir=out / f"logs-{i}-{side}")
            sides.append(run.read_results(run.score(log, ds, cfg, out_dir=out / f"{i}-{side}", cache=False).results_path))
        _, d, _ = run.paired_deltas(*sides)
        r = paired_bootstrap(d, cfg.eval["n_resamples"], cfg.eval["seed"] + i)
        verdicts.append(gate_decision(r, cfg.gate["eps_pts"], cfg.gate["upper_q"]))
        print(f"A/A run {i + 1}/{runs}: {verdicts[-1]} delta {r.delta * 100:+.2f} pts "
              f"(95% CI {r.ci95[0] * 100:+.2f} to {r.ci95[1] * 100:+.2f})")
    rate = verdicts.count("block") / runs
    (out / "aa.json").write_text(json.dumps({"runs": runs, "verdicts": verdicts, "false_block_rate": rate}, indent=2),
                                 encoding="utf-8")
    print(f"A/A false-block rate: {rate:.1%} ({verdicts.count('block')}/{runs}); pass iff <= 5%")
    return 0 if rate <= 0.05 else 1
