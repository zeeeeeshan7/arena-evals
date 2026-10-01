# Arena Eval Harness Implementation Plan

> For agentic workers: REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (- [ ]) syntax for tracking.

**Goal:** Build an automated, statistically sound CI quality gate that posts one PR comment with the paired change in arena-agent task success (with confidence intervals) and blocks merge when a drop of at least 2 pts is real.

**Architecture:** A deterministic synthetic corpus and three tools back four Inspect AI solver variants that differ only in their prompt file. Runs happen in two phases (phase A: the agent writes an Inspect `.eval` log; phase B: this checkout's deterministic scorers plus a certified pointwise LLM judge write `results.jsonl` + `manifest.json`), so baseline and candidate are always scored by identical code. Pure-function statistics (paired bootstrap, gate rule, agreement, bias) feed a GitHub Actions gate that caches the baseline, enforces a cost cap and a rerun policy, posts the comment and sets a commit status.

**Tech Stack:** Python 3.11+ (CI and examples use 3.12), venv + pip, inspect-ai 0.3.272 (pinned), anthropic, numpy, scipy, pydantic v2, PyYAML, arize-phoenix + arize-phoenix-otel + openinference-instrumentation-anthropic, pytest, GitHub Actions.

**Spec:** `docs/superpowers/specs/2026-09-30-arena-eval-harness-design.md` (implements `PRD.md` v0.3; milestones M1-M8, M9 out of scope).

**Conventions for every task:** run all commands from the repository root (the directory that contains `PRD.md`). Every command assumes the project virtualenv created in Task 1 is active (`source .venv/bin/activate`; Windows Git Bash: `source .venv/Scripts/activate`; PowerShell: `.venv\Scripts\Activate.ps1`). "Expected: FAIL/PASS" lines were observed when this plan's code was executed end to end on inspect-ai 0.3.272; if your counts differ, stop and find out why before continuing. Steps marked **(HUMAN)** need a person, an API key, money or GitHub settings: do not simulate them or invent their results. Task 8 (API spike) depends only on Task 1; Tasks 3 and 4 already rely on two of its findings (hook exceptions are swallowed; `system_message()` mangles braces), so running Task 8 right after Task 1 is fine and catches library drift earliest.

## Global Constraints

- Python >= 3.11 (`requires-python = ">=3.11"`); CI runs 3.12; dependencies installed with `pip install -e ".[dev]"` into `.venv`; inspect-ai pinned to 0.3.272 in `pyproject.toml`.
- Gate set n = 300 tasks; splits `gate` 300, `calibration` 120, `dev` 100; separate task sets; the gate split is never used to tune the judge.
- Gate type mix exactly: lookup 90 (30%), multi_hop 75 (25%), arithmetic 45 (15%), conflicting 45 (15%), unanswerable 45 (15%); `datasets check` allows +/- 1 task per type.
- k = 3 repeats per task (Inspect `epochs`); repeats are averaged per task BEFORE bootstrapping over tasks.
- Bootstrap: 10,000 seeded resamples (`configs/eval.yaml: n_resamples: 10000`, `seed: 20260930`), percentile method.
- Every reported score carries a 95% CI; all deltas are absolute percentage points (pts), never relative %.
- Gate rule: **block** iff delta <= -epsilon (epsilon = 2 pts) AND one-sided 97.5% upper bound < 0; **warn** iff delta <= -2 pts AND upper bound >= 0; **pass** otherwise. The two-sided 95% CI is display only.
- One-sided 97.5% upper bound = 97.5th percentile of the resampled mean paired deltas.
- MDE (80% power) = (z_0.975 + z_0.80) x SD(d) / sqrt(n); `mde(0.45, 300)` = 7.3 pts; reported on every run.
- Per-tag metrics are advisory only (Benjamini-Hochberg, q = 0.05) and never affect the verdict.
- Judge certified iff kappa 95% CI lower bound >= 0.4 AND kappa point >= 0.6 on ~100 hand labels (`configs/cert.yaml`), AND verbosity slope 95% CI upper bound <= 2 pts per +50% length, AND length partial-correlation 95% CI lower bound <= 0.
- Gate refuses a stale cert (rubric hash, judge model, judge temperature or labels hash changed) or `certified: false`; it auto-recertifies only when the PR touches `prompts/judge/**` or `configs/models.yaml`.
- Judge errors (schema-invalid twice) are missing, not FAIL; if > 2% of samples in either run are judge errors the outcome is `error`.
- Agent errors, timeouts, limits and missing/invalid `FINAL` lines count as FAIL.
- Pinned config: agent `anthropic/claude-haiku-4-5-20251001` temperature 0.7 seed 1234; judge `anthropic/claude-sonnet-4-5-20250929` temperature 0.0 seed 1234 (`configs/models.yaml`); all recorded in every manifest.
- Cost: `cost.max_usd_per_gate: 20.0` is a starting value, enforced from the first run; Task 17 measures real cost and resets it; candidate pre-flight aborts if baseline cost x 1.2 exceeds the remaining cap.
- Baseline cache key = sha256(base SHA, agent model, judge model, agent temperature, judge temperature, seeds, dataset hash, rubric hash, scorer hash).
- Exit codes: pass 0, warn 0, block 1, error 2; commit status context `arena-eval/gate`.
- Runtime target: gate < 15 min for 300 tasks x 3 repeats (measured in Task 17).
- Hashes are sha256 over file bytes, so `.gitattributes` forces LF line endings.

## Review Focus

- **Zero-variance inputs to the statistics.** All-tie paired deltas (every d_i = 0) must give pass with MDE 0 and no NaN, and a judge that agrees with every human label has zero residual variance, which made the length partial correlation NaN and failed certification for a perfect judge (found while building this plan). Tests: `test_all_ties_zero_variance_passes_without_nan` and `test_constant_negative_deltas_block` (Task 11), `test_partial_corr_controls_for_human_label` (Task 22).
- **Unanswerable task where the agent hallucinates a citation.** `gold_doc_ids` is empty, so citation recall divides by zero, and success must hinge on abstention alone. Tests: `citation_pr(["HR-999"], [], IDS) == (0.0, None)` and the unanswerable rows of `test_task_success_table` (Task 9), plus the `e2e-0005` hallucinated-citation case in `test_e2e_known_verdicts` (Task 12).
- **Tasks with a missing repeat or only judge-error repeats.** Averaging must use the valid repeats, a task with no valid repeat on either side must be dropped from the pairing, and the 2% judge-error ceiling must void the verdict. Tests: `test_task_means_missing_repeat_and_judge_errors` (Task 10), `test_paired_deltas_drops_tasks_without_valid_repeat_on_either_side` (Task 12), `test_judge_error_frac` (Task 25).
- **`calculate()` injection and resource abuse.** `__import__('os')`, attribute access, lambdas, `True + 1`, `2 ** 101`, `10 ** 16`, `1e309`, division by zero, complex results and 300-term expressions must all come back as `error: ...` tool output, never raise into the agent loop or execute code. Tests: `test_rejects_unsafe_or_invalid` (Task 3) and `test_calculate_tool_returns_errors_as_output` (Task 3).
- **Baseline cache key after a scorer, dataset or rubric change.** A hit on a baseline scored by old scorer code would compare apples to oranges and could hide a regression. Tests: `test_cache_key_changes_with_every_input` (Task 25) and the cache hit/miss assertions in `test_gate_passes_on_noop_and_reuses_cached_baseline` (Task 26).

---

## Milestone M1: Arena

### Task 1: Repository scaffold, pinned configs, config loader, CLI skeleton

**Files:**
- Create: `.gitignore`
- Create: `.gitattributes`
- Create: `pyproject.toml`
- Create: `configs/eval.yaml`
- Create: `configs/gate.yaml`
- Create: `configs/models.yaml`
- Create: `configs/cert.yaml`
- Create: `arena_evals/__init__.py`
- Create: `arena_evals/config.py`
- Create: `arena_evals/stats/__init__.py`
- Create: `arena_evals/__main__.py`
- Create: `tests/conftest.py`
- Test: `tests/test_config.py`
- Test: `tests/test_cli.py`

**Interfaces:**
- Consumes: nothing
- Produces: `config.load(root: Path | str = ROOT) -> Config` (dataclass: `root`, `eval`, `gate`, `models`, `cert` dicts; properties `prompts_dir`, `rubric_dir`; `resolved() -> dict`)
- Produces: `config.sha256_bytes(*chunks: bytes) -> str`, `config.sha256_files(*paths) -> str` (format `"sha256:<hex>"`)
- Produces: `config.prompt_hash(prompts_dir, variant) -> str`, `config.rubric_hash(rubric_dir) -> str`, `config.scorer_hash(root=ROOT) -> str`, `config.ROOT`
- Produces: `arena_evals.stats.CI` (frozen dataclass `point`, `lo`, `hi`; `as_dict() -> {"point", "ci95": [lo, hi]}`)
- Produces: CLI `python -m arena_evals {corpus,datasets,generate,run,score,compare,simulate,calibrate,certify,cache-key,gate,aa}` (handlers import their modules lazily, so later tasks make them work)

- [ ] **Step 1: Initialise git and create the virtualenv**

The repository directory already contains `PRD.md` and `docs/`. It is not yet a git repository.

Run:

```bash
git init -b main
python --version
python -m venv .venv
source .venv/bin/activate    # Windows Git Bash: source .venv/Scripts/activate; PowerShell: .venv\Scripts\Activate.ps1
```

Expected: `Initialized empty Git repository in .../.git/`, then `Python 3.11.x` or newer (if older, install Python 3.11+ and create the venv with that interpreter, e.g. `python3.12 -m venv .venv`), then no output from the venv commands; the prompt shows `(.venv)`. Activate the venv in every new shell before running this plan's commands.

- [ ] **Step 2: Create the ignore, attributes and project files**

Create `.gitignore`:

```text
.venv/
__pycache__/
*.pyc
.pytest_cache/
logs/
runs/
.arena-out/
.arena-cache/
.arena-base/
.inspect-cache/
.phoenix/
calibration/free_text_tasks.jsonl
```

Create `.gitattributes`:

```text
# Hashes (dataset, prompt, rubric, scorer) are over file bytes: keep LF everywhere so Windows and CI agree.
* text=auto eol=lf
```

Create `pyproject.toml`:

```toml
[build-system]
requires = ["setuptools>=68"]
build-backend = "setuptools.build_meta"

[project]
name = "arena-evals"
version = "0.1.0"
description = "Eval harness and CI quality gate for the arena stand-in agents"
requires-python = ">=3.11"
dependencies = [
    "inspect-ai==0.3.272",
    "anthropic>=1.9",
    "numpy>=2.0",
    "scipy>=1.13",
    "pydantic>=2.7",
    "pyyaml>=6.0",
    "arize-phoenix>=20.16",
    "arize-phoenix-otel>=0.17",
    "openinference-instrumentation-anthropic>=2.1",
]

[project.optional-dependencies]
dev = ["pytest>=8"]

# pip installs the dependencies above; the code itself is imported from the repo root (pytest `pythonpath`,
# `python -m arena_evals` from the root). That way the gate's base-commit worktree imports ITS code, not HEAD's.
[tool.setuptools.packages.find]
include = ["arena_evals*"]

[tool.pytest.ini_options]
pythonpath = ["."]
testpaths = ["tests"]
markers = ["slow: Monte Carlo simulations (minutes); run with -m slow"]
addopts = "-m 'not slow'"
```

- [ ] **Step 3: Install dependencies**

Run:

```bash
python -m pip install --upgrade pip
pip install -e ".[dev]"
python -c "import inspect_ai, phoenix.otel; import importlib.metadata as m; print(m.version('inspect-ai'))"
```

Expected: pip ends with `Successfully installed ...` (the list includes `inspect-ai-0.3.272`), then the version line prints `0.3.272`. `arena_evals/` does not exist yet, so the editable install carries no code: the package is always imported from the repository root (pytest `pythonpath`, `python -m arena_evals` run from the root). Task 26 relies on this, because the gate's base-commit worktree must import its own code, not HEAD's.

- [ ] **Step 4: Write the failing tests**

Create `tests/conftest.py`:

```python
import os
import tempfile

# Keep tests away from the real Inspect response cache and quiet the progress display.
os.environ.setdefault("INSPECT_CACHE_DIR", tempfile.mkdtemp(prefix="arena-inspect-cache-"))
os.environ["INSPECT_DISPLAY"] = "none"
os.environ.pop("ARENA_TRACE", None)
```

Create `tests/test_config.py`:

```python
from pathlib import Path

import pytest

from arena_evals import config


def test_load_repo_configs():
    cfg = config.load()
    assert cfg.eval["k"] == 3 and cfg.eval["n_resamples"] == 10_000
    assert cfg.gate["eps_pts"] == 2.0 and cfg.gate["upper_q"] == 0.975
    assert cfg.cert["kappa_ci_lower_min"] == 0.4 and cfg.cert["kappa_point_min"] == 0.6
    assert cfg.models["agent"]["model"] != cfg.models["judge"]["model"]
    assert set(cfg.resolved()) == {"eval", "gate", "models", "cert"}


def test_load_rejects_missing_keys(tmp_path: Path):
    (tmp_path / "configs").mkdir()
    for name in ("eval", "gate", "models", "cert"):
        (tmp_path / "configs" / f"{name}.yaml").write_text("{}\n")
    with pytest.raises(ValueError, match="missing 'k'"):
        config.load(tmp_path)


def test_hashes(tmp_path: Path):
    (tmp_path / "a.md").write_bytes(b"alpha")
    (tmp_path / "b.md").write_bytes(b"beta")
    h = config.sha256_files(tmp_path / "a.md", tmp_path / "b.md")
    assert h == config.sha256_bytes(b"alphabeta")
    assert h == "sha256:a4c4aeb92c20500f364b12b3771ef3a11193e2cf04d0f28956a829749993b39f"
    assert h != config.sha256_files(tmp_path / "b.md", tmp_path / "a.md")
    (tmp_path / "system.md").write_bytes(b"s")
    (tmp_path / "rubric.md").write_bytes(b"r")
    assert config.rubric_hash(tmp_path) == config.sha256_bytes(b"sr")
    (tmp_path / "baseline.md").write_bytes(b"p")
    (tmp_path / "_output_contract.md").write_bytes(b"c")
    assert config.prompt_hash(tmp_path, "baseline") == config.sha256_bytes(b"pc")
```

Create `tests/test_cli.py`:

```python
import pytest

from arena_evals.__main__ import main


def test_help_lists_every_command(capsys):
    with pytest.raises(SystemExit) as e:
        main(["--help"])
    assert e.value.code == 0
    out = capsys.readouterr().out
    for cmd in ("corpus", "datasets", "generate", "run", "score", "compare", "simulate", "calibrate", "certify",
                "cache-key", "gate", "aa"):
        assert cmd in out
```

- [ ] **Step 5: Run them to verify they fail**

Run:

```bash
pytest tests/test_config.py tests/test_cli.py -v
```

Expected: FAIL, with `ModuleNotFoundError: No module named 'arena_evals'` in the output.

- [ ] **Step 6: Create the pinned configs**

Create `configs/eval.yaml`:

```yaml
# Run settings (FR5, FR7). Changing this file triggers the gate.
k: 3                      # repeats per task (epochs)
seed: 20260930            # bootstrap + sampling seed
n_resamples: 10000        # bootstrap resamples (seeded)
max_connections: 10       # parallel model calls (Inspect backs off on rate limits)
cache: true               # Inspect response cache, CachePolicy(expiry=None)
limits:
  message_limit: 40
  token_limit: 80000
  time_limit: 300         # seconds per sample
  max_tool_calls: 12
cost:
  max_usd_per_gate: 20.0  # starting value (spec I7); reset after the Task 17 measurement
  preflight_factor: 1.2
sim:
  sd: 0.45                # paired per-task SD; replaced by the Task 17 measurement
```

Create `configs/gate.yaml`:

```yaml
variant: baseline
split: gate
eps_pts: 2.0
upper_q: 0.975
judge_error_max_frac: 0.02
bh_q: 0.05
status_context: arena-eval/gate
```

Create `configs/models.yaml`:

```yaml
# Pinned models (FR7). Agent and judge are configured separately (NFR isolation).
agent:
  model: anthropic/claude-haiku-4-5-20251001
  temperature: 0.7
  seed: 1234
judge:
  model: anthropic/claude-sonnet-4-5-20250929
  temperature: 0.0
  seed: 1234
drafter:
  model: anthropic/claude-sonnet-4-5-20250929
perturber:
  model: anthropic/claude-haiku-4-5-20251001
# USD per million tokens. "default" is charged for any unlisted model so the cap never under-counts.
prices:
  anthropic/claude-haiku-4-5-20251001: {input: 1.0, output: 5.0}
  anthropic/claude-sonnet-4-5-20250929: {input: 3.0, output: 15.0}
  mockllm/model: {input: 0.0, output: 0.0}
  default: {input: 15.0, output: 75.0}
```

Create `configs/cert.yaml`:

```yaml
kappa_ci_lower_min: 0.4
kappa_point_min: 0.6
verbosity_slope_max_pts: 2.0
n_labels: 100
min_per_cell: 5
```

- [ ] **Step 7: Create the package, config loader and CLI skeleton**

Create `arena_evals/__init__.py`:

```python
"""Arena agent eval harness."""
```

Create `arena_evals/stats/__init__.py`:

```python
"""Pure statistics: no I/O."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CI:
    point: float
    lo: float
    hi: float

    def as_dict(self) -> dict:
        return {"point": self.point, "ci95": [self.lo, self.hi]}
```

Create `arena_evals/config.py`:

```python
"""Load configs/*.yaml and compute the content hashes recorded in every run."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parent.parent


@dataclass
class Config:
    root: Path
    eval: dict[str, Any]
    gate: dict[str, Any]
    models: dict[str, Any]
    cert: dict[str, Any]

    @property
    def prompts_dir(self) -> Path:
        return self.root / "prompts"

    @property
    def rubric_dir(self) -> Path:
        return self.root / "prompts" / "judge"

    def resolved(self) -> dict[str, Any]:
        """Full resolved config, written into every manifest."""
        return {"eval": self.eval, "gate": self.gate, "models": self.models, "cert": self.cert}


def load(root: Path | str = ROOT) -> Config:
    root = Path(root)

    def read(name: str) -> dict[str, Any]:
        return yaml.safe_load((root / "configs" / f"{name}.yaml").read_text(encoding="utf-8"))

    cfg = Config(root=root, eval=read("eval"), gate=read("gate"), models=read("models"), cert=read("cert"))
    for key in ("k", "seed", "n_resamples", "limits", "cost"):
        if key not in cfg.eval:
            raise ValueError(f"configs/eval.yaml missing '{key}'")
    for role in ("agent", "judge"):
        if "model" not in cfg.models.get(role, {}):
            raise ValueError(f"configs/models.yaml missing '{role}.model'")
    return cfg


def sha256_bytes(*chunks: bytes) -> str:
    h = hashlib.sha256()
    for c in chunks:
        h.update(c)
    return "sha256:" + h.hexdigest()


def sha256_files(*paths: Path) -> str:
    return sha256_bytes(*(Path(p).read_bytes() for p in paths))


def prompt_hash(prompts_dir: Path, variant: str) -> str:
    return sha256_files(prompts_dir / f"{variant}.md", prompts_dir / "_output_contract.md")


def rubric_hash(rubric_dir: Path) -> str:
    return sha256_files(rubric_dir / "system.md", rubric_dir / "rubric.md")


def scorer_hash(root: Path = ROOT) -> str:
    return sha256_files(root / "arena_evals" / "scorers.py")
```

Create `arena_evals/__main__.py`:

```python
"""CLI: python -m arena_evals <command>. Handlers import lazily so `--help` works before later modules exist."""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def _meter(cfg, max_usd: float | None):
    from arena_evals.agents import CostMeter, set_meter
    return set_meter(CostMeter(cfg.eval["cost"]["max_usd_per_gate"] if max_usd is None else max_usd,
                               cfg.models["prices"]))


def _dataset(cfg, args) -> Path:
    return Path(args.dataset) if getattr(args, "dataset", None) else cfg.root / "datasets" / f"{args.split}.jsonl"


def _summary(rows: list[dict], cfg) -> str:
    import numpy as np
    from arena_evals.run import task_means
    from arena_evals.stats.bootstrap import bootstrap_ci
    means = [v for v in task_means(rows).values() if v is not None]
    ci = bootstrap_ci(np.array(means), cfg.eval["n_resamples"], cfg.eval["seed"])
    return (f"task success {ci.point * 100:.1f}% (95% CI {ci.lo * 100:.1f} to {ci.hi * 100:.1f}), "
            f"n_tasks={len(means)}, samples={len(rows)}")


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(prog="arena_evals")
    sub = p.add_subparsers(dest="cmd", required=True)

    c = sub.add_parser("corpus").add_subparsers(dest="action", required=True)
    c.add_parser("build")

    d = sub.add_parser("datasets").add_subparsers(dest="action", required=True)
    dd = d.add_parser("draft")
    dd.add_argument("--split", required=True, choices=["gate", "calibration", "dev"])
    dd.add_argument("--model")
    ds = d.add_parser("spotcheck")
    ds.add_argument("--split", required=True, choices=["gate", "calibration", "dev"])
    ds.add_argument("--frac", type=float, default=0.2)
    d.add_parser("check")

    for name in ("generate", "run"):
        g = sub.add_parser(name)
        g.add_argument("--split", required=True)
        g.add_argument("--variant", required=True)
        g.add_argument("--dataset")
        g.add_argument("--prompts-dir")
        g.add_argument("--log-dir")
        g.add_argument("--epochs", type=int)
        g.add_argument("--limit", type=int)
        g.add_argument("--max-usd", type=float)
        g.add_argument("--no-cache", action="store_true")
    s = sub.add_parser("score")
    s.add_argument("--log", required=True)
    s.add_argument("--split", required=True)
    s.add_argument("--dataset")
    s.add_argument("--out-dir")
    s.add_argument("--max-usd", type=float)
    s.add_argument("--no-cache", action="store_true")

    cmp_ = sub.add_parser("compare")
    cmp_.add_argument("--base", required=True, help="baseline results.jsonl")
    cmp_.add_argument("--cand", required=True, help="candidate results.jsonl")

    sim = sub.add_parser("simulate")
    sim.add_argument("--sd", type=float)
    sim.add_argument("--from-results", nargs=2, metavar=("BASE", "CAND"))
    sim.add_argument("--n", type=int, default=300)
    sim.add_argument("--trials", type=int, default=1000)

    cal = sub.add_parser("calibrate").add_subparsers(dest="action", required=True)
    ce = cal.add_parser("export")
    ce.add_argument("--max-usd", type=float)
    cl = cal.add_parser("label")
    cl.add_argument("--labeler")
    cert = sub.add_parser("certify")
    cert.add_argument("--max-usd", type=float)

    ck = sub.add_parser("cache-key")
    ck.add_argument("--pr", type=int, required=True)
    gt = sub.add_parser("gate")
    gt.add_argument("--pr", type=int, required=True)
    gt.add_argument("--rerun-reason", default="")
    aa = sub.add_parser("aa")
    aa.add_argument("--runs", type=int, default=20)
    aa.add_argument("--split", default="gate")

    args = p.parse_args(argv)
    from arena_evals import config
    cfg = config.load()

    if args.cmd == "corpus":
        from arena_evals import corpus
        print(f"wrote {len(corpus.build())} docs to {corpus.DOCS_DIR}")
        return 0

    if args.cmd == "datasets":
        from arena_evals import corpus, datasets
        if args.action == "check":
            return datasets.check(cfg.root)
        if args.action == "draft":
            _meter(cfg, None)
            recs = datasets.draft(args.split, corpus.load(), args.model or cfg.models["drafter"]["model"])
            print(f"wrote {len(recs)} drafts to "
                  f"{datasets.write_jsonl(cfg.root / 'datasets' / 'drafts' / f'{args.split}.jsonl', recs)}")
            return 0
        print(f"wrote {datasets.spotcheck(args.split, args.frac)}")
        floor = datasets.label_noise_floor(args.split)
        print(f"label-noise floor {floor['point']:.1%} (95% CI {floor['ci95'][0]:.1%} to {floor['ci95'][1]:.1%}, "
              f"n={floor['n']})")
        return 0

    if args.cmd in ("generate", "run"):
        from arena_evals import run
        from arena_evals.agents import CostCapExceeded
        _meter(cfg, args.max_usd)
        t0 = time.monotonic()
        try:
            log = run.generate(args.split, args.variant, Path(args.prompts_dir) if args.prompts_dir else cfg.prompts_dir,
                               _dataset(cfg, args), cfg, log_dir=Path(args.log_dir) if args.log_dir else None,
                               epochs=args.epochs, limit=args.limit, cache=False if args.no_cache else None)
            if args.cmd == "generate":
                print(log)
                return 0
            out = run.score(log, _dataset(cfg, args), cfg, cache=False if args.no_cache else None)
        except CostCapExceeded as e:
            print(f"error: {e}", file=sys.stderr)
            return 3
        rows = run.read_results(out.results_path)
        print(_summary(rows, cfg))
        print(f"cost ${out.manifest['cost_usd']:.2f}, wall time {time.monotonic() - t0:.0f}s, "
              f"judge errors {out.manifest['judge_error_count']}")
        print(out.results_path)
        return 0

    if args.cmd == "score":
        from arena_evals import run
        _meter(cfg, args.max_usd)
        out = run.score(Path(args.log), _dataset(cfg, args), cfg, out_dir=Path(args.out_dir) if args.out_dir else None,
                        cache=False if args.no_cache else None)
        print(_summary(run.read_results(out.results_path), cfg))
        print(out.results_path)
        return 0

    if args.cmd == "compare":
        from arena_evals import run
        from arena_evals.stats.bootstrap import gate_decision, paired_bootstrap
        base, cand = run.read_results(Path(args.base)), run.read_results(Path(args.cand))
        ids, d, dropped = run.paired_deltas(base, cand)
        r = paired_bootstrap(d, cfg.eval["n_resamples"], cfg.eval["seed"])
        print(f"base: {_summary(base, cfg)}\ncand: {_summary(cand, cfg)}")
        print(f"delta {r.delta * 100:+.2f} pts (95% CI {r.ci95[0] * 100:+.2f} to {r.ci95[1] * 100:+.2f}; "
              f"one-sided 97.5% upper {r.upper_975 * 100:+.2f}), P(delta<0) {r.p_neg:.3f}, n={r.n}, "
              f"dropped={len(dropped)}, SD(d)={r.sd:.3f}, MDE {r.mde * 100:.1f} pts")
        print(f"verdict: {gate_decision(r, cfg.gate['eps_pts'], cfg.gate['upper_q'])}")
        return 0

    if args.cmd == "simulate":
        from arena_evals.stats.simulate import simulate_coverage, simulate_gate
        sd = args.sd if args.sd is not None else cfg.eval["sim"]["sd"]
        if args.from_results:
            from arena_evals import run
            _, d, _ = run.paired_deltas(*(run.read_results(Path(x)) for x in args.from_results))
            sd = float(d.std(ddof=1))
        print(f"SD={sd:.3f} n={args.n} trials={args.trials} eps={cfg.gate['eps_pts']} pts")
        for delta in (-8.0, -7.0, -6.0, -5.0, 0.0):
            rate = simulate_gate(delta, sd, args.n, args.trials, 2000, cfg.gate["eps_pts"], cfg.eval["seed"])
            print(f"true delta {delta:+.0f} pts: blocked {rate:.1%}")
        print(f"95% CI coverage at -5 pts: {simulate_coverage(-5.0, sd, args.n, args.trials, cfg.eval['seed']):.1%}")
        return 0

    if args.cmd == "calibrate":
        from arena_evals import calibration
        if args.action == "export":
            _meter(cfg, args.max_usd)
            print(f"wrote {calibration.export(cfg)}")
            return 0
        print(f"labels in {calibration.label_cli(cfg.root / 'calibration' / 'to_label.jsonl', labeler=args.labeler)}")
        return 0

    if args.cmd == "certify":
        from arena_evals import ci
        _meter(cfg, args.max_usd)
        return ci.certify(cfg)

    if args.cmd == "cache-key":
        from arena_evals import ci, datasets
        from arena_evals.config import rubric_hash, scorer_hash
        base_sha = ci.GitHub().pr(args.pr)["base"]["sha"]
        ds = cfg.root / "datasets" / f"{cfg.gate['split']}.jsonl"
        print(ci.cache_key(base_sha, cfg, datasets.dataset_hash(ds), rubric_hash(cfg.rubric_dir), scorer_hash(cfg.root)))
        return 0

    if args.cmd == "gate":
        from arena_evals import ci
        return ci.gate(args.pr, cfg, rerun_reason=args.rerun_reason)

    if args.cmd == "aa":
        from arena_evals import ci
        return ci.aa(args.runs, cfg, args.split)
    return 2


if __name__ == "__main__":
    sys.exit(main())
```

`__main__.py` is complete from the start. Each command imports its module inside its own branch, so `--help` works now and each command starts working in the task that creates its module.

- [ ] **Step 8: Run the tests to verify they pass**

Run:

```bash
pytest tests/test_config.py tests/test_cli.py -v
```

Expected: PASS (`4 passed`).

- [ ] **Step 9: Commit**

Run:

```bash
git add .gitignore .gitattributes pyproject.toml PRD.md docs configs arena_evals tests
git commit -m "chore: scaffold arena-evals with pinned configs, config loader, CLI"
```

### Task 2: Halcyon Robotics corpus: seed, deterministic build, loader, BM25 index

**Files:**
- Create: `corpus/seed.yaml`
- Create: `arena_evals/corpus.py`
- Create: `corpus/docs/*.md (generated, 40 files)`
- Test: `tests/test_corpus.py`

**Interfaces:**
- Consumes: `config.ROOT`
- Produces: `corpus.build(seed_path=SEED_PATH, out_dir=DOCS_DIR) -> list[Path]`
- Produces: `corpus.load(docs_dir=DOCS_DIR) -> dict[str, Doc]` (`Doc`: `id`, `title`, `kind`, `effective_date`, `supersedes`, `text`, `body`)
- Produces: `corpus.Index(docs).search(query, k=5) -> list[Hit]` (`Hit`: `id`, `title`, `snippet`, `score`)
- Produces: `corpus.current_doc(fact_key, seed_path=SEED_PATH) -> str`, `corpus.tokenize(text) -> list[str]`, `corpus.SEED_PATH`, `corpus.DOCS_DIR`

- [ ] **Step 1: Write the failing test**

Create `tests/test_corpus.py`:

```python
from pathlib import Path

from arena_evals import corpus

CONFLICTS = {  # fact key -> (superseded doc, currently effective doc)
    "parental_leave_weeks": ("HR-002", "HR-009"),
    "remote_stipend": ("HR-003", "HR-010"),
    "c2_battery_hours": ("PRD-002", "PRD-008"),
    "pro_price_per_seat": ("PRC-001", "PRC-006"),
    "head_of_support": ("ORG-002", "ORG-004"),
}


def test_rebuild_is_byte_identical(tmp_path: Path):
    corpus.build(corpus.SEED_PATH, tmp_path)
    committed = {p.name: p.read_bytes() for p in corpus.DOCS_DIR.glob("*.md")}
    rebuilt = {p.name: p.read_bytes() for p in tmp_path.glob("*.md")}
    assert rebuilt == committed


def test_forty_docs_with_kind_counts():
    docs = corpus.load()
    assert len(docs) == 40
    kinds = [d.kind for d in docs.values()]
    assert (kinds.count("hr"), kinds.count("product"), kinds.count("pricing"),
            kinds.count("incident"), kinds.count("org")) == (10, 9, 7, 10, 4)


def test_five_conflict_pairs_resolve_to_current_doc():
    docs = corpus.load()
    assert sum(1 for d in docs.values() if d.supersedes) == 5
    for key, (old, new) in CONFLICTS.items():
        assert docs[new].supersedes == old
        assert corpus.current_doc(key) == new


def test_search_format_and_empty():
    idx = corpus.Index(corpus.load())
    hits = idx.search("parental leave weeks")
    assert {h.id for h in hits[:2]} == {"HR-002", "HR-009"}
    assert all(len(h.snippet) <= 200 for h in hits)
    assert "weeks of fully paid parental leave" in hits[0].snippet
    assert idx.search("zzzz qqqq") == []


def test_search_ties_broken_by_doc_id():
    body = "The Dock charges robots."
    docs = {i: corpus.Doc(i, "t", "product", "2024-01-01", None, body, body) for i in ("PRD-777", "PRD-111")}
    assert [h.id for h in corpus.Index(docs).search("dock charges")] == ["PRD-111", "PRD-777"]
```

- [ ] **Step 2: Run it to verify it fails**

Run:

```bash
pytest tests/test_corpus.py -v
```

Expected: FAIL, with `ImportError: cannot import name 'corpus' from 'arena_evals'` in the output.

- [ ] **Step 3: Create the seed**

Every fact lives here. Fact keys are unique across docs except the 5 conflict pairs, where the newer doc (with `supersedes`) restates the key with a new value. The corpus as-of date is 2026-09-01.

Create `corpus/seed.yaml`:

```yaml
# Every fact in the Halcyon Robotics corpus. `python -m arena_evals corpus build` renders corpus/docs/*.md.
# Fact keys are unique across docs EXCEPT the 5 conflict pairs, where the newer doc (with `supersedes`)
# restates the same key with a new value.
company: Halcyon Robotics
as_of: "2026-09-01"
docs:
  # ---------------- HR policy (10) ----------------
  - id: HR-001
    title: Paid Time Off Policy
    effective_date: "2024-01-01"
    facts:
      - {key: pto_days, value: "20 days", text: "Full-time employees accrue {value} of paid time off per calendar year."}
      - {key: pto_carryover, value: "5 days", text: "Up to {value} of unused paid time off may be carried into the next year."}
    refs: [HR-008]
  - id: HR-002
    title: Parental Leave Policy (2023)
    effective_date: "2023-04-01"
    facts:
      - {key: parental_leave_weeks, value: "12 weeks", text: "Employees with a new child receive {value} of fully paid parental leave."}
      - {key: parental_leave_eligibility, value: "6 months", text: "Parental leave is available after {value} of continuous employment."}
    refs: [HR-001]
  - id: HR-003
    title: Remote Work Policy (2023)
    effective_date: "2023-03-01"
    facts:
      - {key: remote_stipend, value: "500 USD", text: "Remote employees receive a one-time home office stipend of {value}."}
      - {key: remote_office_days, value: "2 days", text: "Hybrid employees are expected in a Halcyon office {value} per week."}
    refs: [HR-005]
  - id: HR-004
    title: Travel and Expenses Policy
    effective_date: "2024-02-15"
    facts:
      - {key: per_diem, value: "75 USD", text: "The meal per diem for business travel is {value} per day."}
      - {key: hotel_cap, value: "220 USD", text: "Hotel bookings are capped at {value} per night without VP approval."}
    refs: [ORG-001]
  - id: HR-005
    title: Equipment Policy
    effective_date: "2024-05-01"
    facts:
      - {key: laptop_refresh, value: "36 months", text: "Company laptops are refreshed every {value}."}
      - {key: equipment_budget, value: "2,400 USD", text: "Each new hire has an equipment budget of {value}."}
    refs: [HR-003]
  - id: HR-006
    title: Learning and Development Budget
    effective_date: "2024-01-01"
    facts:
      - {key: learning_budget, value: "1,500 USD", text: "Each employee has an annual learning budget of {value}."}
      - {key: conference_limit, value: "2 conferences", text: "Employees may attend at most {value} per year on the learning budget."}
    refs: [HR-008]
  - id: HR-007
    title: On-Call Compensation
    effective_date: "2025-01-01"
    facts:
      - {key: oncall_weekend_pay, value: "150 USD", text: "Engineers are paid {value} for each weekend on-call shift."}
      - {key: oncall_rotation, value: "6 engineers", text: "Each on-call rotation has at least {value}."}
    refs: [ORG-003, INC-001]
  - id: HR-008
    title: Performance Review Cycle
    effective_date: "2024-01-01"
    facts:
      - {key: review_months, value: "March and September", text: "Performance reviews are held twice a year, in {value}."}
      - {key: review_raise_cap, value: "8%", text: "Merit raises are capped at {value} per review cycle."}
    refs: [HR-006]
  - id: HR-009
    title: Parental Leave Policy (2025)
    effective_date: "2025-07-01"
    supersedes: HR-002
    facts:
      - {key: parental_leave_weeks, value: "16 weeks", text: "Employees with a new child receive {value} of fully paid parental leave."}
    refs: [HR-001]
  - id: HR-010
    title: Remote Work Policy (2025)
    effective_date: "2025-03-01"
    supersedes: HR-003
    facts:
      - {key: remote_stipend, value: "750 USD", text: "Remote employees receive a one-time home office stipend of {value}."}
    refs: [HR-005]
  # ---------------- Product specs (9) ----------------
  - id: PRD-001
    title: Courier C1 Delivery Robot
    effective_date: "2023-09-01"
    facts:
      - {key: c1_payload, value: "25 kg", text: "The Courier C1 carries a maximum payload of {value}."}
      - {key: c1_speed, value: "6 km/h", text: "The Courier C1 has a top speed of {value}."}
      - {key: c1_battery_hours, value: "10 hours", text: "A full charge runs the Courier C1 for {value}."}
    refs: [PRD-005, PRC-004]
  - id: PRD-002
    title: Courier C2 Delivery Robot (launch spec)
    effective_date: "2024-06-01"
    facts:
      - {key: c2_battery_hours, value: "14 hours", text: "A full charge runs the Courier C2 for {value}."}
      - {key: c2_payload, value: "40 kg", text: "The Courier C2 carries a maximum payload of {value}."}
    refs: [PRD-001]
  - id: PRD-003
    title: Sentinel S1 Security Robot
    effective_date: "2023-11-01"
    facts:
      - {key: s1_camera_count, value: "6 cameras", text: "The Sentinel S1 has {value} covering 360 degrees."}
      - {key: s1_patrol_range, value: "8 km", text: "The Sentinel S1 patrols up to {value} per shift."}
    refs: [PRD-006]
  - id: PRD-004
    title: Harvest H1 Agricultural Robot
    effective_date: "2024-03-01"
    facts:
      - {key: h1_rows_per_hour, value: "12 rows", text: "The Harvest H1 picks {value} of strawberries per hour."}
      - {key: h1_weight, value: "180 kg", text: "The Harvest H1 weighs {value}."}
    refs: [PRD-005]
  - id: PRD-005
    title: Dock Charging Station
    effective_date: "2023-09-01"
    facts:
      - {key: dock_charge_time, value: "90 minutes", text: "The Dock charges any Halcyon robot from empty to full in {value}."}
      - {key: dock_capacity, value: "4 robots", text: "One Dock serves up to {value}."}
    refs: [PRD-001, PRD-004]
  - id: PRD-006
    title: Fleet Console Software
    effective_date: "2024-01-15"
    facts:
      - {key: console_max_robots, value: "500 robots", text: "A single Fleet Console workspace manages up to {value}."}
      - {key: console_uptime_sla, value: "99.9%", text: "Fleet Console has a monthly uptime SLA of {value}."}
    refs: [PRC-001, PRC-002]
  - id: PRD-007
    title: Sentinel S2 Security Robot
    effective_date: "2025-05-01"
    facts:
      - {key: s2_camera_count, value: "8 cameras", text: "The Sentinel S2 has {value} covering 360 degrees."}
      - {key: s2_patrol_range, value: "12 km", text: "The Sentinel S2 patrols up to {value} per shift."}
    refs: [PRD-003]
  - id: PRD-008
    title: Courier C2 Delivery Robot (revised spec)
    effective_date: "2025-06-01"
    supersedes: PRD-002
    facts:
      - {key: c2_battery_hours, value: "12 hours", text: "A full charge runs the Courier C2 for {value}."}
    refs: [PRD-002, INC-004]
  - id: PRD-009
    title: Lift L1 Warehouse Robot
    effective_date: "2025-02-01"
    facts:
      - {key: l1_lift_capacity, value: "600 kg", text: "The Lift L1 lifts pallets of up to {value}."}
      - {key: l1_shelf_height, value: "3.2 m", text: "The Lift L1 reaches shelves up to {value} high."}
    refs: [PRD-005, PRC-004]
  # ---------------- Pricing (7) ----------------
  - id: PRC-001
    title: Fleet Console Pricing (2024)
    effective_date: "2024-01-15"
    facts:
      - {key: pro_price_per_seat, value: "199 USD", text: "The Fleet Console Pro tier costs {value} per seat per month."}
      - {key: pro_min_seats, value: "5 seats", text: "The Pro tier requires a minimum of {value}."}
    refs: [PRD-006]
  - id: PRC-002
    title: Fleet Console Starter Tier
    effective_date: "2024-01-15"
    facts:
      - {key: starter_price_per_seat, value: "49 USD", text: "The Fleet Console Starter tier costs {value} per seat per month."}
      - {key: starter_max_seats, value: "10 seats", text: "The Starter tier is limited to {value}."}
    refs: [PRD-006]
  - id: PRC-003
    title: Fleet Console Enterprise Tier
    effective_date: "2024-01-15"
    facts:
      - {key: enterprise_price_per_seat, value: "159 USD", text: "The Fleet Console Enterprise tier costs {value} per seat per month."}
      - {key: enterprise_min_seats, value: "50 seats", text: "The Enterprise tier requires a minimum of {value}."}
    refs: [PRC-001, PRC-005]
  - id: PRC-004
    title: Robot Hardware Leasing
    effective_date: "2024-04-01"
    facts:
      - {key: c1_lease, value: "890 USD", text: "Leasing a Courier C1 costs {value} per month."}
      - {key: l1_lease, value: "2,150 USD", text: "Leasing a Lift L1 costs {value} per month."}
      - {key: lease_term, value: "24 months", text: "The minimum lease term is {value}."}
    refs: [PRD-001, PRD-009, PRC-007]
  - id: PRC-005
    title: Support Plans
    effective_date: "2024-01-15"
    facts:
      - {key: premium_support_rate, value: "12%", text: "Premium support costs {value} of the annual contract value."}
      - {key: premium_response_time, value: "1 hour", text: "Premium support guarantees a first response within {value}."}
    refs: [ORG-002]
  - id: PRC-006
    title: Fleet Console Pricing (2025)
    effective_date: "2025-01-01"
    supersedes: PRC-001
    facts:
      - {key: pro_price_per_seat, value: "179 USD", text: "The Fleet Console Pro tier costs {value} per seat per month."}
    refs: [PRD-006, PRC-003]
  - id: PRC-007
    title: Volume Discounts
    effective_date: "2024-04-01"
    facts:
      - {key: volume_discount, value: "10%", text: "Orders of more than 100 robots receive a {value} discount on hardware."}
      - {key: volume_discount_large, value: "15%", text: "Orders of more than 500 robots receive a {value} discount on hardware."}
    refs: [PRC-004]
  # ---------------- Incident postmortems (10) ----------------
  - id: INC-001
    title: "Postmortem: Fleet Console Outage"
    effective_date: "2024-02-20"
    facts:
      - {key: inc001_duration, value: "47 minutes", text: "Fleet Console was unavailable for {value}."}
      - {key: inc001_cause, value: "an expired TLS certificate", text: "The root cause was {value}."}
    refs: [PRD-006, HR-007]
  - id: INC-002
    title: "Postmortem: Courier C1 Navigation Fault"
    effective_date: "2024-04-11"
    facts:
      - {key: inc002_robots, value: "31 robots", text: "The navigation fault affected {value} in the Leeds depot."}
      - {key: inc002_cause, value: "a map tile corruption", text: "The root cause was {value}."}
    refs: [PRD-001]
  - id: INC-003
    title: "Postmortem: Dock Overheating"
    effective_date: "2024-07-02"
    facts:
      - {key: inc003_docks, value: "12 docks", text: "Thermal shutdown hit {value} during the heatwave."}
      - {key: inc003_fix, value: "a firmware fan curve update", text: "The fix was {value}."}
    refs: [PRD-005]
  - id: INC-004
    title: "Postmortem: Courier C2 Battery Shortfall"
    effective_date: "2025-05-15"
    facts:
      - {key: inc004_measured_hours, value: "12.3 hours", text: "Field tests measured an average Courier C2 runtime of {value}."}
      - {key: inc004_customers, value: "9 customers", text: "The shortfall was reported by {value}."}
    refs: [PRD-002, PRD-008]
  - id: INC-005
    title: "Postmortem: Billing Double Charge"
    effective_date: "2024-09-03"
    facts:
      - {key: inc005_invoices, value: "214 invoices", text: "A retry bug double-charged {value}."}
      - {key: inc005_refund_days, value: "3 days", text: "All affected customers were refunded within {value}."}
    refs: [PRC-001]
  - id: INC-006
    title: "Postmortem: Sentinel False Alarms"
    effective_date: "2024-11-19"
    facts:
      - {key: inc006_alarms, value: "1,340 alarms", text: "Sentinel S1 units raised {value} over one weekend."}
      - {key: inc006_cause, value: "a motion model trained on summer footage", text: "The root cause was {value}."}
    refs: [PRD-003]
  - id: INC-007
    title: "Postmortem: Harvest H1 Crop Damage"
    effective_date: "2025-03-08"
    facts:
      - {key: inc007_loss, value: "18,000 USD", text: "Gripper misalignment caused crop losses of {value}."}
      - {key: inc007_units, value: "7 units", text: "The fault was found in {value}."}
    refs: [PRD-004]
  - id: INC-008
    title: "Postmortem: Console Data Export Delay"
    effective_date: "2025-06-22"
    facts:
      - {key: inc008_delay, value: "6 hours", text: "Nightly data exports were delayed by {value}."}
      - {key: inc008_cause, value: "a full disk on the export worker", text: "The root cause was {value}."}
    refs: [PRD-006]
  - id: INC-009
    title: "Postmortem: Lift L1 Emergency Stop"
    effective_date: "2025-08-30"
    facts:
      - {key: inc009_site, value: "the Rotterdam warehouse", text: "An emergency stop cascade halted operations at {value}."}
      - {key: inc009_duration, value: "2 hours", text: "Operations were halted for {value}."}
    refs: [PRD-009]
  - id: INC-010
    title: "Postmortem: Support Queue Backlog"
    effective_date: "2026-01-12"
    facts:
      - {key: inc010_tickets, value: "860 tickets", text: "The support backlog peaked at {value}."}
      - {key: inc010_breaches, value: "41 premium SLA breaches", text: "The backlog caused {value}."}
    refs: [PRC-005, ORG-004]
  # ---------------- Org chart (4) ----------------
  - id: ORG-001
    title: Executive Team
    effective_date: "2024-01-01"
    facts:
      - {key: ceo, value: "Mara Lindqvist", text: "The chief executive officer is {value}."}
      - {key: cto, value: "Dev Okafor", text: "The chief technology officer is {value}."}
      - {key: headcount, value: "412 employees", text: "Halcyon Robotics has {value}."}
    refs: [ORG-003]
  - id: ORG-002
    title: Customer Support Organization (2024)
    effective_date: "2024-01-01"
    facts:
      - {key: head_of_support, value: "Priya Raman", text: "Customer Support is led by {value}."}
      - {key: support_regions, value: "3 regions", text: "Support is staffed across {value}: EMEA, Americas and APAC."}
    refs: [PRC-005]
  - id: ORG-003
    title: Engineering Organization
    effective_date: "2024-06-01"
    facts:
      - {key: eng_teams, value: "14 teams", text: "Engineering is organized into {value}."}
      - {key: eng_headcount, value: "168 engineers", text: "Engineering employs {value}."}
    refs: [ORG-001, HR-007]
  - id: ORG-004
    title: Customer Support Organization (2026)
    effective_date: "2026-02-01"
    supersedes: ORG-002
    facts:
      - {key: head_of_support, value: "Tomasz Kowal", text: "Customer Support is led by {value}."}
    refs: [PRC-005, INC-010]
```

- [ ] **Step 4: Create the corpus module**

Create `arena_evals/corpus.py`:

```python
"""Halcyon Robotics corpus: deterministic render from seed.yaml, loader, stdlib BM25 index."""
from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import yaml

from arena_evals.config import ROOT

SEED_PATH = ROOT / "corpus" / "seed.yaml"
DOCS_DIR = ROOT / "corpus" / "docs"
KINDS = {"HR": "hr", "PRD": "product", "PRC": "pricing", "INC": "incident", "ORG": "org"}
KIND_LABELS = {"hr": "HR policy", "product": "Product specification", "pricing": "Pricing",
               "incident": "Incident postmortem", "org": "Org chart"}
_TOKEN = re.compile(r"[a-z0-9]+")


@dataclass(frozen=True)
class Doc:
    id: str
    title: str
    kind: str
    effective_date: str
    supersedes: str | None
    text: str          # full markdown incl. front matter (what get_doc returns)
    body: str          # markdown after the front matter (what search indexes)


@dataclass(frozen=True)
class Hit:
    id: str
    title: str
    snippet: str
    score: float


def _render(seed: dict, d: dict) -> str:
    kind = KINDS[d["id"].split("-")[0]]
    lines = ["---", f"id: {d['id']}", f"title: \"{d['title']}\"", f"kind: {kind}",
             f"effective_date: {d['effective_date']}"]
    if d.get("supersedes"):
        lines.append(f"supersedes: {d['supersedes']}")
    lines += ["---", "", f"# {d['title']}", "",
              f"{seed['company']} {KIND_LABELS[kind]}. Document {d['id']}, effective {d['effective_date']}."]
    if d.get("supersedes"):
        lines += ["", f"This document supersedes {d['supersedes']} from its effective date."]
    for f in d["facts"]:
        lines += ["", f["text"].format(value=f["value"])]
    if d.get("refs"):
        lines += ["", "Related documents: " + ", ".join(d["refs"]) + "."]
    return "\n".join(lines) + "\n"


def build(seed_path: Path = SEED_PATH, out_dir: Path = DOCS_DIR) -> list[Path]:
    seed = yaml.safe_load(Path(seed_path).read_text(encoding="utf-8"))
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for old in out_dir.glob("*.md"):
        old.unlink()
    paths = []
    for d in seed["docs"]:
        p = out_dir / f"{d['id']}.md"
        p.write_text(_render(seed, d), encoding="utf-8", newline="\n")
        paths.append(p)
    return paths


def load(docs_dir: Path = DOCS_DIR) -> dict[str, Doc]:
    docs = {}
    for p in sorted(Path(docs_dir).glob("*.md")):
        text = p.read_text(encoding="utf-8")
        _, front, body = text.split("---\n", 2)
        meta = yaml.safe_load(front)
        docs[meta["id"]] = Doc(id=meta["id"], title=meta["title"], kind=meta["kind"],
                               effective_date=str(meta["effective_date"]), supersedes=meta.get("supersedes"),
                               text=text, body=body.strip())
    return docs


def current_doc(fact_key: str, seed_path: Path = SEED_PATH) -> str:
    """ID of the doc whose value for fact_key is in force at the corpus as-of date."""
    seed = yaml.safe_load(Path(seed_path).read_text(encoding="utf-8"))
    as_of = seed["as_of"]
    cands = [d for d in seed["docs"] if any(f["key"] == fact_key for f in d["facts"]) and d["effective_date"] <= as_of]
    if not cands:
        raise KeyError(f"no effective doc states {fact_key!r}")
    return max(cands, key=lambda d: d["effective_date"])["id"]


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Index:
    """Okapi BM25 over doc bodies (k1=1.5, b=0.75)."""

    def __init__(self, docs: dict[str, Doc], k1: float = 1.5, b: float = 0.75):
        self.docs = docs
        self.k1, self.b = k1, b
        self.tf = {i: Counter(tokenize(d.body)) for i, d in docs.items()}
        self.len = {i: sum(c.values()) for i, c in self.tf.items()}
        self.avgdl = sum(self.len.values()) / max(len(self.len), 1)
        df = Counter(t for c in self.tf.values() for t in c)
        n = len(docs)
        self.idf = {t: math.log(1 + (n - f + 0.5) / (f + 0.5)) for t, f in df.items()}

    def _score(self, doc_id: str, terms: list[str]) -> float:
        tf, dl = self.tf[doc_id], self.len[doc_id]
        s = 0.0
        for t in terms:
            if t in tf:
                f = tf[t]
                s += self.idf[t] * f * (self.k1 + 1) / (f + self.k1 * (1 - self.b + self.b * dl / self.avgdl))
        return s

    def _snippet(self, doc_id: str, terms: set[str]) -> str:
        paras = [p.strip() for p in self.docs[doc_id].body.split("\n\n") if p.strip()]
        best = max(paras, key=lambda p: len(terms & set(tokenize(p))))  # max() keeps the first on ties
        return " ".join(best.split())[:200]

    def search(self, query: str, k: int = 5) -> list[Hit]:
        terms = tokenize(query)
        scored = [(self._score(i, terms), i) for i in self.docs]
        scored = [(s, i) for s, i in scored if s > 0]
        scored.sort(key=lambda si: (-si[0], si[1]))  # ties broken by doc ID
        return [Hit(i, self.docs[i].title, self._snippet(i, set(terms)), s) for s, i in scored[:k]]
```

- [ ] **Step 5: Build the docs**

Run:

```bash
python -m arena_evals corpus build
```

Expected output:

```text
wrote 40 docs to <repo>/corpus/docs
```

- [ ] **Step 6: Run the tests to verify they pass**

Run:

```bash
pytest tests/test_corpus.py -v
```

Expected: PASS (`5 passed`).

- [ ] **Step 7: Commit**

Run:

```bash
git add corpus arena_evals/corpus.py tests/test_corpus.py
git commit -m "feat(corpus): deterministic Halcyon corpus with 5 conflict pairs and BM25 index"
```

### Task 3: Tools, safe calculator and cost meter

**Files:**
- Create: `arena_evals/agents.py`
- Test: `tests/test_calculator.py`
- Test: `tests/test_agents.py`

**Interfaces:**
- Consumes: `corpus.Index`, `corpus.load`, `config.ROOT`
- Produces: `agents.safe_eval(expr: str) -> float` (raises `ValueError`), `agents.format_number(x: float) -> str`
- Produces: `agents.usage_cost(usage, model: str, prices: dict) -> float`
- Produces: `agents.CostCapExceeded(RuntimeError)`, `agents.CostMeter(cap_usd: float, prices: dict | None = None)` with `.charge(usage, model) -> float`, `.check() -> None`, `.spent`, `.prices`
- Produces: `agents.METER` (process-wide meter), `agents.set_meter(meter) -> CostMeter`, `agents.CostHook` (Inspect hook charging every model call)
- Produces: Inspect tools `agents.search_docs()`, `agents.get_doc()`, `agents.calculate()`; `agents._index() -> corpus.Index`
- Produces: Always read the meter as `agents.METER` (module attribute), never `from arena_evals.agents import METER`: `set_meter` rebinds it.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_calculator.py`:

```python
import pytest

from arena_evals.agents import format_number, safe_eval


@pytest.mark.parametrize("expr, expected", [
    ("179 * 25", 4475.0),
    ("(199 - 179) * 12", 240.0),
    ("-3 + 10 / 4", -0.5),
    ("7 // 2 + 7 % 2", 4.0),
    ("2 ** 10", 1024.0),
    ("1.5 * 2", 3.0),
    ("-(2 - 5)", 3.0),
])
def test_arithmetic(expr, expected):
    assert safe_eval(expr) == pytest.approx(expected)


@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo pwned')",
    "abs(-1)",
    "x + 1",
    "(1).__class__",
    "[1, 2]",
    "'a' * 3",
    "True + 1",
    "2 ** 101",
    "10 ** 16",
    "1e309",
    "1 / 0",
    "5 % 0",
    "(-8) ** 0.5",
    "1 +",
    "lambda: 1",
    "1" + "+1" * 300,
])
def test_rejects_unsafe_or_invalid(expr):
    with pytest.raises(ValueError):
        safe_eval(expr)


def test_format_number():
    assert format_number(4475.0) == "4475"
    assert format_number(123456789012.0) == "123456789012"
    assert format_number(2 / 3) == "0.6666666667"
```

Create `tests/test_agents.py`:

```python
import asyncio
from types import SimpleNamespace

import pytest

from arena_evals.agents import CostCapExceeded, CostMeter, calculate, get_doc, search_docs, set_meter


def run(coro):
    return asyncio.run(coro)


@pytest.fixture(autouse=True)
def unlimited_meter():
    set_meter(CostMeter(float("inf")))
    yield
    set_meter(CostMeter(float("inf")))


def test_search_docs_format():
    out = run(search_docs()(query="Pro tier price per seat"))
    first = out.splitlines()[0]
    doc_id, title, snippet = first.split(" | ", 2)
    assert doc_id == "PRC-001" and title == "Fleet Console Pricing (2024)"
    assert len(out.splitlines()) == 5


def test_search_docs_empty_result_is_tool_output_not_exception():
    assert run(search_docs()(query="")) == "no results for query: "
    assert run(search_docs()(query="xylophone quasar")) == "no results for query: xylophone quasar"


def test_get_doc_known_and_unknown():
    assert run(get_doc()(id="PRC-006")).startswith("---\nid: PRC-006\n")
    assert run(get_doc()(id="HR-999")) == "error: no document with id HR-999"


def test_calculate_tool_returns_errors_as_output():
    assert run(calculate()(expr="179 * 25")) == "4475"
    assert run(calculate()(expr="__import__('os')")) == "error: unsupported element: Call"


def test_cost_meter_charges_and_caps():
    m = CostMeter(0.01, {"m": {"input": 1.0, "output": 5.0}, "default": {"input": 15.0, "output": 75.0}})
    assert m.charge(SimpleNamespace(input_tokens=1000, output_tokens=1000), "m") == pytest.approx(0.006)
    m.check()
    m.charge(SimpleNamespace(input_tokens=1000, output_tokens=0), "unknown-model")  # default price: 0.015
    with pytest.raises(CostCapExceeded, match=r"cost cap hit \(\$0.02 of \$0.01\)"):
        m.check()


def test_tools_refuse_after_cap():
    set_meter(CostMeter(0.0))
    with pytest.raises(CostCapExceeded):
        run(get_doc()(id="HR-001"))
```

- [ ] **Step 2: Run them to verify they fail**

Run:

```bash
pytest tests/test_calculator.py tests/test_agents.py -v
```

Expected: FAIL, with `ModuleNotFoundError: No module named 'arena_evals.agents'` in the output.

- [ ] **Step 3: Create `agents.py` with the calculator, cost meter and tools**

The cost cap cannot be enforced from an Inspect hook: an exception raised inside a hook is swallowed (verified in Task 8). So `CostHook` only charges, and `METER.check()` is called at the start of every tool call, every sample and every judge call, which raises `CostCapExceeded` into the sample.

Create `arena_evals/agents.py`:

```python
"""Arena stand-in agent: tools, safe calculator, FINAL-line parser, cost meter, solver factory."""
from __future__ import annotations

import ast
import functools
import math
import operator
from pathlib import Path

from inspect_ai.hooks import Hooks, ModelUsageData, hooks
from inspect_ai.model import CachePolicy, ChatMessageSystem
from inspect_ai.solver import Generate, Solver, TaskState, generate, solver, use_tools
from inspect_ai.tool import Tool, tool
from pydantic import BaseModel, ConfigDict, ValidationError

from arena_evals import corpus
from arena_evals.config import ROOT

PROMPTS_DIR = ROOT / "prompts"

# ---------------------------------------------------------------- safe calculator
_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
MAX_ABS_RESULT = 1e15
MAX_ABS_EXPONENT = 100
MAX_EXPR_CHARS = 500


def safe_eval(expr: str) -> float:
    """Evaluate +, -, *, /, //, %, **, unary minus, parentheses over numeric literals. Raises ValueError."""
    if len(expr) > MAX_EXPR_CHARS:
        raise ValueError(f"expression longer than {MAX_EXPR_CHARS} characters")
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        raise ValueError(f"invalid syntax: {e.msg}") from None

    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            v = node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            v = -ev(node.operand)
        elif isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > MAX_ABS_EXPONENT:
                raise ValueError(f"exponent larger than {MAX_ABS_EXPONENT}")
            try:
                v = _BINOPS[type(node.op)](left, right)
            except ZeroDivisionError:
                raise ValueError("division by zero") from None
            except OverflowError:
                raise ValueError("result too large") from None
        else:
            raise ValueError(f"unsupported element: {type(node).__name__}")
        if isinstance(v, complex) or not math.isfinite(v) or abs(v) > MAX_ABS_RESULT:
            raise ValueError("result too large or not a real number")
        return v

    return float(ev(tree.body))


def format_number(x: float) -> str:
    return str(int(x)) if x.is_integer() else f"{x:.10g}"


# ---------------------------------------------------------------- cost meter
class CostCapExceeded(RuntimeError):
    pass


def usage_cost(usage, model: str, prices: dict) -> float:
    """USD for one ModelUsage. prices: {model: {input, output}} per million tokens; unlisted models pay 'default'."""
    p = prices.get(model) or prices.get("default") or {"input": 0.0, "output": 0.0}
    return (usage.input_tokens * p["input"] + usage.output_tokens * p["output"]) / 1_000_000


class CostMeter:
    """Process-wide spend tracker, charged by CostHook for every agent and judge model call."""

    def __init__(self, cap_usd: float, prices: dict | None = None):
        self.cap_usd = cap_usd
        self.prices = prices or {}
        self.spent = 0.0

    def charge(self, usage, model: str) -> float:
        cost = usage_cost(usage, model, self.prices)
        self.spent += cost
        return cost

    def check(self) -> None:
        if self.spent >= self.cap_usd:
            raise CostCapExceeded(f"cost cap hit (${self.spent:.2f} of ${self.cap_usd:.2f})")


METER = CostMeter(float("inf"))


def set_meter(meter: CostMeter) -> CostMeter:
    global METER
    METER = meter
    return meter


@hooks(name="arena_cost_meter", description="Charge every model call (agent and judge) to the CostMeter")
class CostHook(Hooks):
    async def on_model_usage(self, data: ModelUsageData) -> None:
        METER.charge(data.usage, data.model_name)


# ---------------------------------------------------------------- tools
@functools.cache
def _index() -> corpus.Index:
    return corpus.Index(corpus.load())


@tool
def search_docs() -> Tool:
    async def execute(query: str) -> str:
        """Keyword search over the Halcyon Robotics document corpus.

        Args:
            query: Keywords to search for.

        Returns:
            Up to 5 lines "<id> | <title> | <snippet>", best match first.
        """
        METER.check()
        hits = _index().search(query)
        if not hits:
            return f"no results for query: {query}"
        return "\n".join(f"{h.id} | {h.title} | {h.snippet}" for h in hits)

    return execute


@tool
def get_doc() -> Tool:
    async def execute(id: str) -> str:
        """Fetch the full text of one document.

        Args:
            id: Document ID, e.g. HR-001.

        Returns:
            The document markdown including its front matter.
        """
        METER.check()
        doc = _index().docs.get(id.strip())
        return doc.text if doc else f"error: no document with id {id}"

    return execute


@tool
def calculate() -> Tool:
    async def execute(expr: str) -> str:
        """Evaluate an arithmetic expression.

        Args:
            expr: Numbers with + - * / // % ** and parentheses, e.g. "179 * 25".

        Returns:
            The result, or "error: <reason>".
        """
        METER.check()
        try:
            return format_number(safe_eval(expr))
        except ValueError as e:
            return f"error: {e}"

    return execute
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
pytest tests/test_calculator.py tests/test_agents.py -v
```

Expected: PASS (`30 passed`).

- [ ] **Step 5: Commit**

Run:

```bash
git add arena_evals/agents.py tests/test_calculator.py tests/test_agents.py
git commit -m "feat(agents): search/get_doc/calculate tools, ast-safe calculator, cost meter"
```

### Task 4: Variant prompts, FINAL-line parser, `arena_agent` solver

**Files:**
- Create: `prompts/_output_contract.md`
- Create: `prompts/baseline.md`
- Create: `prompts/concise.md`
- Create: `prompts/verbose.md`
- Create: `prompts/regressed.md`
- Modify: `arena_evals/agents.py`
- Test: `tests/test_solver.py`

**Interfaces:**
- Consumes: tools and `METER` from Task 3
- Produces: `agents.Final` (pydantic: `answer: str`, `citations: list[str]`, `abstain: bool`; strict, no extra fields)
- Produces: `agents.parse_final(text) -> Final | None` (last `FINAL:` line; None -> status `format_error`)
- Produces: `agents.load_prompt(variant, prompts_dir=PROMPTS_DIR) -> str` (variant prompt + output contract)
- Produces: `agents.arena_agent(variant: str, prompts_dir: str = str(PROMPTS_DIR), cache: bool = True) -> Solver` (Task 14 adds `span_attrs: dict | None = None`)

- [ ] **Step 1: Write the failing test**

Create `tests/test_solver.py`:

```python
from arena_evals import agents
from arena_evals.agents import load_prompt, parse_final


def test_parse_final():
    good = 'Answer text\nFINAL: {"answer": "16 weeks", "citations": ["HR-009"], "abstain": false}'
    f = parse_final(good)
    assert (f.answer, f.citations, f.abstain) == ("16 weeks", ["HR-009"], False)
    assert parse_final("no final line") is None
    assert parse_final('FINAL: {"answer": 16, "citations": [], "abstain": false}') is None  # answer must be str
    assert parse_final('FINAL: {"answer": "x", "citations": []}') is None
    assert parse_final("FINAL: not json") is None
    assert parse_final(None) is None


def test_prompts_share_output_contract_and_regressed_drops_cite_and_verify():
    for v in ("baseline", "concise", "verbose", "regressed"):
        assert load_prompt(v).rstrip().endswith('"abstain": false}')
    assert "verify" in load_prompt("baseline").lower()
    assert "verify" not in load_prompt("regressed").lower()
    assert "cite" not in open(agents.PROMPTS_DIR / "regressed.md", encoding="utf-8").read().lower()
```

- [ ] **Step 2: Run it to verify it fails**

Run:

```bash
pytest tests/test_solver.py -v
```

Expected: FAIL, with `ImportError: cannot import name 'load_prompt'` in the output.

- [ ] **Step 3: Create the prompts**

All four variants share the same tools and output contract and differ only in their instructions. `regressed.md` is `baseline.md` minus the cite-and-verify instructions (read full docs, verify each fact, cite the docs relied on); it keeps the conflict, calculator and abstention rules. Task 29 checks empirically that it costs at least 8 pts.

Create `prompts/_output_contract.md`:

```markdown
## Output contract (required)

End your final message with exactly one line in this form, with nothing after it:

FINAL: {"answer": "<answer as a string>", "citations": ["<doc id>"], "abstain": <true or false>}

- `answer` is always a JSON string, including numbers (write "4475 USD", not 4475).
- `citations` lists the IDs of the documents that support the answer. Use [] when abstaining.
- `abstain` is true only when the documents do not contain the answer. Then `answer` is "".

Example:
FINAL: {"answer": "16 weeks", "citations": ["HR-009"], "abstain": false}
```

Create `prompts/baseline.md`:

```markdown
You are the Halcyon Robotics internal assistant. Today is 2026-09-01. You answer employee questions using only the company document corpus, which you reach through three tools: `search_docs`, `get_doc` and `calculate`.

How to work:

1. Search for the relevant documents with `search_docs`.
2. Open every document you rely on with `get_doc` and read the full text. Do not answer from search snippets alone.
3. Verify each fact in your answer against the full document text before you state it.
4. When two documents give different values for the same fact, use the one with the latest `effective_date` that is not after today. A newer document says which document it supersedes.
5. Use `calculate` for every arithmetic step. Do not do arithmetic in your head.
6. Cite the ID of every document your answer relies on, and only those.
7. If the documents do not contain the answer, abstain. Never guess and never use outside knowledge.

Write a short, direct answer of one to three sentences, then the FINAL line.
```

Create `prompts/concise.md`:

```markdown
You are the Halcyon Robotics internal assistant. Today is 2026-09-01. You answer employee questions using only the company document corpus, which you reach through three tools: `search_docs`, `get_doc` and `calculate`.

How to work:

1. Search for the relevant documents with `search_docs`.
2. Open every document you rely on with `get_doc` and read the full text. Do not answer from search snippets alone.
3. Verify each fact in your answer against the full document text before you state it.
4. When two documents give different values for the same fact, use the one with the latest `effective_date` that is not after today. A newer document says which document it supersedes.
5. Use `calculate` for every arithmetic step. Do not do arithmetic in your head.
6. Cite the ID of every document your answer relies on, and only those.
7. If the documents do not contain the answer, abstain. Never guess and never use outside knowledge.

Be as brief as possible: the answer is a single short phrase or number, with no explanation. Then the FINAL line.
```

Create `prompts/verbose.md`:

```markdown
You are the Halcyon Robotics internal assistant. Today is 2026-09-01. You answer employee questions using only the company document corpus, which you reach through three tools: `search_docs`, `get_doc` and `calculate`.

How to work:

1. Search for the relevant documents with `search_docs`.
2. Open every document you rely on with `get_doc` and read the full text. Do not answer from search snippets alone.
3. Verify each fact in your answer against the full document text before you state it.
4. When two documents give different values for the same fact, use the one with the latest `effective_date` that is not after today. A newer document says which document it supersedes.
5. Use `calculate` for every arithmetic step. Do not do arithmetic in your head.
6. Cite the ID of every document your answer relies on, and only those.
7. If the documents do not contain the answer, abstain. Never guess and never use outside knowledge.

Write a thorough answer of two or three paragraphs: state the answer, explain where it comes from, walk through any calculation, and mention relevant context from the documents. Then the FINAL line.
```

Create `prompts/regressed.md`:

```markdown
You are the Halcyon Robotics internal assistant. Today is 2026-09-01. You answer employee questions using only the company document corpus, which you reach through three tools: `search_docs`, `get_doc` and `calculate`.

How to work:

1. Search for the relevant documents with `search_docs`.
2. When two documents give different values for the same fact, use the one with the latest `effective_date` that is not after today. A newer document says which document it supersedes.
3. Use `calculate` for every arithmetic step. Do not do arithmetic in your head.
4. If the documents do not contain the answer, abstain. Never guess and never use outside knowledge.

Write a short, direct answer of one to three sentences, then the FINAL line.
```

- [ ] **Step 4: Add the parser and the solver to `agents.py`**

The solver inserts the system prompt as a `ChatMessageSystem` instead of using Inspect's `system_message()`, because `system_message()` runs the text through a template formatter that turns `{"answer"` into `{{"answer"}` and corrupts the JSON contract (check 3c in Task 8).

Append to the end of `arena_evals/agents.py`:

```python
# ---------------------------------------------------------------- FINAL line + solver
class Final(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    answer: str
    citations: list[str]
    abstain: bool


def parse_final(text: str) -> Final | None:
    """Parse the last `FINAL: {...}` line. None if missing or invalid (-> status format_error)."""
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if line.startswith("FINAL:"):
            try:
                return Final.model_validate_json(line[len("FINAL:"):])
            except ValidationError:
                return None
    return None


def load_prompt(variant: str, prompts_dir: str | Path = PROMPTS_DIR) -> str:
    d = Path(prompts_dir)
    return (d / f"{variant}.md").read_text(encoding="utf-8").rstrip() + "\n\n" + \
        (d / "_output_contract.md").read_text(encoding="utf-8")


@solver
def arena_agent(variant: str, prompts_dir: str = str(PROMPTS_DIR), cache: bool = True) -> Solver:
    # ponytail: inserts ChatMessageSystem directly; inspect's system_message() str-formats and mangles the JSON braces
    prompt = load_prompt(variant, prompts_dir)
    tools = use_tools(search_docs(), get_doc(), calculate())
    loop = generate(tool_calls="loop", cache=CachePolicy(expiry=None) if cache else False)

    async def solve(state: TaskState, generate_fn: Generate) -> TaskState:
        METER.check()
        state.messages.insert(0, ChatMessageSystem(content=prompt))
        state = await tools(state, generate_fn)
        return await loop(state, generate_fn)

    return solve
```

- [ ] **Step 5: Run the tests to verify they pass**

Run:

```bash
pytest tests/test_solver.py tests/test_agents.py tests/test_calculator.py -v
```

Expected: PASS (`32 passed`).

- [ ] **Step 6: Commit**

Run:

```bash
git add prompts arena_evals/agents.py tests/test_solver.py
git commit -m "feat(agents): baseline/concise/verbose/regressed prompts, FINAL parser, arena_agent solver"
```

### Task 5: Task schema, dataset hash, gate mix and near-duplicate check

**Files:**
- Create: `arena_evals/datasets.py`
- Test: `tests/test_datasets.py`

**Interfaces:**
- Consumes: `corpus.load`, `config.sha256_files`, `config.ROOT`
- Produces: `datasets.TaskRecord` (pydantic, `extra="forbid"`: `id`, `input`, `reference: str | None`, `gold_doc_ids: list[str]`, `type`, `answer_kind`, `tags: list[str]`, `split`)
- Produces: `datasets.load(path) -> list[TaskRecord]` (rejects unknown/missing fields and duplicate IDs, reports the line)
- Produces: `datasets.write_jsonl(path, rows) -> Path`, `datasets.dataset_hash(path) -> str`
- Produces: `datasets.mix_counts(split) -> dict[str, int]`, `datasets.check_mix(tasks, split="gate", tol=1) -> list[str]`
- Produces: `datasets.normalize(text) -> str`, `datasets.near_duplicates(gate, others, threshold=0.8) -> list[tuple[str, str, float]]`
- Produces: `datasets.check(root=ROOT) -> int` (CLI `datasets check`; 0 ok, 1 errors)
- Produces: `datasets.TYPES`, `datasets.SHARES`, `datasets.SPLIT_SIZES`, `datasets.PREFIX`

- [ ] **Step 1: Write the failing test**

Create `tests/test_datasets.py`:

```python
import shutil
from pathlib import Path

import pytest

from arena_evals import datasets
from arena_evals.config import ROOT
from arena_evals.datasets import TaskRecord


def rec(i: int, type_: str = "lookup", split: str = "gate", text: str | None = None) -> TaskRecord:
    if type_ == "unanswerable":
        return TaskRecord(id=f"{split}-{i:04d}", input=text or f"unanswerable question number {i}", reference=None,
                          gold_doc_ids=[], type=type_, answer_kind="abstain", tags=[type_], split=split)
    kind = "numeric" if type_ == "arithmetic" else "exact"
    return TaskRecord(id=f"{split}-{i:04d}", input=text or f"question {type_} number {i} about item {i * 7}",
                      reference="42", gold_doc_ids=["HR-001"], type=type_, answer_kind=kind, tags=[type_], split=split)


def gate_set(counts: dict[str, int]) -> list[TaskRecord]:
    out, i = [], 0
    for t, n in counts.items():
        for _ in range(n):
            i += 1
            out.append(rec(i, t))
    return out


def test_schema_rejects_unknown_and_missing_fields():
    good = rec(1).model_dump()
    TaskRecord.model_validate(good)
    with pytest.raises(ValueError):
        TaskRecord.model_validate(good | {"extra": 1})
    with pytest.raises(ValueError):
        TaskRecord.model_validate({k: v for k, v in good.items() if k != "gold_doc_ids"})


def test_schema_type_rules():
    base = rec(1).model_dump()
    with pytest.raises(ValueError):  # unanswerable must abstain with null reference and no gold
        TaskRecord.model_validate(base | {"type": "unanswerable"})
    with pytest.raises(ValueError):
        TaskRecord.model_validate(base | {"type": "arithmetic", "answer_kind": "exact"})
    with pytest.raises(ValueError):
        TaskRecord.model_validate(base | {"type": "conflicting", "gold_doc_ids": ["HR-002", "HR-009"]})
    TaskRecord.model_validate(rec(2, "unanswerable").model_dump())


def test_load_rejects_duplicate_ids_and_reports_line(tmp_path: Path):
    p = datasets.write_jsonl(tmp_path / "x.jsonl", [rec(1), rec(1)])
    with pytest.raises(ValueError, match="x.jsonl:2: duplicate id"):
        datasets.load(p)


def test_dataset_hash_changes_with_content(tmp_path: Path):
    p = datasets.write_jsonl(tmp_path / "a.jsonl", [rec(1)])
    h1 = datasets.dataset_hash(p)
    assert h1.startswith("sha256:") and h1 == datasets.dataset_hash(p)
    datasets.write_jsonl(p, [rec(2)])
    assert datasets.dataset_hash(p) != h1


def test_gate_mix_within_one_task():
    assert datasets.mix_counts("gate") == {"lookup": 90, "multi_hop": 75, "arithmetic": 45,
                                           "conflicting": 45, "unanswerable": 45}
    assert datasets.check_mix(gate_set(datasets.mix_counts("gate"))) == []
    assert datasets.check_mix(gate_set({"lookup": 91, "multi_hop": 74, "arithmetic": 45,
                                        "conflicting": 45, "unanswerable": 45})) == []
    errs = datasets.check_mix(gate_set({"lookup": 92, "multi_hop": 73, "arithmetic": 45,
                                        "conflicting": 45, "unanswerable": 45}))
    assert len(errs) == 2


def test_near_duplicate_planted_and_clean():
    g = [rec(1, text="What is the monthly price of the Pro tier for 25 seats?")]
    dup = [rec(9, split="dev", text="what is the monthly price of the pro tier for 25 seats")]
    far = [rec(8, split="dev", text="Who leads the Customer Support organization?")]
    hits = datasets.near_duplicates(g, dup + far)
    assert [(a, b) for a, b, _ in hits] == [("gate-0001", "dev-0009")]
    assert hits[0][2] == pytest.approx(1.0)
    assert datasets.near_duplicates(g, far) == []


def _fake_repo(tmp_path: Path) -> Path:
    shutil.copytree(ROOT / "corpus", tmp_path / "corpus")
    (tmp_path / "datasets").mkdir()
    return tmp_path


def test_check_fails_on_planted_near_duplicate(tmp_path: Path, capsys):
    root = _fake_repo(tmp_path)
    gate = gate_set(datasets.mix_counts("gate"))
    datasets.write_jsonl(root / "datasets" / "gate.jsonl", gate)
    datasets.write_jsonl(root / "datasets" / "calibration.jsonl", [rec(1, split="calibration", text="unrelated wording entirely here")])
    datasets.write_jsonl(root / "datasets" / "dev.jsonl", [rec(1, split="dev", text=gate[0].input + "?")])
    assert datasets.check(root) == 1
    assert "near-duplicate: gate-0001 ~ dev-0001" in capsys.readouterr().out
```

- [ ] **Step 2: Run it to verify it fails**

Run:

```bash
pytest tests/test_datasets.py -v
```

Expected: FAIL, with `ImportError: cannot import name 'datasets' from 'arena_evals'` in the output.

- [ ] **Step 3: Create the datasets module**

Create `arena_evals/datasets.py`:

```python
"""Task JSONL schema, hashing, gate-mix + near-duplicate checks, LLM drafting, spot-check CLI."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from arena_evals import corpus
from arena_evals.config import ROOT, sha256_files

TYPES = ("lookup", "multi_hop", "arithmetic", "conflicting", "unanswerable")
SHARES = {"lookup": 0.30, "multi_hop": 0.25, "arithmetic": 0.15, "conflicting": 0.15, "unanswerable": 0.15}
SPLIT_SIZES = {"gate": 300, "calibration": 120, "dev": 100}
PREFIX = {"gate": "gate", "calibration": "cal", "dev": "dev"}


class TaskRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")
    id: str
    input: str
    reference: str | None
    gold_doc_ids: list[str]
    type: Literal["lookup", "multi_hop", "arithmetic", "conflicting", "unanswerable"]
    answer_kind: Literal["exact", "numeric", "free_text", "abstain"]
    tags: list[str]
    split: Literal["gate", "calibration", "dev"]

    @model_validator(mode="after")
    def _rules(self) -> "TaskRecord":
        if (self.type == "unanswerable") != (self.answer_kind == "abstain"):
            raise ValueError("type unanswerable <=> answer_kind abstain")
        if self.type == "unanswerable":
            if self.reference is not None or self.gold_doc_ids:
                raise ValueError("unanswerable tasks need reference null and gold_doc_ids []")
        else:
            if not self.reference or not self.gold_doc_ids:
                raise ValueError("answerable tasks need a reference and at least one gold doc")
        if self.type == "arithmetic" and self.answer_kind != "numeric":
            raise ValueError("arithmetic tasks are numeric")
        if self.type == "conflicting" and len(self.gold_doc_ids) != 1:
            raise ValueError("conflicting tasks cite exactly the currently effective doc")
        return self


def load(path: Path | str) -> list[TaskRecord]:
    records, seen = [], set()
    for n, line in enumerate(Path(path).read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            r = TaskRecord.model_validate_json(line)
        except ValidationError as e:
            raise ValueError(f"{path}:{n}: {e}") from None
        if r.id in seen:
            raise ValueError(f"{path}:{n}: duplicate id {r.id}")
        seen.add(r.id)
        records.append(r)
    return records


def write_jsonl(path: Path | str, rows: list) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [r.model_dump_json() if isinstance(r, BaseModel) else json.dumps(r) for r in rows]
    path.write_text("".join(l + "\n" for l in lines), encoding="utf-8", newline="\n")
    return path


def dataset_hash(path: Path | str) -> str:
    return sha256_files(Path(path))


def mix_counts(split: str) -> dict[str, int]:
    return {t: round(SHARES[t] * SPLIT_SIZES[split]) for t in TYPES}


def check_mix(tasks: list[TaskRecord], split: str = "gate", tol: int = 1) -> list[str]:
    want = mix_counts(split)
    have = {t: sum(1 for r in tasks if r.type == t) for t in TYPES}
    return [f"{split}: {t} has {have[t]} tasks, want {want[t]} +/- {tol}" for t in TYPES if abs(have[t] - want[t]) > tol]


def normalize(text: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", "", text.lower()).split())


def _shingles(text: str, n: int = 5) -> set[str]:
    t = normalize(text)
    return {t[i:i + n] for i in range(max(len(t) - n + 1, 1))}


def near_duplicates(gate: list[TaskRecord], others: list[TaskRecord],
                    threshold: float = 0.8) -> list[tuple[str, str, float]]:
    """(gate_id, other_id, jaccard) for every pair whose character 5-gram Jaccard >= threshold. O(n*m)."""
    other_sh = [(o.id, _shingles(o.input)) for o in others]
    out = []
    for g in gate:
        gs = _shingles(g.input)
        for oid, os_ in other_sh:
            j = len(gs & os_) / len(gs | os_)
            if j >= threshold:
                out.append((g.id, oid, j))
    return out


def check(root: Path = ROOT) -> int:
    """`datasets check`: schema, gold IDs, gate mix, cross-split IDs, near-duplicates. Exit code 0/1."""
    errors: list[str] = []
    docs = corpus.load(root / "corpus" / "docs")
    splits: dict[str, list[TaskRecord]] = {}
    for s in SPLIT_SIZES:
        p = root / "datasets" / f"{s}.jsonl"
        if not p.exists():
            errors.append(f"missing {p}")
            continue
        try:
            splits[s] = load(p)
        except ValueError as e:
            errors.append(str(e))
            continue
        errors += [f"{t.id}: split field is {t.split}, file is {s}" for t in splits[s] if t.split != s]
        errors += [f"{t.id}: unknown gold doc {g}" for t in splits[s] for g in t.gold_doc_ids if g not in docs]
    if "gate" in splits:
        errors += check_mix(splits["gate"], "gate")
    ids = [t.id for ts in splits.values() for t in ts]
    errors += [f"id used in more than one split: {i}" for i in sorted({i for i in ids if ids.count(i) > 1})]
    others = splits.get("dev", []) + splits.get("calibration", [])
    for g, o, j in near_duplicates(splits.get("gate", []), others):
        errors.append(f"near-duplicate: {g} ~ {o} (jaccard {j:.2f})")
    for e in errors:
        print(f"ERROR {e}")
    print("datasets check: " + ("FAIL" if errors else "OK") + f" ({sum(len(v) for v in splits.values())} tasks)")
    return 1 if errors else 0
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
pytest tests/test_datasets.py -v
```

Expected: PASS (`7 passed`).

- [ ] **Step 5: Commit**

Run:

```bash
git add arena_evals/datasets.py tests/test_datasets.py
git commit -m "feat(datasets): task schema, dataset hash, gate mix and near-duplicate checks"
```

### Task 6: LLM drafting, spot-check CLI and label-noise floor

**Files:**
- Create: `arena_evals/stats/agreement.py`
- Modify: `arena_evals/datasets.py`
- Test: `tests/test_drafting.py`
- Test: `tests/test_agreement.py`

**Interfaces:**
- Consumes: `TaskRecord`, `mix_counts`, `normalize`, `write_jsonl`, `load` (Task 5); `corpus.load`, `corpus.Doc`
- Produces: `stats.agreement.wilson(k: int, n: int, conf=0.95) -> CI` (Task 18 adds the agreement metrics)
- Produces: `datasets.draft(split, corpus_docs, model, as_of="2026-09-01", batch=15, max_batches_per_type=12) -> list[TaskRecord]` (spec name `draft(split, corpus, model)`; drafts ceil(1.1 x mix) per type)
- Produces: `datasets.spotcheck(split, frac=0.2, seed=20260930, ask=input, root=ROOT) -> Path` (writes `datasets/spotcheck.jsonl` and the final `datasets/<split>.jsonl`)
- Produces: `datasets.label_noise_floor(split, root=ROOT) -> dict | None` (`{"point", "ci95", "n"}`, Wilson 95%)

- [ ] **Step 1: Write the failing tests**

Create `tests/test_agreement.py`:

```python
import pytest

from arena_evals.stats.agreement import wilson


def test_wilson_known_value():
    ci = wilson(6, 60)
    assert ci.point == pytest.approx(0.1)
    assert ci.lo == pytest.approx(0.0466, abs=1e-3)
    assert ci.hi == pytest.approx(0.2012, abs=1e-3)
    with pytest.raises(ValueError):
        wilson(0, 0)
```

Create `tests/test_drafting.py`:

```python
import json
from pathlib import Path

import pytest
from inspect_ai.model import ModelOutput, get_model

from arena_evals import corpus, datasets
from test_datasets import _fake_repo, rec


def test_draft_with_scripted_model_keeps_only_valid_tasks(monkeypatch):
    docs = corpus.load()

    def replies():
        n = 0
        while True:
            n += 1
            items = [
                {"input": f"Question variant {n}-{j} about PTO days", "reference": "20 days",
                 "gold_doc_ids": ["HR-001"], "answer_kind": "exact"} for j in range(15)
            ] + [{"input": "bad gold", "reference": "x", "gold_doc_ids": ["NOPE-1"], "answer_kind": "exact"}]
            yield ModelOutput.from_content("mockllm/model", json.dumps(items))

    monkeypatch.setattr(datasets, "TYPES", ("lookup",))  # script one type only
    model = get_model("mockllm/model", custom_outputs=replies(), memoize=False)
    out = datasets.draft("dev", docs, model)
    assert len(out) == 33  # ceil(1.1 * 30)
    assert all(r.gold_doc_ids == ["HR-001"] and r.tags == ["lookup", "hr"] for r in out)
    assert out[0].id == "dev-0001"


def test_spotcheck_writes_final_split_and_noise_floor(tmp_path: Path):
    root = _fake_repo(tmp_path)
    drafts = []
    i = 0
    for t, n in datasets.mix_counts("dev").items():
        for _ in range(n + 2):
            i += 1
            drafts.append(rec(i, t, split="dev", text=f"draft question {i} of type {t}"))
    datasets.write_jsonl(root / "datasets" / "drafts" / "dev.jsonl", drafts)
    answers = iter(["d", "f", "43", "", "reference was wrong"] + ["o"] * 1000)
    datasets.spotcheck("dev", frac=0.2, seed=1, ask=lambda _: next(answers), root=root)
    final = datasets.load(root / "datasets" / "dev.jsonl")
    assert len(final) == 100 and datasets.check_mix(final, "dev") == []
    sc = [json.loads(l) for l in (root / "datasets" / "spotcheck.jsonl").read_text().splitlines()]
    assert [r["verdict"] for r in sc[:2]] == ["dropped", "fixed"]
    assert sc[1]["task_id"] not in {r.id for r in final} or \
        next(r for r in final if r.id == sc[1]["task_id"]).reference == "43"
    assert sc[0]["task_id"] not in {r.id for r in final}
    floor = datasets.label_noise_floor("dev", root)
    assert floor["n"] == 22 and floor["point"] == pytest.approx(2 / 22)
    assert floor["ci95"][0] < floor["point"] < floor["ci95"][1]
```

- [ ] **Step 2: Run them to verify they fail**

Run:

```bash
pytest tests/test_agreement.py tests/test_drafting.py -v
```

Expected: FAIL, with `ModuleNotFoundError: No module named 'arena_evals.stats.agreement'` in the output.

- [ ] **Step 3: Create the Wilson interval**

Create `arena_evals/stats/agreement.py`:

```python
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
```

- [ ] **Step 4: Extend the imports of `datasets.py`**

In `arena_evals/datasets.py`, replace:

```python
import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from arena_evals import corpus
from arena_evals.config import ROOT, sha256_files
```

with:

```python
import asyncio
import json
import random
import re
from pathlib import Path
from typing import Callable, Literal

from pydantic import BaseModel, ConfigDict, ValidationError, model_validator

from arena_evals import corpus
from arena_evals.config import ROOT, sha256_files
from arena_evals.stats.agreement import wilson
```

- [ ] **Step 5: Add drafting and spot-check to `datasets.py`**

Drafting asks the drafting model (`configs/models.yaml: drafter`) for each task type in batches, rejects drafts whose gold docs are not in the corpus, conflicting drafts that cite a superseded doc, and duplicates, and assigns tags `[type, <kind of each gold doc>]`. The whole draft runs in one event loop: repeated `asyncio.run` calls against a real Anthropic client can fail with a closed-loop error.

Append to the end of `arena_evals/datasets.py`:

```python
# ---------------------------------------------------------------- drafting (LLM)
TYPE_GUIDE = {
    "lookup": "a single fact stated in one document. answer_kind is exact (short span such as a name or date), "
              "numeric (a quantity), or free_text (a sentence-length answer).",
    "multi_hop": "needs facts from two different documents combined, e.g. follow a 'Related documents' reference. "
                 "answer_kind is usually free_text; exact or numeric are allowed.",
    "arithmetic": "needs a calculation over numbers stated in the documents. answer_kind is numeric; reference is "
                  "the computed number with its unit, e.g. \"4475 USD\".",
    "conflicting": "asks for a fact that two documents state differently (one supersedes the other). reference is "
                   "the value from the currently effective document and gold_doc_ids is ONLY that document. "
                   "answer_kind is exact or numeric.",
    "unanswerable": "sounds plausible for this company but the documents do not answer it. answer_kind is abstain, "
                    "reference is null, gold_doc_ids is [].",
}
DRAFT_PROMPT = """You write evaluation questions for an assistant that answers from the Halcyon Robotics corpus below.
The corpus as-of date is {as_of}: a document is in force if its effective_date is not after that date.

Write {n} distinct {type} questions. A {type} question {guide}
Each question must be answerable (or clearly not answerable) from the corpus alone, phrased as an employee would ask it,
and different in wording and target fact from the other questions.

Return ONLY a JSON array. Each element: {{"input": str, "reference": str or null, "gold_doc_ids": [doc ids],
"answer_kind": "exact" | "numeric" | "free_text" | "abstain"}}

CORPUS:
{corpus}
"""


def _effective_ids(docs: dict[str, corpus.Doc], as_of: str) -> set[str]:
    superseded = {d.supersedes for d in docs.values() if d.supersedes and d.effective_date <= as_of}
    return {i for i, d in docs.items() if i not in superseded and d.effective_date <= as_of}


def _parse_drafts(text: str) -> list[dict]:
    try:
        items = json.loads(text[text.index("["): text.rindex("]") + 1])
    except ValueError:
        return []
    return [i for i in items if isinstance(i, dict)]


def draft(split: str, corpus_docs: dict[str, corpus.Doc], model, as_of: str = "2026-09-01",
          batch: int = 15, max_batches_per_type: int = 12) -> list[TaskRecord]:
    """Ask the drafting model for ceil(1.1 x mix) tasks per type; keep only schema-valid, corpus-consistent ones."""
    return asyncio.run(_draft(split, corpus_docs, model, as_of, batch, max_batches_per_type))


async def _draft(split, corpus_docs, model, as_of, batch, max_batches_per_type) -> list[TaskRecord]:
    from inspect_ai.model import GenerateConfig, get_model

    m = get_model(model) if isinstance(model, str) else model
    corpus_text = "\n\n".join(d.text for d in corpus_docs.values())
    effective = _effective_ids(corpus_docs, as_of)
    out: list[TaskRecord] = []
    seen: set[str] = set()
    for t in TYPES:
        want = -(-11 * mix_counts(split)[t] // 10)  # ceil(1.1 x mix), in integers to dodge float error
        got = 0
        for _ in range(max_batches_per_type):
            if got >= want:
                break
            prompt = DRAFT_PROMPT.format(as_of=as_of, n=min(batch, want - got), type=t, guide=TYPE_GUIDE[t],
                                         corpus=corpus_text)
            resp = await m.generate(prompt, config=GenerateConfig(temperature=1.0))
            for item in _parse_drafts(resp.completion):
                if got >= want:
                    break
                gold = item.get("gold_doc_ids") or []
                if any(g not in corpus_docs for g in gold):
                    continue
                if t == "conflicting" and (len(gold) != 1 or gold[0] not in effective):
                    continue
                key = normalize(str(item.get("input", "")))
                if not key or key in seen:
                    continue
                tags = [t] + sorted({corpus_docs[g].kind for g in gold})
                try:
                    rec = TaskRecord(id=f"{PREFIX[split]}-{len(out) + 1:04d}", input=item["input"],
                                     reference=item.get("reference"), gold_doc_ids=gold, type=t,
                                     answer_kind=item.get("answer_kind"), tags=tags, split=split)
                except (ValidationError, KeyError):
                    continue
                seen.add(key)
                out.append(rec)
                got += 1
        if got < want:
            raise RuntimeError(f"drafting produced only {got}/{want} valid {t} tasks; re-run or raise max_batches")
    return out


# ---------------------------------------------------------------- spot-check (HUMAN)
def spotcheck(split: str, frac: float = 0.2, seed: int = 20260930, ask: Callable[[str], str] = input,
              root: Path = ROOT) -> Path:
    """Review a seeded sample of drafts; write datasets/spotcheck.jsonl and the final datasets/<split>.jsonl."""
    drafts = load(root / "datasets" / "drafts" / f"{split}.jsonl")
    docs = corpus.load(root / "corpus" / "docs")
    sample = random.Random(seed).sample(drafts, round(frac * len(drafts)))
    verdicts, fixed = [], {}
    for i, t in enumerate(sample, 1):
        print(f"\n[{i}/{len(sample)}] {t.id} ({t.type}, {t.answer_kind})\nQ: {t.input}\nREF: {t.reference}\nGOLD: {t.gold_doc_ids}")
        for g in t.gold_doc_ids:
            print(f"--- {g}\n{docs[g].body}")
        v = ""
        while v not in ("o", "f", "d"):
            v = ask("[o]k / [f]ix / [d]rop > ").strip().lower()[:1]
        note = ""
        if v == "f":
            ref = ask("correct reference (enter = keep): ").strip()
            gold = ask("correct gold doc ids, comma separated (enter = keep): ").strip()
            note = ask("note: ").strip()
            upd = {}
            if ref:
                upd["reference"] = ref
            if gold:
                upd["gold_doc_ids"] = [g.strip() for g in gold.split(",")]
            fixed[t.id] = TaskRecord.model_validate(t.model_dump() | upd)
        verdicts.append({"split": split, "task_id": t.id, "verdict": {"o": "ok", "f": "fixed", "d": "dropped"}[v],
                         "note": note})
    sc_path = root / "datasets" / "spotcheck.jsonl"
    kept = [json.loads(l) for l in sc_path.read_text(encoding="utf-8").splitlines()] if sc_path.exists() else []
    write_jsonl(sc_path, [r for r in kept if r["split"] != split] + verdicts)
    dropped = {v["task_id"] for v in verdicts if v["verdict"] == "dropped"}
    final: list[TaskRecord] = []
    for t_ in TYPES:
        pool = [fixed.get(r.id, r) for r in drafts if r.type == t_ and r.id not in dropped]
        need = mix_counts(split)[t_]
        if len(pool) < need:
            raise RuntimeError(f"{split}: only {len(pool)} {t_} tasks left after drops, need {need}; draft more")
        final += pool[:need]
    return write_jsonl(root / "datasets" / f"{split}.jsonl", final)


def label_noise_floor(split: str, root: Path = ROOT) -> dict | None:
    """(fixed + dropped) / spot-checked sample, with a 95% Wilson interval. None if never spot-checked."""
    p = root / "datasets" / "spotcheck.jsonl"
    if not p.exists():
        return None
    rows = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]
    rows = [r for r in rows if r["split"] == split]
    if not rows:
        return None
    ci = wilson(sum(r["verdict"] != "ok" for r in rows), len(rows))
    return {"point": ci.point, "ci95": [ci.lo, ci.hi], "n": len(rows)}
```

- [ ] **Step 6: Run the tests to verify they pass**

Run:

```bash
pytest tests/test_agreement.py tests/test_drafting.py tests/test_datasets.py -v
```

Expected: PASS (`10 passed`).

- [ ] **Step 7: Commit**

Run:

```bash
git add arena_evals/stats/agreement.py arena_evals/datasets.py tests/test_agreement.py tests/test_drafting.py
git commit -m "feat(datasets): LLM drafting, seeded spot-check CLI, Wilson label-noise floor"
```

### Task 7: (HUMAN) Author the dev, calibration and gate splits

**Files:**
- Create: `datasets/drafts/{dev,calibration,gate}.jsonl`
- Create: `datasets/{dev,calibration,gate}.jsonl`
- Create: `datasets/spotcheck.jsonl`

**Interfaces:**
- Consumes: CLI `datasets draft`, `datasets spotcheck`, `datasets check` (Tasks 5-6); an Anthropic API key
- Produces: `datasets/dev.jsonl` (100), `datasets/calibration.jsonl` (120), `datasets/gate.jsonl` (300) passing `datasets check`; the spot-check verdicts behind every run's `label_noise_floor`

- [ ] **Step 1: (HUMAN) Provide the API key**

Create an Anthropic API key with a spending limit, then in the shell you will use for the next steps:

Run:

```bash
export ANTHROPIC_API_KEY=sk-ant-...   # PowerShell: $env:ANTHROPIC_API_KEY="sk-ant-..."
```

Never commit the key. It is only read from the environment.

- [ ] **Step 2: (HUMAN) Draft the three splits**

Run (needs ANTHROPIC_API_KEY, costs money):

```bash
python -m arena_evals datasets draft --split dev
python -m arena_evals datasets draft --split calibration
python -m arena_evals datasets draft --split gate
```

Expected: `wrote 112 drafts to .../datasets/drafts/dev.jsonl`, `wrote 133 drafts ...calibration.jsonl`, `wrote 332 drafts ...gate.jsonl`. If a split raises `drafting produced only X/Y valid <type> tasks`, re-run that split's command. Cost was not measured; it is roughly 30-40 calls with the ~10k-token corpus in each prompt.

- [ ] **Step 3: (HUMAN) Spot-check each split**

For each split the CLI shows a seeded 20% sample (dev 22, calibration 27, gate 66 tasks) with the gold documents. Answer `o` if the question, reference and gold IDs are right; `f` to fix (type the corrected reference and/or gold IDs); `d` to drop a task that is ambiguous, wrong or not answerable as typed. Judge unanswerable tasks by checking the corpus really does not answer them. Conflicting tasks must reference the value of the currently effective doc. Take your time: these verdicts are the label-noise floor reported in every run.

Run (interactive):

```bash
python -m arena_evals datasets spotcheck --split dev
python -m arena_evals datasets spotcheck --split calibration
python -m arena_evals datasets spotcheck --split gate
```

Expected after each: `wrote .../datasets/<split>.jsonl` and `label-noise floor X.X% (95% CI a% to b%, n=<sample size>)`. Write the three floors down for the README. If it raises `only N <type> tasks left after drops`, re-run `datasets draft` for that split and spot-check again.

- [ ] **Step 4: Check the datasets**

Run (after the spot-checks):

```bash
python -m arena_evals datasets check
```

Expected last line: `datasets check: OK (520 tasks)`. If a line says `near-duplicate: gate-XXXX ~ dev-YYYY`, reword the gate task's `input` in `datasets/gate.jsonl` (keep the same fact and reference), then re-run until OK.

- [ ] **Step 5: Run the whole suite**

Run:

```bash
pytest -q
```

Expected: PASS (`51 passed`).

- [ ] **Step 6: Commit**

Run (after the HUMAN steps above):

```bash
git add datasets
git commit -m "data: dev/calibration/gate splits with spot-check verdicts"
```

## Milestone M2: Runner skeleton

### Task 8: API spike: verify the Inspect and Phoenix behaviours the harness relies on

**Files:**
- Create: `spikes/verify_apis.py`

**Interfaces:**
- Consumes: installed inspect-ai 0.3.272 and arize-phoenix
- Produces: An observed record of: `CachePolicy(expiry=None)`, mockllm callable `custom_outputs`, solver metadata in logs, 1-based epochs, `system_message()` brace mangling, `inspect_ai.score()` re-scoring, hook semantics, tool exceptions, `phoenix.otel.register` + `phoenix serve`

- [ ] **Step 1: Create the spike script**

Create `spikes/verify_apis.py`:

```python
"""Verify the Inspect/Phoenix behaviours this harness relies on. Run: python spikes/verify_apis.py
Needs `phoenix serve` running on localhost:6006 for check 7 (see Task 8). Prints OK/FAIL per check."""
import importlib.metadata as md
import os
import tempfile

os.environ["INSPECT_CACHE_DIR"] = tempfile.mkdtemp()
os.environ["INSPECT_DISPLAY"] = "none"

from inspect_ai import Task, eval, score  # noqa: E402
from inspect_ai.dataset import MemoryDataset, Sample  # noqa: E402
from inspect_ai.hooks import Hooks, hooks  # noqa: E402
from inspect_ai.log import read_eval_log  # noqa: E402
from inspect_ai.model import CachePolicy, ModelOutput, get_model  # noqa: E402
from inspect_ai.scorer import Score, scorer  # noqa: E402
from inspect_ai.solver import generate, solver, system_message, use_tools  # noqa: E402
from inspect_ai.tool import tool  # noqa: E402

USAGE = []


@hooks(name="spike_usage", description="count model usage events")
class Spike(Hooks):
    async def on_model_usage(self, data):
        USAGE.append(data.model_name)

    async def on_before_model_generate(self, data):
        raise RuntimeError("raised inside a hook")


def check(name, ok, detail=""):
    print(f"{'OK  ' if ok else 'FAIL'} {name} {detail}")


print("inspect-ai", md.version("inspect-ai"), "| arize-phoenix", md.version("arize-phoenix"),
      "| arize-phoenix-otel", md.version("arize-phoenix-otel"))

# 1. CachePolicy(expiry=None) = never expire
check("1 CachePolicy(expiry=None)", CachePolicy(expiry=None).expiry is None)


@tool
def get_doc():
    async def execute(id: str) -> str:
        """Fetch a document.

        Args:
            id: Document ID.
        """
        return f"doc {id}"
    return execute


def scripted(messages, tools, tool_choice, config):
    if not any(m.role == "tool" for m in messages):
        return ModelOutput.for_tool_call("mockllm/model", "get_doc", {"id": "HR-001"})
    return ModelOutput.from_content("mockllm/model", 'done\nFINAL: {"answer": "x", "citations": [], "abstain": false}')


@solver
def stamp():
    async def solve(state, gen):
        state.metadata["trace_id"] = "abc"
        return state
    return solve


# 2+3. mockllm with a callable custom_outputs drives a tool loop; metadata set in a solver reaches the log
model = get_model("mockllm/model", custom_outputs=scripted, memoize=False)
task = Task(dataset=MemoryDataset([Sample(input="q", id="s1")]),
            solver=[system_message('Reply with FINAL: {"answer": "x"}'), stamp(), use_tools(get_doc()),
                    generate(tool_calls="loop", cache=CachePolicy(expiry=None))], epochs=2)
log = eval(task, model=model, log_dir=tempfile.mkdtemp(), fail_on_error=False)[0]
s = log.samples[0]
check("2 mockllm callable custom_outputs + tool loop", [m.role for m in s.messages][-3:] == ["assistant", "tool", "assistant"],
      str([m.role for m in s.messages]))
check("3 solver metadata persists to log", s.metadata.get("trace_id") == "abc")
check("3b epochs are 1-based in the log", sorted(x.epoch for x in log.samples) == [1, 2])
check("3c system_message() mangles JSON braces (so agents.py inserts ChatMessageSystem directly)",
      s.messages[0].text != 'Reply with FINAL: {"answer": "x"}', repr(s.messages[0].text))

# 4. inspect_ai.score() re-scores an existing log with new scorers; dict values with None are allowed


@scorer(metrics=[])
def probe():
    async def f(state, target):
        return Score(value={"precision": None, "recall": 0.5})
    return f


rescored = score(read_eval_log(log.location), [probe()], display="none")
check("4 inspect_ai.score() re-scoring", rescored.samples[0].scores["probe"].value == {"precision": None, "recall": 0.5})

# 5. hooks: on_model_usage fires; an exception raised in a hook does NOT stop the call (so the cost cap is
#    enforced in tools/solver/judge, not in a hook)
check("5 on_model_usage fires per call", len(USAGE) >= 2, f"{len(USAGE)} events")
check("5b exception in hook is swallowed", s.error is None)

# 6. an exception raised inside a tool errors the sample with the class name in the message
class CostCapExceeded(RuntimeError):
    pass


@tool
def boom():
    async def execute(x: str) -> str:
        """Boom.

        Args:
            x: anything.
        """
        raise CostCapExceeded("cap")
    return execute


m2 = get_model("mockllm/model", custom_outputs=lambda *a: ModelOutput.for_tool_call("mockllm/model", "boom", {"x": "1"}),
               memoize=False)
log2 = eval(Task(dataset=MemoryDataset([Sample(input="q", id="b")]), solver=[use_tools(boom()), generate()]),
            model=m2, log_dir=tempfile.mkdtemp(), fail_on_error=False)[0]
err = log2.samples[0].error
check("6 tool exception -> sample error naming CostCapExceeded", bool(err and "CostCapExceeded" in err.message))

# 7. phoenix.otel.register exports spans to a running `phoenix serve`, whose PHOENIX_WORKING_DIR holds phoenix.db
try:
    from phoenix.otel import register
    tp = register(project_name="arena-evals-spike", endpoint="http://localhost:6006/v1/traces", batch=False,
                  verbose=False)
    with tp.get_tracer("spike").start_as_current_span("spike", attributes={"arena.task_id": "t"}) as span:
        tid = format(span.get_span_context().trace_id, "032x")
    tp.force_flush()
    check("7 phoenix.otel.register + span export", len(tid) == 32, f"trace_id={tid}")
except Exception as e:  # noqa: BLE001
    check("7 phoenix.otel.register + span export", False, repr(e))
```

- [ ] **Step 2: Start Phoenix in a second terminal**

In a second terminal (activate the venv there first), run:

```bash
export PHOENIX_WORKING_DIR=$PWD/.phoenix    # PowerShell: $env:PHOENIX_WORKING_DIR="$PWD/.phoenix"
phoenix serve
```

Wait until http://localhost:6006 responds (about 20 s the first time).

- [ ] **Step 3: Run the spike**

Run (needs the Phoenix server from the previous step):

```bash
python spikes/verify_apis.py
```

Expected output (observed on 2026-09-30 and again on 2026-10-01; Inspect also prints a two-line `inspect_ai v0.3.272 / hooks enabled: 1` banner after check 1, which you can ignore):

```text
inspect-ai 0.3.272 | arize-phoenix 20.16.0 | arize-phoenix-otel 0.17.2
OK   1 CachePolicy(expiry=None)
OK   2 mockllm callable custom_outputs + tool loop ['system', 'user', 'assistant', 'tool', 'assistant']
OK   3 solver metadata persists to log
OK   3b epochs are 1-based in the log
OK   3c system_message() mangles JSON braces (so agents.py inserts ChatMessageSystem directly) 'Reply with FINAL: {{"answer"}: "x"}'
OK   4 inspect_ai.score() re-scoring
OK   5 on_model_usage fires per call 4 events
OK   5b exception in hook is swallowed
OK   6 tool exception -> sample error naming CostCapExceeded
OK   7 phoenix.otel.register + span export trace_id=<32 hex chars>
```

Then confirm `.phoenix/phoenix.db` exists. Stop the server with Ctrl+C.

- [ ] **Step 4: If any check prints FAIL, adjust before continuing**

- 1 FAIL: the response cache would expire; in `agents.arena_agent` and `scorers.judge_answer` replace `CachePolicy(expiry=None)` with `CachePolicy(expiry="52W")`.

- 2 FAIL: mockllm no longer accepts a callable; the mock tests (Tasks 10, 12, 16, 19, 20, 23, 26) must switch to a generator per test and `max_connections: 1` in a test copy of `configs/eval.yaml`.

- 3 FAIL: the trace ID cannot travel in `state.metadata`; store it with `inspect_ai.util.store().set("trace_id", ...)` in `arena_agent` and read `sample.store["trace_id"]` in `run.score`.

- 3b FAIL: `run.score` computes `repeat = s.epoch - 1`; change it to `s.epoch`.

- 3c FAIL (braces survive): keep the `ChatMessageSystem` insertion anyway; nothing to change.

- 4 FAIL: phase B cannot re-score; replace `inspect_score(...)` in `run.score` with running the scorers inside the phase-A `Task(scorer=[...])` and read `s.scores` from the log. Baseline and candidate scoring then happen in each checkout, so flag this in the PR comment.

- 5 FAIL (no usage events): charge `agents.METER` in `run.score` from `s.model_usage` instead of `CostHook`. 5b FAIL (hook exceptions propagate): nothing breaks; keep the explicit checks.

- 6 FAIL: `run.generate` cannot detect the cap from sample errors; wrap `inspect_eval` with a check of `agents.METER.spent >= cap` after the call and raise `CostCapExceeded` there.

- 7 FAIL: read the exception. If `register` rejects `endpoint`, pass the collector through the `PHOENIX_COLLECTOR_ENDPOINT` environment variable instead and call `register(project_name=...)`.

- [ ] **Step 5: Commit**

Run:

```bash
git add spikes/verify_apis.py
git commit -m "chore(spike): verify Inspect/Phoenix API behaviours"
```

### Task 9: Deterministic scorers and the per-task success rule

**Files:**
- Create: `arena_evals/scorers.py`
- Test: `tests/test_scorers.py`

**Interfaces:**
- Consumes: `agents.parse_final`, `agents._index` (Tasks 3-4)
- Produces: `scorers.normalize_answer(s)`, `scorers.first_number(s) -> float | None`, `scorers.match_answer(answer, reference, answer_kind) -> bool`
- Produces: `scorers.citation_pr(cited, gold, corpus_ids) -> tuple[float | None, float | None]`
- Produces: `scorers.fetched_doc_ids(messages) -> set[str]`, `scorers.tool_call_count(messages) -> int`
- Produces: `scorers.task_success(task: dict, sample_scores: dict, status: str) -> bool | None` (None = judge error or unjudged free text, treated as missing)
- Produces: Inspect `@scorer`s `answer_match()`, `citation()` (value `{precision, recall}`), `gold_retrieval()`, `abstention()`

- [ ] **Step 1: Write the failing test**

Create `tests/test_scorers.py`:

```python
import pytest

from arena_evals.scorers import citation_pr, first_number, match_answer, normalize_answer, task_success

IDS = {"HR-001", "HR-009", "PRC-006"}


def test_normalize_and_exact():
    assert normalize_answer("  Dev  Okafor. ") == "dev okafor"
    assert match_answer("dev okafor", "Dev Okafor", "exact")
    assert not match_answer("Dev Okafor, CTO", "Dev Okafor", "exact")


@pytest.mark.parametrize("answer, ref, ok", [
    ("$4,475", "4,475 USD", True),
    ("4475.00 USD per month", "4475 USD", True),
    ("4,497 USD", "4475 USD", True),      # within 0.5%
    ("4,500 USD", "4475 USD", False),     # 0.56% off
    ("12%", "12%", True),
    ("0.004", "0", True),                 # absolute floor 0.01
    ("about twenty", "20", False),
    ("-3 degrees", "-3", True),
])
def test_numeric(answer, ref, ok):
    assert match_answer(answer, ref, "numeric") is ok


def test_first_number():
    assert first_number("1,340 alarms") == 1340.0
    assert first_number("£12.5") == 12.5
    assert first_number("none") is None


def test_free_text_never_matches_deterministically():
    assert match_answer("An expired TLS certificate", "An expired TLS certificate", "free_text") is False


def test_citation_precision_recall():
    assert citation_pr(["HR-009"], ["HR-009"], IDS) == (1.0, 1.0)
    assert citation_pr(["HR-009", "HR-001"], ["HR-009"], IDS) == (0.5, 1.0)
    assert citation_pr(["HR-999"], ["HR-009"], IDS) == (0.0, 0.0)       # not in corpus -> wrong
    assert citation_pr([], ["HR-009"], IDS) == (None, 0.0)              # empty citations
    assert citation_pr(["HR-999"], [], IDS) == (0.0, None)              # unanswerable + hallucinated citation
    assert citation_pr([], [], IDS) == (None, None)


def S(**kw):
    base = {"answer_match": True, "abstention": False, "citation_recall": 1.0, "judge": None, "judge_error": False}
    return base | kw


LOOKUP = {"type": "lookup", "answer_kind": "exact"}
FREE = {"type": "multi_hop", "answer_kind": "free_text"}
CONFLICT = {"type": "conflicting", "answer_kind": "exact"}
UNANS = {"type": "unanswerable", "answer_kind": "abstain"}


@pytest.mark.parametrize("task, scores, status, expected", [
    (LOOKUP, S(), "ok", True),
    (LOOKUP, S(answer_match=False), "ok", False),
    (LOOKUP, S(citation_recall=0.0), "ok", False),                  # right answer, no gold citation
    (LOOKUP, S(citation_recall=None), "ok", False),
    (LOOKUP, S(abstention=True), "ok", False),                      # abstained on answerable
    (CONFLICT, S(), "ok", True),
    (CONFLICT, S(answer_match=False), "ok", False),                 # used superseded value
    (UNANS, S(answer_match=False, abstention=True, citation_recall=None), "ok", True),
    (UNANS, S(answer_match=False, abstention=False, citation_recall=None), "ok", False),
    (FREE, S(answer_match=False, judge={"correct": True, "faithful": True}), "ok", True),
    (FREE, S(answer_match=False, judge={"correct": True, "faithful": False}), "ok", False),
    (FREE, S(answer_match=False, judge_error=True), "ok", None),    # judge error -> missing
    (FREE, S(answer_match=False, judge_error=True), "agent_error", False),
    (LOOKUP, S(), "agent_error", False),
    (LOOKUP, S(), "timeout", False),
    (LOOKUP, S(), "limit", False),
    (LOOKUP, S(), "format_error", False),
])
def test_task_success_table(task, scores, status, expected):
    assert task_success(task, scores, status) is expected
```

- [ ] **Step 2: Run it to verify it fails**

Run:

```bash
pytest tests/test_scorers.py -v
```

Expected: FAIL, with `ModuleNotFoundError: No module named 'arena_evals.scorers'` in the output.

- [ ] **Step 3: Create the scorers**

Citation recall is `None` when a task has no gold docs (unanswerable), instead of dividing by zero. An answerable task where the agent abstains is a FAIL even if a judge would pass it.

Create `arena_evals/scorers.py`:

```python
"""Deterministic scorers, the per-task success rule, and (M4) the LLM judge. Changing this file changes scorer_hash."""
from __future__ import annotations

import re

from inspect_ai.model import ChatMessageTool
from inspect_ai.scorer import Score, Target, scorer
from inspect_ai.solver import TaskState

from arena_evals.agents import _index, parse_final

_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


# ---------------------------------------------------------------- pure helpers
def normalize_answer(s: str) -> str:
    return " ".join(re.sub(r"[^\w\s]", "", s.casefold()).split())


def first_number(s: str) -> float | None:
    m = _NUM.search(s.replace("$", "").replace("€", "").replace("£", ""))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def match_answer(answer: str, reference: str, answer_kind: str) -> bool:
    if answer_kind == "exact":
        return normalize_answer(answer) == normalize_answer(reference)
    if answer_kind == "numeric":
        a, r = first_number(answer), first_number(reference)
        return a is not None and r is not None and abs(a - r) <= max(0.005 * abs(r), 0.01)
    return False


def citation_pr(cited: list[str], gold: list[str], corpus_ids: set[str]) -> tuple[float | None, float | None]:
    """precision = |cited & gold| / |cited| (None if nothing cited); recall = |cited & gold| / |gold| (None if no gold).
    Cited IDs that are not in the corpus stay in the denominator, so they count as wrong."""
    cited_set, gold_set = set(cited), set(gold)
    hit = len(cited_set & gold_set & corpus_ids)
    precision = hit / len(cited_set) if cited_set else None
    recall = hit / len(gold_set) if gold_set else None
    return precision, recall


def fetched_doc_ids(messages) -> set[str]:
    return {str(tc.arguments.get("id", "")).strip() for m in messages if m.role == "assistant" and m.tool_calls
            for tc in m.tool_calls if tc.function == "get_doc"}


def tool_call_count(messages) -> int:
    return sum(1 for m in messages if isinstance(m, ChatMessageTool))


def task_success(task: dict, sample_scores: dict, status: str) -> bool | None:
    """Binary success for one repeat (spec 1.5). None = judge error or unjudged free text (treated as missing)."""
    if status != "ok":
        return False
    if task["type"] == "unanswerable":
        return bool(sample_scores["abstention"])
    if sample_scores["abstention"]:
        return False
    if task["answer_kind"] == "free_text":
        j = sample_scores.get("judge")
        if sample_scores.get("judge_error") or j is None:
            return None
        correct = bool(j["correct"] and j["faithful"])
    else:
        correct = bool(sample_scores["answer_match"])
    return correct and (sample_scores["citation_recall"] or 0) > 0


# ---------------------------------------------------------------- Inspect scorers
@scorer(metrics=[])
def answer_match():
    async def score(state: TaskState, target: Target) -> Score:
        task, f = state.metadata, parse_final(state.output.completion)
        ok = bool(f and task["reference"] and match_answer(f.answer, task["reference"], task["answer_kind"]))
        return Score(value=ok, answer=f.answer if f else None)

    return score


@scorer(metrics=[])
def citation():
    async def score(state: TaskState, target: Target) -> Score:
        f = parse_final(state.output.completion)
        p, r = citation_pr(f.citations if f else [], state.metadata["gold_doc_ids"], set(_index().docs))
        return Score(value={"precision": p, "recall": r})

    return score


@scorer(metrics=[])
def gold_retrieval():
    async def score(state: TaskState, target: Target) -> Score:
        return Score(value=bool(fetched_doc_ids(state.messages) & set(state.metadata["gold_doc_ids"])))

    return score


@scorer(metrics=[])
def abstention():
    async def score(state: TaskState, target: Target) -> Score:
        f = parse_final(state.output.completion)
        return Score(value=bool(f and f.abstain))

    return score
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
pytest tests/test_scorers.py -v
```

Expected: PASS (`29 passed`).

- [ ] **Step 5: Commit**

Run:

```bash
git add arena_evals/scorers.py tests/test_scorers.py
git commit -m "feat(scorers): answer match, citation P/R, gold retrieval, abstention, success rule"
```

### Task 10: Runner: phase A generate, phase B score, results JSONL + manifest, mock end-to-end

**Files:**
- Create: `arena_evals/run.py`
- Test: `tests/test_e2e_mock.py`
- Test: `tests/test_results.py`

**Interfaces:**
- Consumes: `agents.arena_agent`, `agents.parse_final`, `agents.usage_cost`, `agents.CostCapExceeded`; scorers (Task 9); `datasets.load/dataset_hash/write_jsonl/label_noise_floor`; `config.Config`, `prompt_hash`, `scorer_hash`
- Produces: `run.generate(split, variant, prompts_dir, dataset, cfg, *, model=None, log_dir=None, run_id=None, epochs=None, cache=None, limit=None) -> Path` (.eval log; raises `CostCapExceeded`)
- Produces: `run.score(log, dataset, cfg, *, out_dir=None, cache=None) -> RunOutput(results_path, manifest)` (Task 16 adds `judge_model=None`)
- Produces: `run.sample_status(sample, max_tool_calls) -> str` (`ok | agent_error | timeout | limit | format_error`)
- Produces: `run.read_results(path) -> list[dict]`, `run.task_means(rows) -> dict[str, float | None]`, `run.new_run_id() -> str`, `run.git_sha(cwd) -> str`
- Produces: `runs/<run_id>/results.jsonl` (one line per sample, spec 2.1 schema) and `runs/<run_id>/manifest.json`

- [ ] **Step 1: Write the failing tests**

The mock agent is Inspect's `mockllm/model` with a callable `custom_outputs`: it sees the conversation, so each question gets its scripted tool call and FINAL line even when samples run concurrently.

Create `tests/test_e2e_mock.py`:

```python
import json
from pathlib import Path

import pytest
from inspect_ai.model import ModelOutput, get_model

from arena_evals import config, datasets, run
from arena_evals.datasets import TaskRecord

M = "mockllm/model"
FINAL = 'FINAL: {"answer": "%s", "citations": %s, "abstain": %s}'

TASKS = [
    TaskRecord(id="e2e-0001", input="How many days of PTO do full-time employees accrue per year?", reference="20 days",
               gold_doc_ids=["HR-001"], type="lookup", answer_kind="exact", tags=["lookup", "hr"], split="dev"),
    TaskRecord(id="e2e-0002", input="Who is the CTO of Halcyon Robotics?", reference="Dev Okafor",
               gold_doc_ids=["ORG-001"], type="multi_hop", answer_kind="exact", tags=["multi_hop", "org"], split="dev"),
    TaskRecord(id="e2e-0003", input="What is the monthly price of the Pro tier for 25 seats?", reference="4475 USD",
               gold_doc_ids=["PRC-006"], type="arithmetic", answer_kind="numeric", tags=["arithmetic", "pricing"],
               split="dev"),
    TaskRecord(id="e2e-0004", input="How many weeks of parental leave do employees get?", reference="16 weeks",
               gold_doc_ids=["HR-009"], type="conflicting", answer_kind="exact", tags=["conflicting", "hr"], split="dev"),
    TaskRecord(id="e2e-0005", input="What is the dental insurance deductible?", reference=None, gold_doc_ids=[],
               type="unanswerable", answer_kind="abstain", tags=["unanswerable"], split="dev"),
    TaskRecord(id="e2e-0006", input="What is the hotel cap per night?", reference="220 USD",
               gold_doc_ids=["HR-004"], type="lookup", answer_kind="exact", tags=["lookup", "hr"], split="dev"),
]

GOOD = {  # question -> (doc to fetch, FINAL line)
    TASKS[0].input: ("HR-001", FINAL % ("20 days", '["HR-001"]', "false")),
    TASKS[1].input: ("ORG-001", FINAL % ("Dev Okafor", '["ORG-001"]', "false")),
    TASKS[2].input: ("PRC-006", FINAL % ("4,475 USD", '["PRC-006"]', "false")),
    TASKS[3].input: ("HR-009", FINAL % ("16 weeks", '["HR-009"]', "false")),
    TASKS[4].input: ("HR-001", FINAL % ("", "[]", "true")),
    TASKS[5].input: ("HR-004", "The cap is 220 USD."),  # no FINAL line -> format_error
}


def scripted_agent(script: dict):
    """mockllm callable: first turn fetches a doc via get_doc, second turn answers with the scripted text."""
    def reply(messages, tools, tool_choice, cfg):
        question = next(m.text for m in messages if m.role == "user")
        doc, final = script[question]
        if not any(m.role == "tool" for m in messages):
            return ModelOutput.for_tool_call(M, "get_doc", {"id": doc})
        return ModelOutput.from_content(M, f"Answer.\n{final}")
    return get_model(M, custom_outputs=reply, memoize=False)


@pytest.fixture
def cfg():
    return config.load()


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    return datasets.write_jsonl(tmp_path / "e2e.jsonl", TASKS)


def run_variant(cfg, dataset, tmp_path, script, name):
    log = run.generate("dev", "baseline", cfg.prompts_dir, dataset, cfg, model=scripted_agent(script),
                       log_dir=tmp_path / f"logs-{name}", epochs=2, cache=False)
    return run.score(log, dataset, cfg, out_dir=tmp_path / name)


def test_generate_and_score_write_results_and_manifest(cfg, dataset, tmp_path):
    out = run_variant(cfg, dataset, tmp_path, GOOD, "base")
    rows = run.read_results(out.results_path)
    assert len(rows) == 12  # 6 tasks x 2 epochs
    required = {"run_id", "variant", "task_id", "repeat", "status", "answer", "citations", "abstain", "scores",
                "success", "latency_s", "tokens_in", "tokens_out", "cost_usd", "trace_id"}
    assert all(set(r) == required for r in rows)
    by = {(r["task_id"], r["repeat"]): r for r in rows}
    assert sorted({r["repeat"] for r in rows}) == [0, 1]
    for tid in ("e2e-0001", "e2e-0002", "e2e-0003", "e2e-0004", "e2e-0005"):
        assert by[(tid, 0)]["status"] == "ok" and by[(tid, 0)]["success"] is True, by[(tid, 0)]
    assert by[("e2e-0006", 0)]["status"] == "format_error" and by[("e2e-0006", 0)]["success"] is False
    assert by[("e2e-0001", 0)]["scores"]["gold_retrieval"] is True
    assert by[("e2e-0003", 0)]["scores"]["citation_precision"] == 1.0
    m = json.loads((tmp_path / "base" / "manifest.json").read_text())
    for key in ("run_id", "git_sha", "variant", "split", "dataset_hash", "prompt_hash", "rubric_hash", "scorer_hash",
                "agent_model", "judge_model", "agent_temperature", "judge_temperature", "k", "seeds",
                "inspect_version", "limits", "cost_usd", "label_noise_floor", "config"):
        assert key in m, key
    assert m["k"] == 2 and m["dataset_hash"] == datasets.dataset_hash(dataset)
```

Create `tests/test_results.py`:

```python
from arena_evals import run


def row(task, success, judge_error=False, trace="t"):
    return {"task_id": task, "success": success, "scores": {"judge_error": judge_error}, "trace_id": trace}


def test_task_means_missing_repeat_and_judge_errors():
    rows = [row("a", True), row("a", False), row("a", None, True),   # judge error = missing
            row("b", True),                                           # only 1 of k repeats present
            row("c", None, True), row("c", None, True)]               # no valid repeat
    assert run.task_means(rows) == {"a": 0.5, "b": 1.0, "c": None}
```

- [ ] **Step 2: Run them to verify they fail**

Run:

```bash
pytest tests/test_e2e_mock.py tests/test_results.py -v
```

Expected: FAIL, with `ImportError: cannot import name 'run' from 'arena_evals'` in the output.

- [ ] **Step 3: Create the runner**

Limits map to Inspect's per-sample `message_limit`, `token_limit` and `time_limit`; the tool-call cap is applied when scoring (`tool_call_count > max_tool_calls` gives status `limit`), with `message_limit` as the hard stop. `rubric_hash`, `judge_model` and `judge_temperature` are `None` in the manifest until Task 16 wires in the judge.

Create `arena_evals/run.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
pytest tests/test_e2e_mock.py tests/test_results.py -v
```

Expected: PASS (`2 passed`).

- [ ] **Step 5: (HUMAN) One real end-to-end smoke run**

Run (needs ANTHROPIC_API_KEY and datasets/dev.jsonl from Task 7; spend capped at $1):

```bash
python -m arena_evals run --split dev --variant baseline --limit 5 --epochs 1 --max-usd 1
```

Expected: a line `task success XX.X% (95% CI a to b), n_tasks=N, samples=5` (free-text tasks are missing until Task 16, so N can be below 5), a line `cost $0.0X, wall time Ns, judge errors 0`, and a `runs/<run_id>/results.jsonl` path. Open that file and check each line has the spec 2.1 fields. If the command exits 3 with `error: cost cap hit`, the cap worked; raise `--max-usd`.

- [ ] **Step 6: Commit**

Run:

```bash
git add arena_evals/run.py tests/test_e2e_mock.py tests/test_results.py
git commit -m "feat(run): two-phase runner writing results.jsonl and manifest, mock e2e test"
```

## Milestone M3: Stats + simulation harness

### Task 11: Bootstrap CI, paired bootstrap, gate rule, MDE, per-tag BH

**Files:**
- Create: `arena_evals/stats/bootstrap.py`
- Test: `tests/test_stats.py`

**Interfaces:**
- Consumes: `stats.CI`
- Produces: `bootstrap_ci(x, n_resamples=10_000, seed=0, alpha=0.05) -> CI`
- Produces: `paired_bootstrap(d, n_resamples=10_000, seed=0) -> PairedResult` (`delta`, `ci95`, `upper_975`, `p_neg`, `mde`, `n`, `sd`, `boot`; `as_dict()` in pts). Units: fractions (0.02 = 2 pts).
- Produces: `gate_decision(r, eps_pts=2.0, upper_q=0.975) -> Literal["pass", "warn", "block"]`
- Produces: `mde(sd, n, alpha_one_sided=0.025, power=0.80) -> float` (fraction)
- Produces: `benjamini_hochberg(pvals, q=0.05) -> list[bool]`, `per_tag(d_by_tag, n_resamples=10_000, seed=0, q=0.05) -> list[dict]`
- Produces: The spec writes `seed: int` without a default after defaulted parameters, which is not valid Python; `seed` defaults to 0 and callers always pass `cfg.eval["seed"]`.

- [ ] **Step 1: Write the failing test**

Create `tests/test_stats.py`:

```python
import numpy as np
import pytest

from arena_evals.stats.bootstrap import (PairedResult, benjamini_hochberg, bootstrap_ci, gate_decision, mde,
                                         paired_bootstrap, per_tag)


def _result(delta: float, boot: list[float]) -> PairedResult:
    return PairedResult(delta=delta, ci95=(0.0, 0.0), upper_975=0.0, p_neg=0.0, mde=0.0, n=300, sd=0.45,
                        boot=np.asarray(boot, dtype=float))


def test_bootstrap_ci_known_data_is_seeded_and_brackets_mean():
    x = np.array([0.0, 1.0] * 150)
    a = bootstrap_ci(x, n_resamples=10_000, seed=7)
    b = bootstrap_ci(x, n_resamples=10_000, seed=7)
    assert a == b
    assert a.point == pytest.approx(0.5)
    # SE of a 0/1 mean at p=0.5, n=300 is 0.0289; 95% half-width ~0.057
    assert a.lo == pytest.approx(0.443, abs=0.01)
    assert a.hi == pytest.approx(0.557, abs=0.01)


def test_mde_matches_prd():
    assert round(mde(0.45, 300) * 100, 1) == 7.3


def test_paired_bootstrap_fields():
    rng = np.random.default_rng(0)
    d = rng.normal(-0.08, 0.45, 300)
    r = paired_bootstrap(d, n_resamples=10_000, seed=1)
    assert r.n == 300
    assert r.ci95[0] < r.delta < r.ci95[1]
    assert r.upper_975 == pytest.approx(r.ci95[1])
    assert 0.0 <= r.p_neg <= 1.0
    assert r.as_dict()["delta_pts"] == pytest.approx(r.delta * 100)


def test_gate_block_at_exactly_minus_two_pts():
    assert gate_decision(_result(-0.02, [-0.03] * 1000)) == "block"


def test_gate_warn_when_upper_bound_exactly_zero():
    boot = [-0.05] * 900 + [0.0] * 100          # 97.5th percentile is exactly 0.0
    assert gate_decision(_result(-0.03, boot)) == "warn"


def test_gate_pass_when_drop_smaller_than_eps():
    assert gate_decision(_result(-0.019, [-0.03] * 1000)) == "pass"


def test_gate_block_via_bootstrap_on_big_drop():
    rng = np.random.default_rng(3)
    r = paired_bootstrap(rng.normal(-0.12, 0.45, 300), seed=3)
    assert gate_decision(r) == "block"


def test_all_ties_zero_variance_passes_without_nan():
    r = paired_bootstrap(np.zeros(300), seed=0)
    assert (r.delta, r.sd, r.mde, r.upper_975, r.p_neg) == (0.0, 0.0, 0.0, 0.0, 0.0)
    assert gate_decision(r) == "pass"


def test_constant_negative_deltas_block():
    r = paired_bootstrap(np.full(300, -0.05), seed=0)
    assert gate_decision(r) == "block"


def test_empty_deltas_raise():
    with pytest.raises(ValueError):
        paired_bootstrap(np.array([]))


def test_benjamini_hochberg():
    assert benjamini_hochberg([0.001, 0.02, 0.04, 0.5], q=0.05) == [True, True, False, False]
    assert benjamini_hochberg([]) == []


def test_per_tag_flags_only_drops():
    rows = per_tag({"pricing": np.full(40, -0.3), "hr": np.full(40, 0.3)}, n_resamples=2000, seed=0)
    by = {r["tag"]: r for r in rows}
    assert by["pricing"]["flagged"] is True
    assert by["hr"]["flagged"] is False
```

- [ ] **Step 2: Run it to verify it fails**

Run:

```bash
pytest tests/test_stats.py -v
```

Expected: FAIL, with `ModuleNotFoundError: No module named 'arena_evals.stats.bootstrap'` in the output.

- [ ] **Step 3: Create the bootstrap module**

`gate_decision` compares `delta * 100 <= -eps_pts + 1e-9`, so an exact -2.0 pts counts as reaching epsilon despite float error; the upper bound is compared with a strict `< 0`, so an upper bound of exactly 0 warns instead of blocking.

Create `arena_evals/stats/bootstrap.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
pytest tests/test_stats.py -v
```

Expected: PASS (`12 passed`).

- [ ] **Step 5: Commit**

Run:

```bash
git add arena_evals/stats/bootstrap.py tests/test_stats.py
git commit -m "feat(stats): percentile + paired bootstrap, gate rule, MDE, BH per-tag"
```

### Task 12: Paired deltas over real harness output, known verdicts, `compare` CLI

**Files:**
- Modify: `arena_evals/run.py`
- Modify: `tests/test_e2e_mock.py`
- Modify: `tests/test_results.py`
- Modify: `tests/test_cli.py`

**Interfaces:**
- Consumes: `run.task_means`, `stats.bootstrap.paired_bootstrap`, `gate_decision`
- Produces: `run.paired_deltas(base_rows, cand_rows) -> tuple[list[str], np.ndarray, list[str]]` (task_ids, d = cand - base, dropped task_ids)
- Produces: CLI `compare --base A.jsonl --cand B.jsonl` printing delta, both intervals, P(delta<0), SD(d), MDE and verdict

- [ ] **Step 1: Write the failing tests**

Append to the end of `tests/test_results.py`:

```python
def test_paired_deltas_drops_tasks_without_valid_repeat_on_either_side():
    base = [row("a", True), row("b", True), row("c", True)]
    cand = [row("a", False), row("b", None, True), row("d", True)]
    ids, d, dropped = run.paired_deltas(base, cand)
    assert ids == ["a"] and d.tolist() == [-1.0] and dropped == ["b", "c", "d"]
```

Append to the end of `tests/test_e2e_mock.py`:

```python
# ---------------------------------------------------------------- M3: paired comparison on real harness output
BAD = dict(GOOD) | {
    TASKS[0].input: ("HR-001", FINAL % ("25 days", '["HR-001"]', "false")),
    TASKS[1].input: ("ORG-001", FINAL % ("Mara Lindqvist", '["ORG-001"]', "false")),
    TASKS[2].input: ("PRC-001", FINAL % ("4975 USD", '["PRC-001"]', "false")),
    TASKS[3].input: ("HR-002", FINAL % ("12 weeks", '["HR-002"]', "false")),
    TASKS[4].input: ("HR-001", FINAL % ("500 USD", '["HR-999"]', "false")),  # hallucinated citation, no abstain
}


def test_e2e_known_verdicts(cfg, dataset, tmp_path):
    from arena_evals.stats.bootstrap import gate_decision, paired_bootstrap

    base = run.read_results(run_variant(cfg, dataset, tmp_path, GOOD, "base").results_path)
    bad = run.read_results(run_variant(cfg, dataset, tmp_path, BAD, "bad").results_path)
    same = run.read_results(run_variant(cfg, dataset, tmp_path, GOOD, "same").results_path)
    hall = next(r for r in bad if r["task_id"] == "e2e-0005")
    assert hall["success"] is False and hall["scores"]["citation_precision"] == 0.0
    assert gate_decision(paired_bootstrap(run.paired_deltas(base, bad)[1], seed=cfg.eval["seed"])) == "block"
    assert gate_decision(paired_bootstrap(run.paired_deltas(base, same)[1], seed=cfg.eval["seed"])) == "pass"
```

Append to the end of `tests/test_cli.py`:

```python
def test_compare_prints_delta_ci_and_verdict(tmp_path, capsys):
    from arena_evals import datasets

    def rows(successes):
        return [{"task_id": f"t{i}", "repeat": 0, "success": s, "scores": {"judge_error": False}}
                for i, s in enumerate(successes)]
    base = datasets.write_jsonl(tmp_path / "b.jsonl", rows([True] * 100))
    cand = datasets.write_jsonl(tmp_path / "c.jsonl", rows([True] * 80 + [False] * 20))
    assert main(["compare", "--base", str(base), "--cand", str(cand)]) == 0
    out = capsys.readouterr().out
    assert "delta -20.00 pts" in out and "verdict: block" in out
```

- [ ] **Step 2: Run them to verify they fail**

Run:

```bash
pytest tests/test_results.py tests/test_e2e_mock.py tests/test_cli.py -v
```

Expected: FAIL, with `AttributeError: module 'arena_evals.run' has no attribute 'paired_deltas'` in the output.

- [ ] **Step 3: Add `paired_deltas` to the runner**

Append to the end of `arena_evals/run.py`:

```python
def paired_deltas(base_rows: list[dict], cand_rows: list[dict]) -> tuple[list[str], np.ndarray, list[str]]:
    """(task_ids, d = cand - base, dropped task_ids). Tasks without a valid repeat on either side are dropped."""
    b, c = task_means(base_rows), task_means(cand_rows)
    ids = sorted(set(b) | set(c))
    keep = [t for t in ids if b.get(t) is not None and c.get(t) is not None]
    return keep, np.array([c[t] - b[t] for t in keep]), [t for t in ids if t not in keep]
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
pytest tests/test_results.py tests/test_e2e_mock.py tests/test_cli.py -v
```

Expected: PASS (`6 passed`).

- [ ] **Step 5: Commit**

Run:

```bash
git add arena_evals/run.py tests/test_results.py tests/test_e2e_mock.py tests/test_cli.py
git commit -m "feat(stats): paired per-task deltas, e2e verdict test, compare CLI"
```

### Task 13: Power and coverage simulation harness

**Files:**
- Create: `arena_evals/stats/simulate.py`
- Test: `tests/test_stats_sim.py`

**Interfaces:**
- Consumes: `paired_bootstrap`, `gate_decision`; `configs/eval.yaml: sim.sd`
- Produces: `simulate_gate(true_delta_pts, sd, n, trials, n_resamples=2000, eps_pts=2.0, seed=0) -> float` (block rate)
- Produces: `simulate_coverage(true_delta_pts, sd, n, trials, seed=0, n_resamples=2000) -> float` (spec signature plus an `n_resamples` keyword; 10k resamples x 1000 trials would take minutes per point)
- Produces: CLI `simulate [--sd X | --from-results BASE CAND] [--n 300] [--trials 1000]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_stats_sim.py`:

```python
import pytest

from arena_evals import config
from arena_evals.stats.simulate import simulate_coverage, simulate_gate

SD = config.load().eval["sim"]["sd"]


@pytest.mark.slow
def test_power_at_minus_8_pts():
    assert simulate_gate(-8.0, SD, 300, trials=1000, n_resamples=2000, eps_pts=2.0, seed=1) >= 0.80


@pytest.mark.slow
def test_false_block_at_zero():
    assert simulate_gate(0.0, SD, 300, trials=1000, n_resamples=2000, eps_pts=2.0, seed=2) <= 0.05


@pytest.mark.slow
def test_ci_coverage():
    assert 0.935 <= simulate_coverage(-5.0, SD, 300, trials=1000, seed=3) <= 0.965


def test_simulation_smoke():
    rate = simulate_gate(-30.0, 0.45, 300, trials=20, n_resamples=500, seed=0)
    assert rate == 1.0
```

- [ ] **Step 2: Run it to verify it fails**

Run:

```bash
pytest tests/test_stats_sim.py -v -m "slow or not slow"
```

Expected: FAIL, with `ModuleNotFoundError: No module named 'arena_evals.stats.simulate'` in the output.

- [ ] **Step 3: Create the simulation module**

Create `arena_evals/stats/simulate.py`:

```python
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
```

- [ ] **Step 4: Run the fast smoke test**

Run:

```bash
pytest tests/test_stats_sim.py -v
```

Expected: PASS (`1 passed, 3 deselected`).

- [ ] **Step 5: Run the slow simulations**

Run (about 40 s):

```bash
pytest tests/test_stats_sim.py -v -m slow
```

Expected: `3 passed, 1 deselected`. These assert, at SD 0.45 and n = 300: block rate >= 80% at -8 pts, <= 5% at 0, and 95% CI coverage in [93.5%, 96.5%] over 1000 trials.

- [ ] **Step 6: Print the full power table**

Run:

```bash
python -m arena_evals simulate --trials 1000
```

Expected output:

```text
SD=0.450 n=300 trials=1000 eps=2.0 pts
true delta -8 pts: blocked 87.x%
true delta -7 pts: blocked 7x.x%
true delta -6 pts: blocked 6x.x%
true delta -5 pts: blocked 4x.x% to 5x.x%
true delta +0 pts: blocked 2.x% to 3.x%
95% CI coverage at -5 pts: 94.x% to 95.x%
```

Observed when this plan was built (seed 1): -8: 87.4%, -7: 78.2%, -6: 66.9%, -5: 50.3%, 0: 2.8%, coverage 94.9%, in line with the PRD's pre-build Monte Carlo (88.5 / 75.6 / 66.0 / 49.0 / 2.2). Exact figures depend on the seed; Task 17 re-runs this with the measured SD.

- [ ] **Step 7: Commit**

Run:

```bash
git add arena_evals/stats/simulate.py tests/test_stats_sim.py
git commit -m "feat(stats): Monte Carlo power and CI coverage harness"
```

## Milestone M4: Judge

### Task 14: Phoenix tracing module and per-sample agent spans

**Files:**
- Create: `arena_evals/tracing.py`
- Modify: `arena_evals/agents.py`
- Test: `tests/test_tracing.py`

**Interfaces:**
- Consumes: `agents.arena_agent` (Task 4)
- Produces: `tracing.init(working_dir=None, endpoint=None) -> None` (registers an OTLP exporter to `$PHOENIX_COLLECTOR_ENDPOINT` or http://localhost:6006 and instruments the Anthropic SDK)
- Produces: `tracing.sample_span(name="arena.sample", **attrs)` context manager yielding the 32-hex trace ID ("" when off); attributes are written as `arena.<key>`
- Produces: `tracing.enabled() -> bool`, `tracing.flush() -> None`
- Produces: `agents.arena_agent(variant, prompts_dir=str(PROMPTS_DIR), cache=True, span_attrs=None)`; the trace ID is stored in `state.metadata["trace_id"]`

- [ ] **Step 1: Write the failing test**

Create `tests/test_tracing.py`:

```python
import pytest
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from arena_evals import tracing


@pytest.fixture
def spans(monkeypatch):
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(tracing, "_TRACER", provider.get_tracer("test"))
    return exporter


def test_sample_span_off_yields_empty_id(monkeypatch):
    monkeypatch.setattr(tracing, "_TRACER", None)
    with tracing.sample_span(task_id="t") as tid:
        assert tid == ""


def test_sample_span_sets_arena_attributes(spans):
    with tracing.sample_span(run_id="r1", task_id="gate-0001", repeat=0, model="m", skipped=None) as tid:
        assert len(tid) == 32
    (span,) = spans.get_finished_spans()
    assert dict(span.attributes) == {"arena.run_id": "r1", "arena.task_id": "gate-0001", "arena.repeat": 0,
                                     "arena.model": "m"}
```

- [ ] **Step 2: Run it to verify it fails**

Run:

```bash
pytest tests/test_tracing.py -v
```

Expected: FAIL, with `ImportError: cannot import name 'tracing' from 'arena_evals'` in the output.

- [ ] **Step 3: Create the tracing module**

`phoenix.otel.register` exports spans to a running Phoenix server; it does not write a database itself. The SQLite trace DB is written by `phoenix serve` into its `PHOENIX_WORKING_DIR` (verified in Task 8), which is why the CI workflow starts `phoenix serve` before the gate and uploads that directory.

Create `arena_evals/tracing.py`:

```python
"""The only module that touches Phoenix/OpenTelemetry. Spans go to a running `phoenix serve`, whose
PHOENIX_WORKING_DIR holds the SQLite trace DB that CI uploads as an artifact."""
from __future__ import annotations

import os
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

_TRACER = None
_PROVIDER = None


def init(working_dir: Path | str | None = None, endpoint: str | None = None) -> None:
    """Register an OTLP exporter to Phoenix and instrument the Anthropic SDK. No-op if already initialised."""
    global _TRACER, _PROVIDER
    if _TRACER is not None:
        return
    from openinference.instrumentation.anthropic import AnthropicInstrumentor
    from phoenix.otel import register

    if working_dir:
        os.environ["PHOENIX_WORKING_DIR"] = str(working_dir)  # read by `phoenix serve`, not by the exporter
    endpoint = endpoint or os.environ.get("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006") + "/v1/traces"
    _PROVIDER = register(project_name="arena-evals", endpoint=endpoint, batch=True, verbose=False)
    AnthropicInstrumentor().instrument(tracer_provider=_PROVIDER)
    _TRACER = _PROVIDER.get_tracer("arena_evals")


def enabled() -> bool:
    return _TRACER is not None


@contextmanager
def sample_span(name: str = "arena.sample", **attrs) -> Iterator[str]:
    """Open a span with arena.<key> attributes; yields the 32-hex trace ID ("" when tracing is off)."""
    if _TRACER is None:
        yield ""
        return
    clean = {f"arena.{k}": (v if isinstance(v, (str, bool, int, float)) else str(v)) for k, v in attrs.items()
             if v is not None}
    with _TRACER.start_as_current_span(name, attributes=clean) as span:
        yield format(span.get_span_context().trace_id, "032x")


def flush() -> None:
    if _PROVIDER is not None:
        _PROVIDER.force_flush()
```

- [ ] **Step 4: Import tracing in `agents.py`**

In `arena_evals/agents.py`, replace:

```python
from arena_evals import corpus
```

with:

```python
from arena_evals import corpus, tracing
```

- [ ] **Step 5: Wrap the solver in a span**

In `arena_evals/agents.py`, replace:

```python
@solver
def arena_agent(variant: str, prompts_dir: str = str(PROMPTS_DIR), cache: bool = True) -> Solver:
    # ponytail: inserts ChatMessageSystem directly; inspect's system_message() str-formats and mangles the JSON braces
    prompt = load_prompt(variant, prompts_dir)
    tools = use_tools(search_docs(), get_doc(), calculate())
    loop = generate(tool_calls="loop", cache=CachePolicy(expiry=None) if cache else False)

    async def solve(state: TaskState, generate_fn: Generate) -> TaskState:
        METER.check()
        state.messages.insert(0, ChatMessageSystem(content=prompt))
        state = await tools(state, generate_fn)
        return await loop(state, generate_fn)

    return solve
```

with:

```python
@solver
def arena_agent(variant: str, prompts_dir: str = str(PROMPTS_DIR), cache: bool = True,
                span_attrs: dict | None = None) -> Solver:
    # ponytail: inserts ChatMessageSystem directly; inspect's system_message() str-formats and mangles the JSON braces
    prompt = load_prompt(variant, prompts_dir)
    tools = use_tools(search_docs(), get_doc(), calculate())
    loop = generate(tool_calls="loop", cache=CachePolicy(expiry=None) if cache else False)

    async def solve(state: TaskState, generate_fn: Generate) -> TaskState:
        METER.check()
        with tracing.sample_span("arena.sample", **(span_attrs or {}), task_id=str(state.sample_id),
                                 repeat=state.epoch - 1) as trace_id:
            state.metadata["trace_id"] = trace_id
            state.messages.insert(0, ChatMessageSystem(content=prompt))
            state = await tools(state, generate_fn)
            state = await loop(state, generate_fn)
        return state

    return solve
```

- [ ] **Step 6: Run the tests to verify they pass**

Run:

```bash
pytest tests/test_tracing.py tests/test_e2e_mock.py tests/test_solver.py -v
```

Expected: PASS (`6 passed`).

- [ ] **Step 7: Commit**

Run:

```bash
git add arena_evals/tracing.py arena_evals/agents.py tests/test_tracing.py
git commit -m "feat(tracing): Phoenix tracing module and per-sample agent spans"
```

### Task 15: Schema-validated pointwise judge with retry-once and cost check

**Files:**
- Create: `prompts/judge/system.md`
- Create: `prompts/judge/rubric.md`
- Modify: `arena_evals/scorers.py`
- Test: `tests/test_judge.py`

**Interfaces:**
- Consumes: `agents.METER`, `agents.usage_cost`, `agents._index`, `agents.parse_final`, `tracing.sample_span`
- Produces: `scorers.JudgeVerdict` (pydantic strict: `correct: bool`, `faithful: bool`, `reason: str`)
- Produces: `scorers.judge_messages(rubric_dir, task_input, reference, answer, cited_docs) -> list`
- Produces: `scorers.parse_verdict(text) -> JudgeVerdict` (raises `ValueError`)
- Produces: `async scorers.judge_answer(model, rubric_dir, task_input, reference, answer, cited_docs, cache=True, prices=None) -> tuple[JudgeVerdict | None, float]` (None = judge error after one retry; float = USD)
- Produces: Inspect `@scorer` `scorers.judge(model, rubric_dir=str(RUBRIC_DIR), temperature=0.0, seed=0, cache=True)` with value `{judged, correct, faithful, error, cost_usd}` and the reason as `explanation`; `scorers.RUBRIC_DIR`
- Produces: `rubric_hash = sha256(system.md bytes + rubric.md bytes)` via `config.rubric_hash`

- [ ] **Step 1: Write the failing test**

Create `tests/test_judge.py`:

````python
import asyncio

import pytest
from inspect_ai.model import ModelOutput, get_model

from arena_evals import agents, config
from arena_evals.agents import CostCapExceeded, CostMeter
from arena_evals.scorers import RUBRIC_DIR, judge_answer, parse_verdict

M = "mockllm/model"


def model(replies):
    it = iter(replies)
    return get_model(M, custom_outputs=lambda *a: ModelOutput.from_content(M, next(it)), memoize=False)


def ask(m):
    return asyncio.run(judge_answer(m, RUBRIC_DIR, "Q?", "ref", "ans", {"HR-001": "doc text"}, cache=False))


@pytest.fixture(autouse=True)
def meter():
    agents.set_meter(CostMeter(float("inf")))
    yield
    agents.set_meter(CostMeter(float("inf")))


def test_parse_verdict_accepts_fenced_json_and_rejects_bad_schema():
    v = parse_verdict('```json\n{"correct": true, "faithful": false, "reason": "r"}\n```')
    assert (v.correct, v.faithful, v.reason) == (True, False, "r")
    for bad in ('{"correct": "yes", "faithful": true, "reason": "r"}',   # strict bool
                '{"correct": true, "faithful": true}',                    # missing reason
                '{"correct": true, "faithful": true, "reason": "r", "score": 5}',
                "no json here"):
        with pytest.raises(ValueError):
            parse_verdict(bad)


def test_retry_once_then_valid():
    v, _ = ask(model(["nope", '{"correct": false, "faithful": true, "reason": "wrong value"}']))
    assert v.correct is False and v.reason == "wrong value"


def test_two_invalid_replies_is_judge_error():
    v, _ = ask(model(["nope", "still nope", '{"correct": true, "faithful": true, "reason": "never reached"}']))
    assert v is None


def test_judge_respects_cost_cap():
    agents.set_meter(CostMeter(0.0))
    with pytest.raises(CostCapExceeded):
        ask(model(['{"correct": true, "faithful": true, "reason": "r"}']))


def test_rubric_hash_covers_both_files():
    h = config.rubric_hash(RUBRIC_DIR)
    assert h == config.sha256_bytes((RUBRIC_DIR / "system.md").read_bytes(), (RUBRIC_DIR / "rubric.md").read_bytes())
````

- [ ] **Step 2: Run it to verify it fails**

Run:

```bash
pytest tests/test_judge.py -v
```

Expected: FAIL, with `ImportError: cannot import name 'RUBRIC_DIR'` in the output.

- [ ] **Step 3: Create the judge prompt and rubric**

Both files are hashed into `rubric_hash`. Any edit makes the judge cert stale (Task 20) and changes the baseline cache key (Task 25).

Create `prompts/judge/system.md`:

```markdown
You are a strict grader for an internal question-answering assistant at Halcyon Robotics. You compare the assistant's answer with a reference answer and with the documents the assistant cited. You reply with a single JSON object and nothing else.
```

Create `prompts/judge/rubric.md`:

```markdown
# Grading rubric (v1)

Grade the assistant answer on two independent criteria.

**correct**: true if the answer states the same facts as the reference answer. Wording, order and extra harmless detail do not matter. Numbers must match the reference (units may be written differently). A missing required fact, a wrong value, or a contradiction makes it false. Hedging between two values is false.

**faithful**: true if every factual claim in the answer is supported by the text of the cited documents shown below. A claim that is true but not supported by any cited document makes it false. If no valid documents were cited, faithful is false.

Length is not a criterion. Do not reward longer answers or penalise short ones.

Reply with exactly this JSON object and nothing else:

{"correct": true or false, "faithful": true or false, "reason": "one or two sentences"}
```

- [ ] **Step 4: Extend the imports of `scorers.py`**

In `arena_evals/scorers.py`, replace:

```python
import re

from inspect_ai.model import ChatMessageTool
from inspect_ai.scorer import Score, Target, scorer
from inspect_ai.solver import TaskState

from arena_evals.agents import _index, parse_final
```

with:

```python
import re
from pathlib import Path

from inspect_ai.model import (CachePolicy, ChatMessageAssistant, ChatMessageSystem, ChatMessageTool,
                              ChatMessageUser, GenerateConfig, get_model)
from inspect_ai.scorer import Score, Target, scorer
from inspect_ai.solver import TaskState
from pydantic import BaseModel, ConfigDict, ValidationError

from arena_evals import agents, tracing
from arena_evals.agents import _index, parse_final, usage_cost
from arena_evals.config import ROOT

RUBRIC_DIR = ROOT / "prompts" / "judge"
```

- [ ] **Step 5: Add the judge to `scorers.py`**

The judge runs only for `answer_kind == "free_text"` answers that parsed and did not abstain. Inspect calls scorers even for errored samples, so this guard also avoids paying for them.

Append to the end of `arena_evals/scorers.py`:

```python
# ---------------------------------------------------------------- LLM judge (M4)
class JudgeVerdict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    correct: bool
    faithful: bool
    reason: str


def judge_messages(rubric_dir: Path, task_input: str, reference: str | None, answer: str,
                   cited_docs: dict[str, str]) -> list:
    system = (rubric_dir / "system.md").read_text(encoding="utf-8")
    rubric = (rubric_dir / "rubric.md").read_text(encoding="utf-8")
    docs = "\n\n".join(f"<doc id=\"{i}\">\n{t}\n</doc>" for i, t in cited_docs.items()) or "(no valid cited documents)"
    user = (f"{rubric}\n\n## Question\n{task_input}\n\n## Reference answer\n{reference}\n\n"
            f"## Assistant answer\n{answer}\n\n## Documents the assistant cited\n{docs}\n")
    return [ChatMessageSystem(content=system), ChatMessageUser(content=user)]


def parse_verdict(text: str) -> JudgeVerdict:
    """Extract the outermost {...} and validate it. Raises ValueError (ValidationError is a ValueError)."""
    start, end = text.find("{"), text.rfind("}")
    if start < 0 or end < start:
        raise ValueError("no JSON object in judge output")
    return JudgeVerdict.model_validate_json(text[start:end + 1])


async def judge_answer(model, rubric_dir: Path, task_input: str, reference: str | None, answer: str,
                       cited_docs: dict[str, str], cache: bool = True, prices: dict | None = None
                       ) -> tuple[JudgeVerdict | None, float]:
    """Two attempts: on invalid output, retry once with the validation error appended. (None, cost) = judge error."""
    messages = judge_messages(rubric_dir, task_input, reference, answer, cited_docs)
    cost = 0.0
    for _ in range(2):
        agents.METER.check()
        out = await model.generate(messages, cache=CachePolicy(expiry=None) if cache else False)
        if out.usage:
            cost += usage_cost(out.usage, str(model), prices or agents.METER.prices)
        try:
            return parse_verdict(out.completion), cost
        except ValueError as e:
            messages = messages + [ChatMessageAssistant(content=out.completion),
                                   ChatMessageUser(content=f"Your reply failed validation: {e}\n"
                                                           "Reply with only the JSON object.")]
    return None, cost


@scorer(metrics=[])
def judge(model, rubric_dir: str = str(RUBRIC_DIR), temperature: float = 0.0, seed: int = 0, cache: bool = True):
    """Pointwise judge, only for free_text answers that did not abstain. Value keys: correct, faithful, error, cost_usd."""

    async def score(state: TaskState, target: Target) -> Score:
        task, f = state.metadata, parse_final(state.output.completion)
        if task["answer_kind"] != "free_text" or f is None or f.abstain:
            return Score(value={"judged": False, "correct": None, "faithful": None, "error": False, "cost_usd": 0.0})
        m = model if not isinstance(model, str) else get_model(
            model, config=GenerateConfig(temperature=temperature, seed=seed))
        docs = {c: _index().docs[c].text for c in f.citations if c in _index().docs}
        with tracing.sample_span("arena.judge", task_id=str(state.sample_id), repeat=state.epoch - 1, model=str(m)):
            verdict, cost = await judge_answer(m, Path(rubric_dir), task["input"], task["reference"], f.answer, docs, cache)
        if verdict is None:
            return Score(value={"judged": True, "correct": None, "faithful": None, "error": True, "cost_usd": cost})
        return Score(value={"judged": True, "correct": verdict.correct, "faithful": verdict.faithful, "error": False,
                            "cost_usd": cost}, explanation=verdict.reason)

    return score
```

- [ ] **Step 6: Run the tests to verify they pass**

Run:

```bash
pytest tests/test_judge.py tests/test_scorers.py -v
```

Expected: PASS (`34 passed`).

- [ ] **Step 7: Commit**

Run:

```bash
git add prompts/judge arena_evals/scorers.py tests/test_judge.py
git commit -m "feat(judge): pointwise judge with pydantic verdict, retry once, cost check"
```

### Task 16: Wire the judge, trace attributes and self-preference warning into the runner

**Files:**
- Create: `arena_evals/bias.py`
- Modify: `arena_evals/run.py`
- Modify: `tests/test_e2e_mock.py`
- Modify: `tests/test_tracing.py`
- Modify: `tests/test_judge.py`

**Interfaces:**
- Consumes: `scorers.judge` (Task 15), `tracing` (Task 14), `config.rubric_hash`
- Produces: `bias.model_family(model_id) -> str`, `bias.self_preference(cfg) -> bool` (Task 22 adds the verbosity tests)
- Produces: `run.score(log, dataset, cfg, *, judge_model=None, out_dir=None, cache=None) -> RunOutput`; manifest gains `rubric_hash`, `judge_model`, `judge_temperature`, judge seed, `judge_error_count`, `self_preference_warning`
- Produces: results rows carry `scores.judge = {correct, faithful, reason}` or `null` and `scores.judge_error`; `cost_usd` = agent + judge
- Produces: `ARENA_TRACE=1` makes `run.generate`/`run.score` call `tracing.init()`

- [ ] **Step 1: Write the failing tests**

Append to the end of `tests/test_e2e_mock.py`:

```python
# ---------------------------------------------------------------- M4: free-text tasks go through the judge
FREE = [
    TaskRecord(id="e2e-0101", input="What caused the Fleet Console outage?", reference="An expired TLS certificate",
               gold_doc_ids=["INC-001"], type="lookup", answer_kind="free_text", tags=["lookup", "incident"],
               split="dev"),
    TaskRecord(id="e2e-0102", input="What caused the export delay?", reference="A full disk on the export worker",
               gold_doc_ids=["INC-008"], type="lookup", answer_kind="free_text", tags=["lookup", "incident"],
               split="dev"),
    TaskRecord(id="e2e-0103", input="What caused the Leeds navigation fault?", reference="A map tile corruption",
               gold_doc_ids=["INC-002"], type="lookup", answer_kind="free_text", tags=["lookup", "incident"],
               split="dev"),
]
FREE_SCRIPT = {
    FREE[0].input: ("INC-001", FINAL % ("An expired TLS certificate took the console down.", '["INC-001"]', "false")),
    FREE[1].input: ("INC-008", FINAL % ("RETRY a full disk on the export worker", '["INC-008"]', "false")),
    FREE[2].input: ("INC-002", FINAL % ("BROKEN corrupted map tiles", '["INC-002"]', "false")),
}


def scripted_judge():
    """First reply for a RETRY answer is invalid (then valid on retry); BROKEN answers are always invalid."""
    def reply(messages, tools, tool_choice, cfg):
        text = "\n".join(m.text for m in messages)
        if "BROKEN" in text or ("RETRY" in text and "failed validation" not in text):
            return ModelOutput.from_content(M, "I think it is correct!")
        return ModelOutput.from_content(M, '{"correct": true, "faithful": true, "reason": "matches INC doc"}')
    return get_model(M, custom_outputs=reply, memoize=False)


def test_judge_scores_free_text_with_retry_and_error(cfg, tmp_path):
    ds = datasets.write_jsonl(tmp_path / "free.jsonl", FREE)
    out = run_variant(cfg, ds, tmp_path, FREE_SCRIPT, "free", judge_model=scripted_judge())
    by = {r["task_id"]: r for r in run.read_results(out.results_path) if r["repeat"] == 0}
    assert by["e2e-0101"]["scores"]["judge"] == {"correct": True, "faithful": True, "reason": "matches INC doc"}
    assert by["e2e-0101"]["success"] is True
    assert by["e2e-0102"]["scores"]["judge"]["correct"] is True        # passed on the retry
    assert by["e2e-0103"]["scores"]["judge_error"] is True               # failed twice
    assert by["e2e-0103"]["success"] is None                             # missing, not FAIL (I2)
    assert out.manifest["judge_error_count"] == 2                        # 1 task x 2 epochs
```

In `tests/test_e2e_mock.py`, replace:

```python
def run_variant(cfg, dataset, tmp_path, script, name):
    log = run.generate("dev", "baseline", cfg.prompts_dir, dataset, cfg, model=scripted_agent(script),
                       log_dir=tmp_path / f"logs-{name}", epochs=2, cache=False)
    return run.score(log, dataset, cfg, out_dir=tmp_path / name)
```

with:

```python
def run_variant(cfg, dataset, tmp_path, script, name, judge_model=M):
    log = run.generate("dev", "baseline", cfg.prompts_dir, dataset, cfg, model=scripted_agent(script),
                       log_dir=tmp_path / f"logs-{name}", epochs=2, cache=False)
    return run.score(log, dataset, cfg, judge_model=judge_model, out_dir=tmp_path / name, cache=False)
```

Append to the end of `tests/test_tracing.py`:

```python
def test_agent_spans_carry_hashes_and_trace_id_reaches_results(spans, tmp_path):
    from arena_evals import config, datasets, run
    from test_e2e_mock import GOOD, TASKS, run_variant

    cfg = config.load()
    ds = datasets.write_jsonl(tmp_path / "e2e.jsonl", TASKS[:1])
    out = run_variant(cfg, ds, tmp_path, GOOD, "traced")
    rows = run.read_results(out.results_path)
    sample_spans = [s for s in spans.get_finished_spans() if s.name == "arena.sample"]
    assert len(sample_spans) == 2
    attrs = dict(sample_spans[0].attributes)
    for key in ("arena.run_id", "arena.dataset_hash", "arena.prompt_hash", "arena.rubric_hash", "arena.model",
                "arena.task_id", "arena.repeat"):
        assert key in attrs, key
    ids = {format(s.context.trace_id, "032x") for s in sample_spans}
    assert {r["trace_id"] for r in rows} == ids
```

Append to the end of `tests/test_judge.py`:

```python
def test_self_preference_warning():
    from arena_evals import bias

    cfg = config.load()
    assert bias.model_family("anthropic/claude-sonnet-4-5-20250929") == "claude"
    cfg.models["agent"]["model"], cfg.models["judge"]["model"] = "anthropic/claude-a", "anthropic/claude-b"
    assert bias.self_preference(cfg) is True
    cfg.models["judge"]["model"] = "openai/gpt-5"
    assert bias.self_preference(cfg) is False
```

- [ ] **Step 2: Run them to verify they fail**

Run:

```bash
pytest tests/test_e2e_mock.py tests/test_tracing.py tests/test_judge.py -v
```

Expected: FAIL, with `TypeError: score() got an unexpected keyword argument 'judge_model'` in the output.

- [ ] **Step 3: Create the self-preference check**

Create `arena_evals/bias.py`:

```python
"""Judge bias checks: self-preference warning (M4) and verbosity tests (M6)."""
from __future__ import annotations

from arena_evals.config import Config


def model_family(model_id: str) -> str:
    """'anthropic/claude-sonnet-4-5-20250929' -> 'claude'."""
    return model_id.split("/")[-1].split("-")[0].lower()


def self_preference(cfg: Config) -> bool:
    """True (warn) when agent and judge models share a family prefix."""
    return model_family(cfg.models["agent"]["model"]) == model_family(cfg.models["judge"]["model"])
```

- [ ] **Step 4: Replace the runner with the judge-aware version**

Replace the entire contents of `arena_evals/run.py` with:

```python
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

from arena_evals import bias, datasets, tracing
from arena_evals.agents import CostCapExceeded, arena_agent, parse_final, usage_cost
from arena_evals.config import Config, prompt_hash, rubric_hash, scorer_hash
from arena_evals.scorers import (abstention, answer_match, citation, gold_retrieval, judge, task_success,
                                 tool_call_count)


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


def _maybe_trace() -> None:
    if os.environ.get("ARENA_TRACE") == "1":
        tracing.init()


def generate(split: str, variant: str, prompts_dir: Path, dataset: Path, cfg: Config, *, model=None,
             log_dir: Path | None = None, run_id: str | None = None, epochs: int | None = None,
             cache: bool | None = None, limit: int | None = None) -> Path:
    """Phase A. Runs the agent on every task x epochs; returns the .eval log path. Raises CostCapExceeded."""
    _maybe_trace()
    prompts_dir, dataset = Path(prompts_dir), Path(dataset)
    records = datasets.load(dataset)
    run_id = run_id or new_run_id()
    cache = cfg.eval["cache"] if cache is None else cache
    lim, agent = cfg.eval["limits"], cfg.models["agent"]
    meta = {"run_id": run_id, "split": split, "variant": variant, "git_sha": git_sha(prompts_dir),
            "dataset_hash": datasets.dataset_hash(dataset), "prompt_hash": prompt_hash(prompts_dir, variant),
            "k": epochs or cfg.eval["k"], "cache": cache}
    span_attrs = {"run_id": run_id, "dataset_hash": meta["dataset_hash"], "prompt_hash": meta["prompt_hash"],
                  "rubric_hash": rubric_hash(cfg.rubric_dir), "model": str(model or agent["model"])}
    task = Task(
        dataset=MemoryDataset([Sample(input=r.input, target=r.reference or "", id=r.id, metadata=r.model_dump())
                               for r in records]),
        solver=arena_agent(variant, str(prompts_dir), cache, span_attrs),
        epochs=meta["k"], message_limit=lim["message_limit"], token_limit=lim["token_limit"],
        time_limit=lim["time_limit"], name=f"arena-{split}-{variant}",
        config=GenerateConfig(temperature=agent["temperature"], seed=agent["seed"]),
    )
    log = inspect_eval(task, model=model or agent["model"], log_dir=str(log_dir or cfg.root / "logs"),
                       fail_on_error=False, max_connections=cfg.eval["max_connections"], limit=limit,
                       metadata=meta)[0]
    tracing.flush()
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


def score(log: Path, dataset: Path, cfg: Config, *, judge_model=None, out_dir: Path | None = None,
          cache: bool | None = None) -> RunOutput:
    """Phase B. Scores a phase-A log with THIS checkout's scorers and judge; writes results.jsonl + manifest.json."""
    _maybe_trace()
    ev = read_eval_log(str(log))
    records = {r.id: r for r in datasets.load(dataset)}
    jcfg, prices = cfg.models["judge"], cfg.models["prices"]
    cache = cfg.eval["cache"] if cache is None else cache
    jmodel = judge_model or jcfg["model"]
    scored = inspect_score(ev, [answer_match(), citation(), gold_retrieval(), abstention(),
                                judge(jmodel, str(cfg.rubric_dir), jcfg["temperature"], jcfg["seed"], cache)],
                           display="none")
    tracing.flush()
    meta = scored.eval.metadata or {}
    run_id = meta.get("run_id") or new_run_id()
    out_dir = Path(out_dir or cfg.root / "runs" / run_id)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for s in scored.samples:
        task = records[str(s.id)]
        sc = {k: v.value for k, v in (s.scores or {}).items()}
        status = sample_status(s, cfg.eval["limits"]["max_tool_calls"])
        j = sc["judge"]
        scores = {"answer_match": sc["answer_match"], "gold_retrieval": sc["gold_retrieval"],
                  "citation_precision": sc["citation"]["precision"], "citation_recall": sc["citation"]["recall"],
                  "abstention": sc["abstention"],
                  "judge": ({"correct": j["correct"], "faithful": j["faithful"],
                             "reason": s.scores["judge"].explanation or ""}
                            if j["judged"] and not j["error"] else None),
                  "judge_error": bool(j["error"])}
        f = parse_final(s.output.completion if s.output else "")
        agent_cost = sum(usage_cost(u, name, prices) for name, u in (s.model_usage or {}).items())
        rows.append({
            "run_id": run_id, "variant": meta.get("variant"), "task_id": task.id, "repeat": s.epoch - 1,
            "status": status, "answer": f.answer if f else None, "citations": f.citations if f else [],
            "abstain": f.abstain if f else False, "scores": scores,
            "success": task_success(task.model_dump(), scores, status),
            "latency_s": s.total_time, "tokens_in": sum(u.input_tokens for u in (s.model_usage or {}).values()),
            "tokens_out": sum(u.output_tokens for u in (s.model_usage or {}).values()),
            "cost_usd": agent_cost + j["cost_usd"], "trace_id": (s.metadata or {}).get("trace_id", ""),
        })
    results_path = datasets.write_jsonl(out_dir / "results.jsonl", rows)
    agent = cfg.models["agent"]
    manifest = {
        "run_id": run_id, "git_sha": meta.get("git_sha", "unknown"), "variant": meta.get("variant"),
        "split": meta.get("split"), "dataset_hash": datasets.dataset_hash(dataset),
        "generated_dataset_hash": meta.get("dataset_hash"), "prompt_hash": meta.get("prompt_hash"),
        "rubric_hash": rubric_hash(cfg.rubric_dir), "scorer_hash": scorer_hash(cfg.root),
        "agent_model": str(scored.eval.model), "judge_model": str(jmodel),
        "agent_temperature": agent["temperature"], "judge_temperature": jcfg["temperature"],
        "k": meta.get("k"), "seeds": {"agent": agent["seed"], "judge": jcfg["seed"], "bootstrap": cfg.eval["seed"]},
        "inspect_version": importlib.metadata.version("inspect-ai"), "limits": cfg.eval["limits"],
        "cache": meta.get("cache"), "n_samples": len(rows), "cost_usd": sum(r["cost_usd"] for r in rows),
        "judge_error_count": sum(r["scores"]["judge_error"] for r in rows),
        "self_preference_warning": bias.self_preference(cfg),
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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run:

```bash
pytest -q
```

Expected: PASS (`108 passed, 3 deselected`).

- [ ] **Step 6: Commit**

Run:

```bash
git add arena_evals/bias.py arena_evals/run.py tests/test_e2e_mock.py tests/test_tracing.py tests/test_judge.py
git commit -m "feat(run): judge scoring, judge-error accounting, trace attributes, self-preference flag"
```

### Task 17: (HUMAN) Measure runtime, cost and paired SD; set the cost cap and simulation SD

**Files:**
- Modify: `configs/eval.yaml`

**Interfaces:**
- Consumes: CLI `run`, `compare`, `simulate`; `datasets/dev.jsonl`; an API key
- Produces: Measured dev-run cost and wall time, a gate-run estimate, the measured paired SD, `cost.max_usd_per_gate` and `sim.sd` set from measurements, simulated power at the measured SD

- [ ] **Step 1: (HUMAN) Start Phoenix so the runs are traced**

In a second terminal (activate the venv there first), run:

```bash
export PHOENIX_WORKING_DIR=$PWD/.phoenix
phoenix serve
```

Keep it running for this task.

- [ ] **Step 2: (HUMAN) Run baseline, a cache-bypassed baseline repeat, and regressed on the dev split**

Run (needs ANTHROPIC_API_KEY, costs money; 100 tasks x 3 repeats each):

```bash
export ARENA_TRACE=1
python -m arena_evals run --split dev --variant baseline --max-usd 20
python -m arena_evals run --split dev --variant baseline --no-cache --max-usd 20
python -m arena_evals run --split dev --variant regressed --max-usd 20
```

Each prints `task success X% (95% CI a to b), n_tasks=..., samples=300`, then `cost $C, wall time Ts, judge errors J`, then its `results.jsonl` path. Record C and T of the first run. If any run shows judge errors above 2% of 300 (more than 6), fix the rubric wording in `prompts/judge/rubric.md` before going further.

- [ ] **Step 3: (HUMAN) Check the traces**

Open http://localhost:6006, project `arena-evals`. Confirm spans named `arena.sample` and `arena.judge` exist, carry `arena.run_id`, `arena.dataset_hash`, `arena.prompt_hash`, `arena.rubric_hash`, `arena.model`, `arena.task_id`, `arena.repeat`, and have Anthropic LLM child spans. Pick one `trace_id` from `results.jsonl` and find it in the UI.

- [ ] **Step 4: (HUMAN) Measure the paired SD**

Run (use the three results paths printed above):

```bash
python -m arena_evals compare --base <baseline results.jsonl> --cand <no-cache baseline results.jsonl>
python -m arena_evals compare --base <baseline results.jsonl> --cand <regressed results.jsonl>
```

Each prints `... SD(d)=S ...`. Take the larger S. The second line also gives a first estimate of the planted regression's effect (checked properly on the gate split in Task 29).

- [ ] **Step 5: (HUMAN) Set the cap and SD from the measurements**

A gate run on a cache miss is 2 sides x 300 tasks x 3 repeats = 6 times the dev run, so estimated gate cost = 6 x C and estimated gate runtime = 6 x T. In `configs/eval.yaml` set `cost.max_usd_per_gate` to 1.5 x (6 x C), rounded up to a whole dollar, and `sim.sd` to the larger S. Replace their comments with `# measured <date>: dev run $C in Ts; SD(d) S`. If 6 x T exceeds 900 s (the 15-minute target), set `max_connections: 20`, re-run the first command and re-measure.

- [ ] **Step 6: Re-run the simulations at the measured SD**

Run (after editing sim.sd):

```bash
python -m arena_evals simulate --trials 1000
pytest tests/test_stats_sim.py -m slow -v
```

Record the block rates. If the -8 pts row is below 80% (for example because the measured SD is well above 0.45), do NOT change epsilon or the bound: the slow test will fail, which is the signal. Record the measured MDE and note in the commit message that n must grow (PRD risk table); keep the slow tests marked `slow` so the default suite stays green.

- [ ] **Step 7: Commit**

Run (after the HUMAN steps above):

```bash
git add configs/eval.yaml
git commit -m "chore(config): cost cap and simulation SD from measured dev runs"
```

## Milestone M5: Calibration

### Task 18: Agreement metrics: kappa, AC1, PABAK, precision/recall, confusion, bootstrap CIs

**Files:**
- Modify: `arena_evals/stats/agreement.py`
- Modify: `tests/test_agreement.py`

**Interfaces:**
- Consumes: `stats.CI`
- Produces: `raw_agreement(a, b)`, `cohen_kappa(a, b)`, `gwet_ac1(a, b)`, `pabak(a, b)` -> float (nan when undefined)
- Produces: `precision_recall(pred, truth) -> tuple[float, float]`, `confusion(pred, truth) -> [[TN, FP], [FN, TP]]` (rows = human fail/pass, cols = judge fail/pass)
- Produces: `agreement_report(pred, truth, n_resamples=10_000, seed=0) -> dict[str, CI]` (keys raw, kappa, ac1, pabak, precision, recall)
- Produces: `wilson(k, n, conf=0.95) -> CI` unchanged

- [ ] **Step 1: Write the failing tests**

Replace the entire contents of `tests/test_agreement.py` with:

```python
import math

import numpy as np
import pytest

from arena_evals.stats.agreement import (agreement_report, cohen_kappa, confusion, gwet_ac1, pabak,
                                         precision_recall, raw_agreement, wilson)

# 100 examples: human pass 54, judge pass 52; TP 47, TN 41, FP 5, FN 7
TRUTH = [1] * 47 + [0] * 41 + [0] * 5 + [1] * 7
PRED = [1] * 47 + [0] * 41 + [1] * 5 + [0] * 7


def test_point_metrics_on_known_table():
    assert raw_agreement(PRED, TRUTH) == pytest.approx(0.88)
    # pe = 0.52*0.54 + 0.48*0.46 = 0.5016; kappa = (0.88-0.5016)/(1-0.5016)
    assert cohen_kappa(PRED, TRUTH) == pytest.approx((0.88 - 0.5016) / 0.4984)
    assert pabak(PRED, TRUTH) == pytest.approx(0.76)
    # pi = 0.53; pe = 2*0.53*0.47 = 0.4982
    assert gwet_ac1(PRED, TRUTH) == pytest.approx((0.88 - 0.4982) / 0.5018)
    assert precision_recall(PRED, TRUTH) == pytest.approx((47 / 52, 47 / 54))
    assert confusion(PRED, TRUTH) == [[41, 5], [7, 47]]


def test_kappa_degenerate_all_same_is_nan_ac1_is_defined():
    assert math.isnan(cohen_kappa([1] * 10, [1] * 10))
    assert gwet_ac1([1] * 10, [1] * 10) == 1.0


def test_agreement_report_has_cis_and_is_seeded():
    a = agreement_report(PRED, TRUTH, n_resamples=2000, seed=5)
    b = agreement_report(PRED, TRUTH, n_resamples=2000, seed=5)
    assert a == b
    for name in ("raw", "kappa", "ac1", "pabak", "precision", "recall"):
        assert a[name].lo <= a[name].point <= a[name].hi


def test_agreement_report_rejects_mismatch():
    with pytest.raises(ValueError):
        agreement_report([1, 0], [1])


def test_wilson_known_value():
    ci = wilson(6, 60)
    assert ci.point == pytest.approx(0.1)
    assert ci.lo == pytest.approx(0.0466, abs=1e-3)
    assert ci.hi == pytest.approx(0.2012, abs=1e-3)
    with pytest.raises(ValueError):
        wilson(0, 0)
```

- [ ] **Step 2: Run them to verify they fail**

Run:

```bash
pytest tests/test_agreement.py -v
```

Expected: FAIL, with `ImportError: cannot import name 'agreement_report'` in the output.

- [ ] **Step 3: Replace the agreement module with the full version**

Replace the entire contents of `arena_evals/stats/agreement.py` with:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
pytest tests/test_agreement.py tests/test_drafting.py -v
```

Expected: PASS (`7 passed`).

- [ ] **Step 5: Commit**

Run:

```bash
git add arena_evals/stats/agreement.py tests/test_agreement.py
git commit -m "feat(stats): kappa, AC1, PABAK, precision/recall, confusion with bootstrap CIs"
```

### Task 19: Calibration export, blind labeling CLI, agreement report

**Files:**
- Create: `arena_evals/calibration.py`
- Test: `tests/test_calibration.py`

**Interfaces:**
- Consumes: `run.generate`, `run.score`, `run.read_results`, `datasets.load/write_jsonl`, `corpus.load`, `stats.agreement.agreement_report/confusion`
- Produces: `calibration.VARIANTS`, `calibration.read_jsonl(path) -> list[dict]`
- Produces: `calibration.stratified_sample(items, cell, n, min_per_cell, seed) -> list[dict]`
- Produces: `calibration.export(cfg, *, model=None, judge_model=None, epochs=1) -> Path` (writes `calibration/to_label.jsonl` blind and `calibration/key.jsonl`)
- Produces: `calibration.label_cli(path, labels_path=None, labeler=None, ask=input) -> Path` (appends `calibration/labels.jsonl`, resumable)
- Produces: `calibration.report(labels, key, n_resamples=10_000, seed=0) -> dict` (`n`, `agreement`, `confusion`)

- [ ] **Step 1: Write the failing test**

Create `tests/test_calibration.py`:

```python
import json
import shutil
from collections import Counter
from pathlib import Path

import pytest
from inspect_ai.model import ModelOutput, get_model

from arena_evals import calibration, config, datasets
from arena_evals.config import ROOT
from arena_evals.datasets import TaskRecord

M = "mockllm/model"


def test_stratified_sample_proportional_with_minimum():
    items = [{"id": i, "cell": "big"} for i in range(170)] + [{"id": 1000 + i, "cell": "small"} for i in range(8)] \
        + [{"id": 2000 + i, "cell": "tiny"} for i in range(3)]
    out = calibration.stratified_sample(items, lambda e: (e["cell"],), n=100, min_per_cell=5, seed=1)
    counts = Counter(e["cell"] for e in out)
    assert sum(counts.values()) == 100
    assert counts["tiny"] == 3 and counts["small"] == 5 and counts["big"] == 92
    assert out == calibration.stratified_sample(items, lambda e: (e["cell"],), 100, 5, 1)  # seeded
    assert len({e["id"] for e in out}) == 100


def test_report_known_table():
    key = [{"example_id": f"e{i}", "judge_pass": p} for i, p in enumerate([1] * 47 + [0] * 41 + [1] * 5 + [0] * 7)]
    labels = [{"example_id": f"e{i}", "human_label": h}
              for i, h in enumerate(["pass"] * 47 + ["fail"] * 41 + ["fail"] * 5 + ["pass"] * 7)]
    rep = calibration.report(labels, key, n_resamples=2000, seed=0)
    assert rep["n"] == 100 and rep["confusion"] == [[41, 5], [7, 47]]
    assert rep["agreement"]["raw"]["point"] == pytest.approx(0.88)
    assert set(rep["agreement"]) == {"raw", "kappa", "ac1", "pabak", "precision", "recall"}


def test_label_cli_is_blind_resumable_and_appends(tmp_path: Path, capsys):
    to_label = datasets.write_jsonl(tmp_path / "to_label.jsonl", [
        {"example_id": f"cal-000{i}-baseline-r0", "task_id": f"cal-000{i}", "variant": "baseline", "repeat": 0,
         "type": "lookup", "input": "q", "reference": "r", "answer": "a", "citations": ["HR-001"]} for i in (1, 2, 3)])
    answers = iter(["p", "", "f", "wrong date", "q"])
    labels = calibration.label_cli(to_label, labeler="zeesh", ask=lambda _: next(answers))
    assert "judge" not in capsys.readouterr().out.lower()
    rows = calibration.read_jsonl(labels)
    assert [(r["example_id"], r["human_label"]) for r in rows] == [("cal-0001-baseline-r0", "pass"),
                                                                    ("cal-0002-baseline-r0", "fail")]
    answers = iter(["p", ""])
    calibration.label_cli(to_label, labeler="zeesh", ask=lambda _: next(answers))  # resumes at #3
    assert len(calibration.read_jsonl(labels)) == 3


def _tmp_root(tmp_path: Path) -> Path:
    for d in ("configs", "prompts", "corpus"):
        shutil.copytree(ROOT / d, tmp_path / d)
    (tmp_path / "arena_evals").mkdir()
    shutil.copy(ROOT / "arena_evals" / "scorers.py", tmp_path / "arena_evals" / "scorers.py")
    return tmp_path


def test_export_runs_four_variants_and_writes_blind_file(tmp_path: Path):
    root = _tmp_root(tmp_path)
    tasks = [TaskRecord(id=f"cal-{i:04d}", input=f"Explain incident number {i}", reference="ref",
                        gold_doc_ids=["INC-001"], type="lookup" if i % 2 else "multi_hop", answer_kind="free_text",
                        tags=["lookup"], split="calibration") for i in range(1, 16)]
    tasks.append(TaskRecord(id="cal-0099", input="How many PTO days?", reference="20 days", gold_doc_ids=["HR-001"],
                            type="lookup", answer_kind="exact", tags=["lookup"], split="calibration"))
    datasets.write_jsonl(root / "datasets" / "calibration.jsonl", tasks)
    cfg = config.load(root)
    cfg.cert["n_labels"] = 30

    def agent(messages, tools, tool_choice, c):
        q = next(m.text for m in messages if m.role == "user")
        good = "GOOD" if int(q.split()[-1]) % 3 else "BAD"
        return ModelOutput.from_content(M, f'x\nFINAL: {{"answer": "{good} answer", "citations": ["INC-001"], "abstain": false}}')

    def judge(messages, tools, tool_choice, c):
        ok = "GOOD answer" in messages[-1].text
        return ModelOutput.from_content(M, json.dumps({"correct": ok, "faithful": True, "reason": "r"}))

    out = calibration.export(cfg, model=get_model(M, custom_outputs=agent, memoize=False),
                             judge_model=get_model(M, custom_outputs=judge, memoize=False))
    blind = calibration.read_jsonl(out)
    key = calibration.read_jsonl(root / "calibration" / "key.jsonl")
    assert len(blind) == 30 and len(key) == 30
    assert all("judge_pass" not in e and "judge" not in e for e in blind)
    assert {e["variant"] for e in blind} <= set(calibration.VARIANTS)
    assert all(not e["task_id"] == "cal-0099" for e in blind)  # only free_text tasks
    cells = Counter((e["type"], k["judge_pass"]) for e, k in zip(blind, key))
    assert all(v >= 5 for v in cells.values())
```

- [ ] **Step 2: Run it to verify it fails**

Run:

```bash
pytest tests/test_calibration.py -v
```

Expected: FAIL, with `ImportError: cannot import name 'calibration'` in the output.

- [ ] **Step 3: Create the calibration module**

`export` runs all four variants with one repeat each on the free_text calibration tasks only (the only judge-scored type), then draws 100 examples stratified by task type x first-pass judge label, proportional with at least 5 per non-empty cell. Using one repeat instead of k = 3 is a cost choice: the pool only needs to exceed 100 examples.

Create `arena_evals/calibration.py`:

```python
"""Judge calibration: stratified export for blind hand-labeling, a terminal labeling CLI, agreement report."""
from __future__ import annotations

import json
import os
import random
from pathlib import Path
from typing import Callable

from arena_evals import corpus, datasets, run
from arena_evals.config import Config
from arena_evals.stats.agreement import agreement_report, confusion

VARIANTS = ("baseline", "concise", "verbose", "regressed")


def read_jsonl(path: Path) -> list[dict]:
    p = Path(path)
    return [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines() if l.strip()] if p.exists() else []


def stratified_sample(items: list[dict], cell: Callable[[dict], tuple], n: int, min_per_cell: int,
                      seed: int) -> list[dict]:
    """Proportional allocation over cells, at least min_per_cell per non-empty cell (capped at cell size)."""
    cells: dict[tuple, list[dict]] = {}
    for it in items:
        cells.setdefault(cell(it), []).append(it)
    n = min(n, len(items))
    alloc = {c: min(len(v), max(min_per_cell, round(n * len(v) / len(items)))) for c, v in cells.items()}
    while sum(alloc.values()) > n:  # shrink the largest allocations first, never below the minimum
        c = max((c for c in alloc if alloc[c] > min(min_per_cell, len(cells[c]))), key=lambda c: alloc[c])
        alloc[c] -= 1
    while sum(alloc.values()) < n:  # grow the cells with the most spare items
        c = max((c for c in alloc if alloc[c] < len(cells[c])), key=lambda c: len(cells[c]) - alloc[c])
        alloc[c] += 1
    rng = random.Random(seed)
    out = []
    for c in sorted(cells):
        out += rng.sample(cells[c], alloc[c])
    return out


def export(cfg: Config, *, model=None, judge_model=None, epochs: int = 1) -> Path:
    """Run all four variants on the free_text calibration tasks, judge them, write to_label.jsonl + key.jsonl."""
    records = datasets.load(cfg.root / "datasets" / "calibration.jsonl")
    free = [r for r in records if r.answer_kind == "free_text"]
    if not free:
        raise RuntimeError("calibration split has no free_text tasks")
    ds = datasets.write_jsonl(cfg.root / "calibration" / "free_text_tasks.jsonl", free)
    by_id = {r.id: r for r in free}
    pool = []
    for v in VARIANTS:
        log = run.generate("calibration", v, cfg.prompts_dir, ds, cfg, model=model, epochs=epochs)
        out = run.score(log, ds, cfg, judge_model=judge_model)
        for r in run.read_results(out.results_path):
            if r["status"] == "ok" and r["scores"]["judge"] is not None:
                j = r["scores"]["judge"]
                t = by_id[r["task_id"]]
                pool.append({"example_id": f"{t.id}-{v}-r{r['repeat']}", "task_id": t.id, "variant": v,
                             "repeat": r["repeat"], "type": t.type, "input": t.input, "reference": t.reference,
                             "answer": r["answer"], "citations": r["citations"],
                             "judge_pass": bool(j["correct"] and j["faithful"]), "judge": j})
    chosen = stratified_sample(pool, lambda e: (e["type"], e["judge_pass"]), cfg.cert["n_labels"],
                               cfg.cert["min_per_cell"], cfg.eval["seed"])
    blind = [{k: v for k, v in e.items() if k not in ("judge_pass", "judge")} for e in chosen]
    datasets.write_jsonl(cfg.root / "calibration" / "key.jsonl",
                         [{"example_id": e["example_id"], "judge_pass": e["judge_pass"], **e["judge"]} for e in chosen])
    return datasets.write_jsonl(cfg.root / "calibration" / "to_label.jsonl", blind)


def label_cli(path: Path, labels_path: Path | None = None, labeler: str | None = None,
              ask: Callable[[str], str] = input) -> Path:
    """Blind labeling: shows question, reference, answer and cited docs; appends to labels.jsonl. Resumable."""
    path = Path(path)
    labels_path = Path(labels_path or path.parent / "labels.jsonl")
    labeler = labeler or os.environ.get("USER") or os.environ.get("USERNAME") or "unknown"
    done = {l["example_id"] for l in read_jsonl(labels_path)}
    todo = [e for e in read_jsonl(path) if e["example_id"] not in done]
    docs = corpus.load()
    for i, e in enumerate(todo, 1):
        print(f"\n[{len(done) + i}/{len(done) + len(todo)}] {e['example_id']}\nQUESTION: {e['input']}\n"
              f"REFERENCE: {e['reference']}\nANSWER: {e['answer']}\nCITED: {e['citations']}")
        for c in e["citations"]:
            print(f"--- {c}\n{docs[c].body if c in docs else '(not in corpus)'}")
        v = ""
        while v not in ("p", "f", "s", "q"):
            v = ask("pass = correct AND faithful to cited docs. [p]ass / [f]ail / [s]kip / [q]uit > ").strip().lower()[:1]
        if v == "q":
            break
        if v == "s":
            continue
        note = ask("note (optional): ").strip()
        with labels_path.open("a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps({"example_id": e["example_id"], "task_id": e["task_id"], "variant": e["variant"],
                                 "human_label": "pass" if v == "p" else "fail", "labeler": labeler,
                                 "note": note}) + "\n")
    return labels_path


def report(labels: list[dict], key: list[dict], n_resamples: int = 10_000, seed: int = 0) -> dict:
    """Agreement of judge_pass (key) with human_label (labels) on examples present in both."""
    judge = {k["example_id"]: k["judge_pass"] for k in key if k.get("judge_pass") is not None}
    pairs = [(int(judge[l["example_id"]]), int(l["human_label"] == "pass")) for l in labels if l["example_id"] in judge]
    if not pairs:
        raise ValueError("no labeled example has a judge verdict")
    pred, truth = [p for p, _ in pairs], [t for _, t in pairs]
    return {"n": len(pairs),
            "agreement": {k: v.as_dict() for k, v in agreement_report(pred, truth, n_resamples, seed).items()},
            "confusion": confusion(pred, truth)}
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
pytest tests/test_calibration.py -v
```

Expected: PASS (`4 passed`).

- [ ] **Step 5: Commit**

Run:

```bash
git add arena_evals/calibration.py tests/test_calibration.py
git commit -m "feat(calibration): stratified blind export, labeling CLI, agreement report"
```

### Task 20: Judge cert status and `certify` (agreement gate)

**Files:**
- Create: `arena_evals/ci.py`
- Test: `tests/test_certify.py`

**Interfaces:**
- Consumes: `calibration.read_jsonl/report`, `scorers.judge_answer`, `corpus.load`, `config.rubric_hash/sha256_files`
- Produces: `ci.cert_path(cfg)`, `ci.labels_path(cfg)`, `ci.current_cert_inputs(cfg) -> dict`
- Produces: `ci.cert_status(cfg) -> tuple[bool, str]` (`(True, "certified")`, or False with `no cert ...`, `stale cert: <keys> changed`, `judge failed certification`)
- Produces: `ci.certify(cfg, *, judge_model=None) -> int` (0 iff certified; writes `configs/judge.cert.json`; Task 23 adds `perturber_model=None` and the bias tests)

- [ ] **Step 1: Write the failing test**

Create `tests/test_certify.py`:

```python
import json
import shutil

import pytest
from inspect_ai.model import ModelOutput, get_model

from arena_evals import ci, config, datasets
from arena_evals.config import ROOT

M = "mockllm/model"


def _repo(tmp_path):
    for d in ("configs", "prompts", "corpus"):
        shutil.copytree(ROOT / d, tmp_path / d)
    return tmp_path


def _examples(n=40):
    ex, labels = [], []
    for i in range(n):
        v = ("baseline", "concise", "verbose", "regressed")[i % 4]
        good = i % 5 != 0
        ex.append({"example_id": f"cal-{i:04d}-{v}-r0", "task_id": f"cal-{i:04d}", "variant": v, "repeat": 0,
                   "type": "lookup", "input": f"q{i}", "reference": "ref",
                   "answer": ("GOOD " if good else "BAD ") + "the outage lasted 47 minutes per INC-001." * (1 + i % 3),
                   "citations": ["INC-001"]})
        labels.append({"example_id": ex[-1]["example_id"], "task_id": ex[-1]["task_id"], "variant": v,
                       "human_label": "pass" if good else "fail", "labeler": "t", "note": ""})
    return ex, labels


def _judge(mode: str):
    """agree: passes GOOD answers (matches the human). contrarian: the opposite. length: also passes long answers."""
    def reply(messages, tools, tool_choice, c):
        ans = messages[-1].text.split("## Assistant answer\n", 1)[1].split("\n\n## Documents", 1)[0]
        good = ans.startswith("GOOD")
        ok = {"agree": good, "contrarian": not good, "length": good or len(ans) > 150}[mode]
        return ModelOutput.from_content(M, json.dumps({"correct": ok, "faithful": True, "reason": "r"}))
    return get_model(M, custom_outputs=reply, memoize=False)


@pytest.fixture
def cfg(tmp_path):
    root = _repo(tmp_path)
    ex, labels = _examples()
    datasets.write_jsonl(root / "calibration" / "to_label.jsonl", ex)
    datasets.write_jsonl(root / "calibration" / "labels.jsonl", labels)
    c = config.load(root)
    c.eval["n_resamples"], c.eval["cache"] = 1000, False
    return c


def test_certify_agreeing_judge_writes_valid_cert_then_goes_stale(cfg):
    assert ci.certify(cfg, judge_model=_judge("agree")) == 0
    cert = json.loads(ci.cert_path(cfg).read_text())
    assert cert["certified"] is True and cert["n_labels"] == 40
    assert cert["rubric_hash"] == config.rubric_hash(cfg.rubric_dir)
    assert cert["agreement"]["kappa"]["point"] == pytest.approx(1.0)
    assert cert["agreement"]["confusion"] == [[8, 0], [0, 32]]
    assert ci.cert_status(cfg) == (True, "certified")
    (cfg.root / "calibration" / "labels.jsonl").write_text("")
    assert ci.cert_status(cfg) == (False, "stale cert: labels_hash changed")


def test_certify_rejects_disagreeing_judge(cfg):
    assert ci.certify(cfg, judge_model=_judge("contrarian")) == 1
    assert json.loads(ci.cert_path(cfg).read_text())["certified"] is False
    assert ci.cert_status(cfg) == (False, "judge failed certification")


def test_cert_status_without_cert(cfg):
    assert ci.cert_status(cfg)[0] is False
```

- [ ] **Step 2: Run it to verify it fails**

Run:

```bash
pytest tests/test_certify.py -v
```

Expected: FAIL, with `ImportError: cannot import name 'ci' from 'arena_evals'` in the output.

- [ ] **Step 3: Create `ci.py` with certification**

Create `arena_evals/ci.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
pytest tests/test_certify.py -v
```

Expected: PASS (`3 passed`).

- [ ] **Step 5: Commit**

Run:

```bash
git add arena_evals/ci.py tests/test_certify.py
git commit -m "feat(certify): judge cert file, staleness check, kappa certification gate"
```

### Task 21: (HUMAN) Export, hand-label ~100 examples, first certification

**Files:**
- Create: `calibration/to_label.jsonl`
- Create: `calibration/key.jsonl`
- Create: `calibration/labels.jsonl`
- Create: `configs/judge.cert.json`

**Interfaces:**
- Consumes: CLI `calibrate export`, `calibrate label`, `certify`; `datasets/calibration.jsonl`; an API key
- Produces: ~100 human labels and a first (agreement-only) `configs/judge.cert.json`

- [ ] **Step 1: (HUMAN) Export the examples**

Run (needs ANTHROPIC_API_KEY, costs money):

```bash
python -m arena_evals calibrate export --max-usd 20
```

Expected: `wrote .../calibration/to_label.jsonl`. It has 100 lines if the calibration split yielded at least 100 judged free-text answers across the four variants; if fewer, `datasets/calibration.jsonl` has too few free_text tasks: draft more calibration multi_hop/lookup tasks (Task 7) and re-export.

- [ ] **Step 2: (HUMAN) Label every example blind**

The label CLI shows the question, the reference answer, the agent answer and the full text of each cited document, never the judge verdict. Label `p` only if the answer is correct (same facts as the reference) AND every claim is supported by the cited documents; otherwise `f`. Label on the rubric's definition, not on style or length. `q` saves and quits; re-running resumes where you stopped. Budget 1-2 hours; do it in one sitting if you can to keep labels consistent.

Run (interactive):

```bash
python -m arena_evals calibrate label --labeler zeesh
```

Optional (PRD FR3 nice-to-have): have a second person label 20 of the same examples into a separate file to estimate human-human agreement. Not required.

- [ ] **Step 3: (HUMAN) Certify**

Run (needs ANTHROPIC_API_KEY):

```bash
python -m arena_evals certify --max-usd 10
```

Expected: `kappa K (95% CI lo to hi)` then `certified: True -> .../configs/judge.cert.json`, exit 0. If `certified: False`: improve `prompts/judge/rubric.md` using only dev-split examples (never the gate split, never by relabeling to match the judge), then re-run `certify`. Every rubric edit makes the cert stale, so always finish with a fresh `certify`.

- [ ] **Step 4: Commit**

Run (after the HUMAN steps above):

```bash
git add calibration/to_label.jsonl calibration/key.jsonl calibration/labels.jsonl configs/judge.cert.json prompts/judge
git commit -m "data(calibration): 100 blind human labels and first judge certification"
```

## Milestone M6: Bias

### Task 22: Verbosity perturbation, slope and partial-correlation tests

**Files:**
- Modify: `arena_evals/bias.py`
- Test: `tests/test_bias.py`

**Interfaces:**
- Consumes: `stats.CI`, `bias.self_preference`
- Produces: `bias.FILLER`, `bias.COMPRESS_PROMPT`, `bias.pad(text, frac, seed) -> str`, `bias.facts_signature(text) -> list[str]`, `bias.length_x(perturbed, original) -> float` (log(len_p / len_o) / log(1.5), i.e. units of +50% length)
- Produces: `bias.slope_ci(x, y, groups, n_resamples=10_000, seed=0) -> CI` (OLS slope, cluster bootstrap by example)
- Produces: `bias.partial_corr_ci(judge_pass, loglen, human, n_resamples=10_000, seed=0) -> CI` (0.0 when residual variance is zero)
- Produces: `bias.verbosity(labels, perturbed, cfg) -> dict` (spec: `verbosity(labels, cfg)`; the re-judged perturbations are passed in because all model calls happen in one event loop inside `ci.certify`)

- [ ] **Step 1: Write the failing test**

Create `tests/test_bias.py`:

```python
import math

import numpy as np
import pytest

from arena_evals import bias


def test_pad_reaches_target_length_and_is_seeded():
    a = bias.pad("The cap is 220 USD per night.", 0.5, seed=3)
    assert len(a) >= 1.5 * len("The cap is 220 USD per night.")
    assert a == bias.pad("The cap is 220 USD per night.", 0.5, seed=3)
    assert bias.facts_signature(a) == bias.facts_signature("The cap is 220 USD per night.")
    padded = bias.pad("x" * 100, 0.5, 0)
    assert bias.length_x(padded, "x" * 100) == pytest.approx(math.log(len(padded) / 100) / math.log(1.5))
    assert bias.length_x(padded, "x" * 100) >= 1.0


def test_facts_signature_detects_changed_numbers_and_ids():
    assert bias.facts_signature("16 weeks per HR-009, 1,340 alarms") == ["1340", "16", "HR-009"]
    assert bias.facts_signature("16 weeks") != bias.facts_signature("12 weeks")


def test_slope_ci_detects_length_bias_and_flat_judge():
    groups = np.repeat(np.arange(100), 3)
    x = np.tile([0.0, 1.0, 1.71], 100)
    flat = np.tile([100.0, 100.0, 100.0], 100)
    s = bias.slope_ci(x, flat, groups, n_resamples=500, seed=0)
    assert (s.point, s.lo, s.hi) == (0.0, 0.0, 0.0)
    rng = np.random.default_rng(0)
    biased = (rng.random(300) < 0.5 + 0.1 * x).astype(float) * 100   # +10 pts per +50% length
    s = bias.slope_ci(x, biased, groups, n_resamples=500, seed=0)
    assert s.lo > 2.0


def test_partial_corr_controls_for_human_label():
    rng = np.random.default_rng(1)
    human = rng.integers(0, 2, 400)
    loglen = rng.normal(5, 1, 400) + human            # longer answers are genuinely better
    judge = human.copy()                             # judge tracks the human label only
    c = bias.partial_corr_ci(judge, loglen, human, n_resamples=500, seed=0)
    assert (c.point, c.lo, c.hi) == (0.0, 0.0, 0.0)             # perfect judge: zero residual variance -> 0
    judge2 = ((loglen + rng.normal(0, 0.5, 400)) > 5.5).astype(int)   # judge rewards length
    c2 = bias.partial_corr_ci(judge2, loglen, human, n_resamples=500, seed=0)
    assert c2.lo > 0
```

- [ ] **Step 2: Run it to verify it fails**

Run:

```bash
pytest tests/test_bias.py -v
```

Expected: FAIL, with `AttributeError: module 'arena_evals.bias' has no attribute 'pad'` in the output.

- [ ] **Step 3: Extend the imports of `bias.py`**

In `arena_evals/bias.py`, replace:

```python
from __future__ import annotations

from arena_evals.config import Config
```

with:

```python
from __future__ import annotations

import math
import random
import re

import numpy as np

from arena_evals.config import Config
from arena_evals.stats import CI
```

- [ ] **Step 4: Add the verbosity tests to `bias.py`**

A compressed answer is kept only if it is at most 70% of the original length and the numbers and doc IDs it mentions are unchanged (`facts_signature`). This is this plan's deterministic reading of the spec's "answer_match/citation extraction changes" rule for free-text answers.

Append to the end of `arena_evals/bias.py`:

```python
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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run:

```bash
pytest tests/test_bias.py tests/test_judge.py -v
```

Expected: PASS (`10 passed`).

- [ ] **Step 6: Commit**

Run:

```bash
git add arena_evals/bias.py tests/test_bias.py
git commit -m "feat(bias): verbosity padding/compression, slope and partial-correlation tests"
```

### Task 23: Wire the bias tests into certification, then re-certify

**Files:**
- Modify: `arena_evals/ci.py`
- Modify: `tests/test_certify.py`
- Modify: `configs/judge.cert.json`

**Interfaces:**
- Consumes: `bias.pad/length_x/facts_signature/COMPRESS_PROMPT/verbosity`, `calibration.report`, `scorers.judge_answer`
- Produces: `ci.certify(cfg, *, judge_model=None, perturber_model=None) -> int`; a bias failure sets `certified: false`; the cert gains `bias.{verbosity_slope_pts, length_partial_corr, self_preference_warning}` and `thresholds.verbosity_slope_max_pts`

- [ ] **Step 1: Write the failing tests**

In `tests/test_certify.py`, replace every occurrence of:

```text
ci.certify(cfg, judge_model=
```

with:

```text
ci.certify(cfg, perturber_model=_perturber(), judge_model=
```

Append to the end of `tests/test_certify.py`:

```python
def _perturber():
    """Compression stand-in: keeps the first sentence only (drops repeated facts, so most are discarded)."""
    def reply(messages, tools, tool_choice, c):
        ans = messages[-1].text.split("ANSWER:\n", 1)[1]
        return ModelOutput.from_content(M, ans.split(".")[0] + ".")
    return get_model(M, custom_outputs=reply, memoize=False)


def test_certify_fails_a_length_biased_judge_on_the_verbosity_slope(cfg):
    assert ci.certify(cfg, perturber_model=_perturber(), judge_model=_judge("length")) == 1
    cert = json.loads(ci.cert_path(cfg).read_text())
    assert cert["certified"] is False
    assert cert["agreement"]["kappa"]["point"] == pytest.approx(1.0)       # agreement alone would have passed
    assert cert["bias"]["verbosity_slope_pts"]["pass"] is False
    assert cert["bias"]["verbosity_slope_pts"]["ci95"][1] > 2.0
    assert set(cert["bias"]) == {"verbosity_slope_pts", "length_partial_corr", "self_preference_warning"}


def test_certify_agreeing_judge_passes_bias_tests(cfg):
    assert ci.certify(cfg, perturber_model=_perturber(), judge_model=_judge("agree")) == 0
    bias_ = json.loads(ci.cert_path(cfg).read_text())["bias"]
    assert bias_["verbosity_slope_pts"]["pass"] and bias_["length_partial_corr"]["pass"]
```

- [ ] **Step 2: Run them to verify they fail**

Run:

```bash
pytest tests/test_certify.py -v
```

Expected: FAIL, with `TypeError: certify() got an unexpected keyword argument 'perturber_model'` in the output.

- [ ] **Step 3: Replace `ci.py` with the bias-aware certification**

Replace the entire contents of `arena_evals/ci.py` with:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
pytest tests/test_certify.py tests/test_bias.py -v
```

Expected: PASS (`9 passed`).

- [ ] **Step 5: Commit the code**

Run:

```bash
git add arena_evals/ci.py tests/test_certify.py
git commit -m "feat(certify): verbosity bias tests gate judge certification"
```

- [ ] **Step 6: (HUMAN) Re-certify with the bias tests**

Run (needs ANTHROPIC_API_KEY, costs money: about 4 judge calls and 1 perturber call per label):

```bash
python -m arena_evals certify --max-usd 20
```

Expected: the kappa line, a `verbosity slope ... pts per +50% length (95% CI lo to hi), pass: True` line, a `length partial corr ... pass: True` line and `certified: True`. If a bias test fails, the judge is uncertified and the gate will refuse to run: strengthen the rubric's "Length is not a criterion" instruction, re-run, and record the effect sizes either way (PRD success metric: "passes or is documented with effect sizes").

- [ ] **Step 7: Commit the new cert**

Run (after the HUMAN steps above):

```bash
git add configs/judge.cert.json prompts/judge
git commit -m "data(calibration): judge re-certified with bias tests"
```

## Milestone M7: Gate

### Task 24: PR comment renderer with hidden marker

**Files:**
- Create: `arena_evals/report.py`
- Test: `tests/test_report.py`

**Interfaces:**
- Consumes: `PairedResult.as_dict()` (Task 11)
- Produces: `report.render(paired, per_tag, worst5, cert, cost, meta) -> str` (markdown; `paired=None` renders an error)
- Produces: `report.marker(meta) -> str`, `report.parse_marker(body) -> dict | None`, `report.replay_note(prior_body) -> str`
- Produces: `report.MARKER_PREFIX = "<!-- arena-eval:v1 "`, `report.REPLAY_NOTE`, `report.MAX_CHARS = 65_000`
- Produces: `meta` keys read: `verdict`, `head_sha`, `run_id`, `eps_pts`, `base_rate`, `cand_rate`, `judge_errors`, `dropped_tasks`, `dataset_hash`, `rubric_hash`, `baseline_cache`, `baseline_rerun_on_head`, `rerun_reason`, `artifact_url`, `error`, `recertified`, `self_preference_warning`; `cost` = `{spent, cap}`

- [ ] **Step 1: Write the failing test**

Create `tests/test_report.py`:

```python
import numpy as np

from arena_evals import report
from arena_evals.stats.bootstrap import paired_bootstrap

CERT = {"certified": True, "n_labels": 100, "agreement": {"kappa": {"point": 0.76, "ci95": [0.63, 0.88]}}}
META = {"verdict": "block", "head_sha": "abc123", "run_id": "gh-1-1", "eps_pts": 2.0, "artifact_url": "https://x/runs/1",
        "base_rate": {"point": 0.82, "ci95": [0.78, 0.86]}, "cand_rate": {"point": 0.75, "ci95": [0.70, 0.79]},
        "judge_errors": {"baseline": 0, "candidate": 1}, "dataset_hash": "sha256:d", "rubric_hash": "sha256:r",
        "baseline_cache": "miss", "baseline_rerun_on_head": True, "dropped_tasks": []}
WORST = [{"task_id": "gate-0042", "base": 1.0, "cand": 0.0, "d_pts": -100.0, "trace_id": "5f1c"}]


def paired():
    return paired_bootstrap(np.random.default_rng(0).normal(-0.073, 0.45, 300), seed=1)


def test_render_block_has_required_fields_and_marker():
    body = report.render(paired(), [], WORST, CERT, {"spent": 12.3, "cap": 20.0}, META)
    assert report.parse_marker(body) == {"head_sha": "abc123", "verdict": "block", "run_id": "gh-1-1"}
    for s in ("dropped task success by", "one-sided 97.5% upper bound", "n=300", "Merge blocked.",
              "Minimum detectable effect", "gate-0042", "`5f1c`", "https://x/runs/1", "certified (kappa 0.76",
              "$12.30 of $20.00", "sha256:d", "sha256:r", "baseline re-run on this PR's dataset/rubric",
              "Judge errors | baseline 0, candidate 1", "phoenix serve", "Baseline success | 82.0% (95% CI"):
        assert s in body, s


def test_render_error_has_no_verdict_numbers():
    body = report.render(None, [], [], {}, {"spent": 20.1, "cap": 20.0},
                         {"verdict": "error", "error": "cost cap hit ($20.10 of $20.00)", "run_id": "r"})
    assert "**No verdict:** cost cap hit ($20.10 of $20.00)" in body
    assert report.parse_marker(body)["verdict"] == "error"


def test_render_truncates_huge_per_tag_table_but_keeps_marker():
    tags = [{"tag": f"tag-{i:05d}-" + "x" * 40, "n": 3, "delta_pts": -1.0, "ci95_pts": [-5.0, 3.0], "flagged": False}
            for i in range(5000)]
    body = report.render(paired(), tags, WORST, CERT, {"spent": 1.0, "cap": 20.0}, META)
    assert len(body) <= 65_536
    assert "more tags omitted" in body
    assert report.parse_marker(body)["verdict"] == "block"
    assert body.rstrip().endswith("http://localhost:6006.")


def test_replay_note_is_idempotent():
    once = report.replay_note("body")
    assert report.replay_note(once) == once and report.REPLAY_NOTE in once


def test_parse_marker_absent():
    assert report.parse_marker("just a human comment") is None
    assert report.parse_marker(None) is None
```

- [ ] **Step 2: Run it to verify it fails**

Run:

```bash
pytest tests/test_report.py -v
```

Expected: FAIL, with `ImportError: cannot import name 'report' from 'arena_evals'` in the output.

- [ ] **Step 3: Create the renderer**

The marker goes first so that truncation (GitHub rejects bodies over 65,536 characters) can never drop it; only the advisory per-tag rows are trimmed.

Create `arena_evals/report.py`:

```python
"""Render the ONE gate PR comment (markdown) and read back its hidden marker. Pure: no I/O."""
from __future__ import annotations

import json
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
    head = [marker(meta), f"## Arena eval gate: {verdict.upper()}", ""]
    if verdict == "error" or paired is None:
        head += [f"**No verdict:** {meta.get('error', 'unknown error')}", ""]
    else:
        p = paired.as_dict()
        je = meta.get("judge_errors", {})
        head += [_headline(p, verdict, meta["eps_pts"]), "",
                 "| | |", "|---|---|",
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
pytest tests/test_report.py -v
```

Expected: PASS (`5 passed`).

- [ ] **Step 5: Commit**

Run:

```bash
git add arena_evals/report.py tests/test_report.py
git commit -m "feat(report): PR comment renderer with marker, CIs, worst 5, size cap"
```

### Task 25: Gate building blocks: GitHub client, cache key, judge-error rate, worst 5, rerun policy

**Files:**
- Modify: `arena_evals/ci.py`
- Test: `tests/test_ci.py`

**Interfaces:**
- Consumes: `report.MARKER_PREFIX`, `run.task_means`, `datasets.TaskRecord`, `config.sha256_bytes`
- Produces: `ci.EXIT = {"pass": 0, "warn": 0, "block": 1, "error": 2}`
- Produces: `ci.GitHub(repo=None, token=None, api=None)` with `pr(n)`, `changed_files(n)`, `comments(n)`, `upsert_comment(n, body)`, `set_status(sha, state, description, context, target_url="")`
- Produces: `ci.cache_key(base_sha, cfg, dataset_hash, rubric_hash_, scorer_hash_) -> str`
- Produces: `ci.judge_error_frac(rows) -> float`, `ci.worst(task_ids, d, base_rows, cand_rows, n=5) -> list[dict]`, `ci.tag_deltas(task_ids, d, records) -> dict[str, np.ndarray]`
- Produces: `ci.rerun_action(prior, head_sha, rerun_reason) -> "replay" | "evaluate"`, `ci.artifact_url() -> str`

- [ ] **Step 1: Write the failing test**

Create `tests/test_ci.py`:

```python
import json
import shutil
from pathlib import Path

import numpy as np
import pytest

from arena_evals import ci, config, datasets, report, run
from arena_evals.agents import CostCapExceeded
from arena_evals.config import ROOT
from test_e2e_mock import BAD, GOOD, TASKS, scripted_agent


def row(task, success, judge_error=False, trace="t"):
    return {"task_id": task, "success": success, "scores": {"judge_error": judge_error}, "trace_id": trace}


def test_judge_error_frac():
    assert ci.judge_error_frac([row("a", True)] * 49 + [row("b", None, True)]) == pytest.approx(0.02)
    assert ci.judge_error_frac([]) == 0.0


def test_cache_key_changes_with_every_input():
    cfg = config.load()
    k = ci.cache_key("base", cfg, "sha256:d", "sha256:r", "sha256:s")
    assert k == ci.cache_key("base", cfg, "sha256:d", "sha256:r", "sha256:s")
    assert k != ci.cache_key("base", cfg, "sha256:d", "sha256:r", "sha256:s2")   # scorer change -> miss
    assert k != ci.cache_key("base", cfg, "sha256:d2", "sha256:r", "sha256:s")
    assert k != ci.cache_key("base", cfg, "sha256:d", "sha256:r2", "sha256:s")
    assert k != ci.cache_key("base2", cfg, "sha256:d", "sha256:r", "sha256:s")
    cfg.models["agent"]["temperature"] = 0.1
    assert k != ci.cache_key("base", cfg, "sha256:d", "sha256:r", "sha256:s")


def test_rerun_action():
    prior = {"head_sha": "h1", "verdict": "block", "run_id": "r"}
    assert ci.rerun_action(prior, "h1", "") == "replay"
    assert ci.rerun_action(prior, "h1", "   ") == "replay"
    assert ci.rerun_action(prior, "h1", "flaky judge, see #12") == "evaluate"
    assert ci.rerun_action(prior, "h2", "") == "evaluate"                        # new commit
    assert ci.rerun_action(prior | {"verdict": "pass"}, "h1", "") == "evaluate"
    assert ci.rerun_action(None, "h1", "") == "evaluate"


def test_worst_lists_only_regressions_sorted():
    base = [row("a", True), row("b", True), row("c", False)]
    cand = [row("a", False, trace="ta"), row("b", True), row("c", True)]
    ids, d, _ = run.paired_deltas(base, cand)
    assert ci.worst(ids, d, base, cand) == [{"task_id": "a", "base": 1.0, "cand": 0.0, "d_pts": -100.0,
                                             "trace_id": "ta"}]
```

- [ ] **Step 2: Run it to verify it fails**

Run:

```bash
pytest tests/test_ci.py -v
```

Expected: FAIL, with `AttributeError: module 'arena_evals.ci' has no attribute` in the output.

- [ ] **Step 3: Extend the imports of `ci.py`**

In `arena_evals/ci.py`, replace:

```python
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path

from inspect_ai.model import GenerateConfig, get_model

from arena_evals import agents, bias, calibration, corpus
from arena_evals.config import Config, rubric_hash, sha256_files
from arena_evals.scorers import judge_answer
```

with:

```python
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
from arena_evals.config import Config, rubric_hash, scorer_hash, sha256_bytes, sha256_files
from arena_evals.scorers import judge_answer
from arena_evals.stats.bootstrap import bootstrap_ci, gate_decision, paired_bootstrap, per_tag
```

- [ ] **Step 4: Add the gate building blocks**

Append to the end of `arena_evals/ci.py`:

```python
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
        mine = [c for c in self.comments(n) if report.MARKER_PREFIX in (c.get("body") or "")]
        if mine:
            self._req("PATCH", f"/repos/{self.repo}/issues/comments/{mine[-1]['id']}", {"body": body})
        else:
            self._req("POST", f"/repos/{self.repo}/issues/{n}/comments", {"body": body})

    def set_status(self, sha: str, state: str, description: str, context: str, target_url: str = "") -> None:
        body = {"state": state, "description": description[:140], "context": context}
        if target_url:
            body["target_url"] = target_url
        self._req("POST", f"/repos/{self.repo}/statuses/{sha}", body)


def cache_key(base_sha: str, cfg: Config, dataset_hash: str, rubric_hash_: str, scorer_hash_: str) -> str:
    a, j = cfg.models["agent"], cfg.models["judge"]
    parts = [base_sha, a["model"], j["model"], a["temperature"], j["temperature"],
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
```

- [ ] **Step 5: Run the tests to verify they pass**

Run:

```bash
pytest tests/test_ci.py -v
```

Expected: PASS (`4 passed`).

- [ ] **Step 6: Commit**

Run:

```bash
git add arena_evals/ci.py tests/test_ci.py
git commit -m "feat(ci): GitHub client, baseline cache key, rerun policy, worst-5 helpers"
```

### Task 26: Gate orchestration: cert check, rerun policy, worktree baseline, cost cap, compare, report

**Files:**
- Modify: `arena_evals/ci.py`
- Modify: `tests/test_ci.py`

**Interfaces:**
- Consumes: everything above; `run.generate/score/paired_deltas/task_means/new_run_id`, `report.render`, `stats.bootstrap.*`, `bias.self_preference`, `agents.set_meter/CostMeter/CostCapExceeded`
- Produces: `ci.generate_baseline(base_sha, cfg, dataset, log_dir, max_usd, cache) -> Path` (git worktree at `.arena-base`, subprocess `python -m arena_evals generate ...`; exit 3 = cost cap)
- Produces: `ci.gate(pr, cfg, *, gh=None, rerun_reason="") -> int` (exit code); writes `.arena-out/{baseline,candidate}/`, `.arena-out/comment.md`, `.arena-out/gate.json`; baseline cache in `.arena-cache/baseline/<key>/`
- Produces: CLI `gate --pr N [--rerun-reason R]` and `cache-key --pr N` (already in `__main__.py`)

- [ ] **Step 1: Write the failing tests**

These run the whole gate against a fake GitHub with scripted agents: block, replay without a reason, re-evaluate with a reason, pass with a baseline cache hit, stale cert, and cost cap. Two more build a throwaway git repo whose `configs/models.yaml` points the agent at `mockllm/model`, and run the real worktree + subprocess baseline path.

Append to the end of `tests/test_ci.py`:

```python
# ---------------------------------------------------------------- full gate() against a fake GitHub
class FakeGitHub:
    def __init__(self, head="h1", changed=()):
        self.head, self.changed, self.bodies, self.statuses = head, list(changed), [], []

    def pr(self, n):
        return {"head": {"sha": self.head}, "base": {"sha": "b0"}}

    def changed_files(self, n):
        return self.changed

    def comments(self, n):
        return [{"id": 1, "body": self.bodies[-1]}] if self.bodies else []

    def upsert_comment(self, n, body):
        self.bodies.append(body)

    def set_status(self, sha, state, description, context, target_url=""):
        self.statuses.append((sha, state, context))


@pytest.fixture
def repo(tmp_path, monkeypatch):
    for d in ("configs", "prompts", "corpus"):
        shutil.copytree(ROOT / d, tmp_path / d)
    (tmp_path / "arena_evals").mkdir()
    shutil.copy(ROOT / "arena_evals" / "scorers.py", tmp_path / "arena_evals" / "scorers.py")
    datasets.write_jsonl(tmp_path / "datasets" / "gate.jsonl", TASKS)
    datasets.write_jsonl(tmp_path / "calibration" / "labels.jsonl", [{"example_id": "x"}])
    cfg = config.load(tmp_path)
    cfg.eval["k"] = 2
    ci.cert_path(cfg).write_text(json.dumps({"certified": True, **ci.current_cert_inputs(cfg)}))
    orig = run.generate
    calls = {"cand": 0, "base": 0, "cand_script": BAD}

    def fake_base(base_sha, cfg_, ds, log_dir, max_usd, cache):
        calls["base"] += 1
        return orig("gate", "baseline", cfg_.prompts_dir, ds, cfg_, model=scripted_agent(GOOD), log_dir=log_dir,
                    cache=False)

    def fake_cand(*a, **k):
        calls["cand"] += 1
        return orig(*a, **(k | {"model": scripted_agent(calls["cand_script"]), "cache": False}))

    monkeypatch.setattr(ci, "generate_baseline", fake_base)
    monkeypatch.setattr(run, "generate", fake_cand)
    return cfg, calls


def test_gate_blocks_then_replays_then_reruns_with_reason(repo):
    cfg, calls = repo
    gh = FakeGitHub()
    assert ci.gate(7, cfg, gh=gh) == 1
    body = gh.bodies[-1]
    assert "Merge blocked." in body and report.parse_marker(body)["verdict"] == "block"
    assert gh.statuses[-1] == ("h1", "failure", "arena-eval/gate")
    assert (cfg.root / ".arena-out" / "candidate" / "results.jsonl").exists()
    assert calls == {"cand": 1, "base": 1, "cand_script": BAD}

    assert ci.gate(7, cfg, gh=gh) == 1                       # same SHA, no reason: replay, no evaluation
    assert calls["cand"] == 1 and report.REPLAY_NOTE in gh.bodies[-1]

    assert ci.gate(7, cfg, gh=gh, rerun_reason="judge outage") == 1   # reason: evaluates, cache bypassed
    assert calls["cand"] == 2 and calls["base"] == 2
    assert "| Rerun reason | judge outage |" in gh.bodies[-1]


def test_gate_passes_on_noop_and_reuses_cached_baseline(repo):
    cfg, calls = repo
    calls["cand_script"] = GOOD
    assert ci.gate(8, cfg, gh=FakeGitHub(head="h2")) == 0
    shutil.rmtree(cfg.root / ".arena-out")
    gh = FakeGitHub(head="h3")
    assert ci.gate(8, cfg, gh=gh) == 0
    assert calls["base"] == 1                                # second run: baseline cache hit
    assert (cfg.root / ".arena-out" / "baseline" / "results.jsonl").exists()   # artifact still has both sides
    assert "cache hit" in gh.bodies[-1] and gh.statuses[-1][1] == "success"


def test_gate_refuses_stale_cert(repo):
    cfg, _ = repo
    (cfg.rubric_dir / "rubric.md").write_text("changed rubric")
    gh = FakeGitHub()
    assert ci.gate(9, cfg, gh=gh) == 2
    assert "judge uncertified (stale cert: rubric_hash changed)" in gh.bodies[-1]


def test_gate_cost_cap_is_error_with_no_score(repo, monkeypatch):
    cfg, _ = repo

    def boom(*a, **k):
        raise CostCapExceeded("cost cap hit ($20.01 of $20.00)")

    monkeypatch.setattr(ci, "generate_baseline", boom)
    gh = FakeGitHub()
    assert ci.gate(10, cfg, gh=gh) == 2
    assert "**No verdict:** cost cap hit ($20.01 of $20.00)" in gh.bodies[-1]
    assert "dropped task success" not in gh.bodies[-1]


def _git_repo(tmp_path: Path, with_harness: bool = True) -> tuple[Path, str]:
    import subprocess
    for d in ("configs", "prompts", "corpus") + (("arena_evals",) if with_harness else ()):
        shutil.copytree(ROOT / d, tmp_path / d, ignore=shutil.ignore_patterns("__pycache__"))
    models = tmp_path / "configs" / "models.yaml"
    models.write_text(models.read_text().replace("anthropic/claude-haiku-4-5-20251001\n  temperature",
                                                 "mockllm/model\n  temperature", 1))
    datasets.write_jsonl(tmp_path / "datasets" / "gate.jsonl", TASKS[:2])
    git = lambda *a: subprocess.run(["git", *a], cwd=tmp_path, check=True, capture_output=True, text=True).stdout
    git("init", "-q")
    git("add", "-A")
    git("-c", "user.name=t", "-c", "user.email=t@t", "commit", "-qm", "base")
    return tmp_path, git("rev-parse", "HEAD").strip()


def test_generate_baseline_runs_base_worktree_via_cli(tmp_path):
    root, sha = _git_repo(tmp_path)
    cfg = config.load(root)
    log = ci.generate_baseline(sha, cfg, root / "datasets" / "gate.jsonl", tmp_path / "logs", 1.0, cache=False)
    assert log.exists() and log.suffix == ".eval"
    assert (root / ".arena-base" / "arena_evals" / "__main__.py").exists()


def test_generate_baseline_errors_when_base_lacks_harness(tmp_path):
    root, sha = _git_repo(tmp_path, with_harness=False)
    cfg = config.load(root)
    with pytest.raises(RuntimeError, match="does not contain the eval harness"):
        ci.generate_baseline(sha, cfg, root / "datasets" / "gate.jsonl", tmp_path / "logs", 1.0, cache=False)
```

- [ ] **Step 2: Run them to verify they fail**

Run:

```bash
pytest tests/test_ci.py -v
```

Expected: FAIL, with `AttributeError: <module 'arena_evals.ci'` in the output.

- [ ] **Step 3: Add the orchestration**

Append to the end of `arena_evals/ci.py`:

```python
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
    return Path(p.stdout.strip().splitlines()[-1])


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

    # 1. cert check (auto-recertify only when this PR touched the judge)
    changed = gh.changed_files(pr)
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
    prior_body = next((c["body"] for c in reversed(gh.comments(pr)) if report.MARKER_PREFIX in (c.get("body") or "")), "")
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
pytest tests/test_ci.py -v
```

Expected: PASS (`10 passed`).

- [ ] **Step 5: Commit**

Run:

```bash
git add arena_evals/ci.py tests/test_ci.py
git commit -m "feat(ci): gate orchestration with worktree baseline, cache, cost cap, rerun policy"
```

### Task 27: GitHub Actions workflows, then (HUMAN) repository settings

**Files:**
- Create: `.github/workflows/gate.yml`
- Create: `.github/workflows/fork-notice.yml`
- Create: `.github/workflows/tests.yml`
- Test: `tests/test_workflows.py`

**Interfaces:**
- Consumes: CLI `cache-key`, `gate`; `datasets check`
- Produces: `arena-eval-gate` workflow (pull_request on the spec paths + workflow_dispatch with `pr` and `rerun_reason`), `arena-eval-fork-notice` (pull_request_target, no checkout, no secrets), `tests` workflow
- Produces: Artifact `arena-eval-gh-<run id>-<attempt>` containing `.arena-out/` incl. `phoenix/phoenix.db`

- [ ] **Step 1: Write the failing test**

Create `tests/test_workflows.py`:

```python
from pathlib import Path

import yaml

WF = Path(__file__).resolve().parent.parent / ".github" / "workflows"


def load(name):
    return yaml.safe_load((WF / name).read_text(encoding="utf-8"))


def test_gate_triggers_on_prompt_agent_config_and_dataset_paths():
    wf = load("gate.yml")
    on = wf[True]  # PyYAML parses the bare key `on` as boolean True
    assert on["pull_request"]["paths"] == ["prompts/**", "agents/**", "arena_evals/agents.py", "configs/**",
                                           "datasets/**"]
    assert set(on["workflow_dispatch"]["inputs"]) == {"pr", "rerun_reason"}


def test_gate_job_skips_fork_prs():
    job = load("gate.yml")["jobs"]["gate"]
    assert "github.event.pull_request.head.repo.full_name == github.repository" in job["if"]


def test_fork_notice_never_checks_out_pr_code():
    wf = load("fork-notice.yml")
    assert "pull_request_target" in wf[True]
    job = wf["jobs"]["notice"]
    assert "!= github.repository" in job["if"]
    assert not any("checkout" in str(s.get("uses", "")) for s in job["steps"])
    assert "ANTHROPIC_API_KEY" not in (WF / "fork-notice.yml").read_text()
    assert "Eval gate skipped: fork PRs have no access to API secrets." in job["steps"][0]["run"]


def test_gate_uploads_artifact_and_saves_baseline_cache_even_on_block():
    steps = load("gate.yml")["jobs"]["gate"]["steps"]
    upload = next(s for s in steps if str(s.get("uses", "")).startswith("actions/upload-artifact"))
    assert upload["if"] == "always()" and upload["with"]["path"] == ".arena-out"
    assert upload["with"]["name"].startswith("arena-eval-")
```

- [ ] **Step 2: Run it to verify it fails**

Run:

```bash
pytest tests/test_workflows.py -v
```

Expected: FAIL, with `FileNotFoundError` in the output.

- [ ] **Step 3: Create the workflows**

The gate job starts `phoenix serve` with `PHOENIX_WORKING_DIR=.arena-out/phoenix`, so the trace DB lands inside the uploaded artifact. The baseline cache key is computed by the CLI before `actions/cache/restore`, and the cache is saved even when the gate blocks (`if: always()`).

Create `.github/workflows/gate.yml`:

```yaml
name: arena-eval-gate

on:
  pull_request:
    paths:
      - "prompts/**"
      - "agents/**"
      - "arena_evals/agents.py"
      - "configs/**"
      - "datasets/**"
  workflow_dispatch:
    inputs:
      pr:
        description: "PR number to re-evaluate"
        required: true
      rerun_reason:
        description: "Why this blocked gate deserves a re-run (logged in the PR comment)"
        required: true

permissions:
  contents: read
  pull-requests: write
  issues: write
  statuses: write

concurrency:
  group: arena-eval-${{ github.event.pull_request.number || inputs.pr }}
  cancel-in-progress: true

jobs:
  gate:
    # Fork PRs get no secrets: skip here, fork-notice.yml explains on the PR.
    if: github.event_name == 'workflow_dispatch' || github.event.pull_request.head.repo.full_name == github.repository
    runs-on: ubuntu-latest
    timeout-minutes: 45
    env:
      ANTHROPIC_API_KEY: ${{ secrets.ANTHROPIC_API_KEY }}
      GITHUB_TOKEN: ${{ secrets.GITHUB_TOKEN }}
      PR: ${{ github.event.pull_request.number || inputs.pr }}
      RERUN_REASON: ${{ inputs.rerun_reason }}
      ARENA_TRACE: "1"
      PHOENIX_WORKING_DIR: ${{ github.workspace }}/.arena-out/phoenix
      INSPECT_CACHE_DIR: ${{ github.workspace }}/.inspect-cache
    steps:
      - uses: actions/checkout@v4
        with:
          ref: ${{ github.event_name == 'workflow_dispatch' && format('refs/pull/{0}/head', inputs.pr) || github.event.pull_request.head.sha }}
          fetch-depth: 0
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
      - run: pip install -e ".[dev]"
      - name: Start Phoenix (trace DB lands in PHOENIX_WORKING_DIR)
        run: |
          mkdir -p "$PHOENIX_WORKING_DIR"
          nohup phoenix serve > .arena-out/phoenix.log 2>&1 &
          for i in $(seq 1 60); do curl -sf http://localhost:6006 >/dev/null && exit 0; sleep 1; done
          echo "phoenix did not start"; cat .arena-out/phoenix.log; exit 1
      - name: Baseline cache key
        id: key
        run: echo "key=$(python -m arena_evals cache-key --pr "$PR" | tail -n 1)" >> "$GITHUB_OUTPUT"
      - uses: actions/cache/restore@v4
        id: baseline
        with:
          path: .arena-cache/baseline
          key: arena-baseline-${{ steps.key.outputs.key }}
      - uses: actions/cache/restore@v4
        with:
          path: .inspect-cache
          key: inspect-cache-${{ github.run_id }}
          restore-keys: inspect-cache-
      - name: Gate
        id: gate
        run: python -m arena_evals gate --pr "$PR" --rerun-reason "$RERUN_REASON"
      - uses: actions/cache/save@v4
        if: always() && steps.baseline.outputs.cache-hit != 'true' && hashFiles('.arena-cache/baseline/**') != ''
        with:
          path: .arena-cache/baseline
          key: arena-baseline-${{ steps.key.outputs.key }}
      - uses: actions/cache/save@v4
        if: always() && hashFiles('.inspect-cache/**') != ''
        with:
          path: .inspect-cache
          key: inspect-cache-${{ github.run_id }}
      - uses: actions/upload-artifact@v4
        if: always()
        with:
          name: arena-eval-gh-${{ github.run_id }}-${{ github.run_attempt }}
          path: .arena-out
          if-no-files-found: warn
```

Create `.github/workflows/fork-notice.yml`:

```yaml
name: arena-eval-fork-notice

on:
  pull_request_target:
    paths:
      - "prompts/**"
      - "agents/**"
      - "arena_evals/agents.py"
      - "configs/**"
      - "datasets/**"

permissions:
  pull-requests: write

jobs:
  notice:
    # No checkout of PR code and no secrets: only posts a comment.
    if: github.event.pull_request.head.repo.full_name != github.repository
    runs-on: ubuntu-latest
    steps:
      - env:
          GH_TOKEN: ${{ secrets.GITHUB_TOKEN }}
        run: >
          gh pr comment ${{ github.event.pull_request.number }} --repo ${{ github.repository }}
          --body "Eval gate skipped: fork PRs have no access to API secrets. A maintainer can re-open this change from a same-repo branch."
```

Create `.github/workflows/tests.yml`:

```yaml
name: tests

on:
  push:
    branches: [main]
  pull_request:

jobs:
  tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: "3.12"
          cache: pip
      - run: pip install -e ".[dev]"
      - run: pytest -q
      - run: python -m arena_evals datasets check
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
pytest tests/test_workflows.py -v
```

Expected: PASS (`4 passed`).

- [ ] **Step 5: Commit**

Run:

```bash
git add .github tests/test_workflows.py
git commit -m "ci: gate, fork-notice and tests workflows"
```

- [ ] **Step 6: (HUMAN) Create the GitHub repository and push**

PRD open question 1 (where the repo lives) is answered here by the maintainer.

Run (replace OWNER):

```bash
gh repo create OWNER/arena-evals --private --source . --push
```

Expected: the repo exists on GitHub and the `tests` workflow runs green on `main`.

- [ ] **Step 7: (HUMAN) Add the API secret**

GitHub -> repo -> Settings -> Secrets and variables -> Actions -> New repository secret. Name `ANTHROPIC_API_KEY`, value: an Anthropic key with a monthly spend limit at least 3x `cost.max_usd_per_gate`. `GITHUB_TOKEN` is provided automatically; the workflow's `permissions:` block grants `statuses: write` and `pull-requests: write`.

- [ ] **Step 8: (HUMAN) Run the gate once, then make it required**

1. Create a branch, make a small wording change in `prompts/concise.md` (not the gated variant, so the verdict should be pass), push and open a PR. The `arena-eval-gate` workflow runs; wait for the PR comment and the `arena-eval/gate` commit status. This first run is a baseline cache miss.

2. Settings -> Branches -> Add branch protection rule (or ruleset) for `main` -> Require status checks to pass before merging -> search for and select `arena-eval/gate` (it only appears after step 1). Also select `tests`. Save.

3. Merge or close the PR. Record the runtime of the gate job; it must be under 15 minutes on a cache hit for the baseline. If the cache-miss run exceeds 15 minutes, raise `max_connections` in `configs/eval.yaml`.

Known limitation (spec gap, not solved by this plan): the gate workflow is path-filtered, so a PR that touches none of the gated paths never gets an `arena-eval/gate` status, and with the status required it waits forever. Until the maintainer decides how to handle that (e.g. merging such PRs with an admin bypass), expect it on docs-only or test-only PRs.

### Task 28: A/A flake test

**Files:**
- Modify: `arena_evals/ci.py`
- Modify: `tests/test_ci.py`

**Interfaces:**
- Consumes: `run.generate/score/paired_deltas`, `stats.bootstrap.paired_bootstrap/gate_decision`
- Produces: `ci.aa(runs, cfg, split="gate") -> int` (0 iff false-block rate <= 5%); writes `.arena-out/aa/aa.json`; CLI `aa --runs 20 [--split gate]`

- [ ] **Step 1: Write the failing test**

Append to the end of `tests/test_ci.py`:

```python
def test_aa_counts_false_blocks(repo):
    cfg, calls = repo
    calls["cand_script"] = GOOD
    assert ci.aa(2, cfg, split="gate") == 0                  # identical scripted agent: never blocks
    out = json.loads((cfg.root / ".arena-out" / "aa" / "aa.json").read_text())
    assert out == {"runs": 2, "verdicts": ["pass", "pass"], "false_block_rate": 0.0}
    assert calls["cand"] == 4                                # 2 runs x 2 sides, cache bypassed
```

- [ ] **Step 2: Run it to verify it fails**

Run:

```bash
pytest tests/test_ci.py -v -k aa
```

Expected: FAIL, with `AttributeError: module 'arena_evals.ci' has no attribute 'aa'` in the output.

- [ ] **Step 3: Add the A/A command**

Append to the end of `arena_evals/ci.py`:

```python
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
```

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```bash
pytest tests/test_ci.py -v
```

Expected: PASS (`11 passed`).

- [ ] **Step 5: Commit**

Run:

```bash
git add arena_evals/ci.py tests/test_ci.py
git commit -m "feat(ci): A/A flake test command"
```

- [ ] **Step 6: (HUMAN) Run the A/A test for real**

Cost: 20 runs x 2 sides x (300 tasks x 3 repeats) = 40 full runs, about 20/3 = 6.7x the Task 17 gate estimate. The command sets its own cap at `runs x cost.max_usd_per_gate`. Approve the spend first.

Run (needs ANTHROPIC_API_KEY, costs money):

```bash
python -m arena_evals aa --runs 20
```

Expected: 20 lines `A/A run i/20: pass|warn|block delta ...` and a final `A/A false-block rate: X% (k/20); pass iff <= 5%`, exit 0 when k <= 1. Expected under the null is about 2.5%. If k >= 2, do not loosen the gate: inspect `.arena-out/aa/*/results.jsonl` for judge errors or flaky tasks and record the result in the README.

## Milestone M8: Demo

### Task 29: (HUMAN) Verify the planted regression costs at least 8 pts on the gate split

**Files:**
- Modify: `prompts/regressed.md (only if the effect is too small)`

**Interfaces:**
- Consumes: CLI `run`, `compare`; `datasets/gate.jsonl`; a certified judge
- Produces: An observed delta for `regressed` vs `baseline` on the gate split, <= -8 pts

- [ ] **Step 1: (HUMAN) Run both variants on the gate split**

Run (needs ANTHROPIC_API_KEY, costs money: about 2x a single gate side each):

```bash
python -m arena_evals run --split gate --variant baseline --max-usd 40
python -m arena_evals run --split gate --variant regressed --max-usd 40
python -m arena_evals compare --base <baseline results.jsonl> --cand <regressed results.jsonl>
```

Expected: `delta D pts (95% CI ...; one-sided 97.5% upper U), ...` and `verdict: block`. The demo needs D <= -8.00.

- [ ] **Step 2: (HUMAN) If D > -8: strengthen the regression and re-measure**

Remove the abstention rule too, so `regressed` also answers unanswerable questions. Replace the entire contents of `prompts/regressed.md` with:

```markdown
You are the Halcyon Robotics internal assistant. Today is 2026-09-01. You answer employee questions using the company document corpus, which you reach through three tools: `search_docs`, `get_doc` and `calculate`.

How to work:

1. Search for the relevant documents with `search_docs`.
2. When two documents give different values for the same fact, use the one with the latest `effective_date` that is not after today. A newer document says which document it supersedes.
3. Use `calculate` for every arithmetic step. Do not do arithmetic in your head.
4. Always give your best answer.

Write a short, direct answer of one to three sentences, then the FINAL line.
```

Then run `pytest tests/test_solver.py -v` (must still pass), re-run the regressed and compare commands above, and commit with `git commit -am "demo: strengthen planted regression"`. Record the final D in the README table.

### Task 30: Demo PRs (planted regression blocked, no-op passes), rerun policy check, README

**Files:**
- Create: `README.md`

**Interfaces:**
- Consumes: the working gate (Tasks 24-28) on GitHub with the required check
- Produces: A blocked `regressed` PR and a passing no-op PR, both real `prompts/**` diffs; a documented README

- [ ] **Step 1: Write the README**

Create `README.md`:

````markdown
# Arena Eval Harness

A quality gate for the arena stand-in agents. A PR that changes `prompts/**`, `agents/**`, `arena_evals/agents.py`, `configs/**` or `datasets/**` gets ONE comment with the paired change in task success (with a two-sided 95% CI and a one-sided 97.5% upper bound), and merge is blocked when a drop of at least 2 pts is statistically real.

Requirements: `PRD.md`. Design: `docs/superpowers/specs/2026-09-30-arena-eval-harness-design.md`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows Git Bash: source .venv/Scripts/activate
pip install -e ".[dev]"
export ANTHROPIC_API_KEY=...          # only for commands that call models
pytest -q                             # unit + mock end-to-end tests, no API calls
```

## Commands

| Command | What it does |
|---|---|
| `python -m arena_evals corpus build` | Render `corpus/docs/*.md` from `corpus/seed.yaml` (deterministic) |
| `python -m arena_evals datasets draft --split S` | LLM-draft tasks into `datasets/drafts/S.jsonl` |
| `python -m arena_evals datasets spotcheck --split S` | Hand-check a seeded 20% sample, write `datasets/S.jsonl` and the label-noise floor |
| `python -m arena_evals datasets check` | Schema, gold IDs, gate mix, cross-split near-duplicates (runs in CI) |
| `python -m arena_evals run --split S --variant V` | Agent run + scoring; prints success with 95% CI, cost, wall time |
| `python -m arena_evals compare --base A --cand B` | Paired bootstrap between two `results.jsonl` files, with the gate verdict |
| `python -m arena_evals simulate [--sd X]` | Monte Carlo power and CI coverage of the gate rule |
| `python -m arena_evals calibrate export` / `calibrate label` | Export 100 stratified judge examples; label them blind in the terminal |
| `python -m arena_evals certify` | Agreement (kappa, AC1, PABAK) + verbosity bias tests, writes `configs/judge.cert.json` |
| `python -m arena_evals gate --pr N` | What CI runs (needs `GITHUB_TOKEN`, `GITHUB_REPOSITORY`) |
| `python -m arena_evals aa --runs 20` | A/A flake test: head vs itself; pass iff false-block rate <= 5% |

## How the gate decides

Per task, success is averaged over its k = 3 repeats; d_i = candidate - baseline. A 10,000-resample seeded paired bootstrap over tasks gives the delta, its two-sided 95% CI (display only) and the one-sided 97.5% upper bound (the 97.5th percentile of resampled mean deltas).

- **block** iff delta <= -2 pts AND upper bound < 0 (exit 1, status failure)
- **warn** iff delta <= -2 pts AND upper bound >= 0 (exit 0, status success, "warn")
- **pass** otherwise (exit 0)
- **error** (exit 2, no verdict): judge uncertified or stale, judge errors above 2% of samples, cost cap hit, base commit without the harness

Per-tag deltas are advisory (Benjamini-Hochberg, q = 0.05) and never gate. Every comment states the minimum detectable effect (80% power) for the observed n and SD; at n = 300 and SD 0.45 it is about 7.3 pts, so smaller real drops can pass.

## Reading the PR comment

The headline gives the delta, both intervals and n. The table gives baseline and candidate success with CIs, P(delta < 0), MDE, judge errors, cost against the cap, baseline cache hit or miss, judge certification, and the dataset and rubric hashes. "Worst 5 regressions" lists task IDs with their trace IDs. To view a trace: download the `arena-eval-<run_id>` artifact from the run, unzip it, run `PHOENIX_WORKING_DIR=<unzipped>/phoenix phoenix serve`, open http://localhost:6006 and search for the trace ID.

## Re-running a blocked gate

Re-running the job on the same commit re-posts the block without evaluating. To re-evaluate, run the `arena-eval-gate` workflow manually (Actions tab, "Run workflow") with the PR number and a `rerun_reason`; the reason is written into the comment and the response cache is bypassed. A new commit is always a fresh evaluation.

## Judge certification

The gate refuses to run unless `configs/judge.cert.json` is certified and matches the current rubric hash, judge model, judge temperature and labels hash. Certified means kappa 95% CI lower bound >= 0.4 AND kappa point >= 0.6 on about 100 hand labels, AND the verbosity slope upper bound <= 2 pts per +50% length, AND the length partial correlation is not significantly > 0. A PR that edits `prompts/judge/**` or `configs/models.yaml` is re-certified automatically against the committed labels; commit the new cert from the run artifact.

## Measured numbers

Fill these in from Tasks 17, 23, 28 and 29 of the implementation plan; do not guess them.

| Quantity | Value | Source |
|---|---|---|
| Dev run (100 tasks x 3) cost / wall time | not measured yet | Task 17 |
| Estimated gate cost / runtime | not measured yet | Task 17 |
| Measured paired SD | not measured yet | Task 17 |
| Simulated power at -8 pts with measured SD | not measured yet | Task 17 |
| Judge kappa (95% CI), n labels | not measured yet | Task 23 |
| A/A false-block rate | not measured yet | Task 28 |
| Planted regression effect on the gate split | not measured yet | Task 29 |

## Fork PRs

Fork PRs get no secrets, so the gate is skipped and a bot comment explains why; the required check stays pending. Re-open the change from a same-repo branch.
````

- [ ] **Step 2: Commit and push**

Run:

```bash
git add README.md
git commit -m "docs: README for the eval gate"
```

Run (needs the GitHub remote from Task 27):

```bash
git push origin main
```

- [ ] **Step 3: (HUMAN) Open the no-op PR**

Run (HUMAN):

```bash
git switch -c demo/noop
sed -i 's/Write a short, direct answer of one to three sentences/Write a direct, short answer of one to three sentences/' prompts/baseline.md
git commit -am "demo: no-op wording change to the baseline prompt"
git push -u origin demo/noop
gh pr create --title "demo: no-op prompt change" --body "Wording-only change; the gate should pass."
```

Expected within 15 minutes: one PR comment headed `## Arena eval gate: PASS` (or, about 2.5% of the time by design, WARN or BLOCK), the `arena-eval/gate` status green. Do not merge; close the PR afterwards.

- [ ] **Step 4: (HUMAN) Open the planted-regression PR**

Run (HUMAN):

```bash
git switch main
git switch -c demo/regressed
cp prompts/regressed.md prompts/baseline.md
git commit -am "demo: drop cite-and-verify from the baseline prompt"
git push -u origin demo/regressed
gh pr create --title "demo: planted regression" --body "Removes cite-and-verify; the gate should block."
```

Expected: a comment whose headline reads like `**This prompt change dropped task success by X pts** (two-sided 95% CI ...; one-sided 97.5% upper bound ..., n=300). **Merge blocked.**`, the worst-5 table with trace IDs, and the `arena-eval/gate` status red so the merge button is disabled.

- [ ] **Step 5: (HUMAN) Check the trace artifact**

Open the gate run -> Artifacts -> download `arena-eval-gh-<run id>-1`, unzip, run `PHOENIX_WORKING_DIR=<unzipped>/phoenix phoenix serve`, open http://localhost:6006 and find one trace ID from the worst-5 table.

- [ ] **Step 6: (HUMAN) Check the rerun policy**

1. On the blocked PR's run, click "Re-run jobs". Expected: the job fails again quickly without evaluating, and the comment gains the line `Re-run on the same head SHA without a rerun_reason: the previous block stands.`

2. Actions -> arena-eval-gate -> Run workflow, `pr` = the PR number, `rerun_reason` = `demo: verify rerun policy`. Expected: a full evaluation (cache bypassed), the comment shows `| Rerun reason | demo: verify rerun policy |`, and it blocks again.

3. Optional power check (PRD success metric, costs about 5 gate runs): repeat step 2 four more times with reasons `power check 1..4` and record how many of the 5 runs blocked (expect at least 4 when the true effect is at least 8 pts).

- [ ] **Step 7: (HUMAN) Close the demo PRs and record results**

Close both PRs without merging and delete the branches. Fill in the README "Measured numbers" table from Tasks 17, 23, 28 and 29, then commit with `git commit -am "docs: measured numbers"` and push.

---

## Self-Review

### How this plan was checked

Every file-creating or file-editing step was replayed mechanically, in order, into an empty directory containing only `PRD.md` and `docs/`. That is 89 file operations; every `Run:` block was executed, every "Expected: FAIL" was checked for its error text and every "Expected: PASS" for its exact pass count. Result: 0 mismatches across Tasks 1-30, 26 commits, clean working tree. The replay used a Python 3.12 venv holding the pinned dependencies; the venv creation and `pip install` lines themselves were not replayed. Separately, `pip install -e ".[dev]"` with this plan's `pyproject.toml` was run in a fresh Python 3.11.15 venv: it installed inspect-ai 0.3.272 and arize-phoenix 20.18.0. On that 3.11 venv, the replayed tree's `pytest -m "slow or not slow"` gave **148 passed**, including the three Monte Carlo tests. Task 8's spike script was run on 2026-10-01 against a live `phoenix serve`, and all 10 checks printed OK. The spike's span was found in `traces` inside `$PHOENIX_WORKING_DIR/phoenix.db`.

Steps that need an API key, a person or GitHub were **not** executed: Tasks 7, 17, 21, 29 and the HUMAN steps of 10, 23, 27, 28 and 30. Their expected outputs describe what should happen; nobody has observed them.

### Spec and PRD coverage

| Spec section / PRD item | Task(s) | Notes |
|---|---|---|
| 1.1 Corpus (40 docs, 5 conflict pairs, deterministic build, byte-identical rebuild) | 2 | |
| 1.2 Tools (BM25 top 5 with ID tie-break, get_doc error text, safe calculator) | 2, 3 | empty search returns `no results for query: ...` (spec silent) |
| 1.3 Variants, output contract, limits, `format_error` | 4, 10 | tool-call cap applied at scoring time; `message_limit` is the hard stop |
| 1.4 Splits/sizes, mix, drafting, spot-check, label-noise floor, near-dup | 5, 6, 7 | |
| 1.5 Deterministic scorers + success table | 9 | recall `None` when no gold docs |
| 2.1 Two-phase run, response cache, CostMeter, results/manifest schema | 3, 10, 16 | cap checked per sample, tool call and judge call (not per agent model call) |
| 2.2 Judge (inputs, rubric hash, pydantic, retry once, judge_error, 2% rule, self-preference) | 15, 16, 26 | |
| 2.3 Calibration export/label, agreement + CIs, certification thresholds | 18, 19, 20, 21 | |
| 2.3 Verbosity bias (padding, compression, slope, partial correlation), cert JSON, staleness | 22, 23 | |
| 2.4 Gate: cert check + auto-recertify, rerun policy, cache key, worktree baseline, pre-flight + cap, compare, BH per-tag, report, status, artifact, exit codes | 24, 25, 26, 27 | |
| 2.5 Tracing (single module, span attributes, trace ID in results) | 14, 16, 27 | |
| 3 Code layout + 3.1 interfaces | all | deviations listed below |
| 5 Error handling table (all 12 rows) | 3, 10, 15, 16, 20, 25, 26, 27 | fork row: workflow `if:` + fork-notice job |
| 6 Testing table (all 9 rows) | 2, 3, 5, 9, 10, 11, 12, 13, 28, 29 | A/A and demo check are HUMAN runs |
| I1 answer_kind / I2 judge error = missing / I3 trigger paths / I4 scorer_hash in key / I5 one-sided verbosity / I6 partial-corr rule / I7 starting cost cap / I8 rerun = same SHA | 5 / 9, 16, 26 / 27 / 25 / 22, 23 / 22 / 1, 17 / 25, 26 | |
| PRD FR1 datasets | 5, 6, 7 | |
| PRD FR2 scorers, errors, secondary metrics | 9, 10, 15, 16 | secondary metrics stored per sample; only success, cost and per-tag deltas are summarized |
| PRD FR3 calibration | 18-21, 26 | human-human double label: optional note in 21 |
| PRD FR4 bias tests | 16, 22, 23 | |
| PRD FR5 statistics, gate rule, MDE, power | 11, 12, 13, 17 | BCa and McNemar cross-check ("consider") not implemented |
| PRD FR6 tracing | 14, 16, 27, 30 | |
| PRD FR7 CI gate, cache, limits, rerun, cost cap, forks, required check | 24-28, 30 | required check + path filter caveat in Task 27 |
| PRD NFR reproducibility, runtime, cost, isolation | 1, 3, 10, 17, 27 | |
| M1 / M2 / M3 / M4 / M5 / M6 / M7 / M8 | 1-7 / 8-10 / 11-13 / 14-17 / 18-21 / 22-23 / 24-28 / 29-30 | 30 tasks |

### Placeholder scan

`grep -nE "TBD|TODO|similar to|appropriate|handle edge cases"` over this file matches only this line, which quotes the pattern. No task, step or code block matches.

### Name and type consistency

The code blocks are the files that ran in the replay, and every test imports the real names, so a mismatched name fails the replay. A regex pass over the 193 `module.name` references in the Interfaces lines resolved all of them against the replayed code except two false positives: `__main__.py` is a file name and `stats.bootstrap` is a submodule path. Signatures that differ from spec 3.1 are listed under deviations.

### Python syntax check

All 76 fenced `python` blocks parse with `ast.parse`. On purpose, the two one-line fragments in Task 23's "replace every occurrence" edit are fenced as `text`, not `python`.

### Review Focus coverage

Each of the five Review Focus lines names tests in the task that owns the code (Tasks 3, 9, 10, 11, 12, 22, 25, 26). All of them ran green in the replay. Other edge cases the review prompt listed also have tests: empty `search_docs` results (`test_agents.py`, Task 3), comments over 65k characters (`test_report.py`, Task 24) and fork PRs (`test_workflows.py`, Task 27).

### Deviations from the spec (deliberate; reviewers should look at these)

- `system_message()` is not used. It mangles the JSON braces of the output contract (observed), so `arena_agent` inserts a `ChatMessageSystem` instead.
- `CostMeter(cap_usd, prices=None)` takes an extra `prices` argument. The cap is checked before each sample, tool call and judge call, not before every agent model call: Inspect swallows exceptions raised in hooks (observed), so the check cannot live in a hook. Overspend is bounded by the calls already in flight.
- `tracing.init(working_dir)` sets `PHOENIX_WORKING_DIR`, but `phoenix serve` writes the database, not the exporter (observed). CI starts `phoenix serve` for that reason.
- The spec has `bias.verbosity(labels, cfg)`; this plan uses `bias.verbosity(labels, perturbed, cfg)`. The re-judged perturbations come from `ci.certify`, which runs all model calls in one event loop.
- `simulate_coverage(..., n_resamples=2000)` gains a keyword. Seeds default to 0 because `seed: int` after defaulted parameters is not valid Python.
- `datasets.draft(split, corpus_docs, model)`: the parameter is renamed so it does not shadow the `corpus` module.
- `calibrate export` uses 1 repeat per variant, not k = 3, to save cost.
- A compression perturbation is kept only if its numbers and doc IDs are unchanged (`facts_signature`). This is this plan's deterministic reading of "answer_match/citation extraction changes" for free text.
- `task_means` and `paired_deltas` live in `run.py`, not `ci.py`, so the `run` and `compare` commands work from M2/M3.
- Extra CLI commands `compare` and `cache-key` are additive; the workflow needs `cache-key` before `actions/cache/restore`.
