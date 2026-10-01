# Arena Eval Harness

A quality gate for the arena stand-in agents. A PR that changes `prompts/**`, `agents/**`, `arena_evals/agents.py`, `configs/**` or `datasets/**` gets ONE comment with the paired change in task success (with a two-sided 95% CI and a one-sided 97.5% upper bound), and merge is blocked when a drop of at least 2 pts is statistically real.

Requirements: `PRD.md`. Design: `docs/superpowers/specs/2026-09-30-arena-eval-harness-design.md`.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate   # Windows Git Bash: source .venv/Scripts/activate
pip install -e ".[dev]"
export ANTHROPIC_BASE_URL=https://api.deepseek.com/anthropic   # DeepSeek's Anthropic-compatible API
export ANTHROPIC_API_KEY=...          # your DeepSeek key; only for commands that call models
pytest -q                             # unit + mock end-to-end tests, no API calls
```

Models are pinned in `configs/models.yaml`: agent `deepseek-flash`, judge `deepseek-v4-pro`, served through the `anthropic/` provider.
Use explicit DeepSeek names: that endpoint maps any Claude model name to `deepseek-flash`, which would make the manifest lie.
Agent and judge share a vendor, so the self-preference warning fires on every run; judge certification (agreement with your hand
labels, verbosity tests) is the check that matters. Prices in `models.yaml` are DeepSeek peak-hour rates; the cost cap is `$4`.

## Datasets and demo scale

This repo runs at **demo scale** to fit a small model budget: splits are `dev` 100, `calibration` 120, `gate` 149
(the PRD's n=300 is the default; set `ARENA_SPLIT_SIZES=gate=300` to rebuild at full size). With n=149 the gate's
minimum detectable effect is larger than the 7.3 pts quoted for n=300, so it reliably catches only big regressions.

- **Spot-check:** a seeded 10% sample of each split was reviewed against the source documents (dev 11, calibration 13,
  gate 17). One dev task was dropped as ambiguous (a "combined raise" question with two valid answers). Label-noise
  floor: dev 9.1% (95% CI 1.6-37.7%), calibration and gate 0 of the sampled tasks needed changes (gate floor 0.0%,
  95% CI 0.0-18.4%). The intervals are wide because the samples are small. Verdicts: `datasets/spotcheck.jsonl`.
- **Small corpus:** 40 documents and 5 conflicting document pairs, so the same facts recur across splits. The
  near-duplicate check (`python -m arena_evals datasets check`) flagged 26 gate tasks that were verbatim or
  near-verbatim repeats of dev/calibration tasks; they were reworded (same fact, same reference). The conflicting
  category therefore tests about five distinct facts, and the splits overlap in *facts* even though not in wording.

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
