# Design: Arena Agent Eval Harness

Status: design v1 · Implements: `PRD.md` v0.3 · Date: 2026-09-30

The PRD is the WHAT (requirements, thresholds, milestones). This document is the HOW. Requirements are referenced by ID (FR1-FR7, NFR, M1-M9) and not restated. Where this design picks one reading of an under-specified PRD point, it says so under "Interpretations".

## 1. Stand-in arena (FR1, FR2, A1, A2; M1)

### 1.1 Corpus: Halcyon Robotics

- 40 markdown docs for the invented company "Halcyon Robotics": HR policy (10), product specs (9), pricing tiers (7), incident postmortems (10), org chart (4).
- Stable doc IDs by kind: `HR-001`, `PRD-001`, `PRC-001`, `INC-001`, `ORG-001`. IDs never change once issued.
- Each doc has YAML front matter: `id`, `title`, `kind`, `effective_date` (ISO date), optional `supersedes` (doc ID). Bodies contain concrete numbers, dates, names and explicit cross-references by doc ID.
- 5 conflicting pairs are included in the 40: an older and a newer doc stating different values for the same fact, distinguished by `effective_date` (newer one carries `supersedes`). The currently effective doc is the one with the latest `effective_date` not after the corpus "as-of" date in `corpus/seed.yaml`.
- Generation is deterministic templating (no LLM): `corpus/seed.yaml` holds every fact (entities, numbers, dates, conflict pairs); `python -m arena_evals corpus build` renders `corpus/docs/*.md`. Output is committed; a test asserts a rebuild is byte-identical.

### 1.2 Tools (Inspect `@tool`s, in `agents.py`)

| Tool | Signature | Behavior |
|---|---|---|
| `search_docs` | `search_docs(query: str) -> str` | BM25 keyword search over doc bodies (stdlib implementation), top 5, ties broken by doc ID. Returns lines `"<id> \| <title> \| <snippet>"`, snippet = first 200 chars of best-matching paragraph. |
| `get_doc` | `get_doc(id: str) -> str` | Full markdown incl. front matter. Unknown ID returns `"error: no document with id <id>"`. |
| `calculate` | `calculate(expr: str) -> str` | Safe arithmetic via `ast`: numeric literals, `+ - * / // % **`, unary minus, parentheses only. No names, calls, attributes. `**` exponent capped at abs 100; results with abs > 1e15 rejected. Errors return `"error: <reason>"`. |

Tool errors are always returned as tool output, never raised (FR7 agent limits).

### 1.3 Agent variants (Inspect solvers)

- One solver factory `arena_agent(variant: str) -> Solver` = `system_message(prompts/<variant>.md)` + `use_tools(search_docs, get_doc, calculate)` + `generate(tool_calls="loop")`. Variants differ only in the prompt file.
- `prompts/baseline.md` (careful, cites doc IDs, abstains when unsupported), `prompts/concise.md`, `prompts/verbose.md`, `prompts/regressed.md` (baseline minus cite-and-verify instructions).
- All prompts share one required output contract, appended from `prompts/_output_contract.md`: the final message ends with a line `FINAL: {"answer": str, "citations": [doc_id, ...], "abstain": bool}`. A missing or unparseable `FINAL` line is a sample FAIL with `status = "format_error"`.
- Limits (FR7) come from `configs/eval.yaml` and map to Inspect `message_limit`, `token_limit`, `time_limit` per sample plus a tool-call counter in the loop. Hitting any limit gives `status = "limit"` and FAIL.
- The gate evaluates one configured variant (`configs/gate.yaml: variant: baseline`) on base vs head. The M8 demo PR replaces the contents of `prompts/baseline.md` with `prompts/regressed.md`, which is a real `prompts/**` diff.

### 1.4 Tasks

- Splits and sizes: `gate` 300, `calibration` 120, `dev` 100. Files `datasets/{gate,calibration,dev}.jsonl`. Gate type mix exactly per FR1 (lookup 90, multi-hop 75, arithmetic 45, conflicting 45, unanswerable 45).
- Authoring: `python -m arena_evals datasets draft --split <s>` asks the drafting model to write tasks from the corpus into `datasets/drafts/<split>.jsonl`. The maintainer spot-checks a seeded random 20% sample per split via `datasets spotcheck` (verdict `ok | fixed | dropped` per task, stored in `datasets/spotcheck.jsonl`). Label-noise floor = (`fixed` + `dropped`) / sample, reported with a 95% Wilson interval in every run manifest.
- Near-duplicate check: `datasets check` normalizes `input` (lowercase, strip punctuation), computes character 5-gram Jaccard for every gate x (dev ∪ calibration) pair, and exits non-zero if any pair is ≥ 0.8. O(n^2) over ~66k pairs, fine at this size. Runs in CI on every PR.

Task JSONL schema (FR1 fields plus `answer_kind`, see Interpretations I1):

```json
{"id": "gate-0042", "input": "What is the monthly price of the Pro tier for 25 seats?",
 "reference": "4,475 USD", "gold_doc_ids": ["PRC-003"], "type": "arithmetic",
 "answer_kind": "numeric", "tags": ["pricing"], "split": "gate"}
```

- `type`: `lookup | multi_hop | arithmetic | conflicting | unanswerable`.
- `answer_kind`: `exact | numeric | free_text | abstain`. `unanswerable` tasks always have `abstain` and `reference: null`, `gold_doc_ids: []`. For `conflicting`, `reference` is the currently effective doc's value and `gold_doc_ids` contains only the currently effective doc.

### 1.5 Scorers (FR2)

Deterministic scorers run first; the judge runs only when `answer_kind == "free_text"`.

| Scorer | Output | Rule |
|---|---|---|
| `answer_match` | bool | `exact`: normalized string equality (casefold, collapse whitespace, strip punctuation). `numeric`: first number parsed from answer (commas, currency, `%` stripped) within `max(0.005 * abs(ref), 0.01)` of reference. |
| `gold_retrieval` | bool | at least one `gold_doc_ids` entry was fetched via `get_doc` during the run. Secondary metric only. |
| `citation` | precision, recall | precision = cited ∩ gold / cited; recall = cited ∩ gold / gold. Cited IDs not in the corpus count as wrong. Empty citations: precision undefined (excluded from mean), recall 0. |
| `abstention` | bool | `FINAL.abstain == true`. |
| `judge` | `{correct, faithful, reason}` or error | Section 2.2. Judge pass = `correct AND faithful`. |

Per-task success per repeat (binary, FR2):

| type | success iff |
|---|---|
| lookup, multi_hop, arithmetic | (answer_match, or judge pass when free_text) AND cites ≥ 1 gold doc |
| conflicting | answer_match against the currently effective value AND cites the currently effective doc (conflict handling) |
| unanswerable | abstention |
| any, `status != "ok"` | FAIL |

## 2. Scoring, certification, gate (FR2-FR7; M2-M7)

### 2.1 Run pipeline

- One Inspect `Task` per (split, variant) with `epochs=k` (default 3, FR5). Each sample = task x repeat.
- Two phases so that baseline and candidate share scoring code (see 2.4): phase A `generate` runs the agent and writes an Inspect `.eval` log; phase B `score` applies the scorers above via `inspect_ai.score()` and writes `results.jsonl` + `manifest.json`.
- Model responses use Inspect's response cache (`cache=CachePolicy(expiry=None)`), keyed by request; the cache dir is persisted with `actions/cache` (NFR reproducibility).
- Cost meter: every model call (agent and judge) adds `usage x price` (prices in `configs/models.yaml`) to a process-wide `CostMeter`; before each call, if spent ≥ cap, it raises `CostCapExceeded`, which aborts the whole run. No partial results are scored.

Run result JSONL (one line per sample):

```json
{"run_id": "r-20260930-ab12", "variant": "baseline", "task_id": "gate-0042", "repeat": 0,
 "status": "ok", "answer": "4,475 USD", "citations": ["PRC-003"], "abstain": false,
 "scores": {"answer_match": true, "gold_retrieval": true, "citation_precision": 1.0,
            "citation_recall": 1.0, "abstention": false, "judge": null, "judge_error": false},
 "success": true, "latency_s": 8.4, "tokens_in": 5120, "tokens_out": 310,
 "cost_usd": 0.021, "trace_id": "5f1c..."}
```

`status`: `ok | agent_error | timeout | limit | format_error`. `judge` when present: `{"correct": bool, "faithful": bool, "reason": str}`; `judge_error: true` means both judge attempts failed schema validation.

`manifest.json`: `run_id`, `git_sha`, `variant`, `split`, `dataset_hash` (sha256 of the split file), `prompt_hash` (sha256 of the variant prompt + output contract), `rubric_hash`, `scorer_hash` (sha256 of `arena_evals/scorers.py`), agent and judge model IDs, temperatures, `k`, seeds, `inspect_version`, limits, `cost_usd`, `label_noise_floor` `{point, ci95}`, full resolved config.

### 2.2 Judge

- Inputs: task `input`, `reference`, the agent's `answer`, and full text of every cited doc that exists (for faithfulness).
- Prompt files: `prompts/judge/system.md` and `prompts/judge/rubric.md`. `rubric_hash = sha256(system.md bytes + rubric.md bytes)`.
- Output: JSON `{"correct": bool, "faithful": bool, "reason": str}` validated with a pydantic model. On invalid output: retry once with the validation error appended; second failure sets `judge_error: true`.
- A judge-error sample is treated as missing, not FAIL (Interpretation I2): the task's success is averaged over its remaining repeats; a task with no valid repeat on either side is dropped from the paired comparison. Judge errors are counted and shown in the comment. If > 2% of samples in either run are judge errors, the gate outcome is `error` (no verdict).
- Judge model and temperature live in `configs/models.yaml`, separate from the agent config (NFR isolation). If agent and judge model IDs share a family prefix (e.g. both `claude-`), the run manifest and PR comment carry a self-preference warning (FR4).

### 2.3 Judge certification (FR3, FR4; M5, M6)

Commands:

- `python -m arena_evals calibrate export` runs all four variants on the `calibration` split, keeps free_text samples, and draws 100 stratified by task type x first-pass judge label (proportional, minimum 5 per non-empty cell). Writes `calibration/to_label.jsonl` (no judge verdict shown, blind labeling) and `calibration/key.jsonl` (judge verdicts).
- The maintainer fills `calibration/labels.jsonl` by hand or via `calibrate label` (a line-by-line terminal prompt).
- `python -m arena_evals certify` re-judges the labeled examples with the current judge + rubric, runs the bias tests, and writes `configs/judge.cert.json`. Exit 0 iff certified.

Label JSONL:

```json
{"example_id": "cal-0017-verbose-r0", "task_id": "cal-0017", "variant": "verbose",
 "human_label": "pass", "labeler": "zeesh", "note": ""}
```

Agreement metrics (stats/agreement.py): raw agreement, Cohen's κ, Gwet's AC1, PABAK, precision/recall of judge pass vs human pass, 2x2 confusion matrix; each with a 10k-resample seeded percentile bootstrap CI over examples. Certified on agreement iff κ 95% CI lower bound ≥ 0.4 AND κ point ≥ 0.6 (`configs/cert.yaml`).

Verbosity bias (FR4):

- Same-content perturbation: for each labeled example, a padded version (+50% and +100% characters by appending sentences from a fixed content-free filler list, seeded) and a compressed version (perturber model instructed to cut ≥ 30% without removing facts; pairs where the deterministic `answer_match`/citation extraction changes are discarded). Each is re-judged. Effect = slope of judge pass (in pts) against `log(len_perturbed / len_orig) / log(1.5)`, i.e. pts per +50% length. Pass iff the bootstrap 95% CI upper bound of the slope ≤ 2 pts.
- Real-output check: on labeled `concise` and `verbose` examples, partial correlation of judge pass with log length controlling for human label (residualize both on human label). Pass iff the 95% bootstrap CI lower bound ≤ 0 (not significantly > 0).
- Any bias failure sets `certified: false`.

Cert JSON (`configs/judge.cert.json`, committed):

```json
{"certified": true, "created_at": "2026-10-20T14:02:00Z", "rubric_hash": "sha256:...",
 "judge_model": "claude-...", "judge_temperature": 0.0, "labels_hash": "sha256:...", "n_labels": 100,
 "thresholds": {"kappa_ci_lower_min": 0.4, "kappa_point_min": 0.6, "verbosity_slope_max_pts": 2.0},
 "agreement": {"raw": {"point": 0.88, "ci95": [0.81, 0.94]}, "kappa": {"point": 0.76, "ci95": [0.63, 0.88]},
               "ac1": {"point": 0.77, "ci95": [0.64, 0.89]}, "pabak": {"point": 0.76, "ci95": [0.62, 0.88]},
               "precision": {"point": 0.90, "ci95": [0.82, 0.97]}, "recall": {"point": 0.87, "ci95": [0.77, 0.95]},
               "confusion": [[41, 5], [7, 47]]},
 "bias": {"verbosity_slope_pts": {"point": 0.6, "ci95": [-0.9, 1.8], "pass": true},
          "length_partial_corr": {"point": 0.03, "ci95": [-0.12, 0.18], "pass": true},
          "self_preference_warning": false}}
```

(The numbers above illustrate format only.) The cert is **stale** if `rubric_hash`, `judge_model`, `judge_temperature`, or `labels_hash` differ from the current files. The gate refuses to run on a stale or `certified: false` cert, except for the auto-recertify path in 2.4 step 1.

### 2.4 Gate flow (FR5, FR7; M7)

Workflow `.github/workflows/gate.yml`, trigger `pull_request` on paths `prompts/**`, `agents/**`, `arena_evals/agents.py`, `configs/**`, `datasets/**` (eval config, Interpretation I3). Job runs only when `head.repo.full_name == github.repository`. A separate job in `.github/workflows/fork-notice.yml` on `pull_request_target` (no checkout of PR code, no secrets used) posts: "Eval gate skipped: fork PRs have no access to API secrets. A maintainer can re-open this change from a same-repo branch."

Steps (`python -m arena_evals gate --pr N`):

1. **Cert check.** Recompute hashes; if the cert is valid, continue. If stale because the PR changed `prompts/judge/**` or the judge model, run `certify` automatically against the committed labels; on pass, continue with the fresh cert (uploaded as artifact; comment tells the author to commit it); on fail, outcome `error: judge uncertified`.
2. **Rerun policy.** Read the hidden marker `<!-- arena-eval:v1 {"head_sha":..., "verdict":..., "run_id":...} -->` in the existing gate comment. If the prior verdict for this same head SHA is `block` and this run is not a `workflow_dispatch` with a non-empty `rerun_reason` input, re-post `block` without evaluating. With a reason, the reason is written to the comment and manifest, and both sides run with the response cache bypassed. A new commit (new head SHA) is a fresh evaluation, not a rerun.
3. **Baseline.** Cache key = `sha256(base_sha, agent_model, judge_model, agent_temperature, judge_temperature, seeds, dataset_hash, rubric_hash, scorer_hash)` (hashes from the head checkout). On hit, restore `results.jsonl` + manifest from `actions/cache`. On miss: `git worktree add base <base_sha>`, run phase A `generate` from the base worktree against the head's dataset file, then phase B `score` with the head's scorers and judge. If the PR changed the dataset or rubric, the comment flags "baseline re-run on this PR's dataset/rubric".
4. **Candidate.** Pre-flight: if baseline measured cost x 1.2 exceeds the remaining cap, abort before starting. Otherwise run both phases on head under the `CostMeter` cap (`configs/eval.yaml: cost.max_usd_per_gate`, covering baseline-if-run + candidate + judge). `CostCapExceeded` gives outcome `error: cost cap hit ($X of $Y)`, no score.
5. **Compare.** Per task: mean success over valid repeats on each side; d_i = cand_i - base_i. `stats.paired_bootstrap(d, n_resamples=10_000, seed)` returns delta, two-sided 95% CI, one-sided 97.5% upper bound (the 97.5th percentile of resampled mean deltas), P(delta < 0), MDE. `stats.gate_decision`:
   - **block** iff delta ≤ -ε (ε = 2 pts) AND one-sided 97.5% upper bound < 0 (equivalently P(delta < 0) ≥ 0.975);
   - **warn** iff delta ≤ -ε AND one-sided 97.5% upper bound ≥ 0;
   - **pass** otherwise.
   The two-sided 95% CI is display-only. MDE (80% power) = (z_0.975 + z_0.80) x SD(d)/√n; at n = 300, SD = 0.45 this is ≈ 7.3 pts, matching the PRD. Per-tag deltas with CIs; one-sided bootstrap p per tag with Benjamini-Hochberg at q = 0.05; per-tag results never affect the verdict.
6. **Report.** Create or update the ONE comment carrying the marker (FR7 contents: headline delta + two-sided 95% CI + one-sided 97.5% bound + n, MDE, advisory per-tag table, worst 5 regressions by d_i with trace IDs and artifact link, cert status + self-preference warning, judge-error count, cost, dataset/rubric hashes, baseline-cache hit/miss, rerun reason if any). Set commit status context `arena-eval/gate` on the head SHA via the API (so the same context works for `pull_request` and `workflow_dispatch` runs); `main` requires this context. Upload `$PHOENIX_WORKING_DIR` (Phoenix SQLite trace DB), both `results.jsonl`, both manifests as the `arena-eval-<run_id>` artifact.

Exit codes: pass 0, warn 0 (status `success`, description "warn"), block 1, error 2 (status `failure`).

Trace links: the artifact is not browsable per-trace, so each regression row shows the artifact URL plus the `trace_id`; the comment footer gives the one-line command to view: `PHOENIX_WORKING_DIR=<unzipped> phoenix serve`.

### 2.5 Tracing (FR6)

`tracing.py` is the only module that touches Phoenix: `phoenix.otel.register()` + OpenInference Anthropic instrumentor, working dir from `PHOENIX_WORKING_DIR`. The solver and judge open a parent span per sample with attributes `arena.run_id`, `arena.dataset_hash`, `arena.prompt_hash`, `arena.rubric_hash`, `arena.model`, `arena.task_id`, `arena.repeat`; its trace ID goes into `results.jsonl`.

## 3. Code layout

```
arena-evals/
  PRD.md
  pyproject.toml                 # inspect-ai, anthropic, numpy, scipy, pydantic, pyyaml, arize-phoenix, openinference-instrumentation-anthropic
  arena_evals/
    __main__.py                  # argparse CLI: corpus, datasets, generate, score, run, calibrate, certify, gate, simulate, aa
    config.py                    # load/validate configs/*.yaml, compute hashes
    corpus.py                    # build docs from seed.yaml; load corpus; BM25 index
    datasets.py                  # load/validate task JSONL, draft, spotcheck, near-dup check
    agents.py                    # tools + arena_agent(variant) solver + FINAL parser + CostMeter
    scorers.py                   # deterministic scorers, judge scorer, success rule
    run.py                       # build Inspect Task, phase A generate, phase B score, write results/manifest
    calibration.py               # export, label CLI, agreement report
    bias.py                      # verbosity perturbation + tests, self-preference check
    report.py                    # render PR comment markdown + marker
    ci.py                        # gate orchestration, baseline cache, rerun policy, GitHub API
    tracing.py                   # Phoenix setup + span helpers
    stats/
      __init__.py
      bootstrap.py               # bootstrap_ci, paired_bootstrap, gate_decision, mde
      agreement.py               # kappa, ac1, pabak, precision_recall, confusion, wilson
      simulate.py                # power / coverage Monte Carlo
  corpus/seed.yaml, corpus/docs/*.md
  prompts/{baseline,concise,verbose,regressed,_output_contract}.md, prompts/judge/{system,rubric}.md
  datasets/{gate,calibration,dev}.jsonl, datasets/spotcheck.jsonl, datasets/drafts/
  calibration/{to_label,key,labels}.jsonl
  configs/{eval,gate,models,cert}.yaml, configs/judge.cert.json
  tests/test_calculator.py, test_scorers.py, test_stats.py, test_stats_sim.py, test_e2e_mock.py, test_corpus.py, test_datasets.py
  .github/workflows/{gate,fork-notice,tests}.yml
```

`agents/**` in the PRD trigger list maps to `arena_evals/agents.py` here; both paths are in the trigger (I3).

### 3.1 Module interfaces

| Module | Public interface | Depends on |
|---|---|---|
| `stats.bootstrap` | `bootstrap_ci(x: np.ndarray, n_resamples=10_000, seed: int, alpha=0.05) -> CI`; `paired_bootstrap(d: np.ndarray, n_resamples=10_000, seed: int) -> PairedResult`; `gate_decision(r: PairedResult, eps_pts=2.0, upper_q=0.975) -> Literal["pass","warn","block"]`; `mde(sd: float, n: int, alpha_one_sided=0.025, power=0.80) -> float` | numpy, scipy. No I/O. |
| `stats.agreement` | `cohen_kappa(a, b)`, `gwet_ac1(a, b)`, `pabak(a, b)`, `precision_recall(pred, truth)`, `confusion(pred, truth)`, `agreement_report(pred, truth, n_resamples, seed) -> dict[str, CI]`, `wilson(k, n) -> CI` | numpy. No I/O. |
| `stats.simulate` | `simulate_gate(true_delta_pts, sd, n, trials, n_resamples, eps_pts, seed) -> float` (block rate); `simulate_coverage(true_delta_pts, sd, n, trials, seed) -> float` | stats.bootstrap |
| `corpus` | `build(seed_path, out_dir) -> list[Path]`; `load(docs_dir) -> dict[str, Doc]`; `Index(docs).search(query, k=5) -> list[Hit]`; `current_doc(fact_key) -> str` | pyyaml |
| `datasets` | `load(path) -> list[TaskRecord]` (pydantic, rejects unknown/missing fields); `dataset_hash(path) -> str`; `near_duplicates(gate, others, threshold=0.8) -> list[tuple[str,str,float]]`; `draft(split, corpus, model) -> list[TaskRecord]`; `spotcheck(split, frac=0.2, seed)` | corpus |
| `agents` | `search_docs`, `get_doc`, `calculate` (`@tool`); `safe_eval(expr: str) -> float`; `arena_agent(variant: str, prompts_dir: Path) -> Solver`; `parse_final(text) -> Final \| None`; `CostMeter(cap_usd).charge(usage, model)` | inspect-ai, corpus, tracing |
| `scorers` | `answer_match`, `citation`, `gold_retrieval`, `abstention`, `judge(model, rubric_dir)` (Inspect `@scorer`s); `task_success(task, sample_scores, status) -> bool \| None` (None = judge error) | inspect-ai, pydantic |
| `run` | `generate(split, variant, prompts_dir, dataset, cfg) -> Path` (.eval log); `score(log, dataset, cfg) -> RunOutput(results_path, manifest)` | agents, scorers, datasets, config |
| `calibration` | `export(cfg) -> Path`; `label_cli(path)`; `report(labels, key) -> dict` | run, stats.agreement |
| `bias` | `verbosity(labels, cfg) -> dict`; `self_preference(cfg) -> bool` | run, scorers, stats |
| `report` | `render(paired, per_tag, worst5, cert, cost, meta) -> str` (markdown incl. marker) | none (pure) |
| `ci` | `gate(pr: int, cfg) -> int` (exit code); `certify(cfg) -> int`; `aa(runs: int, cfg) -> int` | everything above, GitHub REST via `GITHUB_TOKEN` |
| `tracing` | `init(working_dir)`; `sample_span(**attrs)` context manager | arize-phoenix, openinference |

## 4. Data flow

```
corpus/seed.yaml --build--> corpus/docs/*.md --> BM25 index --> tools
datasets/<split>.jsonl --load/validate/near-dup--> Inspect Task (epochs=k)
                                                        |
  prompts/<variant>.md --> arena_agent solver ---------+--> phase A: .eval log --+--> Phoenix spans
                                                                                 |    (trace DB)
  prompts/judge/* --> judge scorer --\                                          v
  scorers.py deterministic ----------+--------------------------> phase B: results.jsonl + manifest
                                                                                 |
  calibration/labels.jsonl --> certify (agreement + bias) --> configs/judge.cert.json
                                                                 |               |
            base worktree (phase A) + head scorers ---> baseline results (or cache hit)
                                                                 v               v
                                         ci.gate: cert check -> rerun check -> paired_bootstrap
                                                                 -> gate_decision -> report.render
                                                                 -> PR comment + commit status + artifact
```

## 5. Error handling

| Condition | Handling | Effect on verdict |
|---|---|---|
| Agent exception / API error after Inspect retries | `status = "agent_error"` | Sample FAIL |
| Timeout or message/token/tool-call limit | `status = "timeout"` / `"limit"` | Sample FAIL |
| Missing/invalid `FINAL` line | `status = "format_error"` | Sample FAIL |
| Tool error (bad doc ID, bad expression) | Returned to agent as `"error: ..."` tool output | None directly |
| Malformed judge output | Retry once with validation error; then `judge_error: true` | Sample missing (I2); > 2% in either run -> outcome `error` |
| Cost cap hit | `CostCapExceeded` aborts run; comment states spend and cap | Outcome `error`, exit 2, no score |
| Stale or failed cert | Auto-recertify if judge files changed in PR, else refuse | Outcome `error: judge uncertified`, exit 2 |
| Baseline cache miss | Run baseline from base worktree | None; comment shows "cache miss" |
| Base SHA lacks the harness | Cannot run baseline | Outcome `error`, exit 2 |
| Dataset or rubric changed in PR | Cache key changes -> baseline re-run on head dataset/rubric | Comment flag |
| Blocked gate re-run without reason | Re-post prior block | Block stands (exit 1) |
| Fork PR | Gate job skipped; fork-notice job comments | No status set (required check stays pending) |

## 6. Testing

| Test | What it proves | Milestone |
|---|---|---|
| `test_calculator.py` | `safe_eval` correct on arithmetic, rejects names/calls/attributes/huge exponents | M1 |
| `test_corpus.py` | Rebuild is byte-identical; 40 docs; 5 conflict pairs resolve to the expected current doc | M1 |
| `test_datasets.py` | Schema validation; gate mix within ±1 task of FR1 shares; near-dup check fails on a planted duplicate | M1 |
| `test_scorers.py` | Each deterministic scorer + `task_success` table in 1.5 on hand-built cases, incl. errors -> FAIL, judge error -> None | M2 |
| `test_e2e_mock.py` | Full `generate` + `score` + `gate_decision` on 6 tasks using Inspect `mockllm/model` with scripted outputs (tool calls + FINAL) and a mocked judge; asserts results schema and a known verdict | M2 |
| `test_stats.py` | Bootstrap CI on known data; `gate_decision` boundary cases (delta = -2.0 exactly, upper bound exactly 0); `mde(0.45, 300)` ≈ 7.3 pts | M3 |
| `test_stats_sim.py` (marked `slow`) | With SD from config (0.45 until M4 measures it), n = 300: block rate ≥ 80% at -8 pts; ≤ 5% at 0; 95% CI coverage in [93.5%, 96.5%] over 1000 trials | M3 |
| `python -m arena_evals aa --runs 20` | Real harness, head vs itself, response cache bypassed; pass iff ≤ 1 block in 20 (false-block ≤ 5%) | M7 |
| Demo check | `regressed` vs `baseline` on the gate split shows a true effect ≥ 8 pts before M8 | M8 |

## 7. Decisions and rejected alternatives

| Decision | Rejected | Why |
|---|---|---|
| Inspect AI as runner | Promptfoo; custom runner | Inspect has solvers, tools, epochs, limits, response cache and log re-scoring built in. Promptfoo is prompt A/B oriented and weak for tool-using agents (kept as M9 advisory). Custom means rebuilding all of that. |
| Pairwise judging out of the gate | Pairwise as gate metric | Pointwise gives an absolute per-task score for paired bootstrap; pairwise adds position bias needing its own certification (PRD non-goal; M9 stretch). |
| Phoenix trace DB as CI artifact | Hosted Phoenix | No infra or secrets to run; ephemeral runners. Cost: no clickable per-trace links (artifact + trace ID instead). |
| One-sided 97.5% upper bound | One-sided 95% | 95% one-sided doubles expected false blocks to ~5%, leaving no margin against the ≤ 5% specificity target; 97.5% gives ~2.5%. Costs power (MDE ~7.3 pts). |
| n = 300 gate tasks | 400-500 | Authoring + spot-check cost and < 15 min runtime at k = 3. 300 gives ~88% power at 8 pts; smaller drops are covered by reporting MDE every run. |
| Split generate/score phases | Run base harness end to end | Guarantees baseline and candidate are scored by identical scorer and judge code. |
| Commit status via API | Job result as the check | Same required context for `pull_request` and `workflow_dispatch` reruns. |

## 8. Interpretations of under-specified PRD points

- **I1** Task schema adds `answer_kind` to FR1's fields, needed to route exact/numeric vs judge scoring.
- **I2** FR2 says judge errors are "reported separately" but not how they enter success. Chosen: missing (not FAIL), with a 2% error ceiling that voids the verdict.
- **I3** "Eval config" in the FR7 trigger = `configs/**` and `datasets/**`; `agents/**` also covers `arena_evals/agents.py`.
- **I4** Cache key adds `scorer_hash` to the FR7 key so a scorer change cannot reuse a baseline scored with old code.
- **I5** Verbosity threshold is one-sided (pro-length bias), matching FR4's "CI upper bound" wording; an anti-length bias is reported but does not fail.
- **I6** FR4's "correlation ... is not significantly > 0" = 95% bootstrap CI lower bound of the partial correlation ≤ 0.
- **I7** Cost cap default `cost.max_usd_per_gate: 20.0` is a starting value; M4 measures actual per-run cost (PRD NFR cost is open) and the maintainer resets it. The cap is enforced from the first run regardless.
- **I8** "Rerun" = a new gate run on the same head SHA. Rerun with a reason bypasses the response cache; otherwise a seeded cached replay would return identical numbers.

## 9. Traceability

| PRD item | Spec section |
|---|---|
| A1, A2 | 1.1-1.3 |
| FR1 datasets | 1.4, I1 |
| FR2 scorers, success, errors | 1.5, 2.1, 2.2, I2 |
| FR3 calibration, certification | 2.3 |
| FR4 bias tests, self-preference | 2.3, 2.2, I5, I6 |
| FR5 statistics, gate rule, MDE, per-tag | 2.4 step 5, 3.1 `stats`, 6 |
| FR6 tracing | 2.5, 2.4 step 6 |
| FR7 CI gate, cache, limits, rerun, cost, forks | 2.4, 1.3, 5, I3, I4, I7, I8 |
| NFR reproducibility, isolation | 2.1, 2.2, 2.5 |
| NFR runtime < 15 min, cost | 2.1 (parallel calls via Inspect `max_connections`), I7; measured M4 |
| M1 | 1.1-1.4, 6 |
| M2 | 1.5, 2.1, 6 |
| M3 | 3.1 `stats`, 6 |
| M4 | 2.2, 2.5, I7 |
| M5 | 2.3 |
| M6 | 2.3 |
| M7 | 2.4, 5, 6 (A/A) |
| M8 | 1.3 (demo PR), 6 (demo check) |
| M9 | Out of scope for this design (PRD stretch) |
