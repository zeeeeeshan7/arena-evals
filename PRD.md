# PRD: Arena Agent Eval Harness

Status: draft v0.3 · Owner: zeesh · Date: 2026-09-30

## 1. Problem

Prompt and agent changes ship with no measurement. "Looks better" is the bar. Regressions land silently, and when scores do exist they come from an uncalibrated LLM judge with no error bars. Cannot tell a real 6-pt drop from noise.

## 2. Goal

A trustworthy, automated quality gate for the arena agents. A PR that changes a prompt gets a comment with per-metric deltas and confidence intervals, and merge is blocked when a drop is statistically real.

**Wow moment:** PR comment reads *"This prompt change dropped task success by 7.3 pts (one-sided 97.5% upper bound −2.2, n=300). Merge blocked."*

**pts** = absolute percentage points (e.g. 82% → 75% is −7 pts). All metric deltas are reported in pts, never relative %.

## 3. Non-goals (v1)

- Training or fine-tuning judges.
- A UI beyond the trace viewer and PR comment.
- Multi-repo or multi-tenant support.
- Human-labeling web app (labeling is a CLI or spreadsheet flow).
- Online/production monitoring (offline evals only).
- Promptfoo (optional advisory stretch only, M9).
- Pairwise judging in the gate, and position-bias certification. Stretch only: a pairwise judge + position-bias test on the concise-vs-verbose pair, advisory.
- Plugging in real (non-stand-in) agents.

## 4. Users

- **Author:** changes a prompt or agent and wants fast feedback on the PR.
- **Reviewer:** reads the PR comment to decide whether to merge.
- **Maintainer (me):** curates datasets, labels calibration data, tunes thresholds.

## 5. Assumptions (correct me)

| # | Assumption | If wrong |
|---|---|---|
| A1 | **Arena = stand-in agents built in this repo.** Tool-using Q&A over a synthetic fictional-company corpus (~40 docs, ~5 deliberately conflicting doc pairs distinguished by effective date). Tools: `search_docs(query)`, `get_doc(id)`, `calculate(expr)`. Variants are Inspect AI solvers sharing one tool loop with different system prompts: `baseline` (careful, cites doc IDs, abstains when unsupported), `concise`, `verbose`, and a planted `regressed` (drops cite-and-verify) used for the demo. Variants are compared **paired-pointwise** on the same tasks. | Corpus/tools change; gate logic unaffected. |
| A2 | Inspect solvers are the agent interface. No separate adapter layer. | Add an adapter when real agents arrive. |
| A3 | Python 3.11+, single repo, agents and evals live together. | Gate needs cross-repo checkout. |
| A4 | Judge model is Claude via the Anthropic API. Judge is a different, or at least separately configured, model from the agent under test. | Swap provider in config. |
| A5 | Stack: **Inspect AI** for the eval runner, **Phoenix** for tracing, **GitHub Actions** for the gate, **numpy/scipy** for bootstrap. | — |

## 6. Functional requirements

### FR1 Golden datasets
- JSONL, one task per line: `id`, `input`, `reference` (optional), `gold_doc_ids`, `type`, `tags`, `split`.
- Versioned in the repo. Content hash recorded in every run.
- Splits: `dev` (iterate on judge prompts), `calibration` (hand-labeled), `gate` (used by CI; never used to tune the judge). Separate task sets.
- Target v1: ~300 gate tasks. Gate mix by type: lookup 30%, multi-hop 25%, arithmetic 15%, conflicting-docs 15%, unanswerable 15%.
- Authoring: LLM-drafted from the corpus, then hand spot-checked. Reference-answer label-noise floor measured on the spot-check sample and reported.
- Near-duplicate check across splits (fails the build if a gate task near-duplicates a dev/calibration task).

### FR2 Scorers
- **Deterministic** scorers: exact/numeric answer, gold-doc retrieval, citation validity (citation precision/recall — the property `regressed` drops), abstention on unanswerable, conflict handling (answer must come from the currently effective doc).
- **LLM-as-judge**, pointwise, **only** for free-text correctness/faithfulness: rubric prompt returns a structured verdict (pass/fail plus reasoning). Rubric is versioned.
- **Rubric hash** = hash of the judge prompt + rubric file; stored in every run log and calibration report.
- **Per-task success** (headline metric **task success rate**):
  - lookup / multi-hop / arithmetic: correct answer (exact/numeric, or judge pass for free text) AND cites at least one gold doc.
  - conflicting-docs: answer drawn from the currently effective doc.
  - unanswerable: correct = abstains.
  - Partial credit: none for success (binary per repeat); citation precision/recall are reported as separate secondary metrics.
- Errored or timed-out agent runs count as **FAIL** (not excluded). Judge-output errors (schema-invalid) retry once, then count as error and are reported separately.
- Secondary metrics: per-tag success, citation precision/recall, latency, cost, tokens.

### FR3 Calibration
- CLI exports ~100 examples from the `calibration` split for hand-labeling, stratified by task type AND label, drawn only from judge-scored task types. Labels come back as JSONL.
- Only the judge-scored subset is calibrated; deterministic scorers are not.
- Report: raw agreement, Cohen's κ, Gwet's AC1 and PABAK (prevalence is skewed), precision/recall against human labels, confusion matrix, each with bootstrap CI.
- Judge is **certified** only if κ CI lower bound ≥ 0.4 AND κ point ≥ 0.6 (configurable). The gate refuses to run with an uncertified judge or a stale rubric hash.
- Calibration re-runs automatically when the judge prompt, rubric, or model changes.
- Human-human agreement: 20-item double-label is nice-to-have, not required (solo maintainer).

### FR4 Bias tests
- **Verbosity bias:** run on the real `concise`/`verbose` variants plus length-perturbed same-content answers (pad with fluff, or compress). Default pass thresholds: judge score change ≤ 2 pts per +50% length on same-content perturbation (CI upper bound), and the correlation of judge score with length, controlling for human label, is not significantly > 0.
- **Self-preference:** config warning if the judge model is the same family as the agent.
- Each test outputs an effect size with a CI plus pass/fail against a threshold. Failing a bias test marks the judge uncertified.
- Position bias: stretch only (see Non-goals).

### FR5 Statistics
- **Every reported score carries a 95% CI.** No bare point estimates anywhere: logs, reports, PR comments. Deltas in pts.
- k repeats per task (default 3). Average the repeats per task first, then bootstrap over tasks (10k resamples, seeded). Repeats cut agent/judge noise but not task-sampling variance. Percentile CI by default; consider BCa, and a McNemar/paired-binomial cross-check.
- Baseline vs candidate: **paired bootstrap** on per-task differences. Reports delta, two-sided 95% CI, one-sided 97.5% upper bound, and P(delta < 0). The two-sided 95% CI is for display only; the gate decision uses the one-sided 97.5% upper bound.
- **Gate rule:** **block iff** point-estimate delta ≤ −ε (default ε = 2 pts) **AND** the one-sided 97.5% upper bound of the paired-bootstrap delta is < 0 (equivalently P(delta < 0) ≥ 0.975). **Warn** if the point estimate ≤ −ε but the one-sided 97.5% upper bound is ≥ 0. Both configurable.
- **Power (estimates, to be confirmed by M3 simulation):** at n=300 with paired per-task SD ~0.45, SE = 0.45/√300 ≈ 2.6 pts, so the upper bound (estimate + 1.96 × 2.6) is < 0 when the estimate < ~−5.1 pts. Target: ≥ 80% detection of a true 8-pt drop (MDE ~7.3 pts); a true 7-pt drop is detected only ~76%, and a 6-pt drop ~65%. Pre-build Monte Carlo (normal per-task diffs, ε = 2 pts; percentile paired bootstrap with 2000 resamples, 1000 trials; normal-approx cross-check, 5000 trials): true −8 pts → 88.5% / 87.2% blocked, −7 → 75.6% / 76.1%, −6 → 66.0% / 64.0%, −5 → 49.0% / 49.2%, 0 → 2.2% / 2.3%. SD 0.45 and normality are assumptions, to be re-checked in M3 with the measured SD.
- Every run reports its **minimum detectable effect** (80% power) given observed n and SD.
- The planted demo regression must produce a real effect of ≥ 8 pts, verified empirically before the demo.
- **Per-tag metrics are advisory only, never gate:** n ≈ 30–60 per tag is too small for reliable per-tag decisions. Any per-tag flag shown uses Holm or BH correction.

### FR6 Tracing
- Every agent run and judge call is traced (Phoenix), with the run ID, dataset hash, prompt hash, rubric hash, and model in span attributes.
- Phoenix is ephemeral in Actions: the trace DB is uploaded as a CI artifact, and the PR comment links to the artifact (not hosted Phoenix in v1).
- Failing or flipped tasks in the PR comment link to their trace.

### FR7 CI gate (GitHub Actions)
- Trigger: PRs touching `agents/**`, `prompts/**`, or the eval config.
- Steps: check out the base and head, run the eval on both (or fetch the cached baseline), compute the paired comparison, post or update **one** PR comment, set the status check.
- Comment contains: headline delta with CI, minimum detectable effect, advisory per-tag table, the worst 5 regressions (task ID plus trace link), judge certification status, run cost, and dataset/rubric hashes.
- Exit non-zero on a block. Required-status-check setting on `main` makes it merge-blocking.
- **Baseline cache key:** base SHA + agent/judge model versions + temperature + seeds + dataset hash + rubric hash. If the PR changes the dataset or rubric, the cached baseline is invalid: re-run the baseline on the head's dataset/rubric config and flag this in the comment.
- **Pinned config:** agent and judge model versions, temperature, and seeds are pinned and recorded in every run.
- **Agent limits:** max tool calls, steps, and tokens per task (configurable). Tool errors are returned to the agent as tool output; hitting a limit counts as FAIL.
- **Rerun policy:** a rerun of a blocked gate does not get a fresh chance to pass without a logged reason (reason recorded in the comment/run log).
- Cost guard: hard cap on spend per run; abort with a clear message rather than a partial score.
- Fork PRs: no secrets, so skip with an explanatory comment (v1). The demo uses same-repo branches.

## 7. Non-functional requirements

- **Reproducibility:** model responses are cached keyed by request hash; a seeded run replayed from cache reproduces the same numbers. Runs log the full config.
- **Runtime:** gate completes in <15 min for 300 tasks × 3 repeats (parallel calls with rate-limit backoff); to be confirmed in M4.
- **Cost:** per-PR-run budget = TBD, measured in M4 (first full agent+judge run). The hard cap is enforced regardless.
- **Isolation:** the judge and agent configs are separate. Swapping the tracer touches one module.

## 8. Architecture (sketch)

```
corpus/ + tools ──┐
datasets/*.jsonl ─┼─> Inspect runner ─> raw results (per task x repeat)
agents/ (solvers) ┤         │                    │
scorers/ (det+judge)┘       ▼                    ▼
                      Phoenix traces      stats/ (bootstrap, paired)
                      (CI artifact)              │
        calibration/ (labels, κ) ──> judge cert ──>┤
        bias/ (verbosity) ─────────> judge cert ──>┤
                                                   ▼
                                      report/ ─> PR comment + status check
```

Modules: `corpus`, `datasets`, `agents`, `scorers`, `calibration`, `bias`, `stats`, `report`, `ci`. `stats` is pure functions with no I/O.

## 9. Milestones

1. **M1 Arena:** synthetic corpus (~40 docs, ~5 conflicting pairs), tools, `baseline`/`concise`/`verbose`/`regressed` solvers, task authoring (dev/calibration/gate) with spot-check and near-duplicate check.
2. **M2 Runner skeleton:** dataset schema, Inspect runner, deterministic scorers, one end-to-end run.
3. **M3 Stats + simulation harness:** averaged-repeat paired bootstrap, gate rule, MDE; simulation harness measuring power and CI coverage (not just a unit test).
4. **M4 Judge:** pointwise LLM judge for free-text correctness/faithfulness, structured output, Phoenix tracing. Measure runtime and per-run cost.
5. **M5 Calibration:** labeling export/import, agreement report (κ, AC1, PABAK), certification gate.
6. **M6 Bias:** verbosity test wired into certification.
7. **M7 Gate:** GitHub Action, PR comment, baseline cache, cost cap, rerun policy, required check, trace artifact. A/A flake test (same code vs itself, N runs) verifying false-block rate ≤ 5%.
8. **M8 Demo:** planted `regressed` PR blocked + no-op PR passes; both are real `prompts/**` diffs so the path filter triggers.
9. **M9 (optional stretch):** advisory Promptfoo prompt A/B, non-gating; optional pairwise judge + position-bias test on concise-vs-verbose.

## 10. Success metrics

- **Gate power:** ≥ 80% block rate at a true 8-pt drop (~76% at 7 pts), by simulation (M3) and by empirical reruns of the demo (M8).
- **Gate specificity:** false-block ≤ 5% on no-op PRs and the A/A test (expected ~2.5% under the null with the one-sided 97.5% bound).
- **CI validity:** the bootstrap CI covers the true value ~95% on synthetic data.
- **Judge quality:** κ CI lower bound ≥ 0.4 (point ≥ 0.6) vs human on ~100 labels; verbosity test passes or is documented with effect sizes.
- **Usability:** PR comment posted within 15 min and contains the delta, CI, and worst 5 regressions with trace links.

## 11. Risks

| Risk | Mitigation |
|---|---|
| Noisy judge makes the gate flaky | Repeats averaged per task, paired bootstrap, ε margin, warn-vs-block tiers, A/A test, rerun policy |
| Overfitting the judge to the gate set | Separate dev, calibration and gate splits; near-duplicate check; the gate split is never used for tuning |
| n=300 still misses real drops < ~7.3 pts (below 80% power; a 7-pt drop is caught only ~76%) | Report the minimum detectable effect in every run; grow the dataset |
| Synthetic tasks carry label noise | Hand spot-check; report the label-noise floor |
| API cost and rate limits | Baseline cache, response cache, hard cost cap, backoff |
| Hand-labeling 100 examples is tedious and inconsistent | Stratified sampling, a simple CLI; optional 20-item double-label |
| Same-family judge favors its own outputs | Config warning; recommend a different judge model |

## 12. Open questions

1. Where does the repo live, and is it GitHub-hosted with a required-checks option?
2. Per-run cost budget?
3. Judge model choice, and is the agent under test the same family (self-preference risk)?
