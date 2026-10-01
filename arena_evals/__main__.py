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
    ca = cal.add_parser("augment")
    ca.add_argument("--max-usd", type=float)
    ca.add_argument("--per-kind", type=int, default=10)
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
            drafts = cfg.root / "datasets" / "drafts"
            partial = drafts / f"{args.split}.partial.jsonl"
            if partial.exists():
                print(f"resuming from {partial} (delete it to start over)", flush=True)
            recs = datasets.draft(args.split, corpus.load(), args.model or cfg.models["drafter"]["model"],
                                  on_progress=lambda m: print(m, flush=True), checkpoint=partial)
            print(f"wrote {len(recs)} drafts to {datasets.write_jsonl(drafts / f'{args.split}.jsonl', recs)}")
            partial.unlink(missing_ok=True)
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
        if args.action == "augment":
            _meter(cfg, args.max_usd)
            print(f"wrote {calibration.augment(cfg, per_kind=args.per_kind)}")
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
