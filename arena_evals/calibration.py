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
