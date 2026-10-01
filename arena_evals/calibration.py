"""Judge calibration: stratified export for blind hand-labeling, a terminal labeling CLI, agreement report."""
from __future__ import annotations

import asyncio
import json
import os
import random
import re
from pathlib import Path
from typing import Callable

from arena_evals import bias, corpus, datasets, run
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


# ---------------------------------------------------------------- negatives (known-bad answers by construction)
KINDS = ("wrong_number", "dropped_fact", "unsupported_claim", "wrong_citation")
_NUM = re.compile(r"\d[\d,]*(?:\.\d+)?")


def _fact_sentences(doc: corpus.Doc) -> list[str]:
    skip = ("#", "Halcyon Robotics", "This document", "Related documents")
    return [l.strip() for l in doc.body.splitlines() if l.strip() and not l.strip().startswith(skip)]


def corrupt(kind: str, e: dict, docs: dict[str, corpus.Doc], rng: random.Random) -> dict | None:
    """A known-bad copy of a real exported example: it fails the judge rubric by construction. None if the example
    has nothing to corrupt that way. wrong_number changes a required number; dropped_fact drops the second required
    fact; unsupported_claim adds a sentence no cited document supports; wrong_citation cites a document that does not
    hold the facts."""
    ans, cites = e["answer"], list(e["citations"])
    if kind == "wrong_number":
        ref_nums = {n.replace(",", "") for n in _NUM.findall(e["reference"] or "")}
        hit = next((m for m in _NUM.finditer(ans) if m.group().replace(",", "") in ref_nums), None)
        if hit is None:
            return None
        ans = ans[:hit.start()] + str(int(float(hit.group().replace(",", ""))) + 7) + ans[hit.end():]
    elif kind == "dropped_fact":
        if ";" not in (e["reference"] or "") or not re.search(r";|, and ", ans):
            return None
        ans = re.split(r";|, and ", ans, maxsplit=1)[0].strip()
    elif kind == "unsupported_claim":
        cited = " ".join(docs[c].text for c in cites if c in docs)
        extra = [s for i, d in docs.items() if i not in cites for s in _fact_sentences(d) if s not in cited]
        if not extra:
            return None
        ans = f"{ans} {rng.choice(extra)}"
    elif kind == "wrong_citation":
        sig = set(bias.facts_signature(ans))
        ok = [i for i, d in docs.items() if i not in cites and not sig & set(bias.facts_signature(d.body))]
        if not ok:
            return None
        cites = [rng.choice(sorted(ok))]
    else:
        raise ValueError(f"unknown corruption kind {kind!r}")
    return e | {"example_id": f"{e['example_id']}~{kind}", "answer": ans, "citations": cites, "synthetic": kind}


def augment(cfg: Config, *, judge_model=None, per_kind: int = 10, seed: int | None = None) -> Path:
    """Append known-bad examples to to_label.jsonl, judge them into key.jsonl, and record their labels
    (labeler "construction", label fail) in labels.jsonl. Idempotent: per_kind is the total per kind."""
    from inspect_ai.model import GenerateConfig, get_model

    from arena_evals import agents
    from arena_evals.config import model_args
    from arena_evals.scorers import judge_answer

    cal = cfg.root / "calibration"
    to_label_path = cal / "to_label.jsonl"
    rows = read_jsonl(to_label_path)
    have = {r["example_id"] for r in rows}
    docs = corpus.load(cfg.root / "corpus" / "docs")
    rng = random.Random(cfg.eval["seed"] if seed is None else seed)
    real = [r for r in rows if "synthetic" not in r]
    new: list[dict] = []
    for kind in KINDS:
        order = real[:]
        rng.shuffle(order)
        made = sum(r.get("synthetic") == kind for r in rows)    # per_kind is the total wanted, so a re-run adds none
        for e in order:
            if made >= per_kind:
                break
            n = corrupt(kind, e, docs, rng)
            if n and n["example_id"] not in have:
                new.append(n)
                have.add(n["example_id"])
                made += 1
    if not new:
        return to_label_path
    jc = cfg.models["judge"]
    jm = judge_model or get_model(jc["model"], config=GenerateConfig(temperature=jc["temperature"], seed=jc["seed"]),
                                  **model_args(jc["model"], jc["temperature"]))

    async def judge_all():
        sem = asyncio.Semaphore(cfg.eval["max_connections"])

        async def one(e):
            async with sem:
                v, _ = await judge_answer(jm, cfg.rubric_dir, e["input"], e["reference"], e["answer"],
                                          {c: docs[c].text for c in e["citations"] if c in docs}, cfg.eval["cache"])
            return e, v
        return await asyncio.gather(*(one(e) for e in new))

    agents.METER.check()
    judged = asyncio.run(judge_all())
    key = read_jsonl(cal / "key.jsonl") + [
        {"example_id": e["example_id"], "judge_pass": None if v is None else bool(v.correct and v.faithful),
         **({} if v is None else {"correct": v.correct, "faithful": v.faithful, "reason": v.reason})}
        for e, v in judged]
    datasets.write_jsonl(cal / "key.jsonl", key)
    datasets.write_jsonl(cal / "labels.jsonl", read_jsonl(cal / "labels.jsonl") + [
        {"example_id": e["example_id"], "task_id": e["task_id"], "variant": e["variant"], "human_label": "fail",
         "labeler": "construction", "note": e["synthetic"]} for e in new])
    return datasets.write_jsonl(to_label_path, rows + new)


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
