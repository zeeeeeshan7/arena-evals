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
