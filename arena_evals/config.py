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


# Every file that decides a sample's success when phase B re-scores a baseline: scorers, the sample-status rule
# (run.py), the FINAL parser (agents.py) and the corpus that defines valid citation IDs.
SCORING_FILES = ("arena_evals/scorers.py", "arena_evals/run.py", "arena_evals/agents.py", "arena_evals/corpus.py")


def scorer_hash(root: Path = ROOT) -> str:
    root = Path(root)
    corpus = sorted(p for p in (root / "corpus").rglob("*") if p.is_file()) if (root / "corpus").is_dir() else []
    return sha256_files(*(root / f for f in SCORING_FILES), *corpus)


def temperature_sent(model_id: str) -> bool:
    """False when Inspect drops `temperature` for this model (Anthropic names it treats as adaptive-thinking-only,
    e.g. deepseek-v4-pro). Asks Inspect rather than guessing from the name; no network call is made."""
    if not model_id.startswith("anthropic/"):
        return True
    import os

    from inspect_ai.model import get_model
    key = os.environ.get("ANTHROPIC_API_KEY")
    os.environ["ANTHROPIC_API_KEY"] = key or "unused-no-call-made"
    try:
        api = get_model(model_id).api
        return not (hasattr(api, "is_claude_4_7_or_later") and api.is_claude_4_7_or_later())
    finally:
        if key is None:
            del os.environ["ANTHROPIC_API_KEY"]


def model_args(model_id: str, temperature: float | None) -> dict:
    """get_model()/eval() model_args that make `temperature` reach the API. Inspect drops it for unrecognised
    Anthropic-provider names, but forwards `extra_body` verbatim, so route it there. Empty when Inspect sends it."""
    if temperature is None or temperature_sent(model_id):
        return {}
    return {"extra_body": {"temperature": temperature}}
