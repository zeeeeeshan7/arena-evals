"""Judge bias checks: self-preference warning (M4) and verbosity tests (M6)."""
from __future__ import annotations

from arena_evals.config import Config


def model_family(model_id: str) -> str:
    """'anthropic/claude-sonnet-4-5-20250929' -> 'claude'."""
    return model_id.split("/")[-1].split("-")[0].lower()


def self_preference(cfg: Config) -> bool:
    """True (warn) when agent and judge models share a family prefix."""
    return model_family(cfg.models["agent"]["model"]) == model_family(cfg.models["judge"]["model"])
