"""Inspect treats any unrecognised Anthropic-provider name (deepseek-flash, deepseek-v4-pro) as a future Claude and
silently drops `temperature`. We send it through `extra_body` instead and record which route a run used."""
import pytest
from inspect_ai.model import get_model

from arena_evals import ci, config


@pytest.mark.parametrize("model,native", [("anthropic/deepseek-v4-pro", False), ("anthropic/deepseek-flash", False),
                                          ("anthropic/claude-haiku-4-5-20251001", True), ("mockllm/model", True)])
def test_temperature_sent_natively_matches_inspect(model, native):
    assert config.temperature_sent(model) is native


def test_model_args_route_temperature_through_extra_body_only_when_inspect_drops_it():
    assert config.model_args("anthropic/deepseek-v4-pro", 0.0) == {"extra_body": {"temperature": 0.0}}
    assert config.model_args("anthropic/deepseek-flash", 0.7) == {"extra_body": {"temperature": 0.7}}
    assert config.model_args("anthropic/claude-haiku-4-5-20251001", 0.7) == {}
    assert config.model_args("mockllm/model", 0.7) == {}
    assert config.model_args("anthropic/deepseek-flash", None) == {}


def test_extra_body_reaches_the_anthropic_request_arguments(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "unused")
    m = get_model("anthropic/deepseek-v4-pro", **config.model_args("anthropic/deepseek-v4-pro", 0.0))
    assert m.api.extra_body == {"temperature": 0.0}


def test_cert_inputs_record_the_temperature_route():
    cfg = config.load()
    assert ci.current_cert_inputs(cfg)["judge_temperature_via_extra_body"] is \
        (not config.temperature_sent(cfg.models["judge"]["model"]))
