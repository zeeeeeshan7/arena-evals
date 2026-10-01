from arena_evals import agents
from arena_evals.agents import load_prompt, parse_final


def test_parse_final():
    good = 'Answer text\nFINAL: {"answer": "16 weeks", "citations": ["HR-009"], "abstain": false}'
    f = parse_final(good)
    assert (f.answer, f.citations, f.abstain) == ("16 weeks", ["HR-009"], False)
    assert parse_final("no final line") is None
    assert parse_final('FINAL: {"answer": 16, "citations": [], "abstain": false}') is None  # answer must be str
    assert parse_final('FINAL: {"answer": "x", "citations": []}') is None
    assert parse_final("FINAL: not json") is None
    assert parse_final(None) is None


def test_prompts_share_output_contract_and_regressed_drops_cite_and_verify():
    for v in ("baseline", "concise", "verbose", "regressed"):
        assert load_prompt(v).rstrip().endswith('"abstain": false}')
    assert "verify" in load_prompt("baseline").lower()
    assert "verify" not in load_prompt("regressed").lower()
    assert "cite" not in open(agents.PROMPTS_DIR / "regressed.md", encoding="utf-8").read().lower()
