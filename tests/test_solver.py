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
    # The gated prompt (baseline) is judged by the eval gate, not by pinning its wording here.
    assert "verify" not in load_prompt("regressed").lower()
    assert "cite" not in open(agents.PROMPTS_DIR / "regressed.md", encoding="utf-8").read().lower()


def test_regressed_prompt_drops_the_supersession_and_abstention_rules():
    """The planted regression must change behaviour on conflicting-document and unanswerable tasks (30% of the
    gate mix); dropping only cite-and-verify changed nothing measurable on answerable tasks."""
    text = open(agents.PROMPTS_DIR / "regressed.md", encoding="utf-8").read().lower()
    for rule in ("effective_date", "supersede", "abstain", "never guess", "outside knowledge"):
        assert rule not in text, rule
    for tool in ("search_docs", "get_doc", "calculate"):
        assert tool in text                                   # same tools: only the discipline is gone
    assert "always give" in text                              # the plausible "be more helpful" edit
    assert "at most once" in text and "do not open" in text   # and the plausible "cut latency" edit


def test_careful_variants_stop_searching_and_abstain_when_searches_find_nothing():
    """Unanswerable questions made the baseline search until it hit a limit and score as a failure.
    The gated prompt (baseline) is deliberately not pinned: the eval gate decides whether an edit to it is a regression."""
    for v in ("concise", "verbose"):
        text = open(agents.PROMPTS_DIR / f"{v}.md", encoding="utf-8").read().lower()
        assert "stop searching" in text and "abstain" in text, v
