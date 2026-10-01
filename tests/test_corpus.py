from pathlib import Path

from arena_evals import corpus

CONFLICTS = {  # fact key -> (superseded doc, currently effective doc)
    "parental_leave_weeks": ("HR-002", "HR-009"),
    "remote_stipend": ("HR-003", "HR-010"),
    "c2_battery_hours": ("PRD-002", "PRD-008"),
    "pro_price_per_seat": ("PRC-001", "PRC-006"),
    "head_of_support": ("ORG-002", "ORG-004"),
}


def test_rebuild_is_byte_identical(tmp_path: Path):
    corpus.build(corpus.SEED_PATH, tmp_path)
    committed = {p.name: p.read_bytes() for p in corpus.DOCS_DIR.glob("*.md")}
    rebuilt = {p.name: p.read_bytes() for p in tmp_path.glob("*.md")}
    assert rebuilt == committed


def test_forty_docs_with_kind_counts():
    docs = corpus.load()
    assert len(docs) == 40
    kinds = [d.kind for d in docs.values()]
    assert (kinds.count("hr"), kinds.count("product"), kinds.count("pricing"),
            kinds.count("incident"), kinds.count("org")) == (10, 9, 7, 10, 4)


def test_five_conflict_pairs_resolve_to_current_doc():
    docs = corpus.load()
    assert sum(1 for d in docs.values() if d.supersedes) == 5
    for key, (old, new) in CONFLICTS.items():
        assert docs[new].supersedes == old
        assert corpus.current_doc(key) == new


def test_search_format_and_empty():
    idx = corpus.Index(corpus.load())
    hits = idx.search("parental leave weeks")
    assert {h.id for h in hits[:2]} == {"HR-002", "HR-009"}
    assert all(len(h.snippet) <= 200 for h in hits)
    assert "weeks of fully paid parental leave" in hits[0].snippet
    assert idx.search("zzzz qqqq") == []


def test_search_ties_broken_by_doc_id():
    body = "The Dock charges robots."
    docs = {i: corpus.Doc(i, "t", "product", "2024-01-01", None, body, body) for i in ("PRD-777", "PRD-111")}
    assert [h.id for h in corpus.Index(docs).search("dock charges")] == ["PRD-111", "PRD-777"]
