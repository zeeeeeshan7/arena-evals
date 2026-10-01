"""The public report page (GitHub Pages serves /docs)."""
import re
from pathlib import Path

DOCS = Path(__file__).resolve().parent.parent / "docs"
URL = "https://zeeeeeshan7.github.io/arena-evals/"


def test_report_page_is_a_complete_static_document():
    html = (DOCS / "index.html").read_text(encoding="utf-8")
    assert html.lstrip().lower().startswith("<!doctype html>")
    assert "<title>Arena Eval Gate</title>" in html and 'name="viewport"' in html and 'charset="utf-8"' in html
    assert html.count("<script>") == 1 and html.count("</script>") == 1
    assert "claude.ai" not in html                                     # no link back to the private artifact
    assert re.search(r"body\s*\{[^}]*margin:\s*0", html)             # the artifact skeleton zeroed it; a bare page must


def test_pages_does_not_run_jekyll_and_readme_links_the_page():
    assert (DOCS / ".nojekyll").exists()
    assert URL in (DOCS.parent / "README.md").read_text(encoding="utf-8")
