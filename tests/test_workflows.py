from pathlib import Path

import yaml

WF = Path(__file__).resolve().parent.parent / ".github" / "workflows"


def load(name):
    return yaml.safe_load((WF / name).read_text(encoding="utf-8"))


def test_gate_triggers_on_prompt_agent_config_and_dataset_paths():
    wf = load("gate.yml")
    on = wf[True]  # PyYAML parses the bare key `on` as boolean True
    assert on["pull_request"]["paths"] == ["prompts/**", "agents/**", "arena_evals/agents.py", "configs/**",
                                           "datasets/**"]
    assert set(on["workflow_dispatch"]["inputs"]) == {"pr", "rerun_reason"}


def test_gate_job_skips_fork_prs():
    job = load("gate.yml")["jobs"]["gate"]
    assert "github.event.pull_request.head.repo.full_name == github.repository" in job["if"]


def test_fork_notice_never_checks_out_pr_code():
    wf = load("fork-notice.yml")
    assert "pull_request_target" in wf[True]
    job = wf["jobs"]["notice"]
    assert "!= github.repository" in job["if"]
    assert not any("checkout" in str(s.get("uses", "")) for s in job["steps"])
    assert "ANTHROPIC_API_KEY" not in (WF / "fork-notice.yml").read_text()
    assert "Eval gate skipped: fork PRs have no access to API secrets." in job["steps"][0]["run"]


def test_gate_uploads_artifact_and_saves_baseline_cache_even_on_block():
    steps = load("gate.yml")["jobs"]["gate"]["steps"]
    upload = next(s for s in steps if str(s.get("uses", "")).startswith("actions/upload-artifact"))
    assert upload["if"] == "always()" and upload["with"]["path"] == ".arena-out"
    assert upload["with"]["name"].startswith("arena-eval-")
