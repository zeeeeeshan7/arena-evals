import json
from pathlib import Path

import pytest
from inspect_ai.model import ModelOutput, get_model

from arena_evals import config, datasets, run
from arena_evals.datasets import TaskRecord

M = "mockllm/model"
FINAL = 'FINAL: {"answer": "%s", "citations": %s, "abstain": %s}'

TASKS = [
    TaskRecord(id="e2e-0001", input="How many days of PTO do full-time employees accrue per year?", reference="20 days",
               gold_doc_ids=["HR-001"], type="lookup", answer_kind="exact", tags=["lookup", "hr"], split="dev"),
    TaskRecord(id="e2e-0002", input="Who is the CTO of Halcyon Robotics?", reference="Dev Okafor",
               gold_doc_ids=["ORG-001"], type="multi_hop", answer_kind="exact", tags=["multi_hop", "org"], split="dev"),
    TaskRecord(id="e2e-0003", input="What is the monthly price of the Pro tier for 25 seats?", reference="4475 USD",
               gold_doc_ids=["PRC-006"], type="arithmetic", answer_kind="numeric", tags=["arithmetic", "pricing"],
               split="dev"),
    TaskRecord(id="e2e-0004", input="How many weeks of parental leave do employees get?", reference="16 weeks",
               gold_doc_ids=["HR-009"], type="conflicting", answer_kind="exact", tags=["conflicting", "hr"], split="dev"),
    TaskRecord(id="e2e-0005", input="What is the dental insurance deductible?", reference=None, gold_doc_ids=[],
               type="unanswerable", answer_kind="abstain", tags=["unanswerable"], split="dev"),
    TaskRecord(id="e2e-0006", input="What is the hotel cap per night?", reference="220 USD",
               gold_doc_ids=["HR-004"], type="lookup", answer_kind="exact", tags=["lookup", "hr"], split="dev"),
]

GOOD = {  # question -> (doc to fetch, FINAL line)
    TASKS[0].input: ("HR-001", FINAL % ("20 days", '["HR-001"]', "false")),
    TASKS[1].input: ("ORG-001", FINAL % ("Dev Okafor", '["ORG-001"]', "false")),
    TASKS[2].input: ("PRC-006", FINAL % ("4,475 USD", '["PRC-006"]', "false")),
    TASKS[3].input: ("HR-009", FINAL % ("16 weeks", '["HR-009"]', "false")),
    TASKS[4].input: ("HR-001", FINAL % ("", "[]", "true")),
    TASKS[5].input: ("HR-004", "The cap is 220 USD."),  # no FINAL line -> format_error
}


def scripted_agent(script: dict):
    """mockllm callable: first turn fetches a doc via get_doc, second turn answers with the scripted text."""
    def reply(messages, tools, tool_choice, cfg):
        question = next(m.text for m in messages if m.role == "user")
        doc, final = script[question]
        if not any(m.role == "tool" for m in messages):
            return ModelOutput.for_tool_call(M, "get_doc", {"id": doc})
        return ModelOutput.from_content(M, f"Answer.\n{final}")
    return get_model(M, custom_outputs=reply, memoize=False)


@pytest.fixture
def cfg():
    return config.load()


@pytest.fixture
def dataset(tmp_path: Path) -> Path:
    return datasets.write_jsonl(tmp_path / "e2e.jsonl", TASKS)


def run_variant(cfg, dataset, tmp_path, script, name):
    log = run.generate("dev", "baseline", cfg.prompts_dir, dataset, cfg, model=scripted_agent(script),
                       log_dir=tmp_path / f"logs-{name}", epochs=2, cache=False)
    return run.score(log, dataset, cfg, out_dir=tmp_path / name)


def test_generate_and_score_write_results_and_manifest(cfg, dataset, tmp_path):
    out = run_variant(cfg, dataset, tmp_path, GOOD, "base")
    rows = run.read_results(out.results_path)
    assert len(rows) == 12  # 6 tasks x 2 epochs
    required = {"run_id", "variant", "task_id", "repeat", "status", "answer", "citations", "abstain", "scores",
                "success", "latency_s", "tokens_in", "tokens_out", "cost_usd", "trace_id"}
    assert all(set(r) == required for r in rows)
    by = {(r["task_id"], r["repeat"]): r for r in rows}
    assert sorted({r["repeat"] for r in rows}) == [0, 1]
    for tid in ("e2e-0001", "e2e-0002", "e2e-0003", "e2e-0004", "e2e-0005"):
        assert by[(tid, 0)]["status"] == "ok" and by[(tid, 0)]["success"] is True, by[(tid, 0)]
    assert by[("e2e-0006", 0)]["status"] == "format_error" and by[("e2e-0006", 0)]["success"] is False
    assert by[("e2e-0001", 0)]["scores"]["gold_retrieval"] is True
    assert by[("e2e-0003", 0)]["scores"]["citation_precision"] == 1.0
    m = json.loads((tmp_path / "base" / "manifest.json").read_text())
    for key in ("run_id", "git_sha", "variant", "split", "dataset_hash", "prompt_hash", "rubric_hash", "scorer_hash",
                "agent_model", "judge_model", "agent_temperature", "judge_temperature", "k", "seeds",
                "inspect_version", "limits", "cost_usd", "label_noise_floor", "config"):
        assert key in m, key
    assert m["k"] == 2 and m["dataset_hash"] == datasets.dataset_hash(dataset)


# ---------------------------------------------------------------- M3: paired comparison on real harness output
BAD = dict(GOOD) | {
    TASKS[0].input: ("HR-001", FINAL % ("25 days", '["HR-001"]', "false")),
    TASKS[1].input: ("ORG-001", FINAL % ("Mara Lindqvist", '["ORG-001"]', "false")),
    TASKS[2].input: ("PRC-001", FINAL % ("4975 USD", '["PRC-001"]', "false")),
    TASKS[3].input: ("HR-002", FINAL % ("12 weeks", '["HR-002"]', "false")),
    TASKS[4].input: ("HR-001", FINAL % ("500 USD", '["HR-999"]', "false")),  # hallucinated citation, no abstain
}


def test_e2e_known_verdicts(cfg, dataset, tmp_path):
    from arena_evals.stats.bootstrap import gate_decision, paired_bootstrap

    base = run.read_results(run_variant(cfg, dataset, tmp_path, GOOD, "base").results_path)
    bad = run.read_results(run_variant(cfg, dataset, tmp_path, BAD, "bad").results_path)
    same = run.read_results(run_variant(cfg, dataset, tmp_path, GOOD, "same").results_path)
    hall = next(r for r in bad if r["task_id"] == "e2e-0005")
    assert hall["success"] is False and hall["scores"]["citation_precision"] == 0.0
    assert gate_decision(paired_bootstrap(run.paired_deltas(base, bad)[1], seed=cfg.eval["seed"])) == "block"
    assert gate_decision(paired_bootstrap(run.paired_deltas(base, same)[1], seed=cfg.eval["seed"])) == "pass"
