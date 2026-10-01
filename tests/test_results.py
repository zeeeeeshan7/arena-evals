from arena_evals import run


def row(task, success, judge_error=False, trace="t"):
    return {"task_id": task, "success": success, "scores": {"judge_error": judge_error}, "trace_id": trace}


def test_task_means_missing_repeat_and_judge_errors():
    rows = [row("a", True), row("a", False), row("a", None, True),   # judge error = missing
            row("b", True),                                           # only 1 of k repeats present
            row("c", None, True), row("c", None, True)]               # no valid repeat
    assert run.task_means(rows) == {"a": 0.5, "b": 1.0, "c": None}


def test_paired_deltas_drops_tasks_without_valid_repeat_on_either_side():
    base = [row("a", True), row("b", True), row("c", True)]
    cand = [row("a", False), row("b", None, True), row("d", True)]
    ids, d, dropped = run.paired_deltas(base, cand)
    assert ids == ["a"] and d.tolist() == [-1.0] and dropped == ["b", "c", "d"]
