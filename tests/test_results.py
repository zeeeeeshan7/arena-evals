from arena_evals import run


def row(task, success, judge_error=False, trace="t"):
    return {"task_id": task, "success": success, "scores": {"judge_error": judge_error}, "trace_id": trace}


def test_task_means_missing_repeat_and_judge_errors():
    rows = [row("a", True), row("a", False), row("a", None, True),   # judge error = missing
            row("b", True),                                           # only 1 of k repeats present
            row("c", None, True), row("c", None, True)]               # no valid repeat
    assert run.task_means(rows) == {"a": 0.5, "b": 1.0, "c": None}
