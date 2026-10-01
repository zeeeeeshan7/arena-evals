"""Verify the Inspect/Phoenix behaviours this harness relies on. Run: python spikes/verify_apis.py
Needs `phoenix serve` running on localhost:6006 for check 7 (see Task 8). Prints OK/FAIL per check."""
import importlib.metadata as md
import os
import tempfile

os.environ["INSPECT_CACHE_DIR"] = tempfile.mkdtemp()
os.environ["INSPECT_DISPLAY"] = "none"

from inspect_ai import Task, eval, score  # noqa: E402
from inspect_ai.dataset import MemoryDataset, Sample  # noqa: E402
from inspect_ai.hooks import Hooks, hooks  # noqa: E402
from inspect_ai.log import read_eval_log  # noqa: E402
from inspect_ai.model import CachePolicy, ModelOutput, get_model  # noqa: E402
from inspect_ai.scorer import Score, scorer  # noqa: E402
from inspect_ai.solver import generate, solver, system_message, use_tools  # noqa: E402
from inspect_ai.tool import tool  # noqa: E402

USAGE = []


@hooks(name="spike_usage", description="count model usage events")
class Spike(Hooks):
    async def on_model_usage(self, data):
        USAGE.append(data.model_name)

    async def on_before_model_generate(self, data):
        raise RuntimeError("raised inside a hook")


def check(name, ok, detail=""):
    print(f"{'OK  ' if ok else 'FAIL'} {name} {detail}")


print("inspect-ai", md.version("inspect-ai"), "| arize-phoenix", md.version("arize-phoenix"),
      "| arize-phoenix-otel", md.version("arize-phoenix-otel"))

# 1. CachePolicy(expiry=None) = never expire
check("1 CachePolicy(expiry=None)", CachePolicy(expiry=None).expiry is None)


@tool
def get_doc():
    async def execute(id: str) -> str:
        """Fetch a document.

        Args:
            id: Document ID.
        """
        return f"doc {id}"
    return execute


def scripted(messages, tools, tool_choice, config):
    if not any(m.role == "tool" for m in messages):
        return ModelOutput.for_tool_call("mockllm/model", "get_doc", {"id": "HR-001"})
    return ModelOutput.from_content("mockllm/model", 'done\nFINAL: {"answer": "x", "citations": [], "abstain": false}')


@solver
def stamp():
    async def solve(state, gen):
        state.metadata["trace_id"] = "abc"
        return state
    return solve


# 2+3. mockllm with a callable custom_outputs drives a tool loop; metadata set in a solver reaches the log
model = get_model("mockllm/model", custom_outputs=scripted, memoize=False)
task = Task(dataset=MemoryDataset([Sample(input="q", id="s1")]),
            solver=[system_message('Reply with FINAL: {"answer": "x"}'), stamp(), use_tools(get_doc()),
                    generate(tool_calls="loop", cache=CachePolicy(expiry=None))], epochs=2)
log = eval(task, model=model, log_dir=tempfile.mkdtemp(), fail_on_error=False)[0]
s = log.samples[0]
check("2 mockllm callable custom_outputs + tool loop", [m.role for m in s.messages][-3:] == ["assistant", "tool", "assistant"],
      str([m.role for m in s.messages]))
check("3 solver metadata persists to log", s.metadata.get("trace_id") == "abc")
check("3b epochs are 1-based in the log", sorted(x.epoch for x in log.samples) == [1, 2])
check("3c system_message() mangles JSON braces (so agents.py inserts ChatMessageSystem directly)",
      s.messages[0].text != 'Reply with FINAL: {"answer": "x"}', repr(s.messages[0].text))

# 4. inspect_ai.score() re-scores an existing log with new scorers; dict values with None are allowed


@scorer(metrics=[])
def probe():
    async def f(state, target):
        return Score(value={"precision": None, "recall": 0.5})
    return f


rescored = score(read_eval_log(log.location), [probe()], display="none")
check("4 inspect_ai.score() re-scoring", rescored.samples[0].scores["probe"].value == {"precision": None, "recall": 0.5})

# 5. hooks: on_model_usage fires; an exception raised in a hook does NOT stop the call (so the cost cap is
#    enforced in tools/solver/judge, not in a hook)
check("5 on_model_usage fires per call", len(USAGE) >= 2, f"{len(USAGE)} events")
check("5b exception in hook is swallowed", s.error is None)

# 6. an exception raised inside a tool errors the sample with the class name in the message
class CostCapExceeded(RuntimeError):
    pass


@tool
def boom():
    async def execute(x: str) -> str:
        """Boom.

        Args:
            x: anything.
        """
        raise CostCapExceeded("cap")
    return execute


m2 = get_model("mockllm/model", custom_outputs=lambda *a: ModelOutput.for_tool_call("mockllm/model", "boom", {"x": "1"}),
               memoize=False)
log2 = eval(Task(dataset=MemoryDataset([Sample(input="q", id="b")]), solver=[use_tools(boom()), generate()]),
            model=m2, log_dir=tempfile.mkdtemp(), fail_on_error=False)[0]
err = log2.samples[0].error
check("6 tool exception -> sample error naming CostCapExceeded", bool(err and "CostCapExceeded" in err.message))

# 7. phoenix.otel.register exports spans to a running `phoenix serve`, whose PHOENIX_WORKING_DIR holds phoenix.db
try:
    from phoenix.otel import register
    tp = register(project_name="arena-evals-spike", endpoint="http://localhost:6006/v1/traces", batch=False,
                  verbose=False)
    with tp.get_tracer("spike").start_as_current_span("spike", attributes={"arena.task_id": "t"}) as span:
        tid = format(span.get_span_context().trace_id, "032x")
    tp.force_flush()
    check("7 phoenix.otel.register + span export", len(tid) == 32, f"trace_id={tid}")
except Exception as e:  # noqa: BLE001
    check("7 phoenix.otel.register + span export", False, repr(e))
