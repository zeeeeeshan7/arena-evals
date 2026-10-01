"""Arena stand-in agent: tools, safe calculator, FINAL-line parser, cost meter, solver factory."""
from __future__ import annotations

import ast
import functools
import math
import operator
from pathlib import Path

from inspect_ai.hooks import Hooks, ModelUsageData, hooks
from inspect_ai.model import CachePolicy, ChatMessageSystem
from inspect_ai.solver import Generate, Solver, TaskState, generate, solver, use_tools
from inspect_ai.tool import Tool, tool
from pydantic import BaseModel, ConfigDict, ValidationError

from arena_evals import corpus
from arena_evals.config import ROOT

PROMPTS_DIR = ROOT / "prompts"

# ---------------------------------------------------------------- safe calculator
_BINOPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv,
           ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod, ast.Pow: operator.pow}
MAX_ABS_RESULT = 1e15
MAX_ABS_EXPONENT = 100
MAX_EXPR_CHARS = 500


def safe_eval(expr: str) -> float:
    """Evaluate +, -, *, /, //, %, **, unary minus, parentheses over numeric literals. Raises ValueError."""
    if len(expr) > MAX_EXPR_CHARS:
        raise ValueError(f"expression longer than {MAX_EXPR_CHARS} characters")
    try:
        tree = ast.parse(expr.strip(), mode="eval")
    except SyntaxError as e:
        raise ValueError(f"invalid syntax: {e.msg}") from None

    def ev(node: ast.AST) -> float:
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            v = node.value
        elif isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            v = -ev(node.operand)
        elif isinstance(node, ast.BinOp) and type(node.op) in _BINOPS:
            left, right = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow) and abs(right) > MAX_ABS_EXPONENT:
                raise ValueError(f"exponent larger than {MAX_ABS_EXPONENT}")
            try:
                v = _BINOPS[type(node.op)](left, right)
            except ZeroDivisionError:
                raise ValueError("division by zero") from None
            except OverflowError:
                raise ValueError("result too large") from None
        else:
            raise ValueError(f"unsupported element: {type(node).__name__}")
        if isinstance(v, complex) or not math.isfinite(v) or abs(v) > MAX_ABS_RESULT:
            raise ValueError("result too large or not a real number")
        return v

    return float(ev(tree.body))


def format_number(x: float) -> str:
    return str(int(x)) if x.is_integer() else f"{x:.10g}"


# ---------------------------------------------------------------- cost meter
class CostCapExceeded(RuntimeError):
    pass


def usage_cost(usage, model: str, prices: dict) -> float:
    """USD for one ModelUsage. prices: {model: {input, output}} per million tokens; unlisted models pay 'default'."""
    p = prices.get(model) or prices.get("default") or {"input": 0.0, "output": 0.0}
    return (usage.input_tokens * p["input"] + usage.output_tokens * p["output"]) / 1_000_000


class CostMeter:
    """Process-wide spend tracker, charged by CostHook for every agent and judge model call."""

    def __init__(self, cap_usd: float, prices: dict | None = None):
        self.cap_usd = cap_usd
        self.prices = prices or {}
        self.spent = 0.0

    def charge(self, usage, model: str) -> float:
        cost = usage_cost(usage, model, self.prices)
        self.spent += cost
        return cost

    def check(self) -> None:
        if self.spent >= self.cap_usd:
            raise CostCapExceeded(f"cost cap hit (${self.spent:.2f} of ${self.cap_usd:.2f})")


METER = CostMeter(float("inf"))


def set_meter(meter: CostMeter) -> CostMeter:
    global METER
    METER = meter
    return meter


@hooks(name="arena_cost_meter", description="Charge every model call (agent and judge) to the CostMeter")
class CostHook(Hooks):
    async def on_model_usage(self, data: ModelUsageData) -> None:
        METER.charge(data.usage, data.model_name)


# ---------------------------------------------------------------- tools
@functools.cache
def _index() -> corpus.Index:
    return corpus.Index(corpus.load())


@tool
def search_docs() -> Tool:
    async def execute(query: str) -> str:
        """Keyword search over the Halcyon Robotics document corpus.

        Args:
            query: Keywords to search for.

        Returns:
            Up to 5 lines "<id> | <title> | <snippet>", best match first.
        """
        METER.check()
        hits = _index().search(query)
        if not hits:
            return f"no results for query: {query}"
        return "\n".join(f"{h.id} | {h.title} | {h.snippet}" for h in hits)

    return execute


@tool
def get_doc() -> Tool:
    async def execute(id: str) -> str:
        """Fetch the full text of one document.

        Args:
            id: Document ID, e.g. HR-001.

        Returns:
            The document markdown including its front matter.
        """
        METER.check()
        doc = _index().docs.get(id.strip())
        return doc.text if doc else f"error: no document with id {id}"

    return execute


@tool
def calculate() -> Tool:
    async def execute(expr: str) -> str:
        """Evaluate an arithmetic expression.

        Args:
            expr: Numbers with + - * / // % ** and parentheses, e.g. "179 * 25".

        Returns:
            The result, or "error: <reason>".
        """
        METER.check()
        try:
            return format_number(safe_eval(expr))
        except ValueError as e:
            return f"error: {e}"

    return execute


# ---------------------------------------------------------------- FINAL line + solver
class Final(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    answer: str
    citations: list[str]
    abstain: bool


def parse_final(text: str) -> Final | None:
    """Parse the last `FINAL: {...}` line. None if missing or invalid (-> status format_error)."""
    for line in reversed((text or "").splitlines()):
        line = line.strip()
        if line.startswith("FINAL:"):
            try:
                return Final.model_validate_json(line[len("FINAL:"):])
            except ValidationError:
                return None
    return None


def load_prompt(variant: str, prompts_dir: str | Path = PROMPTS_DIR) -> str:
    d = Path(prompts_dir)
    return (d / f"{variant}.md").read_text(encoding="utf-8").rstrip() + "\n\n" + \
        (d / "_output_contract.md").read_text(encoding="utf-8")


@solver
def arena_agent(variant: str, prompts_dir: str = str(PROMPTS_DIR), cache: bool = True) -> Solver:
    # ponytail: inserts ChatMessageSystem directly; inspect's system_message() str-formats and mangles the JSON braces
    prompt = load_prompt(variant, prompts_dir)
    tools = use_tools(search_docs(), get_doc(), calculate())
    loop = generate(tool_calls="loop", cache=CachePolicy(expiry=None) if cache else False)

    async def solve(state: TaskState, generate_fn: Generate) -> TaskState:
        METER.check()
        state.messages.insert(0, ChatMessageSystem(content=prompt))
        state = await tools(state, generate_fn)
        return await loop(state, generate_fn)

    return solve
