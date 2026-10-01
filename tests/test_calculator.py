import pytest

from arena_evals.agents import format_number, safe_eval


@pytest.mark.parametrize("expr, expected", [
    ("179 * 25", 4475.0),
    ("(199 - 179) * 12", 240.0),
    ("-3 + 10 / 4", -0.5),
    ("7 // 2 + 7 % 2", 4.0),
    ("2 ** 10", 1024.0),
    ("1.5 * 2", 3.0),
    ("-(2 - 5)", 3.0),
])
def test_arithmetic(expr, expected):
    assert safe_eval(expr) == pytest.approx(expected)


@pytest.mark.parametrize("expr", [
    "__import__('os').system('echo pwned')",
    "abs(-1)",
    "x + 1",
    "(1).__class__",
    "[1, 2]",
    "'a' * 3",
    "True + 1",
    "2 ** 101",
    "10 ** 16",
    "1e309",
    "1 / 0",
    "5 % 0",
    "(-8) ** 0.5",
    "1 +",
    "lambda: 1",
    "1" + "+1" * 300,
])
def test_rejects_unsafe_or_invalid(expr):
    with pytest.raises(ValueError):
        safe_eval(expr)


def test_format_number():
    assert format_number(4475.0) == "4475"
    assert format_number(123456789012.0) == "123456789012"
    assert format_number(2 / 3) == "0.6666666667"
