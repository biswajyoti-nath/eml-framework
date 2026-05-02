"""Domain validation utilities for the EML exp-log fragment.

This module enforces the positive-domain assumptions that underpin the EML
representation.  All log arguments and non-integer power bases must be provably
positive for the rewrite rules to be semantically correct.

Public API
----------
assert_domain       — raise DomainError if the expression is not domain-safe.
is_positive_expr    — symbolic positivity test for a single expression.
is_domain_safe      — full domain-safety report (symbolic).
validate_domain     — combined symbolic + numeric validation report.
validate_log_domain — numeric log-argument positivity check.
validate_transformed_domain — full EML-expression numeric well-definedness check.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import sympy as sp

from .config import DEFAULT_SAMPLE_COUNT, DEFAULT_SAMPLE_HIGH, DEFAULT_SAMPLE_LOW
from .core import extract_log_arguments
from .exceptions import DomainError

logger = logging.getLogger(__name__)

__all__ = [
    "assert_domain",
    "is_positive_expr",
    "is_domain_safe",
    "validate_domain",
    "validate_log_domain",
    "validate_transformed_domain",
]


# ---------------------------------------------------------------------------
# Symbolic positivity helpers
# ---------------------------------------------------------------------------


def is_positive_expr(expr: sp.Expr) -> bool:
    """Return True when *expr* is provably positive under exp-log assumptions.

    This is a best-effort symbolic check.  It returns False for expressions
    whose sign cannot be determined, to stay conservative (fail-safe).

    Parameters
    ----------
    expr:
        Any SymPy expression.

    Returns
    -------
    bool
        True only when positivity is proven; False when unknown or negative.

    Notes
    -----
    - ``exp(…)`` is always positive.
    - Even powers are non-negative but not strictly positive; we return False.
    - Falls back to ``sympy.ask(Q.positive)`` for general cases.
    """
    if expr.is_positive:
        return True
    if expr.func == sp.exp:
        return True
    if expr.func == sp.Abs:
        return False  # Abs can be zero
    if expr.is_Pow and expr.args[1].is_integer and expr.args[1] % 2 == 0:
        return False  # even powers include zero
    result: bool | None = sp.ask(sp.Q.positive(expr))
    return result is True


def is_domain_safe(expr: sp.Expr) -> tuple[bool, dict[str, Any]]:
    """Symbolically check EML domain assumptions for *expr*.

    Traverses the expression tree and checks:

    * All ``log`` arguments are provably positive.
    * All non-integer ``Pow`` bases are provably positive.

    Parameters
    ----------
    expr:
        SymPy expression to validate.

    Returns
    -------
    safe : bool
        True when both log and power domains are satisfied.
    conditions : dict
        Detailed breakdown with keys ``log_args``, ``nonint_pow_bases``,
        ``log_safe``, ``pow_safe``.

    Limitations
    -----------
    This is a conservative check — it may report unsafe for expressions that
    are actually safe on restricted domains (e.g. user-constrained variables).
    """
    log_args: list[sp.Expr] = []
    nonint_pow_bases: list[sp.Expr] = []

    for node in sp.preorder_traversal(expr):
        if node.func == sp.log:
            log_args.append(node.args[0])
        if node.func == sp.Pow and not node.args[1].is_integer:
            nonint_pow_bases.append(node.args[0])

    log_safe = all(is_positive_expr(arg) for arg in log_args)
    pow_safe = all(is_positive_expr(base) for base in nonint_pow_bases)
    safe = log_safe and pow_safe

    logger.debug(
        "is_domain_safe: log_safe=%s, pow_safe=%s, log_args=%s, pow_bases=%s",
        log_safe,
        pow_safe,
        log_args,
        nonint_pow_bases,
    )

    return safe, {
        "log_args": log_args,
        "nonint_pow_bases": nonint_pow_bases,
        "log_safe": log_safe,
        "pow_safe": pow_safe,
    }


# ---------------------------------------------------------------------------
# Numeric helpers
# ---------------------------------------------------------------------------



def validate_log_domain(
    expr: sp.Expr,
    symbol: sp.Symbol,
    sample_points: np.ndarray | None = None,
) -> bool:
    """Check that all log arguments in *expr* are positive on *sample_points*.

    Parameters
    ----------
    expr:
        Expression to validate.
    symbol:
        The free variable to substitute sample values for.
    sample_points:
        1-D array of real values to test. Defaults to 100 points in [0.1, 5.0].

    Returns
    -------
    bool
        True when all log arguments evaluate to a strictly positive value at
        every sample point; False if any are ≤ 0 or numerically problematic.
    """
    log_args = extract_log_arguments(expr)
    if not log_args:
        return True

    if sample_points is None:
        sample_points = np.linspace(DEFAULT_SAMPLE_LOW, DEFAULT_SAMPLE_HIGH, DEFAULT_SAMPLE_COUNT)

    lambdas = [sp.lambdify(symbol, arg, [{"Abs": np.abs, "sqrt": np.sqrt}]) for arg in log_args]

    for fn in lambdas:
        try:
            values = fn(sample_points)
        except Exception:
            values = np.vectorize(lambda x, _fn=fn: _fn(np.asarray(x, dtype=float)))(  # type: ignore[misc]
                sample_points
            )
        if np.any(np.asarray(values) <= 0):
            logger.debug("validate_log_domain: non-positive log argument detected")
            return False

    return True


def _evaluate_numeric(
    expr: sp.Expr,
    symbol: sp.Symbol,
    sample_points: np.ndarray,
) -> np.ndarray:
    """Lambdify and evaluate *expr* on *sample_points*, suppressing numeric errors."""
    f = sp.lambdify(
        symbol,
        expr,
        [{"exp": np.exp, "log": np.log, "Abs": np.abs, "sqrt": np.sqrt}],
    )
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        try:
            values = np.asarray(f(sample_points))
        except Exception:
            vectorized = np.vectorize(lambda x: f(np.asarray(x, dtype=float)))  # type: ignore[misc]
            values = vectorized(sample_points)
    return values


# ---------------------------------------------------------------------------
# Combined validation
# ---------------------------------------------------------------------------


def validate_domain(
    expr: sp.Expr,
    symbol: sp.Symbol,
    sample_points: np.ndarray | None = None,
) -> dict[str, Any]:
    """Validate EML domain safety using symbolic checks and sampled values.

    Parameters
    ----------
    expr:
        Expression to validate.
    symbol:
        The free variable symbol.
    sample_points:
        Optional custom sample grid. Defaults to 100 points in [0.1, 5.0].

    Returns
    -------
    dict with keys:
        ``valid_domain`` — True when both symbolic and numeric checks pass.
        ``symbolic_safe`` — result of :func:`is_domain_safe`.
        ``numeric_ok``   — result of :func:`validate_log_domain`.
        ``conditions``   — detailed symbolic breakdown dict.
    """
    symbolic_safe, conditions = is_domain_safe(expr)
    numeric_ok = validate_log_domain(expr, symbol, sample_points)
    return {
        "valid_domain": symbolic_safe and numeric_ok,
        "symbolic_safe": symbolic_safe,
        "numeric_ok": numeric_ok,
        "conditions": conditions,
    }


def validate_transformed_domain(
    expr: sp.Expr,
    symbol: sp.Symbol,
    sample_points: np.ndarray | None = None,
) -> bool:
    """Validate that a transformed EML expression is numerically well-defined.

    Evaluates the EML semantic (substituting EML nodes with their exp/log
    expansions) and checks that all log arguments are positive *and* all output
    values are finite on the sample grid.

    Parameters
    ----------
    expr:
        EML-encoded expression (may contain ``eml(…)`` nodes).
    symbol:
        Free variable symbol.
    sample_points:
        Sample grid. Defaults to 100 points in [0.1, 5.0].

    Returns
    -------
    bool
        True when the expression is domain-safe and numerically finite.
    """
    from .core import evaluate_eml  # local import to avoid circular dependency

    if sample_points is None:
        sample_points = np.linspace(DEFAULT_SAMPLE_LOW, DEFAULT_SAMPLE_HIGH, DEFAULT_SAMPLE_COUNT)

    evaluated = evaluate_eml(expr)

    if not validate_log_domain(evaluated, symbol, sample_points):
        return False

    values = _evaluate_numeric(evaluated, symbol, sample_points)
    if values.shape != sample_points.shape:
        return False

    return bool(np.all(np.isfinite(values)))


# ---------------------------------------------------------------------------
# Enforcement (raises)
# ---------------------------------------------------------------------------


def assert_domain(expr: sp.Expr) -> None:
    """Raise :class:`~eml.exceptions.DomainError` if *expr* is not domain-safe.

    This is the strict enforcement entry point.  Use it at the boundary of any
    function that requires all log arguments and non-integer power bases to be
    provably positive.

    Parameters
    ----------
    expr:
        Expression to validate.

    Raises
    ------
    DomainError
        When any log argument or non-integer power base cannot be proven
        positive by SymPy's assumption system.

    Notes
    -----
    The check is *conservative* — expressions that are safe on a restricted
    domain but not globally may trigger this error.  In such cases, use
    :func:`validate_domain` with explicit ``sample_points`` instead.
    """
    safe, conditions = is_domain_safe(expr)
    if not safe:
        problem_parts: list[str] = []
        if not conditions["log_safe"]:
            problem_parts.append(f"unsafe log arguments: {conditions['log_args']}")
        if not conditions["pow_safe"]:
            problem_parts.append(f"unsafe power bases: {conditions['nonint_pow_bases']}")
        detail = "; ".join(problem_parts)
        raise DomainError(
            f"Expression violates EML positive-domain assumptions — {detail}. "
            "Ensure all log arguments and non-integer power bases are provably positive."
        )
