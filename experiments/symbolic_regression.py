"""Symbolic regression experiment: EML vs baseline expression search.

Uses an evolutionary search (random generation + mutation + elite selection) to
find expressions that approximate target functions, comparing the EML grammar
against a standard SymPy exp-log grammar.

Run with::

    python -m experiments.symbolic_regression

All randomness is seeded via :data:`~eml.config.RANDOM_SEED` for
reproducibility.  Search hyperparameters are centralized in :mod:`eml.config`.
"""

from __future__ import annotations

import logging
import random
from typing import Any

import numpy as np
import sympy as sp

from eml import configure_logging, evaluate_eml, total_node_count
from eml.config import (
    RANDOM_SEED,
    SEARCH_BUDGET,
    SEARCH_ELITE_SIZE,
    SEARCH_MAX_DEPTH,
    SEARCH_POP_SIZE,
    SEARCH_SAMPLE_POINTS,
)
from eml.core import eml

logger = logging.getLogger(__name__)

__all__ = [
    "random_eml_expr",
    "random_sympy_expr",
    "evaluate_candidate",
    "evaluate_standard_expr",
    "target_functions",
    "mse",
    "search_for_target",
    "run_search",
    # EML grammar primitives (used by tests)
    "log_expr",
    "exp_expr",
    "mul_expr",
    "pow_expr",
]


# ---------------------------------------------------------------------------
# EML grammar primitives
# ---------------------------------------------------------------------------


def log_expr(expr: sp.Expr) -> sp.Expr:
    """Encode ``log(expr)`` using the EML primitive."""
    return sp.Integer(1) - eml(sp.Integer(0), expr)


def exp_expr(expr: sp.Expr) -> sp.Expr:
    """Encode ``exp(expr)`` using the EML primitive."""
    return eml(expr, sp.Integer(1))


def mul_expr(a: sp.Expr, b: sp.Expr) -> sp.Expr:
    """Encode ``a * b`` using the EML exp-log identity."""
    return exp_expr(log_expr(a) + log_expr(b))


def pow_expr(base: sp.Expr, exponent: sp.Expr) -> sp.Expr:
    """Encode ``base ** exponent`` using the EML exp-log identity."""
    return exp_expr(log_expr(base) * exponent)


# ---------------------------------------------------------------------------
# Random expression generators
# ---------------------------------------------------------------------------


def _random_leaf(x: sp.Symbol) -> sp.Expr:
    """Return a random atomic EML leaf: x, 1, or 2."""
    return random.choice([x, sp.Integer(1), sp.Integer(2)])


def random_eml_expr(depth: int, x: sp.Symbol) -> sp.Expr:
    """Generate a random expression in the EML grammar.

    Parameters
    ----------
    depth:
        Maximum remaining tree depth (base case at ≤ 0).
    x:
        Free variable symbol.

    Returns
    -------
    sp.Expr
        A random EML expression tree.
    """
    if depth <= 0:
        return _random_leaf(x)

    node_type: str = random.choices(
        ["leaf", "eml", "add", "log", "exp", "mul", "pow", "neg"],
        weights=[0.12, 0.12, 0.13, 0.2, 0.08, 0.1, 0.2, 0.05],
        k=1,
    )[0]

    if node_type == "leaf":
        return _random_leaf(x)
    if node_type == "eml":
        return eml(random_eml_expr(depth - 1, x), random_eml_expr(depth - 1, x))
    if node_type == "add":
        return random_eml_expr(depth - 1, x) + random_eml_expr(depth - 1, x)
    if node_type == "log":
        return log_expr(random_eml_expr(depth - 1, x))
    if node_type == "exp":
        return exp_expr(random_eml_expr(depth - 1, x))
    if node_type == "mul":
        return mul_expr(random_eml_expr(depth - 1, x), random_eml_expr(depth - 1, x))
    if node_type == "pow":
        exponent: sp.Expr = random.choice([sp.Integer(2), sp.Integer(3)])
        return pow_expr(random_eml_expr(depth - 1, x), exponent)
    if node_type == "neg":
        return -random_eml_expr(depth - 1, x)

    return _random_leaf(x)


def random_sympy_expr(depth: int, x: sp.Symbol) -> sp.Expr:
    """Generate a random expression in the standard SymPy exp-log grammar.

    Parameters
    ----------
    depth:
        Maximum remaining tree depth (base case at ≤ 0).
    x:
        Free variable symbol.

    Returns
    -------
    sp.Expr
        A random SymPy expression tree.
    """
    if depth <= 0:
        return _random_leaf(x)

    node_type = random.choices(
        ["leaf", "add", "exp", "log", "mul", "pow", "neg"],
        weights=[0.2, 0.15, 0.1, 0.15, 0.2, 0.15, 0.05],
        k=1,
    )[0]

    if node_type == "leaf":
        return _random_leaf(x)
    if node_type == "add":
        return random_sympy_expr(depth - 1, x) + random_sympy_expr(depth - 1, x)
    if node_type == "exp":
        return sp.exp(random_sympy_expr(depth - 1, x))
    if node_type == "log":
        return sp.log(sp.Abs(random_sympy_expr(depth - 1, x)) + 0.1)
    if node_type == "mul":
        return random_sympy_expr(depth - 1, x) * random_sympy_expr(depth - 1, x)
    if node_type == "pow":
        exponent = random.choice([sp.Integer(2), sp.Integer(3), sp.Integer(-1)])
        return random_sympy_expr(depth - 1, x) ** exponent
    if node_type == "neg":
        return -random_sympy_expr(depth - 1, x)

    return _random_leaf(x)


# ---------------------------------------------------------------------------
# Candidate evaluation
# ---------------------------------------------------------------------------


def _safe_vectorized_evaluate(f: Any, xs: np.ndarray) -> np.ndarray:
    """Evaluate *f* on *xs*, falling back to vectorized scalar calls on error.

    Parameters
    ----------
    f:
        Lambdified numeric function.
    xs:
        Input sample array.

    Returns
    -------
    np.ndarray
        Output values (may contain NaN / Inf).
    """
    try:
        return np.asarray(f(xs))
    except Exception:
        vectorized = np.vectorize(lambda x: f(np.asarray(x, dtype=float)))  # type: ignore[misc]
        return np.asarray(vectorized(xs))


def evaluate_candidate(expr: sp.Expr, x: sp.Symbol, xs: np.ndarray) -> np.ndarray | None:
    """Evaluate an EML candidate expression numerically on *xs*.

    Evaluates the EML semantics, lambdifies, and checks for complex / nan /
    inf values.  Returns None for any numerically degenerate candidate.

    Parameters
    ----------
    expr:
        EML-encoded candidate expression.
    x:
        Free variable symbol.
    xs:
        Sample grid.

    Returns
    -------
    np.ndarray or None
        Evaluated output array, or None if evaluation failed or produced
        non-finite / complex values.
    """
    candidate: sp.Expr = evaluate_eml(expr)
    try:
        f = sp.lambdify(
            x,
            candidate,
            [{"exp": np.exp, "log": np.log, "Abs": np.abs, "sqrt": np.sqrt}],
        )
    except Exception:
        return None

    try:
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            y = _safe_vectorized_evaluate(f, xs)
    except Exception:
        return None

    if y.shape != xs.shape:
        return None
    if np.any(np.iscomplex(y)) or np.any(np.isnan(y)) or np.any(np.isinf(y)):
        return None
    return y


def evaluate_standard_expr(expr: sp.Expr, x: sp.Symbol, xs: np.ndarray) -> np.ndarray | None:
    """Evaluate a standard SymPy expression numerically on *xs*.

    Parameters
    ----------
    expr:
        Standard SymPy expression (no EML nodes).
    x:
        Free variable symbol.
    xs:
        Sample grid.

    Returns
    -------
    np.ndarray or None
        Evaluated output array, or None on failure.
    """
    try:
        f = sp.lambdify(
            x,
            expr,
            [{"exp": np.exp, "log": np.log, "Abs": np.abs, "sqrt": np.sqrt}],
        )
    except Exception:
        return None

    try:
        with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
            y = _safe_vectorized_evaluate(f, xs)
    except Exception:
        return None

    if y.shape != xs.shape:
        return None
    if np.any(np.iscomplex(y)) or np.any(np.isnan(y)) or np.any(np.isinf(y)):
        return None
    return y


# ---------------------------------------------------------------------------
# Target functions and fitness
# ---------------------------------------------------------------------------


def target_functions() -> list[dict[str, Any]]:
    """Return the target function suite for symbolic regression.

    Returns
    -------
    list[dict]
        Each entry: ``name`` (str), ``expr`` (sp.Expr), ``range`` (tuple[float, float]).
    """
    x: sp.Symbol = sp.symbols("x")
    return [
        {"name": "x^2 + x", "expr": x**2 + x, "range": (0.1, 3.0)},
        {"name": "exp(x)", "expr": sp.exp(x), "range": (-2.0, 2.0)},
        {"name": "log(x)", "expr": sp.log(x), "range": (0.1, 3.0)},
        {"name": "x^2 * log(x)", "expr": x**2 * sp.log(x), "range": (0.1, 3.0)},
        {"name": "exp(x) + log(x)", "expr": sp.exp(x) + sp.log(x), "range": (0.1, 3.0)},
    ]


def mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute mean squared error between two arrays.

    Parameters
    ----------
    y_true:
        Ground-truth values.
    y_pred:
        Predicted values.

    Returns
    -------
    float
        Mean squared error.
    """
    return float(np.mean((y_true - y_pred) ** 2))


# ---------------------------------------------------------------------------
# Evolutionary search
# ---------------------------------------------------------------------------


def _mutate_candidate(expr: sp.Expr, x: sp.Symbol) -> sp.Expr:
    """Apply a random mutation to an EML candidate expression.

    Parameters
    ----------
    expr:
        Parent expression to mutate.
    x:
        Free variable symbol.

    Returns
    -------
    sp.Expr
        Mutated expression.
    """
    choice = random.random()
    if choice < 0.25:
        return expr + random_eml_expr(1, x)
    if choice < 0.5:
        return eml(expr, random_eml_expr(1, x))
    if choice < 0.7:
        return mul_expr(expr, random_eml_expr(1, x))
    if choice < 0.9:
        exponent: sp.Expr = random.choice([sp.Integer(2), sp.Integer(3)])
        return pow_expr(expr, exponent)
    return -expr


def search_for_target(
    target: dict[str, Any],
    candidate_generator: Any,
    evaluator: Any,
    budget: int = SEARCH_BUDGET,
    pop_size: int = SEARCH_POP_SIZE,
    elite_size: int = SEARCH_ELITE_SIZE,
    max_depth: int = SEARCH_MAX_DEPTH,
    seed: int = RANDOM_SEED,
) -> dict[str, Any]:
    """Run the evolutionary symbolic regression search for *target*.

    Uses an elite-retention evolutionary strategy: generate a population,
    score all candidates, carry over the top ``elite_size`` expressions, and
    fill the rest with mutations.

    Parameters
    ----------
    target:
        Target function dict (``name``, ``expr``, ``range``).
    candidate_generator:
        Callable ``(depth, x) → sp.Expr`` for initial population.
    evaluator:
        Callable ``(expr, x, xs) → np.ndarray | None`` for numeric evaluation.
    budget:
        Total number of candidate evaluations allowed.
    pop_size:
        Population size per generation.
    elite_size:
        Number of top candidates retained each generation.
    max_depth:
        Maximum expression tree depth.
    seed:
        Random seed for reproducibility.

    Returns
    -------
    dict
        Best result with keys ``expr`` (sp.Expr or None), ``mse`` (float),
        ``y`` (np.ndarray or None).
    """
    random.seed(seed)
    np.random.seed(seed)

    x: sp.Symbol = sp.symbols("x")
    low, high = target["range"]
    xs: np.ndarray = np.linspace(low, high, SEARCH_SAMPLE_POINTS)
    y_true: np.ndarray = sp.lambdify(x, target["expr"], "numpy")(xs)

    def score(expr: sp.Expr) -> tuple[float, np.ndarray] | None:
        y_pred = evaluator(expr, x, xs)
        if y_pred is None:
            return None
        return mse(y_true, y_pred), y_pred

    population: list[sp.Expr] = [candidate_generator(max_depth, x) for _ in range(pop_size)]
    best: dict[str, Any] = {"expr": None, "mse": float("inf"), "y": None}
    evals = 0

    while evals < budget:
        scored: list[tuple[float, sp.Expr, np.ndarray]] = []

        for expr in population:
            result = score(expr)
            evals += 1
            if result is None:
                continue
            error, y_pred = result
            scored.append((error, expr, y_pred))
            if error < best["mse"]:
                best.update({"expr": expr, "mse": error, "y": y_pred})
            if evals >= budget:
                break

        if not scored:
            population = [candidate_generator(max_depth, x) for _ in range(pop_size)]
            continue

        scored.sort(key=lambda item: item[0])
        elites: list[sp.Expr] = [expr for _, expr, _ in scored[:elite_size]]
        population = elites.copy()

        while len(population) < pop_size:
            parent = random.choice(elites)
            population.append(_mutate_candidate(parent, x))

    return best


# ---------------------------------------------------------------------------
# Experiment runner
# ---------------------------------------------------------------------------


def run_search() -> list[dict[str, Any]]:
    """Run the full symbolic regression experiment and return structured results.

    Searches for each target function using both the EML grammar and the
    standard SymPy baseline grammar, then logs a summary.

    Returns
    -------
    list[dict]
        One entry per target with keys:
        ``target``, ``eml_mse``, ``eml_expr``, ``eml_tree_size``,
        ``baseline_mse``, ``baseline_expr``, ``baseline_tree_size``.
    """
    results: list[dict[str, Any]] = []

    for target in target_functions():
        logger.info("Searching for target: %s", target["name"])

        best_eml = search_for_target(
            target,
            candidate_generator=random_eml_expr,
            evaluator=evaluate_candidate,
        )
        best_baseline = search_for_target(
            target,
            candidate_generator=random_sympy_expr,
            evaluator=evaluate_standard_expr,
        )

        if best_eml["expr"] is None:
            logger.warning("No valid EML candidate found for %s", target["name"])
        else:
            logger.info(
                "EML     — MSE: %.3e | tree: %d | expr: %s",
                best_eml["mse"],
                total_node_count(best_eml["expr"]),
                best_eml["expr"],
            )

        if best_baseline["expr"] is None:
            logger.warning("No baseline candidate found for %s", target["name"])
        else:
            logger.info(
                "Baseline — MSE: %.3e | tree: %d | expr: %s",
                best_baseline["mse"],
                total_node_count(best_baseline["expr"]),
                best_baseline["expr"],
            )

        results.append(
            {
                "target": target["name"],
                "eml_mse": best_eml["mse"],
                "eml_expr": str(best_eml["expr"]) if best_eml["expr"] is not None else None,
                "eml_tree_size": (
                    total_node_count(best_eml["expr"]) if best_eml["expr"] is not None else None
                ),
                "baseline_mse": best_baseline["mse"],
                "baseline_expr": (
                    str(best_baseline["expr"]) if best_baseline["expr"] is not None else None
                ),
                "baseline_tree_size": (
                    total_node_count(best_baseline["expr"])
                    if best_baseline["expr"] is not None
                    else None
                ),
            }
        )

    return results


if __name__ == "__main__":
    configure_logging()
    run_search()
