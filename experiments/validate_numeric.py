"""Numeric validation experiment: EML representation vs original expression.

For each benchmark expression and numeric range, verifies that:

1. The original expression is domain-safe (symbolic + numeric).
2. The EML-encoded and evaluated expression matches the original numerically.

Run with::

    python -m experiments.validate_numeric

Returns structured results suitable for further analysis or logging.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import sympy as sp

from eml import configure_logging, evaluate_eml, validate_domain
from eml.config import BENCHMARK_SAMPLE_POINTS
from eml.transform import transform
from experiments.benchmarks import benchmark_expressions

logger = logging.getLogger(__name__)


def _safe_lambdify(expr: sp.Expr, symbol: sp.Symbol) -> Any:
    """Return a NumPy-compatible lambda for *expr*.

    Parameters
    ----------
    expr:
        Standard SymPy expression (no EML nodes).
    symbol:
        Free variable symbol.

    Returns
    -------
    callable
        Vectorized numeric function.
    """
    return sp.lambdify(
        symbol,
        expr,
        [{"exp": np.exp, "log": np.log, "sqrt": np.sqrt, "Abs": np.abs}],
    )


def validate_numeric() -> list[dict[str, Any]]:
    """Validate all benchmark expressions numerically and return structured results.

    For each (expression, range) pair:

    - Checks symbolic and numeric domain safety for the original expression.
    - Checks symbolic and numeric domain safety for the EML-evaluated form.
    - Computes mean and max absolute error between original and EML forms.

    Returns
    -------
    list[dict]
        One entry per (expression, range) with keys:
        ``expression``, ``range``, ``status``, and optionally
        ``mean_error``, ``max_error``, ``conditions``, ``error``.
    """
    x: sp.Symbol = sp.symbols("x")
    results: list[dict[str, Any]] = []

    for entry in benchmark_expressions():
        expr: sp.Expr = entry["expr"]
        eml_expr: sp.Expr = transform(expr)
        eml_eval: sp.Expr = evaluate_eml(eml_expr)

        for low, high in entry["ranges"]:
            xs: np.ndarray = np.linspace(low, high, BENCHMARK_SAMPLE_POINTS)
            original_domain = validate_domain(expr, x, sample_points=xs)
            transformed_domain = validate_domain(eml_eval, x, sample_points=xs)

            if not original_domain["symbolic_safe"]:
                logger.warning(
                    "SKIP %s on [%.2f, %.2f]: original symbolically unsafe",
                    entry["name"],
                    low,
                    high,
                )
                results.append(
                    {
                        "expression": entry["name"],
                        "range": (low, high),
                        "status": "original_symbolic_unsafe",
                        "conditions": original_domain["conditions"],
                    }
                )
                continue

            if not original_domain["numeric_ok"]:
                logger.warning(
                    "SKIP %s on [%.2f, %.2f]: original numerically unsafe",
                    entry["name"],
                    low,
                    high,
                )
                results.append(
                    {
                        "expression": entry["name"],
                        "range": (low, high),
                        "status": "original_numeric_unsafe",
                        "conditions": original_domain["conditions"],
                    }
                )
                continue

            if not transformed_domain["symbolic_safe"]:
                logger.warning(
                    "SKIP %s on [%.2f, %.2f]: transformed symbolically unsafe",
                    entry["name"],
                    low,
                    high,
                )
                results.append(
                    {
                        "expression": entry["name"],
                        "range": (low, high),
                        "status": "transformed_symbolic_unsafe",
                        "conditions": transformed_domain["conditions"],
                    }
                )
                continue

            if not transformed_domain["numeric_ok"]:
                logger.warning(
                    "SKIP %s on [%.2f, %.2f]: transformed numerically unsafe",
                    entry["name"],
                    low,
                    high,
                )
                results.append(
                    {
                        "expression": entry["name"],
                        "range": (low, high),
                        "status": "transformed_numeric_unsafe",
                        "conditions": transformed_domain["conditions"],
                    }
                )
                continue

            f_orig = _safe_lambdify(expr, x)
            f_eml = _safe_lambdify(eml_eval, x)
            with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
                y_orig = np.asarray(f_orig(xs), dtype=float)
                y_eml = np.asarray(f_eml(xs), dtype=float)
            error: np.ndarray = np.abs(y_orig - y_eml)
            mean_err = float(np.nanmean(error))
            max_err = float(np.nanmax(error))
            logger.info(
                "OK  %s on [%.2f, %.2f]: mean=%.2e, max=%.2e",
                entry["name"],
                low,
                high,
                mean_err,
                max_err,
            )
            results.append(
                {
                    "expression": entry["name"],
                    "range": (low, high),
                    "mean_error": mean_err,
                    "max_error": max_err,
                    "status": "ok",
                }
            )

    return results


if __name__ == "__main__":
    configure_logging()
    validate_numeric()
