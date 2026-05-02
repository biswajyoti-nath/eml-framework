"""Benchmark expression catalogue for EML complexity analysis.

Each entry defines a named expression and the numeric ranges to validate it on.
Ranges are chosen to ensure domain safety (positive log arguments, etc.) as
well as to include boundary cases.
"""

from __future__ import annotations

from typing import Any

import sympy as sp

__all__ = ["benchmark_expressions"]


def benchmark_expressions() -> list[dict[str, Any]]:
    """Return the canonical benchmark suite for EML complexity evaluation.

    Returns
    -------
    list[dict]
        Each entry contains:

        ``name`` : str
            Human-readable label for the expression.
        ``expr`` : sp.Expr
            SymPy expression within the EML exp-log fragment.
        ``ranges`` : list[tuple[float, float]]
            List of ``(low, high)`` numeric validation intervals.
    """
    x: sp.Symbol = sp.symbols("x")
    return [
        {
            "name": "cubic",
            "expr": x**3 + 2 * x - 1,
            "ranges": [(-2.0, 2.0), (-5.0, 5.0)],
        },
        {
            "name": "square",
            "expr": x**2,
            "ranges": [(-3.0, 3.0), (0.1, 5.0)],
        },
        {
            "name": "cube",
            "expr": x**3,
            "ranges": [(-3.0, 3.0), (0.1, 5.0)],
        },
        {
            "name": "rational",
            "expr": x / (1 + x**2),
            "ranges": [(-2.0, 2.0), (-5.0, 5.0)],
        },
        {
            "name": "exp_sum",
            "expr": sp.exp(x) + sp.exp(-x),
            "ranges": [(-2.0, 2.0), (-5.0, 5.0)],
        },
        {
            "name": "exp_log_identity",
            "expr": sp.exp(sp.log(x + 2)),
            "ranges": [(0.1, 3.0), (1.0, 5.0)],
        },
        {
            "name": "log_sum",
            "expr": sp.log(x + 3) + sp.log(x + 1),
            "ranges": [(0.1, 3.0), (1.0, 4.0)],
        },
        {
            "name": "cubic_shifted",
            "expr": (x + 2) ** 3,
            "ranges": [(-2.0, 2.0), (-1.0, 3.0)],
        },
        {
            "name": "x_exp_x",
            "expr": x * sp.exp(x),
            "ranges": [(-2.0, 2.0), (-1.0, 4.0)],
        },
        {
            "name": "exp_log_product",
            "expr": sp.exp(x * sp.log(x + 1)),
            "ranges": [(0.1, 3.0), (1.0, 4.0)],
        },
        {
            "name": "log_of_product",
            "expr": sp.log((x + 1) * (x + 2)),
            "ranges": [(0.1, 3.0), (1.0, 5.0)],
        },
        {
            "name": "sqrt_via_pow",
            "expr": (x**2 + 1) ** sp.Rational(1, 2),
            "ranges": [(-3.0, 3.0), (-1.5, 2.5)],
        },
        {
            "name": "log_square",
            "expr": sp.log(x + 2) ** 2,
            "ranges": [(0.1, 3.0), (1.0, 4.0)],
        },
        {
            "name": "division",
            "expr": (x + 1) / (x + 2),
            "ranges": [(-1.9, 1.9), (0.1, 3.0)],
        },
        {
            "name": "log_of_exp_plus_one",
            "expr": sp.log(sp.exp(x) + 1),
            "ranges": [(-2.0, 2.0), (-1.0, 3.0)],
        },
        {
            "name": "negative_power",
            "expr": x ** (-1),
            "ranges": [(-5.0, -0.2), (0.2, 5.0)],
        },
        {
            "name": "log_of_quad",
            "expr": sp.log(x**2 + 1),
            "ranges": [(-3.0, 3.0), (-5.0, 5.0)],
        },
        {
            "name": "exp_times_log",
            "expr": sp.exp(x) * sp.log(x + 4),
            "ranges": [(0.1, 3.0), (1.0, 5.0)],
        },
        {
            "name": "mixed_power",
            "expr": (x + 1) ** sp.Rational(3, 2),
            "ranges": [(-1.0, 3.0), (-0.5, 4.0)],
        },
        {
            "name": "gaussian",
            "expr": sp.exp(-(x**2)),
            "ranges": [(-2.0, 2.0), (-5.0, 5.0)],
        },
        {
            "name": "log_of_exp",
            "expr": sp.log(sp.exp(x) + 2),
            "ranges": [(-2.0, 2.0), (-1.0, 4.0)],
        },
        {
            "name": "power_times_exp",
            "expr": x**2 * sp.exp(x),
            "ranges": [(-2.0, 2.0), (-1.0, 4.0)],
        },
        {
            "name": "log_abs",
            "expr": sp.log(sp.Abs(x)),
            "ranges": [(-5.0, -0.1), (0.1, 5.0)],
        },
    ]
