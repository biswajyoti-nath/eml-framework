"""Tests for the symbolic regression experiment module."""

from __future__ import annotations

import numpy as np
import sympy as sp

from experiments.symbolic_regression import (
    evaluate_expr,
    compute_mse,
    exp_expr,
    log_expr,
    target_functions,
)


def test_evaluate_expr_exp_expr_matches_numpy() -> None:
    """exp_expr(x) should evaluate numerically to numpy.exp(xs)."""
    x = sp.symbols("x")
    xs = np.linspace(0.1, 2.0, 25)

    y_pred = evaluate_expr(exp_expr(x), x, xs, is_eml=True)
    assert y_pred is not None
    assert np.allclose(y_pred, np.exp(xs), atol=1e-8, rtol=1e-8)


def test_evaluate_expr_log_expr_matches_numpy() -> None:
    """log_expr(x) should evaluate numerically to numpy.log(xs)."""
    x = sp.symbols("x")
    xs = np.linspace(0.1, 3.0, 25)

    y_pred = evaluate_expr(log_expr(x), x, xs, is_eml=True)
    assert y_pred is not None
    assert np.allclose(y_pred, np.log(xs), atol=1e-8, rtol=1e-8)


def test_target_functions_have_valid_ranges() -> None:
    """All target functions must have a name and valid (low < high) float ranges."""
    functions = target_functions()
    assert any(target["name"] == "exp(x)" for target in functions)
    assert any(target["name"] == "log(x)" for target in functions)

    for target in functions:
        low, high = target["range"]
        assert low < high
        assert isinstance(low, float)
        assert isinstance(high, float)
