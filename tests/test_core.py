"""Tests for core EML primitives and complexity metrics (eml.core).

Covers:
- eml(), exp_eml(), log_eml() constructor correctness
- evaluate_eml() round-trip for exp and log
- total_node_count, nonlinear_node_count, tree_depth
- eml_node_count, eml_depth, operator_count
- weighted_cost, unique_subexpression_count
- extract_log_arguments
- compute_metrics unified dict interface
"""

from __future__ import annotations

import sympy as sp

from eml import (
    EML,
    compute_metrics,
    eml,
    eml_depth,
    eml_node_count,
    evaluate_eml,
    exp_eml,
    extract_log_arguments,
    log_eml,
    nonlinear_node_count,
    operator_count,
    total_node_count,
    tree_depth,
    unique_subexpression_count,
    weighted_cost,
)

# ---------------------------------------------------------------------------
# EML primitive constructors
# ---------------------------------------------------------------------------


def test_eml_returns_eml_node(x: sp.Symbol) -> None:
    """eml(x, 1) should produce an EML function node."""
    result = eml(x, sp.Integer(1))
    assert result.func == EML


def test_exp_eml_is_eml_x_one(x: sp.Symbol) -> None:
    """exp_eml(x) should equal eml(x, 1)."""
    assert exp_eml(x) == eml(x, sp.Integer(1))


def test_log_eml_is_one_minus_eml_zero_x(x: sp.Symbol) -> None:
    """log_eml(x) should equal 1 - eml(0, x)."""
    assert log_eml(x) == sp.Integer(1) - eml(sp.Integer(0), x)


# ---------------------------------------------------------------------------
# evaluate_eml round-trips
# ---------------------------------------------------------------------------


def test_evaluate_eml_exp_roundtrip(x: sp.Symbol) -> None:
    """evaluate_eml(exp_eml(x)) should equal exp(x)."""
    result = evaluate_eml(exp_eml(x))
    assert result.equals(sp.exp(x))


def test_evaluate_eml_log_roundtrip(x: sp.Symbol) -> None:
    """evaluate_eml(log_eml(x)) should equal log(x)."""
    result = evaluate_eml(log_eml(x))
    assert result.equals(sp.log(x))


def test_evaluate_eml_no_eml_nodes_unchanged(x: sp.Symbol) -> None:
    """evaluate_eml on an expression with no EML nodes returns it unchanged."""
    expr = sp.exp(x) + sp.log(x + 1)
    assert evaluate_eml(expr).equals(expr)


# ---------------------------------------------------------------------------
# total_node_count
# ---------------------------------------------------------------------------


def test_total_node_count_atom() -> None:
    """An atom has exactly 1 node."""
    assert total_node_count(sp.Integer(1)) == 1


def test_total_node_count_exp(x: sp.Symbol) -> None:
    """exp(x) has 2 nodes: exp and x."""
    assert total_node_count(sp.exp(x)) == 2


# ---------------------------------------------------------------------------
# nonlinear_node_count
# ---------------------------------------------------------------------------


def test_nonlinear_node_count_no_nonlinear(x: sp.Symbol) -> None:
    """A polynomial has zero nonlinear nodes."""
    assert nonlinear_node_count(x**2 + x) == 0


def test_nonlinear_node_count_exp_log(x: sp.Symbol) -> None:
    """exp(x) + log(x) has exactly 2 nonlinear nodes."""
    assert nonlinear_node_count(sp.exp(x) + sp.log(x)) == 2


def test_nonlinear_node_count_eml(x: sp.Symbol) -> None:
    """An EML node counts as a nonlinear node."""
    assert nonlinear_node_count(eml(x, sp.Integer(1))) >= 1


# ---------------------------------------------------------------------------
# tree_depth
# ---------------------------------------------------------------------------


def test_tree_depth_atom() -> None:
    """Atomic expression has depth 0."""
    assert tree_depth(sp.Integer(5)) == 0


def test_tree_depth_exp(x: sp.Symbol) -> None:
    """exp(x) has depth 1."""
    assert tree_depth(sp.exp(x)) == 1


def test_tree_depth_nested(x: sp.Symbol) -> None:
    """exp(x + 1) has depth 2: exp → Add → x.

    Note: exp(log(x)) simplifies to x in SymPy (depth 0), so we use
    a non-simplifiable nesting to verify the depth counter.
    """
    assert tree_depth(sp.exp(x + 1)) == 2


# ---------------------------------------------------------------------------
# eml_node_count
# ---------------------------------------------------------------------------


def test_eml_node_count_zero_for_no_eml(x: sp.Symbol) -> None:
    """Standard expressions contain zero EML nodes."""
    assert eml_node_count(sp.exp(x) + sp.log(x)) == 0


def test_eml_node_count_one_for_single_eml(x: sp.Symbol) -> None:
    """A single eml() call should count as exactly 1."""
    assert eml_node_count(eml(x, sp.Integer(1))) == 1


def test_eml_node_count_nested(x: sp.Symbol) -> None:
    """Nested eml() calls should each be counted."""
    inner = eml(x, sp.Integer(1))
    outer = eml(inner, sp.Integer(1))
    assert eml_node_count(outer) == 2


# ---------------------------------------------------------------------------
# eml_depth
# ---------------------------------------------------------------------------


def test_eml_depth_zero_for_no_eml(x: sp.Symbol) -> None:
    """No EML nodes → depth 0."""
    assert eml_depth(sp.exp(x)) == 0


def test_eml_depth_one_for_single_eml(x: sp.Symbol) -> None:
    """A top-level EML node has depth 1."""
    assert eml_depth(eml(x, sp.Integer(1))) == 1


def test_eml_depth_nested(x: sp.Symbol) -> None:
    """eml(eml(x, 1), 1) has depth 2."""
    inner = eml(x, sp.Integer(1))
    assert eml_depth(eml(inner, sp.Integer(1))) == 2


# ---------------------------------------------------------------------------
# operator_count
# ---------------------------------------------------------------------------


def test_operator_count_polynomial(x: sp.Symbol) -> None:
    """x**2 + x has Add and Pow — at least 2 operators."""
    assert operator_count(x**2 + x) >= 2


def test_operator_count_atom_is_zero() -> None:
    """Atom has no operators."""
    assert operator_count(sp.Integer(1)) == 0


# ---------------------------------------------------------------------------
# weighted_cost
# ---------------------------------------------------------------------------


def test_weighted_cost_positive(x: sp.Symbol) -> None:
    """Weighted cost for any non-trivial expression should be > 0."""
    assert weighted_cost(eml(x, sp.Integer(1))) > 0


def test_weighted_cost_atom_is_zero() -> None:
    """Atom contributes zero to circuit cost."""
    assert weighted_cost(sp.Integer(1)) == 0


def test_weighted_cost_eml_more_than_exp(x: sp.Symbol) -> None:
    """EML node costs more than a plain exp node."""
    assert weighted_cost(eml(x, sp.Integer(1))) > weighted_cost(sp.exp(x))


# ---------------------------------------------------------------------------
# unique_subexpression_count
# ---------------------------------------------------------------------------


def test_unique_subexpression_count_atom() -> None:
    """An atom has exactly 1 unique subexpression."""
    assert unique_subexpression_count(sp.Integer(2)) == 1


def test_unique_subexpression_count_exp(x: sp.Symbol) -> None:
    """exp(x) has 2 unique subexpressions: exp(x) and x."""
    assert unique_subexpression_count(sp.exp(x)) == 2


# ---------------------------------------------------------------------------
# extract_log_arguments
# ---------------------------------------------------------------------------


def test_extract_log_arguments_empty(x: sp.Symbol) -> None:
    """No log nodes → empty list."""
    assert extract_log_arguments(sp.exp(x)) == []


def test_extract_log_arguments_single(x: sp.Symbol) -> None:
    """log(x + 1) should yield [x + 1] as the log argument."""
    args = extract_log_arguments(sp.log(x + 1))
    assert len(args) == 1
    assert args[0].equals(x + 1)


def test_extract_log_arguments_multiple(x: sp.Symbol) -> None:
    """log(x) + log(x + 1) should yield two log arguments."""
    args = extract_log_arguments(sp.log(x) + sp.log(x + 1))
    assert len(args) == 2


# ---------------------------------------------------------------------------
# Unified metrics interface
# ---------------------------------------------------------------------------


def test_compute_metrics_keys(x: sp.Symbol) -> None:
    """compute_metrics returns a dict with all six required keys."""
    expr = exp_eml(x)  # eml(x, 1)
    result = compute_metrics(expr)
    required_keys = {
        "node_count",
        "depth",
        "nonlinear_nodes",
        "eml_nodes",
        "unique_subexpressions",
        "weighted_cost",
    }
    assert required_keys == set(result.keys())
    assert all(isinstance(v, int) for v in result.values())


def test_compute_metrics_consistency(x: sp.Symbol) -> None:
    """compute_metrics values match individually called metric functions."""
    expr = exp_eml(x)  # eml(x, 1)
    result = compute_metrics(expr)
    assert result["node_count"] == total_node_count(expr)
    assert result["depth"] == tree_depth(expr)
    assert result["nonlinear_nodes"] == nonlinear_node_count(expr)
    assert result["eml_nodes"] == eml_node_count(expr)
    assert result["unique_subexpressions"] == unique_subexpression_count(expr)
    assert result["weighted_cost"] == weighted_cost(expr)
