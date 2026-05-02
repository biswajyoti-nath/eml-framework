"""Core EML primitives and symbolic complexity metrics.

This module defines the EML symbolic function, its exp/log encodings, and the
tree-level complexity metrics used to compare EML representations against
standard exp/log forms.

Supported fragment
------------------
The EML framework targets the *positive-domain exp-log fragment*: expressions
built from exp, log, Abs, Add, Mul, integer Pow, and derivatives.  Functions
outside this fragment (e.g. sin, gamma) are not supported.

Domain assumptions
------------------
- All log arguments must be strictly positive.
- Non-integer power bases must be strictly positive.
- These are *not* enforced here — see :mod:`eml.validation` for enforcement.
"""

from __future__ import annotations

import logging

import sympy as sp

from .config import (
    ABS_OPERATOR_COST,
    ADD_OPERATOR_COST,
    EML_OPERATOR_COST,
    EXP_LOG_OPERATOR_COST,
    MUL_OPERATOR_COST,
    POW_OPERATOR_COST,
)

logger = logging.getLogger(__name__)

# The symbolic EML function node.  SymPy treats it as an opaque function until
# evaluate_eml() substitutes the exp/log semantics.
EML: sp.core.function.UndefinedFunction = sp.Function("eml")

__all__ = [
    "EML",
    "eml",
    "exp_eml",
    "log_eml",
    "evaluate_eml",
    "total_node_count",
    "nonlinear_node_count",
    "tree_depth",
    "eml_node_count",
    "operator_count",
    "eml_depth",
    "weighted_cost",
    "unique_subexpression_count",
    "extract_log_arguments",
]


# ---------------------------------------------------------------------------
# EML primitive constructors
# ---------------------------------------------------------------------------


def eml(x: sp.Expr, y: sp.Expr) -> sp.Expr:
    """Return the symbolic EML primitive ``eml(x, y) = exp(x) - log(y)``.

    Parameters
    ----------
    x:
        Exponent argument.
    y:
        Log argument (must be positive in the domain).

    Returns
    -------
    sp.Expr
        An opaque ``eml(x, y)`` node; evaluate with :func:`evaluate_eml`.
    """
    return EML(x, y)  # type: ignore[return-value]


def exp_eml(x: sp.Expr) -> sp.Expr:
    """Encode ``exp(x)`` as an EML primitive.

    Semantics: ``exp_eml(x) = eml(x, 1) = exp(x) - log(1) = exp(x)``.

    Parameters
    ----------
    x:
        Exponent argument.
    """
    return eml(x, sp.Integer(1))


def log_eml(x: sp.Expr) -> sp.Expr:
    """Encode ``log(x)`` as an EML primitive.

    Semantics: ``log_eml(x) = 1 - eml(0, x) = 1 - (exp(0) - log(x)) = log(x)``.

    Parameters
    ----------
    x:
        Log argument (must be positive in the domain).
    """
    return sp.Integer(1) - eml(sp.Integer(0), x)


def evaluate_eml(expr: sp.Expr) -> sp.Expr:
    """Substitute all EML nodes with their exp/log semantics.

    Replaces every ``eml(x, y)`` node with ``exp(x) - log(y)`` and returns the
    resulting standard SymPy expression.

    Parameters
    ----------
    expr:
        EML-encoded expression (may contain ``eml(…)`` nodes).

    Returns
    -------
    sp.Expr
        Equivalent expression using only standard SymPy functions.
    """
    return expr.replace(
        lambda node: node.func == EML,
        lambda node: sp.exp(node.args[0]) - sp.log(node.args[1]),
    )


# ---------------------------------------------------------------------------
# Complexity metrics
# ---------------------------------------------------------------------------


def total_node_count(expr: sp.Expr) -> int:
    """Count all nodes in the symbolic expression tree.

    Parameters
    ----------
    expr:
        Any SymPy expression.

    Returns
    -------
    int
        Total number of nodes (atoms + internal nodes) in the tree.
    """
    return sum(1 for _ in sp.preorder_traversal(expr))


def nonlinear_node_count(expr: sp.Expr) -> int:
    """Count nonlinear exp-log nodes: ``exp``, ``log``, and EML primitives.

    Parameters
    ----------
    expr:
        Any SymPy expression.

    Returns
    -------
    int
        Number of nonlinear nodes in the tree.
    """
    return sum(1 for node in sp.preorder_traversal(expr) if node.func in {EML, sp.exp, sp.log})


def tree_depth(expr: sp.Expr) -> int:
    """Compute the maximum depth of the symbolic expression tree.

    Leaves (atoms with no children) have depth 0.

    Parameters
    ----------
    expr:
        Any SymPy expression.

    Returns
    -------
    int
        Maximum tree depth.
    """
    if not expr.args:
        return 0
    return 1 + max(tree_depth(arg) for arg in expr.args)


def eml_node_count(expr: sp.Expr) -> int:
    """Count EML primitive occurrences in the expression tree.

    Parameters
    ----------
    expr:
        Any SymPy expression (may contain ``eml(…)`` nodes).

    Returns
    -------
    int
        Number of ``eml(…)`` nodes.
    """
    return sum(1 for node in sp.preorder_traversal(expr) if node.func == EML)


def operator_count(expr: sp.Expr) -> int:
    """Count core symbolic operators within the supported exp-log fragment.

    Counts: Add, Mul, Pow, Derivative nodes.

    Parameters
    ----------
    expr:
        Any SymPy expression.

    Returns
    -------
    int
        Number of operator nodes.
    """
    return sum(
        1
        for node in sp.preorder_traversal(expr)
        if node.is_Add or node.is_Mul or node.is_Pow or node.is_Derivative
    )


def eml_depth(expr: sp.Expr) -> int:
    """Compute the maximum nesting depth of EML primitive calls.

    An EML node at the top level has depth 1. Each nested EML call inside
    another adds 1 to the depth.

    Parameters
    ----------
    expr:
        Any SymPy expression (may contain ``eml(…)`` nodes).

    Returns
    -------
    int
        Maximum EML nesting depth; 0 if no EML nodes are present.
    """
    if expr.func == EML:
        return 1 + max((eml_depth(arg) for arg in expr.args), default=0)
    if not expr.args:
        return 0
    return max(eml_depth(arg) for arg in expr.args)


def weighted_cost(expr: sp.Expr) -> int:
    """Estimate symbolic circuit cost with per-operator weights.

    Weights (from :mod:`eml.config`):

    * EML primitive: :data:`~eml.config.EML_OPERATOR_COST`
    * exp / log:     :data:`~eml.config.EXP_LOG_OPERATOR_COST`
    * Abs:           :data:`~eml.config.ABS_OPERATOR_COST`
    * Add:           :data:`~eml.config.ADD_OPERATOR_COST`
    * Mul:           :data:`~eml.config.MUL_OPERATOR_COST`
    * Pow:           :data:`~eml.config.POW_OPERATOR_COST`

    Parameters
    ----------
    expr:
        Any SymPy expression.

    Returns
    -------
    int
        Estimated total circuit cost.
    """
    cost = 0
    for node in sp.preorder_traversal(expr):
        if node.func == EML:
            cost += EML_OPERATOR_COST
        elif node.func in {sp.exp, sp.log}:
            cost += EXP_LOG_OPERATOR_COST
        elif node.func == sp.Abs:
            cost += ABS_OPERATOR_COST
        elif node.is_Add:
            cost += ADD_OPERATOR_COST
        elif node.is_Mul:
            cost += MUL_OPERATOR_COST
        elif node.is_Pow:
            cost += POW_OPERATOR_COST
    return cost


def unique_subexpression_count(expr: sp.Expr) -> int:
    """Count unique subexpressions in the symbolic expression tree.

    Two nodes are considered equal if they are structurally identical as SymPy
    objects (using SymPy's built-in equality / hashing).

    Parameters
    ----------
    expr:
        Any SymPy expression.

    Returns
    -------
    int
        Number of distinct subexpressions.
    """
    return len(set(sp.preorder_traversal(expr)))


def extract_log_arguments(expr: sp.Expr) -> list[sp.Expr]:
    """Collect all direct log arguments from the expression tree.

    Parameters
    ----------
    expr:
        Any SymPy expression.

    Returns
    -------
    list[sp.Expr]
        List of arguments ``a`` for each ``log(a)`` node found.
    """
    return [node.args[0] for node in sp.preorder_traversal(expr) if node.func == sp.log]


