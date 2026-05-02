"""Schrödinger equation helpers for the EML representation.

Provides the canonical 1D Schrödinger operator and convenience constructors
for the test case ``psi(x) = exp(-x²)``, ``V(x) = x²``.

All module-level state is encapsulated inside functions — there are no
side-effecting imports.
"""

from __future__ import annotations

import logging

import sympy as sp

from .transform import transform

logger = logging.getLogger(__name__)

__all__ = [
    "schrodinger_operator",
    "original_schrodinger",
    "eml_schrodinger",
]


def schrodinger_operator(
    psi_expr: sp.Expr,
    potential_expr: sp.Expr,
    variable: sp.Symbol,
) -> sp.Expr:
    """Build the 1D Schrödinger operator ``H[psi] = -psi'' + V * psi``.

    Parameters
    ----------
    psi_expr:
        Wave-function expression (e.g. ``exp(-x**2)``).
    potential_expr:
        Potential energy expression (e.g. ``x**2``).
    variable:
        The spatial variable to differentiate with respect to.

    Returns
    -------
    sp.Expr
        Symbolic expression for ``H[psi]``, with the second derivative
        evaluated (``doit()`` called).

    Notes
    -----
    The derivative is evaluated eagerly so the result is a standard
    expression tree that the EML transformer can process directly.
    """
    second_deriv: sp.Expr = sp.diff(psi_expr, variable, 2)
    return -second_deriv + potential_expr * psi_expr


def original_schrodinger() -> sp.Expr:
    """Return the symbolic Schrödinger expression for the canonical test case.

    Uses ``psi(x) = exp(-x²)`` and ``V(x) = x²`` with the real variable ``x``.

    Returns
    -------
    sp.Expr
        ``H[psi] = -d²/dx²[exp(-x²)] + x² * exp(-x²)`` in simplified form.
    """
    x: sp.Symbol = sp.symbols("x")
    psi: sp.Expr = sp.exp(-(x**2))
    potential: sp.Expr = x**2
    expr = schrodinger_operator(psi, potential, x)
    logger.debug("original_schrodinger: %s", expr)
    return expr


def eml_schrodinger() -> sp.Expr:
    """Return the Schrödinger expression rewritten with the EML transformer.

    Applies :func:`~eml.transform.transform` to :func:`original_schrodinger`.

    Returns
    -------
    sp.Expr
        EML-encoded Schrödinger operator expression.
    """
    expr = original_schrodinger()
    result = transform(expr)
    logger.debug("eml_schrodinger: %s", result)
    return result
