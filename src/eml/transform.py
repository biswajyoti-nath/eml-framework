"""Symbolic transformation from standard exp-log form to EML representation.

This module implements the core recursive rewrite that encodes expressions in
the EML exp-log fragment.  It also provides the standard exp/log baseline
rewrite for complexity comparison.

Supported operations
--------------------
- Atoms (symbols, numbers): preserved as-is.
- Derivative: evaluated (``doit()``) then transformed.
- Add: terms transformed independently; addition preserved as combinator.
- Mul: encoded via exp(log(a) + log(b)) identity — requires positive operands.
- Pow: integer powers use sign/|·| rewrite; non-integer requires positive base.
- exp, log, Abs: mapped to EML primitives or preserved.

Domain assumptions
------------------
- Multiplication transform requires provably positive operands.
  Violation raises :class:`~eml.exceptions.DomainError`.
- Non-integer power transform requires a provably positive base.
  Violation raises :class:`NotImplementedError` (by design — caller must
  restrict the domain or use Abs rewrite explicitly).

Unsupported functions
---------------------
Any SymPy function not in ``SUPPORTED_EML_FUNCS`` raises
:class:`~eml.exceptions.UnsupportedExpressionError`.
"""

from __future__ import annotations

import logging

import sympy as sp

from .core import exp_eml, log_eml
from .exceptions import DomainError, UnsupportedExpressionError
from .validation import is_positive_expr

logger = logging.getLogger(__name__)

#: The set of SymPy function types natively handled by the EML transformer.
SUPPORTED_EML_FUNCS: frozenset[type] = frozenset({sp.exp, sp.log, sp.Abs})

__all__ = [
    "transform",
    "exp_log_rewrite",
    "SUPPORTED_EML_FUNCS",
]


# ---------------------------------------------------------------------------
# Main recursive transformer
# ---------------------------------------------------------------------------


def transform(expr: sp.Expr) -> sp.Expr:
    """Recursively rewrite *expr* into the scoped EML exp-log form.

    This rewrite targets the EML exp-log fragment, not a general symbolic
    engine.  Addition is preserved as an ambient combinator, while
    multiplication, power, division, exp, log, and absolute value are encoded
    through EML-aware transformations.

    Power rewrites are intentionally limited to positive-domain or
    absolute-value cases for real-valued correctness.

    Parameters
    ----------
    expr:
        A SymPy expression built from the supported fragment.

    Returns
    -------
    sp.Expr
        EML-encoded equivalent expression.

    Raises
    ------
    DomainError
        When a multiplication operand cannot be proven positive.
    UnsupportedExpressionError
        When the expression contains a function outside the EML fragment.
    NotImplementedError
        When a non-integer power base cannot be proven positive.
    """
    if expr.is_Atom:
        return expr

    if expr.is_Derivative:
        logger.debug("transform: evaluating derivative before transform")
        return transform(expr.doit())

    if expr.is_Add:
        return transform_add([transform(arg) for arg in expr.args])

    if expr.is_Mul:
        return transform_mul([transform(arg) for arg in expr.args])

    if expr.is_Pow:
        return transform_pow(expr)

    if expr.func == sp.exp:
        return exp_eml(transform(expr.args[0]))

    if expr.func == sp.log:
        return log_eml(transform(expr.args[0]))

    if expr.func == sp.Abs:
        return sp.Abs(transform(expr.args[0]))

    return _transform_unsupported(expr)


# ---------------------------------------------------------------------------
# Operation-specific transformers
# ---------------------------------------------------------------------------


def transform_add(terms: list[sp.Expr]) -> sp.Expr:
    """Preserve addition as a standard combinator over transformed terms.

    Parameters
    ----------
    terms:
        Already-transformed additive terms.

    Returns
    -------
    sp.Expr
        Sum of all terms (standard SymPy Add).
    """
    return sp.Add(*terms)


def transform_mul(factors: list[sp.Expr]) -> sp.Expr:
    """Encode multiplication in exp-log form using EML primitives.

    Uses the identity ``a * b = exp(log(a) + log(b))``, which is only
    semantically valid when all operands are provably positive.

    Parameters
    ----------
    factors:
        Already-transformed multiplicative factors.

    Returns
    -------
    sp.Expr
        EML-encoded product.

    Raises
    ------
    DomainError
        When any factor cannot be proven positive under the EML domain
        assumptions.  Silent failures are not acceptable — the caller must
        guarantee positivity or restructure the expression.

    Notes
    -----
    Integer and rational constants (e.g. ``-1``, ``1/2``) often trigger this
    error when multiplied with EML terms.  The standard approach is to handle
    sign factors separately via :func:`transform_pow` before calling this.
    """
    for factor in factors:
        if not is_positive_expr(factor):
            raise DomainError(
                f"Multiplication transform requires all operands to be provably positive "
                f"under the EML domain assumptions, but got: {factor!r}. "
                "Restructure the expression or apply a sign/Abs rewrite first."
            )
    result = factors[0]
    for factor in factors[1:]:
        result = exp_eml(log_eml(result) + log_eml(factor))
    return result


def transform_pow(expr: sp.Expr) -> sp.Expr:
    """Encode a power expression using exp-log identities.

    Strategy:

    * Exponent is 0 → return 1.
    * Exponent is a non-zero integer → ``sign(base)^n * exp(n * log(|base|))``.
    * Exponent is non-integer → ``exp(exponent * log(base))``, requires
      positive base.

    Parameters
    ----------
    expr:
        A ``Pow`` node whose arguments will be decomposed.

    Returns
    -------
    sp.Expr
        EML-encoded power.

    Raises
    ------
    NotImplementedError
        When the exponent is non-integer and the base is not provably positive.
    """
    base, exponent = expr.args
    base_t: sp.Expr = transform(base)
    exponent_t: sp.Expr = transform(exponent)

    if exponent.is_Integer:
        if exponent == 0:
            return sp.Integer(1)
        sign_factor: sp.Expr = sp.sign(base_t) ** exponent
        return sign_factor * exp_eml(exponent_t * log_eml(sp.Abs(base_t)))

    if not is_positive_expr(base_t):
        raise NotImplementedError(
            f"Non-integer power transform requires a positive base under the EML "
            f"positive-domain exp-log fragment, but got base: {base_t!r}. "
            "Use sp.Abs(base) or restrict the domain to positive values."
        )

    return exp_eml(exponent_t * log_eml(base_t))


def transform_abs(expr: sp.Expr) -> sp.Expr:
    """Preserve absolute value when encoding EML expressions.

    Parameters
    ----------
    expr:
        Already-transformed argument.

    Returns
    -------
    sp.Expr
        ``Abs(expr)``.
    """
    return sp.Abs(expr)


def _transform_unsupported(expr: sp.Expr) -> sp.Expr:
    """Fallback for functions not in the EML fragment.

    Raises :class:`~eml.exceptions.UnsupportedExpressionError` for any
    function outside :data:`SUPPORTED_EML_FUNCS`.  This is intentional — silent
    pass-through would produce semantically incorrect EML representations.

    Parameters
    ----------
    expr:
        Expression node whose function is not handled by the main dispatcher.

    Raises
    ------
    UnsupportedExpressionError
        Always, unless the function happens to be in ``SUPPORTED_EML_FUNCS``
        (in which case it recurses with :func:`transform`).
    """
    if expr.func not in SUPPORTED_EML_FUNCS:
        raise UnsupportedExpressionError(
            f"Function '{expr.func}' is not supported in the EML exp-log fragment. "
            "Supported functions: exp, log, Abs, Add, Mul, Pow, Derivative."
        )
    logger.debug("_transform_unsupported: recursing into supported func %s", expr.func)
    return expr.func(*[transform(arg) for arg in expr.args])


# ---------------------------------------------------------------------------
# Baseline: standard exp/log rewrite (for comparison)
# ---------------------------------------------------------------------------


def exp_log_rewrite(expr: sp.Expr) -> sp.Expr:
    """Rewrite *expr* into standard exp/log form for baseline comparison.

    This is the non-EML baseline representation used for complexity comparison
    within the same exp-log fragment.  It is not a complete symbolic rewrite
    engine for arbitrary functions.

    Parameters
    ----------
    expr:
        Expression within the supported exp-log fragment.

    Returns
    -------
    sp.Expr
        Equivalent expression using standard SymPy exp/log (no EML nodes).
    """
    if expr.is_Atom:
        return expr

    if expr.is_Derivative:
        return exp_log_rewrite(expr.doit())

    if expr.is_Add:
        return sp.Add(*[exp_log_rewrite(arg) for arg in expr.args])

    if expr.is_Mul:
        return _exp_log_rewrite_mul([exp_log_rewrite(arg) for arg in expr.args])

    if expr.is_Pow:
        return _exp_log_rewrite_pow(expr)

    if expr.func in {sp.exp, sp.log}:
        return expr.func(exp_log_rewrite(expr.args[0]))

    if expr.func == sp.Abs:
        return sp.Abs(exp_log_rewrite(expr.args[0]))

    return expr.func(*[exp_log_rewrite(arg) for arg in expr.args])


def _exp_log_rewrite_mul(factors: list[sp.Expr]) -> sp.Expr:
    """Encode a product list using ``exp(log(a) + log(b))`` (baseline).

    Parameters
    ----------
    factors:
        Already-rewritten factors.

    Returns
    -------
    sp.Expr
        Baseline exp/log encoding of the product.
    """
    result = factors[0]
    for factor in factors[1:]:
        result = sp.exp(sp.log(result) + sp.log(factor))
    return result


def _exp_log_rewrite_pow(expr: sp.Expr) -> sp.Expr:
    """Encode a power using exp/log identities (baseline).

    Parameters
    ----------
    expr:
        A ``Pow`` node.

    Returns
    -------
    sp.Expr
        Baseline exp/log encoding of the power.
    """
    base, exponent = expr.args
    base_t: sp.Expr = exp_log_rewrite(base)
    exponent_t: sp.Expr = exp_log_rewrite(exponent)

    if exponent.is_Integer:
        if exponent == 0:
            return sp.Integer(1)
        if exponent < 0:
            return sp.exp(exponent_t * sp.log(base_t))
        if exponent % 2 == 0:
            return sp.exp(exponent_t * sp.log(sp.Abs(base_t)))
        return _exp_log_rewrite_mul([base_t, sp.exp((exponent - 1) * sp.log(sp.Abs(base_t)))])

    return sp.exp(exponent_t * sp.log(base_t))
