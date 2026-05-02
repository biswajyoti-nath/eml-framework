"""Tests for the domain validation module (eml.validation).

Covers:
- assert_domain() raises DomainError for unsafe expressions
- assert_domain() passes for provably safe expressions
- is_domain_safe() detailed breakdown
- validate_domain() combined dict output
- validate_log_domain() numeric checks
- validate_transformed_domain() full EML well-definedness
- is_positive_expr() symbolic positivity tests
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from eml import (
    DomainError,
    assert_domain,
    is_domain_safe,
    is_positive_expr,
    validate_domain,
    validate_log_domain,
    validate_transformed_domain,
)
from eml.transform import transform

# ---------------------------------------------------------------------------
# assert_domain
# ---------------------------------------------------------------------------


def test_assert_domain_raises_on_log_negative(x: sp.Symbol) -> None:
    """assert_domain must raise DomainError for log with non-positive argument."""
    expr = sp.log(x - 5)  # x - 5 not provably positive
    with pytest.raises(DomainError, match="log argument"):
        assert_domain(expr)


def test_assert_domain_passes_for_exp(x: sp.Symbol) -> None:
    """assert_domain should not raise for exp(x) — no log arguments."""
    assert_domain(sp.exp(x))  # must not raise


def test_assert_domain_passes_for_log_of_exp(x: sp.Symbol) -> None:
    """log(exp(x)) has a provably positive argument — must not raise."""
    assert_domain(sp.log(sp.exp(x)))  # exp(x) > 0 always


def test_assert_domain_passes_for_polynomial(x: sp.Symbol) -> None:
    """Pure polynomials contain no log/Pow hazards — must not raise."""
    assert_domain(x**3 + 2 * x - 1)


def test_assert_domain_raises_on_noninteger_pow_nonpositive_base(x: sp.Symbol) -> None:
    """x**(1/2) has a non-positive base and must raise DomainError."""
    with pytest.raises(DomainError, match="power base"):
        assert_domain(x ** sp.Rational(1, 2))


def test_assert_domain_message_lists_unsafe_log_args(x: sp.Symbol) -> None:
    """DomainError message should contain detail about the unsafe argument."""
    expr = sp.log(x - 10)
    with pytest.raises(DomainError) as exc_info:
        assert_domain(expr)
    assert "log argument" in str(exc_info.value).lower() or "unsafe" in str(exc_info.value).lower()


# ---------------------------------------------------------------------------
# is_domain_safe
# ---------------------------------------------------------------------------


def test_is_domain_safe_returns_false_for_unbounded_log(x: sp.Symbol) -> None:
    """log(x) with unbounded x is not provably safe."""
    safe, conditions = is_domain_safe(sp.log(x))
    assert safe is False
    assert conditions["log_safe"] is False
    assert len(conditions["log_args"]) == 1


def test_is_domain_safe_returns_true_for_log_of_exp(x: sp.Symbol) -> None:
    """log(exp(x)) is provably safe since exp(x) > 0."""
    safe, conditions = is_domain_safe(sp.log(sp.exp(x)))
    assert safe is True
    assert conditions["log_safe"] is True


def test_is_domain_safe_tracks_noninteger_pow_bases(x: sp.Symbol) -> None:
    """Non-integer power base should appear in conditions dict."""
    safe, conditions = is_domain_safe(x ** sp.Rational(1, 2))
    assert safe is False
    assert conditions["pow_safe"] is False
    assert len(conditions["nonint_pow_bases"]) == 1


# ---------------------------------------------------------------------------
# validate_domain (combined)
# ---------------------------------------------------------------------------


def test_validate_domain_returns_all_keys(x: sp.Symbol) -> None:
    """validate_domain result must contain the expected dict keys."""
    result = validate_domain(sp.exp(x), x)
    assert {"valid_domain", "symbolic_safe", "numeric_ok", "conditions"} <= result.keys()


def test_validate_domain_ok_for_log_plus_two_numeric(
    x: sp.Symbol, positive_sample_points: np.ndarray
) -> None:
    """log(x + 2) on [0.1, 5.0] is numerically safe (x+2 > 0), but may not be
    symbolically proven positive since x is unbounded in SymPy's assumption
    system.  Verify numeric_ok is True.
    """
    result = validate_domain(sp.log(x + 2), x, sample_points=positive_sample_points)
    # x+2 > 0 on [0.1, 5.0] — numeric domain is safe even if not symbolically proven
    assert result["numeric_ok"] is True


# ---------------------------------------------------------------------------
# validate_log_domain (numeric)
# ---------------------------------------------------------------------------


def test_validate_log_domain_passes_when_no_logs(
    x: sp.Symbol, positive_sample_points: np.ndarray
) -> None:
    """Expressions without log nodes trivially pass log-domain validation."""
    assert validate_log_domain(sp.exp(x), x, sample_points=positive_sample_points) is True


def test_validate_log_domain_detects_non_positive(x: sp.Symbol) -> None:
    """log(x - 1) on [-1, 2] includes x ≤ 1 — should return False."""
    xs = np.linspace(-1.0, 2.0, 50)
    assert validate_log_domain(sp.log(x - 1), x, sample_points=xs) is False


def test_validate_log_domain_passes_for_log_x_plus_constant(
    x: sp.Symbol, positive_sample_points: np.ndarray
) -> None:
    """log(x + 3) on [0.1, 5.0] is always positive — should return True."""
    assert validate_log_domain(sp.log(x + 3), x, sample_points=positive_sample_points) is True


# ---------------------------------------------------------------------------
# validate_transformed_domain
# ---------------------------------------------------------------------------


def test_validate_transformed_domain_passes_for_exp(
    x: sp.Symbol, positive_sample_points: np.ndarray
) -> None:
    """exp(x) transforms to an EML form that is numerically well-defined."""
    eml_expr = transform(sp.exp(x))
    assert validate_transformed_domain(eml_expr, x, sample_points=positive_sample_points)


def test_validate_transformed_domain_fails_for_sqrt_on_negatives(x: sp.Symbol) -> None:
    """x**(1/2) is rejected by transform itself (non-positive base), so the
    test verifies that transform raises before domain validation is reached.
    """
    with pytest.raises(NotImplementedError, match="positive base"):
        transform(x ** sp.Rational(1, 2))


# ---------------------------------------------------------------------------
# is_positive_expr
# ---------------------------------------------------------------------------


def test_is_positive_expr_for_exp(x: sp.Symbol) -> None:
    """exp(x) is always positive."""
    assert is_positive_expr(sp.exp(x)) is True


def test_is_positive_expr_for_positive_constant() -> None:
    """Positive integer constants are provably positive."""
    assert is_positive_expr(sp.Integer(3)) is True


def test_is_positive_expr_for_zero_returns_false() -> None:
    """Zero is not strictly positive."""
    assert is_positive_expr(sp.Integer(0)) is False


def test_is_positive_expr_for_abs_returns_false(x: sp.Symbol) -> None:
    """Abs(x) is non-negative but not strictly positive (can be 0)."""
    assert is_positive_expr(sp.Abs(x)) is False
