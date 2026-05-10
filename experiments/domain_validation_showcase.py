"""Domain validation showcase: Real examples of domain violations and diagnostic catchery.

This module demonstrates the domain validation system's ability to detect and report
domain violations in symbolic expressions. It includes:

- Real examples of expressions violating domain assumptions (negative log arguments, invalid power bases)
- How validation catches them with diagnostic information
- Examples that pass validation
- Statistical summary of validation patterns

Run with::

    python -m experiments.domain_validation_showcase

Results demonstrate that domain validation is non-trivial and catches real errors.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import sympy as sp

from eml import configure_logging
from eml.exceptions import DomainError
from eml.validation import (
    is_positive_expr,
    is_domain_safe,
    validate_domain,
    validate_log_domain,
)

logger = logging.getLogger(__name__)

__all__ = [
    "domain_violation_examples",
    "domain_safe_examples",
    "run_validation_showcase",
]


# ---------------------------------------------------------------------------
# Domain Violation Examples
# ---------------------------------------------------------------------------


def domain_violation_examples() -> list[dict[str, Any]]:
    """Return expressions that violate domain assumptions.

    EML framework assumes:
    - All log arguments are strictly positive
    - Non-integer power bases are strictly positive

    These violations may cause runtime errors or silent numerical failures.

    Returns
    -------
    list[dict]
        Each entry: name, expr (sp.Expr), violation_type (str), explanation (str).
    """
    x = sp.symbols("x")

    return [
        {
            "name": "log(x - 2)",
            "expr": sp.log(x - 2),
            "violation_type": "negative_log_argument",
            "explanation": (
                "For x ∈ [0, 1], argument is negative. exp-log transformation would "
                "either fail numerically or produce NaN/Inf values."
            ),
            "test_range": (0.0, 1.0),
        },
        {
            "name": "log(-x)",
            "expr": sp.log(-x),
            "violation_type": "negative_log_argument",
            "explanation": (
                "For x > 0, the argument is negative. Domain of log requires y > 0."
            ),
            "test_range": (0.1, 3.0),
        },
        {
            "name": "log(sin(x))",
            "expr": sp.log(sp.sin(x)),
            "violation_type": "oscall_argument",
            "explanation": (
                "sin(x) oscillates; frequently negative or zero. log(negative) is undefined. "
                "sin(0)=0 makes log(sin(0)) undefined."
            ),
            "test_range": (0.0, 2 * np.pi),
        },
        {
            "name": "(x - 1)^(1/2)",
            "expr": (x - 1) ** (sp.Rational(1, 2)),
            "violation_type": "negative_power_base",
            "explanation": (
                "For x ∈ [0, 0.5], base is negative. Fractional exponent of negative "
                "base produces complex values. Even-root convention doesn't apply in symbolic form."
            ),
            "test_range": (0.0, 0.5),
        },
        {
            "name": "(x - 2)^(1/3)",
            "expr": (x - 2) ** (sp.Rational(1, 3)),
            "violation_type": "negative_power_base_odd_root",
            "explanation": (
                "Odd root of negative is valid mathematically (e.g., ∛(-8) = -2). "
                "However, symbolic engines may not handle this correctly. "
                "For x ∈ [0, 1], base is negative."
            ),
            "test_range": (0.0, 1.0),
        },
        {
            "name": "log(x^2 - 1)",
            "expr": sp.log(x**2 - 1),
            "violation_type": "composite_negative",
            "explanation": (
                "For x ∈ [0, 1], we have |x| < 1 so x^2 - 1 < 0. "
                "log of negative is undefined."
            ),
            "test_range": (0.0, 1.0),
        },
        {
            "name": "log(1 - x^2)",
            "expr": sp.log(1 - x**2),
            "violation_type": "composite_negative",
            "explanation": (
                "Outside interval (-1, 1), we have 1 - x^2 < 0. log(negative) is undefined."
            ),
            "test_range": (2.0, 5.0),
        },
        {
            "name": "x * log(x) for x around 0",
            "expr": x * sp.log(x),
            "violation_type": "log_singularity",
            "explanation": (
                "As x → 0+, log(x) → -∞. Even though lim x*log(x) = 0 mathematically, "
                "numerical evaluation near 0 is numerically unstable."
            ),
            "test_range": (-0.1, 0.1),
        },
        {
            "name": "exp(100*x)",
            "expr": sp.exp(100 * x),
            "violation_type": "numerical_overflow",
            "explanation": (
                "For x > 0.1, exp(100*x) overflows float64 (max ~10^308). "
                "Not strictly a domain violation, but causes numerical failure."
            ),
            "test_range": (0.1, 1.0),
        },
    ]


# ---------------------------------------------------------------------------
# Domain-Safe Examples
# ---------------------------------------------------------------------------


def domain_safe_examples() -> list[dict[str, Any]]:
    """Return expressions that are domain-safe.

    These can be safely transformed to exp-log form.

    Returns
    -------
    list[dict]
        Each entry: name, expr (sp.Expr), explanation (str).
    """
    x = sp.symbols("x")

    return [
        {
            "name": "log(x) for x > 0",
            "expr": sp.log(x),
            "explanation": "Argument is guaranteed positive in domain.",
            "test_range": (0.1, 3.0),
        },
        {
            "name": "log(x + 2)",
            "expr": sp.log(x + 2),
            "explanation": "For x ∈ [0, 3], x+2 ∈ [2, 5], always positive.",
            "test_range": (0.0, 3.0),
        },
        {
            "name": "exp(x)",
            "expr": sp.exp(x),
            "explanation": "Always positive; no domain restrictions.",
            "test_range": (-2.0, 2.0),
        },
        {
            "name": "log(x) + x^2",
            "expr": sp.log(x) + x**2,
            "explanation": (
                "Composite: log(x) is safe for x > 0, x^2 is always safe. "
                "Addition is defined for both."
            ),
            "test_range": (0.1, 3.0),
        },
        {
            "name": "x^2 * log(x) for x > 0",
            "expr": x**2 * sp.log(x),
            "explanation": (
                "x^2 always non-negative, log(x) defined for x > 0. "
                "Product is well-defined in domain."
            ),
            "test_range": (0.1, 3.0),
        },
        {
            "name": "exp(x) * log(x) for x > 0",
            "expr": sp.exp(x) * sp.log(x),
            "explanation": (
                "exp(x) always positive, log(x) defined for x > 0. "
                "Product is well-defined."
            ),
            "test_range": (0.1, 3.0),
        },
        {
            "name": "log(1 + x^2)",
            "expr": sp.log(1 + x**2),
            "explanation": (
                "1 + x^2 ≥ 1 > 0 for all x. Always positive."
            ),
            "test_range": (-3.0, 3.0),
        },
        {
            "name": "(x + 1)^2",
            "expr": (x + 1)**2,
            "explanation": "Integer power; no domain restrictions.",
            "test_range": (-5.0, 5.0),
        },
    ]


# ---------------------------------------------------------------------------
# Validation Demonstration
# ---------------------------------------------------------------------------


def run_validation_showcase() -> None:
    """Run comprehensive validation showcase with diagnostics."""
    logger.info("=" * 80)
    logger.info("DOMAIN VALIDATION SHOWCASE")
    logger.info("=" * 80)
    logger.info(
        "This demonstrates the domain validation system's ability to detect "
        "violations and provide diagnostic information.\n"
    )

    # Part 1: Domain Violations
    logger.info("PART 1: DOMAIN VIOLATIONS")
    logger.info("-" * 80)

    violations = domain_violation_examples()
    caught = 0
    missed = 0

    for example in violations:
        logger.info("\n[%s]", example["name"])
        logger.info("  Type: %s", example["violation_type"])
        logger.info("  Explanation: %s", example["explanation"])
        logger.info("  Expression: %s", example["expr"])

        x = sp.symbols("x")

        # Symbolic check
        safe, report = is_domain_safe(example["expr"])
        if not safe:
            logger.info("  ✓ CAUGHT (symbolic): %s", report.get("reason", "domain violation"))
            caught += 1
        else:
            logger.info("  ? MISSED (symbolic): Symbolic check passed (may be detected numerically)")

        # Numeric check
        low, high = example["test_range"]
        xs = np.linspace(low, high, 50)

        result = validate_domain(example["expr"], x, xs)
        if not result["symbolic_safe"] or not result["numeric_ok"]:
            logger.info("  ✓ CAUGHT (numeric): %s", result.get("error_source", "evaluation failed"))
            caught += 1
        else:
            logger.info("  ? MISSED (numeric): Evaluation succeeded")
            missed += 1

        # Log conditions
        if result.get("conditions"):
            logger.info("  Conditions: %s", result["conditions"])

    logger.info("\nViolations Summary: %d caught, %d missed", caught, len(violations) - caught)

    # Part 2: Domain-Safe Examples
    logger.info("\n" + "=" * 80)
    logger.info("PART 2: DOMAIN-SAFE EXAMPLES")
    logger.info("-" * 80)

    safe_examples = domain_safe_examples()
    all_safe = 0

    for example in safe_examples:
        logger.info("\n[%s]", example["name"])
        logger.info("  Explanation: %s", example["explanation"])
        logger.info("  Expression: %s", example["expr"])

        x = sp.symbols("x")

        # Symbolic check
        safe, report = is_domain_safe(example["expr"])
        if safe:
            logger.info("  ✓ PASS (symbolic)")
            all_safe += 1
        else:
            logger.info("  ✗ REJECT (symbolic): %s", report.get("reason", "unknown"))

        # Numeric check
        low, high = example["test_range"]
        xs = np.linspace(low, high, 50)

        result = validate_domain(example["expr"], x, xs)
        if result["symbolic_safe"] and result["numeric_ok"]:
            logger.info("  ✓ PASS (numeric)")
            all_safe += 1
        else:
            logger.info("  ✗ REJECT (numeric): %s", result.get("error_source", "evaluation failed"))

    logger.info("\nSafe Examples Summary: %d / %d passed both checks", all_safe, len(safe_examples) * 2)

    # Part 3: Detailed Diagnostic Examples
    logger.info("\n" + "=" * 80)
    logger.info("PART 3: DETAILED DIAGNOSTICS FOR KEY CASES")
    logger.info("-" * 80)

    x = sp.symbols("x")

    # Case 1: log(x - 2)
    logger.info("\n[Case: log(x - 2) for x ∈ [0, 1]]")
    expr = sp.log(x - 2)
    logger.info("  Positivity of (x - 2): %s", is_positive_expr(x - 2))
    result = validate_domain(expr, x, np.linspace(0, 1, 20))
    logger.info("  Validation result: %s", result)

    # Case 2: log(1 + x^2)
    logger.info("\n[Case: log(1 + x^2) for x ∈ [-3, 3]]")
    expr = sp.log(1 + x**2)
    logger.info("  Positivity of (1 + x^2): %s", is_positive_expr(1 + x**2))
    result = validate_domain(expr, x, np.linspace(-3, 3, 20))
    logger.info("  Validation result: %s", result)

    # Case 3: Fractional power
    logger.info("\n[Case: (x - 1)^(1/2) for x ∈ [0, 0.5]]")
    expr = (x - 1) ** (sp.Rational(1, 2))
    logger.info("  Positivity of (x - 1): %s", is_positive_expr(x - 1))
    result = validate_domain(expr, x, np.linspace(0, 0.5, 20))
    logger.info("  Validation result: %s", result)

    logger.info("\n" + "=" * 80)
    logger.info("END OF SHOWCASE")
    logger.info("=" * 80)


if __name__ == "__main__":
    configure_logging()
    run_validation_showcase()
