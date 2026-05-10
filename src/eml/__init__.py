"""EML framework — representation study of the Exp-Minus-Log operator.

Public API
----------
Core primitives:
    eml, exp_eml, log_eml, evaluate_eml

Complexity metrics:
    compute_metrics, total_node_count, nonlinear_node_count, tree_depth,
    eml_node_count, operator_count, eml_depth, weighted_cost,
    unique_subexpression_count, extract_log_arguments

Transform:
    transform, exp_log_rewrite

Domain validation:
    assert_domain, is_domain_safe, is_positive_expr,
    validate_domain, validate_log_domain, validate_transformed_domain

Exceptions:
    DomainError, UnsupportedExpressionError

Configuration:
    configure_logging  (call once at application entry-point)
"""

from .config import RANDOM_SEED
from .core import (
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
from .exceptions import DomainError, UnsupportedExpressionError
from .logging_config import configure_logging

from .transform import exp_log_rewrite, transform
from .validation import (
    assert_domain,
    is_domain_safe,
    is_positive_expr,
    validate_domain,
    validate_log_domain,
    validate_transformed_domain,
)

__all__ = [
    # Core primitives
    "EML",
    "eml",
    "exp_eml",
    "log_eml",
    "evaluate_eml",
    # Complexity metrics
    "compute_metrics",
    "total_node_count",
    "nonlinear_node_count",
    "tree_depth",
    "eml_node_count",
    "operator_count",
    "eml_depth",
    "weighted_cost",
    "unique_subexpression_count",
    "extract_log_arguments",
    # Transform
    "transform",
    "exp_log_rewrite",
    # Domain validation
    "assert_domain",
    "is_domain_safe",
    "is_positive_expr",
    "validate_domain",
    "validate_log_domain",
    "validate_transformed_domain",
    # Exceptions
    "DomainError",
    "UnsupportedExpressionError",
    # Config / logging
    "configure_logging",
    "RANDOM_SEED",
]
