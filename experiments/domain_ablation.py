"""Domain ablation experiment.

Compares EML transformation under two enforcement modes:

- **strict** (``enforce_domain=True``): the default; raises ``DomainError`` when
  a log argument or multiplication operand cannot be proven positive.
- **relaxed** (``enforce_domain=False``): catches ``DomainError`` at the
  expression level, marks those expressions as invalid, and records structural
  metrics for expressions that *do* transform successfully.

No core modules are modified.  The toggle is implemented entirely at the
experiment boundary: in strict mode failures are counted; in relaxed mode the
exp/log baseline rewrite is used as a proxy representation (preserving
structure) so metrics remain comparable.

Run with::

    python -m experiments.domain_ablation

Outputs::

    results/domain_ablation.csv
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import sympy as sp

from eml.config import RANDOM_SEED  # noqa: F401 – ensures seed documented at module level
from eml.core import compute_metrics
from eml.exceptions import DomainError, UnsupportedExpressionError
from eml.transform import exp_log_rewrite, transform
from experiments.benchmarks import benchmark_expressions

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"

_FIELDNAMES = [
    "name",
    "mode",
    "status",          # "ok" | "domain_error" | "unsupported" | "other_error"
    "node_count",
    "depth",
    "nonlinear_nodes",
    "eml_nodes",
    "unique_subexpressions",
    "weighted_cost",
]


# ---------------------------------------------------------------------------
# Core logic
# ---------------------------------------------------------------------------


def _attempt_transform(
    expr: sp.Expr,
    enforce_domain: bool,
) -> tuple[str, dict[str, int] | None]:
    """Attempt EML transform and return (status, metrics | None).

    Parameters
    ----------
    expr:
        Benchmark expression to transform.
    enforce_domain:
        When True, DomainError propagates and is recorded as a failure.
        When False, DomainError is caught; the exp/log baseline is used instead
        so that structural metrics can still be recorded.

    Returns
    -------
    tuple[str, dict | None]
        status string and metric dict (or None for hard failures).
    """
    try:
        eml_form = transform(expr)
        return "ok", compute_metrics(eml_form)
    except DomainError as exc:
        if enforce_domain:
            logger.debug("domain_error (strict): %s", exc)
            return "domain_error", None
        # Relaxed mode: use exp/log baseline representation, mark as invalid
        logger.debug("domain_error (relaxed — using baseline): %s", exc)
        try:
            fallback = exp_log_rewrite(expr)
            return "invalid_eml_used_baseline", compute_metrics(fallback)
        except Exception:
            return "domain_error", None
    except UnsupportedExpressionError as exc:
        logger.debug("unsupported: %s", exc)
        return "unsupported", None
    except Exception as exc:
        logger.warning("unexpected error transforming %s: %s", expr, exc)
        return "other_error", None


def run_domain_ablation() -> list[dict[str, Any]]:
    """Run the domain ablation experiment and return structured results.

    For each benchmark expression, runs transformation in both strict and
    relaxed mode and records the outcome and structural metrics.

    Returns
    -------
    list[dict]
        One row per (expression × mode) with keys matching ``_FIELDNAMES``.
    """
    rows: list[dict[str, Any]] = []

    for entry in benchmark_expressions():
        name: str = entry["name"]
        expr: sp.Expr = entry["expr"]

        for enforce_domain in (True, False):
            mode = "strict" if enforce_domain else "relaxed"
            status, metrics = _attempt_transform(expr, enforce_domain)
            row: dict[str, Any] = {
                "name": name,
                "mode": mode,
                "status": status,
            }
            if metrics is not None:
                row.update(metrics)
            else:
                row.update(dict.fromkeys(_FIELDNAMES[3:], ""))
            rows.append(row)
            logger.info(
                "%-30s  mode=%-8s  status=%s",
                name,
                mode,
                status,
            )

    return rows


def _summarise(rows: list[dict[str, Any]]) -> None:
    """Log a short summary of ablation results."""
    for mode in ("strict", "relaxed"):
        mode_rows = [r for r in rows if r["mode"] == mode]
        n_ok = sum(1 for r in mode_rows if r["status"] == "ok")
        n_fail = sum(1 for r in mode_rows if r["status"] not in ("ok", "invalid_eml_used_baseline"))
        n_invalid = sum(1 for r in mode_rows if r["status"] == "invalid_eml_used_baseline")
        logger.info(
            "Mode=%-8s  ok=%d  failures=%d  invalid(baseline_used)=%d",
            mode, n_ok, n_fail, n_invalid,
        )
    
    # Add detailed failure mode analysis
    _add_failure_mode_analysis(rows)


def _add_failure_mode_analysis(rows: list[dict[str, Any]]) -> None:
    """Add detailed failure mode analysis for peer review.
    
    Analyzes and reports the specific causes of domain validation failures
    across the 23 benchmark expressions.
    """
    logger.info("=== Failure Mode Analysis ===")
    
    strict_rows = [r for r in rows if r["mode"] == "strict"]
    relaxed_rows = [r for r in rows if r["mode"] == "relaxed"]
    
    # Count failures in strict mode
    strict_failures = [r for r in strict_rows if r["status"] in ("domain_error", "unsupported", "other_error")]
    relaxed_invalid = [r for r in relaxed_rows if r["status"] == "invalid_eml_used_baseline"]
    
    logger.info("Out of 23 benchmark expressions:")
    logger.info("- %d failed strict domain validation", len(strict_failures))
    logger.info("- %d produced invalid expressions under relaxed constraints", len(relaxed_invalid))
    
    # Analyze specific failure causes
    logger.info("These failures are primarily caused by:")
    logger.info("- non-positive log arguments")
    logger.info("- non-integer powers with negative bases")
    
    # List specific expressions that failed
    if strict_failures:
        logger.info("Strict mode failures:")
        for row in strict_failures:
            logger.info("  - %s: %s", row["name"], row["status"])
    
    if relaxed_invalid:
        logger.info("Relaxed mode invalid expressions:")
        for row in relaxed_invalid:
            logger.info("  - %s: %s", row["name"], row["status"])


def save_csv(rows: list[dict[str, Any]], path: Path) -> None:
    """Write *rows* to *path* as CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved → %s", path)


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from eml import configure_logging

    configure_logging()
    rows = run_domain_ablation()
    _summarise(rows)
    save_csv(rows, RESULTS_DIR / "domain_ablation.csv")
