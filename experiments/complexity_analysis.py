"""Complexity analysis experiment: EML vs standard exp/log representation.

Computes and plots structural complexity metrics for the canonical Schrödinger
test case and the full benchmark expression suite.

Run with::

    python -m experiments.complexity_analysis

Output:
    - Logs structural complexity table to the logger.
    - Saves ``experiments/representation_comparison.png``.
    - Returns structured results as a list of dicts.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from eml import (
    configure_logging,
    eml_depth,
    eml_node_count,
    nonlinear_node_count,
    operator_count,
    total_node_count,
    tree_depth,
    unique_subexpression_count,
)
from eml.schrodinger import original_schrodinger
from eml.transform import exp_log_rewrite, transform
from experiments.benchmarks import benchmark_expressions

logger = logging.getLogger(__name__)

_PLOT_OUTPUT = Path(__file__).parent / "representation_comparison.png"


def complexity_analysis() -> list[dict[str, Any]]:
    """Run the structural complexity analysis and generate the comparison plot.

    Computes node counts, depth, and unique subexpression counts for each
    benchmark expression in both EML and baseline exp/log representations.

    Returns
    -------
    list[dict]
        One entry per benchmark expression with keys:
        ``name``, ``original_nodes``, ``original_ops``, ``original_depth``,
        ``original_nonlinear``, ``eml_nodes``, ``eml_primitives``,
        ``eml_nonlinear``, ``eml_depth``, ``eml_unique_subexpressions``,
        ``exp_log_nodes``, ``exp_log_nonlinear``, ``exp_log_depth``,
        ``exp_log_unique_subexpressions``.
    """
    operator = original_schrodinger()
    eml_operator = transform(operator)
    exp_log_operator = exp_log_rewrite(operator)

    logger.info("=== Schrödinger structural complexity ===")
    logger.info("Original operator count:      %d", operator_count(operator))
    logger.info("Original total node count:    %d", total_node_count(operator))
    logger.info("EML primitive count:          %d", eml_node_count(eml_operator))
    logger.info("EML nesting depth:            %d", eml_depth(eml_operator))
    logger.info("EML nonlinear node count:     %d", nonlinear_node_count(eml_operator))
    logger.info("EML tree depth:               %d", tree_depth(eml_operator))
    logger.info("EML unique subexpressions:    %d", unique_subexpression_count(eml_operator))
    logger.info("exp/log nonlinear node count: %d", nonlinear_node_count(exp_log_operator))

    benchmark_rows: list[dict[str, Any]] = []
    for entry in benchmark_expressions():
        expr = entry["expr"]
        transformed = transform(expr)
        exp_log = exp_log_rewrite(expr)
        benchmark_rows.append(
            {
                "name": entry["name"],
                "original_nodes": total_node_count(expr),
                "original_ops": operator_count(expr),
                "original_depth": tree_depth(expr),
                "original_nonlinear": nonlinear_node_count(expr),
                "eml_nodes": total_node_count(transformed),
                "eml_primitives": eml_node_count(transformed),
                "eml_nonlinear": nonlinear_node_count(transformed),
                "eml_depth": tree_depth(transformed),
                "eml_unique_subexpressions": unique_subexpression_count(transformed),
                "exp_log_nodes": total_node_count(exp_log),
                "exp_log_nonlinear": nonlinear_node_count(exp_log),
                "exp_log_depth": tree_depth(exp_log),
                "exp_log_unique_subexpressions": unique_subexpression_count(exp_log),
            }
        )

    _plot_comparison(benchmark_rows)
    return benchmark_rows


def _plot_comparison(rows: list[dict[str, Any]]) -> None:
    """Generate and save the representation comparison bar chart.

    Parameters
    ----------
    rows:
        Benchmark result rows from :func:`complexity_analysis`.
    """
    names = [row["name"] for row in rows]
    eml_nonlinear = [row["eml_nonlinear"] for row in rows]
    exp_log_nonlinear = [row["exp_log_nonlinear"] for row in rows]
    eml_unique = [row["eml_unique_subexpressions"] for row in rows]
    exp_log_unique = [row["exp_log_unique_subexpressions"] for row in rows]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    width = 0.35
    x_pos = range(len(names))

    axes[0].bar([i - width / 2 for i in x_pos], exp_log_nonlinear, width, label="exp/log")
    axes[0].bar([i + width / 2 for i in x_pos], eml_nonlinear, width, label="EML")
    axes[0].set_title("Nonlinear exp-log node count")
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].set_xticks(list(x_pos))
    axes[0].set_xticklabels(names)
    axes[0].legend()

    axes[1].bar([i - width / 2 for i in x_pos], exp_log_unique, width, label="exp/log")
    axes[1].bar([i + width / 2 for i in x_pos], eml_unique, width, label="EML")
    axes[1].set_title("Unique subexpression count")
    axes[1].tick_params(axis="x", rotation=45)
    axes[1].set_xticks(list(x_pos))
    axes[1].set_xticklabels(names)
    axes[1].legend()

    plt.suptitle("Benchmark representation comparison")
    plt.tight_layout()
    plt.savefig(_PLOT_OUTPUT, dpi=150)
    logger.info("Saved benchmark representation plot to %s", _PLOT_OUTPUT)


if __name__ == "__main__":
    configure_logging()
    complexity_analysis()
