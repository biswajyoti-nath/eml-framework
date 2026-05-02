"""Generate all figures used in the paper.

Run from the repository root::

    python paper/generate_figures.py

Outputs (saved to paper/figures/):
    complexity_comparison.png  — nonlinear node count, unique subexpressions, tree depth
    regression_results.png     — symbolic regression MSE comparison (EML vs baseline)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

matplotlib.use("Agg")

# ── resolve package root ──────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("generate_figures")

FIGURE_DIR = Path(__file__).parent / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

# ── style constants ───────────────────────────────────────────────────────────
BLUE = "#2563EB"  # EML
AMBER = "#D97706"  # exp/log baseline

plt.rcParams.update(
    {
        "font.family": "serif",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
    }
)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Structural complexity comparison
# ─────────────────────────────────────────────────────────────────────────────


def generate_complexity_figure() -> None:
    """Generate and save the structural complexity comparison figure."""
    from eml import nonlinear_node_count, tree_depth, unique_subexpression_count
    from eml.exceptions import DomainError, UnsupportedExpressionError
    from eml.transform import exp_log_rewrite, transform
    from experiments.benchmarks import benchmark_expressions

    names: list[str] = []
    eml_nl: list[int] = []
    exp_nl: list[int] = []
    eml_uniq: list[int] = []
    exp_uniq: list[int] = []
    eml_depth: list[int] = []
    exp_depth: list[int] = []

    for entry in benchmark_expressions():
        try:
            eml_form = transform(entry["expr"])
            explog_form = exp_log_rewrite(entry["expr"])
        except (DomainError, UnsupportedExpressionError, NotImplementedError) as exc:
            logger.warning("Skipped %s: %s", entry["name"], exc)
            continue

        names.append(entry["name"].replace("_", "\n"))
        eml_nl.append(nonlinear_node_count(eml_form))
        exp_nl.append(nonlinear_node_count(explog_form))
        eml_uniq.append(unique_subexpression_count(eml_form))
        exp_uniq.append(unique_subexpression_count(explog_form))
        eml_depth.append(tree_depth(eml_form))
        exp_depth.append(tree_depth(explog_form))

    n = len(names)
    x = np.arange(n)
    w = 0.38

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle(
        "EML vs exp/log baseline — structural representation metrics",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    for ax, y_eml, y_exp, title, ylabel in [
        (axes[0], eml_nl, exp_nl, "Nonlinear node count", "Count"),
        (axes[1], eml_uniq, exp_uniq, "Unique subexpressions", "Count"),
        (axes[2], eml_depth, exp_depth, "Tree depth", "Depth"),
    ]:
        ax.bar(x - w / 2, y_exp, w, label="exp/log baseline", color=AMBER, alpha=0.85)
        ax.bar(x + w / 2, y_eml, w, label="EML", color=BLUE, alpha=0.85)
        ax.set_title(title, fontweight="bold")
        ax.set_ylabel(ylabel)
        ax.set_xticks(x)
        ax.set_xticklabels(names, fontsize=7, rotation=45, ha="right")
        ax.legend(fontsize=8)

    plt.tight_layout()
    out = FIGURE_DIR / "complexity_comparison.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Symbolic regression MSE comparison
# ─────────────────────────────────────────────────────────────────────────────


def generate_regression_figure() -> None:
    """Generate and save the symbolic regression results figure."""
    from experiments.symbolic_regression import run_search

    logger.info("Running symbolic regression experiment (this may take ~60 s)…")
    results = run_search()

    targets = [r["target"] for r in results]
    eml_mse = [r["eml_mse"] for r in results]
    baseline_mse = [r["baseline_mse"] for r in results]
    eml_size = [r["eml_tree_size"] if r["eml_tree_size"] is not None else 0 for r in results]
    base_size = [
        r["baseline_tree_size"] if r["baseline_tree_size"] is not None else 0 for r in results
    ]

    n = len(targets)
    x = np.arange(n)
    w = 0.38
    labels = [t.replace(" + ", "+\n").replace(" * ", "×\n") for t in targets]

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(
        "Evolutionary symbolic regression — EML vs exp/log baseline",
        fontsize=13,
        fontweight="bold",
        y=1.02,
    )

    # Panel 1: final MSE (log scale)
    ax = axes[0]
    ax.bar(x - w / 2, baseline_mse, w, label="exp/log baseline", color=AMBER, alpha=0.85)
    ax.bar(x + w / 2, eml_mse, w, label="EML", color=BLUE, alpha=0.85)
    ax.set_yscale("log")
    ax.set_title("Final MSE (lower is better)", fontweight="bold")
    ax.set_ylabel("MSE (log scale)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax.legend(fontsize=8)
    ax.yaxis.set_minor_formatter(ticker.NullFormatter())

    # Panel 2: best-found tree size
    ax = axes[1]
    ax.bar(x - w / 2, base_size, w, label="exp/log baseline", color=AMBER, alpha=0.85)
    ax.bar(x + w / 2, eml_size, w, label="EML", color=BLUE, alpha=0.85)
    ax.set_title("Best-found tree size (node count)", fontweight="bold")
    ax.set_ylabel("Nodes")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=8, rotation=30, ha="right")
    ax.legend(fontsize=8)

    plt.tight_layout()
    out = FIGURE_DIR / "regression_results.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", out)


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("=== Generating paper figures ===")
    generate_complexity_figure()
    generate_regression_figure()
    logger.info("=== Done. Figures written to %s ===", FIGURE_DIR)
