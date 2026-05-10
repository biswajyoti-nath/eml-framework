"""Depth/size vs error analysis and success rate metrics.

Uses the symbolic regression experiment results to:

1. Build datasets correlating tree depth and node count with final MSE.
2. Plot scatter charts: error vs depth and error vs node count.
3. Compute per-grammar success rates (exact recovery: MSE < 1e-10).

No SR logic is duplicated — all data comes from
:func:`experiments.symbolic_regression.run_search`.

Run with::

    python -m experiments.depth_size_analysis

Outputs::

    results/depth_vs_error.csv
    results/size_vs_error.csv
    results/success_rates.csv
    figures/depth_vs_error.png
    figures/size_vs_error.png
"""

from __future__ import annotations

import csv
import logging
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt

matplotlib.use("Agg")

from eml.config import RANDOM_SEED  # noqa: F401
from experiments.symbolic_regression import run_search

logger = logging.getLogger(__name__)

RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"

#: MSE threshold below which a recovery is considered exact.
EXACT_THRESHOLD: float = 1e-10

_BLUE = "#2563EB"
_AMBER = "#D97706"


# ---------------------------------------------------------------------------
# Dataset builders
# ---------------------------------------------------------------------------


def build_analysis_datasets(
    sr_results: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Build depth/size/success datasets from symbolic regression results.

    Parameters
    ----------
    sr_results:
        Output of :func:`~experiments.symbolic_regression.run_search`.

    Returns
    -------
    depth_rows:
        One row per (target × grammar) with keys
        ``target``, ``grammar``, ``depth``, ``mse``.
    size_rows:
        One row per (target × grammar) with keys
        ``target``, ``grammar``, ``node_count``, ``mse``.
    success_rows:
        One row per grammar with keys
        ``grammar``, ``exact_count``, ``total``, ``success_rate_pct``,
        ``avg_node_count``.
    """
    depth_rows: list[dict[str, Any]] = []
    size_rows: list[dict[str, Any]] = []

    eml_exact = 0
    baseline_exact = 0
    eml_nodes: list[int] = []
    baseline_nodes: list[int] = []
    n_targets = len(sr_results)

    for result in sr_results:
        target = result["target"]

        eml_depth = result.get("eml_depth_mean")
        eml_size = result.get("eml_nodes_mean")
        eml_mse = result["eml_mse_mean"]
        base_depth = result.get("base_depth_mean")
        base_size = result.get("base_nodes_mean")
        base_mse = result["base_mse_mean"]

        if eml_depth is not None and eml_size is not None:
            depth_rows.append({"target": target, "grammar": "EML",
                                "depth": eml_depth, "mse": eml_mse})
            size_rows.append({"target": target, "grammar": "EML",
                               "node_count": eml_size, "mse": eml_mse})
            eml_nodes.append(eml_size)
        if base_depth is not None and base_size is not None:
            depth_rows.append({"target": target, "grammar": "baseline",
                                "depth": base_depth, "mse": base_mse})
            size_rows.append({"target": target, "grammar": "baseline",
                               "node_count": base_size, "mse": base_mse})
            baseline_nodes.append(base_size)

        if eml_mse < EXACT_THRESHOLD:
            eml_exact += 1
        if base_mse < EXACT_THRESHOLD:
            baseline_exact += 1

    success_rows: list[dict[str, Any]] = [
        {
            "grammar": "EML",
            "exact_count": eml_exact,
            "total": n_targets,
            "success_rate_pct": round(100.0 * eml_exact / n_targets, 1) if n_targets else 0.0,
            "avg_node_count": round(sum(eml_nodes) / len(eml_nodes), 1) if eml_nodes else "",
        },
        {
            "grammar": "baseline",
            "exact_count": baseline_exact,
            "total": n_targets,
            "success_rate_pct": round(100.0 * baseline_exact / n_targets, 1) if n_targets else 0.0,
            "avg_node_count": (
                round(sum(baseline_nodes) / len(baseline_nodes), 1) if baseline_nodes else ""
            ),
        },
    ]

    return depth_rows, size_rows, success_rows


# ---------------------------------------------------------------------------
# Plotting (separated from computation)
# ---------------------------------------------------------------------------


def _scatter(
    rows: list[dict[str, Any]],
    x_key: str,
    x_label: str,
    title: str,
    out_path: Path,
) -> None:
    """Save a scatter plot of MSE vs *x_key* coloured by grammar."""
    eml_rows = [r for r in rows if r["grammar"] == "EML"]
    base_rows = [r for r in rows if r["grammar"] == "baseline"]

    fig, ax = plt.subplots(figsize=(8, 5))
    if eml_rows:
        ax.scatter(
            [r[x_key] for r in eml_rows],
            [r["mse"] for r in eml_rows],
            color=_BLUE, label="EML", alpha=0.85, s=80, zorder=3,
        )
    if base_rows:
        ax.scatter(
            [r[x_key] for r in base_rows],
            [r["mse"] for r in base_rows],
            color=_AMBER, label="exp/log baseline", alpha=0.85, s=80, marker="^", zorder=3,
        )
    ax.set_xlabel(x_label)
    ax.set_ylabel("Final MSE")
    ax.set_title(title, fontweight="bold")
    ax.set_yscale("symlog", linthresh=1e-12)
    ax.legend()
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", out_path)


# ---------------------------------------------------------------------------
# CSV helpers
# ---------------------------------------------------------------------------


def _write_csv(rows: list[dict[str, Any]], path: Path, fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    logger.info("Saved → %s", path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def run_depth_size_analysis(
    sr_results: list[dict[str, Any]] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Run the depth/size vs error analysis and success rate computation.

    Parameters
    ----------
    sr_results:
        Pre-computed symbolic regression results.  If None, calls
        :func:`~experiments.symbolic_regression.run_search` automatically.

    Returns
    -------
    dict with keys ``depth_rows``, ``size_rows``, ``success_rows``.
    """
    if sr_results is None:
        logger.info("Running symbolic regression (this may take ~60 s)…")
        sr_results = run_search()

    depth_rows, size_rows, success_rows = build_analysis_datasets(sr_results)

    # CSVs
    _write_csv(
        depth_rows, RESULTS_DIR / "depth_vs_error.csv",
        ["target", "grammar", "depth", "mse"],
    )
    _write_csv(
        size_rows, RESULTS_DIR / "size_vs_error.csv",
        ["target", "grammar", "node_count", "mse"],
    )
    _write_csv(
        success_rows, RESULTS_DIR / "success_rates.csv",
        ["grammar", "exact_count", "total", "success_rate_pct", "avg_node_count"],
    )

    # Plots
    _scatter(
        depth_rows, "depth", "Tree depth",
        "Final MSE vs tree depth — EML vs baseline",
        FIGURES_DIR / "depth_vs_error.png",
    )
    _scatter(
        size_rows, "node_count", "Node count",
        "Final MSE vs node count — EML vs baseline",
        FIGURES_DIR / "size_vs_error.png",
    )

    # Log success rate summary
    for row in success_rows:
        logger.info(
            "Grammar=%-10s  exact=%d/%d  (%.1f%%)  avg_nodes=%s",
            row["grammar"], row["exact_count"], row["total"],
            row["success_rate_pct"], row["avg_node_count"],
        )

    return {"depth_rows": depth_rows, "size_rows": size_rows, "success_rows": success_rows}


if __name__ == "__main__":
    from eml import configure_logging

    configure_logging()
    run_depth_size_analysis()
