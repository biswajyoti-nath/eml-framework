"""Experiment orchestrator — runs all analyses and generates results/summary.md.

Runs (in order):
1. Domain ablation              → results/domain_ablation.csv
2. Symbolic regression          → results/symbolic_regression_per_target.csv
3. Depth/size vs error          → results/depth_vs_error.csv, results/size_vs_error.csv
                                  results/success_rates.csv
                                  figures/depth_vs_error.png, figures/size_vs_error.png
4. Domain validation showcase   → Console report with diagnostic analysis
5. Summary                      → results/summary.md

Run with::

    python -m experiments.run_all

All outputs are deterministic (RANDOM_SEED = 42).
"""

from __future__ import annotations

import logging
import textwrap
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from eml import configure_logging
from eml.config import RANDOM_SEED, SEARCH_BUDGET, SEARCH_MAX_DEPTH, SEARCH_POP_SIZE
from experiments.depth_size_analysis import (
    EXACT_THRESHOLD,
    FIGURES_DIR,
    RESULTS_DIR,
    run_depth_size_analysis,
)
from experiments.domain_ablation import run_domain_ablation
from experiments.domain_ablation import save_csv as save_ablation_csv
from experiments.symbolic_regression import run_search
from experiments.domain_validation_showcase import run_validation_showcase

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Summary generator
# ---------------------------------------------------------------------------


def _ablation_stats(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, int]]:
    """Return failure/invalid/ok counts keyed by mode."""
    stats: dict[str, dict[str, int]] = {}
    for mode in ("strict", "relaxed"):
        mode_rows = [r for r in rows if r["mode"] == mode]
        stats[mode] = {
            "ok": sum(1 for r in mode_rows if r["status"] == "ok"),
            "domain_error": sum(1 for r in mode_rows if r["status"] == "domain_error"),
            "invalid_baseline": sum(
                1 for r in mode_rows if r["status"] == "invalid_eml_used_baseline"
            ),
            "other": sum(
                1
                for r in mode_rows
                if r["status"] not in ("ok", "domain_error", "invalid_eml_used_baseline")
            ),
        }
    return stats


def generate_summary(
    ablation_rows: list[dict[str, Any]],
    analysis: dict[str, list[dict[str, Any]]],
) -> str:
    """Render the results summary as a Markdown string."""
    ab = _ablation_stats(ablation_rows)
    sr = analysis["success_rows"]
    eml_row = next(r for r in sr if r["grammar"] == "EML")
    base_row = next(r for r in sr if r["grammar"] == "baseline")

    depth_rows = analysis["depth_rows"]
    eml_depths = [r["depth"] for r in depth_rows if r["grammar"] == "EML"]
    base_depths = [r["depth"] for r in depth_rows if r["grammar"] == "baseline"]
    avg_eml_depth = round(sum(eml_depths) / len(eml_depths), 1) if eml_depths else "n/a"
    avg_base_depth = round(sum(base_depths) / len(base_depths), 1) if base_depths else "n/a"

    size_rows = analysis["size_rows"]
    eml_sizes = [r["node_count"] for r in size_rows if r["grammar"] == "EML"]
    base_sizes = [r["node_count"] for r in size_rows if r["grammar"] == "baseline"]
    avg_eml_size = round(sum(eml_sizes) / len(eml_sizes), 1) if eml_sizes else "n/a"
    avg_base_size = round(sum(base_sizes) / len(base_sizes), 1) if base_sizes else "n/a"

    timestamp = datetime.now(tz=UTC).strftime("%Y-%m-%d %H:%M UTC")

    md = textwrap.dedent(f"""\
        # EML Representation Study — Results Summary

        *Generated: {timestamp}*
        *Seed: {RANDOM_SEED} | Budget: {SEARCH_BUDGET} total evals | Pop: {SEARCH_POP_SIZE} | Max depth: {SEARCH_MAX_DEPTH}*

        ---

        ## 1. Symbolic Regression: Structural Characterization

        | Metric | EML | exp/log baseline |
        |---|---|---|
        | Exact recoveries (MSE < {EXACT_THRESHOLD:.0e}) | {eml_row["exact_count"]} / {eml_row["total"]} | {base_row["exact_count"]} / {base_row["total"]} |
        | Success rate | {eml_row["success_rate_pct"]}% | {base_row["success_rate_pct"]}% |
        | Average node count (best found) | {eml_row["avg_node_count"]} | {base_row["avg_node_count"]} |
        | Average tree depth (best found) | {avg_eml_depth} | {avg_base_depth} |

        The EML grammar produces structurally larger trees (avg {avg_eml_size} nodes
        vs {avg_base_size} for baseline). Under a shared true evaluation budget,
        differences in rejection rate directly affect the number of generations each
        grammar can complete.

        ---

        ## 2. Depth and Node Count Effects

        | Metric | EML | baseline |
        |---|---|---|
        | Average depth (best-found trees) | {avg_eml_depth} | {avg_base_depth} |
        | Average node count (best-found trees) | {avg_eml_size} | {avg_base_size} |

        Structural inflation is a direct consequence of the EML encoding rules:
        - `exp(x)` → `eml(x, 1)` — adds one nesting level
        - `ln(x)` → `1 − eml(0, x)` — adds two nesting levels and an extra constant

        See `figures/depth_vs_error.png` and `figures/size_vs_error.png`.

        ---

        ## 3. Domain Constraint Effects

        | Mode | Successful transforms | Domain errors | Invalid (baseline used) |
        |---|---|---|---|
        | Strict (enforce\\_domain=True)  | {ab["strict"]["ok"]} | {ab["strict"]["domain_error"]} | 0 |
        | Relaxed (enforce\\_domain=False) | {ab["relaxed"]["ok"]} | 0 | {ab["relaxed"]["invalid_baseline"]} |

        Domain constraints are a primary shaping force on EML representations.
        The strict positivity requirements for logarithm arguments directly
        determine which expressions can be absorbed into the EML primitive.

        ---

        ## 4. Reproducibility

        All results in this directory were generated deterministically with:

        ```bash
        python -m experiments.run_all
        ```

        Seed: `{RANDOM_SEED}`.

        ---

        ## Output Files

        | File | Description |
        |---|---|
        | `results/domain_ablation.csv` | Per-expression transform outcomes (strict vs relaxed) |
        | `results/symbolic_regression_per_target.csv` | Per-target SR results with rejection metrics |
        | `results/depth_vs_error.csv` | Tree depth and MSE per target × grammar |
        | `results/size_vs_error.csv` | Node count and MSE per target × grammar |
        | `results/success_rates.csv` | Exact recovery counts and success rates |
        | `figures/depth_vs_error.png` | Scatter: MSE vs tree depth |
        | `figures/size_vs_error.png` | Scatter: MSE vs node count |
    """)

    return md


def save_summary(md: str, path: Path) -> None:
    """Write *md* to *path*."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8")
    logger.info("Saved → %s", path)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------


def run_all() -> None:
    """Run all experiments and generate all output artifacts."""
    logger.info("=== Step 1/5: Domain ablation ===")
    ablation_rows = run_domain_ablation()
    save_ablation_csv(ablation_rows, RESULTS_DIR / "domain_ablation.csv")

    logger.info("=== Step 2/5: Symbolic regression ===")
    sr_results = run_search()

    logger.info("=== Step 3/5: Depth/size analysis and success rates ===")
    analysis = run_depth_size_analysis(sr_results=sr_results)

    logger.info("=== Step 4/5: Domain validation showcase ===")
    run_validation_showcase()

    logger.info("=== Step 5/5: Generating summary ===")
    md = generate_summary(ablation_rows, analysis)
    save_summary(md, RESULTS_DIR / "summary.md")

    logger.info("=== All done. Results in %s/ and %s/ ===", RESULTS_DIR, FIGURES_DIR)


if __name__ == "__main__":
    configure_logging()
    run_all()
