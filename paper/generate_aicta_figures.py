"""Generate AICTA 2026 paper figures from existing experimental data.

Run from the repository root::

    python paper/generate_aicta_figures.py

Outputs (saved to paper/figures/):
    rejection_vs_generations.pdf   — rejection rate vs productive generations
    rejection_distribution.pdf     — boxplot of rejection rates by grammar
    rejection_composition.pdf      — rejection-type breakdown by grammar
    tree_comparison.pdf            — side-by-side tree structure visualization
"""

from __future__ import annotations

import csv
import logging
import sys
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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
logger = logging.getLogger("generate_aicta_figures")

FIGURE_DIR = Path(__file__).parent / "figures"
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

DATA_CSV = ROOT / "results" / "symbolic_regression_per_target.csv"

# ── style constants (restrained academic palette) ─────────────────────────────
EML_COLOR = "#2563EB"      # steel blue
BASE_COLOR = "#D97706"     # muted amber
EML_MARKER = "o"
BASE_MARKER = "s"

plt.rcParams.update({
    "font.family": "serif",
    "font.size": 9,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.labelsize": 10,
    "axes.titlesize": 11,
    "legend.fontsize": 8,
    "figure.dpi": 300,
})


# ── data loader ───────────────────────────────────────────────────────────────

def load_data() -> list[dict]:
    """Load per-target SR results from CSV."""
    rows = []
    with open(DATA_CSV, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append({
                "target": row["target"].strip(),
                "eml_mse": float(row["eml_mse_mean"]),
                "base_mse": float(row["base_mse_mean"]),
                "eml_nodes": float(row["eml_nodes_mean"]),
                "base_nodes": float(row["base_nodes_mean"]),
                "eml_depth": float(row["eml_depth_mean"]),
                "base_depth": float(row["base_depth_mean"]),
                "eml_gens": float(row["eml_generations_mean"]),
                "base_gens": float(row["base_generations_mean"]),
                "eml_rej": float(row["eml_rejection_rate_pct"]),
                "base_rej": float(row["base_rejection_rate_pct"]),
                "eml_domain": float(row["eml_domain_mean"]),
                "base_domain": float(row["base_domain_mean"]),
                "eml_depth_rej": float(row["eml_depth_rej_mean"]),
                "base_depth_rej": float(row["base_depth_rej_mean"]),
                "eml_numeric": float(row["eml_numeric_mean"]),
                "base_numeric": float(row["base_numeric_mean"]),
                "eml_accepted": float(row["eml_accepted_mean"]),
                "base_accepted": float(row["base_accepted_mean"]),
            })
    return rows


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Rejection Rate vs Generations Completed (CENTERPIECE)
# ─────────────────────────────────────────────────────────────────────────────

def generate_rejection_vs_generations(data: list[dict]) -> None:
    """Scatter plot: rejection rate vs productive generations for both grammars."""
    fig, ax = plt.subplots(figsize=(5.5, 4.0))

    eml_rej = [d["eml_rej"] for d in data]
    eml_gen = [d["eml_gens"] for d in data]
    base_rej = [d["base_rej"] for d in data]
    base_gen = [d["base_gens"] for d in data]

    # Plot data points — legend comes from the label kwarg only
    ax.scatter(base_rej, base_gen, c=BASE_COLOR, marker=BASE_MARKER,
               s=50, alpha=0.85, edgecolors="white", linewidths=0.5,
               label="exp/log baseline", zorder=3)
    ax.scatter(eml_rej, eml_gen, c=EML_COLOR, marker=EML_MARKER,
               s=50, alpha=0.85, edgecolors="white", linewidths=0.5,
               label="EML grammar", zorder=3)

    # Trend line
    all_rej = np.array(eml_rej + base_rej)
    all_gen = np.array(eml_gen + base_gen)
    z = np.polyfit(all_rej, all_gen, 1)
    p = np.poly1d(z)
    x_fit = np.linspace(min(all_rej) - 2, max(all_rej) + 2, 100)
    ax.plot(x_fit, p(x_fit), color="gray", linewidth=1.0, linestyle="-",
            alpha=0.5, zorder=1, label=None)  # no legend entry for trend

    ax.set_xlabel("Rejection rate (%)")
    ax.set_ylabel("Completed evolutionary generations")
    ax.legend(loc="upper right", framealpha=0.9, edgecolor="#cccccc")

    plt.tight_layout()
    out = FIGURE_DIR / "rejection_vs_generations.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Rejection Rate Distribution (boxplot)
# ─────────────────────────────────────────────────────────────────────────────

def generate_rejection_distribution(data: list[dict]) -> None:
    """Violin + boxplot comparing rejection rate distributions between grammars."""
    fig, ax = plt.subplots(figsize=(4.0, 3.8))

    eml_rej = [d["eml_rej"] for d in data]
    base_rej = [d["base_rej"] for d in data]
    all_data = [base_rej, eml_rej]
    positions = [1, 2]
    colors = [BASE_COLOR, EML_COLOR]
    grammar_labels = ["exp/log\nbaseline", "EML\ngrammar"]

    # Violin bodies for distribution shape
    vp = ax.violinplot(all_data, positions=positions, showextrema=False,
                       widths=0.7)
    for i, body in enumerate(vp["bodies"]):
        body.set_facecolor(colors[i])
        body.set_alpha(0.2)
        body.set_edgecolor(colors[i])
        body.set_linewidth(0.8)

    # Overlay compact boxplots
    bp = ax.boxplot(
        all_data, positions=positions,
        widths=0.25, patch_artist=True,
        medianprops=dict(color="black", linewidth=1.5),
        whiskerprops=dict(linewidth=1.0, color="#555555"),
        capprops=dict(linewidth=1.0, color="#555555"),
        flierprops=dict(marker="", markersize=0),  # hide fliers, show as scatter
    )
    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(colors[i])
        box.set_alpha(0.45)
        box.set_edgecolor(colors[i])

    # Overlay individual data points
    rng = np.random.default_rng(42)
    for i, (vals, color) in enumerate(zip(all_data, colors)):
        jitter = rng.normal(0, 0.04, len(vals))
        ax.scatter(
            np.full(len(vals), positions[i]) + jitter, vals,
            c=color, s=22, alpha=0.75, edgecolors="white",
            linewidths=0.4, zorder=4,
        )

    # Annotate medians
    for i, vals in enumerate(all_data):
        med = np.median(vals)
        ax.text(positions[i], med + 2.5, f"{med:.1f}%",
                ha="center", va="bottom", fontsize=7.5, color="#333333",
                fontweight="bold")

    ax.set_xticks(positions)
    ax.set_xticklabels(grammar_labels)
    ax.set_ylabel("Rejection rate (%)")
    ax.set_ylim(0, 100)
    ax.set_xlim(0.3, 2.7)

    plt.tight_layout()
    out = FIGURE_DIR / "rejection_distribution.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Rejection Composition Stacked Bar Chart (NEW)
# ─────────────────────────────────────────────────────────────────────────────

def generate_rejection_composition(data: list[dict]) -> None:
    """Stacked bar chart showing composition of rejections (share of total rejected)."""
    fig, ax = plt.subplots(figsize=(5.0, 4.0))

    # Aggregate rejection types across all targets
    eml_domain = sum(d["eml_domain"] for d in data)
    eml_depth = sum(d["eml_depth_rej"] for d in data)
    eml_numeric = sum(d["eml_numeric"] for d in data)

    base_domain = sum(d["base_domain"] for d in data)
    base_depth = sum(d["base_depth_rej"] for d in data)
    base_numeric = sum(d["base_numeric"] for d in data)

    # Normalize to share of total rejected candidates (sum to 100%)
    eml_total_rej = eml_domain + eml_depth + eml_numeric
    base_total_rej = base_domain + base_depth + base_numeric

    eml_domain_share = 100.0 * eml_domain / eml_total_rej
    eml_depth_share = 100.0 * eml_depth / eml_total_rej
    eml_numeric_share = 100.0 * eml_numeric / eml_total_rej

    base_domain_share = 100.0 * base_domain / base_total_rej
    base_depth_share = 100.0 * base_depth / base_total_rej
    base_numeric_share = 100.0 * base_numeric / base_total_rej

    # Colors for rejection types
    domain_color = "#DC2626"  # red
    depth_color = "#2563EB"   # blue
    numeric_color = "#D97706"  # amber

    x = np.array([1, 2])
    width = 0.5

    # Stacked bars for baseline (normalized shares)
    ax.bar(x[0], base_domain_share, width, label="Domain", color=domain_color, alpha=0.85)
    ax.bar(x[0], base_depth_share, width, bottom=base_domain_share, label="Depth", color=depth_color, alpha=0.85)
    ax.bar(x[0], base_numeric_share, width, bottom=base_domain_share + base_depth_share, label="Numeric", color=numeric_color, alpha=0.85)

    # Stacked bars for EML (normalized shares)
    ax.bar(x[1], eml_domain_share, width, color=domain_color, alpha=0.85)
    ax.bar(x[1], eml_depth_share, width, bottom=eml_domain_share, color=depth_color, alpha=0.85)
    ax.bar(x[1], eml_numeric_share, width, bottom=eml_domain_share + eml_depth_share, color=numeric_color, alpha=0.85)

    # Labels
    ax.set_xticks(x)
    ax.set_xticklabels(["exp/log\nbaseline", "EML\ngrammar"])
    ax.set_ylabel("Share of rejected candidates (%)")
    ax.set_ylim(0, 100)

    # Add percentage labels on bars
    ax.text(x[0], base_domain_share / 2, f"{base_domain_share:.1f}%", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    ax.text(x[0], base_domain_share + base_depth_share / 2, f"{base_depth_share:.1f}%", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    ax.text(x[0], base_domain_share + base_depth_share + base_numeric_share / 2, f"{base_numeric_share:.1f}%", ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    ax.text(x[1], eml_domain_share / 2, f"{eml_domain_share:.1f}%", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    ax.text(x[1], eml_domain_share + eml_depth_share / 2, f"{eml_depth_share:.1f}%", ha="center", va="center", fontsize=8, color="white", fontweight="bold")
    ax.text(x[1], eml_domain_share + eml_depth_share + eml_numeric_share / 2, f"{eml_numeric_share:.1f}%", ha="center", va="center", fontsize=8, color="white", fontweight="bold")

    ax.legend(loc="upper right", framealpha=0.9, edgecolor="#cccccc", title="Rejection type")
    plt.tight_layout()
    out = FIGURE_DIR / "rejection_composition.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Tree Comparison (exp(log(x+1)) — native vs EML)
# ─────────────────────────────────────────────────────────────────────────────

def _draw_tree(ax, tree, x=0.5, y=0.95, dx=0.2, dy=0.13, level=0):
    """Recursively draw a symbolic tree represented as nested tuples."""
    label, children = tree
    node_color = "#e3f0ff" if label == "eml" else "#f5f5f5"
    border_color = EML_COLOR if label == "eml" else "#555555"

    ax.text(x, y, label, ha="center", va="center", fontsize=7.5,
            fontweight="bold" if label == "eml" else "normal",
            bbox=dict(boxstyle="round,pad=0.25", facecolor=node_color,
                      edgecolor=border_color, linewidth=1.0),
            zorder=4)

    if children:
        n = len(children)
        child_positions = []
        for i, child in enumerate(children):
            # Spread children evenly
            cx = x + (i - (n - 1) / 2) * dx
            cy = y - dy
            child_positions.append((cx, cy))
            ax.plot([x, cx], [y - 0.02, cy + 0.02], color="#999999",
                    linewidth=0.8, zorder=1)
            _draw_tree(ax, child, cx, cy, dx * 0.75, dy, level + 1)


def generate_tree_comparison() -> None:
    """Side-by-side tree visualization: native exp/log vs EML form."""
    # Expression: exp(log(x + 1))
    native_tree = ("exp", [
        ("log", [
            ("+", [
                ("x", []),
                ("1", []),
            ]),
        ]),
    ])

    # EML form: eml(1 - eml(0, x+1), 1)
    eml_tree = ("eml", [
        ("\u2212", [
            ("1", []),
            ("eml", [
                ("0", []),
                ("+", [
                    ("x", []),
                    ("1", []),
                ]),
            ]),
        ]),
        ("1", []),
    ])

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(6.5, 3.6))

    for ax in (ax1, ax2):
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_aspect("equal")
        ax.axis("off")

    ax1.set_title("Native exp/log", fontsize=10, fontweight="bold", pad=8)
    ax2.set_title("EML-expanded representation", fontsize=10, fontweight="bold", pad=8)

    _draw_tree(ax1, native_tree, x=0.5, y=0.88, dx=0.28, dy=0.20)
    _draw_tree(ax2, eml_tree, x=0.5, y=0.88, dx=0.38, dy=0.16)

    # Metrics as aligned table below trees
    col_headers = ["Representation", "Nodes", "Depth"]
    row_data = [
        ["Native exp/log", "4", "3"],
        ["EML-expanded", "7", "4"],
    ]
    # Add table to the figure itself to guarantee centering
    table = fig.add_axes([0.15, 0.05, 0.7, 0.2]).table(
        cellText=row_data,
        colLabels=col_headers,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    # Hide the axes we created for the table
    fig.axes[-1].axis("off")
    
    table.auto_set_font_size(False)
    table.set_fontsize(10.5)
    for key, cell in table.get_celld().items():
        cell.set_linewidth(0.5)
        cell.set_edgecolor("#888888")
        if key[0] == 0:  # header row
            cell.set_facecolor("#f0f0f0")
            cell.set_text_props(fontweight="bold")
        else:
            cell.set_facecolor("white")

    plt.tight_layout(rect=[0, 0.25, 1, 1])
    out = FIGURE_DIR / "tree_comparison.pdf"
    fig.savefig(out, bbox_inches="tight")
    plt.close(fig)
    logger.info("Saved → %s", out)


# ─────────────────────────────────────────────────────────────────────────────
# Summary statistics (printed for LaTeX table)
# ─────────────────────────────────────────────────────────────────────────────

def print_summary_table(data: list[dict]) -> None:
    """Print summary statistics for the structural inflation table."""
    eml_nodes = np.mean([d["eml_nodes"] for d in data])
    base_nodes = np.mean([d["base_nodes"] for d in data])
    eml_depth = np.mean([d["eml_depth"] for d in data])
    base_depth = np.mean([d["base_depth"] for d in data])
    eml_rej = np.mean([d["eml_rej"] for d in data])
    base_rej = np.mean([d["base_rej"] for d in data])
    eml_gens = np.mean([d["eml_gens"] for d in data])
    base_gens = np.mean([d["base_gens"] for d in data])

    # Calculate rejection type percentages from raw counts
    # Compute total evals from accepted count and rejection rate: total = accepted / (1 - rej_rate)
    eml_total_evals = sum(d["eml_accepted"] / (1 - d["eml_rej"]/100.0) for d in data)
    base_total_evals = sum(d["base_accepted"] / (1 - d["base_rej"]/100.0) for d in data)
    eml_domain = sum(d["eml_domain"] for d in data)
    eml_depth_rej = sum(d["eml_depth_rej"] for d in data)
    eml_numeric = sum(d["eml_numeric"] for d in data)
    base_domain = sum(d["base_domain"] for d in data)
    base_depth_rej = sum(d["base_depth_rej"] for d in data)
    base_numeric = sum(d["base_numeric"] for d in data)

    eml_domain_pct = 100.0 * eml_domain / eml_total_evals
    eml_depth_pct = 100.0 * eml_depth_rej / eml_total_evals
    eml_numeric_pct = 100.0 * eml_numeric / eml_total_evals
    base_domain_pct = 100.0 * base_domain / base_total_evals
    base_depth_pct = 100.0 * base_depth_rej / base_total_evals
    base_numeric_pct = 100.0 * base_numeric / base_total_evals

    print("\n" + "=" * 70)
    print("STRUCTURAL INFLATION SUMMARY TABLE (for LaTeX)")
    print("=" * 70)
    print(f"{'Grammar':<20} {'Nodes':>8} {'Depth':>8} {'DomRej':>8} {'DepRej':>8} {'NumRej':>8} {'TotRej':>8} {'Gens':>8}")
    print("-" * 72)
    print(f"{'exp/log baseline':<20} {base_nodes:>8.1f} {base_depth:>8.1f} "
          f"{base_domain_pct:>8.1f} {base_depth_pct:>8.1f} {base_numeric_pct:>8.1f} {base_rej:>8.1f} {base_gens:>8.1f}")
    print(f"{'EML grammar':<20} {eml_nodes:>8.1f} {eml_depth:>8.1f} "
          f"{eml_domain_pct:>8.1f} {eml_depth_pct:>8.1f} {eml_numeric_pct:>8.1f} {eml_rej:>8.1f} {eml_gens:>8.1f}")
    print("-" * 72)
    print(f"{'Ratio (EML/base)':<20} {eml_nodes/base_nodes:>8.1f}x "
          f"{eml_depth/base_depth:>7.1f}x "
          f"{eml_domain_pct/base_domain_pct:>7.1f}x "
          f"{eml_depth_pct/base_depth_pct:>7.1f}x "
          f"{eml_numeric_pct/base_numeric_pct:>7.1f}x "
          f"{eml_rej/base_rej:>7.1f}x "
          f"{eml_gens/base_gens:>7.1f}x")
    print("=" * 72)

    # Also print LaTeX-ready table (updated with rejection breakdown)
    print("\n% LaTeX table code (updated with rejection breakdown):")
    print(r"\begin{table}[t]")
    print(r"\centering")
    print(r"\small")
    print(r"\caption{Structural inflation and rejection dynamics across 18 targets. "
          r"Mean values over 5 independent trials per target. "
          r"Rejection rates are disaggregated into domain (symbolic constraint violations), "
          r"depth (structural limits), and numeric (evaluation failures) components. "
          r"Higher domain rejection under EML reflects the sparser valid region imposed by "
          r"strict positivity constraints, while depth rejection reflects structural inflation "
          r"consuming the depth budget.}")
    print(r"\label{tab:inflation_summary}")
    print(r"\begin{tabular}{lrrrrrrr}")
    print(r"\toprule")
    print(r"\textbf{Grammar} & \textbf{Nodes} & \textbf{Depth} "
          r"& \textbf{Domain Rej.~(\%)} & \textbf{Depth Rej.~(\%)} & \textbf{Num.~Rej.~(\%)} "
          r"& \textbf{Total Rej.~(\%)} & \textbf{Gens} \\")
    print(r"\midrule")
    print(f"exp/log baseline & {base_nodes:.1f} & {base_depth:.1f} "
          f"& {base_domain_pct:.1f} & {base_depth_pct:.1f} & {base_numeric_pct:.1f} "
          f"& {base_rej:.1f} & {base_gens:.1f} \\\\")
    print(f"EML grammar & {eml_nodes:.1f} & {eml_depth:.1f} "
          f"& {eml_domain_pct:.1f} & {eml_depth_pct:.1f} & {eml_numeric_pct:.1f} "
          f"& {eml_rej:.1f} & {eml_gens:.1f} \\\\")
    print(r"\midrule")
    print(f"Ratio (EML / baseline) & {eml_nodes/base_nodes:.1f}$\\times$ "
          f"& {eml_depth/base_depth:.1f}$\\times$ "
          f"& {eml_domain_pct/base_domain_pct:.1f}$\\times$ "
          f"& {eml_depth_pct/base_depth_pct:.1f}$\\times$ "
          f"& {eml_numeric_pct/base_numeric_pct:.1f}$\\times$ "
          f"& {eml_rej/base_rej:.1f}$\\times$ "
          f"& {eml_gens/base_gens:.1f}$\\times$ \\\\")
    print(r"\bottomrule")
    print(r"\end{tabular}")
    print(r"\end{table}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    logger.info("=== Generating AICTA 2026 figures ===")
    data = load_data()
    generate_rejection_vs_generations(data)
    generate_rejection_distribution(data)
    generate_rejection_composition(data)
    generate_tree_comparison()
    print_summary_table(data)
    logger.info("=== Done. Figures written to %s ===", FIGURE_DIR)
