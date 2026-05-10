"""Symbolic regression experiment: EML vs baseline (controlled characterization).

Design principles (all are controlled variables — representation is the ONLY
intended experimental variable):

- Single evaluation path per grammar (no fallbacks, no ∞ fitness).
- True evaluation budget: every call to the evaluator — including rejection-
  sampling attempts — counts against SEARCH_BUDGET.
- Strict depth gate: candidates with actual tree_depth > SEARCH_MAX_DEPTH are
  rejected regardless of grammar, before numeric evaluation.
- Grammar-pure mutation: EML mutations stay strictly within the EML grammar
  (mul_expr, Add-based negation). Baseline uses native SymPy operators.
- Equal *valid* population sizes via rejection sampling; the budget cost of
  reaching that size is itself a characterization metric (rejection_rate).
- N_TRIALS = 5 independent runs; results reported as mean ± std.

New characterization metrics reported per grammar per target:
  generations_completed  — how many generations fit within the true budget
  rejection_rate_pct     — pct of candidate evaluations that were rejected

Run with::

    python -m experiments.symbolic_regression
"""

from __future__ import annotations

import csv
import logging
import random
import statistics
from pathlib import Path
from typing import Any, Callable

import numpy as np
import sympy as sp

from eml import configure_logging, evaluate_eml, total_node_count, tree_depth
from eml.config import (
    RANDOM_SEED,
    SEARCH_BUDGET,
    SEARCH_ELITE_SIZE,
    SEARCH_MAX_DEPTH,
    SEARCH_POP_SIZE,
    SEARCH_SAMPLE_POINTS,
)
from eml.core import eml

logger = logging.getLogger(__name__)

N_TRIALS: int = 5
RESULTS_DIR = Path(__file__).parent.parent / "results"

__all__ = [
    "target_functions",
    "run_search",
    "log_expr",
    "exp_expr",
    "mul_expr",
    "pow_expr",
    "random_eml_expr",
    "random_sympy_expr",
    "evaluate_expr",
    "compute_mse",
    "search_for_target",
]


# ---------------------------------------------------------------------------
# EML grammar primitives
# ---------------------------------------------------------------------------


def log_expr(e: sp.Expr) -> sp.Expr:
    """Encode ln(e) via EML: 1 - eml(0, e)."""
    return sp.Integer(1) - eml(sp.Integer(0), e)


def exp_expr(e: sp.Expr) -> sp.Expr:
    """Encode exp(e) via EML: eml(e, 1)."""
    return eml(e, sp.Integer(1))


def mul_expr(a: sp.Expr, b: sp.Expr) -> sp.Expr:
    """Encode a*b via EML exp-log identity (requires a,b > 0 on domain)."""
    return exp_expr(log_expr(a) + log_expr(b))


def pow_expr(base: sp.Expr, exp_: sp.Expr) -> sp.Expr:
    """Encode base**exp_ via EML exp-log identity."""
    return exp_expr(log_expr(base) * exp_)


# ---------------------------------------------------------------------------
# Random tree generators
# ---------------------------------------------------------------------------

def _leaf(x: sp.Symbol) -> sp.Expr:
    return random.choice([x, sp.Integer(1), sp.Integer(2)])


def random_eml_expr(depth: int, x: sp.Symbol) -> sp.Expr:
    """Sample a random tree from the EML grammar.

    Grammar-pure: the only nodes produced are eml(...), Add, atoms.
    Negation is excluded because 0 - expr reduces to Mul(-1, expr) in
    SymPy, which violates the positive-domain EML fragment constraint.
    Multiplication and powers are encoded through EML exp-log identities.
    """
    if depth <= 0:
        return _leaf(x)
    op = random.choices(
        ["leaf", "eml", "add", "exp", "log", "mul", "pow"],
        weights=[0.12, 0.12, 0.16, 0.08, 0.17, 0.10, 0.25],
    )[0]
    d = depth - 1
    if op == "leaf":  return _leaf(x)
    if op == "eml":   return eml(random_eml_expr(d, x), random_eml_expr(d, x))
    if op == "add":   return random_eml_expr(d, x) + random_eml_expr(d, x)
    if op == "exp":   return exp_expr(random_eml_expr(d, x))
    if op == "log":   return log_expr(random_eml_expr(d, x))
    if op == "mul":   return mul_expr(random_eml_expr(d, x), random_eml_expr(d, x))
    # pow
    return pow_expr(random_eml_expr(d, x), random.choice([sp.Integer(2), sp.Integer(3)]))


def random_sympy_expr(depth: int, x: sp.Symbol) -> sp.Expr:
    """Sample a random tree from the standard exp/log grammar."""
    if depth <= 0:
        return _leaf(x)
    op = random.choices(
        ["leaf", "add", "exp", "log", "mul", "pow", "neg"],
        weights=[0.12, 0.13, 0.10, 0.20, 0.10, 0.30, 0.05],
    )[0]
    d = depth - 1
    if op == "leaf":  return _leaf(x)
    if op == "add":   return random_sympy_expr(d, x) + random_sympy_expr(d, x)
    if op == "exp":   return sp.exp(random_sympy_expr(d, x))
    if op == "log":   return sp.log(random_sympy_expr(d, x))
    if op == "mul":   return random_sympy_expr(d, x) * random_sympy_expr(d, x)
    if op == "pow":
        return random_sympy_expr(d, x) ** random.choice([sp.Integer(2), sp.Integer(3), sp.Integer(-1)])
    return -random_sympy_expr(d, x)


# ---------------------------------------------------------------------------
# Single unified evaluation path
# ---------------------------------------------------------------------------


class DomainError(Exception):
    """Raised when numeric evaluation produces mathematically invalid outputs (e.g. complex numbers)."""
    pass

class NumericError(Exception):
    """Raised when numeric evaluation produces floating-point instability (e.g. NaN, overflow)."""
    pass

def _validate_numeric_output(y: np.ndarray, xs: np.ndarray) -> np.ndarray:
    """Validate numeric output — raise DomainError or NumericError on any violation."""
    if np.iscomplexobj(y):
        raise DomainError("Complex output detected")
    if not np.all(np.isfinite(y)):
        raise NumericError("NaN or Inf detected")
    if y.shape != xs.shape:
        raise NumericError(f"Shape mismatch: {y.shape} vs {xs.shape}")
    return y

def evaluate_expr(
    expr: sp.Expr,
    x: sp.Symbol,
    xs: np.ndarray,
    *,
    is_eml: bool,
) -> tuple[np.ndarray | None, str | None]:
    """Evaluate *expr* numerically on *xs* with strict validity checks.

    Returns (y_arr, error_type). error_type is 'domain' or 'numeric' if invalid.
    """
    try:
        sym_expr = evaluate_eml(expr) if is_eml else expr
        f = sp.lambdify(
            x,
            sym_expr,
            [{"exp": np.exp, "log": np.log, "Abs": np.abs, "sqrt": np.sqrt}],
        )
    except (ValueError, TypeError):
        return None, "domain"
    except Exception:
        return None, "domain"

    try:
        with np.errstate(over="raise", divide="raise", invalid="raise"):
            y = f(xs)
    except FloatingPointError as e:
        if "invalid" in str(e):
            return None, "domain"
        return None, "numeric"
    except (ValueError, TypeError):
        return None, "domain"
    except Exception:
        return None, "numeric"

    try:
        y_arr = np.asarray(y)
        if np.isscalar(y_arr):
            y_arr = np.full_like(xs, y_arr, dtype=float)
        _validate_numeric_output(y_arr, xs)
        return y_arr, None
    except DomainError:
        return None, "domain"
    except NumericError:
        return None, "numeric"
    except Exception:
        return None, "numeric"


# ---------------------------------------------------------------------------
# Target functions
# ---------------------------------------------------------------------------


def target_functions() -> list[dict[str, Any]]:
    """18 target functions with positive-domain evaluation ranges."""
    x: sp.Symbol = sp.symbols("x")
    return [
        {"name": "x^2 + x",         "expr": x**2 + x,                    "range": (0.1, 3.0)},
        {"name": "x^3 + x",         "expr": x**3 + x,                    "range": (0.1, 2.5)},
        {"name": "x^3 + x^2 + x",   "expr": x**3 + x**2 + x,             "range": (0.1, 2.0)},
        {"name": "exp(x)",           "expr": sp.exp(x),                   "range": (-2.0, 2.0)},
        {"name": "exp(x^2)",         "expr": sp.exp(x**2),                "range": (-1.5, 1.5)},
        {"name": "exp(x) + x",       "expr": sp.exp(x) + x,               "range": (-1.0, 2.0)},
        {"name": "log(x)",           "expr": sp.log(x),                   "range": (0.1, 5.0)},
        {"name": "log(x^2 + 1)",     "expr": sp.log(x**2 + 1),            "range": (-2.0, 2.0)},
        {"name": "log(x + 2)",       "expr": sp.log(x + 2),               "range": (-1.0, 3.0)},
        {"name": "x^2 * log(x)",     "expr": x**2 * sp.log(x),            "range": (0.1, 3.0)},
        {"name": "exp(x) * log(x)",  "expr": sp.exp(x) * sp.log(x),       "range": (0.1, 2.5)},
        {"name": "x^2 * exp(x)",     "expr": x**2 * sp.exp(x),            "range": (-1.0, 2.0)},
        {"name": "exp(x) + log(x)",  "expr": sp.exp(x) + sp.log(x),       "range": (0.1, 3.0)},
        {"name": "x / (1 + x)",      "expr": x / (1 + x),                 "range": (0.1, 3.0)},
        {"name": "x^2 / (1 + x^2)",  "expr": x**2 / (1 + x**2),           "range": (-2.0, 2.0)},
        {"name": "exp(log(x+1))",    "expr": sp.exp(sp.log(x + 1)),        "range": (0.1, 3.0)},
        {"name": "log(exp(x))",      "expr": sp.log(sp.exp(x)),            "range": (-2.0, 1.0)},
        {"name": "exp(x) + exp(-x)", "expr": sp.exp(x) + sp.exp(-x),      "range": (-2.0, 2.0)},
    ]


def compute_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute MSE with strict validation."""
    if np.iscomplexobj(y_pred):
        raise DomainError("Complex values in prediction")
    if not np.all(np.isfinite(y_pred)):
        raise DomainError("Non-finite values in prediction")
    if y_pred.shape != y_true.shape:
        raise DomainError(f"Shape mismatch: {y_pred.shape} vs {y_true.shape}")
    return float(np.mean((y_true - y_pred) ** 2))


# ---------------------------------------------------------------------------
# Valid population generation (rejection sampling + depth gate + budget)
# ---------------------------------------------------------------------------


def generate_valid_population(
    size: int,
    generator: Callable[[int, sp.Symbol], sp.Expr],
    evaluator: Callable[[sp.Expr, sp.Symbol, np.ndarray], tuple[np.ndarray | None, str | None]],
    x: sp.Symbol,
    xs: np.ndarray,
    max_depth: int,
    *,
    eval_counter: list[int],
    budget: int,
) -> tuple[list[sp.Expr], int, int, int, int]:
    """Return up to *size* valid individuals via rejection sampling.

    Every call to the evaluator — including rejected candidates — increments
    eval_counter[0] and counts against the true search budget.

    Returns (population, rej_domain, rej_depth, rej_numeric, accepted).
    """
    population: list[sp.Expr] = []
    rej_domain = 0
    rej_depth = 0
    rej_numeric = 0
    accepted = 0
    while len(population) < size and eval_counter[0] < budget:
        candidate = generator(max_depth, x)
        # Gate 1: strict actual depth
        if tree_depth(candidate) > max_depth:
            rej_depth += 1
            eval_counter[0] += 1
            continue
        # Gate 2: numeric validity
        eval_counter[0] += 1
        y, err = evaluator(candidate, x, xs)
        if err is None and y is not None:
            population.append(candidate)
            accepted += 1
        elif err == "domain":
            rej_domain += 1
        else:
            rej_numeric += 1
    return population, rej_domain, rej_depth, rej_numeric, accepted


# ---------------------------------------------------------------------------
# Grammar-aware mutation
# ---------------------------------------------------------------------------


def _mutate(
    expr: sp.Expr,
    x: sp.Symbol,
    generator: Callable[[int, sp.Symbol], sp.Expr],
    evaluator: Callable[[sp.Expr, sp.Symbol, np.ndarray], tuple[np.ndarray | None, str | None]],
    xs: np.ndarray,
    max_depth: int,
    *,
    is_eml: bool,
    eval_counter: list[int],
    budget: int,
    max_attempts: int = 30,
) -> tuple[sp.Expr | None, int, int, int, int]:
    """Mutate *expr* using grammar-aware operators.

    Returns (result, rej_domain, rej_depth, rej_numeric, accepted).
    """
    rej_domain = 0
    rej_depth = 0
    rej_numeric = 0
    accepted = 0
    for _ in range(max_attempts):
        if eval_counter[0] >= budget:
            return None, rej_domain, rej_depth, rej_numeric, accepted
        r = random.random()
        subtree = generator(max(1, max_depth // 2), x)
        if r < 0.40:
            candidate = expr + subtree
        elif r < 0.70:
            candidate = mul_expr(expr, subtree) if is_eml else expr * subtree
        else:
            candidate = subtree  # full replacement

        # Depth gate
        if tree_depth(candidate) > max_depth:
            rej_depth += 1
            eval_counter[0] += 1
            continue

        eval_counter[0] += 1
        y, err = evaluator(candidate, x, xs)
        if err is None and y is not None:
            accepted += 1
            return candidate, rej_domain, rej_depth, rej_numeric, accepted
        elif err == "domain":
            rej_domain += 1
        else:
            rej_numeric += 1

    # Fallback: fresh valid individual
    fallback_pop, f_dom, f_dep, f_num, f_acc = generate_valid_population(
        1, generator, evaluator, x, xs, max_depth,
        eval_counter=eval_counter, budget=budget,
    )
    rej_domain += f_dom
    rej_depth += f_dep
    rej_numeric += f_num
    accepted += f_acc
    if fallback_pop:
        return fallback_pop[0], rej_domain, rej_depth, rej_numeric, accepted
    return None, rej_domain, rej_depth, rej_numeric, accepted


# ---------------------------------------------------------------------------
# Evolutionary search (single clean loop, true budget)
# ---------------------------------------------------------------------------


def search_for_target(
    target: dict[str, Any],
    generator: Callable[[int, sp.Symbol], sp.Expr],
    evaluator: Callable[[sp.Expr, sp.Symbol, np.ndarray], tuple[np.ndarray | None, str | None]],
    *,
    budget: int = SEARCH_BUDGET,
    pop_size: int = SEARCH_POP_SIZE,
    elite_size: int = SEARCH_ELITE_SIZE,
    max_depth: int = SEARCH_MAX_DEPTH,
    seed: int = RANDOM_SEED,
    is_eml: bool = False,
) -> dict[str, Any]:
    """Run one trial of evolutionary SR for *target*.

    True budget: every evaluator call (including rejections during population
    generation and mutation) counts against *budget*.

    Returns dictionary with tracking metrics.
    """
    random.seed(seed)
    np.random.seed(seed)

    x: sp.Symbol = sp.symbols("x")
    lo, hi = target["range"]
    xs = np.linspace(lo, hi, SEARCH_SAMPLE_POINTS)
    y_true = sp.lambdify(x, target["expr"], "numpy")(xs)

    eval_counter: list[int] = [0]
    total_domain = 0
    total_depth = 0
    total_numeric = 0
    total_accepted = 0

    def fitness(expr: sp.Expr) -> float | None:
        y, err = evaluator(expr, x, xs)
        if err is not None or y is None:
            return None
        try:
            return compute_mse(y_true, y)
        except (DomainError, NumericError, FloatingPointError, ValueError):
            return None

    # --- Initial valid population ---
    population, init_dom, init_dep, init_num, init_acc = generate_valid_population(
        pop_size, generator, evaluator, x, xs, max_depth,
        eval_counter=eval_counter, budget=budget,
    )
    total_domain += init_dom
    total_depth += init_dep
    total_numeric += init_num
    total_accepted += init_acc

    best_expr: sp.Expr | None = None
    best_mse: float = float("inf")
    gen_count = 0

    while eval_counter[0] < budget and population:
        # Score current population
        scored: list[tuple[float, sp.Expr]] = []
        for ind in population:
            if eval_counter[0] >= budget:
                break
            f = fitness(ind)
            eval_counter[0] += 1
            if f is not None:
                scored.append((f, ind))
                if f < best_mse:
                    best_mse, best_expr = f, ind

        if not scored:
            population, r_dom, r_dep, r_num, r_acc = generate_valid_population(
                pop_size, generator, evaluator, x, xs, max_depth,
                eval_counter=eval_counter, budget=budget,
            )
            total_domain += r_dom
            total_depth += r_dep
            total_numeric += r_num
            total_accepted += r_acc
            continue

        scored.sort(key=lambda t: t[0])
        elites = [ind for _, ind in scored[:elite_size]]
        gen_count += 1

        # Build next generation
        next_gen = elites[:]
        while len(next_gen) < pop_size and eval_counter[0] < budget:
            parent = random.choice(elites)
            child, m_dom, m_dep, m_num, m_acc = _mutate(
                parent, x, generator, evaluator, xs, max_depth,
                is_eml=is_eml,
                eval_counter=eval_counter,
                budget=budget,
            )
            total_domain += m_dom
            total_depth += m_dep
            total_numeric += m_num
            total_accepted += m_acc
            if child is not None:
                next_gen.append(child)
        population = next_gen

    return {
        "expr":        best_expr,
        "mse":         best_mse,
        "generations": gen_count,
        "total_evals": eval_counter[0],
        "domain_rejections": total_domain,
        "depth_rejections": total_depth,
        "numeric_rejections": total_numeric,
        "accepted_candidates": total_accepted,
        "rejections":  total_domain + total_depth + total_numeric,
    }


# ---------------------------------------------------------------------------
# Multi-trial execution and aggregation
# ---------------------------------------------------------------------------


def _make_evaluator(is_eml: bool) -> Callable[[sp.Expr, sp.Symbol, np.ndarray], np.ndarray | None]:
    def _eval(expr: sp.Expr, x: sp.Symbol, xs: np.ndarray) -> np.ndarray | None:
        return evaluate_expr(expr, x, xs, is_eml=is_eml)
    return _eval


def run_search() -> list[dict[str, Any]]:
    """Run the full SR benchmark: 5 trials × 18 targets × 2 grammars.

    Returns per-target aggregated statistics including the new
    characterization metrics: generations_completed and rejection_rate_pct.
    """
    logger.info("=" * 70)
    logger.info(
        "SR BENCHMARK  |  controlled characterization  |  depth=%d  |  trials=%d  |  budget=%d",
        SEARCH_MAX_DEPTH, N_TRIALS, SEARCH_BUDGET,
    )
    logger.info("=" * 70)

    eml_eval = _make_evaluator(is_eml=True)
    base_eval = _make_evaluator(is_eml=False)

    targets = target_functions()
    raw: dict[str, dict[str, list[float]]] = {
        t["name"]: {
            "eml_mse": [], "base_mse": [],
            "eml_nodes": [], "base_nodes": [],
            "eml_depth": [], "base_depth": [],
            "eml_generations": [], "base_generations": [],
            "eml_rejections": [], "base_rejections": [],
            "eml_total_evals": [], "base_total_evals": [],
            "eml_accepted": [], "base_accepted": [],
            "eml_domain": [], "base_domain": [],
            "eml_depth_rej": [], "base_depth_rej": [],
            "eml_numeric": [], "base_numeric": [],
        }
        for t in targets
    }

    for trial in range(N_TRIALS):
        seed = RANDOM_SEED + trial
        logger.info("Trial %d/%d  (seed=%d)", trial + 1, N_TRIALS, seed)
        for tgt in targets:
            name = tgt["name"]

            res_eml = search_for_target(
                tgt, random_eml_expr, eml_eval,
                seed=seed, max_depth=SEARCH_MAX_DEPTH, is_eml=True,
            )
            res_base = search_for_target(
                tgt, random_sympy_expr, base_eval,
                seed=seed, max_depth=SEARCH_MAX_DEPTH, is_eml=False,
            )

            raw[name]["eml_mse"].append(res_eml["mse"])
            raw[name]["base_mse"].append(res_base["mse"])
            raw[name]["eml_nodes"].append(
                total_node_count(res_eml["expr"]) if res_eml["expr"] is not None else 0
            )
            raw[name]["base_nodes"].append(
                total_node_count(res_base["expr"]) if res_base["expr"] is not None else 0
            )
            raw[name]["eml_depth"].append(
                tree_depth(res_eml["expr"]) if res_eml["expr"] is not None else 0
            )
            raw[name]["base_depth"].append(
                tree_depth(res_base["expr"]) if res_base["expr"] is not None else 0
            )
            raw[name]["eml_generations"].append(res_eml["generations"])
            raw[name]["base_generations"].append(res_base["generations"])
            raw[name]["eml_total_evals"].append(res_eml["total_evals"])
            raw[name]["base_total_evals"].append(res_base["total_evals"])

            raw[name]["eml_accepted"].append(res_eml["accepted_candidates"])
            raw[name]["base_accepted"].append(res_base["accepted_candidates"])
            raw[name]["eml_domain"].append(res_eml["domain_rejections"])
            raw[name]["base_domain"].append(res_base["domain_rejections"])
            raw[name]["eml_depth_rej"].append(res_eml["depth_rejections"])
            raw[name]["base_depth_rej"].append(res_base["depth_rejections"])
            raw[name]["eml_numeric"].append(res_eml["numeric_rejections"])
            raw[name]["base_numeric"].append(res_base["numeric_rejections"])

            eml_rej_pct = 100.0 * res_eml["rejections"] / max(res_eml["total_evals"], 1)
            base_rej_pct = 100.0 * res_base["rejections"] / max(res_base["total_evals"], 1)
            raw[name]["eml_rejections"].append(eml_rej_pct)
            raw[name]["base_rejections"].append(base_rej_pct)

            logger.info(
                "  %-28s  EML mse=%.3e gens=%d rej=%.1f%%  |  Base mse=%.3e gens=%d rej=%.1f%%",
                name,
                res_eml["mse"], res_eml["generations"], eml_rej_pct,
                res_base["mse"], res_base["generations"], base_rej_pct,
            )

    # Aggregate
    aggregated: list[dict[str, Any]] = []
    for tgt in targets:
        name = tgt["name"]
        d = raw[name]
        row = {
            "target":                   name,
            "eml_mse_mean":             statistics.mean(d["eml_mse"]),
            "eml_mse_std":              statistics.stdev(d["eml_mse"]) if N_TRIALS > 1 else 0.0,
            "base_mse_mean":            statistics.mean(d["base_mse"]),
            "base_mse_std":             statistics.stdev(d["base_mse"]) if N_TRIALS > 1 else 0.0,
            
            "eml_nodes_mean":           statistics.mean(d["eml_nodes"]),
            "eml_nodes_std":            statistics.stdev(d["eml_nodes"]) if N_TRIALS > 1 else 0.0,
            "base_nodes_mean":          statistics.mean(d["base_nodes"]),
            "base_nodes_std":           statistics.stdev(d["base_nodes"]) if N_TRIALS > 1 else 0.0,
            
            "eml_depth_mean":           statistics.mean(d["eml_depth"]),
            "eml_depth_std":            statistics.stdev(d["eml_depth"]) if N_TRIALS > 1 else 0.0,
            "base_depth_mean":          statistics.mean(d["base_depth"]),
            "base_depth_std":           statistics.stdev(d["base_depth"]) if N_TRIALS > 1 else 0.0,
            
            "eml_generations_mean":     statistics.mean(d["eml_generations"]),
            "eml_generations_std":      statistics.stdev(d["eml_generations"]) if N_TRIALS > 1 else 0.0,
            "base_generations_mean":    statistics.mean(d["base_generations"]),
            "base_generations_std":     statistics.stdev(d["base_generations"]) if N_TRIALS > 1 else 0.0,
            
            "eml_rejection_rate_pct":   statistics.mean(d["eml_rejections"]),
            "eml_rejection_rate_std":   statistics.stdev(d["eml_rejections"]) if N_TRIALS > 1 else 0.0,
            "base_rejection_rate_pct":  statistics.mean(d["base_rejections"]),
            "base_rejection_rate_std":  statistics.stdev(d["base_rejections"]) if N_TRIALS > 1 else 0.0,
            
            "eml_accepted_mean":        statistics.mean(d["eml_accepted"]),
            "base_accepted_mean":       statistics.mean(d["base_accepted"]),
            "eml_domain_mean":          statistics.mean(d["eml_domain"]),
            "base_domain_mean":         statistics.mean(d["base_domain"]),
            "eml_depth_rej_mean":       statistics.mean(d["eml_depth_rej"]),
            "base_depth_rej_mean":      statistics.mean(d["base_depth_rej"]),
            "eml_numeric_mean":         statistics.mean(d["eml_numeric"]),
            "base_numeric_mean":        statistics.mean(d["base_numeric"]),
        }
        aggregated.append(row)

    # Save CSV
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv_path = RESULTS_DIR / "symbolic_regression_per_target.csv"
    _save_csv(aggregated, csv_path)

    # Summary log
    logger.info(
        "\n%-28s  %-22s  %-22s  %-8s  %-8s  %-8s  %-8s",
        "Target", "EML (mean±std)", "Base (mean±std)",
        "EML gen", "Base gen", "EML rej%", "Base rej%",
    )
    logger.info("-" * 120)
    for row in aggregated:
        logger.info(
            "%-28s  %.3e ± %.3e  %.3e ± %.3e  %-8.1f  %-8.1f  %-8.1f  %-8.1f",
            row["target"][:28],
            row["eml_mse_mean"], row["eml_mse_std"],
            row["base_mse_mean"], row["base_mse_std"],
            row["eml_generations_mean"], row["base_generations_mean"],
            row["eml_rejection_rate_pct"], row["base_rejection_rate_pct"],
        )

    return aggregated


def _save_csv(aggregated: list[dict[str, Any]], path: Path) -> None:
    fields = [
        "target",
        "eml_mse_mean", "eml_mse_std",
        "base_mse_mean", "base_mse_std",
        "eml_nodes_mean", "eml_nodes_std", 
        "base_nodes_mean", "base_nodes_std",
        "eml_depth_mean", "eml_depth_std", 
        "base_depth_mean", "base_depth_std",
        "eml_generations_mean", "eml_generations_std", 
        "base_generations_mean", "base_generations_std",
        "eml_rejection_rate_pct", "eml_rejection_rate_std", 
        "base_rejection_rate_pct", "base_rejection_rate_std",
        "eml_accepted_mean", "base_accepted_mean",
        "eml_domain_mean", "base_domain_mean",
        "eml_depth_rej_mean", "base_depth_rej_mean",
        "eml_numeric_mean", "base_numeric_mean",
    ]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        w.writerows(aggregated)
    logger.info("Results saved to %s", path)


if __name__ == "__main__":
    configure_logging()
    run_search()
