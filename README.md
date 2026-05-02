# EML Framework: A Representation Study of the Exp-Minus-Log Operator

A research implementation studying the structural and computational behavior
of the **EML operator** — `eml(x, y) = exp(x) − ln(y)` — under practical,
real-domain constraints. We measure how enforcing a single nonlinear
primitive reshapes symbolic structure, tree depth, and search-space topology
across a 23-expression benchmark suite and an exploratory symbolic regression
experiment.

> **Scope statement.** This repository implements a *restricted*,
> *real-domain* instantiation of the EML operator. It is a representation
> study, not a universal symbolic computation system.

**Quick start**

```bash
pip install -e ".[dev]"
pytest tests/            # 67 tests, all pass
python paper/generate_figures.py   # reproduces paper figures → paper/figures/
```

---

## Background

Odrzywołek (2026) shows that the binary operator
([arXiv:2603.21852](https://arxiv.org/abs/2603.21852))

```
eml(x, y) = exp(x) − ln(y)
```

together with the constant `1`, is sufficient to generate the full repertoire
of elementary functions — including arithmetic, transcendentals, and algebraic
functions — as uniform binary trees under the grammar
`S → 1 | eml(S, S)` [[arXiv:2603.21852]](#citation).

**This repository does not implement that universal system.**

It studies a *constrained subset*: the positive-domain exp–log fragment,
in which the above operator is applied to expressions involving `exp`, `ln`,
`Abs`, `Mul`, and integer and rational `Pow`, under the assumption that all
`ln` arguments and non-integer power bases are strictly positive. The
Schrödinger operator `H[ψ] = −ψ″ + V·ψ` is used as the primary test case
for transformation and structural analysis.

---

## Positioning

This project:

- studies the **structural and representational** effects of enforcing the
  EML primitive on a restricted exp–log algebra
- **does not** implement the full universal system described in the original
  paper
- **does not** claim universality, completeness, or performance superiority
- operates strictly within a **real-valued, positive-domain** fragment
- treats symbolic regression as an **exploratory** comparison of search-space
  behavior, not a claim of algorithmic advantage

---

## Key Idea

Within the supported fragment, standard exp–log operations are encoded as
EML expressions:

| Operation | EML encoding | Domain constraint |
|---|---|---|
| `exp(f(x))` | `eml(f(x), 1)` | none |
| `ln(f(x))` | `1 − eml(0, f(x))` | `f(x) > 0` |
| `a · b` | `exp(ln(a) + ln(b))` | `a, b > 0` |
| `a / b` | `exp(ln(a) − ln(b))` | `a, b > 0` |
| `xⁿ` (integer *n*) | `sign(x)ⁿ · exp(n · ln(\|x\|))` | none |
| `xᵃ` (non-integer *a*) | `exp(a · ln(x))` | `x > 0` |

Addition is **not encoded** inside the EML primitive; it is preserved as the
ambient additive combinator outside the rewrite.

The encoding yields uniform binary tree structures amenable to structural
comparison. This repository measures the effect of that uniformity on
tree-level complexity metrics.

---

## What This Repository Does

- **Transforms** supported symbolic expressions (SymPy) into EML form via
  a recursive rewrite over the exp–log fragment
- **Evaluates** transformed expressions back to standard `exp`/`ln` form to
  verify semantic preservation
- **Validates** correctness through symbolic domain checks (SymPy assumption
  system) combined with sampled numeric evaluation
- **Analyzes** structural complexity via node counts, nonlinear node counts,
  tree depth, unique subexpression counts, and a weighted circuit cost model
- **Compares** EML representations against a standard exp/log rewrite
  baseline across a 23-expression benchmark suite
- **Explores** EML search-space behavior in a simple evolutionary symbolic
  regression setting, compared against a SymPy grammar baseline

---

## What This Repository Does NOT Do

- Does **not** implement the full universal EML system of Odrzywołek (2026)
- Does **not** encode addition inside the EML primitive
- Does **not** support trigonometric, hyperbolic, or other non-exp-log
  elementary functions
- Does **not** operate on the complex domain
- Does **not** lift domain restrictions on `ln` arguments or non-integer bases
- Is **not** a complete machine learning or symbolic regression system
- Makes **no** performance or efficiency claims relative to alternative
  symbolic representations

---

## Supported Fragment

The rewrite system is restricted to a **positive-domain exp–log algebra**.

**Supported input types:**

- `exp(·)`, `ln(·)`, `Abs(·)`
- `Add`, `Mul`, integer and rational `Pow`
- `Derivative` (evaluated via `doit()` before transformation)

**Domain constraints enforced at transform time:**

- All `ln` arguments must be provably positive (SymPy assumption system).
  Violation raises `DomainError`.
- All non-integer `Pow` bases must be provably positive.
  Violation raises `NotImplementedError`.
- All `Mul` operands must be provably positive.
  Violation raises `DomainError`.

**Unsupported functions:**

Any function outside the above set (e.g., `sin`, `cos`, `gamma`) raises
`UnsupportedExpressionError` immediately. There is no silent pass-through.

**Note on symbolic positivity.** SymPy's assumption system is conservative.
A symbol `x` without an explicit positivity assumption is treated as
unbounded; expressions involving such symbols may fail domain checks even
when they are safe on a restricted input range. Use `validate_domain` with
explicit sample points for numeric verification in such cases.

---

## Repository Structure

```
eml-framework/
│
├── src/
│   └── eml/
│       ├── __init__.py          ← full public API re-export
│       ├── core.py              ← EML primitives and complexity metrics
│       ├── transform.py         ← recursive EML rewrite + exp/log baseline
│       ├── schrodinger.py       ← 1D Schrödinger operator helpers
│       ├── validation.py        ← domain safety checks; assert_domain()
│       ├── exceptions.py        ← DomainError, UnsupportedExpressionError
│       ├── config.py            ← experiment parameters and cost constants
│       └── logging_config.py    ← configure_logging() entrypoint
│
├── experiments/
│   ├── benchmarks.py            ← 23-expression benchmark catalogue
│   ├── validate_numeric.py      ← numeric equivalence + domain validation
│   ├── complexity_analysis.py   ← structural metric comparison
│   └── symbolic_regression.py  ← evolutionary search: EML vs baseline grammar
│
├── paper/
│   ├── eml_representation_study.tex  ← full LaTeX paper source
│   ├── generate_figures.py           ← reproduces all paper figures from scratch
│   └── figures/
│       ├── complexity_comparison.png ← Fig. 1: structural metrics bar chart
│       └── regression_results.png   ← Fig. 2: symbolic regression MSE + tree size
│
├── tests/
│   ├── conftest.py              ← shared fixtures (x symbol, sample grids)
│   ├── test_core.py             ← 27 unit tests: primitives and metrics
│   ├── test_transform.py        ← transform correctness and error semantics
│   ├── test_validation.py       ← 22 unit tests: domain validation API
│   └── test_symbolic_regression.py
│
├── notebooks/
│   └── demo.ipynb
│
├── pyproject.toml               ← package config; black/ruff/isort/mypy settings
├── requirements.txt             ← runtime dependencies
├── requirements-dev.txt         ← development dependencies
└── .pre-commit-config.yaml      ← pre-commit hooks
```

---

## Installation

```bash
# Runtime only
pip install -r requirements.txt

# Development (testing, linting, type checking)
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

**Requirements:** Python ≥ 3.10, SymPy ≥ 1.11, NumPy ≥ 1.24,
Matplotlib ≥ 3.7.

---

## Usage

```python
import sympy as sp
from eml import (
    transform,
    evaluate_eml,
    assert_domain,
    DomainError,
    original_schrodinger,
    eml_schrodinger,
)

x = sp.symbols("x")

# --- Basic transformation ---
expr = sp.exp(x)
eml_expr  = transform(expr)       # → eml(x, 1)
recovered = evaluate_eml(eml_expr)  # → exp(x)  [semantic round-trip]

# --- Schrödinger test case ---
# H[ψ] = −ψ″ + x²·ψ,  ψ = exp(−x²)
H_orig = original_schrodinger()   # standard SymPy expression
H_eml  = eml_schrodinger()        # same, rewritten in EML form

# --- Domain enforcement ---
try:
    assert_domain(sp.log(x - 5))  # x − 5 not provably positive
except DomainError as e:
    print(e)  # explicit rejection; no silent failure

# --- Combined symbolic + numeric domain check ---
from eml import validate_domain
import numpy as np

xs = np.linspace(0.1, 5.0, 200)
result = validate_domain(sp.log(x + 1), x, sample_points=xs)
# {'valid_domain': True, 'symbolic_safe': ..., 'numeric_ok': True, ...}
```

---

## Experiments

All experiments are framed as **representation analysis**. No experiment
makes performance, efficiency, or approximation-quality claims.

### Numeric Validation

Verifies that the EML-encoded and semantically evaluated expression matches
the original expression numerically across 23 benchmark expressions and
multiple input ranges. Reports mean and maximum absolute error where both
forms are domain-safe.

```bash
python -m experiments.validate_numeric
```

### Structural Complexity Comparison

Computes tree-level structural metrics — nonlinear node count, unique
subexpression count, tree depth, and weighted circuit cost — for EML
encodings versus standard exp/log rewrites across the benchmark suite.

```bash
python -m experiments.complexity_analysis
```

### Symbolic Regression (Exploratory)

Applies an elite-retention evolutionary search to find expressions
approximating five target functions, comparing the EML grammar against a
standard SymPy exp/log grammar. Results report MSE and tree size.
This experiment examines **search-space structure**, not learning efficacy.

```bash
python -m experiments.symbolic_regression
```

### Paper Figures

Reproduces both figures used in the paper (`paper/figures/`) from scratch.
Runs the benchmark complexity analysis and the full symbolic regression
experiment, then saves publication-ready PNGs.

```bash
python paper/generate_figures.py
# → paper/figures/complexity_comparison.png
# → paper/figures/regression_results.png
```

---

## Testing

```bash
pytest tests/ -v
```

```
67 passed in 0.59s
```

| Test module | Count | Coverage |
|---|---|---|
| `test_core.py` | 27 | EML primitives, evaluate\_eml round-trips, all metrics |
| `test_validation.py` | 22 | assert\_domain, is\_domain\_safe, validate\_domain, validate\_log\_domain |
| `test_transform.py` | 15 | transform correctness, DomainError, UnsupportedExpressionError, edge cases |
| `test_symbolic_regression.py` | 3 | candidate evaluation, target function schema |

---

## Code Quality

```bash
black --check src/ tests/ experiments/   # PEP 8 formatting (100 col)
ruff check src/ tests/ experiments/      # linting (E/W/F/B/C4/SIM/UP)
isort --check-only src/ tests/           # import ordering
mypy src/eml/                            # static type checking
```

All checks pass. Full type annotations on all public functions.
Target: Python 3.10+.

---

## Configuration

All experiment parameters are centralized in `src/eml/config.py`.
No magic numbers appear in experiment or source files.

| Constant | Default | Description |
|---|---|---|
| `RANDOM_SEED` | 42 | Global seed for all experiments |
| `SEARCH_BUDGET` | 4 000 | Max candidate evaluations per search run |
| `SEARCH_POP_SIZE` | 100 | Evolutionary search population size |
| `SEARCH_ELITE_SIZE` | 15 | Elite candidates retained per generation |
| `SEARCH_MAX_DEPTH` | 4 | Maximum expression tree depth |
| `BENCHMARK_SAMPLE_POINTS` | 301 | Sample points for numeric validation |
| `EML_OPERATOR_COST` | 3 | Weighted circuit cost: EML node |
| `EXP_LOG_OPERATOR_COST` | 2 | Weighted circuit cost: exp/ln node |

---

## Limitations

- **Domain restrictions** are mandatory for all `ln` arguments and for
  non-integer `Pow` bases. Expressions involving unbounded symbols will fail
  symbolic positivity checks even when numerically safe on a restricted range.
- **Addition** is not encoded inside the EML primitive and remains an external
  operator. The EML-uniform-tree property of the original paper does not hold
  for additive expressions in this implementation.
- **Function coverage** is limited to the exp–log fragment.
  Trigonometric, hyperbolic, and other elementary functions are unsupported.
- **Symbolic positivity** is checked conservatively via SymPy's assumption
  system. This may produce false negatives for expressions whose positivity
  depends on domain restrictions not captured by SymPy's global assumptions.
- **Symbolic regression** is exploratory. The evolutionary search is a basic
  elite-retention strategy; results reflect structural search-space differences,
  not algorithmic optimization claims.
- **`transform_pow`** is guaranteed only for integer exponents and for
  non-integer exponents with provably positive bases. The general non-integer,
  non-positive case is explicitly rejected.

---

## Relation to Prior Work

Odrzywołek (2026) establishes that `eml(x, y) = exp(x) − ln(y)` together
with the constant `1` is a *universal* binary operator for elementary
functions over the complex field, and demonstrates gradient-based symbolic
regression using EML trees as differentiable circuits.

This repository studies a **practical, restricted instantiation** of that
operator:

- over the real-valued, positive-domain exp–log fragment only
- without the complex-field universality result
- without gradient-based optimization
- with the specific goal of analyzing structural representation differences
  in a controlled software engineering context

The theoretical contribution belongs entirely to the original paper.

---

## Citation

If you use this repository in research that references the EML operator,
please cite the original work:

```bibtex
@misc{odrzywolek2026eml,
  title         = {All elementary functions from a single binary operator},
  author        = {Andrzej Odrzywo{\l}ek},
  year          = {2026},
  eprint        = {2603.21852},
  archivePrefix = {arXiv},
  primaryClass  = {cs.SC},
  doi           = {10.48550/arXiv.2603.21852},
  url           = {https://arxiv.org/abs/2603.21852}
}
```

---

## License

See [LICENSE](LICENSE).
