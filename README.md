# EML Framework: Operator-Space Constraints in Symbolic Regression

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.19991157.svg)](https://doi.org/10.5281/zenodo.19991157)

A controlled empirical study of how constraining symbolic regression to a single nonlinear primitive reshapes symbolic structure and search behaviour.

> **This is a characterization study, not a performance evaluation.**
> No claims of computational superiority are made.

## What This Studies

The EML operator, `eml(x, y) = exp(x) − ln(y)`, was shown by [Odrzywołek (2026)](https://arxiv.org/abs/2603.21852) to generate all elementary functions from a single binary primitive over the complex field. This work asks: what happens when you enforce this restriction in a real-valued symbolic regression system where logarithm arguments must be strictly positive and complex identities are unavailable?

We measure three things:

1. **Rejection dynamics** — how many candidates fail domain or depth checks under each grammar
2. **Structural inflation** — how much larger EML-encoded trees become compared to standard exp/log
3. **Search efficiency** — how many productive generations a fixed evaluation budget allows

## Key Findings

| Property | EML grammar | Standard exp/log |
|---|---|---|
| Rejection rate | ~90% | ~70% |
| Avg. tree size (best found) | 5–6× larger | baseline |
| Operator variety | 1 nonlinear type | multiple types |

- Operator restriction produces **sparser valid search space** (higher rejection rate)
- EML **homogenizes** the nonlinear layer but **inflates** node count and depth
- Domain constraints are a **primary structural force**, not a peripheral inconvenience
- Rejection rates are treated as **characterization findings**, not hidden confounds

## Scope & Limitations

- Restricted to the positive-domain exp–log fragment
- Addition remains external to the EML primitive
- Trigonometric, hyperbolic, and complex-field functions are excluded
- The evolutionary search is deliberately simple — results describe search-space structure, not performance ceilings
- Depth cap at 6 may disproportionately constrain the EML grammar

## Repository Structure

```
src/eml/              Core library
  ├── core.py          EML primitives and structural metrics
  ├── transform.py     Recursive domain-enforcing rewriter
  ├── validation.py    Symbolic + numeric domain validation
  ├── config.py        Experiment configuration and constants
  ├── exceptions.py    DomainError, UnsupportedExpressionError
  └── logging_config.py
experiments/           Experiment scripts
  ├── symbolic_regression.py   Grammar-pure SR with true budget accounting
  ├── depth_size_analysis.py   Depth/size vs error + success rates
  ├── domain_ablation.py       Strict vs relaxed domain enforcement
  ├── validate_numeric.py      Numeric correctness validation
  ├── domain_validation_showcase.py  Domain violation diagnostics
  ├── benchmarks.py            Benchmark expression catalogue
  └── run_all.py               Run full experiment suite
paper/                 Manuscript and figures
  ├── eml_representation_study_aicta2026.tex  AICTA submission source
  ├── generate_aicta_figures.py               Generate paper figures
  └── figures/                               Generated figures
tests/                 Unit tests covering all public interfaces
results/               Generated CSV data (gitignored)
```

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
PYTHONPATH=src pytest tests/
PYTHONPATH=src python -m experiments.run_all     # full benchmark suite
python paper/generate_aicta_figures.py             # reproduce paper figures
```

## Reproducing the Paper

```bash
# Generate all experimental data
PYTHONPATH=src python -m experiments.run_all

# Generate figures
python paper/generate_aicta_figures.py

# Compile AICTA submission
cd paper && pdflatex eml_representation_study_aicta2026.tex && pdflatex eml_representation_study_aicta2026.tex
```

## Citation

If you use this code in your work, please cite:

```bibtex
@software{eml_framework_2026,
  author    = {Nath, Biswajyoti},
  title     = {{EML Framework: Operator-Space Constraints in
                Symbolic Regression}},
  year      = {2026},
  publisher = {Zenodo},
  doi       = {10.5281/zenodo.19991157},
  url       = {https://doi.org/10.5281/zenodo.19991157}
}
```

Also cite the original theoretical work:

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

## License

See [LICENSE](LICENSE).
