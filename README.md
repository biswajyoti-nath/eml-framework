# EML Framework

A research-oriented framework for encoding symbolic mathematical expressions using a single nonlinear operator (EML), with applications in symbolic regression and differentiable computation.

---

## 📌 Overview

This project implements a constructive framework based on the operator:


eml(x, y) = exp(x) - ln(y)


The core idea is that **all elementary functions can be represented using only this operator** through recursive composition.

This repository provides tools to:

* Transform symbolic expressions into EML trees
* Evaluate EML representations numerically
* Analyze structural complexity (depth, node count)
* Visualize expression trees

---

## 🚀 Features

* Symbolic → EML transformation (via SymPy)
* Numerical evaluation and equivalence checking
* Tree-based representation of expressions
* Visualization of EML computational graphs
* Complexity analysis (depth and size)

---

## 📁 Project Structure

```
eml-framework/
│
├── eml/
│   ├── core.py
│   ├── transform.py
│   ├── evaluate.py
│   ├── tree.py
│   └── visualize.py
│
├── experiments/
│   ├── correctness.py
│   └── complexity.py
│
├── notebooks/
│   └── demo.ipynb
│
├── tests/
│   ├── test_transform.py
│   └── test_evaluate.py
│
├── requirements.txt
└── README.md
```

---

## ⚙️ Installation

```bash
git clone https://github.com/biswajyoti-nath/eml-framework.git
cd eml-framework
pip install -r requirements.txt
```

---

## 🧠 Usage

### Example: Transform an Expression

```python
import sympy as sp
from eml.transform import to_eml

x = sp.symbols('x')
expr = x**2 + sp.exp(x)

eml_expr = to_eml(expr)
print(eml_expr)
```

---

### Example: Evaluate Equivalence

```python
from eml.evaluate import evaluate_pair

evaluate_pair(expr, eml_expr, x=1.5)
```

---

### Example: Visualize Tree

```python
from eml.visualize import plot_tree

plot_tree(eml_expr)
```

---

## 🔬 Experiments

### 1. Correctness

* Random expressions are generated
* Converted into EML form
* Evaluated numerically

Expected result:

* Numerical error ≈ 1e-6 or lower

Run:

```bash
python experiments/correctness.py
```

---

### 2. Complexity

Measures:

* Tree depth
* Node count

Shows exponential growth in naive expansion.

Run:

```bash
python experiments/complexity.py
```

---

## 📊 Key Insight

The EML representation is:

* **Expressive**: can encode all elementary functions
* **Uniform**: single operator across all expressions
* **Composable**: naturally forms tree structures

However:

* Naive expansion leads to exponential growth
* Tree/DAG representations are required for efficiency

---

## 🔧 Future Work

* DAG-based compression and subtree reuse
* Integration with symbolic regression models
* Differentiable EML trees for neural optimization
* GPU acceleration for large expression graphs

---

## 📄 Related Work

This implementation is inspired by recent theoretical work demonstrating that elementary functions can be constructed from a single operator.

---

## 📜 License

This project is released under the MIT License.

---

## 🤝 Contributing

Contributions are welcome. Please open an issue or submit a pull request for improvements or new features.

---

## ⭐ Citation

If you use this work, please cite:

```
@article{eml_framework,
  title={A Constructive Representation of the Schrödinger Operator Using a Single Nonlinear Primitive},
  author={Biswajyoti Nath},
  year={2026}
}
```

---

## ⚠️ Disclaimer

This is an early-stage research implementation. Some components may be experimental or subject to change.
