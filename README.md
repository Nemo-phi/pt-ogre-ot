# Computational Optimal Transport

M2 PT OGRE Research Project 2026–2027

## Topics

1. Discrete Optimal Transport
2. Entropic OT and Sinkhorn Algorithms
3. Wasserstein Gradient Flows and JKO
4. Crowd Motion with Congestion

## Structure

- `otlib/`: Algorithms and numerical methods
- `tests/`: Unit tests
- `notebooks/`: Numerical experiments

## Installation

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

## Phase 1 numerical tasks 1.1 and 1.2

- [Solver](otlib/transport.py) computes the transport plan and dual prices.
- [Notebook](notebooks/transport.ipynb) shows the examples, plot, and numerical results.
- [Transport tests](tests/test_transport.py) check the plan, cost, and route count.

Open the notebook with the environment where the package is installed.
Run the transport tests from the repository root with `python3 -m pytest -q`.
