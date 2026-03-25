# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

VertiBench is a Python library for benchmarking vertical federated learning (VFL). It generates synthetic VFL datasets with tunable feature importance imbalance and inter-party correlation, then evaluates the quality of vertical data partitions along those two dimensions.

## Build & Development Commands

```bash
# Install from source (editable)
pip install -e .

# Install with test dependencies (adds xgboost)
pip install -e ".[test]"

# Build distribution
python -m build

# Run all tests
python -m unittest discover test/

# Run individual test files
python -m unittest test.test_splitter
python -m unittest test.test_evaluator
python -m unittest test.test_evaluate_alpha

# Run a single test case
python -m unittest test.test_splitter.TestImportanceSplitter.test_split_tabular
```

No linter or formatter is configured for this project.

## Architecture

The library lives in `src/vertibench/` and has two core modules:

### Splitter.py — Vertical Data Partitioning

Abstract base class `Splitter` defines the interface: `split_indices()` returns per-party feature index lists, and `split()` applies them to datasets.

Three implementations:
- **ImportanceSplitter** — Uses Dirichlet distribution to assign features to parties with controllable importance imbalance. The `weights` parameter controls expected importance per party (higher weight = more features).
- **CorrelationSplitter** — Uses BRKGA (pymoo) genetic algorithm to find partitions that match a target inter/intra-party correlation ratio. Parameter `beta` ∈ [0,1] controls the balance. Requires `fit()` on data before splitting.
- **SimpleSplitter** — Uniform contiguous split of features across parties.

### Evaluator.py — Split Quality Assessment

- **ImportanceEvaluator** — Computes per-party feature importance using SHAP Permutation explainer. `evaluate_alpha()` recovers the Dirichlet concentration parameter from importance scores.
- **CorrelationEvaluator** — Computes correlation matrices and scores inner vs. inter-party correlation. `evaluate_beta()` recovers the correlation concentration metric. Supports GPU acceleration via PyTorch (`gpu_id` parameter). Uses multiple SVD strategies depending on feature count (exact for <100, randomized for larger).

### Key Data Flow

1. Generate data (e.g., `sklearn.datasets.make_classification`)
2. `Splitter.split(X)` → list of per-party feature matrices `Xs`
3. `Evaluator.evaluate(Xs, ...)` → quality scores
4. `evaluate_alpha()` / `evaluate_beta()` → concentration metrics

### Design Patterns

- `Splitter` uses ABC + template method: concrete classes implement `split_indices()`, base class handles `split()` logic.
- `CorrelationSplitter` composes a `CorrelationEvaluator` internally for optimization.
- Correlation computation has multiple backends: Spearman (pandas), Pearson (numpy/torch), with CPU/GPU variants.

## Testing

Tests use `unittest` with `subTest()` for parameterized variants. Test data is generated synthetically via `generate_data()` and `split_data()` helpers in each test file. The evaluator tests train actual XGBoost models, so the `[test]` extras are required.

## Dependencies

Key: numpy, scipy, scikit-learn, torch, shap, pymoo, matplotlib. Python >= 3.9.
