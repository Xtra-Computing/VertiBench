# VertiBench: Vertical Federated Learning Benchmark

## Introduction

VertiBench is a benchmark for [federated learning](https://ieeexplore.ieee.org/abstract/document/9599369/), [split learning](https://arxiv.org/abs/1912.12115), and [assisted learning](https://proceedings.neurips.cc/paper_files/paper/2022/hash/4d6938f94ab47d32128c239a4bfedae0-Abstract-Conference.html) on vertical partitioned data. It provides tools to synthetic vertical partitioned data from a given global dataset. VertiBench supports partition under various **imbalance** and **correlation** level, effectively simulating a wide-range of real-world vertical federated learning scenarios. 


![data-dist-full.png](fig%2Fdata-dist-full.png)

## Installation

VertiBench has already been published on PyPI. The installation requires the installation of `python>=3.9`. To further install VertiBench, run the following command:

```bash
pip install vertibench
```

## Getting Started

This examples includes the pipeline of split and evaluate. First,
 load your datasets or generate synthetic datasets. 

```python
from sklearn.datasets import make_classification

# Generate a large dataset
X, y = make_classification(n_samples=10000, n_features=10)
```

To split the dataset by importance,

```python
from vertibench.Splitter import ImportanceSplitter

imp_splitter = ImportanceSplitter(num_parties=4, weights=[1, 1, 1, 3])
Xs = imp_splitter.split(X)
```

To split the dataset by correlation,

```python
from vertibench.Splitter import CorrelationSplitter

corr_splitter = CorrelationSplitter(num_parties=4)
Xs = corr_splitter.fit_split(X)
```

To evaluate a feature split `Xs` in terms of party importance,

```python
from vertibench.Evaluator import ImportanceEvaluator
from sklearn.linear_model import LogisticRegression
import numpy as np

model = LogisticRegression()
X = np.concatenate(Xs, axis=1)
model.fit(X, y)
imp_evaluator = ImportanceEvaluator()
imp_scores = imp_evaluator.evaluate(Xs, model.predict)
alpha = imp_evaluator.evaluate_alpha(scores=imp_scores)
print(f"Importance scores: {imp_scores}, alpha: {alpha}")
```

To evaluate a feature split in terms of correlation,

```python
from vertibench.Evaluator import CorrelationEvaluator

corr_evaluator = CorrelationEvaluator()
corr_scores = corr_evaluator.fit_evaluate(Xs)
beta = corr_evaluator.evaluate_beta()
print(f"Correlation scores: {corr_scores}, beta: {beta}")
```