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

