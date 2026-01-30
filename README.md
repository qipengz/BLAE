# Bi-Lipschitz Autoencoder (BLAE)

[![ICLR 2026](https://img.shields.io/badge/ICLR-2026-blue.svg)](https://iclr.cc/)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **"Bi-Lipschitz Autoencoder with Injectivity Guarantee"** (ICLR 2026).

## Overview

BLAE is a novel autoencoder framework that addresses the fundamental bottleneck of non-injective encoders in dimensionality reduction. By combining **injective regularization** with **bi-Lipschitz constraints**, BLAE achieves:

- ✅ **Superior manifold preservation** - Maintains intrinsic geometric structure
- ✅ **Robustness to distribution shifts** - Consistent embeddings across varying data distributions
- ✅ **Elimination of pathological local minima** - Avoids non-injective encoder collapse
- ✅ **Efficient dimensionality reduction** - Linear complexity O(m) vs quadratic O(m²) for isometric methods

## Key Features

### 1. Injective Regularization
Eliminates non-injective mappings through a separation criterion that prevents distant manifold regions from collapsing into nearby latent codes:
```math
L_{inj}(δ, ϵ) = E[ReLU(log(ϵ·d_M(x,y) / d_N(E(x),E(y)))) · 1_{d_M(x,y)>δ}]
```

### 2. Bi-Lipschitz Regularization
Ensures admissible geometric preservation while maintaining linear dimensionality scaling:
```math
L_{bi-Lip}(κ) = E[ReLU(1/κ - σ_{min}(J_D))² + ReLU(σ_{max}(J_D) - κ)²]
```

### 3. Combined Framework
```math
L_{BLAE} = L_{recon} + λ_{reg} · L_{reg} + λ_{bi-Lip} · L_{bi-Lip}
```

## Project Structure
```
BLAE/
├── data/
│   ├── data_class.py          # Dataset loaders
├── models/
│   ├── autoencoders.py        # Autoencoder models
│   ├── coders.py              # Encoder/decoder architectures
├── trainers/
│   ├── trainers.py            # Training loops
├── utils/
│   ├── datasets.py            # Data generation utilities
│   ├── functionals.py         # Loss functions
│   ├── kernels.py             # Kernel methods
│   ├── measures.py            # Evaluation metrics
│   ├── regularizations.py     # Regularization terms
│   └── visualization.py       # Plotting utilities
└── swiss_roll.ipynb           # Demo notebook
```
