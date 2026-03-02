# ACE: Adapting Sampling for Counterfactual Explanations (Python)

This repository contains Python code associated with the papers:

1) **ACE: Adapting Sampling for Counterfactual Explanations** (accepted at **L4DC 2026**)
   Margarita A. Guerrero, Cristian R. Rojas  
📄 [arXiv link](https://arxiv.org/abs/2509.26322)  
✅ Accepted for publication in *IEEE Control Systems Letters (LCSS), 2025 Edition*.

2) **Sample-Efficient Counterfactual Tuning for Compressor Pressure Control** (industrial case study)
   Margarita A. Guerrero, Rodrigo A. González, Cristian R. Rojas
   📄 [arXiv link](https://arxiv.org/abs/2512.03747)  


## Overview

This code implements **ACE: Adaptive Sampling for Counterfactual Explanations**, a sample-efficient method to compute a **Counterfactual Explanation (CFE)** for **classification** tasks.

Given a trained black-box classifier \( h: \mathcal{X} \to \{0,1\} \) and a fixed input instance \( \tilde{x} \), ACE aims to find the closest point \( x \) that **induces a classification flip**, i.e., \( h(x) \neq h(\tilde{x}) \). In other words, ACE returns the **smallest change needed in the input feature space** to change the classifier outcome.

ACE is designed for settings where **queries to the black-box model are limited or expensive**, and it adaptively samples informative points using a Gaussian-process surrogate and Bayesian optimization.

In addition to standard ML benchmarks, ACE is used as a **controller retuning engine** in a safety-critical industrial-inspired setting: the algorithm searches for the smallest actionable change in controller parameters that flips the outcome from “fail” to “pass” with a small number of closed-loop experiments.


## Contents

- **Main implementation**
  - `ACE_Github.py`  (main script; includes a runnable example at the end)

- **Datasets**
  - `datasets/`  (CSV datasets used in the seminal ACE paper and the tuning case study)

- **Support code for the compressor tuning case**
  - `LDG.py`         (bounds / utilities for PIDF tuning)
  - `LDG_cascade.py` (bounds / utilities for cascade PIDF tuning)

> Note: `ACE_Github.py` imports these helper modules from the same folder, and reads datasets from `datasets/<name>.csv`.


## Installation

A typical setup is:

```bash
pip install numpy scipy scikit-learn matplotlib pandas pyDOE scikit-image
