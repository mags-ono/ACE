# ACE: Adapting Sampling for Counterfactual Explanations (Python)

This repository contains Python code associated with the papers:

  **“ACE: Adapting Sampling for Counterfactual Explanations”**  
  Margarita A. Guerrero,Cristian R. Rojas  
  📄 [arXiv link](https://arxiv.org/abs/2509.26322)  
  ✅ Accepted for publication in *8th Annual Learning for Dynamics & Control Conference, 2026 Edition*.
  
  **“Sample-Efficient Counterfactual Tuning for Compressor Pressure Control”**  
  Margarita A. Guerrero, Braghadeesh Lakshminarayanan, Cristian R. Rojas  
  📄 [arXiv link](https://arxiv.org/abs/2512.03747)  

## Overview

This code implements **ACE: Adaptive Sampling for Counterfactual Explanations**, a sample-efficient method to compute a **Counterfactual Explanation (CFE)** for **classification** tasks.

Given a trained black-box classifier $h: \mathcal{X} \to \{0,1\}$ and a fixed input instance $\tilde{x}$, ACE aims to find the closest point $x$ that **induces a classification flip**, i.e., $h(x) \neq h(\tilde{x})$. In other words, ACE returns the **smallest change needed in the input feature space** to change the classifier outcome.

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


## Main Functions

Below is the key function used to run ACE end-to-end.

## Main Function

Below is the key function used to run ACE end-to-end.

### `iterative_main_loop`

**Signature**
```python
iterative_main_loop(
    dataset,
    target,
    initial_instance_index,
    target_val,
    ini_ponts,
    seed=0,
    action=None
)
```
**Inputs (most important)**
- `dataset` (str): dataset name (expects `datasets/<dataset>.csv`)
  - examples: `"pidf"`, `"pid_cascade"`, `"diabetes"`, etc.
- `target` (`Target`): target specification (classification)
- `initial_instance_index` (int): index of the factual instance \(\tilde{x}\) in the dataset
- `target_val` (int / str): desired target class (e.g., `1`)
- `ini_ponts` (int): initial number of samples \(\mathcal{H}\) used to initialize the search
- `seed` (int): random seed
- `action` (list[int] or empty list): indices of **immutable/frozen** features  
  - if `action=[]`, then **all features are actionable**

**Outputs (most important)**
Returns a tuple:
- `output[0]`: a **dict** containing the run results, including (at least)
  - `instance`: the factual instance \(\tilde{x}\)
  - `x_s`: the computed counterfactual \(x\) (closest found)
  - `elapsed_time`: wall-clock time
  - `X`: sampled points across iterations
- `output[1]`: the **predictor** (black-box evaluator used for the selected dataset)


## Example Included: PID Cascade Tuning

At the end of `ACE_Github.py`, a minimal executable example computes the closest CFE for the cascade PID tuning dataset.

The example uses the factual controller parameter vector:

\[
\theta_0 = [[8.4533, 0.1358, 1.7270, 0.0104, 3.9855, 2.8415, 0.2635, 0.0200]]
\]
corresponding to:
\[
(Kp_1, Ki_1, Kd_1, Tf_1, \; Kp_2, Ki_2, Kd_2, Tf_2)
\]

This example obtains the CFE for:
- `dataset="pid_cascade"`
- `ind=47` (instance index; corresponds to \(\theta_{0,1}\) in *Sample-Efficient Counterfactual Tuning for Compressor Pressure Control*)
- `target_val=1`
- `inip=30` (initial number of samples \(\mathcal{H}\))
- `action=[]` (all 8 parameters are actionable)

Copy-paste snippet:

```python
inip = 30
seed = 0

# PIDF
# dataset = "pidf"
# ind = 162

# Cascade
dataset = "pid_cascade"
ind = 47

action = []
target_val = 1
t = Target(target_type="classification", target_feature="Class", target_value=target_val)

output = iterative_main_loop(dataset, t, ind, target_val, inip, seed, action)

result = output[0]   # dict
knn    = output[1]   # predictor
X = result["X"][inip+1:]

print("Result index", ind, ":")
print("instance:", fmt4(result["instance"]))
print("x_s:", fmt4(result["x_s"]))
print("Elapsed_time:", fmt4(result["elapsed_time"]))
