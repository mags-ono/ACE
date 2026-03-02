# ACE: Adapting Sampling for Counterfactual Explanations (Python)

This repository contains Python code associated with the papers:

  **“ACE: Adapting Sampling for Counterfactual Explanations”**  
  Margarita A. Guerrero,Cristian R. Rojas  
  📄 [arXiv link](https://arxiv.org/abs/2509.26322)  
  ✅ Accepted for publication in *8th Annual Learning for Dynamics & Control Conference, 2026 Edition*.
  
  **“Sample-Efficient Counterfactual Tuning for Compressor Pressure Control”**  
  Margarita A. Guerrero, Braghadeesh Lakshminarayanan, Cristian R. Rojas  
  📄 [arXiv link](https://arxiv.org/abs/2512.03747)  

---

## Overview

This code implements **ACE: Adaptive Sampling for Counterfactual Explanations**, a sample-efficient method to compute a **Counterfactual Explanation (CFE)** for **classification** tasks.

Given a trained black-box classifier $h: \mathcal{X} \to \{0,1\}$ and a fixed input instance $\tilde{x}$, ACE aims to find the closest point $x$ that **induces a classification flip**, i.e., $h(x) \neq h(\tilde{x})$. In other words, ACE returns the **smallest change needed in the input feature space** to change the classifier outcome.

ACE is designed for settings where **queries to the black-box model are limited or expensive**, and it adaptively samples informative points using a Gaussian-process surrogate and Bayesian optimization.

In addition to standard ML benchmarks, ACE is used as a **controller retuning engine** in a safety-critical industrial-inspired setting: the algorithm searches for the smallest actionable change in controller parameters that flips the outcome from “fail” to “pass” with a small number of closed-loop experiments.

---

## Contents

- **Main implementation**
  - `ACE_Github.py`  (main script; includes a runnable example at the end)

- **Datasets**
  - `datasets/`  (CSV datasets used in the seminal ACE paper and the tuning case study)

- **Support code for the compressor tuning case**
  - `LDG.py`         (bounds / utilities for PIDF tuning)
  - `LDG_cascade.py` (bounds / utilities for cascade PIDF tuning)

> Note: `ACE_Github.py` imports these helper modules from the same folder, and reads datasets from `datasets/<name>.csv`.

---

# Installation (ACE Repository)

This repository runs on a standard Python scientific stack. The notes below include a recommended folder layout and how to verify package versions from Spyder.


## Tested Environment

The repository was tested with:

- Python 3.10.19 (Anaconda)
- Windows 10
- numpy 2.2.6
- scipy 1.15.2
- pandas 2.3.3
- scikit-learn 1.7.2
- matplotlib 3.10.8
- pyDOE 0.3.8


## 1) Install dependencies

If you already have a Python environment, install the required packages with:

```bash
pip install numpy scipy pandas scikit-learn matplotlib pyDOE
```


## 2) Folder layout (important)

Keep the following files/folders at the **same repository level**:

- `ACE_Github.py`
- `LDG.py`
- `LDG_cascade.py`
- `datasets/`

The code expects datasets at:

- `datasets/<dataset>.csv`  
  (e.g., `datasets/pid_cascade.csv`)


## 3) Quick run

From the repository root:

```bash
python ACE_Github.py
```


## Check package versions (Spyder)

Run the following in Spyder’s IPython console:

```python
import sys, platform
import numpy, scipy, pandas, sklearn, matplotlib
import pyDOE

print("Python:", sys.version)
print("Platform:", platform.platform())
print("numpy:", numpy.__version__)
print("scipy:", scipy.__version__)
print("pandas:", pandas.__version__)
print("scikit-learn:", sklearn.__version__)
print("matplotlib:", matplotlib.__version__)
print("pyDOE:", pyDOE.__version__ if hasattr(pyDOE, "__version__") else "unknown")
```


## Export exact environment (optional)

To export all installed packages (useful for reproducibility):

```python
import sys, subprocess
subprocess.check_call([sys.executable, "-m", "pip", "freeze"])
```


## Optional: `requirements.txt` (pinned)

If you want a pinned `requirements.txt` matching the tested versions:

```txt
numpy==2.2.6
scipy==1.15.2
pandas==2.3.3
scikit-learn==1.7.2
matplotlib==3.10.8
pyDOE==0.3.8
```

---

## Main Function

Below is the key function used to run ACE end-to-end.

## `iterative_main_loop`

**Purpose**  
Runs **ACE end-to-end** for a *single* factual instance: it initializes the surrogate/model using \(\mathcal{H}\) initial samples, then iteratively proposes new candidate points (via an acquisition function), queries the black-box evaluator, and returns the closest valid counterfactual found.

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
- `output[0]` (dict): run results, typically including:
  - `instance`: factual instance \(\tilde{x}\)
  - `x_s`: computed counterfactual \(x\) (closest found)
  - `elapsed_time`: wall-clock time
  - `X`: sampled points across iterations
- `output[1]`: the predictor / black-box evaluator used for the selected dataset

---

## Helper Function

Below is the key function used to run ACE end-to-end.

## `run_100_fixed_iterations`

**Purpose**  
Runs ACE on a **single fixed factual instance** (given by `fixed_index`) for **100 different seeds**, saves run-level artifacts (e.g., NPZ + optional boxplots), and returns aggregated statistics.

**Signature**
```python
run_100_fixed_iterations(
    dataset,
    target,
    target_val,
    fixed_index,
    ini_ponts=30,
    action=None,
    save_dir="results",
    save_plots=True,
    show_plots=True,
    save_pdf=True,
    save_png=False
)
```

**Inputs (most important)**
- `dataset` (str): dataset name (expects `datasets/<dataset>.csv`)
- `target` (`Target`): classification target specification
- `target_val` (int/str): desired target class (e.g., `1`)
- `fixed_index` (int): index of the factual instance to keep fixed across the 100 runs
- `ini_ponts` (int): initial number of samples used to initialize ACE in each run
- `action` (list[int] or empty list): indices of **immutable/frozen** features  
  - if `action=[]`, then all features are actionable
- `save_dir` (str): folder where NPZ and figures are saved
- `save_plots`, `show_plots`, `save_pdf`, `save_png` (bool): figure controls

**Outputs (most important)**  
Returns a dict with summary statistics and saved-path info (keys may include):
- `l2_mean`, `l2_std` : mean/std L2 distance between CFE and factual instance
- `l1_mean`, `l1_std` : mean/std L1 distance between CFE and factual instance
- `validity_rate`, `validity_std` : fraction of runs that achieved a valid flip (plus variability)
- `lof_mean`, `lof_std` : mean/std LOF-based affinity of returned CFEs
- `iters_mean`, `iters_std` : mean/std of total iterations
- `real_iters_mean`, `real_iters_std` : mean/std of iterations after initialization
- `seeds` : array/list of seeds used
- `npz_path` : path to the saved NPZ file (if saved)
- `fig_prefix` : prefix used for saved plots (if saved)


## `optimize_acquisition_cat`

**Purpose**  
Optimizes the acquisition function (Expected Improvement) when some features are **categorical / integer-like**.  
A common approach is: continuous optimization (e.g., L‑BFGS‑B) for real-valued coordinates + local refinement for categorical coordinates (e.g., floor/ceil candidates), while respecting bounds and optionally frozen features.

**Signature**
```python
optimize_acquisition_cat(
    X,
    t,
    categorical_columns,
    X_test,
    kernel,
    bound_vals,
    x_s,
    MC,
    factor,
    lambd=10,
    n_neighbors=20,
    action=None,
    sampling_method="lhs",
    gtol=1e-20
)
```

**Inputs (most important)**
- `X` (np.ndarray): current sampled points
- `t` (np.ndarray): corresponding targets for `X`
- `categorical_columns` (list[int]): indices of categorical/integer features
- `X_test` (np.ndarray): candidate pool / initialization points for the optimizer
- `kernel`: GP kernel object
- `bound_vals` (np.ndarray): bounds array of shape `(d, 2)`
- `x_s` (np.ndarray): reference point (typically the factual instance) used in the acquisition
- `MC` (int): Monte Carlo sample count used inside EI
- `factor` (float): scaling factor used inside the optimizer workflow
- `lambd` (float): regularization / trade-off parameter used in EI
- `action` (list[int] or empty list): indices of **immutable/frozen** features
- `sampling_method` (str): e.g., `"lhs"` (Latin Hypercube)
- `gtol` (float): optimizer tolerance

**Outputs (most important)**  
Returns a tuple:
- `best_x` (np.ndarray): best candidate point found (next query point)
- `fx` (float): auxiliary EI output (as returned by the EI routine)
- `best_ei` (float): best (max) expected improvement achieved
- `x_min` (np.ndarray): baseline point used by EI (depending on implementation)


## `expected_improvement_mc_l1`

**Purpose**  
Computes **Expected Improvement (EI)** at candidate points using **Monte Carlo sampling** and an **L1-based sparsity penalty**, encouraging counterfactuals that change **fewer** features (or change them less in an L1 sense).

**Signature**
```python
expected_improvement_mc_l1(
    X2,
    X,
    t,
    kernel,
    x_s,
    lambda_,
    n_samples,
    alpha=5
)
```

**Inputs (most important)**
- `X2` (np.ndarray): candidate points where EI is evaluated
- `X` (np.ndarray): current sampled points (training inputs for the surrogate)
- `t` (np.ndarray): targets for `X`
- `kernel`: GP kernel object
- `x_s` (np.ndarray): reference point (factual instance)
- `lambda_` (float): regularization weight
- `n_samples` (int): number of Monte Carlo samples
- `alpha` (float): weight for the L1 / sparsity term

**Outputs (most important)**  
Returns a tuple:
- `mean_improvement` (float): EI value (mean improvement)
- `fx` (float): auxiliary value returned by the EI routine
- `x_min` (np.ndarray): baseline point used internally for the improvement term


## `compute_lof_affinity`

**Purpose**  
Computes a **Local Outlier Factor (LOF)** based affinity score for a candidate point, relative to a reference dataset.  
Intuition:
- affinity $\approx 1$ means “typical / inlier”
- affinity $> 1$ means “stronger inlier”
- affinity $ 1$ means “more outlier-like”

**Signature**
```python
compute_lof_affinity(
    new_point,
    X,
    n_neighbors=20
)
```

**Inputs (most important)**
- `new_point` (np.ndarray): point to evaluate (e.g., a candidate CFE)
- `X` (np.ndarray): reference data used to fit the LOF model
- `n_neighbors` (int): LOF neighborhood size

**Output**
- `affinity` (float): LOF-based affinity score

---

## Example Included: PID Cascade Tuning

At the end of `ACE_Github.py`, a minimal executable example computes the closest CFE for the cascade PID tuning dataset.

The example uses the factual controller parameter vector:

$\theta_0 = [[8.4533, 0.1358, 1.7270, 0.0104, 3.9855, 2.8415, 0.2635, 0.0200]]$

corresponding to:

$(Kp_1, Ki_1, Kd_1, Tf_1, \; Kp_2, Ki_2, Kd_2, Tf_2)$

This example obtains the CFE for:
- `dataset="pid_cascade"`
- `ind=47` (instance index; corresponds to $\theta_{01}$ in *Sample-Efficient Counterfactual Tuning for Compressor Pressure Control*)
- `target_val=1`
- `inip=30` (initial number of samples $\mathcal{H}$)
- `action=[]` (all 8 parameters are actionable)

Copy-paste snippet:

```python
inip = 30

# -------------- CASE 1: PIDF --------------
# dataset = "pidf"
# ind = 162 #This is the first instance for Case 1

# -------------- CASE 2: Cascade PIDFs --------------
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
```


### Citation

If you use this code in your work, please cite:

```bibtex
@inproceedings{Guerrero-25-ACE,
  title     = {{ACE}: Adapting Sampling for Counterfactual Explanations},
  author    = {Guerrero, Margarita A. and Rojas, Cristian R.},
  booktitle = {8th Annual Learning for Dynamics \& Control Conference (L4DC)},
  year      = {2026},
  note      = {Also available as arXiv:2509.26322}
}
```
