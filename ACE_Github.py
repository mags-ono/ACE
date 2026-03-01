# -*- coding: utf-8 -*-
"""
Created on Mon Jul  1 12:07:07 2024

@author: mags3
"""
import numpy as np
seed = 1
np.random.seed(seed)
import os
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
import warnings
warnings.filterwarnings("ignore")


import os, sys, importlib.util, pathlib

# Asegura que la carpeta del script está en sys.path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

try:
    from LDG import BOUNDS as PIDF_BOUNDS
except ModuleNotFoundError:
    # Fallback: importar por ruta absoluta al archivo LDG.py
    ldg_path = pathlib.Path(__file__).with_name("LDG.py")
    spec = importlib.util.spec_from_file_location("LDG", ldg_path)
    ldg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(ldg)
    PIDF_BOUNDS = ldg.BOUNDS
    
# try normal import; si no, fallback por ruta como con LDG
try:
    from LDG_cascade import BOUNDS_8 as PIDF_BOUNDS_8D
except ModuleNotFoundError:
    ldg8_path = pathlib.Path(__file__).with_name("LDG_cascade.py")
    spec8 = importlib.util.spec_from_file_location("LDG_cascade", ldg8_path)
    ldg8 = importlib.util.module_from_spec(spec8)
    spec8.loader.exec_module(ldg8)
    PIDF_BOUNDS_8D = ldg8.BOUNDS_8

from scipy.optimize import minimize
from sklearn.preprocessing import LabelEncoder
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern
import matplotlib.pyplot as plt
from scipy.special import expit as sigmoid
from sklearn.datasets import make_moons
from scipy.spatial.distance import cdist
from pyDOE import lhs
from sklearn.neighbors import LocalOutlierFactor
import time
from itertools import product
import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, LabelEncoder

from common.Target import Target
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import truncnorm
import warnings
from sklearn.exceptions import DataConversionWarning
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from skimage.transform import resize
from gaussian_processes_util import (
    plot_data_1D,
    plot_data_2D,
    plot_pt_2D,
    plot_db_2D)
#import tensorflow as tf
import pickle
import threadpoolctl
#from ucimlrepo import fetch_ucirepo
from functools import lru_cache
from simulation_2 import evaluate_pidf, evaluate_pidf_cascade, SIGMA_Y_DEFAULT, N_DEFAULT, TS_DEFAULT, R_STEP_DEFAULT, SIGMA_Y_DEFAULT_2, SETTLE_EPS_DEFAULT
#from LDG import BOUNDS as PIDF_BOUNDS


SIGMA_FOR_BO = SIGMA_Y_DEFAULT  # 0.01
N_FOR_BO     = N_DEFAULT        # 8000
TS_FOR_BO    = TS_DEFAULT       # el mismo Ts
R_FOR_BO     = R_STEP_DEFAULT   # 1.0




# Suprimir los DataConversionWarning de sklearn
#warnings.filterwarnings(action='ignore', category=DataConversionWarning)

#tf.random.set_seed(seed)
#threadpoolctl.threadpool_limits(limits=1, user_api='blas')

def compute_lof_affinity(new_point, X, n_neighbors=20):
    """
    Computes an affinity score where:
    - Affinity = 1 when LOF score = -1
    - Affinity > 1 when LOF score > -1 (more inlier)
    - Affinity < 1 when LOF score < -1 (more outlier)
    """
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
    lof.fit(X)

    score = lof.score_samples(new_point.reshape(1, -1))[0]

    # Apply shifted exponential to center affinity at score = -1
    affinity = np.exp(1 + score)
    return affinity


def plot_image(image, title=""):
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    plt.title(title)
    plt.show()
#seed=None


def train_kernel(X, t,opt, length_scale=1, tol=1e-15):
    """
    Trains a Gaussian Process classifier using a Matérn 5/2 kernel.

    Parameters:
    X : np.array
        Training input data of shape (n_samples, n_features).
    t : np.array
        Training target labels of shape (n_samples,).
    length_scale : float
        The length scale of the Matérn kernel.
    tol : float
        Tolerance for convergence of the optimizer.
        
    Returns:
    tuple:
        - GaussianProcessClassifier: The trained Gaussian Process model.
    """
    if len(np.unique(t)) < 2:
        print("Only one class present in training data. Skipping this instance.")
        return None, None  # Agregamos un flag indicando error
    #kernel = Matern(length_scale=length_scale, nu=2.5) #2.5
    #kernel = RBF(length_scale=length_scale)
    kernel = Matern(length_scale=length_scale, nu=2.5)
    if opt:
        model = GaussianProcessClassifier(kernel=kernel, optimizer='fmin_l_bfgs_b')
    else:
        model = GaussianProcessClassifier(kernel=kernel, optimizer=None)  # Disable optimizer
    
    model.fit(X, t)
    return model, model.kernel_

def W(a):
    """
    Compute the diagonal weight matrix for the Laplace approximation.
    
    Parameters:
    a : np.array
        Logit values, an array of shape (n_samples,).

    Returns:
    np.array
        Diagonal matrix of second derivatives of the logistic sigmoid function
        with respect to the logit values, used for the Hessian approximation.
    """    
    sig=sigmoid(a)*(1-sigmoid(a))
    return np.diag(sig.ravel())

def a_t(X, t,K_a, max_iter=10 , tol=1e-6):
    """
    Iteratively refine the estimate of the latent vector 'a' using the Newton-Raphson method.

    Parameters:
    X : np.array
        Input feature matrix of shape (n_samples, n_features).
    t : np.array
        Target binary labels for the training data of shape (n_samples,).
    K : np.array
        Covariance matrix of the training data, shape (n_samples, n_samples).
    max_iter : int
        Maximum number of iterations for Newton-Raphson method.
    tol : float
        Convergence tolerance for the iterative method.

    Returns:
    np.array
        Converged value of the latent variable vector 'a'.
    """
    '''
    a=np.zeros_like(t)
    I=np.eye(X.shape[0])
    
    for i in range(max_iter):
        W_a=W(a)
        F1=np.linalg.inv(I + W_a @ K)
        a_new=(K @ F1) @ (t-sigmoid(a)+ W_a @ a)
        diff=np.abs(a_new-a)
        a=a_new
        
        if np.linalg.norm(diff) < tol:
           break
       
    return a
    '''
    a = np.zeros_like(t)
    I = np.eye(X.shape[0])
    for i in range(max_iter):
        W_a = W(a)
        F1 = np.linalg.inv(I + W_a @ K_a)
        a_new = (K_a @ F1) @ (t - sigmoid(a) + W_a @ a)
        diff = np.abs(a_new - a)
        a = a_new
        
        if np.linalg.norm(diff) < tol:
            break
    
    return a

def posterior(X,t,X2,kernel):
    """
    Compute the posterior mean and variance for new data points using the Laplace approximation
    in Gaussian processes for classification.

    Parameters:
    X : np.array
        Training input data of shape (n_samples, n_features).
    t : np.array
        Training target labels of shape (n_samples,).
    X2 : np.array
        New input data for which to predict, of shape (n_samples_new, n_features).

    Returns:
    tuple
        A tuple (mu, var) where 'mu' is the posterior mean vector for the new data points,
        and 'var' is the covariance matrix of the predictions.
    """
    
    K=kernel(X,X)
    a=a_t(X,t,K)
    
    Ks=kernel(X,X2)
    Kss=kernel(X2,X2)
    
    W_inv=np.linalg.inv(W(a))
    F1=np.linalg.inv(W_inv+K)
    
    mu=(Ks.T) @ (t-sigmoid(a))
    
    diagonal=np.diag(Kss)
    
    var= diagonal.reshape(-1,1) - np.sum((F1 @ Ks) * Ks, axis=0).reshape(-1, 1)
    #return mu, var
    kappa = 1.0 / np.sqrt(1.0 + np.pi * var / 8)
    
    mu_real=sigmoid(kappa * mu)
    var_real = var * (mu_real * (1 - mu_real))**2 #La derivada de sigmoide es sigmoide*(1-sigmoide) y la incertidumbre se propaga como (df/dx)^2 * var_latente^2
    return mu_real,var_real, F1

def feature_normalized_distance(X, X_prime, std_devs,epsilon=1e-10):
    
    """
    Compute the feature-normalized Euclidean distance between points in X and X_prime.
    
    Parameters:
    X : np.array
        Input data points of shape (n_samples, n_features).
    X_prime : np.array
        Input data points of shape (n_samples, n_features).
    std_devs : np.array
        Standard deviations of each feature, shape (n_features,).
    
    Returns:
    np.array
        Normalized Euclidean distance between points in X and X_prime.
    """
    valid_indices = std_devs > epsilon  # Ignorar características con desviación estándar menor que epsilon
    if not np.any(valid_indices):
        raise ValueError("All features have zero standard deviation, cannot compute distance.")
    
    normalized_diff = (X[:, valid_indices] - X_prime[:, valid_indices]) / std_devs[valid_indices]
    squared_diff = np.square(normalized_diff)
    return np.sqrt(np.sum(squared_diff, axis=1)).reshape(-1, 1)

def feature_normalized_l1_distance(X, X_prime, std_devs, epsilon=1e-10):
    """
    Compute the feature-normalized L1 distance between points in X and X_prime.

    Parameters:
    X : np.array
        Input data points of shape (n_samples, n_features).
    X_prime : np.array
        Reference points of shape (n_samples, n_features).
    std_devs : np.array
        Standard deviations of each feature, shape (n_features,).
    epsilon : float
        Threshold to ignore features with near-zero std devs.

    Returns:
    np.array
        Normalized L1 distance between points in X and X_prime, shape (n_samples, 1).
    """
    valid_indices = std_devs > epsilon  # Ignore features with zero or near-zero std
    if not np.any(valid_indices):
        raise ValueError("All features have zero standard deviation, cannot compute distance.")

    normalized_diff = np.abs(X[:, valid_indices] - X_prime[:, valid_indices]) / std_devs[valid_indices]
    return np.sum(normalized_diff, axis=1).reshape(-1, 1)


def filter_outliers(new_point, X, n_neighbors=20):
    lof = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
    lof.fit(X)
    return lof.predict(new_point.reshape(1, -1)) == 1


def optimize_acquisition_cat(X, t, categorical_columns, X_test, kernel, bound_vals, x_s, MC, factor, lambd=10, n_neighbors=20,action=None, sampling_method='lhs', gtol=1e-20):
    """
    Uses a BFGS optimizer to find the point that maximizes the expected improvement,
    starting from the point in X2 that initially has the highest EI.

    Parameters:
    X : np.array
        Points over which to optimize EI.
    t : np.array
        Targets corresponding to X.
    categorical_columns : list
        Indices of columns that are categorical.
    X_test : np.array
        Test data for initial sampling.
    kernel : object
        Kernel used for GP modeling.
    bound_vals : np.array
        Bounds for each dimension during optimization.
    x_s : np.array
        The starting point for optimization.
    MC : int
        Number of Monte Carlo samples for expected improvement calculation.
    factor : float
        Factor for sampling standard deviation.
    lambd : float
        Scaling factor for expected improvement calculation.
    n_neighbors : int
        Number of neighbors for LOF.
    sampling_method : str
        Method for generating the initial point.
    gtol : float
        Gradient tolerance for the optimizer.
    action : list
        Indices of columns to be frozen.

    Returns:
    np.array: The point that maximizes the EI.
    """
    def objective(x):
        x = x.reshape(1, -1)
        #return -expected_improvement_mc(x, X, t, kernel, x_s, lambd, MC)[0]
        return -expected_improvement_mc_l1(x, X, t, kernel, x_s, lambd, MC)[0]

    best_result = None
    best_x = None
    unique_points = {tuple(y) for y in X}
    unique_points.add(tuple(x_s[0]))
    
    # Exclude categorical columns that are in `action`
    effective_categorical_columns = [col for col in categorical_columns if action is None or col not in action]

    # Sample an initial point
    for _ in range(10):
        if sampling_method == 'lhs':
            initial_point = latin_hypercube_sample(bound_vals, 1)[0]
        elif sampling_method == 'normal':
            std = np.sqrt(np.abs(x_s.ravel()))
            initial_point = truncated_normal(x_s.ravel(), std, bound_vals[:, 0], bound_vals[:, 1], 1, factor).ravel()
        elif sampling_method == 'random':
            initial_point = np.random.uniform(bound_vals[:, 0], bound_vals[:, 1])
        elif sampling_method == 'test':
            initial_point = X_test[np.random.choice(X_test.shape[0])]
            while tuple(initial_point) in unique_points:
                initial_point = X_test[np.random.choice(X_test.shape[0])]

        if tuple(initial_point) in unique_points:
            continue

        # Initial minimization
        res = minimize(
            objective,
            initial_point,
            method='L-BFGS-B',
            bounds=bound_vals,
            options={'gtol': gtol}
        )

        improvement = -res.fun
        if improvement <= 0:
            continue

        if best_result is None or res.fun < best_result:
            if filter_outliers(res.x, X, n_neighbors):
                best_result = res.fun
                best_x = res.x
                if not effective_categorical_columns:
                    break

                # Refinement for categorical columns
                if best_x is not None:
                    modified_point = best_x
                    refined_bounds = bound_vals.copy()

                    for cat_col in categorical_columns:
                        if action is not None and cat_col in action:
                            continue  # Skip frozen columns

                        candidate_solutions = []

                        # Evaluate floor and ceil for the current categorical column
                        for value in [np.floor(modified_point[cat_col]), np.ceil(modified_point[cat_col])]:
                            if refined_bounds[cat_col, 0] <= value <= refined_bounds[cat_col, 1]:
                                refined_point = modified_point.copy()
                                refined_point[cat_col] = value

                                # Freeze all previously refined columns
                                temp_bounds = refined_bounds.copy()
                                temp_bounds[cat_col, :] = [value, value]

                                refined_res = minimize(
                                    objective,
                                    refined_point,
                                    method='L-BFGS-B',
                                    bounds=temp_bounds,
                                    options={'gtol': gtol}
                                )

                                refined_improvement = -refined_res.fun
                                if refined_improvement > 0:
                                    candidate_solutions.append((refined_res.x, refined_improvement))

                        # Choose the best value for the current categorical column
                        if candidate_solutions:
                            best_candidate = max(candidate_solutions, key=lambda x: x[1])
                            modified_point = best_candidate[0]  # Update with the best candidate
                            refined_bounds[cat_col, :] = [modified_point[cat_col], modified_point[cat_col]]  # Freeze the column
                        else:
                            refined_point = modified_point.copy()
                            refined_point[cat_col] = np.random.choice([np.floor(modified_point[cat_col]), np.ceil(modified_point[cat_col])])
                            modified_point=refined_point
                            refined_bounds[cat_col, :] = [modified_point[cat_col], modified_point[cat_col]]

                    best_x = modified_point
                    best_result = -objective(modified_point)
                break

    if best_result is None:
        fx = 0
        return best_x, fx, best_result, 0
    else:
        _, fx, x_min = expected_improvement_mc(best_x.reshape(1, -1), X, t, kernel, x_s, lambd, MC)
        return best_x, fx, -best_result, x_min

def optimize_acquisition_bb2(X, t, categorical_columns, X_test, kernel, bound_vals, x_s, MC, factor, lambd=10, n_neighbors=20, action=None, sampling_method='lhs', gtol=1e-20):
    """
    Optimizes the acquisition function using a true Branch and Bound approach to find the point that maximizes the expected improvement,
    starting from the root point found by L-BFGS-B.

    Parameters:
    X : np.array
        Points over which to optimize EI.
    t : np.array
        Targets corresponding to X.
    categorical_columns : list
        Indices of columns that are categorical.
    X_test : np.array
        Test data for initial sampling.
    kernel : object
        Kernel used for GP modeling.
    bound_vals : np.array
        Bounds for each dimension during optimization.
    x_s : np.array
        The starting point for optimization.
    MC : int
        Number of Monte Carlo samples for expected improvement calculation.
    factor : float
        Factor for sampling standard deviation.
    lambd : float
        Scaling factor for expected improvement calculation.
    n_neighbors : int
        Number of neighbors for LOF.
    sampling_method : str
        Method for generating the initial point.
    gtol : float
        Gradient tolerance for the optimizer.
    action : list
        Indices of columns to be frozen.

    Returns:
    np.array: The point that maximizes the EI.
    """
    def objective(x):
        x = x.reshape(1, -1)
        return -expected_improvement_mc(x, X, t, kernel, x_s, lambd, MC)[0]
        #return -expected_improvement_mc_l1(x, X, t, kernel, x_s, lambd, MC)[0]
        

    # Exclude categorical columns that are in `action`
    effective_categorical_columns = [col for col in categorical_columns if action is None or col not in action]

    best_result = None
    best_x = None
    unique_points = {tuple(y) for y in X}
    unique_points.add(tuple(x_s[0]))

    # Sample an initial point
    for _ in range(10):
        if sampling_method == 'lhs':
            initial_point = latin_hypercube_sample(bound_vals, 1)[0]
        elif sampling_method == 'normal':
           # if not np.allclose(X[-1], x_s):
           #     initial_point = X[-1]
           # else:
            std = np.sqrt(np.abs(x_s.ravel()))
            initial_point = truncated_normal(x_s.ravel(), std, bound_vals[:, 0], bound_vals[:, 1], 1, factor).ravel()
        elif sampling_method == 'random':
            initial_point = np.random.uniform(bound_vals[:, 0], bound_vals[:, 1])
        elif sampling_method == 'test':
            initial_point = X_test[np.random.choice(X_test.shape[0])]
            while tuple(initial_point) in unique_points:
                initial_point = X_test[np.random.choice(X_test.shape[0])]

        if tuple(initial_point) in unique_points:
            continue

        # Initial minimization to find the root point
        res = minimize(
            objective,
            initial_point,
            method='L-BFGS-B',
            bounds=bound_vals,
            options={'gtol': gtol}
        )

        improvement = -res.fun
        if improvement <= 0:
            continue

        if best_result is None or res.fun < best_result:
            if filter_outliers(res.x, X, n_neighbors):
                best_result = res.fun
                best_x = res.x
                if not effective_categorical_columns:
                    break

                root_point = res.x.copy()
                # Reset best_result and best_x to ensure only feasible solutions are compared
                best_result = None
                best_x = None

                # Step 3: Branch and Bound
                def branch_and_bound(current_point, current_bounds, level):
                    nonlocal best_result, best_x

                    # If all categorical variables are fixed, optimize remaining continuous variables
                    if level == len(effective_categorical_columns):
                        refined_res = minimize(
                            objective,
                            current_point,
                            method='L-BFGS-B',
                            bounds=current_bounds,
                            options={'gtol': gtol}
                        )

                        refined_improvement = -refined_res.fun
                        # Update only at the lowest level
                        if best_result is None or refined_improvement > -best_result:
                            best_result = -refined_improvement
                            best_x = refined_res.x
                        return

                    # Branch over the current categorical variable
                    current_col = effective_categorical_columns[level]
                    int_floor = int(np.floor(current_point[current_col]))
                    int_ceil = int(np.ceil(current_point[current_col]))

                    for value in range(int_floor, int_ceil + 1):
                        modified_point = current_point.copy()
                        modified_point[current_col] = value
                        current_bounds[current_col, :] = [value, value]

                        # Optimize for other variables before branching further
                        refined_res = minimize(
                            objective,
                            modified_point,
                            method='L-BFGS-B',
                            bounds=current_bounds,
                            options={'gtol': gtol}
                        )
                        refined_point = refined_res.x

                        # Recurse to fix the next categorical variable
                        branch_and_bound(refined_point, current_bounds.copy(), level + 1)

                # Start branching from the root point
                branch_and_bound(root_point, bound_vals.copy(), 0)

    if best_x is None:
        return None, 0, None, 0
    else:
        _, fx, x_min = expected_improvement_mc(best_x.reshape(1, -1), X, t, kernel, x_s, lambd, MC)
        return best_x, fx, -best_result, x_min

def optimize_acquisition_bb(X, t, categorical_columns, X_test, kernel, bound_vals, x_s, MC, factor, lambd=10, n_neighbors=20, action=None, sampling_method='lhs', gtol=1e-20):
    """
    Optimizes the acquisition function using Branch and Bound to find the point that maximizes the expected improvement,
    starting from the point in X2 that initially has the highest EI.

    Parameters:
    X : np.array
        Points over which to optimize EI.
    t : np.array
        Targets corresponding to X.
    categorical_columns : list
        Indices of columns that are categorical.
    X_test : np.array
        Test data for initial sampling.
    kernel : object
        Kernel used for GP modeling.
    bound_vals : np.array
        Bounds for each dimension during optimization.
    x_s : np.array
        The starting point for optimization.
    MC : int
        Number of Monte Carlo samples for expected improvement calculation.
    factor : float
        Factor for sampling standard deviation.
    lambd : float
        Scaling factor for expected improvement calculation.
    n_neighbors : int
        Number of neighbors for LOF.
    sampling_method : str
        Method for generating the initial point.
    gtol : float
        Gradient tolerance for the optimizer.
    action : list
        Indices of columns to be frozen.

    Returns:
    np.array: The point that maximizes the EI.
    """
    def objective(x):
        x = x.reshape(1, -1)
        return -expected_improvement_mc(x, X, t, kernel, x_s, lambd, MC)[0]

    # Exclude categorical columns that are in `action`
    effective_categorical_columns = [col for col in categorical_columns if action is None or col not in action]

    best_result = None
    best_x = None
    unique_points = {tuple(y) for y in X}
    unique_points.add(tuple(x_s[0]))

    # Sample an initial point
    for _ in range(10):
        if sampling_method == 'lhs':
            initial_point = latin_hypercube_sample(bound_vals, 1)[0]
        elif sampling_method == 'normal':
            std = np.sqrt(np.abs(x_s.ravel()))
            initial_point = truncated_normal(x_s.ravel(), std, bound_vals[:, 0], bound_vals[:, 1], 1, factor).ravel()
        elif sampling_method == 'random':
            initial_point = np.random.uniform(bound_vals[:, 0], bound_vals[:, 1])
        elif sampling_method == 'test':
            initial_point = X_test[np.random.choice(X_test.shape[0])]
            while tuple(initial_point) in unique_points:
                initial_point = X_test[np.random.choice(X_test.shape[0])]

        if tuple(initial_point) in unique_points:
            continue

        # Initial minimization
        res = minimize(
            objective,
            initial_point,
            method='L-BFGS-B',
            bounds=bound_vals,
            options={'gtol': gtol}
        )

        improvement = -res.fun
        if improvement <= 0:
            continue

        if best_result is None or res.fun < best_result:
            if filter_outliers(res.x, X, n_neighbors):
                best_result = res.fun
                best_x = res.x
                if not effective_categorical_columns:
                    break

                # Branch and Bound
                if best_x is not None:
                    # Refine bounds based on floor and ceil of the root point
                    root_point = best_x.copy()
                    refined_bounds = bound_vals.copy()

                    all_combinations = list(product(
                        *[
                            range(int(np.floor(root_point[col])), int(np.ceil(root_point[col])) + 1)
                            for col in effective_categorical_columns
                        ]
                    ))

                    # Initialize best_bb_result and best_bb_point
                    best_bb_result = None
                    best_bb_point = None

                    for combination in all_combinations:
                        modified_point = root_point.copy()

                        # Freeze categorical variables according to the current combination
                        for idx, cat_col in enumerate(effective_categorical_columns):
                            modified_point[cat_col] = combination[idx]
                            refined_bounds[cat_col, :] = [combination[idx], combination[idx]]

                        # Refine the point using the optimizer
                        refined_res = minimize(
                            objective,
                            modified_point,
                            method='L-BFGS-B',
                            bounds=refined_bounds,
                            options={'gtol': gtol}
                        )

                        refined_improvement = -refined_res.fun

                        if best_bb_result is None or refined_improvement > best_bb_result:
                            best_bb_result = refined_improvement
                            best_bb_point = refined_res.x

                    # Update best_result and best_x if a better branch and bound result is found
                    if best_bb_result is not None and (best_result is None or best_bb_result > -best_result):
                        best_result = -best_bb_result
                        best_x = best_bb_point

                # Exit after evaluating all combinations for the current initial point
                break

    if best_x is None:
        return None, 0, None, 0
    else:
        _, fx, x_min = expected_improvement_mc(best_x.reshape(1, -1), X, t, kernel, x_s, lambd, MC)
        return best_x, fx, -best_result, x_min

def optimize_acquisition(X, t,categorical_column ,X_test, kernel, bound_vals, x_s, MC, factor, lambd=10, n_neighbors=20, sampling_method='lhs', gtol=1e-20):
    """
    Uses a BFGS optimizer to find the point that maximizes the expected improvement,
    starting from the point in X2 that initially has the highest EI.

    Parameters:
    X : np.array
        Points over which to optimize EI.
    model : GaussianProcessClassifier
        Trained Gaussian Process model.
    bound_vals : np.array
        Bounds for each dimension during optimization.
    lambd : float
        Scaling factor for expected improvement calculation.
    gtol : float
        Gradient tolerance for the optimizer.
    n_neighbors : int
        Number of neighbors for LOF.
    sampling_method : str
        Method for generating the initial point, 'lhs' for Latin Hypercube Sampling or 'uniform' for uniform random sampling.

    Returns:
    np.array: The point that maximizes the EI.
    """
    def objective(x):
        x = x.reshape(1, -1)
        return -expected_improvement_mc(x, X, t, kernel, x_s, lambd, MC)[0]
    
    
    best_result = None
    best_x = None
    unique_points = {tuple(y) for y in X}
    unique_points.add(tuple(x_s[0]))
    if sampling_method=='pca':
        
        #without SMOTE
        #pca = PCA(n_components=50)
        pca = PCA(n_components=15)
        X_reduced = pca.fit_transform(X_test)
        x_s_reduced = pca.transform(x_s)
        reduced_bound_vals = bounds_from_model(X_reduced)
        '''
        #with SMOTE
        smote = SMOTE()
        X_resampled, y_resampled = smote.fit_resample(X_test, y_test.ravel())

        # Apply PCA to the resampled data
        pca = PCA(n_components=50)
        X_reduced = pca.fit_transform(X_resampled)
        x_s_reduced = pca.transform(x_s)
        reduced_bound_vals = bounds_from_model(X_reduced)
        '''
    for _ in range(10): #20
        if sampling_method == 'lhs':
            initial_point = latin_hypercube_sample(bound_vals, 1)[0]
        elif sampling_method == 'normal': #rest
            std = np.sqrt(np.abs(x_s.ravel()))
            initial_point = truncated_normal(x_s.ravel(), std, bound_vals[:,0], bound_vals[:,1], 1, factor).ravel()
        elif sampling_method == 'pca': #mnist
            mean = np.mean(reduced_bound_vals, axis=1)
            std = np.sqrt(np.abs(mean))
            initial_point_reduced = truncated_normal(x_s_reduced.ravel(), std, reduced_bound_vals[:, 0], reduced_bound_vals[:, 1], 1, factor).ravel()
            initial_point = pca.inverse_transform(initial_point_reduced)
        elif sampling_method == 'random':
            initial_point = np.random.uniform(bound_vals[:, 0], bound_vals[:, 1])
        elif sampling_method == 'test':
            initial_point = X_test[np.random.choice(X_test.shape[0])]
            while tuple(initial_point) in unique_points:
                initial_point = X_test[np.random.choice(X_test.shape[0])]        
        
        if tuple(initial_point) in unique_points:
            continue
        
        res = minimize(
            objective, 
            initial_point, 
            method='L-BFGS-B', 
            bounds=bound_vals, 
            options={'gtol': gtol}
        )
        
        improvement = -res.fun
        if improvement <= 0:
            continue

        if best_result is None or res.fun < best_result:
            if filter_outliers(res.x, X, n_neighbors):
                best_result = res.fun
                best_x = res.x
                break
    if best_result is None:
        fx = 0
        return best_x, fx, best_result,0
    else:
        _,fx,x_min = expected_improvement_mc(best_x.reshape(1, -1), X, t, kernel, x_s, lambd, MC)
        return best_x, fx, -best_result,x_min
    
def sample_gp_correlated2(mu, cov, n_samples):
    """
    Generate samples from a correlated Gaussian Process given the mean and covariance.
    
    Parameters:
    mu : np.array
        Mean values (k, ).
    cov : np.array
        Covariance matrix (k, k).
    n_samples : int
        Number of samples to generate.
    
    Returns:
    np.array
        Samples generated from the GP (n_samples, k).
    """
    L = np.linalg.cholesky(cov)  # Adding a small jitter for numerical stability
    uncorrelated_samples = np.random.normal(size=(n_samples, len(mu)))
    correlated_samples = mu + uncorrelated_samples @ L.T
    return correlated_samples

def sample_gp_correlated(mu, cov, n_samples, jitter=1e-6):
    """
    Generate samples from a correlated Gaussian Process given the mean and covariance.

    Parameters:
    mu : np.array
        Mean values (k, ).
    cov : np.array
        Covariance matrix (k, k).
    n_samples : int
        Number of samples to generate.
    jitter : float
        Small value added to the diagonal for numerical stability.

    Returns:
    np.array
        Samples generated from the GP (n_samples, k).
    """
    try:
        L = np.linalg.cholesky(cov)  # Attempt Cholesky decomposition
    except np.linalg.LinAlgError:
        # Add jitter to the diagonal and retry
        cov += np.eye(cov.shape[0]) * jitter
        L = np.linalg.cholesky(cov)

    uncorrelated_samples = np.random.normal(size=(n_samples, len(mu)))
    correlated_samples = mu + uncorrelated_samples @ L.T
    return correlated_samples

def expected_improvement_mc_l1(X2, X, t, kernel, x_s, lambda_, n_samples,alpha=5):
    """
    Calculate Expected Improvement using Monte Carlo sampling with correlated Gaussian variables.

    Parameters:
    X2 : np.array
        Points where EI is evaluated.
    X : np.array
        Training points.
    t : np.array
        Training targets.
    kernel : sklearn.gaussian.process.kernels object
        Kernel used in the Gaussian Process.
    x_s : np.array
        Reference point for distance calculations.
    lambda_ : float
        Regularization parameter.
    n_samples : int
        Number of Monte Carlo samples.
    
    Returns:
    float
        Maximum improvement value.
    """
    mu_train, sigma_train,F1 = posterior(X, t, X, kernel)
    mu_star, sigma_star,_ = posterior(X, t, X2, kernel)
    std_devs_s = np.std(X, axis=0)
    
    d = feature_normalized_distance(X, x_s, std_devs_s)
    #reg_s=(0.5-mu_train)**2
    g=feature_normalized_l1_distance(X,x_s, std_devs_s) #Sparsity term
       
    min_idx=np.argmin(d+alpha*g+lambda_*np.abs(0.5-mu_train))
    #min_idx=np.argmin(d+lambda_*reg_s)
    x_min=X[min_idx].reshape(1,-1)
    
    mu_min=mu_train[min_idx]
    var_min=sigma_train[min_idx]
    
    K_star_min = kernel(X2, x_min)
    K_star_X = kernel(X2, X)
    K_min_X = kernel(X, x_min)
    
    cov_star_min = K_star_min - K_star_X @ F1 @ K_min_X
    covar_real = cov_star_min * (mu_star * (1 - mu_star))*(mu_min*(1-mu_min))
    
    mu_combined = np.hstack([mu_star.ravel(), mu_min.ravel()])
    
    cov_combined = np.block([
        [sigma_star, covar_real],
        [covar_real.T, var_min]
    ])

    samples = sample_gp_correlated(mu_combined, cov_combined, n_samples)
    # Separate the samples for f(x) and f(x*)
    f_star = samples[:, 0]  # First column for f(x)
    f_min = samples[:, 1]      # All but the last column for f(x*)

    d_s = feature_normalized_distance(X2, x_s, std_devs_s)
    d_min = feature_normalized_distance(x_min, x_s, std_devs_s)
    
    g_s=feature_normalized_l1_distance(X2,x_s, std_devs_s); #new point
    g_min=feature_normalized_l1_distance(x_min,x_s, std_devs_s); #new point
    
    c_s = d_s.reshape(-1,1) + alpha*g_s.reshape(-1,1) + lambda_ * np.abs(0.5- f_star.reshape(-1,1))
    #c_s = d_s.reshape(-1,1) - lambda_ * (f_star.reshape(-1,1) - 0.5)**2
    c_min = d_min.reshape(-1,1) + alpha*g_min.reshape(-1,1) + lambda_ * np.abs(0.5 - f_min.reshape(-1,1))
    #c_min = d_min.reshape(-1,1) + lambda_ * (f_min.reshape(-1,1) - 0.5)**2
    #fx=f_star[np.argmax(c_min-c_s)]
    # Calculate the improvement
    #improvement = np.maximum(0, np.max(c_min-c_s))
    improvements = np.maximum(0, c_min - c_s)
    mean_improvement = np.mean(improvements)
    fx = f_star[np.argmax(improvements)]
    
    return mean_improvement,fx,x_min

def expected_improvement_mc(X2, X, t, kernel, x_s, lambda_, n_samples=100):
    """
    Calculate Expected Improvement using Monte Carlo sampling with correlated Gaussian variables.

    Parameters:
    X2 : np.array
        Points where EI is evaluated.
    X : np.array
        Training points.
    t : np.array
        Training targets.
    kernel : sklearn.gaussian.process.kernels object
        Kernel used in the Gaussian Process.
    x_s : np.array
        Reference point for distance calculations.
    lambda_ : float
        Regularization parameter.
    n_samples : int
        Number of Monte Carlo samples.
    
    Returns:
    float
        Maximum improvement value.
    """
    mu_train, sigma_train,F1 = posterior(X, t, X, kernel)
    mu_star, sigma_star,_ = posterior(X, t, X2, kernel)
    std_devs_s = np.std(X, axis=0)
    
    d = feature_normalized_distance(X, x_s, std_devs_s)
    #reg_s=(0.5-mu_train)**2
    min_idx=np.argmin(d+lambda_*np.abs(0.5-mu_train))
    #min_idx=np.argmin(d+lambda_*reg_s)
    x_min=X[min_idx].reshape(1,-1)
    
    mu_min=mu_train[min_idx]
    var_min=sigma_train[min_idx]
    
    K_star_min = kernel(X2, x_min)
    K_star_X = kernel(X2, X)
    K_min_X = kernel(X, x_min)
    
    cov_star_min = K_star_min - K_star_X @ F1 @ K_min_X
    covar_real = cov_star_min * (mu_star * (1 - mu_star))*(mu_min*(1-mu_min))
    
    mu_combined = np.hstack([mu_star.ravel(), mu_min.ravel()])
    
    cov_combined = np.block([
        [sigma_star, covar_real],
        [covar_real.T, var_min]
    ])

    samples = sample_gp_correlated(mu_combined, cov_combined, n_samples)
    # Separate the samples for f(x) and f(x*)
    f_star = samples[:, 0]  # First column for f(x)
    f_min = samples[:, 1]      # All but the last column for f(x*)

    d_s = feature_normalized_distance(X2, x_s, std_devs_s)
    d_min = feature_normalized_distance(x_min, x_s, std_devs_s)
    
    c_s = d_s.reshape(-1,1) + lambda_ * np.abs(0.5- f_star.reshape(-1,1))
    #c_s = d_s.reshape(-1,1) - lambda_ * (f_star.reshape(-1,1) - 0.5)**2
    c_min = d_min.reshape(-1,1) + lambda_ * np.abs(0.5 - f_min.reshape(-1,1))
    #c_min = d_min.reshape(-1,1) + lambda_ * (f_min.reshape(-1,1) - 0.5)**2
    #fx=f_star[np.argmax(c_min-c_s)]
    # Calculate the improvement
    #improvement = np.maximum(0, np.max(c_min-c_s))
    improvements = np.maximum(0, c_min - c_s)
    mean_improvement = np.mean(improvements)
    fx = f_star[np.argmax(improvements)]
    
    return mean_improvement,fx,x_min

# def h(X,knn):
#     """
#     Use the trained KNeighborsClassifier to predict labels for the given data points.
    
#     Parameters:
#     X : np.array
#         Input feature matrix of shape (n_samples, n_features).

#     Returns:
#     np.array
#         Predicted labels for the input data.
#     """
#     return knn.predict(X.reshape(1, -1) if X.ndim == 1 else X).reshape(-1, 1)


def h(X, knn):
    X2 = X.reshape(1, -1) if X.ndim == 1 else X
    if callable(knn):
        # si 'knn' ES una función/llamable → la llamamos
        return knn(X2).reshape(-1, 1)
    else:
        # si 'knn' es un modelo sklearn → usamos .predict
        return knn.predict(X2).reshape(-1, 1)


def Ini_Model(knn,categorical_columns,x_s,t_s,bound_vals,X_test,y_test,mitad=1,n=5,action=[]):
    
    if X_test is not None:
        if mitad==0:
            assert X_test is not None and y_test is not None, "X_test and y_test cannot be None"
            
            indices = np.random.choice(X_test.shape[0], n, replace=False)
            
            X = X_test[indices]
            t = y_test[indices].reshape(-1, 1)
    
            X = np.vstack([X, x_s])
            t = np.vstack([t, t_s])
        else:
            assert X_test is not None and y_test is not None, "X_test and y_test cannot be None"
        
            positive_indices = np.where(y_test == 1)[0]
            negative_indices = np.where(y_test == 0)[0]
        
            # Ensure that there are enough samples of each class
            if len(positive_indices) < n // 2 or len(negative_indices) < n // 2:
                raise ValueError("Not enough positive or negative samples in X_test to choose from")
        
            positive_indices = np.random.choice(positive_indices, n // 2, replace=False)
            negative_indices = np.random.choice(negative_indices, n // 2, replace=False)
        
            selected_indices = np.concatenate([positive_indices, negative_indices])
        
            X = X_test[selected_indices]
            t = y_test[selected_indices].reshape(-1, 1)
        
            X = np.vstack([X, x_s])
            t = np.vstack([t, t_s])
    else:
        # Inicializar X
        X = np.zeros((n, bound_vals.shape[0]))

        for i in range(bound_vals.shape[0]):
            if action is not None and i in action:
                # Freeze column values based on x_s
                X[:, i] = x_s[0, i]
            elif categorical_columns is not None and i in categorical_columns:
                # Para columnas categóricas, muestrea valores enteros en el rango [min, max]
                min_val, max_val = int(bound_vals[i, 0]), int(bound_vals[i, 1])
                X[:, i] = np.random.choice(range(min_val, max_val + 1), size=n, replace=True)
            else:
                # Para columnas continuas, usa truncated_normal
                mean = np.array([np.mean(bound_vals[i])])
                std = np.array([np.sqrt(np.abs(mean))])
                lower = np.array([bound_vals[i, 0]])
                upper = np.array([bound_vals[i, 1]])
        
                X[:, i] = truncated_normal(mean, std, lower, upper, n).ravel()

        # Calcular las etiquetas para los puntos generados
        t = h(X, knn)

        # Añadir el punto inicial x_s
        X = np.vstack([X, x_s])
        t = np.vstack([t, t_s])
    if X.shape[0]>=5:
        #model, _ = train_kernel(X, t, 1)
        model, _ = train_kernel(X, t, 1)
    else:
        model, _ = train_kernel(X, t, 0 ,0.5)
    return model, X, t

#def bounds_from_model(X_train, categorical_columns, action, x_s, extension_factor=1.0, dataset=None):
def bounds_from_model(X_train, categorical_columns, action, x_s, extension_factor=1.0, 
                      dataset=None,pid_lb=None, pid_ub=None):
    """
    Estimates bounds for generating new points based on the trained model's data,
    while keeping specified columns fixed.

    Parameters:
    X_train : np.ndarray
        The input dataset (training data).
    categorical_columns : list
        List of indices of categorical columns.
    action : list
        List of column indices to keep fixed (unchanged).
    x_s : np.ndarray
        The specific instance for which bounds are being calculated.
    extension_factor : float, optional
        Factor to extend the bounds for continuous columns.
    dataset : str, optional
        The name of the dataset. If "tictactoe", bounds are forced to [0, 1].

    Returns:
    np.ndarray
        Array of tuples specifying the lower and upper bounds for each dimension.
    """
    bounds = []

    if dataset in ("pidf_cascade", "pid_cascade"):
        n_features = X_train.shape[1]
        B = np.asarray(PIDF_BOUNDS_8D, dtype=float)
        assert B.shape[0] == n_features == 8, "PIDF_CASCADE bounds size mismatch."
    
        LB, UB = B[:, 0], B[:, 1]
        x0 = np.asarray(x_s, float).reshape(1, -1)  # shape (1, 8)
    
        local_half_width = [1.5, 1.0, 1.0, None,  1.5, 1.0, 1.0, None]
    
        for i in range(n_features):
            if action is not None and i in action:
                v = float(x0[0, i])
                bounds.append((v, v))  # fijar esta dimensión
                continue
    
            if local_half_width[i] is None:
                lo, hi = float(LB[i]), float(UB[i])  # Tf usa rango global completo
            else:
                hw = float(local_half_width[i])
                lo = max(float(LB[i]), float(x0[0, i] - hw))
                hi = min(float(UB[i]), float(x0[0, i] + hw))
                if lo > hi:
                    lo, hi = float(LB[i]), float(UB[i])
    
            bounds.append((lo, hi))
    
        return np.array(bounds, dtype=float)

    if dataset in ("pid", "pidf"):
        n_features = X_train.shape[1]
        LB = np.array([0.5, 0.1, 0.1, 0.01], dtype=float) if pid_lb is None else np.asarray(pid_lb, float)
        UB = np.array([10.0, 4.0, 5.0, 0.10], dtype=float) if pid_ub is None else np.asarray(pid_ub, float)
        assert LB.shape[0] == n_features and UB.shape[0] == n_features, "PIDF bounds size mismatch."
    
        x0 = np.asarray(x_s, float).reshape(1, -1)  # ensure shape (1, n_features)
        local_half_width = [1.5, 1.0, 1.0, None]

        for i in range(n_features):
            if action is not None and i in action:
                v = float(x0[0, i])
                bounds.append((v, v))  # freeze this feature
                continue
    
            if local_half_width[i] is None:
                lo, hi = float(LB[i]), float(UB[i])  # Tf uses entire global range
            else:
                hw = float(local_half_width[i])
                lo = max(float(LB[i]), float(x0[0, i] - hw))
                hi = min(float(UB[i]), float(x0[0, i] + hw))

                if lo > hi:
                    lo, hi = float(LB[i]), float(UB[i])
    
            bounds.append((lo, hi))
    
        return np.array(bounds, dtype=float)
    
    for i in range(X_train.shape[1]):
        if dataset == "tictactoe":
            # Force bounds for Tic-Tac-Toe to [0, 1]
            if i in action:
                # Keep the column fixed at its value in x_s
                bounds.append((x_s[0, i], x_s[0, i]))
            else:
                # All other columns can vary between 0 and 1
                bounds.append((0, 2))
        else:
            # General case for other datasets
            if i in action:
                # Keep the column fixed at its value in x_s
                bounds.append((x_s[0, i], x_s[0, i]))
            elif i in categorical_columns:
                # For categorical columns, use the min and max unique values
                min_val = np.min(X_train[:, i])
                max_val = np.max(X_train[:, i])
                bounds.append((min_val, max_val))
            else:
                # For continuous columns, extend bounds based on percentiles
                lower_bound = np.quantile(X_train[:, i], 0.05)
                upper_bound = np.quantile(X_train[:, i], 0.95)
                bounds.append((lower_bound, upper_bound))

    return np.array(bounds)

    #lower_bounds = np.min(X_train, axis=0)
    #upper_bounds = np.max(X_train, axis=0)
    
    #lower_bounds= np.quantile(X_train, 0.05, axis=0)
    #upper_bounds = np.quantile(X_train, 0.95, axis=0)
    
    #return np.vstack((lower_bounds, upper_bounds)).T

def truncated_normal(mean, std_dev, lower_bound, upper_bound, size, sampling_factor=1.0,min_std=1e-3):
    """
    Genera muestras de una distribución normal truncada.

    Parameters:
    mean : np.array
        La media de la distribución normal.
    std_dev : np.array
        La desviación estándar de la distribución normal.
    lower_bound : np.array
        El límite inferior para la truncación.
    upper_bound : np.array
        El límite superior para la truncación.
    size : int o tuple
        El número de muestras a generar o la forma de las muestras.
    sampling_factor : float
        Factor para ajustar la desviación estándar.

    Returns:
    np.array:
        Array de muestras de la distribución normal truncada.
    """
    # Ajustar la desviación estándar con el factor de muestreo
    adjusted_std_dev = std_dev * sampling_factor
    
    samples = np.zeros((size, len(mean)))
    
    for i in range(size):
        sample = np.zeros(len(mean))
        for j in range(len(mean)):
            if lower_bound[j] == upper_bound[j] or adjusted_std_dev[j]==0 :
                sample[j] = mean[j]
            else:
                while True:
                        sample[j] = truncnorm(
                            (lower_bound[j] - mean[j]) / adjusted_std_dev[j], 
                            (upper_bound[j] - mean[j]) / adjusted_std_dev[j], 
                            loc=mean[j], 
                            scale=adjusted_std_dev[j]
                        ).rvs(1)[0]
                        if lower_bound[j] <= sample[j] <= upper_bound[j]:
                            break
        samples[i] = sample
        
    return samples


def detect_categorical_columns(X_train):
    """
    Detects categorical columns in a NumPy array.
    A column is considered categorical if at least one of the first 5 values is a string.

    Parameters:
    X_train : np.ndarray
        The input NumPy array.

    Returns:
    list
        List of indices for categorical columns.
    """
    categorical_columns = []
    for col in range(X_train.shape[1]):
        # Check if any of the first 5 values in the column is a string
        if any(isinstance(val, str) for val in X_train[:5, col]):
            categorical_columns.append(col)
    return categorical_columns


def latin_hypercube_sample2(bounds, n_points=100):
    """
    Generate a Latin Hypercube Sample within given bounds.

    Parameters:
    bounds : list of tuples
        List of tuples specifying the lower and upper bounds for each dimension.
    n_points : int
        Number of points to generate.

    Returns:
    np.array
        Array of shape (n_points, n_dimensions) containing the Latin Hypercube Sample.
    """
    n_dim = len(bounds)
    lhs_sample = lhs(n_dim, samples=n_points)
    
    # Scale the sample to the bounds
    scaled_sample = np.zeros_like(lhs_sample)
    for i, (lower, upper) in enumerate(bounds):
        scaled_sample[:, i] = lhs_sample[:, i] * (upper - lower) + lower
    
    #scaled_sample=sobol_sample(bounds, n_points)
    return scaled_sample


from scipy.stats import qmc


def latin_hypercube_sample(bounds, n_points=100, categorical_columns=None, action=None, seed=None):
    """
    Generate a Latin Hypercube Sample within given bounds, handling both continuous and categorical variables,
    and allowing certain columns to be fixed (frozen) as specified.

    Parameters:
    bounds : list of tuples
        List of tuples specifying the lower and upper bounds for each dimension.
    n_points : int
        Number of points to generate.
    categorical_columns : list of ints, optional
        Indices of columns that are categorical. If None, all columns are considered continuous.
    action : list of ints, optional
        Indices of columns to be fixed (frozen). If None, no columns are fixed.
    seed : int, optional
        Seed for reproducibility.

    Returns:
    np.array
        Array of shape (n_points, n_dimensions) containing the generated sample.
    """
    n_dim = len(bounds)
    sampler = qmc.LatinHypercube(d=n_dim, seed=seed)
    lhs_sample = sampler.random(n=n_points)
    
    # Initialize the scaled sample array
    scaled_sample = np.zeros_like(lhs_sample)
    
    for i, (lower, upper) in enumerate(bounds):
        if action and i in action:
            # Fix the column to the specified value (assuming lower == upper for fixed columns)
            scaled_sample[:, i] = lower
        elif categorical_columns and i in categorical_columns:
            # Handle categorical variables
            unique_values = np.arange(lower, upper + 1)
            n_unique = len(unique_values)
            # Scale the LHS sample to the number of categories and assign discrete values
            indices = np.round(lhs_sample[:, i] * (n_unique - 1)).astype(int)
            scaled_sample[:, i] = unique_values[indices]
        else:
            # Handle continuous variables
            scaled_sample[:, i] = lhs_sample[:, i] * (upper - lower) + lower
    
    return scaled_sample

def sobol_sample(bounds, n_points=100, categorical_columns=None, action=None, seed=None):
    """
    Generate a sample using the Sobol' sequence, handling both continuous and categorical variables,
    and allowing certain columns to be fixed (frozen) as specified.

    Parameters:
    bounds : list of tuples
        List of tuples specifying the lower and upper bounds for each dimension.
    n_points : int
        Number of points to generate. Should be a power of 2 for Sobol' sequence.
    categorical_columns : list of ints, optional
        Indices of columns that are categorical. If None, all columns are considered continuous.
    action : list of ints, optional
        Indices of columns to be fixed (frozen). If None, no columns are fixed.
    seed : int, optional
        Seed for reproducibility.

    Returns:
    np.array
        Array of shape (n_points, n_dimensions) containing the generated sample.
    """
    n_dim = len(bounds)
    sampler = qmc.Sobol(d=n_dim, scramble=True, seed=seed)
    # Calculate the exponent for the nearest power of 2 greater than or equal to n_points
    m = int(np.ceil(np.log2(n_points)))
    # Generate 2^m samples
    sobol_sample = sampler.random_base2(m=m)
    # Select the first n_points samples
    sobol_sample = sobol_sample[:n_points]
    
    # Initialize the array for the scaled sample
    scaled_sample = np.zeros_like(sobol_sample)
    
    for i, (lower, upper) in enumerate(bounds):
        if action and i in action:
            # Fix the column to the specified value (assuming lower == upper for fixed columns)
            scaled_sample[:, i] = lower
        elif categorical_columns and i in categorical_columns:
            # Handle categorical variables
            unique_values = np.arange(lower, upper + 1)
            n_unique = len(unique_values)
            # Scale the Sobol' sample to the number of categories and assign discrete values
            indices = np.round(sobol_sample[:, i] * (n_unique - 1)).astype(int)
            scaled_sample[:, i] = unique_values[indices]
        else:
            # Handle continuous variables
            scaled_sample[:, i] = sobol_sample[:, i] * (upper - lower) + lower
    
    return scaled_sample



def E_N(X,knn, t,x_s, kernel, grid_points, minprob , mnist=1):
    """
    Evaluate the posterior mean on a grid of points and find the point with the highest posterior mean.

    Parameters:
    model : GaussianProcessClassifier
        The trained Gaussian Process model.
    bounds : np.array
        The bounds for each dimension.
    n_points : int
        Number of points to evaluate in each dimension.

    Returns:
    np.array
        The point with the highest posterior mean.
    """
    #gpc = GaussianProcessClassifier(kernel=kernel)
    #gpc.fit(X, t.ravel())
    #pt_test_gpc = gpc.predict_proba(grid_points)[:,1]
    flag=0
    
    if mnist==1:
        grid_points=X
    #else:
    #    grid_points=np.concatenate((grid_points, X), axis=0)
    probs,var,_ = posterior(X, t, grid_points, kernel)
    
    #if np.all(probs == 0.5):
        #print("All probs are 0.5")
        #return 0, 0,0, 0,np.inf,0
    
    max_index = np.argmax(probs)
    X_max=grid_points[max_index]
    # Find points with probability closest to 0.5
    closest_to_0_5_indices = np.where((probs > minprob) & (probs < 1))[0]

    if closest_to_0_5_indices.size == 0:
        closest_to_0_5_indices = np.where((probs > 0.40) & (probs <= minprob))[0]
        if closest_to_0_5_indices.size == 0:
           #print('error')
           return np.zeros(1), np.zeros(1),np.zeros(1), np.zeros(1),np.inf,np.zeros(1),flag,None,None
        else:
            #print('no probs over 0.5')
            closest_to_0_5_points = grid_points[closest_to_0_5_indices]
            distances = cdist(x_s, closest_to_0_5_points)

    else:
        closest_to_0_5_points = grid_points[closest_to_0_5_indices]
        distances = cdist(x_s, closest_to_0_5_points)
 

    # Get the indices of the 5 shortest distances
    sorted_indices = np.argsort(distances[0])[:5]  # Get indices of 5 smallest distances
    closest_points = closest_to_0_5_points[sorted_indices]  # Select the closest 5 points
    closest_probs = probs[closest_to_0_5_indices][sorted_indices]  # Corresponding probabilities
    closest_vars = var[closest_to_0_5_indices][sorted_indices]  # Corresponding variances
    
    # Evaluate h for each of the closest points
    h_values = h(closest_points, knn).ravel()  # Get the predicted labels (1 or 0)
    
    # Check if there are any points with h == 1
    if np.any(h_values == 1):
        # Get the index of the first point with h == 1
        closest_with_h_1_index = np.where(h_values == 1)[0][0]
        selected_point = closest_points[closest_with_h_1_index]
        selected_distance = distances[0][sorted_indices][closest_with_h_1_index]
        flag=1
    else:
        # If no points have h == 1, return 0
        closest_distance_index = np.argmin(distances[0])
        closest_to_0_5_point = closest_to_0_5_points[closest_distance_index]
        closest_to_0_5_prob = probs[closest_to_0_5_indices][closest_distance_index]
        closest_to_0_5_var = var[closest_to_0_5_indices][closest_distance_index]        
        return X_max, probs[max_index], closest_to_0_5_point, closest_to_0_5_prob, np.min(distances[0]),closest_to_0_5_var,flag,closest_points,h_values

    return (
        X_max,
        probs[max_index],
        selected_point,
        closest_probs if selected_point is not None else None,
        selected_distance,
        closest_vars if selected_point is not None else None,
        flag,
        closest_points,
        h_values
    )
         
    
    #closest_distance_index = np.argmin(distances[0])
    #closest_to_0_5_point = closest_to_0_5_points[closest_distance_index]
    #closest_to_0_5_prob = probs[closest_to_0_5_indices][closest_distance_index]
    #closest_to_0_5_var = var[closest_to_0_5_indices][closest_distance_index]        
    #return X_max, probs[max_index], closest_to_0_5_point, closest_to_0_5_prob, np.min(distances[0]),closest_to_0_5_var


def plot_and_save_image(image, title, filename):
    plt.imshow(image, cmap='gray')
    plt.title(title)
    plt.axis('off')
    plt.savefig(filename, format='pdf', bbox_inches='tight')
    plt.close()

def plot_distances_with_lines(X, y, x_s, ce,lambd):
    distances = cdist(X[y == 1], x_s).flatten()
    ce_distance = cdist(ce, x_s).flatten()[0]
    
    # Sort distances in descending order
    sorted_indices = np.argsort(distances)[::-1]
    sorted_distances = distances[sorted_indices]

    # Find the index of the counterfactual example distance in the sorted distances
    ce_index = np.where(sorted_distances == ce_distance)[0][0]

    plt.figure(figsize=(10, 6))
    plt.plot(range(len(sorted_distances)), sorted_distances, 'o-', label='Distances', markersize=5)
    plt.plot(ce_index, ce_distance, 'ro', markersize=10, label='Counterfactual Example Distance')
    plt.axhline(y=ce_distance, color='red', linewidth=2, linestyle='--')
    plt.xlabel('Index')
    plt.ylabel('Distance')
    plt.title(f'Distances to Counterfactual Example (Lambda={lambd})')
    plt.legend()
    plt.show()
    #save_plot_as_pdf(plt.gcf(), f'2_distances_with_lines_lambda_{lambd}.pdf')
    

def encode_categorical_data(dataframe, ordinal_mapping=None, label_columns=None, one_hot_columns=None):
    """
    Encodes categorical data in a DataFrame.
    Applies Ordinal Encoding for ordered features and Label Encoding for unordered features.

    Parameters:
    dataframe : pd.DataFrame
        The input DataFrame containing categorical data.
    ordinal_mapping : dict
        A dictionary where keys are column names and values are lists
        specifying the desired order of categories for Ordinal Encoding.
    label_columns : list
        List of column names to encode with Label Encoding.

    Returns:
    pd.DataFrame
        A DataFrame with encoded categorical data.
    """
    # Apply Ordinal Encoding if ordinal_mapping is provided
    if ordinal_mapping:
        for col in ordinal_mapping.keys():
            dataframe[col].fillna("NA", inplace=True)
        
        ordinal_encoder = OrdinalEncoder(categories=list(ordinal_mapping.values()))
        ordinal_columns = list(ordinal_mapping.keys())
        dataframe[ordinal_columns] = ordinal_encoder.fit_transform(dataframe[ordinal_columns])
    
    # Apply Label Encoding
    if label_columns:
        for col in label_columns:
            label_encoder = LabelEncoder()
            dataframe[col] = label_encoder.fit_transform(dataframe[col].astype(str))  # Ensure consistent types
    
        # Apply One-Hot Encoding : este es el que incrementa las columnas
    if one_hot_columns:
        one_hot_encoded = pd.get_dummies(dataframe[one_hot_columns], prefix=one_hot_columns, drop_first=False)
        
        # Convert boolean values to integers (0 and 1)
        one_hot_encoded = one_hot_encoded.astype(int)
        
        dataframe = dataframe.drop(columns=one_hot_columns).join(one_hot_encoded)
    
    return dataframe


def preprocess_images(X_train, X_test, method='pca', n_components=50, new_shape=(10, 10)):
    """
    Preprocess images using PCA or resampling.

    Parameters:
    X_train : np.array
        Training input data of shape (n_samples, 28*28).
    X_test : np.array
        Test input data of shape (n_samples, 28*28).
    method : str
        Method to use for dimensionality reduction ('pca' or 'resample').
    n_components : int
        Number of components for PCA.
    new_shape : tuple
        New shape for resampling.

    Returns:
    tuple
        Processed X_train and X_test, and the PCA model if method is 'pca'.
    """
    pca_model = None

    if method == 'pca':
        pca = PCA(n_components=n_components)
        X_train_processed = pca.fit_transform(X_train)
        X_test_processed = pca.transform(X_test)
        pca_model = pca  # Guardar el modelo PCA
    elif method == 'resample':
        X_train_reshaped = X_train.reshape(-1, 28, 28)
        X_test_reshaped = X_test.reshape(-1, 28, 28)
        X_train_processed = np.array([resize(image, new_shape, anti_aliasing=True).flatten() for image in X_train_reshaped])
        X_test_processed = np.array([resize(image, new_shape, anti_aliasing=True).flatten() for image in X_test_reshaped])
        
    else:
        raise ValueError("Invalid method. Choose 'pca' or 'resample'.")

    return X_train_processed, X_test_processed, pca_model


def J(X,x_min,t,lambd,x_s,kernel,fx=None):
    std_devs_s = np.std(X, axis=0)
    if fx!=None:
        d = feature_normalized_distance(X[-1].reshape(1, -1), x_s, std_devs_s)
        return d + lambd*np.abs(0.5-fx)
    else:
        mu_train, _,_ = posterior(X, t, x_min, kernel)
        d = feature_normalized_distance(x_min, x_s, std_devs_s)
        #return np.min( d + lambd*np.abs(0.5-mu_train) )
        return d
    
def preprocess_data(X):
    X = X / 255.0
    return X



@lru_cache(maxsize=4096)
def _eval_pidf_cached(kp, ki, kd, tf, sigma, N, Ts, r_value):
    lab, _, _ = evaluate_pidf(kp, ki, kd, tf, sigma_y=sigma, N=N, Ts=Ts, r_value=r_value, seed=0)
    return int(lab)

def pid_predictor(X, sigma=SIGMA_Y_DEFAULT, N=N_DEFAULT, Ts=TS_DEFAULT, r_value=R_STEP_DEFAULT):
    X = np.atleast_2d(X).astype(float)
    y = [_eval_pidf_cached(*row[:4], sigma, N, Ts, r_value) for row in X]
    return np.array(y, dtype=int).reshape(-1, 1)

# ------- 8D (cascade) - NUEVO -------
@lru_cache(maxsize=4096)
def _eval_pidf_cascade_cached(kp1, ki1, kd1, tf1, kp2, ki2, kd2, tf2, sigma1, sigma2, N, Ts, r_value):
    lab, _, _ = evaluate_pidf_cascade(
        kp1, ki1, kd1, tf1, kp2, ki2, kd2, tf2,
        sigma_y1=sigma1, sigma_y2=sigma2,
        N=N, Ts=Ts, r_value=r_value, seed=0
    )
    return int(lab)

def pid_predictor_8d(
    X,
    sigma1=SIGMA_Y_DEFAULT,      # outer noise
    sigma2=SIGMA_Y_DEFAULT_2,    # inner noise (tu nuevo default = 0.005)
    N=N_DEFAULT, Ts=TS_DEFAULT, r_value=R_STEP_DEFAULT
):
    X = np.atleast_2d(X).astype(float)
    y = [
        _eval_pidf_cascade_cached(*row[:8], sigma1, sigma2, N, Ts, r_value)
        for row in X
    ]
    return np.array(y, dtype=int).reshape(-1, 1)

# ------- Wrapper selector (conserva ambos modos) -------
# def make_pid_predictor(mode: str = "single"):
#     """
#     mode: 'single' -> 4D predictor, 'cascade' -> 8D predictor
#     """
#     mode = (mode or "single").lower()
#     if mode == "cascade":
#         return pid_predictor_8d
#     return pid_predictor

def main_loop(x_s,x_new,t_new,categorical_columns, sample,knn,ini_ponts,lambd,action,cont,MC, bound_vals,factor,mask,X_test,y_test,grid_x, grid_y, grid, initial_X=None, initial_t=None, initial_kernel=None):

    num_it = 30#30
    cost = []
    flag=0
    
    t_s=h(x_s, knn)
    #print(t_s)
    lambd=10
    #lambda=10
    
    minprob=0.5
    improvement=[]
    max_lambda = 0
    epsilon=1e-3
    #impro_min=0
    #impro_max=1e15
    distances=[]
    exi=0
    
    for i in range(num_it):
        if lambd>1e30:#30
            break
        if i == 0:
            if initial_X is None or initial_t is None:
                model, X, t = Ini_Model(knn,categorical_columns, x_s,t_s,bound_vals,X_test,y_test,0,ini_ponts,action)
                if model==None:
                    return x_s.reshape(-1, X.shape[1]), 0, X, t, None, 0,0,x_s.reshape(-1, X.shape[1]), 0,0
            else:
                X, t = initial_X, initial_t
                #X = np.vstack([X, x_new])
                #t = np.vstack([t, t_new])
                model, _ = train_kernel(X, t, 0)
                if model==None:
                    return x_s.reshape(-1, X.shape[1]), 0, X, t, None, 0,0,x_s.reshape(-1, X.shape[1]), 0,0

            kernel = model.kernel_
            initial_params = kernel.get_params()

        next_point,_,_,x_min= optimize_acquisition_cat(X,t,categorical_columns,X_test, kernel, bound_vals, x_s,MC,factor, lambd, ini_ponts,action, sampling_method='normal')
        
        if next_point is None:
            lambd=lambd**1.5
            exi=exi+1
            if exi>2:
                #print('early exit')
                break
            else:
                continue
        
        new_target = h(next_point, knn)
        
        X = np.vstack([X, next_point])
        t = np.vstack([t, new_target])

        
        if X.shape[0]==np.round(ini_ponts*1.2):
            model, _ = train_kernel(X, t, 1)
            kernel=model.kernel_
            flag=1
        
        if flag==1 and X.shape[0]==np.round(ini_ponts*1.5):
            model, _ = train_kernel(X, t, 1)
            kernel=model.kernel_

        #print(new_dist)
        if np.linalg.norm(X[-1]-X[-2])<epsilon:
            #print(f"Terminating early at iteration {i+1} with cost difference {abs(cost[-1] - cost[-2])}")
            break
        lambd=lambd**1.5



    max_point,_,best_point,_,new_dist,_,flag2,points,t_points = E_N(X,knn, t, x_s, kernel, sample, minprob,0)
    
    if points is not None:
        X = np.vstack([X, points])
        t = np.vstack([t, t_points.reshape(-1,1)])

    if best_point.size<=1:
        if best_point==0:
            best_point=x_s
            max_point=x_s
    return best_point.reshape(-1, X.shape[1]), flag2, X, t, model.kernel_, i + 1,0, max_point.reshape(-1, X.shape[1]), lambd,improvement

def iterative_main_loop(dataset, target, initial_instance_index,target_val, ini_ponts,seed=0,action=None):
    
    points = 8000

    lambd=1
    factor=1
    counter=0
    
    #svc
    if dataset=='moons':
        T, ht = make_moons(n_samples=200, shuffle=True, noise=0.05, random_state=0)
        T = (T.copy() - T.mean(axis=0)) / T.std(axis=0)
        X_train, X_test, y_train, y_test = train_test_split(T, ht, test_size=0.3, random_state=0)
        
        
        knn = SVC(gamma=1, probability=True)
        knn.fit(X_train, y_train)
        
        X_combined = np.vstack([X_train, X_test])
        y_combined = np.hstack([y_train, y_test])
        
        bound_vals = bounds_from_model(X_train, extension_factor=1.0)
        initial_x_s=np.array([[-0.18558001,  1.47538963]])
        #initial_x_s=np.array([[-0.18558001,  1.1]])
        #initial_x_s=np.array([[0,  0]])
        #initial_x_s=np.array([[0.55,  0.25]])
        x_s = initial_x_s
        factor=7
        X_combined=X_combined
        y_combined=y_combined
          
        
    elif dataset=='german':
        dataframe = pd.read_csv("datasets/" + "german" + ".csv")
        #categorical_columns = detect_categorical_columns(dataframe.drop(['Unnamed: 0', target.target_feature()], axis=1).values)
        categorical_columns = [1, 2, 3, 4, 5, 8]
        ordinal_mapping = {
            "Housing": ["free", "rent", "own"],
            "Saving accounts": ["NA", "little", "moderate", "quite rich", "rich"],
            "Checking account": ["NA", "little", "moderate", "quite rich", "rich"]}
        label_columns = ["Sex","Purpose"]
        dataframe= encode_categorical_data(dataframe, ordinal_mapping, label_columns)
        
        X_train = dataframe.drop(['Unnamed: 0', target.target_feature()], axis=1).values
        
        #label_encoders = {}  # Guardar los codificadores para decodificar si es necesario
        ht = dataframe[[target.target_feature()]].values.ravel()
        #for col in categorical_columns:
        #    le = LabelEncoder()
        #    X_train[:, col] = le.fit_transform(X_train[:, col].astype(str))
        #    label_encoders[col] = le
            
        if target_val=='good':
            y_train = np.where(ht == 'good', 1, 0)
        else:
            y_train = np.where(ht == 'good', 0, 1)
        knn = RandomForestClassifier()
        knn.fit(X_train, y_train)
        initial_x_s=X_train[initial_instance_index].reshape(1, -1)
        x_s = initial_x_s
        print(h(x_s,knn))
        bound_vals = bounds_from_model(X_train, categorical_columns,action,x_s,extension_factor=1.0)
        factor=7
        #X_combined=X_train
        #y_combined=y_train    
        X_combined=None
        y_combined=None
    
    elif dataset=='cmc':
        
        #758 -> 1
        #820 -> 1.41421356
        #887 -> 2.82
        #900 -> 1
        #699 -> 1
        dataframe = pd.read_csv("datasets/" + "cmc" + ".csv")
        
        X_train = dataframe.drop([target.target_feature()], axis=1).values
        categorical_columns=[1,2,4,5,6,7,8]
        #label_encoders = {}  # Guardar los codificadores para decodificar si es necesario
        ht = dataframe[[target.target_feature()]].values.ravel()
        y_train=ht
        knn = RandomForestClassifier()
        knn.fit(X_train, y_train)
        initial_x_s=X_train[initial_instance_index].reshape(1, -1)
        x_s = initial_x_s
        #print(h(x_s,knn))
        
        # #Calcular desviaciones estándar por feature
        # std_devs = np.std(X_train, axis=0)
        
        # # Guardar en la raíz
        # std_file_path = f"std_{dataset}.npy"
        # np.save(std_file_path, std_devs)
        
        # index_file = f"selected_indices_{dataset}.npy"

        # if not os.path.exists(index_file):
        #     # Índices fijos que quieres incluir
        #     fixed_indices = [900, 699, 758]
        
        #     # Buscar todos los índices que cumplen y_train == 0 y no son los fijos
        #     all_class_0_indices = np.where(y_train == 0)[0]
        #     remaining_indices = [idx for idx in all_class_0_indices if idx not in fixed_indices]
        
        #     # Elegir 97 aleatorios (sin reemplazo) de los restantes
        #     np.random.seed(42)  # Para reproducibilidad
        #     random_indices = np.random.choice(remaining_indices, size=97, replace=False)
        
        #     # Combinar con los índices fijos
        #     selected_indices = np.array(fixed_indices + random_indices.tolist())
        #     np.save(index_file, selected_indices)
        #     print(f"Saved Indexes {index_file}. Total: {len(selected_indices)}")
        # else:
        #     selected_indices = np.load(index_file)
        #     print(f"Load Indexes {index_file}. Total: {len(selected_indices)}")
        
        bound_vals = bounds_from_model(X_train, categorical_columns,action,x_s,extension_factor=1.0)
        factor=7
        #X_combined=X_train
        #y_combined=y_train    
        X_combined=None
        y_combined=None  
        
    elif dataset=='pidf':
        dataframe = pd.read_csv("datasets/" + dataset + ".csv")
        X_train = dataframe.drop([target.target_feature()], axis=1).values
        ht = dataframe[[target.target_feature()]].values.ravel()
        categorical_columns = []
        if target_val==1:
            y_train = np.where(ht == 1, 1, 0)
        else:
            y_train = np.where(ht == 0, 0, 1)
            
        
        knn = lambda X: pid_predictor(X, sigma=SIGMA_FOR_BO, N=N_FOR_BO, Ts=TS_FOR_BO, r_value=R_FOR_BO)

        
        initial_x_s = X_train[initial_instance_index].reshape(1, -1)

        x_s = initial_x_s
        
        lb = PIDF_BOUNDS[:, 0]
        ub = PIDF_BOUNDS[:, 1]
        bound_vals = bounds_from_model(
            X_train, categorical_columns, action, x_s,
            dataset="pidf",
            pid_lb=lb, pid_ub=ub
        )

        
        X_combined = X_train
        y_combined = y_train
        
        factor = 7  # o el que venías usando
        
    elif dataset=='pid_cascade':
        dataframe = pd.read_csv("datasets/" + dataset + ".csv")
        X_train = dataframe.drop([target.target_feature()], axis=1).values
        ht = dataframe[[target.target_feature()]].values.ravel()
        categorical_columns = []
        if target_val==1:
            y_train = np.where(ht == 1, 1, 0)
        else:
            y_train = np.where(ht == 0, 0, 1)
            
        
        knn = lambda X: pid_predictor_8d(X, sigma1=SIGMA_FOR_BO,sigma2=SIGMA_Y_DEFAULT_2, N=N_FOR_BO, Ts=TS_FOR_BO, r_value=R_FOR_BO)
        

        initial_x_s = X_train[initial_instance_index].reshape(1, -1) 
        x_s = initial_x_s
        
        lb = PIDF_BOUNDS_8D[:, 0]
        ub = PIDF_BOUNDS_8D[:, 1]
        bound_vals = bounds_from_model(
            X_train, categorical_columns, action, x_s,
            dataset="pid_cascade",
            pid_lb=lb, pid_ub=ub
        )
        

        
        X_combined = X_train
        y_combined = y_train
        
        factor = 7  # o el que venías usando        

        
        
    elif dataset=='blood':
        dataframe = pd.read_csv("datasets/" + dataset + ".csv")
        categorical_columns = []
        action=[]
        X_train = dataframe.drop([target.target_feature()], axis=1).values
        ht = dataframe[[target.target_feature()]].values.ravel()
        
        if target_val==2:
            y_train = np.where(ht == 2, 1, 0)
        else:
            y_train = np.where(ht == 2, 0, 1)
        knn = RandomForestClassifier()
        knn.fit(X_train, y_train)
        initial_x_s=X_train[initial_instance_index].reshape(1, -1)
        x_s = initial_x_s
        
        bound_vals = bounds_from_model(X_train, categorical_columns,action,x_s,extension_factor=1.0)
        factor=7
        X_combined=X_train
        y_combined=y_train
    elif dataset=="eyes_new":
        dataframe = pd.read_csv("datasets/" + dataset + ".csv")
        X_train = dataframe.drop([target.target_feature()], axis=1).values
        ht = dataframe[[target.target_feature()]].values.ravel()
        if target_val=="2":
            y_train = np.where(ht == 2, 1, 0)
        else:
            y_train = np.where(ht == 2, 0, 1)
        factor=7#7
        knn = RandomForestClassifier()
        knn.fit(X_train, y_train)
        initial_x_s=X_train[initial_instance_index].reshape(1, -1)
        x_s = initial_x_s
        bound_vals = bounds_from_model(X_train, extension_factor=1.0)
        X_combined=None
        y_combined=None
    elif dataset=='diabetes':
        dataframe = pd.read_csv("datasets/" + dataset + ".csv")
        categorical_columns = detect_categorical_columns(dataframe.drop([target.target_feature()], axis=1).values)
        X_train = dataframe.drop([target.target_feature()], axis=1).values
        ht = dataframe[[target.target_feature()]].values.ravel()
        if target_val=="tested_positive":
            y_train = np.where(ht == 'tested_positive', 1, 0)
        else:
            y_train = np.where(ht == 'tested_positive', 0, 1)
        #factor=7#7
        knn = RandomForestClassifier()
        knn.fit(X_train, y_train)
        

        initial_x_s=X_train[initial_instance_index].reshape(1, -1)
        x_s = initial_x_s
            
        # # Calcular desviaciones estándar por feature
        # std_devs = np.std(X_train, axis=0)
        
        # # Guardar en la raíz
        # std_file_path = f"std_{dataset}.npy"
        # np.save(std_file_path, std_devs)
        
        # index_file = "selected_indices_diabetes.npy"

        # if not os.path.exists(index_file):
        #     # Índices fijos que quieres incluir
        #     fixed_indices = [1, 3, 5]
        
        #     # Buscar todos los índices que cumplen y_train == 0 y no son los fijos
        #     all_class_0_indices = np.where(y_train == 0)[0]
        #     remaining_indices = [idx for idx in all_class_0_indices if idx not in fixed_indices]
        
        #     # Elegir 97 aleatorios (sin reemplazo) de los restantes
        #     np.random.seed(42)  # Para reproducibilidad
        #     random_indices = np.random.choice(remaining_indices, size=97, replace=False)
        
        #     # Combinar con los índices fijos
        #     selected_indices = np.array(fixed_indices + random_indices.tolist())
        #     np.save(index_file, selected_indices)
        #     print(f"Saved Indexes {index_file}. Total: {len(selected_indices)}")
        # else:
        #     selected_indices = np.load(index_file)
        #     print(f"Load Indexes {index_file}. Total: {len(selected_indices)}")
        
        bound_vals = bounds_from_model(X_train, categorical_columns,action,x_s,extension_factor=1.0)
        #X_combined=X_train
        #y_combined=y_train
        X_combined=None
        y_combined=None
        factor=7
    elif dataset=='biomed2':
        dataframe = pd.read_csv("datasets/" + dataset + ".csv")
        X_train = dataframe.drop([target.target_feature()], axis=1).values
        ht = dataframe[[target.target_feature()]].values.ravel()
        if target_val=="normal":
            y_train = np.where(ht == 'normal', 1, 0)
        else:
            y_train = np.where(ht == 'normal', 0, 1)
        #factor=7#7
        knn = RandomForestClassifier()
        knn.fit(X_train, y_train)
        
        initial_x_s=X_train[initial_instance_index].reshape(1, -1)
        x_s = initial_x_s
        bound_vals = bounds_from_model(X_train, extension_factor=1.0)
        #X_combined=X_train
        #y_combined=y_train
        X_combined=X_train
        y_combined=y_train
        factor=7
    elif dataset=="kc2":
        dataframe = pd.read_csv("datasets/" + dataset + ".csv")
        X_train = dataframe.drop([target.target_feature()], axis=1).values
        #X_train,_,pca=preprocess_images(X_train, X_train, 'pca', 10)
        #pca=None
        ht = dataframe[[target.target_feature()]].values.ravel()
        if target_val=="yes":
            y_train = np.where(ht == 'yes', 1, 0)
        else:
            y_train = np.where(ht == 'yes', 0, 1)
        #factor=25
        #factor=25
        factor=7
        #smote = SMOTE(random_state=42)
        #X_train, y_train = smote.fit_resample(X_train, y_train)
        
        # Calcular desviaciones estándar por feature
        # std_devs = np.std(X_train, axis=0)
        
        # # Guardar en la raíz
        # std_file_path = f"std_{dataset}.npy"
        # np.save(std_file_path, std_devs)
        
        # index_file = f"selected_indices_{dataset}.npy"

        # if not os.path.exists(index_file):
        #     # Índices fijos que quieres incluir
        #     fixed_indices = [0, 10, 15]
        
        #     # Buscar todos los índices que cumplen y_train == 0 y no son los fijos
        #     all_class_0_indices = np.where(y_train == 0)[0]
        #     remaining_indices = [idx for idx in all_class_0_indices if idx not in fixed_indices]
        
        #     # Elegir 97 aleatorios (sin reemplazo) de los restantes
        #     np.random.seed(42)  # Para reproducibilidad
        #     random_indices = np.random.choice(remaining_indices, size=97, replace=False)
        
        #     # Combinar con los índices fijos
        #     selected_indices = np.array(fixed_indices + random_indices.tolist())
        #     np.save(index_file, selected_indices)
        #     print(f"Saved Indexes {index_file}. Total: {len(selected_indices)}")
        # else:
        #     selected_indices = np.load(index_file)
        #     print(f"Load Indexes {index_file}. Total: {len(selected_indices)}")

        knn = RandomForestClassifier()
        knn.fit(X_train, y_train)
        
        initial_x_s=X_train[initial_instance_index].reshape(1, -1)
        x_s = initial_x_s
        categorical_columns=[]
        action=[]
        bound_vals = bounds_from_model(X_train, categorical_columns,action,x_s,extension_factor=1.0)
        #X_combined=X_train
        #y_combined=y_train
        X_combined=None
        y_combined=None
    elif dataset=="nursery":
    # Cargar datos del dataset Nursery
        nursery = fetch_ucirepo(id=76)
        
        # Combinar características y target en un solo DataFrame
        dataframe = pd.concat([nursery.data.features, nursery.data.targets], axis=1)
        
        # Corregir error tipográfico en la columna 'form'
        dataframe['form'] = dataframe['form'].replace('completed', 'complete')
        
        # Detectar columnas categóricas
        categorical_columns = detect_categorical_columns(dataframe.drop([target.target_feature()], axis=1).values)
        
        # Definir el mapeo ordinal para las columnas categóricas
        ordinal_mapping = {
            "parents": ["usual", "pretentious", "great_pret"],
            "has_nurs": ["very_crit", "critical", "improper", "less_proper", "proper"],
            "form": ["incomplete", "complete", "foster"],
            "children": ["more", "3", "2", "1"],  # Más hijos es mejor
            "housing": ["critical", "less_conv", "convenient"],
            "finance": ["inconv", "convenient"],
            "social": ["problematic", "slightly_prob", "nonprob"],
            "health": ["not_recom", "priority", "recommended"]
        }
        
        # Transformar la clase objetivo
        #dataframe["class"] = dataframe["class"].map(lambda x: 1 if x in ["recommended", "priority"] else 0)
        ht = dataframe[[target.target_feature()]].values.ravel()

        # Definir clases que se consideran positivas
        positive_classes = ["recommend", "priority", "spec_prior", "very_recom"]
        
        y_train = np.where(np.isin(ht, positive_classes), 1, 0)

        # Aplicar codificación a las columnas categóricas
        label_columns = None  # No utilizaremos Label Encoding en este caso
        dataframe_encoded = encode_categorical_data(dataframe, ordinal_mapping=ordinal_mapping, label_columns=label_columns)
        # Separar las características (X_train) y el target (y_train)
        X_train = dataframe_encoded.drop([target.target_feature()], axis=1).values
        
        # #Calcular desviaciones estándar por feature
        # std_devs = np.std(X_train, axis=0)
        
        # # Guardar en la raíz
        # std_file_path = f"std_{dataset}.npy"
        # np.save(std_file_path, std_devs)
        
        # index_file = f"selected_indices_{dataset}.npy"

        # if not os.path.exists(index_file):
        #     # Índices fijos que quieres incluir
        #     fixed_indices = [5, 8, 20]
        
        #     # Buscar todos los índices que cumplen y_train == 0 y no son los fijos
        #     all_class_0_indices = np.where(y_train == 0)[0]
        #     remaining_indices = [idx for idx in all_class_0_indices if idx not in fixed_indices]
        
        #     # Elegir 97 aleatorios (sin reemplazo) de los restantes
        #     np.random.seed(42)  # Para reproducibilidad
        #     random_indices = np.random.choice(remaining_indices, size=97, replace=False)
        
        #     # Combinar con los índices fijos
        #     selected_indices = np.array(fixed_indices + random_indices.tolist())
        #     np.save(index_file, selected_indices)
        #     print(f"Saved Indexes {index_file}. Total: {len(selected_indices)}")
        # else:
        #     selected_indices = np.load(index_file)
        #     print(f"Load Indexes {index_file}. Total: {len(selected_indices)}")
        
        knn = RandomForestClassifier()
        knn.fit(X_train, y_train)
        initial_x_s=X_train[initial_instance_index].reshape(1, -1)
        x_s = initial_x_s
        #print(h(x_s,knn))
        bound_vals = bounds_from_model(X_train, categorical_columns,action,x_s,extension_factor=1.0)
        factor=7
        #X_combined=X_train
        #y_combined=y_train    
        X_combined=None
        y_combined=None  
                           

    elif dataset=="mnist":
        '''
        X, y = fetch_openml(data_id=554, return_X_y=True, as_frame=True)
        y = y.astype(int)  # Asegurarse de que las etiquetas sean enteros

        # Filter data
        classes_ = [8, 9]
        indices = np.random.choice(X[y.isin(classes_)].index, 3000)
        X2, y2 = X.loc[indices, :], y.loc[indices].astype(int)

        # Normalize data
        X2, y2 = preprocess_data(X2, y2)

        # convert tags: 8 -> 0 y 9 -> 1
        y2 = np.where(y2 == 8, 0, 1)

        # Dividir datos en conjunto de entrenamiento y prueba
        X_train, X_test, y_train, y_test = train_test_split(X2, y2, test_size=0.3, random_state=0)
        
        pca=None
        # pandas to numpy
        X_train = X_train.to_numpy()
        X_test= X_test.to_numpy()
        
        #X_train_meth, X_test_meth, pca = preprocess_images(X_train, X_test, 'pca', 50)
        #resampling
        
        X_train_meth=X_train
        X_test_meth=X_test
        
        X_combined = np.vstack([X_train_meth, X_test_meth])
        y_combined = np.hstack([y_train, y_test])
        '''
        (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    
        # Filtrar el conjunto de datos para que solo contenga los dígitos 8 y 9
        train_filter = np.where((y_train == 8) | (y_train == 9))
        test_filter = np.where((y_test == 8) | (y_test == 9))
        
        x_train, y_train = x_train[train_filter], y_train[train_filter]
        x_test, y_test = x_test[test_filter], y_test[test_filter]
        
        # Convertir etiquetas: 8 -> 0 y 9 -> 1
        y_train = np.where(y_train == 8, 0, 1)
        y_test = np.where(y_test == 8, 0, 1)
        
        # Normalizar datos
        x_train = preprocess_data(x_train)
        x_test = preprocess_data(x_test)
        
        # Aplanar las imágenes para PCA
        x_train = x_train.reshape(x_train.shape[0], -1)
        x_test = x_test.reshape(x_test.shape[0], -1)
        
        model_filename = "mnist_svc_model.pkl"
        #model_filename = "mnist_svc_model.pkl"
        
        if os.path.exists(model_filename):
            with open(model_filename, 'rb') as f:
                knn = pickle.load(f)
            print("Model loaded from", model_filename)
        else:
            knn = SVC(kernel='rbf', probability=True)
            knn.fit(x_train, y_train)
            with open(model_filename, 'wb') as f:
                pickle.dump(knn, f)
            print("Model trained and saved to", model_filename)
    
        
        #idx_8 = np.where(y_test == 0)[0][1]
        idx_8 = np.where(y_test == 1)[0][1]
        #idx = test_indices_8[np.random.randint(0, len(test_indices_8))]
        #x_s=X_test[idx,:].reshape(1, -1)
        x_s = x_test[idx_8, :].reshape(1, -1)
        initial_x_s=x_s
        #x_s = initial_x_s
        X_combined=x_train
        y_combined=y_train
        factor=5
        '''
        opposite_label_points = X_combined[y_combined == 1]
        distances = cdist(opposite_label_points, x_s)
        min_index = np.argmin(distances)
        closest_opposite_point = opposite_label_points[min_index]
        bound_vals = bounds_from_model(X_train, extension_factor=1.0)
        
        '''
        #factor=7
        bound_vals = bounds_from_model(x_train, extension_factor=1.0)
    elif dataset=="breast":
        dataframe = pd.read_csv("datasets/breast.csv", delimiter=';')
        categorical_columns = detect_categorical_columns(dataframe.drop([target.target_feature()], axis=1).values)
        X_train = dataframe.iloc[:, :-1].values
        ht = dataframe.iloc[:, -1].values
        if target_val == "benign":
            y_train = np.where(ht == 'benign', 1, 0)
        else:
            y_train = np.where(ht == 'benign', 0, 1)
        factor=7#7
        knn = RandomForestClassifier()
        knn.fit(X_train, y_train)
        
        initial_x_s=X_train[initial_instance_index].reshape(1, -1)
        x_s = initial_x_s
        action=[]
        
        
        
        bound_vals = bounds_from_model(X_train,categorical_columns,action,x_s, extension_factor=1.0)        
        pca=None
        X_combined=X_train
        y_combined=y_train
    pca=None    
    
    
    start_time = time.time()

    sample=sobol_sample(bound_vals, points,categorical_columns,action,seed)
    MC=1000

    X, t, kernel = None, None, None
    contador = 0
    all_costs = []
    lambdas=[]
    impros=[]
    flag=0
    
    grid_x = np.array([0])
    grid_y=grid_x
    grid = grid_x
    
    flag1=0
    x_new=None
    t_new=None
    
    X_ant=X
    #x_s_ant=x_s
    t_ant=t
    kernel_ant=kernel
    
    while True:
        x_s, flag2, X, t, kernel, cont, cost, max_prob,lambda_over,improvement = main_loop(initial_x_s,x_new,t_new,categorical_columns,sample, knn,ini_ponts,lambd,action, contador,MC,bound_vals,factor,flag,X_combined,y_combined,grid_x, grid_y, grid, X, t, kernel)  
        contador += cont
        lambdas.append(lambda_over)
        impros.append(improvement)
            
        if x_new is not None and (cdist(x_s,initial_x_s) >= cdist(x_new,initial_x_s) or flag2==0):
            x_s=x_new
            
            X=X_ant
            t=t_ant
            kernel=kernel_ant
            break
        


        if flag2:
            flag1=1;
            if x_new is None:
                x_new=x_s
            else:
                if cdist(x_s,initial_x_s)<=cdist(x_new,initial_x_s):
                    x_new=x_s
                    
        if counter>=10 and flag1:
            break
        elif counter>=10 and not flag1:
            print('No counterfactual found...')
            x_s=initial_x_s
            break
        
        counter=counter+1
        factor=factor*1
        X_ant=X
        #x_s_ant=x_s
        t_ant=t
        kernel_ant=kernel
        
    lambda_ = max(lambdas)
    end_time = time.time()
    elapsed_time = end_time - start_time


    
    if dataset=='moons':
        #grid_x, grid_y = np.mgrid[-2:3:120j, -1.5:2:120j]
        grid_x, grid_y = np.mgrid[-2:2:120j, -1.7:1.5:120j]
        grid = np.stack([grid_x, grid_y], axis=-1)    
        plt.figure(figsize=(9, 7))   
        # Graficar solo al final de todas las iteraciones
        mu_test, var_test, _ = posterior(X, t, grid.reshape(-1, 2), kernel)
        mu_test = mu_test.reshape(grid_x.shape)
        var_test = var_test.reshape(grid_x.shape)

        plot_pt_2D(grid_x, grid_y, mu_test)
        plot_db_2D(grid_x, grid_y, mu_test, decision_boundary=0.5)
        plot_data_2D(X, t)
        #plt.title((f'Iterations = {contador} with $\lambda$ = {lambd} and {ini_ponts} initial points (seed={seed})'))
        #plt.title('ASCE ')
        real_model_contour = plt.contour(grid_x, grid_y, knn.predict_proba(grid.reshape(-1, 2))[:, 1].reshape(grid_x.shape), levels=[0.5], colors='crimson', linestyles='dotted',linewidths=2)
        plt.clabel(real_model_contour, fmt={0.5: '0.5'}, fontsize=15, colors='crimson')

        #plt.plot(max_prob[0][0], max_prob[0][1],  marker='o', markerfacecolor='yellow', markeredgecolor='black', label='Max Point')
        plt.plot(initial_x_s[0][0], initial_x_s[0][1], marker='o', markersize=12, markerfacecolor='#32CD32', markeredgecolor='black', label='Instance')
        plt.plot(x_s[0][0], x_s[0][1], marker='o',markersize=12, markerfacecolor='yellow', markeredgecolor='black', label='Counterfactual')
        plt.legend()
        plt.savefig("output_plot.pdf", bbox_inches='tight', pad_inches=0, format='pdf')
        plt.show()
        print(x_s)
        print(cdist(initial_x_s,x_s))
        
    dist=cdist(x_s,initial_x_s)
    
    if dataset=="mnist":
        plot_image(initial_x_s.reshape(28, 28), title="Original Image (8)")
        #plot_and_save_image(initial_x_s.reshape(28, 28), "Original Image (8)", f'Instance_Image_factor_{factor}_ini_points_{ini_ponts}_final.pdf')
        plot_image(x_s.reshape(28, 28), title="Counterfactual Explanation (9)")
        #plot_and_save_image(x_s.reshape(28, 28), "Counterfactual Explanation (9)", f'Counterfactual_Image_factor_{factor}_ini_points_{ini_ponts}_final.pdf')
        print('Distance between counterfactual and original:', dist, ' and the factor is', factor)
    

    result_f = {
        'lambda': lambda_,
        'elapsed_time': elapsed_time,
        'cont': contador,
        'x_s': x_s,
        'instance' : initial_x_s,
        'cost': all_costs,
        'improv':impros,
        'X':X,
        't':t,
        'dist':dist,
        'factor':factor,
        'kernel':kernel
    }
    

    return result_f,knn


def fmt4(a):
    """Devuelve el array como string con 4 decimales."""
    return np.array2string(
        np.asarray(a, dtype=float),
        formatter={'float_kind': lambda x: f"{x:.4f}"},
        separator=", "
    )
def run_100_mixed_instances(dataset, target, target_val, selected_indices, std_file, ini_ponts=30, action=None, save_dir="results"):
    l2_distances, l1_distances, validity, lof_scores, cfe_solutions, n_iterations, instances = [], [], [], [], [], [], []

    std_devs = np.load(f"std_{dataset}.npy")
    
    np.random.seed(0)

    for i, idx in enumerate(selected_indices):
        result = iterative_main_loop(
            dataset=dataset,
            target=target,
            initial_instance_index=idx,
            target_val=target_val,
            ini_ponts=ini_ponts,
            seed=0,  # fijo
            action=action
        )

        instance = result[0]['instance']
        cfe = result[0]['x_s']
        dist_l2 = feature_normalized_distance(cfe.reshape(1, -1), instance.reshape(1, -1), std_devs)[0][0]
        dist_l1 = feature_normalized_l1_distance(cfe.reshape(1, -1), instance.reshape(1, -1), std_devs)[0][0]
        valid = not np.array_equal(cfe, instance)
        lof = compute_lof_affinity(cfe, result[0]['X'], 30)
        iters = result[0]['X'].shape[0] - 1

        l2_distances.append(dist_l2)
        l1_distances.append(dist_l1)
        validity.append(valid)
        lof_scores.append(lof)
        cfe_solutions.append(cfe)
        n_iterations.append(iters)
        instances.append(instance)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{dataset}_mixed_instances.npz")
    np.savez(save_path,
             l2_distances=np.array(l2_distances),
             l1_distances=np.array(l1_distances),
             validity=np.array(validity),
             lof_scores=np.array(lof_scores),
             cfe_solutions=np.array(cfe_solutions),
             n_iterations=np.array(n_iterations),
             instances=np.array(instances))
    print(f"Resultados guardados en: {save_path}")

    return {
        "l2_mean": np.mean(l2_distances),
        "l2_std": np.std(l2_distances),
        "l1_mean": np.mean(l1_distances),
        "l1_std": np.std(l1_distances),
        "validity_rate": np.mean(validity),
        "validity_std": np.std(validity),
        "lof_mean": np.mean(lof_scores),
        "lof_std": np.std(lof_scores),
        "iters_mean": np.mean(n_iterations),
        "iters_std": np.std(n_iterations),
        "cfes": np.array(cfe_solutions)
    }



def _get_bounds_local(instance: np.ndarray,
                      dataset: str,
                      PIDF_BOUNDS: np.ndarray,
                      PIDF_BOUNDS_8D: np.ndarray):
    x0 = np.asarray(instance, float).ravel()
    if dataset in ("pid", "pidf"):
        B = np.asarray(PIDF_BOUNDS, float)          # (4,2)
        local_half_width = [1.5, 1.0, 1.0, None]
    elif dataset in ("pid_cascade", "pidf_cascade"):
        B = np.asarray(PIDF_BOUNDS_8D, float)       # (8,2)
        local_half_width = [1.5, 1.0, 1.0, None, 1.5, 1.0, 1.0, None]
    else:
        raise ValueError(f"Unknown dataset '{dataset}' for local normalization.")

    lb_g, ub_g = B[:, 0], B[:, 1]
    d = len(lb_g)
    lb_loc = np.empty(d, float)
    ub_loc = np.empty(d, float)

    for i in range(d):
        if local_half_width[i] is None:
            lb_loc[i], ub_loc[i] = lb_g[i], ub_g[i]
        else:
            w = float(local_half_width[i])
            lb_loc[i] = max(lb_g[i], x0[i] - w)
            ub_loc[i] = min(ub_g[i], x0[i] + w)
            if ub_loc[i] <= lb_loc[i]:
                lb_loc[i], ub_loc[i] = lb_g[i], ub_g[i]

    rng = ub_loc - lb_loc
    rng = np.where(rng > 0, rng, (ub_g - lb_g))
    return lb_loc, ub_loc, rng


def run_100_fixed_iterations(dataset, target, target_val, fixed_index,
                             ini_ponts=30, action=None, save_dir="results",
                             save_plots=True, show_plots=True,
                             save_pdf=True, save_png=False):
    """
    Fixed-instance test (100 seeds). Calcula y guarda:
      - DELTA: D = X_out - instance
      - DELTA normalizado: Dn = D / range_local (ventana local por corrida)
      - CFEs (uno por seed) + resúmenes (mean/std) en físico y normalizado
    También guarda NPZ + (opcional) PDFs/PNGs de boxplots sin márgenes.
    """
    os.makedirs(save_dir, exist_ok=True)

    l2_distances, l1_distances, validity, lof_scores, cfe_solutions, n_iterations, real_iterations = [], [], [], [], [], [], []

    Dn_out_list = []   # Δ_norm por corrida
    D_out_list  = []   # Δ por corrida
    Dn_runs_mu  = []   # medias por corrida de Δ_norm
    dbar_runs   = []   # distancia media por corrida (Δ_norm)
    seeds_used  = []
    dims        = None
    instance_ref = None  # fijada en la primera corrida válida

    for i in range(100):
        np.random.seed(i)
        result = iterative_main_loop(
            dataset=dataset,
            target=target,
            initial_instance_index=fixed_index,
            target_val=target_val,
            ini_ponts=ini_ponts,
            seed=i,
            action=action
        )

        res0 = result[0] if isinstance(result, (list, tuple)) else result
        X = res0.get("X", None) if isinstance(res0, dict) else None
        if X is None and isinstance(result, dict):
            X = result.get("X", None)
        if X is None:
            raise RuntimeError("iterative_main_loop must return dict/list with key 'X'.")

        instance = res0["instance"]
        cfe = res0["x_s"]
        inip = res0.get("inip", None)
        if inip is None and isinstance(result, dict):
            inip = result.get("inip", None)
        if inip is None:
            inip = ini_ponts

        if dims is None:
            dims = instance.size
        if instance_ref is None:
            instance_ref = instance.copy()

        # Métricas clásicas del CFE sobre el contraejemplo (cfe vs instance)
        dist_l2 = np.linalg.norm(cfe - instance)
        dist_l1 = np.sum(np.abs(cfe - instance))
        valid = not np.array_equal(cfe, instance)
        lof = compute_lof_affinity(cfe, X, 30)
        iters = X.shape[0] - 1
        iters_real = iters - inip

        l2_distances.append(dist_l2)
        l1_distances.append(dist_l1)
        validity.append(valid)
        lof_scores.append(lof)
        cfe_solutions.append(cfe)
        n_iterations.append(iters)
        real_iterations.append(iters_real)

        # Puntos nuevos del CFE (para Δ y Δ_norm)
        X_out = X[inip+1:]
        if X_out.size == 0:
            continue

        # Normalización LOCAL por corrida (según dataset)
        _, _, range_local = _get_bounds_local(instance, dataset, PIDF_BOUNDS, PIDF_BOUNDS_8D)

        # ====== DELTA ======
        D  = X_out - instance.reshape(1, -1)              # Δ (físico)
        Dn = D / range_local.reshape(1, -1)               # Δ normalizado (local)
        # ===================

        D_out_list.append(D)
        Dn_out_list.append(Dn)
        seeds_used.append(i)

        mu_s   = Dn.mean(axis=0)
        dbar_s = np.linalg.norm(Dn, axis=1).mean()
        Dn_runs_mu.append(mu_s)
        dbar_runs.append(dbar_s)

    # Concatenaciones pooled
    dims = dims or (Dn_out_list[0].shape[1] if Dn_out_list else 0)
    absDn_pooled = np.vstack([np.abs(M) for M in Dn_out_list]) if Dn_out_list else np.empty((0, dims))
    D_pooled     = np.vstack(D_out_list)                   if D_out_list  else np.empty((0, dims))
    mu_mat       = np.asarray(Dn_runs_mu)                  if Dn_runs_mu   else np.empty((0, dims))
    dbar_vec     = np.asarray(dbar_runs)                   if dbar_runs    else np.empty((0,))

    labels = ["Kp","Ki","Kd","Tf"] if (dims == 4) else ["Kp1","Ki1","Kd1","Tf1","Kp2","Ki2","Kd2","Tf2"]

    # ==== Resumen de CFEs (uno por seed) ====
    cfe_mat = np.vstack(cfe_solutions) if cfe_solutions else np.empty((0, dims))
    if cfe_mat.size:
        # Delta del CFE respecto al instance de referencia (misma instancia para todas las corridas)
        cfe_delta = cfe_mat - instance_ref.reshape(1, -1)
        cfe_mean  = cfe_mat.mean(axis=0)
        cfe_std   = cfe_mat.std(axis=0)
        # normalización local para el resumen del CFE (ventana basada en instance_ref)
        _, _, range_local_ref = _get_bounds_local(instance_ref, dataset, PIDF_BOUNDS, PIDF_BOUNDS_8D)
        cfe_mean_norm = (cfe_mean - instance_ref) / range_local_ref
        cfe_std_norm  = cfe_std / range_local_ref
    else:
        cfe_delta     = np.empty((0, dims))
        cfe_mean      = np.array([])
        cfe_std       = np.array([])
        cfe_mean_norm = np.array([])
        cfe_std_norm  = np.array([])

    # Guardado NPZ
    #base = os.path.join(save_dir, f"{dataset}_fixed_instance_{fixed_index}")
    base = os.path.join(save_dir, f"{dataset}_fixed_instance_{fixed_index}_inip_{ini_ponts}")
    npz_path = base + ".npz"
    np.savez(npz_path,
             l2_distances=np.array(l2_distances),
             l1_distances=np.array(l1_distances),
             validity=np.array(validity),
             lof_scores=np.array(lof_scores),
             cfe_solutions=np.array(cfe_solutions),
             n_iterations=np.array(n_iterations),
             real_iterations=np.array(real_iterations),
             absDn_pooled=absDn_pooled,   # |Δ_norm| pooled
             D_pooled=D_pooled,           # Δ (físico) pooled
             mu_runs=mu_mat,              # medias por corrida (Δ_norm)
             dbar_runs=dbar_vec,          # dist. media por corrida (Δ_norm)
             seeds=np.array(seeds_used),
             # ---- CFEs resumen ----
             cfe_mean=cfe_mean,
             cfe_std=cfe_std,
             cfe_delta=cfe_delta,
             cfe_mean_norm=cfe_mean_norm,
             cfe_std_norm=cfe_std_norm,
             instance_ref=instance_ref
             )
    print(f"[saved] NPZ in: {npz_path}")

    # Helpers de guardado (PDF/PNG sin márgenes)
    def _savefig(fig_path):
        plt.tight_layout(pad=0)
        plt.savefig(fig_path, bbox_inches="tight", pad_inches=0, dpi=150)

    # Plots
    if absDn_pooled.size:
        plt.figure(figsize=(10, 4))
        plt.boxplot(absDn_pooled, labels=labels, showfliers=False)
        plt.ylabel("|Δ| normalized by LOCAL window")
        plt.title("Pooled spread of CFE proposals around instance (normalized)")
        plt.axhline(0, linewidth=1)
        if save_plots:
            if save_pdf: _savefig(base + "_box_absDn.pdf")
            if save_png: _savefig(base + "_box_absDn.png")
        if show_plots: plt.show()
        else:          plt.close()

    if D_pooled.size:
        plt.figure(figsize=(10, 4))
        plt.boxplot(D_pooled, labels=labels, showfliers=False)
        plt.ylabel("Δ (physical units)")
        plt.title("Pooled spread of CFE proposals around instance (non-normalized)")
        plt.axhline(0, linewidth=1)
        if save_plots:
            if save_pdf: _savefig(base + "_box_D.pdf")
            if save_png: _savefig(base + "_box_D.png")
        if show_plots: plt.show()
        else:          plt.close()

    if mu_mat.size:
        plt.figure(figsize=(10, 4))
        plt.boxplot(mu_mat, labels=labels, showfliers=False)
        plt.ylabel("Run-level mean Δ (normalized)")
        plt.title("Run-to-run bias per dimension (means of proposals)")
        plt.axhline(0, linewidth=1)
        if save_plots:
            if save_pdf: _savefig(base + "_box_mu.pdf")
            if save_png: _savefig(base + "_box_mu.png")
        if show_plots: plt.show()
        else:          plt.close()

    if dbar_vec.size:
        plt.figure(figsize=(6, 4))
        plt.boxplot(dbar_vec.reshape(-1, 1), labels=["mean distance"])
        plt.ylabel("Mean ||Δ||₂ (normalized by LOCAL window)")
        plt.title("Run-level mean distance to instance")
        if save_plots:
            if save_pdf: _savefig(base + "_box_dbar.pdf")
            if save_png: _savefig(base + "_box_dbar.png")
        if show_plots: plt.show()
        else:          plt.close()

    return {
        "l2_mean": float(np.mean(l2_distances)) if l2_distances else np.nan,
        "l2_std":  float(np.std(l2_distances))  if l2_distances else np.nan,
        "l1_mean": float(np.mean(l1_distances)) if l1_distances else np.nan,
        "l1_std":  float(np.std(l1_distances))  if l1_distances else np.nan,
        "validity_rate": float(np.mean(validity)) if validity else np.nan,
        "validity_std":  float(np.std(validity))  if validity else np.nan,
        "lof_mean": float(np.mean(lof_scores)) if lof_scores else np.nan,
        "lof_std":  float(np.std(lof_scores))  if lof_scores else np.nan,
        "iters_mean": float(np.mean(n_iterations)) if n_iterations else np.nan,
        "iters_std":  float(np.std(n_iterations))  if n_iterations else np.nan,
        "real_iters_mean": float(np.mean(real_iterations)) if n_iterations else np.nan,
        "real_iters_std":  float(np.std(real_iterations))  if n_iterations else np.nan,
        "seeds": np.array(seeds_used),
        "npz_path": npz_path,
        "fig_prefix": base,
        # ---- CFEs resumen ----
        "cfe_mean": cfe_mean,
        "cfe_std":  cfe_std,
        "cfe_mean_norm": cfe_mean_norm,
        "cfe_std_norm":  cfe_std_norm
    }


### ----------------------------------------------         MAIN    ----------------------------------------------###

inip=30

#PIDF
#dataset="pidf"
#ind=162

#Cascade
dataset="pid_cascade"
ind=47


action=[]
target_val=1
t = Target(target_type="classification", target_feature="Class", target_value=target_val) #Kp=3.8649,Ki=1.1121,Kd=1.5611,Tf=0.0152,Output=0
output=iterative_main_loop(dataset, t,ind,target_val, inip,seed,action)

result = output[0]   # dict
knn    = output[1]   # predictor
X=result['X'][inip+1:]
    
print("Result index",ind, ":")
print("instance:", fmt4(result['instance']))
print("x_s:", fmt4(result['x_s']))
print("Elapsed_time:", fmt4(result['elapsed_time']))

# Result index 162 :
# instance: [[3.8649, 1.1121, 1.5611, 0.0152]]
# x_s: [[3.9178, 0.9712, 1.5096, 0.0118]]
# Elapsed_time: 19.9096

# Result index 47 :
# instance: [[8.4533, 0.1358, 1.7270, 0.0104, 3.9855, 2.8415, 0.2635, 0.0200]]
# x_s: [[9.5320, 0.4416, 2.2861, 0.0367, 4.0923, 1.8915, 0.3642, 0.0210]]
# Elapsed_time: 17.0462