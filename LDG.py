# -*- coding: utf-8 -*-
"""
Created on Mon Oct  6 14:58:55 2025

@author: mags3
"""

# === pid_dataset_local.py ===
# Densify pid.csv locally around one or more anchor points using truncated normal
# (never leaves [LB, UB]), then simulate and append (4-decimal, no duplicates).

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import truncnorm
from simulation_2 import (
    evaluate_pidf,              # <-- ADD
    evaluate_pidf_identify,
    SIGMA_Y_DEFAULT, N_DEFAULT, TS_DEFAULT, R_STEP_DEFAULT
)

# --- Bounds for [Kp, Ki, Kd, Tf]
BOUNDS = np.array([
    [0.5, 10.0],  # Kp
    [0.1,  4.0],  # Ki
    [0.1,  5.0],  # Kd
    [0.01, 0.10], # Tf
], dtype=float)

DECIMALS = 4
def fmt(x: float) -> str:
    return f"{float(x):.{DECIMALS}f}"

def truncated_around(x_center: np.ndarray,
                     n: int = 100,
                     radius: float = 0.10,
                     seed: int = 0,
                     bounds: np.ndarray = BOUNDS) -> np.ndarray:
    """
    Draw n samples around x_center using a truncated normal within 'bounds'.
    - radius: fraction of (ub - lb) used as per-dimension std.
    """
    rng = np.random.default_rng(seed)
    bounds = np.asarray(bounds, float)
    lb, ub = bounds[:,0], bounds[:,1]

    x_center = np.asarray(x_center, float).ravel()
    std = np.maximum(radius * (ub - lb), 1e-6)  # avoid zero std
    a = (lb - x_center) / std
    b = (ub - x_center) / std

    out = np.zeros((n, 4))
    for i in range(4):
        out[:, i] = truncnorm.rvs(a[i], b[i], loc=x_center[i], scale=std[i],
                                  size=n, random_state=rng)
    return np.clip(out, lb, ub)

def truncated_around_vec(x_center: np.ndarray,
                         n: int = 100,
                         radius: float = 0.08,
                         radius_vec: np.ndarray | None = None,
                         seed: int = 0,
                         bounds: np.ndarray = BOUNDS) -> np.ndarray:
    """
    Draw n samples around x_center using a truncated normal within 'bounds'.
    You can use a per-dimension radius via radius_vec (fraction of (ub-lb)).
    """
    rng = np.random.default_rng(seed)
    bounds = np.asarray(bounds, float)
    lb, ub = bounds[:, 0], bounds[:, 1]

    x_center = np.asarray(x_center, float).ravel()
    base = (ub - lb)
    if radius_vec is None:
        std = np.maximum(radius * base, 1e-6)
    else:
        rv = np.asarray(radius_vec, float).ravel()
        std = np.maximum(rv * base, 1e-6)

    a = (lb - x_center) / std
    b = (ub - x_center) / std

    out = np.zeros((n, 4))
    for i in range(4):
        out[:, i] = truncnorm.rvs(a[i], b[i], loc=x_center[i], scale=std[i],
                                  size=n, random_state=rng)
    return np.clip(out, lb, ub)

def directional_pushes(x_center: np.ndarray,
                       bounds: np.ndarray = BOUNDS,
                       scales: tuple[float, float, float] = (0.25, 0.35, -0.30)) -> np.ndarray:
    """
    Build a small deterministic set of 'pushed' points to likely induce Class=0:
    - Increase Kp by ~25%, Ki by ~35%, decrease Tf by ~30% (clip to bounds).
    - Also a couple of stronger pushes.
    """
    lb, ub = bounds[:,0], bounds[:,1]
    x = np.asarray(x_center, float).ravel().copy()

    kp_s, ki_s, tf_s = scales
    # Base pushes
    P = []
    p1 = x.copy()
    p1[0] *= (1.0 + kp_s)   # Kp up
    p1[1] *= (1.0 + ki_s)   # Ki up
    p1[3] *= (1.0 + tf_s)   # Tf down if tf_s negative
    P.append(np.clip(p1, lb, ub))

    # Stronger Kp,Ki increase; Tf decrease
    p2 = x.copy()
    p2[0] *= 1.50
    p2[1] *= 1.50
    p2[3] *= 0.70
    P.append(np.clip(p2, lb, ub))

    # Moderate Kd up with small Tf down
    p3 = x.copy()
    p3[2] *= 1.40
    p3[3] *= 0.85
    P.append(np.clip(p3, lb, ub))

    # Small Kp up only
    p4 = x.copy()
    p4[0] *= 1.20
    P.append(np.clip(p4, lb, ub))

    return np.vstack(P)


def densify_smart(points: np.ndarray,
                  n_per_point: int = 200,
                  radii: tuple[float, float] = (0.03, 0.08),
                  radius_vec: np.ndarray | None = np.array([0.08, 0.10, 0.10, 0.04]),
                  seed: int = 0,
                  r_value: float = 1.0,
                  sigma_y: float = 0.01,
                  N: int = 1200,
                  Ts: float = 0.1,
                  csv_name: str = "pid.csv",
                  bounds: np.ndarray = BOUNDS,
                  target_pos_range: tuple[float, float] = (0.25, 0.75)) -> None:
    """
    For each anchor point:
      - sample half points with small radius, half with medium radius (optionally radius_vec),
      - add a few directional pushes,
      - simulate and append (4-dec, no duplicates).
    Prints class ratio for the batch; aim to get both 0 and 1 locally.
    """
    rows = []
    rng = np.random.default_rng(seed)

    for p in np.asarray(points, float):
        # 1) stochastic: half small-radius, half medium-radius
        n_small = n_per_point // 2
        n_med   = n_per_point - n_small

        X_small = truncated_around_vec(p, n=n_small, radius=radii[0],
                                       radius_vec=radius_vec, seed=int(rng.integers(1e9)),
                                       bounds=bounds)
        X_med   = truncated_around_vec(p, n=n_med,   radius=radii[1],
                                       radius_vec=radius_vec, seed=int(rng.integers(1e9)),
                                       bounds=bounds)

        # 2) deterministic: directional pushes
        X_push  = directional_pushes(p, bounds=bounds)

        Xloc = np.vstack([X_small, X_med, X_push])

        # 3) simulate all
        for Kp, Ki, Kd, Tf in Xloc:
            try:
                label, mets, theta_hat, _ = evaluate_pidf_identify(
                    Kp, Ki, Kd, Tf,
                    r_value=R_STEP_DEFAULT,
                    sigma_y=SIGMA_Y_DEFAULT,
                    seed=0,
                    N=N_DEFAULT,
                    Ts=TS_DEFAULT
                )
                Kp_h, Ki_h, Kd_h, Tf_h = theta_hat
                rows.append([fmt(Kp_h), fmt(Ki_h), fmt(Kd_h), fmt(Tf_h), int(label)])
            except Exception as e:
                # skip any numerical failure
                pass

    _append_rows_to_pid_csv(rows, csv_name=csv_name)

    # report local balance for this batch
    df_batch = pd.DataFrame(rows, columns=["Kp","Ki","Kd","Tf","Class"])
    if not df_batch.empty:
        c = df_batch["Class"].astype(int).to_numpy()
        pos = (c == 1).mean()
        print(f"[batch] positives={pos*100:.1f}%  ({c.sum()}/{len(c)})  target≈{int(100*target_pos_range[0])}–{int(100*target_pos_range[1])}%")


def _append_rows_to_pid_csv(rows, csv_name: str = "pid.csv") -> None:
    """
    Append rows to pid.csv (next to this script) avoiding duplicates on (Kp,Ki,Kd,Tf).
    Robust to Windows network/offline files: no Path.exists(); try-read instead.
    """
    # Write next to this script (not CWD) to avoid network/offline surprises
    base_dir = Path(__file__).resolve().parent
    csv_path = (base_dir / csv_name)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    # Build new batch as strings (4 decimals)
    df_new = pd.DataFrame(rows, columns=["Kp","Ki","Kd","Tf","Class"]).astype(str)
    if df_new.empty:
        print("[pid.csv] nothing to append (rows empty).")
        return

    try:
        # Try to read existing CSV; if it fails, we'll create a new one
        df_old = pd.read_csv(csv_path, dtype=str)
        # Normalize whitespace
        for c in ["Kp","Ki","Kd","Tf","Class"]:
            if c in df_old.columns:
                df_old[c] = df_old[c].astype(str).str.strip()
        for c in ["Kp","Ki","Kd","Tf","Class"]:
            df_new[c] = df_new[c].astype(str).str.strip()

        # Concatenate and de-duplicate by the 4 feature columns
        before = len(df_old)
        df_all = pd.concat([df_old, df_new], axis=0, ignore_index=True)
        df_all.drop_duplicates(subset=["Kp","Ki","Kd","Tf"], keep="first", inplace=True)
        df_all.to_csv(csv_path, index=False)
        added = len(df_all) - before
        print(f"[pid.csv] total={len(df_all)} (added {added}).  -> {csv_path}")
    except (FileNotFoundError, OSError, pd.errors.EmptyDataError):
        # File doesn't exist or is unavailable: create it now
        df_new.to_csv(csv_path, index=False)
        print(f"[pid.csv] created with {len(df_new)} rows.  -> {csv_path}")


def densify_around_points(points: np.ndarray,
                          n_per_point: int = 100,
                          radius: float = 0.08,
                          seed: int = 0,
                          r_value: float = 1.0,
                          sigma_y: float = 0.01,
                          N: int = 1200,
                          Ts: float = 0.1,
                          csv_name: str = "pid.csv",
                          bounds: np.ndarray = BOUNDS) -> None:
    """
    For each anchor point, draw n_per_point truncated-normal samples in 'bounds',
    simulate via evaluate_pidf, and append to pid.csv avoiding duplicates.
    """
    rows = []
    s = int(seed)
    for p in np.asarray(points, float):
        Xloc = truncated_around(p, n=n_per_point, radius=radius, seed=s, bounds=bounds)
        s += 1
        for Kp, Ki, Kd, Tf in Xloc:
            label, mets, theta_hat, _ = evaluate_pidf_identify(
                Kp, Ki, Kd, Tf,
                r_value=R_STEP_DEFAULT,
                sigma_y=SIGMA_Y_DEFAULT,
                seed=0,
                N=N_DEFAULT,
                Ts=TS_DEFAULT
            )
            Kp_h, Ki_h, Kd_h, Tf_h = theta_hat
            rows.append([fmt(Kp_h), fmt(Ki_h), fmt(Kd_h), fmt(Tf_h), int(label)])

    _append_rows_to_pid_csv(rows, csv_name=csv_name)

def sanity_check_theta_ldg(theta):
    kp, ki, kd, tf = map(float, theta)

    # Ambas evalúan métricas SOBRE LA MISMA SIMULACIÓN;
    # la versión *_identify además devuelve theta_hat (gains identificados).
    lab_eval, mets_eval, _ = evaluate_pidf(
        kp, ki, kd, tf,
        sigma_y=SIGMA_Y_DEFAULT, N=N_DEFAULT, Ts=TS_DEFAULT, r_value=R_STEP_DEFAULT
    )

    lab_id, mets_id, theta_hat, _ = evaluate_pidf_identify(
        kp, ki, kd, tf,
        sigma_y=SIGMA_Y_DEFAULT, N=N_DEFAULT, Ts=TS_DEFAULT, r_value=R_STEP_DEFAULT
    )

    print("\n[LDG sanity]")
    print(f"theta_in    = [{kp:.4f}, {ki:.4f}, {kd:.4f}, {tf:.4f}]")
    print(f"label_eval  = {lab_eval}  (evaluation only)")
    print(f"label_ident = {lab_id}    (simulate + identify, metrics from simulation)")
    print(f"theta_hat   = [{theta_hat[0]:.4f}, {theta_hat[1]:.4f}, {theta_hat[2]:.4f}, {theta_hat[3]:.4f}]")

    if lab_eval != lab_id:
        print("WARNING: labels differ -> revisa que SIGMA/N/Ts/r_value/u_min/u_max sean iguales.")


# if __name__ == "__main__":
#     theta = [3.7033, 0.806, 1.3528, 0.0103]
#     sanity_check_theta_ldg(theta)
    
if __name__ == "__main__":
    anchor = np.array([4.1012, 0.8, 1.9999, 0.01], float)

    densify_smart([anchor],
        n_per_point=250,
        radii=(0.03, 0.08),
        radius_vec=np.array([0.25, 0.2, 0.2, 0.04]),
        seed=123,
        r_value=R_STEP_DEFAULT,
        sigma_y=SIGMA_Y_DEFAULT,
        N=N_DEFAULT,
        Ts=TS_DEFAULT,
        csv_name="pid.csv",
        bounds=BOUNDS,
        target_pos_range=(0.25, 0.75)
    )


# anchor = np.array([4.1012, 0.8, 1.9999, 0.01], float)

# densify_smart([anchor],
#     n_per_point=250,
#     radii=(0.03, 0.08),
#     #radius_vec=np.array([0.08, 0.10, 0.10, 0.04]),
#     radius_vec = np.array([0.25, 0.2, 0.2, 0.04]),
#     seed=123,
#     r_value=R_STEP_DEFAULT,    # 1.0
#     sigma_y=SIGMA_Y_DEFAULT,   # 0.01
#     N=N_DEFAULT,               # 8000
#     Ts=TS_DEFAULT,             # tu Ts del modelo
#     csv_name="pid.csv",
#     bounds=BOUNDS,             # en LDG: [0.5,0.1,0.1,0.01]–[10,4,5,0.1]
#     target_pos_range=(0.25, 0.75)
# )
# --- Example usage:
# x_s = np.array([1.2, 0.4, 0.05, 0.02])
# densify_around_points([x_s], n_per_point=200, radius=0.07, seed=123)

# df = pd.read_csv("pid.csv")
# pos = df[df["Class"]=="1"][["Kp","Ki","Kd","Tf"]].astype(float).values
# idx = np.random.default_rng(0).choice(len(pos), size=min(5,len(pos)), replace=False)
# seed_points = pos[idx]
# densify_around_points(seed_points, n_per_point=150, radius=0.06, seed=999)
