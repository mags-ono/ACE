# -*- coding: utf-8 -*-
"""
LDG_cascade.py — Local Densification Generator for CASCADED PIDF (8D)

Generates labeled data around one or more 8D anchor points:
  theta = [Kp1, Ki1, Kd1, Tf1, Kp2, Ki2, Kd2, Tf2]

For each sampled point, it:
  1) Simulates the cascaded loops (outer C1, inner C2) with those gains,
  2) Identifies BOTH controllers via fit_pidf_gains:
        - C1 from (r1, u1, y1)
        - C2 from (r2, u2, y2)
  3) Computes pass/fail label using OUTER LOOP metrics and your criteria,
  4) Appends the IDENTIFIED gains (theta1_hat + theta2_hat) + label to CSV,
     avoiding duplicates on the 8 features.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import truncnorm

from simulation_2 import (
    evaluate_pidf_identify_cascade, evaluate_pidf_cascade,
    SIGMA_Y_DEFAULT, SIGMA_Y_DEFAULT_2, N_DEFAULT, TS_DEFAULT, R_STEP_DEFAULT, SETTLE_EPS_DEFAULT
)

# -------- Bounds for 8D [C1 then C2]: [Kp, Ki, Kd, Tf] x 2 --------
# Mirror the single-loop bounds for each controller.
BOUNDS_8 = np.array([
    [0.5, 10.0],  # Kp1
    [0.1,  4.0],  # Ki1
    [0.1,  5.0],  # Kd1
    [0.01, 0.10], # Tf1
    [0.5, 10.0],  # Kp2
    [0.1,  4.0],  # Ki2
    [0.1,  5.0],  # Kd2
    [0.01, 0.10], # Tf2
], dtype=float)

DECIMALS = 4
# def fmt(x: float) -> str:
#     return f"{float(x):.{DECIMALS}f}"
def fmt(x: float) -> float:
    return round(float(x), DECIMALS)

def _truncnorm_vec(center: np.ndarray, lb: np.ndarray, ub: np.ndarray, std: np.ndarray, n: int, rng) -> np.ndarray:
    a = (lb - center) / std
    b = (ub - center) / std
    out = np.zeros((n, center.size))
    for i in range(center.size):
        out[:, i] = truncnorm.rvs(a[i], b[i], loc=center[i], scale=std[i],
                                  size=n, random_state=rng)
    return np.clip(out, lb, ub)

def truncated_around_vec_8d(x_center: np.ndarray,
                            n: int = 100,
                            radius: float = 0.08,
                            radius_vec: np.ndarray | None = None,
                            seed: int = 0,
                            bounds: np.ndarray = BOUNDS_8) -> np.ndarray:
    """
    Draw n samples around 8D center using truncated normal within 'bounds'.
    - radius (or radius_vec) is a fraction of (ub - lb) per dimension.
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

    return _truncnorm_vec(x_center, lb, ub, std, n, rng)

def directional_pushes_8d(x_center: np.ndarray,
                          bounds: np.ndarray = BOUNDS_8) -> np.ndarray:
    """
    Deterministic 'pushes' likely to produce both stable and borderline behavior.
    We nudge Kp/Ki up and Tf down for both loops, and a couple of Kd pushes.
    """
    lb, ub = bounds[:, 0], bounds[:, 1]
    x = np.asarray(x_center, float).ravel().copy()

    P = []

    # Push 1: Both loops slightly more aggressive (Kp↑, Ki↑, Tf↓)
    p1 = x.copy()
    p1[0] *= 1.25; p1[1] *= 1.25; p1[3] *= 0.80   # C1
    p1[4] *= 1.20; p1[5] *= 1.20; p1[7] *= 0.85   # C2
    P.append(np.clip(p1, lb, ub))

    # Push 2: Stronger Kp/Ki on C1, moderate on C2
    p2 = x.copy()
    p2[0] *= 1.50; p2[1] *= 1.40; p2[3] *= 0.75
    p2[4] *= 1.20; p2[5] *= 1.10; p2[7] *= 0.90
    P.append(np.clip(p2, lb, ub))

    # Push 3: Emphasize derivative on both
    p3 = x.copy()
    p3[2] *= 1.40; p3[6] *= 1.40
    p3[3] *= 0.90; p3[7] *= 0.90
    P.append(np.clip(p3, lb, ub))

    # Push 4: Mild Kp1 only and Kp2 only
    p4 = x.copy(); p4[0] *= 1.15; P.append(np.clip(p4, lb, ub))
    p5 = x.copy(); p5[4] *= 1.15; P.append(np.clip(p5, lb, ub))

    return np.vstack(P)

def _append_rows_to_csv_8d(rows, csv_name: str = "pid_cascade.csv") -> None:
    """
    Append rows with IDENTIFIED (HAT) gains + Class to an existing CSV,
    de-duplicating by the 8 HAT parameters (like LDG hace con 4D).
    Mantiene floats y redondea a 4 decimales al escribir.
    """
    base_dir = Path(__file__).resolve().parent
    csv_path = (base_dir / csv_name)
    csv_path.parent.mkdir(parents=True, exist_ok=True)

    cols = ["Kp1","Ki1","Kd1","Tf1","Kp2","Ki2","Kd2","Tf2","Class"]
    feat_cols = ["Kp1","Ki1","Kd1","Tf1","Kp2","Ki2","Kd2","Tf2"]

    df_new = pd.DataFrame(rows, columns=cols)
    if df_new.empty:
        print("[pid_cascade.csv] nothing to append (rows empty).")
        return

    # asegurar tipos numéricos y redondeo a 4 decimales
    df_new[feat_cols] = df_new[feat_cols].astype(float).round(4)
    df_new["Class"]   = df_new["Class"].astype(int)

    try:
        df_old = pd.read_csv(csv_path)  # deja que pandas infiera floats/ints
        before = len(df_old)
        df_all = pd.concat([df_old, df_new], axis=0, ignore_index=True)

        # >>> igual que en LDG: dedup por features (acá 8 HAT), sin usar Class
        df_all.drop_duplicates(subset=feat_cols, keep="first", inplace=True)
        # <<<

        df_all.to_csv(csv_path, index=False)
        added = len(df_all) - before
        print(f"[pid_cascade.csv] total={len(df_all)} (added {added}).  -> {csv_path}")
    except (FileNotFoundError, OSError, pd.errors.EmptyDataError):
        df_new.to_csv(csv_path, index=False)
        print(f"[pid_cascade.csv] created with {len(df_new)} rows.  -> {csv_path}")


def densify_smart_cascade(points8: np.ndarray,
                          n_per_point: int = 200,
                          radii: tuple[float, float] = (0.03, 0.08),
                          radius_vec: np.ndarray | None = np.array([
                              0.25, 0.20, 0.20, 0.04,   # C1 per-dim radii
                              0.25, 0.20, 0.20, 0.04    # C2 per-dim radii
                          ]),
                          seed: int = 0,
                          r_value: float = R_STEP_DEFAULT,
                          sigma_y1: float = SIGMA_Y_DEFAULT,
                          sigma_y2: float = SIGMA_Y_DEFAULT_2,
                          N: int = N_DEFAULT,
                          Ts: float = TS_DEFAULT,
                          csv_name: str = "pid_cascade.csv",
                          bounds8: np.ndarray = BOUNDS_8,
                          target_pos_range: tuple[float, float] = (0.25, 0.75)
                          ) -> None:
    """
    For each 8D anchor point:
      - sample half with small radius, half with medium radius (optionally per-dim radius_vec),
      - add deterministic directional pushes,
      - simulate (cascade), identify BOTH controllers,
      - **RE-LABEL on the IDENTIFIED gains (HAT) with the SAME seed_i**,
      - append HAT gains + Class to CSV.
    """
    rows = []
    rng = np.random.default_rng(seed)

    attempts = 0
    success  = 0

    for p in np.asarray(points8, float):
        # 1) stochastic: half small-radius, half medium-radius
        n_small = n_per_point // 2
        n_med   = n_per_point - n_small

        X_small = truncated_around_vec_8d(p, n=n_small, radius=radii[0],
                                          radius_vec=radius_vec, seed=int(rng.integers(1_000_000_000)),
                                          bounds=bounds8)
        X_med   = truncated_around_vec_8d(p, n=n_med,   radius=radii[1],
                                          radius_vec=radius_vec, seed=int(rng.integers(1_000_000_000)),
                                          bounds=bounds8)

        # 2) deterministic pushes
        X_push  = directional_pushes_8d(p, bounds=bounds8)

        Xloc = np.vstack([X_small, X_med, X_push])

        # 3) simulate + identify + RE-LABEL on HAT
        for row in Xloc:
            attempts += 1
            Kp1, Ki1, Kd1, Tf1, Kp2, Ki2, Kd2, Tf2 = map(float, row[:8])
            try:
                # unique seed per sample to diversify noise/trajectories
                seed_i = int(rng.integers(1_000_000_000))

                # (a) simulate + identify (this computes label on TRUE internally, but we won't use it)
                _, _, th1_hat, th2_hat, _ = evaluate_pidf_identify_cascade(
                    Kp1, Ki1, Kd1, Tf1, Kp2, Ki2, Kd2, Tf2,
                    r_value=r_value, N=N, Ts=Ts,
                    sigma_y1=sigma_y1, sigma_y2=sigma_y2,
                    seed=seed_i, settle_eps=SETTLE_EPS_DEFAULT, trim_ic=100
                )
                Kp1h, Ki1h, Kd1h, Tf1h = th1_hat
                Kp2h, Ki2h, Kd2h, Tf2h = th2_hat

                # (b) RE-LABEL **on HAT** with the **same seed_i**
                label_hat, _, _ = evaluate_pidf_cascade(
                    Kp1h, Ki1h, Kd1h, Tf1h, Kp2h, Ki2h, Kd2h, Tf2h,
                    r_value=r_value, N=N, Ts=Ts,
                    sigma_y1=sigma_y1, sigma_y2=sigma_y2,
                    seed=seed_i, settle_eps=SETTLE_EPS_DEFAULT
                )

                # (c) store HAT + label_hat (DECIMALS=4 via fmt, CSV writer redondea floats)
                rows.append([
                    fmt(Kp1h), fmt(Ki1h), fmt(Kd1h), fmt(Tf1h),
                    fmt(Kp2h), fmt(Ki2h), fmt(Kd2h), fmt(Tf2h),
                    int(label_hat)
                ])
                success += 1

            except Exception:
                # Skip numerical failures silently to keep generation robust
                pass

    _append_rows_to_csv_8d(rows, csv_name=csv_name)

    # Report local class balance and counters for this batch
    if rows:
        df_batch = pd.DataFrame(rows, columns=["Kp1","Ki1","Kd1","Tf1","Kp2","Ki2","Kd2","Tf2","Class"])
        c = df_batch["Class"].astype(int).to_numpy()
        pos = (c == 1).mean()
        lo, hi = target_pos_range
        print(f"[batch] attempted={attempts}, succeeded={success}, positives={pos*100:.1f}%  ({c.sum()}/{len(c)})  target≈{int(100*lo)}–{int(100*hi)}%")
    else:
        print(f"[batch] attempted={attempts}, succeeded={success}, no rows produced.")

# def densify_smart_cascade(points8: np.ndarray,
#                           n_per_point: int = 200,
#                           radii: tuple[float, float] = (0.03, 0.08),
#                           radius_vec: np.ndarray | None = np.array([
#                               0.25, 0.20, 0.20, 0.04,   # C1 per-dim radii
#                               0.25, 0.20, 0.20, 0.04    # C2 per-dim radii
#                           ]),
#                           seed: int = 0,
#                           r_value: float = R_STEP_DEFAULT,
#                           sigma_y1: float = 0.01,
#                           sigma_y2: float = 0.005,
#                           N: int = N_DEFAULT,
#                           Ts: float = TS_DEFAULT,
#                           csv_name: str = "pid_cascade.csv",
#                           bounds8: np.ndarray = BOUNDS_8,
#                           target_pos_range: tuple[float, float] = (0.25, 0.75)
#                           ) -> None:
#     """
#     For each 8D anchor point:
#       - sample half with small radius, half with medium radius (optionally per-dim radius_vec),
#       - add a few deterministic directional pushes,
#       - simulate (cascade), identify both controllers, and append IDENTIFIED gains + Class.
#     """
#     rows = []
#     rng = np.random.default_rng(seed)

#     attempts = 0
#     success  = 0

#     for p in np.asarray(points8, float):
#         # 1) stochastic: half small-radius, half medium-radius
#         n_small = n_per_point // 2
#         n_med   = n_per_point - n_small

#         X_small = truncated_around_vec_8d(p, n=n_small, radius=radii[0],
#                                           radius_vec=radius_vec, seed=int(rng.integers(1_000_000_000)),
#                                           bounds=bounds8)
#         X_med   = truncated_around_vec_8d(p, n=n_med,   radius=radii[1],
#                                           radius_vec=radius_vec, seed=int(rng.integers(1_000_000_000)),
#                                           bounds=bounds8)

#         # 2) deterministic pushes
#         X_push  = directional_pushes_8d(p, bounds=bounds8)

#         Xloc = np.vstack([X_small, X_med, X_push])

#         # 3) simulate + identify + label
#         for row in Xloc:
#             attempts += 1
#             Kp1, Ki1, Kd1, Tf1, Kp2, Ki2, Kd2, Tf2 = map(float, row[:8])
#             try:
#                 # unique seed per sample to diversify noise/trajectories
#                 seed_i = int(rng.integers(1_000_000_000))
#                 label, mets, th1_hat, th2_hat, _ = evaluate_pidf_identify_cascade(
#                     Kp1, Ki1, Kd1, Tf1, Kp2, Ki2, Kd2, Tf2,
#                     r_value=r_value, N=N, Ts=Ts,
#                     sigma_y1=sigma_y1, sigma_y2=sigma_y2,
#                     seed=seed_i, settle_eps=0.05, trim_ic=100
#                 )
#                 Kp1h, Ki1h, Kd1h, Tf1h = th1_hat
#                 Kp2h, Ki2h, Kd2h, Tf2h = th2_hat
#                 rows.append([
#                     fmt(Kp1h), fmt(Ki1h), fmt(Kd1h), fmt(Tf1h),
#                     fmt(Kp2h), fmt(Ki2h), fmt(Kd2h), fmt(Tf2h),
#                     int(label)
#                 ])
#                 success += 1
#             except Exception:
#                 # Skip numerical failures silently to keep generation robust
#                 pass

#     _append_rows_to_csv_8d(rows, csv_name=csv_name)

#     # Report local class balance for this batch
#     if rows:
#         df_batch = pd.DataFrame(rows, columns=["Kp1","Ki1","Kd1","Tf1","Kp2","Ki2","Kd2","Tf2","Class"])
#         c = df_batch["Class"].astype(int).to_numpy()
#         pos = (c == 1).mean()
#         lo, hi = target_pos_range
#         print(f"[batch] attempted={attempts}, succeeded={success}, positives={pos*100:.1f}%  ({c.sum()}/{len(c)})  target≈{int(100*lo)}–{int(100*hi)}%")
#     else:
#         print(f"[batch] attempted={attempts}, succeeded={success}, no rows produced.")


def sanity_check_theta_ldg_cascade(theta8):
    """
    Quick check: run one identify call and print the result.
    theta8 = [Kp1,Ki1,Kd1,Tf1, Kp2,Ki2,Kd2,Tf2]
    """
    Kp1, Ki1, Kd1, Tf1, Kp2, Ki2, Kd2, Tf2 = map(float, theta8)
    label, mets, th1_hat, th2_hat, _ = evaluate_pidf_identify_cascade(
        Kp1, Ki1, Kd1, Tf1, Kp2, Ki2, Kd2, Tf2,
        r_value=R_STEP_DEFAULT, N=N_DEFAULT, Ts=TS_DEFAULT,
        sigma_y1=SIGMA_Y_DEFAULT, sigma_y2=SIGMA_Y_DEFAULT_2,
        seed=0, settle_eps=SETTLE_EPS_DEFAULT, trim_ic=100
    )
    print("\n[LDG cascade sanity]")
    print(f"theta_in  = [{Kp1:.4f},{Ki1:.4f},{Kd1:.4f},{Tf1:.4f}, {Kp2:.4f},{Ki2:.4f},{Kd2:.4f},{Tf2:.4f}]")
    print(f"label     = {label}")
    print(f"theta1hat = [{th1_hat[0]:.4f},{th1_hat[1]:.4f},{th1_hat[2]:.4f},{th1_hat[3]:.4f}]")
    print(f"theta2hat = [{th2_hat[0]:.4f},{th2_hat[1]:.4f},{th2_hat[2]:.4f},{th2_hat[3]:.4f}]")

if __name__ == "__main__":
    # Example anchors (outer C1 near your healthy point; inner C2 a bit tighter/faster)
    anchor_c1 = np.array([4, 0.5, 1.1, 0.01], float)
    anchor_c2 = np.array([3, 1, 0.50, 0.02], float)
    anchor8   = np.r_[anchor_c1, anchor_c2]

    # Single-anchor densification (adjust n_per_point / radii as needed)
    densify_smart_cascade([anchor8],
        n_per_point=700,
        radii=(0.03, 0.08),
        radius_vec=np.array([0.3, 0.2, 0.2, 0.04, 0.3, 0.5, 0.8, 0.04]),
        seed=0,
        r_value=R_STEP_DEFAULT,
        sigma_y1=SIGMA_Y_DEFAULT,
        sigma_y2=SIGMA_Y_DEFAULT_2,
        N=N_DEFAULT,
        Ts=TS_DEFAULT,
        csv_name="pid_cascade.csv",
        bounds8=BOUNDS_8,
        target_pos_range=(0.25, 0.75)
    )

    # Example sanity check (optional):
    # sanity_check_theta_ldg_cascade(anchor8)
