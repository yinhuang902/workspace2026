"""
==============================================================
Two-Scenario Cubic Optimization — Linear Surrogate
==============================================================

Problem Formulation
-------------------
minimize  v(y) = v1(y) + v2(y)    over  0 <= y <= 3

where:
    v1(y) = 1.00694*y^3 - 4.74589*y^2 + 5.17523*y
    v2(y) = -0.677232*y^3 + 3.03949*y^2 - 3.02338*y

Equivalent total:
    v(y) = 0.329708*y^3 - 1.7064*y^2 + 2.15185*y

Global optimum (analytical):
    y* ≈ 2.6200,  v(y*) ≈ -0.1459

Algorithm: Iterative piecewise linear underestimator splitting.
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from dataclasses import dataclass
from typing import Tuple
from scipy.optimize import minimize_scalar

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# ========================
# Problem Definition
# ========================

def v1(y: float) -> float:
    return 1.00694*y**3 - 4.74589*y**2 + 5.17523*y

def v2(y: float) -> float:
    return -0.677232*y**3 + 3.03949*y**2 - 3.02338*y

P1_weight = 1.0
P2_weight = 1.0

def expected_v(y: float) -> float:
    return P1_weight * v1(y) + P2_weight * v2(y)

# ========================
# Segment Data Structure
# ========================

@dataclass
class Segment:
    Y: Tuple[float, float]
    a1: float; b1: float; c1: float; ms1: float; y_ms1: float
    a2: float; b2: float; c2: float; ms2: float; y_ms2: float
    lb: float

# ========================
# Core Algorithm
# ========================

def fit_linear_interpolant(v_func, Y: Tuple[float, float]) -> Tuple[float, float, float]:
    """Fits L_s(y) = b*y + c using the two endpoints of interval Y. Returns (a=0, b, c)."""
    lo, hi = Y
    f_lo = v_func(lo)
    f_hi = v_func(hi)
    if abs(hi - lo) < 1e-12:
        return 0.0, 0.0, f_lo
    b = (f_hi - f_lo) / (hi - lo)
    c = f_lo - b * lo
    return 0.0, b, c

def compute_ms_shift(v_func, a: float, b: float, c: float, Y: Tuple[float, float]) -> Tuple[float, float]:
    """Computes ms = min(v_s - L_s) using scipy minimize_scalar."""
    lo, hi = Y
    diff_func = lambda y: v_func(y) - (a*y**2 + b*y + c)
    res = minimize_scalar(diff_func, bounds=(lo, hi), method='bounded',
                          options={'xatol': 1e-10, 'maxiter': 1000})
    vals = [(diff_func(lo), lo), (diff_func(hi), hi), (res.fun, res.x)]
    best = min(vals, key=lambda x: x[0])
    return float(best[0]), float(best[1])

def min_of_linear_on_interval(a: float, b: float, c: float, Y: Tuple[float, float]) -> float:
    lo, hi = Y
    vals = [a*lo**2 + b*lo + c, a*hi**2 + b*hi + c]
    if a > 0:
        ystar = -b / (2.0*a)
        if lo <= ystar <= hi:
            vals.append(a*ystar**2 + b*ystar + c)
    return min(vals)

def compute_interval_lower_bound(a1, b1, c1, ms1, a2, b2, c2, ms2, Y) -> float:
    expected_a = P1_weight * a1 + P2_weight * a2
    expected_b = P1_weight * b1 + P2_weight * b2
    expected_c = P1_weight * (c1 + ms1) + P2_weight * (c2 + ms2)
    return min_of_linear_on_interval(expected_a, expected_b, expected_c, Y)

def create_segment(Y: Tuple[float, float]) -> Segment:
    a1, b1, c1 = fit_linear_interpolant(v1, Y)
    a2, b2, c2 = fit_linear_interpolant(v2, Y)
    ms1, y_ms1 = compute_ms_shift(v1, a1, b1, c1, Y)
    ms2, y_ms2 = compute_ms_shift(v2, a2, b2, c2, Y)
    lb = compute_interval_lower_bound(a1, b1, c1, ms1, a2, b2, c2, ms2, Y)
    return Segment(Y, a1, b1, c1, ms1, y_ms1, a2, b2, c2, ms2, y_ms2, lb)

# ========================
# Main Iterative Splitting
# ========================

def run_iterative_splitting(Y0: Tuple[float, float], num_iterations: int = 20):
    output_log_file = os.path.join(SCRIPT_DIR, "cubic_linear_iteration_summary.txt")
    with open(output_log_file, "w") as f:
        f.write("=== Cubic Problem — Linear Surrogate Iterative Splitting Log ===\n")
        f.write(f"Initial interval: [{Y0[0]}, {Y0[1]}]\n")
        f.write("Surrogate type: LINEAR (endpoint interpolation)\n")

    active_segments = [create_segment(Y0)]
    records = []

    for k in range(num_iterations):
        active_segments.sort(key=lambda s: s.lb)
        worst_seg = active_segments.pop(0)

        candidates = [
            (worst_seg.ms1, worst_seg.y_ms1, "Scene 1"),
            (worst_seg.ms2, worst_seg.y_ms2, "Scene 2"),
        ]
        candidates.sort(key=lambda x: x[0])

        all_existing_endpoints = set()
        all_existing_endpoints.add(worst_seg.Y[0])
        all_existing_endpoints.add(worst_seg.Y[1])
        for seg in active_segments:
            all_existing_endpoints.add(seg.Y[0])
            all_existing_endpoints.add(seg.Y[1])

        Y = worst_seg.Y
        seg_width = Y[1] - Y[0]
        tol = 0.01 * seg_width

        def collides(y_candidate):
            for ep in all_existing_endpoints:
                if abs(y_candidate - ep) < tol:
                    return ep
            return None

        collision_log = []
        chosen_ms = candidates[0][0]
        y_split = candidates[0][1]
        chosen_scene = candidates[0][2]

        collided_ep = collides(y_split)
        if collided_ep is not None:
            collision_log.append(
                f"    {chosen_scene} ms point y={y_split:.6f} collides with endpoint {collided_ep:.6f}"
            )
            found_valid = False
            for ms_val, y_val, scene_name in candidates[1:]:
                ep = collides(y_val)
                if ep is not None:
                    collision_log.append(
                        f"    {scene_name} ms point y={y_val:.6f} also collides with endpoint {ep:.6f}"
                    )
                else:
                    y_split = y_val
                    chosen_scene = scene_name
                    chosen_ms = ms_val
                    collision_log.append(
                        f"    -> Selected {scene_name} ms point y={y_val:.6f} (no collision)"
                    )
                    found_valid = True
                    break
            if not found_valid:
                y_split = (Y[0] + Y[1]) / 2.0
                chosen_scene = "Midpoint (fallback)"
                chosen_ms = candidates[0][0]
                collision_log.append(
                    f"    -> All candidates collide. Using midpoint y={y_split:.6f}"
                )

        new_segs = []
        for new_Y in [(Y[0], y_split), (y_split, Y[1])]:
            if new_Y[1] - new_Y[0] > 1e-6:
                new_segs.append(create_segment(new_Y))

        all_segs_after = sorted(active_segments + new_segs, key=lambda s: s.Y[0])

        all_endpoints = set()
        for seg in all_segs_after:
            all_endpoints.add(seg.Y[0])
            all_endpoints.add(seg.Y[1])
        ub = min(expected_v(y) for y in all_endpoints)
        lb = min(seg.lb for seg in all_segs_after)

        abs_gap = ub - lb
        rel_gap = abs_gap / (max(abs(ub), abs(lb)) + 1e-12)

        log_lines = []
        log_lines.append(f"\n{'='*60}")
        log_lines.append(f"Iteration {k}")
        log_lines.append(f"{'='*60}")
        log_lines.append(f"\n[1] ms values (on worst segment [{worst_seg.Y[0]:.4f}, {worst_seg.Y[1]:.4f}]):")
        log_lines.append(f"    Scene 1:  ms1 = {worst_seg.ms1:.6f},  y_ms1 = {worst_seg.y_ms1:.6f}")
        log_lines.append(f"    Scene 2:  ms2 = {worst_seg.ms2:.6f},  y_ms2 = {worst_seg.y_ms2:.6f}")
        log_lines.append(f"\n[2] Chosen split point:")
        log_lines.append(f"    y_split = {y_split:.6f}  (from {chosen_scene}, ms = {chosen_ms:.6f})")
        if collision_log:
            log_lines.append(f"\n    *** COLLISION DETECTED ***")
            log_lines.extend(collision_log)
        log_lines.append(f"\n[3] Endpoints (post-split, {len(all_segs_after)} segments):")
        for j, seg in enumerate(all_segs_after):
            log_lines.append(f"    Segment {j+1} [{seg.Y[0]:.4f}, {seg.Y[1]:.4f}]:")
            log_lines.append(f"      endpoints = [{seg.Y[0]:.4f}, {seg.Y[1]:.4f}]")
            log_lines.append(f"      L1(y) = {seg.b1:.6f}*y + {seg.c1:.6f}")
            log_lines.append(f"      L2(y) = {seg.b2:.6f}*y + {seg.c2:.6f}")
            log_lines.append(f"      ms1 = {seg.ms1:.6f} @ y = {seg.y_ms1:.6f}")
            log_lines.append(f"      ms2 = {seg.ms2:.6f} @ y = {seg.y_ms2:.6f}")
            log_lines.append(f"      local LB = {seg.lb:.6f}")
        log_lines.append(f"\n[4] Bounds:")
        log_lines.append(f"    UB = {ub:.6f}  (min of expected_v at all segment endpoints)")
        log_lines.append(f"    LB = {lb:.6f}  (min local LB across all segments)")
        log_lines.append(f"    Abs Gap = {abs_gap:.6f}")
        log_lines.append(f"    Rel Gap = {rel_gap:.6f}")

        with open(output_log_file, "a") as f:
            f.write("\n".join(log_lines) + "\n")

        active_segments.extend(new_segs)
        active_segments.sort(key=lambda s: s.Y[0])

        records.append({
            'Iteration': k,
            'Worst_Interval': f"[{worst_seg.Y[0]:.4f}, {worst_seg.Y[1]:.4f}]",
            'split_point_y': y_split,
            'chosen_from': chosen_scene,
            'LB': lb, 'UB': ub,
            'Abs Gap': abs_gap, 'Rel Gap': rel_gap,
            'Num Segments': len(active_segments)
        })

        # Plotting
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        ys_global = np.linspace(Y0[0], Y0[1], 1000)
        axes[0].plot(ys_global, [v1(y) for y in ys_global], 'k-', lw=4, alpha=0.3, label='True v1')
        axes[1].plot(ys_global, [v2(y) for y in ys_global], 'k-', lw=4, alpha=0.3, label='True v2')
        axes[2].plot(ys_global, [expected_v(y) for y in ys_global], 'k-', lw=3, label='True Expected Total')

        for i, seg in enumerate(active_segments):
            ys_seg = np.linspace(seg.Y[0], seg.Y[1], 100)

            lbl_A = 'L1 (PW Linear)' if i == 0 else None
            lbl_U = 'U1 (PW Underestimator)' if i == 0 else None
            lbl_E = 'Endpoints' if i == 0 else None
            p1_interp = seg.a1*ys_seg**2 + seg.b1*ys_seg + seg.c1
            p1_under = p1_interp + seg.ms1
            axes[0].plot(ys_seg, p1_interp, '--', lw=2, color='orange', label=lbl_A)
            axes[0].plot(ys_seg, p1_under, 'b-', lw=2, label=lbl_U)
            axes[0].plot([seg.Y[0], seg.Y[1]], [v1(seg.Y[0]), v1(seg.Y[1])], 'ro', markersize=5, label=lbl_E)
            axes[0].axvline(seg.Y[0], color='gray', linestyle=':', alpha=0.5)

            lbl_A = 'L2 (PW Linear)' if i == 0 else None
            lbl_U = 'U2 (PW Underestimator)' if i == 0 else None
            lbl_E = 'Endpoints' if i == 0 else None
            p2_interp = seg.a2*ys_seg**2 + seg.b2*ys_seg + seg.c2
            p2_under = p2_interp + seg.ms2
            axes[1].plot(ys_seg, p2_interp, '--', lw=2, color='orange', label=lbl_A)
            axes[1].plot(ys_seg, p2_under, 'b-', lw=2, label=lbl_U)
            axes[1].plot([seg.Y[0], seg.Y[1]], [v2(seg.Y[0]), v2(seg.Y[1])], 'ro', markersize=5, label=lbl_E)
            axes[1].axvline(seg.Y[0], color='gray', linestyle=':', alpha=0.5)

            lbl_U = 'Total PW Underestimator' if i == 0 else None
            expected_under = P1_weight * p1_under + P2_weight * p2_under
            axes[2].plot(ys_seg, expected_under, 'b-', lw=2, label=lbl_U)
            axes[2].axvline(seg.Y[0], color='gray', linestyle=':', alpha=0.5)

        axes[2].axvline(y_split, color='purple', linestyle='--', lw=2, label=f'Split ({chosen_scene}, y={y_split:.2f})')
        axes[2].plot(y_split, expected_v(y_split), 'm*', markersize=15, label='Worst ms Location')

        axes[0].set_title(f'Scene 1 (Iter {k})')
        axes[0].legend()
        axes[1].set_title(f'Scene 2 (Iter {k})')
        axes[1].legend()
        axes[2].set_title(f'Average Scenario (Iter {k})')
        axes[2].legend()
        plt.tight_layout()
        plt.savefig(os.path.join(SCRIPT_DIR, f'cubic_linear_iteration_{k}.png'), dpi=150)
        plt.close(fig)

    df = pd.DataFrame(records)
    final_lb = records[-1]['LB']
    final_ub = records[-1]['UB']
    final_abs = records[-1]['Abs Gap']
    final_rel = records[-1]['Rel Gap']

    print("\n=== Iteration Summary Table ===")
    print(df.to_string(index=False))
    print(f"\n=== Final Result ===")
    print(f"  LB = {final_lb:.6f}")
    print(f"  UB = {final_ub:.6f}")
    print(f"  Absolute Gap = {final_abs:.6f}")
    print(f"  Relative Gap = {final_rel:.6f}")

    with open(output_log_file, "a") as f:
        f.write("\n\n=== Final Summary Table ===\n")
        f.write(df.to_string(index=False))
        f.write(f"\n\n=== Final Result ===\n")
        f.write(f"  LB = {final_lb:.6f}\n")
        f.write(f"  UB = {final_ub:.6f}\n")
        f.write(f"  Absolute Gap = {final_abs:.6f}\n")
        f.write(f"  Relative Gap = {final_rel:.6f}\n")

    return df

if __name__ == "__main__":
    run_iterative_splitting((0.0, 3.0), num_iterations=20)
    print("Results saved as cubic_linear_iteration_summary.txt and PNG files generated.")
