import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from dataclasses import dataclass
from typing import Tuple

# Scenario 1
def v1(y: float) -> float:
    return -np.sqrt(np.abs(y))

# Scenario 2
def v2(y: float) -> float:
    return np.sqrt(np.abs(y))

P1_weight = 0.5
P2_weight = 0.5

# Average expected value function
def expected_v(y: float) -> float:
    return P1_weight * v1(y) + P2_weight * v2(y)

@dataclass
class Segment:
    Y: Tuple[float, float]
    # Local models for scene 1 and 2
    a1: float
    b1: float
    c1: float
    ms1: float
    y_ms1: float
    
    a2: float
    b2: float
    c2: float
    ms2: float
    y_ms2: float
    
    lb: float # the local expected lower bound of this segment

def fit_quadratic_interpolant(v_func, Y: Tuple[float, float], n_samples: int = 10) -> Tuple[float, float, float]:
    """Fits A_s(y) = a*y^2 + b*y + c using a uniform grid of n_samples on interval Y."""
    lo, hi = Y
    ys = np.linspace(lo, hi, n_samples)
    fs = np.array([v_func(y) for y in ys])
    
    X = np.vstack((ys**2, ys, np.ones_like(ys))).T
    coefs, _, _, _ = np.linalg.lstsq(X, fs, rcond=None)
    a, b, c = coefs
    return a, b, c

import gurobipy as gp
from gurobipy import GRB

def compute_ms_shift(v_func, a: float, b: float, c: float, Y: Tuple[float, float], n_grid: int = 5000) -> Tuple[float, float]:
    """
    Computes ms = min(v_s - A_s) using Gurobi.
    v_s is either -sqrt(|y|) or sqrt(|y|).
    A_s(y) = a*y^2 + b*y + c.
    We minimize v_s(y) - A_s(y) over y in Y.
    
    Since sqrt(|y|) is nonsmooth at 0, we split the interval at 0
    and solve separate subproblems on each piece where |y| is linear.
    """
    lo, hi = Y
    
    # Determine sign of v_func to know if it's -sqrt or +sqrt
    test_val = v_func(1.0)
    is_neg_sqrt = (test_val < 0)  # v1 = -sqrt(|y|)
    
    best_ms = np.inf
    best_y = lo
    
    # Split interval at 0 if it spans both sides
    sub_intervals = []
    if lo < 0 and hi > 0:
        if lo < 0:
            sub_intervals.append((lo, 0.0, 'neg'))   # y <= 0, |y| = -y
        if hi > 0:
            sub_intervals.append((0.0, hi, 'pos'))    # y >= 0, |y| = y
    elif hi <= 0:
        sub_intervals.append((lo, hi, 'neg'))          # |y| = -y
    else:
        sub_intervals.append((lo, hi, 'pos'))          # |y| = y
    
    for (sub_lo, sub_hi, sign) in sub_intervals:
        try:
            m = gp.Model("ms_shift")
            m.Params.OutputFlag = 0
            m.Params.NonConvex = 2
            m.Params.FuncNonlinear = 1
            
            y = m.addVar(lb=sub_lo, ub=sub_hi, name="y")
            
            # t = |y|: on 'neg' side t = -y, on 'pos' side t = y
            t = m.addVar(lb=0.0, name="t")  # t = |y| >= 0
            if sign == 'neg':
                m.addConstr(t == -y, "abs_y")
            else:
                m.addConstr(t == y, "abs_y")
            
            # s = sqrt(t), modeled via s^2 = t  (i.e., t = s^2, power constraint)
            s = m.addVar(lb=0.0, name="s")  # s = sqrt(|y|) >= 0
            m.addGenConstrPow(s, t, 2.0, name="sqrt_constr")  # t = s^2
            
            # v_s(y) = ±s, A_s(y) = a*y^2 + b*y + c
            # objective: minimize v_s(y) - A_s(y)
            if is_neg_sqrt:
                # v_s = -s, so obj = -s - (a*y^2 + b*y + c)
                m.setObjective(-s - a*y*y - b*y - c, GRB.MINIMIZE)
            else:
                # v_s = s, so obj = s - (a*y^2 + b*y + c)
                m.setObjective(s - a*y*y - b*y - c, GRB.MINIMIZE)
            
            m.optimize()
            
            if m.Status == GRB.OPTIMAL:
                obj_val = m.ObjVal
                y_val = y.X
                if obj_val < best_ms:
                    best_ms = obj_val
                    best_y = y_val
        except gp.GurobiError:
            # Fallback to grid search if Gurobi fails
            ys = np.linspace(sub_lo, sub_hi, n_grid)
            A_s = a*ys**2 + b*ys + c
            v_s = np.array([v_func(y_) for y_ in ys])
            diff = v_s - A_s
            idx = np.argmin(diff)
            if diff[idx] < best_ms:
                best_ms = diff[idx]
                best_y = ys[idx]
    
    return float(best_ms), float(best_y)

def min_of_quadratic_on_interval(a: float, b: float, c: float, Y: Tuple[float, float]) -> float:
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
    return min_of_quadratic_on_interval(expected_a, expected_b, expected_c, Y)

def create_segment(Y: Tuple[float, float]) -> Segment:
    a1, b1, c1 = fit_quadratic_interpolant(v1, Y, n_samples=10)
    a2, b2, c2 = fit_quadratic_interpolant(v2, Y, n_samples=10)
    ms1, y_ms1 = compute_ms_shift(v1, a1, b1, c1, Y)
    ms2, y_ms2 = compute_ms_shift(v2, a2, b2, c2, Y)
    lb = compute_interval_lower_bound(a1, b1, c1, ms1, a2, b2, c2, ms2, Y)
    return Segment(Y, a1, b1, c1, ms1, y_ms1, a2, b2, c2, ms2, y_ms2, lb)

def run_iterative_splitting(Y0: Tuple[float, float], num_iterations: int = 20):
    output_log_file = "p4_iteration_summary.txt"
    with open(output_log_file, "w") as f:
        f.write("=== P4 Iterative Splitting Log ===\n")
        f.write(f"Initial interval: [{Y0[0]}, {Y0[1]}]\n")
    
    active_segments = [create_segment(Y0)]
    records = []
    
    for k in range(num_iterations):
        active_segments.sort(key=lambda s: s.lb)
        worst_seg = active_segments.pop(0)
        
        # Build candidate list: (ms_value, y_ms, scene_name), sorted by ms (smallest first)
        candidates = [
            (worst_seg.ms1, worst_seg.y_ms1, "Scene 1"),
            (worst_seg.ms2, worst_seg.y_ms2, "Scene 2"),
        ]
        candidates.sort(key=lambda x: x[0])  # smallest ms first (worst overestimate)
        
        # Collect ALL existing endpoints from all active segments + worst segment
        all_existing_endpoints = set()
        all_existing_endpoints.add(worst_seg.Y[0])
        all_existing_endpoints.add(worst_seg.Y[1])
        for seg in active_segments:
            all_existing_endpoints.add(seg.Y[0])
            all_existing_endpoints.add(seg.Y[1])
        
        Y = worst_seg.Y
        seg_width = Y[1] - Y[0]
        tol = 0.01 * seg_width  # 1% of segment width
        
        # Check collision: a candidate collides if it's within tol of ANY existing endpoint
        def collides(y_candidate):
            for ep in all_existing_endpoints:
                if abs(y_candidate - ep) < tol:
                    return ep  # return the endpoint it collided with
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
            # Try other candidates in order of ms value
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
                # All candidates collide, use midpoint
                y_split = (Y[0] + Y[1]) / 2.0
                chosen_scene = "Midpoint (fallback)"
                chosen_ms = candidates[0][0]
                collision_log.append(
                    f"    -> All candidates collide. Using midpoint y={y_split:.6f}"
                )
        
        # Split: create children
        new_segs = []
        for new_Y in [(Y[0], y_split), (y_split, Y[1])]:
            if new_Y[1] - new_Y[0] > 1e-6:
                new_segs.append(create_segment(new_Y))
        
        all_segs_after = sorted(active_segments + new_segs, key=lambda s: s.Y[0])
        
        # --- Compute UB: min of expected_v at all segment endpoints ---
        all_endpoints = set()
        for seg in all_segs_after:
            all_endpoints.add(seg.Y[0])
            all_endpoints.add(seg.Y[1])
        ub = min(expected_v(y) for y in all_endpoints)
        
        # --- Compute LB: min of all segment local LBs ---
        lb = min(seg.lb for seg in all_segs_after)
        
        abs_gap = ub - lb
        rel_gap = abs_gap / (max(abs(ub), abs(lb)) + 1e-12)  # avoid division by zero
        
        # Build log for this iteration
        log_lines = []
        log_lines.append(f"\n{'='*60}")
        log_lines.append(f"Iteration {k}")
        log_lines.append(f"{'='*60}")
        
        # 1. ms value and point for each scenario on the worst segment
        log_lines.append(f"\n[1] ms values (on worst segment [{worst_seg.Y[0]:.4f}, {worst_seg.Y[1]:.4f}]):")
        log_lines.append(f"    Scene 1:  ms1 = {worst_seg.ms1:.6f},  y_ms1 = {worst_seg.y_ms1:.6f}")
        log_lines.append(f"    Scene 2:  ms2 = {worst_seg.ms2:.6f},  y_ms2 = {worst_seg.y_ms2:.6f}")
        
        # 2. Chosen ms point with which scenario
        log_lines.append(f"\n[2] Chosen split point:")
        log_lines.append(f"    y_split = {y_split:.6f}  (from {chosen_scene}, ms = {chosen_ms:.6f})")
        if collision_log:
            log_lines.append(f"\n    *** COLLISION DETECTED ***")
            log_lines.extend(collision_log)
        
        # 3. Sample points per segment (post-split)
        log_lines.append(f"\n[3] Sample points (post-split, {len(all_segs_after)} segments):")
        for j, seg in enumerate(all_segs_after):
            samples = np.linspace(seg.Y[0], seg.Y[1], 10)
            sample_list = [round(float(s), 4) for s in samples]
            log_lines.append(f"    Segment {j+1} [{seg.Y[0]:.4f}, {seg.Y[1]:.4f}]:")
            log_lines.append(f"      samples = {sample_list}")
            log_lines.append(f"      ms1 = {seg.ms1:.6f} @ y = {seg.y_ms1:.6f}")
            log_lines.append(f"      ms2 = {seg.ms2:.6f} @ y = {seg.y_ms2:.6f}")
            log_lines.append(f"      local LB = {seg.lb:.6f}")
        
        # 4. Bounds
        log_lines.append(f"\n[4] Bounds:")
        log_lines.append(f"    UB = {ub:.6f}  (min of expected_v at all segment endpoints)")
        log_lines.append(f"    LB = {lb:.6f}  (min local LB across all segments)")
        log_lines.append(f"    Abs Gap = {abs_gap:.6f}")
        log_lines.append(f"    Rel Gap = {rel_gap:.6f}")
        
        with open(output_log_file, "a") as f:
            f.write("\n".join(log_lines) + "\n")
        
        # Update active_segments with the new children
        active_segments.extend(new_segs)
        active_segments.sort(key=lambda s: s.Y[0])
            
        records.append({
            'Iteration': k,
            'Worst_Interval': f"[{worst_seg.Y[0]:.4f}, {worst_seg.Y[1]:.4f}]",
            'split_point_y': y_split,
            'chosen_from': chosen_scene,
            'LB': lb,
            'UB': ub,
            'Abs Gap': abs_gap,
            'Rel Gap': rel_gap,
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
            ys_samples = np.linspace(seg.Y[0], seg.Y[1], 10)
            
            lbl_A = 'A1 (PW Interpolant)' if i == 0 else None
            lbl_U = 'U1 (PW Underestimator)' if i == 0 else None
            lbl_S = 'Samples (10 pts)' if i == 0 else None
            lbl_EP = 'Seg Endpoints' if i == 0 else None
            
            p1_interp = seg.a1*ys_seg**2 + seg.b1*ys_seg + seg.c1
            p1_under = p1_interp + seg.ms1
            axes[0].plot(ys_seg, p1_interp, '--', lw=2, color='orange', label=lbl_A)
            axes[0].plot(ys_seg, p1_under, 'b-', lw=2, label=lbl_U)
            axes[0].plot(ys_samples, [v1(y) for y in ys_samples], 'ro', markersize=3, label=lbl_S)
            axes[0].plot([seg.Y[0], seg.Y[1]], [v1(seg.Y[0]), v1(seg.Y[1])], 'gD', markersize=7, label=lbl_EP)
            axes[0].axvline(seg.Y[0], color='gray', linestyle=':', alpha=0.5)
            
            lbl_A = 'A2 (PW Interpolant)' if i == 0 else None
            lbl_U = 'U2 (PW Underestimator)' if i == 0 else None
            lbl_S = 'Samples (10 pts)' if i == 0 else None
            lbl_EP = 'Seg Endpoints' if i == 0 else None
            
            p2_interp = seg.a2*ys_seg**2 + seg.b2*ys_seg + seg.c2
            p2_under = p2_interp + seg.ms2
            axes[1].plot(ys_seg, p2_interp, '--', lw=2, color='orange', label=lbl_A)
            axes[1].plot(ys_seg, p2_under, 'b-', lw=2, label=lbl_U)
            axes[1].plot(ys_samples, [v2(y) for y in ys_samples], 'ro', markersize=3, label=lbl_S)
            axes[1].plot([seg.Y[0], seg.Y[1]], [v2(seg.Y[0]), v2(seg.Y[1])], 'gD', markersize=7, label=lbl_EP)
            axes[1].axvline(seg.Y[0], color='gray', linestyle=':', alpha=0.5)
            
            lbl_U = 'Total PW Underestimator' if i == 0 else None
            lbl_EP = 'Seg Endpoints' if i == 0 else None
            expected_under = P1_weight * p1_under + P2_weight * p2_under
            axes[2].plot(ys_seg, expected_under, 'b-', lw=2, label=lbl_U)
            axes[2].plot([seg.Y[0], seg.Y[1]], [expected_v(seg.Y[0]), expected_v(seg.Y[1])], 'gD', markersize=7, label=lbl_EP)
            axes[2].axvline(seg.Y[0], color='gray', linestyle=':', alpha=0.5)
            
        axes[2].axvline(y_split, color='purple', linestyle='--', lw=2, label=f'Split ({chosen_scene}, y={y_split:.2f})')
        axes[2].plot(y_split, expected_v(y_split), 'm*', markersize=15, label='Worst ms Location')
            
        n_segs = len(active_segments)
        axes[0].set_title(f'Scene 1 (Iter {k}, {n_segs} segments)')
        axes[0].legend()
        axes[1].set_title(f'Scene 2 (Iter {k}, {n_segs} segments)')
        axes[1].legend()
        axes[2].set_title(f'Average (Iter {k}, {n_segs} segments)')
        axes[2].legend()
        
        plt.tight_layout()
        plt.savefig(f'p4_iteration_{k}.png', dpi=150)
        plt.close(fig)
        
    df = pd.DataFrame(records)
    
    # Final report
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
    run_iterative_splitting((-10.0, 10.0), num_iterations=20)
    print(f"Results saved as p4_iteration_summary.txt and PNG files generated.")

