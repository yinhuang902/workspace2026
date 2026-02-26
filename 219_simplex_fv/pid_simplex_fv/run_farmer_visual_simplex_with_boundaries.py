#!/usr/bin/env python
"""
run_farmer_visual_simplex_with_boundaries.py
=============================================
Run a simplex-style 2D triangulation iteration on the Farmer budget face
(x_w + x_c + x_b = 500, x >= 0), producing per-iteration PNG plots.

Pre-computed boundary segments are LOADED from a JSON file produced by
``extract_farmer_linear_boundaries.py``.  This script does NOT recompute
boundaries.

Usage (from repo root):
    python run_farmer_visual_simplex_with_boundaries.py
    python run_farmer_visual_simplex_with_boundaries.py --target_nodes 30
    python run_farmer_visual_simplex_with_boundaries.py \
        --boundary_json farmer_boundary_cache/boundaries_grid50.json
"""

import argparse
import io
import json
import math
import os
import sys
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
import matplotlib
matplotlib.use("Agg")  # safe headless backend
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

# ---------------------------------------------------------------------------
# Windows console encoding fix
# ---------------------------------------------------------------------------
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding=sys.stdout.encoding, errors="replace")
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding=sys.stderr.encoding, errors="replace")

# ---------------------------------------------------------------------------
# Problem parameters  (must match run_farmer_case.py exactly)
# ---------------------------------------------------------------------------
TOTAL = 500.0

SCENARIOS = {
    "good": {"yield": 1.2, "probability": 1 / 3},
    "fair": {"yield": 1.0, "probability": 1 / 3},
    "bad":  {"yield": 0.8, "probability": 1 / 3},
}
S = len(SCENARIOS)

KNOWN_OPTIMAL = -108390.0  # Birge & Louveaux (1997)

DEFAULT_BOUNDARY_JSON = "farmer_boundary_cache/boundaries_grid25.json"

# ---------------------------------------------------------------------------
# Solver configuration  (direct Pyomo solve — no BaseBundle)
# ---------------------------------------------------------------------------
SOLVER_NAME = "gurobi"
SOLVER_OPTS = {"TimeLimit": 30}


# ---------------------------------------------------------------------------
# Build models  (same as run_farmer_case.py)
# ---------------------------------------------------------------------------
def build_farmer_models():
    """Build scenario models and return (names, models, first_vars)."""
    from farmer_problem import TwoStageFarmer

    scenario_names = list(SCENARIOS.keys())
    model_list = []
    first_vars_list = []
    for sname in scenario_names:
        scen_info = SCENARIOS[sname]
        farmer = TwoStageFarmer(scen_info["yield"])
        m = farmer.model
        m.obj_expr = pyo.Expression(expr=m.obj.expr)
        fvars = [m.x["wheat"], m.x["corn"], m.x["beets"]]
        model_list.append(m)
        first_vars_list.append(fvars)
    return scenario_names, model_list, first_vars_list


# ---------------------------------------------------------------------------
# Direct Pyomo helpers  (no BaseBundle)
# ---------------------------------------------------------------------------
def make_solver():
    """Create and configure a Pyomo solver."""
    solver = SolverFactory(SOLVER_NAME)
    for k, v in SOLVER_OPTS.items():
        solver.options[k] = v
    return solver


def set_first_stage_x(m, w, c, b):
    """Fix first-stage acreage variables on model *m*."""
    m.x["wheat"].fix(w)
    m.x["corn"].fix(c)
    m.x["beets"].fix(b)


def unfix_first_stage_x(m):
    """Unfix first-stage acreage variables on model *m*."""
    m.x["wheat"].unfix()
    m.x["corn"].unfix()
    m.x["beets"].unfix()


def solve_scenario_model(solver, m):
    """Solve model *m* and return objective value.  Variables are populated."""
    res = solver.solve(m, tee=False)
    return float(pyo.value(m.obj_expr))


# ---------------------------------------------------------------------------
# Evaluate expected Q at a 2D point
# ---------------------------------------------------------------------------
def eval_Q_at_wc(w, c, solver, model_list, first_vars_list):
    """Return per-scenario Q values at (wheat, corn, beets=TOTAL-w-c)."""
    b = TOTAL - w - c
    qs = []
    for s_idx in range(S):
        m = model_list[s_idx]
        set_first_stage_x(m, w, c, b)
        obj = solve_scenario_model(solver, m)
        unfix_first_stage_x(m)
        qs.append(obj)
    return qs


# ---------------------------------------------------------------------------
# Load pre-computed boundary segments
# ---------------------------------------------------------------------------
def load_boundaries(json_path):
    """Load boundary segments from a JSON file."""
    p = Path(json_path)
    if not p.exists():
        raise FileNotFoundError(
            f"Boundary file not found: {p}\n"
            f"Run extract_farmer_linear_boundaries.py first.")
    with open(p, "r", encoding="utf-8") as f:
        data = json.load(f)
    # segments is a list of [[w1, c1], [w2, c2]]
    segments = [tuple(tuple(pt) for pt in seg) for seg in data["segments"]]
    print(f"  Loaded {len(segments)} boundary segments from {p}")
    print(f"  Grid: {data.get('GRID_N', '?')},  "
          f"distinct signatures: {data.get('num_distinct_signatures', '?')}")
    return segments


# ---------------------------------------------------------------------------
# Local triangle split  (no scipy Delaunay)
# ---------------------------------------------------------------------------
def split_triangle(triangles, sel_idx, new_node_idx):
    """
    Replace triangle at sel_idx with three sub-triangles formed by
    inserting new_node_idx.

    triangle (a, b, c) -> (a, b, p), (b, c, p), (c, a, p)
    """
    a, b, c = triangles[sel_idx]
    p = new_node_idx
    new_tris = [(a, b, p), (b, c, p), (c, a, p)]
    # Remove old, add new
    result = [t for i, t in enumerate(triangles) if i != sel_idx]
    result.extend(new_tris)
    return result


# ---------------------------------------------------------------------------
# SurLB computation  (affine interpolation — vertex minimum)
# ---------------------------------------------------------------------------
def compute_sur_lb(tri_verts_indices, f_at_node):
    """
    Compute SurLB for a triangle using affine interpolation.

    For a linear function on a simplex, the minimum occurs at a vertex.
    SurLB = min over vertices of expected Q values (= f_at_node).
    """
    return min(f_at_node[vi] for vi in tri_verts_indices)


# ---------------------------------------------------------------------------
# Longest-edge midpoint selection
# ---------------------------------------------------------------------------
def longest_edge_midpoint(nodes_wc, tri_verts):
    """
    Return the midpoint of the longest edge of the triangle.

    Given triangle vertex indices (a, b, c) and node coords,
    find the longest edge and return its midpoint in (w, c) space.
    """
    a, b, c = tri_verts
    pa, pb, pc = (np.array(nodes_wc[a]), np.array(nodes_wc[b]),
                  np.array(nodes_wc[c]))

    edges = [(pa, pb), (pb, pc), (pc, pa)]
    lengths = [np.linalg.norm(e[1] - e[0]) for e in edges]
    best = int(np.argmax(lengths))
    u, v = edges[best]
    mid = (u + v) / 2.0
    return (float(mid[0]), float(mid[1]))


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def save_iteration_plot(it, nodes_wc, triangles, sel_idx, best_idx,
                        new_pt_wc, true_ub, sur_lb_global,
                        f_at_node, boundary_segments, tri_lb_map,
                        plot_dir):
    """Save a single iteration 2D plot."""
    fig, ax = plt.subplots(figsize=(9, 8))

    # 1) Feasible triangle boundary
    tri_bnd = plt.Polygon(
        [(0, 0), (TOTAL, 0), (0, TOTAL)],
        fill=False, edgecolor='black', lw=2, ls='-')
    ax.add_patch(tri_bnd)

    # 2) Regime boundary segments (light gray)
    if boundary_segments:
        lc_bnd = LineCollection(
            boundary_segments, colors='lightgray', linewidths=0.6,
            alpha=0.5, zorder=1, label='kink boundaries')
        ax.add_collection(lc_bnd)

    # 3) Current triangulation edges (dark thin)
    edges_set = set()
    for tri_v in triangles:
        for a, b in combinations(tri_v, 2):
            edges_set.add((min(a, b), max(a, b)))
    tri_lines = [[nodes_wc[a], nodes_wc[b]] for a, b in edges_set]
    lc_tri = LineCollection(
        tri_lines, colors='steelblue', linewidths=0.8,
        zorder=2, label='triangulation')
    ax.add_collection(lc_tri)

    # 4) Highlight selected triangle (thick red)
    if sel_idx is not None:
        si = triangles[sel_idx]
        sel_lines = [[nodes_wc[si[a]], nodes_wc[si[b]]]
                     for a, b in combinations(range(3), 2)]
        lc_sel = LineCollection(
            sel_lines, colors='red', linewidths=2.5,
            zorder=4, label='selected')
        ax.add_collection(lc_sel)

    # 5) Active / fathomed triangle shading
    for ti, tri_v in enumerate(triangles):
        lb_val = tri_lb_map.get(ti, float('inf'))
        is_active = lb_val <= true_ub + 1e-6
        poly = plt.Polygon(
            [nodes_wc[tri_v[0]], nodes_wc[tri_v[1]], nodes_wc[tri_v[2]]],
            closed=True,
            facecolor='lightyellow' if is_active else 'lavender',
            edgecolor='none', alpha=0.3, zorder=0)
        ax.add_patch(poly)

    # 6) All evaluated nodes (black dots)
    nw = [pt[0] for pt in nodes_wc]
    nc = [pt[1] for pt in nodes_wc]
    ax.scatter(nw, nc, c='black', s=15, zorder=5,
               label=f'nodes ({len(nodes_wc)})')

    # 7) Best node (red star)
    ax.scatter([nodes_wc[best_idx][0]], [nodes_wc[best_idx][1]],
               marker='*', c='red', s=200, zorder=7, label='best node')

    # 8) New point (if any)
    if new_pt_wc is not None:
        ax.scatter([new_pt_wc[0]], [new_pt_wc[1]],
                   marker='D', c='lime', s=120, edgecolors='darkgreen',
                   linewidths=1.5, zorder=8, label='new point')

    # 9) Per-triangle SurLB annotation at centroid
    for ti, tri_v in enumerate(triangles):
        lb_val = tri_lb_map.get(ti, float('inf'))
        cx = sum(nodes_wc[tri_v[j]][0] for j in range(3)) / 3
        cy = sum(nodes_wc[tri_v[j]][1] for j in range(3)) / 3
        ax.text(cx, cy, f'{lb_val:.0f}', fontsize=6, ha='center',
                va='center', color='gray', zorder=6)

    ax.set_xlim(-20, TOTAL + 20)
    ax.set_ylim(-20, TOTAL + 20)
    ax.set_xlabel("wheat (acres)", fontsize=11)
    ax.set_ylabel("corn (acres)", fontsize=11)
    ax.set_aspect('equal')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(alpha=0.15)

    gap_str = (f"{true_ub - sur_lb_global:.1f}"
               if math.isfinite(sur_lb_global) else "inf")
    ax.set_title(
        f"Iter {it} | {len(nodes_wc)} nodes  {len(triangles)} tris | "
        f"UB={true_ub:.1f}  LB={sur_lb_global:.1f}  gap={gap_str}",
        fontsize=10)

    out_path = plot_dir / f"iter_{it:03d}.png"
    fig.savefig(str(out_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out_path


# ---------------------------------------------------------------------------
# Convergence plot
# ---------------------------------------------------------------------------
def save_convergence_plot(ub_hist, lb_hist, node_hist, plot_dir):
    """Save UB/LB convergence and gap plots."""
    if not ub_hist:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax1.plot(node_hist, ub_hist, "ro-", ms=4, label="UB (best F)")
    ax1.plot(node_hist, lb_hist, "bs-", ms=4, label="SurLB")
    ax1.axhline(KNOWN_OPTIMAL, color="green", ls="--", lw=1.5,
                label=f"F*={KNOWN_OPTIMAL:.0f}")
    ax1.set_xlabel("# Nodes"); ax1.set_ylabel("Objective")
    ax1.set_title("Convergence"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    gaps = [ub - lb for ub, lb in zip(ub_hist, lb_hist)]
    ax2.semilogy(node_hist, [max(g, 1e-16) for g in gaps], "mo-", ms=4)
    ax2.set_xlabel("# Nodes"); ax2.set_ylabel("UB - LB")
    ax2.set_title("UB-LB Surrogate Gap"); ax2.grid(alpha=0.3)

    plt.tight_layout()
    conv_path = plot_dir / "convergence.png"
    fig.savefig(str(conv_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Convergence plot saved: {conv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="2D visual simplex on Farmer budget face (loads boundaries)")
    parser.add_argument(
        "--boundary_json", type=str, default=DEFAULT_BOUNDARY_JSON,
        help=f"Path to boundary JSON (default: {DEFAULT_BOUNDARY_JSON})")
    parser.add_argument(
        "--target_nodes", type=int, default=50,
        help="Max number of nodes (default: 50)")
    parser.add_argument(
        "--max_iters", type=int, default=50,
        help="Max iterations (default: 50)")
    parser.add_argument(
        "--gap_tol", type=float, default=None,
        help="UB-LB gap stopping tolerance (default: None = disabled)")
    parser.add_argument(
        "--time_limit", type=float, default=600.0,
        help="Wall-clock time limit in seconds (default: 600)")
    args = parser.parse_args()

    # Ensure local imports work
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    plot_dir = Path.cwd() / "farmer_region_plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("  Farmer 2D Simplex Visualization (with boundaries)")
    print("=" * 60)

    # ── Load boundaries ───────────────────────────────────────────
    print("\n[1] Loading boundary segments ...")
    boundary_segments = load_boundaries(args.boundary_json)

    # ── Build models ──────────────────────────────────────────────
    print("\n[2] Building Farmer scenario models ...")
    scenario_names, model_list, first_vars_list = build_farmer_models()
    print(f"    {S} scenarios built")

    # ── Create solver ─────────────────────────────────────────────
    print("\n[3] Creating solver ...")
    solver = make_solver()

    # ── Initial nodes ─────────────────────────────────────────────
    # Corners of the budget face in (w, c) space
    nodes_wc = [(0.0, 0.0), (TOTAL, 0.0), (0.0, TOTAL)]
    f_at_node = {}  # node_idx -> expected Q

    print("\n[4] Evaluating initial nodes ...")
    for i, (wv, cv) in enumerate(nodes_wc):
        qs = eval_Q_at_wc(wv, cv, solver, model_list, first_vars_list)
        f_at_node[i] = sum(qs) / S
        print(f"    node {i}: ({wv:.0f}, {cv:.0f}, {TOTAL-wv-cv:.0f})  "
              f"F={f_at_node[i]:.2f}")

    # Initial triangulation: single triangle
    triangles = [(0, 1, 2)]
    print(f"    Initial triangulation: {len(triangles)} triangle(s)")

    # ── Main simplex loop ─────────────────────────────────────────
    ub_hist, lb_hist, node_hist = [], [], []
    t_start = perf_counter()
    it = 0

    print(f"\n{'=' * 60}")
    print(f"  Running 2D simplex loop "
          f"(target={args.target_nodes} nodes, "
          f"max_iters={args.max_iters}, "
          f"gap_tol={args.gap_tol})")
    print(f"{'=' * 60}\n")

    while len(nodes_wc) < args.target_nodes and it < args.max_iters:
        N = len(nodes_wc)
        N_tri = len(triangles)

        # --- UB = best evaluated node ---
        true_ub = min(f_at_node.values())
        best_idx = min(f_at_node, key=f_at_node.get)

        # --- Compute per-triangle SurLB ---
        tri_lb_map = {}  # tri_index -> SurLB
        for ti, tri_v in enumerate(triangles):
            tri_lb_map[ti] = compute_sur_lb(tri_v, f_at_node)

        # Global SurLB
        finite_lbs = [v for v in tri_lb_map.values() if math.isfinite(v)]
        sur_lb_global = min(finite_lbs) if finite_lbs else float('inf')

        # Record
        ub_hist.append(true_ub)
        lb_hist.append(sur_lb_global)
        node_hist.append(N)

        sur_gap = true_ub - sur_lb_global

        print(f"[Iter {it:3d}] nodes={N}  tris={N_tri}  "
              f"UB={true_ub:.2f}  LB={sur_lb_global:.2f}  gap={sur_gap:.2f}")

        # Per-triangle diagnostic (abbreviated for large counts)
        if N_tri <= 20:
            for ti, tri_v in enumerate(triangles):
                lb_val = tri_lb_map[ti]
                is_active = lb_val <= true_ub + 1e-6
                tag = " *" if is_active else ""
                print(f"    t{ti} verts={list(tri_v)}  "
                      f"LB={lb_val:.2f}{tag}")

        # --- Optional gap_tol stopping (disabled by default) ---
        if args.gap_tol is not None and sur_gap <= args.gap_tol:
            save_iteration_plot(
                it, nodes_wc, triangles, None, best_idx, None,
                true_ub, sur_lb_global, f_at_node, boundary_segments,
                tri_lb_map, plot_dir)
            print(f"  STOP: UB-LB gap converged "
                  f"({sur_gap:.4e} <= {args.gap_tol}).")
            break

        # --- Time limit ---
        if args.time_limit and (perf_counter() - t_start) >= args.time_limit:
            save_iteration_plot(
                it, nodes_wc, triangles, None, best_idx, None,
                true_ub, sur_lb_global, f_at_node, boundary_segments,
                tri_lb_map, plot_dir)
            print("  STOP: time limit.")
            break

        # --- Select triangle with LOWEST SurLB ---
        valid_tis = [ti for ti in tri_lb_map
                     if math.isfinite(tri_lb_map[ti])]
        sel_idx = min(valid_tis, key=lambda ti: tri_lb_map[ti]) \
            if valid_tis else 0
        sel_tri = triangles[sel_idx]
        print(f"    -> selected tri {sel_idx}  verts={list(sel_tri)}  "
              f"LB={tri_lb_map[sel_idx]:.2f}")

        # --- New point: longest-edge midpoint ---
        new_pt_wc = longest_edge_midpoint(nodes_wc, sel_tri)

        # Collision check — fall back to centroid, then perturbed centroid
        COLLISION_TOL = 1e-8
        if any(np.linalg.norm(np.array(new_pt_wc) - np.array(n))
               < COLLISION_TOL for n in nodes_wc):
            # Fall back to centroid
            new_pt_wc = (sum(nodes_wc[vi][0] for vi in sel_tri) / 3,
                         sum(nodes_wc[vi][1] for vi in sel_tri) / 3)
            if any(np.linalg.norm(np.array(new_pt_wc) - np.array(n))
                   < COLLISION_TOL for n in nodes_wc):
                # Slight perturbation
                new_pt_wc = (new_pt_wc[0] + 1e-6, new_pt_wc[1] + 1e-6)
                b_check = TOTAL - new_pt_wc[0] - new_pt_wc[1]
                if b_check < -1e-8:
                    print("  Collision unresolvable — stopping.")
                    save_iteration_plot(
                        it, nodes_wc, triangles, sel_idx, best_idx, None,
                        true_ub, sur_lb_global, f_at_node,
                        boundary_segments, tri_lb_map, plot_dir)
                    break

        # --- Save iteration plot (BEFORE inserting new node) ---
        save_iteration_plot(
            it, nodes_wc, triangles, sel_idx, best_idx, new_pt_wc,
            true_ub, sur_lb_global, f_at_node, boundary_segments,
            tri_lb_map, plot_dir)
        print(f"    -> new point ({new_pt_wc[0]:.1f}, {new_pt_wc[1]:.1f})  "
              f"[saved iter_{it:03d}.png]")

        # --- Insert new node ---
        new_idx = len(nodes_wc)
        nodes_wc.append(new_pt_wc)
        qs = eval_Q_at_wc(new_pt_wc[0], new_pt_wc[1],
                          solver, model_list, first_vars_list)
        f_at_node[new_idx] = sum(qs) / S

        # --- Local triangle split ---
        triangles = split_triangle(triangles, sel_idx, new_idx)

        it += 1

    dt_total = perf_counter() - t_start

    # ── Results summary ───────────────────────────────────────────
    final_ub = min(f_at_node.values())
    final_lb = lb_hist[-1] if lb_hist else float('nan')
    final_gap = final_ub - final_lb

    print(f"\n{'=' * 60}")
    print(f"  RESULTS - {len(nodes_wc)} nodes, {len(triangles)} triangles")
    print(f"{'=' * 60}")
    print(f"  Known F*:      {KNOWN_OPTIMAL:.2f}")
    print(f"  Best UB:       {final_ub:.2f}")
    print(f"  SurLB:         {final_lb:.2f}")
    print(f"  Gap (UB-LB):   {final_gap:.2e}")
    if abs(final_ub - KNOWN_OPTIMAL) < 100:
        print(f"  Near known optimum "
              f"(delta = {final_ub - KNOWN_OPTIMAL:.2f})")
    else:
        print(f"  Far from known optimum "
              f"(delta = {final_ub - KNOWN_OPTIMAL:.2f})")
    print(f"  Iterations:    {it}")
    print(f"  Wall time:     {dt_total:.1f}s")
    print(f"  Plots saved:   {plot_dir}")
    print(f"{'=' * 60}")

    # ── Convergence plot ──────────────────────────────────────────
    save_convergence_plot(ub_hist, lb_hist, node_hist, plot_dir)

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        import traceback
        print("\n" + "=" * 60)
        print("FATAL ERROR:")
        print("=" * 60)
        traceback.print_exc()
        sys.exit(1)
