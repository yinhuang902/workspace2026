# run_farmer_visual_simplex.py  –  Standalone 2D visual simplex on Farmer budget face
"""
Classic Farmer problem (3 scenarios good/fair/bad) visualized on the
2D budget face  x_w + x_c + x_b = 500,  x ≥ 0.

Works in the (wheat, corn) plane with  beets = 500 − wheat − corn.
Each iteration produces a PNG in  farmer_region_plots/.

Paste the entire file into a Jupyter notebook cell and run.
"""

# %%  ======= Imports & setup ===================================================
import io, math, os, sys
from pathlib import Path
from time import perf_counter
from itertools import combinations

import numpy as np
import matplotlib
matplotlib.use("Agg")                    # safe headless backend
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection
from scipy.spatial import Delaunay
import pyomo.environ as pyo

# Make sure local modules are importable
script_dir = Path.cwd()
if str(script_dir) not in sys.path:
    sys.path.insert(0, str(script_dir))

np.random.seed(42)

# %%  ======= Problem parameters ================================================
TOTAL = 500.0

SCENARIOS = {
    "good": {"yield": 1.2, "probability": 1 / 3},
    "fair": {"yield": 1.0, "probability": 1 / 3},
    "bad":  {"yield": 0.8, "probability": 1 / 3},
}
S = len(SCENARIOS)

# Runner knobs
TARGET_NODES = 50
GAP_STOP_TOL = 1e-2
TIME_LIMIT   = 600.0

UB_SOLVER_OPTS = {}
LB_SOLVER_OPTS = {"TimeLimit": 30}

KNOWN_OPTIMAL = -108390.0       # Birge & Louveaux (1997)

PLOT_DIR = Path.cwd() / "farmer_region_plots"

# Grid resolution for kink boundary precomputation
GRID_N = 80

# %%  ======= Build scenario models =============================================

def build_farmer_models():
    from farmer_problem import TwoStageFarmer

    scenario_names = list(SCENARIOS.keys())
    model_list, first_vars_list = [], []
    for sname in scenario_names:
        farmer = TwoStageFarmer(SCENARIOS[sname]["yield"])
        m = farmer.model
        # Patch obj_expr (needed by BaseBundle / MSBundle)
        m.obj_expr = pyo.Expression(expr=m.obj.expr)
        fvars = [m.x["wheat"], m.x["corn"], m.x["beets"]]
        model_list.append(m)
        first_vars_list.append(fvars)
    return scenario_names, model_list, first_vars_list


print("Building farmer scenario models …")
scenario_names, model_list, first_vars_list = build_farmer_models()
print(f"  {S} scenarios built")

# %%  ======= Build bundles =====================================================

from bundles import BaseBundle, MSBundle
from utils import evaluate_Q_at
from simplex_specialstart import ms_on_tetra_for_scene, solve_surrogate_lb_for_tet

print("Building BaseBundle + MSBundle …")
base_bundles = [BaseBundle(m, UB_SOLVER_OPTS) for m in model_list]
ms_bundles = [
    MSBundle(m, fv, LB_SOLVER_OPTS, scenario_index=s)
    for s, (m, fv) in enumerate(zip(model_list, first_vars_list))
]
print("  Done.")

# %%  ======= Helper: evaluate Q_s at a 2D point ===============================

def eval_Q_at_wc(w, c):
    """Return per-scenario Q values at (wheat, corn, beets=TOTAL-w-c)."""
    b = TOTAL - w - c
    pt3 = (w, c, b)
    return [evaluate_Q_at(base_bundles[s], first_vars_list[s], pt3)
            for s in range(S)]


def F_true(w, c):
    """Expected recourse F = (1/S) sum_s Q_s."""
    return sum(eval_Q_at_wc(w, c)) / S


# %%  ======= Precompute kink / linear-region boundaries ========================

print(f"\nPrecomputing regime boundaries (grid {GRID_N}×{GRID_N})  …")
t0_grid = perf_counter()

# --- Build dense grid on the 2D feasible triangle ---
ww = np.linspace(0, TOTAL, GRID_N)
cc = np.linspace(0, TOTAL, GRID_N)
grid_pts = []       # (idx, w, c)
grid_idx = {}       # (iw, ic) -> idx
for iw, wv in enumerate(ww):
    for ic, cv in enumerate(cc):
        if wv + cv <= TOTAL + 1e-8:
            idx = len(grid_pts)
            grid_pts.append((wv, cv))
            grid_idx[(iw, ic)] = idx

n_grid = len(grid_pts)
print(f"  {n_grid} grid points …")

# --- Evaluate each grid point and extract regime signature ---
# Regime per scenario: which recourse decisions are non-zero?
#   wheat:  buy_w > tol  or  sell_w > tol
#   corn:   buy_c > tol  or  sell_c > tol
#   beets:  beets_favorable near 6000 (quota binding)

def regime_signature_at(w, c):
    """Solve each scenario, extract regime bits, return combined signature."""
    b = TOTAL - w - c
    pt3 = (w, c, b)
    sigs = []
    for s_idx in range(S):
        bb = base_bundles[s_idx]
        # Fix first-stage vars and solve
        fvars = first_vars_list[s_idx]
        # Use eval_at with return_meta to get the solved model state
        q_val = bb.eval_at(fvars, pt3)
        m = bb.model

        tol = 1e-4
        bits = []
        # Wheat buy/sell
        try:
            bits.append(1 if pyo.value(m.y["wheat"]) > tol else 0)
        except:
            bits.append(-1)
        try:
            bits.append(1 if pyo.value(m.w["wheat"]) > tol else 0)
        except:
            bits.append(-1)
        # Corn buy/sell
        try:
            bits.append(1 if pyo.value(m.y["corn"]) > tol else 0)
        except:
            bits.append(-1)
        try:
            bits.append(1 if pyo.value(m.w["corn"]) > tol else 0)
        except:
            bits.append(-1)
        # Beets favorable at quota?
        try:
            bits.append(1 if pyo.value(m.w["beets_favorable"]) > 6000 - tol else 0)
        except:
            bits.append(-1)
        sigs.append(tuple(bits))
    return tuple(sigs)


# Evaluate signatures (this is the slow part)
signatures = [None] * n_grid
for pi, (wv, cv) in enumerate(grid_pts):
    if pi % 200 == 0:
        print(f"    grid point {pi}/{n_grid} …")
    signatures[pi] = regime_signature_at(wv, cv)

# --- Extract boundary edges ---
# Build adjacency from (iw,ic) grid: connect horizontal and vertical neighbours
boundary_segments = []
for (iw, ic), idx_a in grid_idx.items():
    for diw, dic in [(1, 0), (0, 1), (1, 1), (1, -1)]:
        nb = (iw + diw, ic + dic)
        if nb in grid_idx:
            idx_b = grid_idx[nb]
            if signatures[idx_a] != signatures[idx_b]:
                pa = grid_pts[idx_a]
                pb = grid_pts[idx_b]
                # Midpoint on the edge (approximate boundary location)
                boundary_segments.append(((pa[0], pa[1]), (pb[0], pb[1])))

dt_grid = perf_counter() - t0_grid
print(f"  {len(boundary_segments)} boundary segments found in {dt_grid:.1f}s")

# %%  ======= Initial nodes on 2D triangle =====================================

# Corners of the budget face in (w, c) space
nodes_wc = [(0.0, 0.0), (TOTAL, 0.0), (0.0, TOTAL)]

# Evaluate Q at initial nodes
scen_values = {}   # scen_values[s][node_idx] = Q_s
for s_idx in range(S):
    scen_values[s_idx] = {}
for i, (wv, cv) in enumerate(nodes_wc):
    qs = eval_Q_at_wc(wv, cv)
    for s_idx in range(S):
        scen_values[s_idx][i] = qs[s_idx]

f_at_node = {i: sum(scen_values[s][i] for s in range(S)) / S
             for i in range(len(nodes_wc))}

# Initial Delaunay
tri_del = Delaunay(np.array(nodes_wc))
triangles = [tuple(s) for s in tri_del.simplices]

print(f"\nInitial nodes ({len(nodes_wc)}):")
for i, (wv, cv) in enumerate(nodes_wc):
    print(f"  ({wv:.0f}, {cv:.0f}, {TOTAL-wv-cv:.0f})  F={f_at_node[i]:.2f}")
print(f"Initial triangles: {len(triangles)}")

# %%  ======= Plotting helper ===================================================

PLOT_DIR.mkdir(parents=True, exist_ok=True)


def save_iteration_plot(it, nodes_wc, triangles, sel_rec, best_idx, new_pt_wc,
                        true_ub, sur_lb_s, f_at_node, boundary_segments,
                        tri_records, fname=None):
    """Save a single iteration 2D plot."""
    fig, ax = plt.subplots(figsize=(9, 8))

    # 1) Feasible triangle boundary
    tri_bnd = plt.Polygon([(0, 0), (TOTAL, 0), (0, TOTAL)],
                           fill=False, edgecolor='black', lw=2, ls='-')
    ax.add_patch(tri_bnd)

    # 2) Kink / regime boundary segments (light gray)
    if boundary_segments:
        lc_bnd = LineCollection(boundary_segments,
                                colors='lightgray', linewidths=0.6, alpha=0.5,
                                zorder=1, label='kink boundaries')
        ax.add_collection(lc_bnd)

    # 3) Current triangulation edges
    edges_set = set()
    for tri_v in triangles:
        for a, b in combinations(tri_v, 2):
            edges_set.add((min(a, b), max(a, b)))
    tri_lines = [[nodes_wc[a], nodes_wc[b]] for a, b in edges_set]
    lc_tri = LineCollection(tri_lines, colors='steelblue', linewidths=0.8,
                            zorder=2, label='triangulation')
    ax.add_collection(lc_tri)

    # 4) Highlight selected triangle
    if sel_rec is not None:
        si = sel_rec["vert_idxs"]
        sel_lines = [[nodes_wc[si[a]], nodes_wc[si[b]]]
                      for a, b in combinations(range(3), 2)]
        lc_sel = LineCollection(sel_lines, colors='red', linewidths=2.5,
                                zorder=4, label='selected')
        ax.add_collection(lc_sel)

    # 5) Active vs fathomed triangles shading
    for rec in tri_records:
        vi = rec["vert_idxs"]
        is_active = rec["SurLB"] / S <= true_ub + 1e-6
        poly = plt.Polygon([nodes_wc[vi[0]], nodes_wc[vi[1]], nodes_wc[vi[2]]],
                            closed=True,
                            facecolor='lightyellow' if is_active else 'lavender',
                            edgecolor='none', alpha=0.3, zorder=0)
        ax.add_patch(poly)

    # 6) All evaluated nodes (black dots)
    nw = [nodes_wc[i][0] for i in range(len(nodes_wc))]
    nc = [nodes_wc[i][1] for i in range(len(nodes_wc))]
    ax.scatter(nw, nc, c='black', s=15, zorder=5, label=f'nodes ({len(nodes_wc)})')

    # 7) Best node (star)
    ax.scatter([nodes_wc[best_idx][0]], [nodes_wc[best_idx][1]],
               marker='*', c='red', s=200, zorder=7, label='best node')

    # 8) New point (if any)
    if new_pt_wc is not None:
        ax.scatter([new_pt_wc[0]], [new_pt_wc[1]],
                   marker='D', c='lime', s=120, edgecolors='darkgreen',
                   linewidths=1.5, zorder=8, label='new point')

    # 9) Per-triangle SurLB/S annotation (small text at centroid)
    for rec in tri_records:
        vi = rec["vert_idxs"]
        cx = sum(nodes_wc[vi[j]][0] for j in range(3)) / 3
        cy = sum(nodes_wc[vi[j]][1] for j in range(3)) / 3
        ax.text(cx, cy, f'{rec["SurLB"]/S:.0f}', fontsize=6, ha='center',
                va='center', color='gray', zorder=6)

    ax.set_xlim(-20, TOTAL + 20)
    ax.set_ylim(-20, TOTAL + 20)
    ax.set_xlabel("wheat (acres)", fontsize=11)
    ax.set_ylabel("corn (acres)", fontsize=11)
    ax.set_aspect('equal')
    ax.legend(fontsize=7, loc='upper right')
    ax.grid(alpha=0.15)

    gap_str = f"{true_ub - sur_lb_s:.1f}" if math.isfinite(sur_lb_s) else "∞"
    ax.set_title(
        f"Iter {it} | {len(nodes_wc)} nodes  {len(triangles)} tris | "
        f"UB={true_ub:.1f}  LB={sur_lb_s:.1f}  gap={gap_str}",
        fontsize=10)

    out = fname or (PLOT_DIR / f"iter_{it:03d}.png")
    fig.savefig(str(out), dpi=150, bbox_inches='tight')
    plt.close(fig)
    return out

# %%  ======= Main simplex loop ================================================

ms_cache = {}
true_ub_hist, sur_lb_hist, node_count_hist = [], [], []
t_start = perf_counter()
it = 0

print(f"\n{'='*70}")
print(f"  Running 2D simplex loop (target={TARGET_NODES} nodes, tol={GAP_STOP_TOL})")
print(f"{'='*70}\n")

while len(nodes_wc) < TARGET_NODES:
    N = len(nodes_wc)
    N_tri = len(triangles)

    # --- UB = best evaluated node ---
    true_ub = min(f_at_node.values())
    best_idx = min(f_at_node, key=f_at_node.get)

    # --- Build per-triangle records ---
    tri_records = []
    for t_idx, tri_verts in enumerate(triangles):
        idxs = list(tri_verts)
        # 3D vertices (w, c, b)
        verts_3d = [(nodes_wc[vi][0], nodes_wc[vi][1],
                      TOTAL - nodes_wc[vi][0] - nodes_wc[vi][1])
                     for vi in idxs]
        fverts_per_scene = [[scen_values[s_idx][vi] for vi in idxs]
                            for s_idx in range(S)]

        ms_scene, xms_scene, c_scene, cpts_scene = [], [], [], []
        for w in range(S):
            cache_key = (w, tuple(sorted(idxs)))
            if cache_key in ms_cache:
                ms_val, pt_ms, c_val, c_pt = ms_cache[cache_key]
            else:
                ms_val, pt_ms, c_val, c_pt = ms_on_tetra_for_scene(
                    ms_bundles[w], verts_3d, fverts_per_scene[w])
                ms_cache[cache_key] = (ms_val, pt_ms, c_val, c_pt)
            ms_scene.append(ms_val)
            xms_scene.append(pt_ms)
            c_scene.append(c_val)
            cpts_scene.append(c_pt)

        sur_lb = solve_surrogate_lb_for_tet(fverts_per_scene, ms_scene, c_scene)
        best_sc = int(np.argmin(ms_scene))

        tri_records.append({
            "vert_idxs": idxs, "verts_3d": verts_3d,
            "SurLB": sur_lb,
            "ms_per_scene": list(ms_scene),
            "fverts_per_scene": [list(row) for row in fverts_per_scene],
            "x_ms_best_scene": xms_scene[best_sc],
            "c_point_per_scene": cpts_scene,
        })

    if not tri_records:
        print("  No triangles — stopping."); break

    sur_lb_vals = [r["SurLB"] for r in tri_records if math.isfinite(r["SurLB"])]
    sur_lb_global = min(sur_lb_vals) if sur_lb_vals else float('inf')

    # Record
    true_ub_hist.append(true_ub)
    sur_lb_hist.append(sur_lb_global / S)
    node_count_hist.append(N)

    sur_gap = true_ub - sur_lb_global / S

    print(f"[Iter {it:3d}] nodes={N}  tris={N_tri}  "
          f"UB={true_ub:.2f}  LB/S={sur_lb_global/S:.2f}  gap={sur_gap:.2f}")

    # Per-triangle diagnostic
    for ti, rec in enumerate(tri_records):
        is_active = rec["SurLB"] / S <= true_ub + 1e-6
        tag = " *" if is_active else ""
        print(f"    t{ti} verts={rec['vert_idxs']}  LB/S={rec['SurLB']/S:.2f}{tag}")

    # --- Stopping ---
    if GAP_STOP_TOL and sur_gap <= GAP_STOP_TOL:
        # Plot final state then stop
        save_iteration_plot(it, nodes_wc, triangles, None, best_idx, None,
                            true_ub, sur_lb_global / S, f_at_node,
                            boundary_segments, tri_records)
        print(f"  STOP: UB-LB gap converged ({sur_gap:.4e} <= {GAP_STOP_TOL}).")
        break
    if TIME_LIMIT and (perf_counter() - t_start) >= TIME_LIMIT:
        save_iteration_plot(it, nodes_wc, triangles, None, best_idx, None,
                            true_ub, sur_lb_global / S, f_at_node,
                            boundary_segments, tri_records)
        print("  STOP: time limit.")
        break

    # --- Select triangle with LOWEST SurLB ---
    valid_recs = [r for r in tri_records if math.isfinite(r["SurLB"])]
    sel = min(valid_recs, key=lambda r: r["SurLB"]) if valid_recs else tri_records[0]
    sv3 = sel["verts_3d"]
    print(f"    → select tri verts={sel['vert_idxs']}  LB/S={sel['SurLB']/S:.2f}")

    # --- Candidate point ---
    cand_pt = sel["x_ms_best_scene"]
    if cand_pt is None:
        for cp in sel["c_point_per_scene"]:
            if cp is not None:
                cand_pt = cp; break
    if cand_pt is None:
        # Fallback: centroid of selected triangle in 3D
        cand_pt = tuple(np.mean(sv3, axis=0))

    # Project to 2D (w, c), keep beets = TOTAL - w - c
    new_w = max(0.0, float(cand_pt[0]))
    new_c = max(0.0, float(cand_pt[1]))
    # Ensure feasibility: w + c <= TOTAL
    if new_w + new_c > TOTAL + 1e-10:
        scale = TOTAL / (new_w + new_c)
        new_w *= scale; new_c *= scale
    new_pt_wc = (new_w, new_c)

    # Collision check
    if any(np.linalg.norm(np.array(new_pt_wc) - np.array(n)) < 1e-8
           for n in nodes_wc):
        # Use centroid instead
        si = sel["vert_idxs"]
        new_pt_wc = (sum(nodes_wc[si[j]][0] for j in range(3)) / 3,
                     sum(nodes_wc[si[j]][1] for j in range(3)) / 3)
        if any(np.linalg.norm(np.array(new_pt_wc) - np.array(n)) < 1e-8
               for n in nodes_wc):
            print("  Collision — stopping.")
            break

    # --- Save iteration plot (BEFORE inserting new node) ---
    save_iteration_plot(it, nodes_wc, triangles, sel, best_idx, new_pt_wc,
                        true_ub, sur_lb_global / S, f_at_node,
                        boundary_segments, tri_records)
    print(f"    → new point ({new_pt_wc[0]:.1f}, {new_pt_wc[1]:.1f})  "
          f"[saved iter_{it:03d}.png]")

    # --- Insert new node ---
    new_idx = len(nodes_wc)
    nodes_wc.append(new_pt_wc)
    qs = eval_Q_at_wc(new_pt_wc[0], new_pt_wc[1])
    for s_idx in range(S):
        scen_values[s_idx][new_idx] = qs[s_idx]
    f_at_node[new_idx] = sum(qs) / S

    # Re-triangulate
    ms_cache.clear()
    try:
        tri_del = Delaunay(np.array(nodes_wc))
        triangles = [tuple(s) for s in tri_del.simplices]
    except Exception as e:
        print(f"  Delaunay failed: {e}"); break

    it += 1

dt_total = perf_counter() - t_start
print(f"\n--- Done in {dt_total:.1f}s,  {len(nodes_wc)} nodes,  "
      f"{len(triangles)} triangles ---")

# %%  ======= Results summary ===================================================

final_ub = min(f_at_node.values())
final_lb = sur_lb_hist[-1] if sur_lb_hist else float('nan')
final_gap = final_ub - final_lb

print("\n" + "=" * 60)
print(f"  Known F*:      {KNOWN_OPTIMAL:.2f}")
print(f"  Best UB:       {final_ub:.2f}")
print(f"  SurLB/S:       {final_lb:.2f}")
print(f"  Gap (UB-LB):   {final_gap:.2e}")
if abs(final_ub - KNOWN_OPTIMAL) < 100:
    print(f"  ✔  Near known optimum (Δ = {final_ub - KNOWN_OPTIMAL:.2f})")
else:
    print(f"  ✗  Far from known optimum (Δ = {final_ub - KNOWN_OPTIMAL:.2f})")
print(f"  Final nodes:   {len(nodes_wc)}")
print(f"  Final tris:    {len(triangles)}")
print(f"  Plots saved:   {PLOT_DIR}")
print("=" * 60)

# %%  ======= Convergence plot ==================================================

if true_ub_hist:
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))

    ax1.plot(node_count_hist, true_ub_hist, "ro-", ms=4, label="UB (best F)")
    ax1.plot(node_count_hist, sur_lb_hist, "bs-", ms=4, label="SurLB/S")
    ax1.axhline(KNOWN_OPTIMAL, color="green", ls="--", lw=1.5,
                label=f"F*={KNOWN_OPTIMAL:.0f}")
    ax1.set_xlabel("# Nodes"); ax1.set_ylabel("Objective")
    ax1.set_title("Convergence"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    gaps = [ub - lb for ub, lb in zip(true_ub_hist, sur_lb_hist)]
    ax2.semilogy(node_count_hist, [max(g, 1e-16) for g in gaps], "mo-", ms=4)
    ax2.set_xlabel("# Nodes"); ax2.set_ylabel("UB − LB")
    ax2.set_title("UB–LB Surrogate Gap"); ax2.grid(alpha=0.3)

    plt.tight_layout()
    conv_path = PLOT_DIR / "convergence.png"
    fig.savefig(str(conv_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nConvergence plot saved: {conv_path}")

# %%  ======= Final node distribution on 2D triangle ===========================

fig, ax = plt.subplots(figsize=(8, 8))
tri_bnd = plt.Polygon([(0, 0), (TOTAL, 0), (0, TOTAL)],
                       fill=False, edgecolor='black', lw=2)
ax.add_patch(tri_bnd)

# Kink boundaries
if boundary_segments:
    lc = LineCollection(boundary_segments, colors='lightgray',
                        linewidths=0.6, alpha=0.5, zorder=1)
    ax.add_collection(lc)

# Triangulation edges
edges_set = set()
for trv in triangles:
    for a, b in combinations(trv, 2):
        edges_set.add((min(a, b), max(a, b)))
lines = [[nodes_wc[a], nodes_wc[b]] for a, b in edges_set]
lc2 = LineCollection(lines, colors='steelblue', linewidths=0.5, zorder=2)
ax.add_collection(lc2)

# Nodes
nw = [nodes_wc[i][0] for i in range(len(nodes_wc))]
nc = [nodes_wc[i][1] for i in range(len(nodes_wc))]
ax.scatter(nw, nc, c='black', s=12, zorder=5)

# Best
best_idx_f = min(f_at_node, key=f_at_node.get)
ax.scatter([nodes_wc[best_idx_f][0]], [nodes_wc[best_idx_f][1]],
           marker='*', c='red', s=250, zorder=7, label='best')

# Known optimum region (approximate x_w=120, x_c=80, x_b=300)
ax.scatter([120], [80], marker='D', c='gold', s=120, edgecolors='black',
           zorder=8, label='known opt ≈(120,80,300)')

ax.set_xlim(-20, TOTAL + 20); ax.set_ylim(-20, TOTAL + 20)
ax.set_xlabel("wheat (acres)"); ax.set_ylabel("corn (acres)")
ax.set_title(f"Final node distribution — {len(nodes_wc)} nodes")
ax.legend(fontsize=8); ax.set_aspect('equal'); ax.grid(alpha=0.15)

final_path = PLOT_DIR / "final_distribution.png"
fig.savefig(str(final_path), dpi=150, bbox_inches='tight')
plt.close(fig)
print(f"Final distribution plot saved: {final_path}")
