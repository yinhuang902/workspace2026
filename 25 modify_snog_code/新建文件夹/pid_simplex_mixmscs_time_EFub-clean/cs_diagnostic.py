"""
cs_diagnostic.py
================
Standalone diagnostic script to reproduce the Gurobi CS (constant-cut) bug.

Specifically:
  - Builds the same scenario models as app.ipynb
  - Constructs simplex T3 with the exact same vertices
  - Solves  min_{K in T3} Q_s(K)  for each scenario s using:
      (A) Gurobi "loose" (NonConvex=2, MIPGap=0.1) — same as the algorithm
      (B) Gurobi "tight" (NonConvex=2, MIPGap=1e-6) — reference solver
  - Also evaluates Q_s at the EF optimal point K_EF
  - Compares all results to identify which solver/scenario is wrong

Usage:
    python cs_diagnostic.py
"""

import pyomo.environ as pyo
import numpy as np
from time import perf_counter

from modeling import build_models_from_csv
from bundles import BaseBundle, MSBundle
from utils import evaluate_Q_at, tighten_bounds_one_model

# ==========================================================================
# 1. Model Construction (identical to app.ipynb)
# ==========================================================================
csv_path = "data.csv"
max_scenarios = 2

bounds = {
    "Kp": (-10.0, 10.0),
    "Ki": (-100.0, 100.0),
    "Kd": (-100.0, 100.0),
    "x": (-2.5, 2.5),
    "u": (-5.0, 5.0),
    "e": (None, None),
    "I": (None, None),
}
weights = (10.0, 0.01)
T = 15.0
nfe = 20

print("=" * 70)
print("CS DIAGNOSTIC: Reproducing Gurobi NonConvex=2 false-optimal bug")
print("=" * 70)

print("\n[1] Building scenario models...")
t0 = perf_counter()
model_list, first_stg_vars_list, m_tmpl_list, nfe = build_models_from_csv(
    csv_path=csv_path, T=T, nfe=nfe, weights=weights, bounds=bounds,
    sp0=0.0, sp1=0.5,
    tau_xs_col="tau_xs", tau_us_col="tau_us", tau_ds_col="tau_ds",
    disturb_prefix="disturbance_", setpoint_change_col="setpoint_change",
    max_scenarios=max_scenarios, skip=0,
)
print(f"    Done in {perf_counter()-t0:.3f}s. {len(model_list)} scenarios.")

# ==========================================================================
# 2. Simplex T3 definition
# ==========================================================================
T3_vertices = [
    (-10.0, 100.0, 100.0),
    (10.0, 100.0, -100.0),
    (10.0, 100.0, 100.0),
    (10.0, -100.0, -100.0),
]

# K_EF from IPOPT EF solve (the UB point that triggered the bug):
K_EF = (9.99993804110578, 99.9998865292316, -3.939378908186571e-06)

print(f"\n[2] Simplex T3 vertices:")
for i, v in enumerate(T3_vertices):
    print(f"    v{i}: {v}")
print(f"    K_EF (IPOPT EF optimal): {K_EF}")

# ==========================================================================
# 3. Build solvers
# ==========================================================================
print("\n[3] Building solvers...")

# --- Gurobi BaseBundle (for independent Q evaluation, no MIPGap) ---
ub_options = {'NonConvex': 2}
base_bundles = [BaseBundle(m, ub_options) for m in model_list]

# --- Gurobi MSBundle LOOSE (same as the algorithm: MIPGap=0.1) ---
loose_options = {
    'NonConvex': 2,
    'MIPGap': 1e-1,
    'TimeLimit': 15,
}
ms_bundles_loose = [MSBundle(m, yvars, loose_options)
                    for m, yvars in zip(model_list, first_stg_vars_list)]

# --- Gurobi MSBundle TIGHT (reference: MIPGap=1e-6) ---
tight_options = {
    'NonConvex': 2,
    'MIPGap': 1e-6,
    'TimeLimit': 120,   # Allow more time for tight solve
}
ms_bundles_tight = [MSBundle(m, yvars, tight_options)
                    for m, yvars in zip(model_list, first_stg_vars_list)]

print("    Gurobi solvers ready (loose + tight).")

# ==========================================================================
# 4. Evaluate Q_s at K_EF (independent, per-scenario)
# ==========================================================================
print("\n" + "=" * 70)
print("[4] Independent Q_s evaluation at K_EF")
print("=" * 70)

q_ef_per_scen = []
for s in range(len(model_list)):
    q_val = evaluate_Q_at(base_bundles[s], first_stg_vars_list[s], K_EF)
    q_ef_per_scen.append(q_val)
    print(f"    scen {s}: Q_s(K_EF) = {q_val:.12f}")

print(f"    SUM = {sum(q_ef_per_scen):.12f}")
print(f"    AVG = {sum(q_ef_per_scen)/len(q_ef_per_scen):.12f}")

# ==========================================================================
# 5. Vertex Q evaluation (needed by update_tetra)
# ==========================================================================
vertex_Q = np.zeros((len(model_list), 4))
for s in range(len(model_list)):
    for j, v in enumerate(T3_vertices):
        vertex_Q[s, j] = evaluate_Q_at(base_bundles[s], first_stg_vars_list[s], v)
    print(f"    scen {s} vertex Q values: {vertex_Q[s]}")

# ==========================================================================
# 6. Gurobi CS solve LOOSE: min Q_s(K) s.t. K in T3 (MIPGap=0.1)
# ==========================================================================
print("\n" + "=" * 70)
print("[5] Gurobi CS solve LOOSE (NonConvex=2, MIPGap=0.1)")
print("=" * 70)

gurobi_loose = []
for s in range(len(model_list)):
    msb = ms_bundles_loose[s]
    msb.update_tetra(T3_vertices, vertex_Q[s])

    t0 = perf_counter()
    ok, c_val, cand_pt = msb.solve_const_cut()
    dt = perf_counter() - t0

    meta = msb.last_cs_meta
    dual_bound = meta.get("dual_bound", None)
    primal_obj = meta.get("primal_obj", None)
    status = meta.get("status", "?")

    result = {
        "scen": s, "ok": ok, "status": status,
        "dual_bound": dual_bound, "primal_obj": primal_obj,
        "c_val": c_val, "cand_pt": cand_pt, "time": dt,
    }
    gurobi_loose.append(result)

    print(f"\n  scen {s}:")
    print(f"    status     = {status}")
    print(f"    dual_bound = {dual_bound}")
    print(f"    primal_obj = {primal_obj}")
    print(f"    c_val      = {c_val}")
    print(f"    K_gurobi   = {cand_pt}")
    print(f"    time       = {dt:.3f}s")

    if cand_pt is not None:
        q_indep = evaluate_Q_at(base_bundles[s], first_stg_vars_list[s], cand_pt)
        print(f"    indep_Q(K_gurobi) = {q_indep:.12f}")
        result["indep_Q"] = q_indep

# ==========================================================================
# 7. Gurobi CS solve TIGHT: min Q_s(K) s.t. K in T3 (MIPGap=1e-6)
# ==========================================================================
print("\n" + "=" * 70)
print("[6] Gurobi CS solve TIGHT (NonConvex=2, MIPGap=1e-6)")
print("=" * 70)

gurobi_tight = []
for s in range(len(model_list)):
    msb = ms_bundles_tight[s]
    msb.update_tetra(T3_vertices, vertex_Q[s])

    t0 = perf_counter()
    ok, c_val, cand_pt = msb.solve_const_cut()
    dt = perf_counter() - t0

    meta = msb.last_cs_meta
    dual_bound = meta.get("dual_bound", None)
    primal_obj = meta.get("primal_obj", None)
    status = meta.get("status", "?")

    result = {
        "scen": s, "ok": ok, "status": status,
        "dual_bound": dual_bound, "primal_obj": primal_obj,
        "c_val": c_val, "cand_pt": cand_pt, "time": dt,
    }
    gurobi_tight.append(result)

    print(f"\n  scen {s}:")
    print(f"    status     = {status}")
    print(f"    dual_bound = {dual_bound}")
    print(f"    primal_obj = {primal_obj}")
    print(f"    c_val      = {c_val}")
    print(f"    K_gurobi   = {cand_pt}")
    print(f"    time       = {dt:.3f}s")

    if cand_pt is not None:
        q_indep = evaluate_Q_at(base_bundles[s], first_stg_vars_list[s], cand_pt)
        print(f"    indep_Q(K_tight) = {q_indep:.12f}")
        result["indep_Q"] = q_indep

# ==========================================================================
# 8. Comparison Summary
# ==========================================================================
print("\n" + "=" * 70)
print("[7] COMPARISON SUMMARY")
print("=" * 70)

header = (f"{'':4s}  {'Gurobi LOOSE':>14s}  {'K_loose':>38s}  "
          f"{'Gurobi TIGHT':>14s}  {'K_tight':>38s}  {'Q(K_EF)':>14s}")
print(f"\n{header}")
print("-" * 135)

sum_loose = 0.0
sum_tight = 0.0
sum_q_ef = 0.0

for s in range(len(model_list)):
    gl = gurobi_loose[s]
    gt = gurobi_tight[s]
    qef = q_ef_per_scen[s]

    l_cs = gl["c_val"] if gl["c_val"] is not None else float('nan')
    l_K = gl["cand_pt"]
    t_cs = gt["c_val"] if gt["c_val"] is not None else float('nan')
    t_K = gt["cand_pt"]

    l_K_str = f"({l_K[0]:.4f}, {l_K[1]:.4f}, {l_K[2]:.4f})" if l_K else "N/A"
    t_K_str = f"({t_K[0]:.4f}, {t_K[1]:.4f}, {t_K[2]:.4f})" if t_K else "N/A"

    flags = []
    if qef < l_cs - 1e-8:
        flags.append("⚠ Q(K_EF)<LOOSE")
    if t_cs < l_cs - 1e-8:
        flags.append("⚠ TIGHT<LOOSE")
    flag_str = "  " + ", ".join(flags) if flags else ""

    print(f"  s={s}  {l_cs:14.9f}  {l_K_str:>38s}  {t_cs:14.9f}  {t_K_str:>38s}  {qef:14.9f}{flag_str}")

    sum_loose += l_cs
    sum_tight += t_cs
    sum_q_ef += qef

print("-" * 135)
print(f"  SUM  {sum_loose:14.9f}  {'':38s}  {sum_tight:14.9f}  {'':38s}  {sum_q_ef:14.9f}")
print(f"  AVG  {sum_loose/len(model_list):14.9f}  {'':38s}  {sum_tight/len(model_list):14.9f}  {'':38s}  {sum_q_ef/len(model_list):14.9f}")

# ==========================================================================
# 9. Detailed Verdict
# ==========================================================================
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
for s in range(len(model_list)):
    gl = gurobi_loose[s]
    gt = gurobi_tight[s]
    qef = q_ef_per_scen[s]
    l_cs = gl["c_val"] if gl["c_val"] is not None else float('nan')
    t_cs = gt["c_val"] if gt["c_val"] is not None else float('nan')

    print(f"\n  Scenario {s}:")
    print(f"    Gurobi LOOSE c_s = {l_cs:.9f}  (MIPGap=0.1)")
    print(f"    Gurobi TIGHT c_s = {t_cs:.9f}  (MIPGap=1e-6)")
    print(f"    Q_s(K_EF)        = {qef:.9f}")

    if qef < l_cs - 1e-6:
        print(f"    → ⚠ LOOSE DUAL IS WRONG: Q_s(K_EF)={qef:.9f} < c_s_loose={l_cs:.9f}")
        print(f"       K_EF is inside T3 but has a LOWER Q than Gurobi's 'global optimum'")
        print(f"       Gurobi NonConvex=2 with MIPGap=0.1 returned a FALSE optimal")
        if t_cs <= qef + 1e-6:
            print(f"       Tight MIPGap (1e-6) FIXES the issue: c_s_tight={t_cs:.9f} ≤ Q(K_EF)={qef:.9f}")
        else:
            print(f"       ⚠ Even tight MIPGap doesn't fix it: c_s_tight={t_cs:.9f} > Q(K_EF)={qef:.9f}")
    elif t_cs < l_cs - 1e-6:
        print(f"    → ⚠ LOOSE found suboptimal: tight={t_cs:.9f} < loose={l_cs:.9f}")
    else:
        print(f"    → ✓ Results consistent")

print("\nDone.")
