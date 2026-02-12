"""
cs_diagnostic.py
================
Standalone diagnostic script to reproduce the Gurobi CS (constant-cut) bug.

Solves  min_{K in T3} Q_s(K)  for each scenario s using:
  (A) Gurobi (NonConvex=2, MIPGap=0.1)  — same as the algorithm
  (B) IPOPT (default settings)           — same as the EF solver
Also evaluates Q_s at the EF optimal point K_EF.

Usage:
    python cs_diagnostic.py
"""

import pyomo.environ as pyo
import numpy as np
from time import perf_counter
import shutil

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
S = len(model_list)
print(f"    Done in {perf_counter()-t0:.3f}s. {S} scenarios.")

# ==========================================================================
# 2. Simplex T3 definition
# ==========================================================================
T3_vertices = [
    (-10.0, 100.0, 100.0),
    (10.0, 100.0, -100.0),
    (10.0, 100.0, 100.0),
    (10.0, -100.0, -100.0),
]

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

# --- Gurobi MSBundle (for CS solve: NonConvex=2, MIPGap=0.1) ---
lb_options = {
    'NonConvex': 2,
    'MIPGap': 1e-1,
    'TimeLimit': 15,
}
ms_bundles = [MSBundle(m, yvars, lb_options)
              for m, yvars in zip(model_list, first_stg_vars_list)]

# --- Check IPOPT availability ---
ipopt_available = False
for ipopt_name in ["ipopt", "cyipopt"]:
    try:
        _test = pyo.SolverFactory(ipopt_name)
        if _test.available():
            ipopt_available = True
            ipopt_solver_name = ipopt_name
            break
    except Exception:
        continue

if not ipopt_available:
    # Try finding ipopt in conda env
    ipopt_path = shutil.which("ipopt")
    if ipopt_path:
        ipopt_available = True
        ipopt_solver_name = "ipopt"

print(f"    Gurobi solvers ready.")
print(f"    IPOPT available: {ipopt_available}" +
      (f" (using '{ipopt_solver_name}')" if ipopt_available else
       " — will use Gurobi tight (MIPGap=1e-6) as fallback"))

# ==========================================================================
# 4. Evaluate Q_s at K_EF (independent, per-scenario)
# ==========================================================================
print("\n" + "=" * 70)
print("[4] Independent Q_s evaluation at K_EF")
print("=" * 70)

q_ef_per_scen = []
for s in range(S):
    q_val = evaluate_Q_at(base_bundles[s], first_stg_vars_list[s], K_EF)
    q_ef_per_scen.append(q_val)
    print(f"    scen {s}: Q_s(K_EF) = {q_val:.12f}")

print(f"    SUM = {sum(q_ef_per_scen):.12f}")
print(f"    AVG = {sum(q_ef_per_scen)/S:.12f}")

# ==========================================================================
# 5. Vertex Q evaluation (needed by update_tetra)
# ==========================================================================
vertex_Q = np.zeros((S, 4))
for s in range(S):
    for j, v in enumerate(T3_vertices):
        vertex_Q[s, j] = evaluate_Q_at(base_bundles[s], first_stg_vars_list[s], v)
    print(f"    scen {s} vertex Q values: {vertex_Q[s]}")

# ==========================================================================
# 6. Gurobi CS solve: min Q_s(K) s.t. K in T3 (MIPGap=0.1)
# ==========================================================================
print("\n" + "=" * 70)
print("[5] Gurobi CS solve (NonConvex=2, MIPGap=0.1)")
print("=" * 70)

gurobi_results = []
for s in range(S):
    msb = ms_bundles[s]
    msb.update_tetra(T3_vertices, vertex_Q[s])

    t0 = perf_counter()
    ok, c_val, cand_pt = msb.solve_const_cut()
    dt = perf_counter() - t0

    meta = msb.last_cs_meta
    result = {
        "scen": s, "ok": ok, "status": meta.get("status", "?"),
        "dual_bound": meta.get("dual_bound"), "primal_obj": meta.get("primal_obj"),
        "c_val": c_val, "cand_pt": cand_pt, "time": dt,
    }
    gurobi_results.append(result)

    print(f"\n  scen {s}:")
    print(f"    status     = {result['status']}")
    print(f"    dual_bound = {result['dual_bound']}")
    print(f"    primal_obj = {result['primal_obj']}")
    print(f"    c_val      = {c_val}")
    print(f"    K_gurobi   = {cand_pt}")
    print(f"    time       = {dt:.3f}s")

    if cand_pt is not None:
        q_indep = evaluate_Q_at(base_bundles[s], first_stg_vars_list[s], cand_pt)
        print(f"    indep_Q(K_gurobi) = {q_indep:.12f}")
        result["indep_Q"] = q_indep

# ==========================================================================
# 7. Reference solve: IPOPT (default) or Gurobi tight (MIPGap=1e-6)
# ==========================================================================
def solve_per_scenario_in_simplex(s, solver_name, solver_opts, warm_start_K):
    """Solve min Q_s(K) s.t. K in T3 using the given solver."""
    m_clone = model_list[s].clone()
    Kp = m_clone.find_component(first_stg_vars_list[s][0].name)
    Ki = m_clone.find_component(first_stg_vars_list[s][1].name)
    Kd = m_clone.find_component(first_stg_vars_list[s][2].name)

    # Barycentric simplex constraints
    m_clone.lam_idx = pyo.RangeSet(0, 3)
    m_clone.lam = pyo.Var(m_clone.lam_idx, domain=pyo.NonNegativeReals, bounds=(0, 1))
    m_clone.lam_sum = pyo.Constraint(
        expr=sum(m_clone.lam[j] for j in m_clone.lam_idx) == 1.0)

    V = T3_vertices
    m_clone.link_kp = pyo.Constraint(
        expr=Kp == sum(m_clone.lam[j] * V[j][0] for j in range(4)))
    m_clone.link_ki = pyo.Constraint(
        expr=Ki == sum(m_clone.lam[j] * V[j][1] for j in range(4)))
    m_clone.link_kd = pyo.Constraint(
        expr=Kd == sum(m_clone.lam[j] * V[j][2] for j in range(4)))

    if hasattr(m_clone, 'obj'):
        m_clone.del_component('obj')
    m_clone.obj_min = pyo.Objective(expr=m_clone.obj_expr, sense=pyo.minimize)

    # Warm start
    for j in range(4):
        m_clone.lam[j].value = 0.25
    Kp.value = warm_start_K[0]
    Ki.value = warm_start_K[1]
    Kd.value = warm_start_K[2]

    solver = pyo.SolverFactory(solver_name)
    if solver_opts:
        for k, v in solver_opts.items():
            solver.options[k] = v

    t0 = perf_counter()
    res = solver.solve(m_clone, tee=False)
    dt = perf_counter() - t0

    term = str(res.solver.termination_condition)
    status = str(res.solver.status)

    kp_val = float(pyo.value(Kp))
    ki_val = float(pyo.value(Ki))
    kd_val = float(pyo.value(Kd))
    obj_val = float(pyo.value(m_clone.obj_expr))

    return {
        "scen": s, "status": status, "term": term,
        "obj": obj_val, "K": (kp_val, ki_val, kd_val), "time": dt,
    }


if ipopt_available:
    ref_solver_name = ipopt_solver_name
    ref_solver_opts = {}  # IPOPT default settings (same as EF solver)
    ref_label = f"IPOPT (default, solver='{ipopt_solver_name}')"
else:
    ref_solver_name = "gurobi"
    ref_solver_opts = {'NonConvex': 2, 'MIPGap': 1e-6, 'TimeLimit': 120}
    ref_label = "Gurobi TIGHT (MIPGap=1e-6, fallback since IPOPT unavailable)"

print("\n" + "=" * 70)
print(f"[6] Reference solve: {ref_label}")
print("=" * 70)

ref_results = []
for s in range(S):
    # Try with K_EF warm start
    result = solve_per_scenario_in_simplex(s, ref_solver_name, ref_solver_opts, K_EF)
    ref_results.append(result)

    print(f"\n  scen {s} (warm start = K_EF):")
    print(f"    status = {result['status']}, term = {result['term']}")
    print(f"    obj    = {result['obj']:.12f}")
    K = result['K']
    print(f"    K      = ({K[0]:.6f}, {K[1]:.6f}, {K[2]:.6f})")
    print(f"    time   = {result['time']:.3f}s")

    q_indep = evaluate_Q_at(base_bundles[s], first_stg_vars_list[s], K)
    print(f"    indep_Q = {q_indep:.12f}")
    result["indep_Q"] = q_indep

    # Also try centroid warm start
    centroid = tuple(np.mean(T3_vertices, axis=0))
    result2 = solve_per_scenario_in_simplex(s, ref_solver_name, ref_solver_opts, centroid)
    K2 = result2['K']
    q2 = evaluate_Q_at(base_bundles[s], first_stg_vars_list[s], K2)
    print(f"  scen {s} (warm start = centroid):")
    print(f"    obj={result2['obj']:.12f}, K=({K2[0]:.4f},{K2[1]:.4f},{K2[2]:.4f}), indep_Q={q2:.12f}")

    # Keep the best result
    if result2["obj"] < result["obj"]:
        result2["indep_Q"] = q2
        ref_results[s] = result2
        print(f"    → centroid start found BETTER solution!")

# ==========================================================================
# 8. Comparison Summary
# ==========================================================================
print("\n" + "=" * 70)
print("[7] COMPARISON SUMMARY")
print("=" * 70)

ref_short = "IPOPT" if ipopt_available else "Grb-tight"
header = (f"{'':4s}  {'Gurobi c_s':>14s}  {'K_gurobi':>38s}  "
          f"{ref_short+' obj':>14s}  {ref_short+' K':>38s}  {'Q(K_EF)':>14s}")
print(f"\n{header}")
print("-" * 135)

sum_gurobi = 0.0
sum_ref = 0.0
sum_q_ef = 0.0

for s in range(S):
    gr = gurobi_results[s]
    rr = ref_results[s]
    qef = q_ef_per_scen[s]

    g_cs = gr["c_val"] if gr["c_val"] is not None else float('nan')
    g_K = gr["cand_pt"]
    r_obj = rr["obj"]
    r_K = rr["K"]

    g_K_str = f"({g_K[0]:.4f}, {g_K[1]:.4f}, {g_K[2]:.4f})" if g_K else "N/A"
    r_K_str = f"({r_K[0]:.4f}, {r_K[1]:.4f}, {r_K[2]:.4f})"

    flags = []
    if qef < g_cs - 1e-8:
        flags.append("⚠Q(K_EF)<Grb")
    if r_obj < g_cs - 1e-8:
        flags.append(f"⚠{ref_short}<Grb")
    flag_str = "  " + ", ".join(flags) if flags else ""

    print(f"  s={s}  {g_cs:14.9f}  {g_K_str:>38s}  {r_obj:14.9f}  {r_K_str:>38s}  {qef:14.9f}{flag_str}")

    sum_gurobi += g_cs
    sum_ref += r_obj
    sum_q_ef += qef

print("-" * 135)
print(f"  SUM  {sum_gurobi:14.9f}  {'':38s}  {sum_ref:14.9f}  {'':38s}  {sum_q_ef:14.9f}")
print(f"  AVG  {sum_gurobi/S:14.9f}  {'':38s}  {sum_ref/S:14.9f}  {'':38s}  {sum_q_ef/S:14.9f}")

# ==========================================================================
# 9. Verdict
# ==========================================================================
print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
for s in range(S):
    gr = gurobi_results[s]
    rr = ref_results[s]
    qef = q_ef_per_scen[s]
    g_cs = gr["c_val"] if gr["c_val"] is not None else float('nan')
    r_obj = rr["obj"]

    print(f"\n  Scenario {s}:")
    print(f"    Gurobi CS c_s    = {g_cs:.9f}  (NonConvex=2, MIPGap=0.1)")
    print(f"    {ref_short} min in T3 = {r_obj:.9f}")
    print(f"    Q_s(K_EF)        = {qef:.9f}")

    if qef < g_cs - 1e-6:
        print(f"    → ⚠ GUROBI CS IS WRONG: Q_s(K_EF)={qef:.9f} < c_s={g_cs:.9f}")
        print(f"       K_EF is inside T3 but has LOWER Q than Gurobi's 'optimal'")
    elif r_obj < g_cs - 1e-6:
        print(f"    → ⚠ GUROBI CS IS WRONG: {ref_short} found {r_obj:.9f} < c_s={g_cs:.9f}")
    else:
        print(f"    → ✓ Consistent")

print("\nDone.")
