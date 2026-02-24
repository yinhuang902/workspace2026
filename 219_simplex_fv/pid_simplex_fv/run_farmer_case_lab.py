#!/usr/bin/env python
"""
run_farmer_case.py — Classic Farmer stochastic programming test for the simplex method.

Adapts the SNoGloDe farmer example (3 scenarios: good/fair/bad weather)
to the simplex method framework. Results go to results/farmer_{mode}/.

Usage:
    python run_farmer_case.py --mode smoke
    python run_farmer_case.py --mode full
"""
import argparse, csv, io, math, os, sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pyomo.environ as pyo

# ---------------------------------------------------------------------------
# Windows console encoding fix
# ---------------------------------------------------------------------------
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding=sys.stdout.encoding, errors="replace")
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding=sys.stderr.encoding, errors="replace")

# ---------------------------------------------------------------------------
# Problem parameters
# ---------------------------------------------------------------------------
SCENARIOS = {
    "good": {"yield": 1.2, "probability": 1 / 3},
    "fair": {"yield": 1.0, "probability": 1 / 3},
    "bad":  {"yield": 0.8, "probability": 1 / 3},
}

# Solver options (Gurobi) — LP problem, no NonConvex needed
UB_SOLVER_OPTS = {}   # LP — default Gurobi settings suffice
LB_SOLVER_OPTS = {
    "MIPGap": 1e-3,
    "TimeLimit": 30,
}

#                            SMOKE            FULL
# ─────────────────────────────────────────────────────
MODE_PARAMS = {
    "smoke": {
        "target_nodes":   50,
        "gap_stop_tol":   1e-4,
        "time_limit":     None,      # no wall-clock cap
        "enable_3d_plot": False,
        "enable_ef_ub":   True,
        "ef_time_ub":     30.0,
        "use_exact_opt":  False,
    },
    "full": {
        "target_nodes":   500,
        "gap_stop_tol":   1e-10,
        "time_limit":     3600.0,    # 1 hour
        "enable_3d_plot": False,
        "enable_ef_ub":   True,
        "ef_time_ub":     60.0,
        "use_exact_opt":  False,
    },
}

RANDOM_SEED = 42

# Known analytic solution (Birge & Louveaux, 1997)
# Expected cost = -108,390 (minimum of expected total cost)
KNOWN_OPTIMAL = -108390.0


# ---------------------------------------------------------------------------
# Model builder
# ---------------------------------------------------------------------------
def build_farmer_models():
    """Build scenario models and return (model_list, first_vars_list).

    Each model gets ``model.obj_expr`` patched (needed by BaseBundle / MSBundle).
    """
    from farmer_problem import TwoStageFarmer

    scenario_names = list(SCENARIOS.keys())
    model_list = []
    first_vars_list = []

    for sname in scenario_names:
        scen_info = SCENARIOS[sname]
        farmer = TwoStageFarmer(scen_info["yield"])
        m = farmer.model

        # --- Patch: extract objective expression into obj_expr ---
        # BaseBundle/MSBundle expect model.obj_expr as a raw Expression.
        # The farmer model has model.obj as a Pyomo Objective.
        obj_expression = m.obj.expr
        m.obj_expr = pyo.Expression(expr=obj_expression)

        # First-stage vars: acreage allocated to wheat, corn, beets
        # Same order as planting_crops: ["wheat", "corn", "beets"]
        fvars = [m.x["wheat"], m.x["corn"], m.x["beets"]]
        model_list.append(m)
        first_vars_list.append(fvars)

    return scenario_names, model_list, first_vars_list


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Farmer regression test runner (smoke / full)")
    parser.add_argument(
        "--mode", choices=["smoke", "full"], default="smoke",
        help="Run mode: 'smoke' (quick) or 'full' (exhaustive)")
    args = parser.parse_args()
    mode = args.mode
    params = MODE_PARAMS[mode]

    # Ensure working directory is the script's directory
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)
    sys.path.insert(0, str(script_dir))

    np.random.seed(RANDOM_SEED)

    # Results directory
    results_dir = script_dir / "results" / f"farmer_{mode}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Print header ──────────────────────────────────────────────────────
    print("=" * 70)
    print(f"  Farmer Regression Test — mode={mode.upper()}")
    print("=" * 70)
    print(f"  Results dir:     {results_dir}")
    print(f"  Random seed:     {RANDOM_SEED}")
    print(f"  Scenarios:       {list(SCENARIOS.keys())}")
    print(f"  target_nodes:    {params['target_nodes']}")
    print(f"  gap_stop_tol:    {params['gap_stop_tol']}")
    print(f"  time_limit:      {params['time_limit'] or 'None (unlimited)'}")
    print(f"  enable_ef_ub:    {params['enable_ef_ub']}")
    print(f"  ef_time_ub:      {params['ef_time_ub']}")
    print(f"  Known optimal:   {KNOWN_OPTIMAL}")
    print("=" * 70)

    # ── Build models ──────────────────────────────────────────────────────
    print("\n[1] Building farmer scenario models...")
    t0 = perf_counter()
    scenario_names, model_list, first_vars_list = build_farmer_models()
    S = len(model_list)
    dt_build = perf_counter() - t0
    print(f"    {S} scenarios built in {dt_build:.2f}s")

    # Print first-stage var info
    fv0 = first_vars_list[0]
    print(f"    First-stage dim: {len(fv0)}")
    for i, fv in enumerate(fv0):
        lb = fv.lb if fv.lb is not None else "-inf"
        ub = fv.ub if fv.ub is not None else "+inf"
        print(f"      var[{i}] = {fv.name}, bounds=[{lb}, {ub}]")

    # ── Build solver bundles ──────────────────────────────────────────────
    print("\n[2] Building solver bundles (BaseBundle + MSBundle)...")
    t0 = perf_counter()

    from bundles import BaseBundle, MSBundle

    base_bundles = [BaseBundle(m, UB_SOLVER_OPTS) for m in model_list]
    ms_bundles = [
        MSBundle(m, yvars, LB_SOLVER_OPTS, scenario_index=s)
        for s, (m, yvars) in enumerate(zip(model_list, first_vars_list))
    ]
    dt_bundles = perf_counter() - t0
    print(f"    Done in {dt_bundles:.2f}s")

    # ── Run simplex algorithm ─────────────────────────────────────────────
    print("\n[3] Running simplex algorithm...")
    print("-" * 70)
    t0 = perf_counter()

    from simplex_specialstart import run_pid_simplex_3d
    from utils import SimplexTracker

    tracker = SimplexTracker()

    # Feasible tetrahedron corners: {x >= 0, x1+x2+x3 <= 500}
    corner_nodes = [
        (0.0, 0.0, 0.0),
        (500.0, 0.0, 0.0),
        (0.0, 500.0, 0.0),
        (0.0, 0.0, 500.0),
    ]

    # ── Lab experiment: small tetrahedron near the known optimal ──
    # Known Birge & Louveaux optimal: x* = (170, 80, 250), EQ = -108,390
    # Note: x* is ON the feasibility boundary (170+80+250 = 500),
    # so it sits on the face {v1,v2,v3} of this tetrahedron.
    #
    #   v0 = (130, 50, 200)  interior point  (sum = 380)
    #   v1 = (220, 50, 230)  boundary point  (sum = 500, high wheat)
    #   v2 = (130, 120, 250) boundary point  (sum = 500, high corn)
    #   v3 = (130, 50, 320)  boundary point  (sum = 500, high beets)
    #
    # Verified: optimal (170,80,250) is inside face {v1,v2,v3}
    #   with barycentric coords μ₁=4/9, μ₂=3/7, μ₃=8/63 (all > 0)
    all_initial = [
        (130.0, 50.0, 200.0),   # v0: interior
        (220.0, 50.0, 230.0),   # v1: boundary, high wheat
        (130.0, 120.0, 250.0),  # v2: boundary, high corn
        (130.0, 50.0, 320.0),   # v3: boundary, high beets
    ]
    print(f"    Lab: small tetrahedron with {len(all_initial)} vertices near optimal (170, 80, 250)")
    print(f"    Vertex sums: {[sum(v) for v in all_initial]}")

    result = run_pid_simplex_3d(
        base_bundles=base_bundles,
        ms_bundles=ms_bundles,
        model_list=model_list,
        first_vars_list=first_vars_list,
        target_nodes=params["target_nodes"],
        verbose=True,
        gap_stop_tol=params["gap_stop_tol"],
        tracker=tracker,
        enable_3d_plot=True,
        plot_every=1,
        use_exact_opt=params["use_exact_opt"],
        time_limit=params["time_limit"],
        enable_ef_ub=params["enable_ef_ub"],
        ef_time_ub=params["ef_time_ub"],
        initial_nodes=all_initial,
        output_csv_path=str(results_dir / "simplex_result.csv"),
        split_mode=2,  # Mode 2: custom initial points + Mode 1 splitting
        plot_output_dir=str(results_dir / "plots"),
        axis_labels=("wheat (acres)", "corn (acres)", "beets (acres)"),
    )

    dt_run = perf_counter() - t0
    print("-" * 70)
    print(f"    Simplex run completed in {dt_run:.2f}s")

    # ── Extract final metrics ─────────────────────────────────────────────
    LB_hist = result["LB_hist"]
    UB_hist = result["UB_hist"]

    n_iters = len(LB_hist)
    final_LB = LB_hist[-1] / S if LB_hist else float("nan")
    final_UB = UB_hist[-1] / S if UB_hist else float("nan")
    final_gap_abs = final_UB - final_LB
    final_gap_rel = final_gap_abs / (abs(final_UB) + 1e-16)

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  RESULTS SUMMARY — mode={mode.upper()}")
    print("=" * 70)
    print(f"  Iterations:      {n_iters}")
    print(f"  Final nodes:     {result['node_count'][-1] if result['node_count'] else 'N/A'}")
    print(f"  Final LB (avg):  {final_LB:.6f}")
    print(f"  Final UB (avg):  {final_UB:.6f}")
    print(f"  Gap (abs):       {final_gap_abs:.6e}")
    print(f"  Gap (rel):       {final_gap_rel:.6e}")
    print(f"  Known optimal:   {KNOWN_OPTIMAL:.2f}")
    if math.isfinite(final_UB):
        opt_gap = abs(final_UB - KNOWN_OPTIMAL) / (abs(KNOWN_OPTIMAL) + 1e-16)
        print(f"  UB vs optimal:   {opt_gap:.6e}")
    print(f"  Wall time:       {dt_run:.2f}s")
    print(f"  Termination:     {result.get('termination_reason', 'unknown')}")
    print("=" * 70)

    # ── Write results text ────────────────────────────────────────────────
    txt_path = results_dir / "simplex_result.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write(f"mode={mode}\n")
        f.write(f"scenarios={list(SCENARIOS.keys())}\n")
        f.write(f"iterations={n_iters}\n")
        f.write(f"final_nodes={result['node_count'][-1] if result['node_count'] else 'N/A'}\n")
        f.write(f"final_LB_avg={final_LB}\n")
        f.write(f"final_UB_avg={final_UB}\n")
        f.write(f"gap_abs={final_gap_abs}\n")
        f.write(f"gap_rel={final_gap_rel}\n")
        f.write(f"known_optimal={KNOWN_OPTIMAL}\n")
        f.write(f"wall_time_s={dt_run:.2f}\n")

    csv_path = results_dir / "simplex_result.csv"
    print(f"\n  Results written to:")
    print(f"    {txt_path}")
    print(f"    {csv_path}")

    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        import traceback
        print("\n" + "=" * 70)
        print("FATAL ERROR — full traceback below:")
        print("=" * 70)
        traceback.print_exc()
        # Also write to crash_log.txt
        crash_path = Path(__file__).resolve().parent / "crash_log.txt"
        with open(crash_path, "w", encoding="utf-8") as f:
            traceback.print_exc(file=f)
        print(f"\nCrash log written to: {crash_path}")
        sys.exit(1)
