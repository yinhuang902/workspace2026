#!/usr/bin/env python
"""
run_newsvendor_case.py — 3-item (3D) newsvendor LP recourse test for the simplex method.

A deliberately simple LP recourse problem whose value function has
clean, controllable kinks at the demand breakpoints.  Designed as a
unit-test/sanity-check for the simplex refinement algorithm.

Usage:
    python run_newsvendor_case.py --mode smoke
    python run_newsvendor_case.py --mode full
    python run_newsvendor_case.py --mode smoke --solve_ef
"""
import argparse, csv, io, math, os, sys
from pathlib import Path
from time import perf_counter
from itertools import product

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

# ==========================================================================
#  Problem parameters  (edit these to change the instance)
# ==========================================================================

# Budget constraint: x1 + x2 + x3 <= B,  x_i >= 0
B = 10.0

# Scenario probabilities (EQUAL weights — dividing LB/UB by S is correct).
# If you change these to unequal weights, replace the `/S` aggregation in the
# results section with a proper weighted sum: sum_s p_s * Q_s.
PROBS = [1 / 3, 1 / 3, 1 / 3]

# Shortage (backorder) and holding costs — same for all items.
C_SHORT = [10.0, 10.0, 10.0]
C_HOLD  = [ 1.0,  1.0,  1.0]

# Scenario demand vectors  d^(s) in R^3_+
DEMANDS = [
    [2.0, 3.0, 5.0],   # scenario 1
    [4.0, 4.0, 2.0],   # scenario 2
    [6.0, 1.0, 3.0],   # scenario 3
]

# Scenario names (cosmetic)
SCENARIO_NAMES = ["s1", "s2", "s3"]

# ---------------------------------------------------------------------------
# Solver options  (LP — no MIP settings needed)
# ---------------------------------------------------------------------------
UB_SOLVER_OPTS = {}
LB_SOLVER_OPTS = {
    "MIPGap": 1e-1,
    "TimeLimit": 30,
}

# ---------------------------------------------------------------------------
# Mode presets  (same style as run_farmer_case.py)
# ---------------------------------------------------------------------------
MODE_PARAMS = {
    "smoke": {
        "target_nodes":   80,
        "gap_stop_tol":   1e-2,
        "time_limit":     None,
        "enable_3d_plot": False,
        "enable_ef_ub":   True,
        "ef_time_ub":     30.0,
        "use_exact_opt":  False,
    },
    "full": {
        "target_nodes":   500,
        "gap_stop_tol":   1e-10,
        "time_limit":     3600.0,
        "enable_3d_plot": False,
        "enable_ef_ub":   True,
        "ef_time_ub":     60.0,
        "use_exact_opt":  False,
    },
}

RANDOM_SEED = 42

# Analytic optimal (computed by hand / small LP; overwrite once you verify).
KNOWN_OPTIMAL = None  # set after first successful EF solve if desired


# ==========================================================================
#  Model builder  (one Pyomo model per scenario, mirroring TwoStageFarmer)
# ==========================================================================
ITEMS = [1, 2, 3]   # item indices


def build_newsvendor_models():
    """Build one Pyomo ``ConcreteModel`` per scenario.

    Each model contains:
      * First-stage vars  x[i] in [0, B]   with budget constraint sum(x) <= B.
      * Second-stage vars u[i] >= 0 (shortage), v[i] >= 0 (holding).
      * Linking:  x[i] + u[i] - v[i] == d_s[i].
      * Objective (minimise): sum_i c_short[i]*u[i] + c_hold[i]*v[i].
      * model.obj_expr patch required by BaseBundle / MSBundle.

    Non-anticipativity of x is enforced externally by the simplex
    algorithm, which evaluates every scenario model at the *same* x
    point (via ``eval_at``).

    Returns
    -------
    scenario_names : list[str]
    model_list     : list[pyo.ConcreteModel]
    first_vars_list: list[list[pyo.Var]]
    """
    model_list = []
    first_vars_list = []

    for s, sname in enumerate(SCENARIO_NAMES):
        d_s = DEMANDS[s]

        m = pyo.ConcreteModel(name=f"newsvendor_{sname}")

        # --- First-stage variables: order quantities ------------------
        m.x = pyo.Var(ITEMS, within=pyo.NonNegativeReals, bounds=(0, B))

        # --- Budget / simplex constraint (critical): ------------------
        @m.Constraint()
        def budget(m):
            return sum(m.x[i] for i in ITEMS) <= B

        # --- Second-stage variables -----------------------------------
        m.u = pyo.Var(ITEMS, within=pyo.NonNegativeReals)  # shortage
        m.v = pyo.Var(ITEMS, within=pyo.NonNegativeReals)  # holding

        # --- Linking constraints: x_i + u_i - v_i == d_i^(s) ---------
        @m.Constraint(ITEMS)
        def demand_balance(m, i):
            return m.x[i] + m.u[i] - m.v[i] == d_s[i - 1]

        # --- Second-stage objective -----------------------------------
        obj_expression = sum(
            C_SHORT[i - 1] * m.u[i] + C_HOLD[i - 1] * m.v[i]
            for i in ITEMS
        )
        m.obj = pyo.Objective(expr=obj_expression, sense=pyo.minimize)

        # --- Patch obj_expr for BaseBundle / MSBundle -----------------
        m.obj_expr = pyo.Expression(expr=obj_expression)

        # First-stage variable list (ordered consistently across scenarios)
        fvars = [m.x[1], m.x[2], m.x[3]]

        model_list.append(m)
        first_vars_list.append(fvars)

    return list(SCENARIO_NAMES), model_list, first_vars_list


# ==========================================================================
#  Optional: Extensive-form (EF) reference solve
# ==========================================================================
def solve_ef_newsvendor(verbose=True):
    """Solve the deterministic equivalent (extensive form) as a single LP.

    x is SHARED across all scenarios (non-anticipativity), while u/v are
    scenario-specific.  Objective: sum_s p_s * Q_s(x).

    Returns the optimal objective value or None on failure.
    """
    ef = pyo.ConcreteModel(name="newsvendor_EF")

    S = len(DEMANDS)
    SCENS = list(range(S))

    # Shared first-stage: x[i]
    ef.x = pyo.Var(ITEMS, within=pyo.NonNegativeReals, bounds=(0, B))

    @ef.Constraint()
    def budget(ef):
        return sum(ef.x[i] for i in ITEMS) <= B

    # Per-scenario second-stage: u[s,i], v[s,i]
    ef.u = pyo.Var(SCENS, ITEMS, within=pyo.NonNegativeReals)
    ef.v = pyo.Var(SCENS, ITEMS, within=pyo.NonNegativeReals)

    @ef.Constraint(SCENS, ITEMS)
    def demand_balance(ef, s, i):
        return ef.x[i] + ef.u[s, i] - ef.v[s, i] == DEMANDS[s][i - 1]

    # Weighted objective: sum_s p_s * Q_s(x)
    ef.obj = pyo.Objective(
        expr=sum(
            PROBS[s] * (
                sum(C_SHORT[i - 1] * ef.u[s, i] + C_HOLD[i - 1] * ef.v[s, i]
                    for i in ITEMS)
            )
            for s in SCENS
        ),
        sense=pyo.minimize,
    )

    # Try Gurobi first, then IPOPT, then GLPK
    for solver_name in ("gurobi", "ipopt", "glpk"):
        solver = pyo.SolverFactory(solver_name)
        if solver.available():
            if verbose:
                print(f"  [EF] Solving with {solver_name} ...")
            result = solver.solve(ef, tee=False)
            if (result.solver.status == pyo.SolverStatus.ok and
                    result.solver.termination_condition in (
                        pyo.TerminationCondition.optimal,
                        pyo.TerminationCondition.locallyOptimal)):
                obj_val = pyo.value(ef.obj)
                x_star = tuple(pyo.value(ef.x[i]) for i in ITEMS)
                if verbose:
                    print(f"  [EF] Optimal objective (E[Q]): {obj_val:.6f}")
                    print(f"  [EF] Optimal x*: {x_star}")
                return obj_val
            else:
                if verbose:
                    print(f"  [EF] Solver {solver_name} returned status "
                          f"{result.solver.termination_condition}")
    if verbose:
        print("  [EF] WARNING: no solver produced an optimal solution.")
    return None


# ==========================================================================
#  Initial nodes construction
# ==========================================================================
def build_initial_nodes():
    """Build initial node set: 4 budget-simplex corners + demand-kink grid.

    Kink breakpoints per dimension come from the demand values across
    scenarios:
        dim 1 (x1): {d_1^(s)} = {2, 4, 6}
        dim 2 (x2): {d_2^(s)} = {3, 4, 1} → {1, 3, 4}
        dim 3 (x3): {d_3^(s)} = {5, 2, 3} → {2, 3, 5}

    All (x1, x2, x3) combinations from the kink sets satisfying
    x1+x2+x3 <= B and x_i >= 0 are included, then deduplicated and
    merged with the 4 simplex corners.
    """
    # Budget-simplex corners
    corners = {
        (0.0, 0.0, 0.0),
        (B,   0.0, 0.0),
        (0.0, B,   0.0),
        (0.0, 0.0, B),
    }

    # Kink breakpoints from demand vectors
    kink1 = sorted({d[0] for d in DEMANDS})   # {2, 4, 6}
    kink2 = sorted({d[1] for d in DEMANDS})   # {1, 3, 4}
    kink3 = sorted({d[2] for d in DEMANDS})   # {2, 3, 5}

    kink_pts = set()
    for x1, x2, x3 in product(kink1, kink2, kink3):
        if x1 >= 0 and x2 >= 0 and x3 >= 0 and x1 + x2 + x3 <= B + 1e-8:
            kink_pts.add((round(x1, 6), round(x2, 6), round(x3, 6)))

    all_initial = sorted(corners | kink_pts)
    return all_initial, corners, kink_pts, (kink1, kink2, kink3)


# ==========================================================================
#  Main
# ==========================================================================
def main():
    parser = argparse.ArgumentParser(
        description="Newsvendor LP recourse test runner (smoke / full)")
    parser.add_argument(
        "--mode", choices=["smoke", "full"], default="smoke",
        help="Run mode: 'smoke' (quick) or 'full' (exhaustive)")
    parser.add_argument(
        "--solve_ef", action="store_true", default=False,
        help="Solve extensive form once for a reference optimal value")
    args = parser.parse_args()
    mode = args.mode
    params = MODE_PARAMS[mode]

    # Ensure working directory is the script's directory
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)
    sys.path.insert(0, str(script_dir))

    np.random.seed(RANDOM_SEED)

    # Results directory
    results_dir = script_dir / "results" / f"newsvendor_{mode}"
    results_dir.mkdir(parents=True, exist_ok=True)

    # ── Print header ──────────────────────────────────────────────────────
    print("=" * 70)
    print(f"  Newsvendor Regression Test — mode={mode.upper()}")
    print("=" * 70)
    print(f"  Results dir:     {results_dir}")
    print(f"  Random seed:     {RANDOM_SEED}")
    print(f"  Budget B:        {B}")
    print(f"  Scenarios:       {SCENARIO_NAMES}")
    print(f"  Probabilities:   {PROBS}  (equal → /S aggregation)")
    print(f"  c_short:         {C_SHORT}")
    print(f"  c_hold:          {C_HOLD}")
    for idx, d in enumerate(DEMANDS):
        print(f"  demand[{idx}]:       {d}")
    print(f"  target_nodes:    {params['target_nodes']}")
    print(f"  gap_stop_tol:    {params['gap_stop_tol']}")
    print(f"  time_limit:      {params['time_limit'] or 'None (unlimited)'}")
    print(f"  enable_ef_ub:    {params['enable_ef_ub']}")
    print(f"  ef_time_ub:      {params['ef_time_ub']}")
    print(f"  Known optimal:   {KNOWN_OPTIMAL}")
    print("=" * 70)

    # ── Optional EF solve ─────────────────────────────────────────────────
    ef_obj = None
    if args.solve_ef:
        print("\n[0] Solving extensive form (EF) for reference optimal...")
        ef_obj = solve_ef_newsvendor(verbose=True)
        if ef_obj is not None:
            print(f"    EF optimal E[Q(x*)] = {ef_obj:.6f}")
        else:
            print("    EF solve failed — continuing without reference value.")

    # ── Build models ──────────────────────────────────────────────────────
    print("\n[1] Building newsvendor scenario models...")
    t0 = perf_counter()
    scenario_names, model_list, first_vars_list = build_newsvendor_models()
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

    # ── Build initial nodes ───────────────────────────────────────────────
    all_initial, corners, kink_pts, (kink1, kink2, kink3) = build_initial_nodes()
    print(f"\n    Initial nodes: {len(corners)} corners + "
          f"{len(kink_pts)} kink points = {len(all_initial)} total")
    print(f"    Kink planes — dim1: {kink1}, dim2: {kink2}, dim3: {kink3}")

    # ── Run simplex algorithm ─────────────────────────────────────────────
    print("\n[3] Running simplex algorithm...")
    print("-" * 70)
    t0 = perf_counter()

    from simplex_specialstart import run_pid_simplex_3d
    from utils import SimplexTracker

    tracker = SimplexTracker()

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
        plot_every=5,
        use_exact_opt=params["use_exact_opt"],
        time_limit=params["time_limit"],
        enable_ef_ub=params["enable_ef_ub"],
        ef_time_ub=params["ef_time_ub"],
        initial_nodes=all_initial,
        output_csv_path=str(results_dir / "simplex_result.csv"),
        split_mode=2,  # Mode 2: custom initial points + Mode 1 splitting
        plot_output_dir=str(results_dir / "plots"),
        axis_labels=("x1 (item 1)", "x2 (item 2)", "x3 (item 3)"),
    )

    dt_run = perf_counter() - t0
    print("-" * 70)
    print(f"    Simplex run completed in {dt_run:.2f}s")

    # ── Extract final metrics ─────────────────────────────────────────────
    LB_hist = result["LB_hist"]
    UB_hist = result["UB_hist"]

    n_iters = len(LB_hist)
    # NOTE: With equal probabilities p_s = 1/S the per-scenario sum
    # is divided by S to obtain E[Q].  If probabilities become unequal,
    # replace `/S` with a proper weighted sum.
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
    if ef_obj is not None:
        print(f"  EF optimal:      {ef_obj:.6f}")
        if math.isfinite(final_UB):
            opt_gap = abs(final_UB - ef_obj) / (abs(ef_obj) + 1e-16)
            print(f"  UB vs EF opt:    {opt_gap:.6e}")
    elif KNOWN_OPTIMAL is not None:
        print(f"  Known optimal:   {KNOWN_OPTIMAL:.6f}")
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
        f.write(f"scenarios={SCENARIO_NAMES}\n")
        f.write(f"probabilities={PROBS}\n")
        f.write(f"budget_B={B}\n")
        f.write(f"demands={DEMANDS}\n")
        f.write(f"c_short={C_SHORT}\n")
        f.write(f"c_hold={C_HOLD}\n")
        f.write(f"iterations={n_iters}\n")
        f.write(f"final_nodes={result['node_count'][-1] if result['node_count'] else 'N/A'}\n")
        f.write(f"final_LB_avg={final_LB}\n")
        f.write(f"final_UB_avg={final_UB}\n")
        f.write(f"gap_abs={final_gap_abs}\n")
        f.write(f"gap_rel={final_gap_rel}\n")
        if ef_obj is not None:
            f.write(f"ef_optimal={ef_obj}\n")
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
