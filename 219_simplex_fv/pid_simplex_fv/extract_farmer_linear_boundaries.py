#!/usr/bin/env python
"""
extract_farmer_linear_boundaries.py
====================================
Compute/approximate the linear-region boundaries of the Farmer recourse
value function on the 2D budget face:

    x_w + x_c + x_b = TOTAL (= 500),   x >= 0

Uses a grid-based regime-signature change detection.  The boundary
segments are saved to a JSON file that can be loaded by a separate
simplex-visualization script.

Usage (from repo root):
    python extract_farmer_linear_boundaries.py              # default grid 25
    python extract_farmer_linear_boundaries.py --grid_n 50  # finer grid
    python extract_farmer_linear_boundaries.py --grid_n 10 --save_signatures
"""

import argparse
import io
import json
import os
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
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

# ---------------------------------------------------------------------------
# Solver configuration  (direct Pyomo solve — no BaseBundle)
# ---------------------------------------------------------------------------
SOLVER_NAME = "gurobi"
SOLVER_OPTS = {"TimeLimit": 30}

# ---------------------------------------------------------------------------
# Variable name mapping for regime-signature extraction
# ---------------------------------------------------------------------------
# These are the Pyomo variable names defined in farmer_problem.TwoStageFarmer.
# Model has:
#   m.y[crop]  — purchasing  (crops: "wheat", "corn")
#   m.w[crop]  — selling     (crops: "wheat", "corn", "beets_favorable",
#                                     "beets_unfavorable")
# If the model changes, update this mapping and the script will FAIL LOUDLY.
VAR_MAP = {
    "buy_wheat":       ("y", "wheat"),               # m.y["wheat"]
    "sell_wheat":      ("w", "wheat"),               # m.w["wheat"]
    "buy_corn":        ("y", "corn"),                # m.y["corn"]
    "sell_corn":       ("w", "corn"),                # m.w["corn"]
    "sell_beets_fav":  ("w", "beets_favorable"),     # m.w["beets_favorable"]
}

# Tolerance for classifying a variable as "active"
DEFAULT_TOL = 1e-4


# ---------------------------------------------------------------------------
# Build scenario models  (same as run_farmer_case.py)
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
        # Patch obj_expr as in run_farmer_case.py
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
# Variable introspection helper
# ---------------------------------------------------------------------------
def introspect_model_vars(model):
    """Print all variable names on the model (for debugging)."""
    print("\n  [Introspect] Variables on the Farmer model:")
    for v in model.component_objects(pyo.Var, active=True):
        indices = list(v.keys()) if v.is_indexed() else ["(scalar)"]
        print(f"    {v.name}: indices = {indices}")
    print()


# ---------------------------------------------------------------------------
# Strict VAR_MAP validation
# ---------------------------------------------------------------------------
def validate_var_map(model):
    """
    Ensure every entry in VAR_MAP resolves to an actual Pyomo component.
    Raises RuntimeError with a clear message if anything is missing.
    """
    for key, (comp_name, idx) in VAR_MAP.items():
        comp = model.find_component(comp_name)
        if comp is None:
            available = [v.name for v in model.component_objects(pyo.Var)]
            raise RuntimeError(
                f"VAR_MAP['{key}'] references component '{comp_name}', "
                f"but it does not exist on the model.  "
                f"Available variable components: {available}")
        if hasattr(comp, '__getitem__'):
            if idx not in comp:
                raise RuntimeError(
                    f"VAR_MAP['{key}'] references '{comp_name}[\"{idx}\"]', "
                    f"but index '{idx}' is not present.  "
                    f"Available indices: {list(comp.keys())}")
        print(f"    OK  {key:20s} -> m.{comp_name}[\"{idx}\"]")


# ---------------------------------------------------------------------------
# Regime signature extraction
# ---------------------------------------------------------------------------
def _get_var_value(model, comp_name, idx):
    """Read a solved variable value — assumes model has just been solved."""
    comp = getattr(model, comp_name)
    return float(pyo.value(comp[idx]))


def regime_signature_at(w, c, solver, model_list, first_vars_list,
                        tol=DEFAULT_TOL):
    """
    Solve each scenario at (w, c, TOTAL-w-c), extract regime booleans.

    Returns a tuple of per-scenario sub-signatures:
        ((bool, bool, bool, bool, bool),   # good
         (bool, bool, bool, bool, bool),   # fair
         (bool, bool, bool, bool, bool))   # bad

    Each sub-signature: (buy_wheat>tol, sell_wheat>tol, buy_corn>tol,
                         sell_corn>tol, sell_beets_fav>tol).
    """
    b = TOTAL - w - c
    S = len(model_list)

    sigs = []
    for s_idx in range(S):
        m = model_list[s_idx]
        set_first_stage_x(m, w, c, b)
        solve_scenario_model(solver, m)
        # Variables are now populated on m

        bits = []
        for key, (comp_name, idx) in VAR_MAP.items():
            val = _get_var_value(m, comp_name, idx)
            bits.append(val > tol)

        unfix_first_stage_x(m)
        sigs.append(tuple(bits))

    return tuple(sigs)


# ---------------------------------------------------------------------------
# Grid construction on the 2D feasible triangle
# ---------------------------------------------------------------------------
def build_triangle_grid(grid_n):
    """
    Build a uniform grid on the triangle w >= 0, c >= 0, w+c <= TOTAL.

    Returns:
        grid_pts : list of (w, c)
        grid_idx : dict  (iw, ic) -> index into grid_pts
    """
    ww = np.linspace(0, TOTAL, grid_n)
    cc = np.linspace(0, TOTAL, grid_n)
    grid_pts = []
    grid_idx = {}
    for iw, wv in enumerate(ww):
        for ic, cv in enumerate(cc):
            if wv + cv <= TOTAL + 1e-8:
                idx = len(grid_pts)
                grid_pts.append((float(wv), float(cv)))
                grid_idx[(iw, ic)] = idx
    return grid_pts, grid_idx


# ---------------------------------------------------------------------------
# Boundary extraction from grid signatures  (with deduplication)
# ---------------------------------------------------------------------------
def extract_boundaries(grid_pts, grid_idx, signatures):
    """
    Find boundary segments: edges between adjacent grid points whose
    regime signatures differ.

    Adjacency: horizontal, vertical, and both diagonals in (iw, ic) space.
    Segments are deduplicated by sorting endpoints and rounding coords.
    """
    ROUND_DIGITS = 8
    seen = set()
    boundary_segments = []

    for (iw, ic), idx_a in grid_idx.items():
        for diw, dic in [(1, 0), (0, 1), (1, 1), (1, -1)]:
            nb = (iw + diw, ic + dic)
            if nb in grid_idx:
                idx_b = grid_idx[nb]
                if signatures[idx_a] != signatures[idx_b]:
                    pa = (round(grid_pts[idx_a][0], ROUND_DIGITS),
                          round(grid_pts[idx_a][1], ROUND_DIGITS))
                    pb = (round(grid_pts[idx_b][0], ROUND_DIGITS),
                          round(grid_pts[idx_b][1], ROUND_DIGITS))
                    # Canonical order for deduplication
                    canon = (min(pa, pb), max(pa, pb))
                    if canon not in seen:
                        seen.add(canon)
                        boundary_segments.append(
                            [[pa[0], pa[1]], [pb[0], pb[1]]])

    return boundary_segments


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Extract linear-region boundaries for the Farmer problem")
    parser.add_argument(
        "--grid_n", type=int, default=25,
        help="Grid resolution (default: 25)")
    parser.add_argument(
        "--tol", type=float, default=DEFAULT_TOL,
        help=f"Tolerance for variable activity (default: {DEFAULT_TOL})")
    parser.add_argument(
        "--save_signatures", action="store_true",
        help="Also save per-grid-point signatures to a debug JSON")
    args = parser.parse_args()

    grid_n = args.grid_n
    tol = args.tol

    # Ensure local imports work
    script_dir = Path(__file__).resolve().parent
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    print("=" * 60)
    print("  Farmer Boundary Extraction")
    print("=" * 60)
    print(f"  TOTAL:   {TOTAL}")
    print(f"  GRID_N:  {grid_n}")
    print(f"  tol:     {tol}")
    print(f"  Solver:  {SOLVER_NAME}")
    print(f"  Scenarios: {list(SCENARIOS.keys())}")
    print("=" * 60)

    # ── Build models ──────────────────────────────────────────────
    print("\n[1] Building Farmer scenario models ...")
    t0 = perf_counter()
    scenario_names, model_list, first_vars_list = build_farmer_models()
    dt_build = perf_counter() - t0
    print(f"    {len(model_list)} scenarios built in {dt_build:.2f}s")

    # ── Introspect variable names (once, for the first scenario) ──
    print("\n[2] Introspecting model variables (scenario 0) ...")
    introspect_model_vars(model_list[0])

    # ── Strictly validate VAR_MAP ─────────────────────────────────
    print("[3] Validating VAR_MAP entries ...")
    validate_var_map(model_list[0])
    print()

    # ── Create solver ─────────────────────────────────────────────
    print("[4] Creating solver ...")
    solver = make_solver()
    # Quick smoke-test: solve scenario 0 at a test point
    set_first_stage_x(model_list[0], 100.0, 100.0, 300.0)
    test_obj = solve_scenario_model(solver, model_list[0])
    unfix_first_stage_x(model_list[0])
    print(f"    Smoke-test obj at (100,100,300) scenario 0 = {test_obj:.2f}")

    # ── Build grid ────────────────────────────────────────────────
    print(f"\n[5] Building triangle grid ({grid_n} x {grid_n}) ...")
    grid_pts, grid_idx = build_triangle_grid(grid_n)
    n_grid = len(grid_pts)
    print(f"    {n_grid} feasible grid points")

    # ── Evaluate signatures ───────────────────────────────────────
    print(f"\n[6] Evaluating regime signatures ({n_grid} points) ...")
    t0_grid = perf_counter()

    # Cache: avoid re-solving the same (w, c) point
    sig_cache = {}
    signatures = [None] * n_grid

    for pi, (wv, cv) in enumerate(grid_pts):
        if pi % max(1, n_grid // 10) == 0:
            elapsed = perf_counter() - t0_grid
            print(f"    point {pi:5d}/{n_grid}  ({elapsed:.1f}s)")
        cache_key = (round(wv, 8), round(cv, 8))
        if cache_key in sig_cache:
            signatures[pi] = sig_cache[cache_key]
        else:
            sig = regime_signature_at(wv, cv, solver, model_list,
                                      first_vars_list, tol=tol)
            sig_cache[cache_key] = sig
            signatures[pi] = sig

    dt_grid = perf_counter() - t0_grid
    print(f"    Done in {dt_grid:.1f}s  ({n_grid} points, "
          f"{len(sig_cache)} unique solves)")

    # ── Extract boundaries (deduplicated) ─────────────────────────
    print("\n[7] Extracting boundary segments ...")
    boundary_segments = extract_boundaries(grid_pts, grid_idx, signatures)
    print(f"    {len(boundary_segments)} unique boundary segments found")

    # Count distinct signatures
    unique_sigs = set(signatures)
    print(f"    {len(unique_sigs)} distinct regime signatures")

    # ── Save to JSON ──────────────────────────────────────────────
    cache_dir = Path.cwd() / "farmer_boundary_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)

    out_file = cache_dir / f"boundaries_grid{grid_n}.json"
    payload = {
        "TOTAL": TOTAL,
        "GRID_N": grid_n,
        "tol": tol,
        "scenarios": list(SCENARIOS.keys()),
        "num_grid_points": n_grid,
        "num_unique_solves": len(sig_cache),
        "num_segments": len(boundary_segments),
        "num_distinct_signatures": len(unique_sigs),
        "segments": boundary_segments,
    }

    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"\n[8] Saved boundary data to: {out_file}")
    print(f"    Segments: {len(boundary_segments)}")

    # ── Optional: save full signatures for debugging ──────────────
    if args.save_signatures:
        # Convert tuple signatures to JSON-serializable lists
        sig_data = {
            "TOTAL": TOTAL,
            "GRID_N": grid_n,
            "tol": tol,
            "points_and_signatures": [
                {"w": grid_pts[i][0], "c": grid_pts[i][1],
                 "signature": signatures[i]}
                for i in range(n_grid)
            ],
        }
        sig_file = cache_dir / f"signatures_grid{grid_n}.json"
        with open(sig_file, "w", encoding="utf-8") as f:
            json.dump(sig_data, f, indent=1)
        print(f"    Signatures saved to: {sig_file}")

    total_time = perf_counter() - t0
    print(f"\n{'=' * 60}")
    print(f"  Total time: {total_time:.1f}s")
    print(f"{'=' * 60}")

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
