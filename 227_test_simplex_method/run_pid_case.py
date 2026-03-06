"""
run_pid_case.py — PID regression test runner (smoke + full).

Usage:
    python run_pid_case.py --mode smoke      # fast regression check (~2 min)
    python run_pid_case.py --mode full       # formal run (~30-60 min)
    python run_pid_case.py                   # defaults to smoke

All parameters are at the top of this file for easy editing.
Results written to results/pid_{mode}/ directory.
"""

import os
import sys

# ---------------------------------------------------------------------------
# Windows encoding fix: wrap stdout/stderr to handle Unicode chars (✓, ≤, →)
# in the core algorithm's print statements without crashing.
# ---------------------------------------------------------------------------
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    import io
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding=sys.stdout.encoding, errors="replace")
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding=sys.stderr.encoding, errors="replace")

import argparse
import json
import math
import time
from pathlib import Path
from time import perf_counter

import numpy as np

# ---------------------------------------------------------------------------
# 0. PARAMETERS — edit this section to tune smoke / full runs
# ---------------------------------------------------------------------------

# Path to PID scenario data (relative to this script)
DATA_CSV = "data.csv"

# Common parameters
COMMON = {
    "T": 15.0,             # time horizon
    "nfe": 20,             # finite elements
    "weights": (10.0, 0.01),
    "bounds": {
        "Kp": (-10.0, 10.0),
        "Ki": (-100.0, 100.0),
        "Kd": (-100.0, 100.0),
        "x": (-2.5, 2.5),
        "u": (-5.0, 5.0),
        "e": (None, None),
        "I": (None, None),
    },
    "sp0": 0.0,
    "sp1": 0.5,
}

# Solver options (Gurobi)
UB_SOLVER_OPTS = {"NonConvex": 2}
LB_SOLVER_OPTS = {
    "NonConvex": 2,
    "MIPGap": 1e-1,
    "TimeLimit": 30,
}

# Penalty value for infeasible Q evaluations (passed to BaseBundle)
Q_MAX = 1e3

#                            SMOKE            FULL
# ─────────────────────────────────────────────────────
MODE_PARAMS = {
    "smoke": {
        "max_scenarios":   2,        # few scenarios for speed
        "skip":            0,
        "target_nodes":    120,       # small mesh
        "gap_stop_tol":    1e-8,     # tight gap (matches app.ipynb)
        "time_limit":      30,      # no wall-clock cap — run until gap or target_nodes
        "enable_3d_plot":  False,    # no plotting
        "enable_ef_ub":    True,
        "ef_time_ub":      30.0,     # short EF solve
        "use_exact_opt":   False,
    },
    "full": {
        "max_scenarios":   50,     # all scenarios in CSV
        "skip":            0,
        "target_nodes":    1000,     # matches app.ipynb
        "gap_stop_tol":    1e-6,     # matches app.ipynb
        "time_limit":      60*10*1,  # 7 hours (matches app.ipynb)
        "enable_3d_plot":  False,    # disable for headless regression
        "enable_ef_ub":    True,
        "ef_time_ub":      60.0,     # standard EF time
        "use_exact_opt":   False,
    },
}

# Fixed random seed for reproducibility
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# 1. MAIN
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="PID regression test runner (smoke / full)")
    parser.add_argument(
        "--mode", choices=["smoke", "full"], default="full",
        help="Run mode: 'smoke' (fast regression) or 'full' (formal run)")
    args = parser.parse_args()
    mode = args.mode
    params = MODE_PARAMS[mode]

    # Ensure working directory is the script's directory
    script_dir = Path(__file__).resolve().parent
    os.chdir(script_dir)
    sys.path.insert(0, str(script_dir))

    np.random.seed(RANDOM_SEED)

    # Results directory
    results_dir = script_dir / "results" / f"pid_{mode}"
    results_dir.mkdir(parents=True, exist_ok=True)

    csv_path = str(script_dir / DATA_CSV)
    if not os.path.isfile(csv_path):
        print(f"ERROR: Data file not found: {csv_path}")
        sys.exit(1)

    # ── Print header ──────────────────────────────────────────────────────
    print("=" * 70)
    print(f"  PID Regression Test — mode={mode.upper()}")
    print("=" * 70)
    print(f"  Data file:       {csv_path}")
    print(f"  Results dir:     {results_dir}")
    print(f"  Random seed:     {RANDOM_SEED}")
    print(f"  T={COMMON['T']}, nfe={COMMON['nfe']}, "
          f"weights={COMMON['weights']}")
    print(f"  max_scenarios:   {params['max_scenarios'] or 'ALL'}")
    print(f"  target_nodes:    {params['target_nodes']}")
    print(f"  gap_stop_tol:    {params['gap_stop_tol']}")
    print(f"  time_limit:      {params['time_limit'] or 'None (unlimited)'}")
    print(f"  enable_ef_ub:    {params['enable_ef_ub']}")
    print(f"  ef_time_ub:      {params['ef_time_ub']}")
    print(f"  enable_3d_plot:  {params['enable_3d_plot']}")
    print(f"  use_exact_opt:   {params['use_exact_opt']}")
    print("=" * 70)

    # ── Build models ──────────────────────────────────────────────────────
    print("\n[1] Building PID scenario models...")
    t0 = perf_counter()

    from modeling import build_models_from_csv
    model_list, first_stg_vars_list, m_tmpl_list, nfe = build_models_from_csv(
        csv_path=csv_path,
        T=COMMON["T"],
        nfe=COMMON["nfe"],
        weights=COMMON["weights"],
        bounds=COMMON["bounds"],
        sp0=COMMON["sp0"],
        sp1=COMMON["sp1"],
        max_scenarios=params["max_scenarios"],
        skip=params["skip"],
    )
    S = len(model_list)
    dt_build = perf_counter() - t0
    print(f"    {S} scenarios built in {dt_build:.2f}s")

    # ── Build solver bundles ──────────────────────────────────────────────
    print("\n[2] Building solver bundles (BaseBundle + MSBundle)...")
    t0 = perf_counter()

    from bundles import BaseBundle, MSBundle

    base_bundles = [BaseBundle(m, UB_SOLVER_OPTS, q_max=Q_MAX) for m in model_list]
    ms_bundles = [
        MSBundle(m, yvars, LB_SOLVER_OPTS, scenario_index=s)
        for s, (m, yvars) in enumerate(zip(model_list, first_stg_vars_list))
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

    result = run_pid_simplex_3d(
        base_bundles=base_bundles,
        ms_bundles=ms_bundles,
        model_list=model_list,
        first_vars_list=first_stg_vars_list,
        target_nodes=params["target_nodes"],
        verbose=True,
        gap_stop_tol=params["gap_stop_tol"],
        tracker=tracker,
        enable_3d_plot=params["enable_3d_plot"],
        use_exact_opt=params["use_exact_opt"],
        time_limit=params["time_limit"],
        enable_ef_ub=params["enable_ef_ub"],
        ef_time_ub=params["ef_time_ub"],
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
    final_gap_abs = (final_UB - final_LB)
    final_gap_rel = final_gap_abs / (abs(final_UB) + 1e-16)

    # ── Print summary ─────────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print(f"  RESULTS SUMMARY — mode={mode.upper()}")
    print("=" * 70)
    print(f"  Iterations:      {n_iters}")
    print(f"  Final nodes:     {result['node_count'][-1] if result['node_count'] else 'N/A'}")
    print(f"  Final LB (avg):  {final_LB:.9f}")
    print(f"  Final UB (avg):  {final_UB:.9f}")
    print(f"  Abs gap (avg):   {final_gap_abs:.9f}")
    print(f"  Rel gap:         {final_gap_rel*100:.4f}%")
    print(f"  Total time:      {dt_run:.2f}s")
    print(f"  Build time:      {dt_build + dt_bundles:.2f}s")
    print("=" * 70)

    # ── Write result file ─────────────────────────────────────────────────
    summary = {
        "mode": mode,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "scenarios": S,
        "random_seed": RANDOM_SEED,
        "params": {k: v for k, v in params.items()},
        "results": {
            "iterations": n_iters,
            "final_nodes": int(result["node_count"][-1]) if result["node_count"] else 0,
            "final_LB_avg": final_LB,
            "final_UB_avg": final_UB,
            "gap_abs_avg": final_gap_abs,
            "gap_rel": final_gap_rel,
            "total_runtime_s": dt_run,
            "build_time_s": dt_build + dt_bundles,
        },
        "histories": {
            "LB_per_iter": [lb / S for lb in LB_hist],
            "UB_per_iter": [ub / S for ub in UB_hist],
            "node_count": result["node_count"],
        },
    }

    result_file = results_dir / "result.json"
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, default=_json_safe)
    print(f"\n  Result file: {result_file}")

    # Also write a quick human-readable text summary
    txt_file = results_dir / "result.txt"
    with open(txt_file, "w", encoding="utf-8") as f:
        f.write(f"PID Regression — mode={mode}\n")
        f.write(f"{'='*50}\n")
        f.write(f"Scenarios:   {S}\n")
        f.write(f"Iterations:  {n_iters}\n")
        f.write(f"Final LB:    {final_LB:.9f}\n")
        f.write(f"Final UB:    {final_UB:.9f}\n")
        f.write(f"Abs gap:     {final_gap_abs:.9f}\n")
        f.write(f"Rel gap:     {final_gap_rel*100:.4f}%\n")
        f.write(f"Runtime:     {dt_run:.2f}s\n")
        f.write(f"Timestamp:   {summary['timestamp']}\n")
        f.write(f"\nLB history (avg per scenario):\n")
        for i, lb in enumerate(LB_hist):
            ub = UB_hist[i]
            f.write(f"  iter {i:3d}: LB={lb/S:.9f}  UB={ub/S:.9f}\n")
    print(f"  Text file:   {txt_file}")

    # Write per-iteration CSV
    import csv
    csv_file = results_dir / "simplex_result.csv"
    iter_times = result.get("iter_time_hist", [])
    with open(csv_file, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "iter", "time_s", "nodes", "LB", "UB",
            "gap_abs", "gap_rel_pct",
            "active_ratio",
        ])
        for i in range(n_iters):
            lb_i = LB_hist[i] / S
            ub_i = UB_hist[i] / S
            gap_a = ub_i - lb_i
            gap_r = gap_a / (abs(ub_i) + 1e-16) * 100
            nc = result["node_count"][i] if i < len(result["node_count"]) else ""
            ar = (result["active_ratio_hist"][i]
                  if i < len(result["active_ratio_hist"]) else "")
            t_i = f"{iter_times[i]:.3f}" if i < len(iter_times) else ""
            writer.writerow([
                i, t_i, nc, f"{lb_i:.9f}", f"{ub_i:.9f}",
                f"{gap_a:.9f}", f"{gap_r:.6f}",
                f"{ar:.6f}" if isinstance(ar, float) else ar,
            ])
    print(f"  CSV file:    {csv_file}")

    print(f"\n  ✓ PID {mode} regression complete.")
    return 0


def _json_safe(obj):
    """JSON encoder fallback for numpy/inf/nan types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, float):
        if math.isnan(obj):
            return "NaN"
        if math.isinf(obj):
            return "Inf" if obj > 0 else "-Inf"
    return str(obj)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception:
        import traceback
        tb = traceback.format_exc()
        # Print to stdout (stderr may be swallowed by PowerShell encoding)
        print("\n" + "=" * 70)
        print("FATAL ERROR — full traceback below:")
        print("=" * 70)
        print(tb)
        # Also write to file in case stdout is also lost
        crash_file = Path(__file__).resolve().parent / "crash_log.txt"
        with open(crash_file, "w", encoding="utf-8") as f:
            f.write(tb)
        print(f"\nCrash log written to: {crash_file}")
        sys.exit(1)
