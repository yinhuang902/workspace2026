# Adding a New Problem to the Simplex Method Solver

> **Audience**: anyone who wants to solve a new 2-stage stochastic
> programming problem using this codebase without modifying the core algorithm.

---

## 1. Overview

### What problems does this code support?

This codebase solves **two-stage stochastic programs** of the form:

```
min_x  E_ω[ Q(x, ω) ]
s.t.   x ∈ X ⊂ R^d       (first-stage; continuous; finite bounds)
       Q(x, ω) = min_y { f(x, y, ω) : g(x, y, ω) ≤ 0 }   (second-stage)
```

The first stage is **continuous** with **finite variable bounds** (the algorithm
generates corner nodes from `[lb, ub]` of each first-stage variable). The
second-stage subproblems can be **nonconvex** or **mixed-integer**, depending on
the solver settings (Gurobi with `NonConvex=2` is the default).

The scenarios ω₁, …, ω_S are provided explicitly as a list of Pyomo
`ConcreteModel` instances. Equal probabilities (1/S) are assumed unless
overridden.

### High-level algorithm loop

The algorithm maintains a **simplicial mesh** (Delaunay triangulation) over the
first-stage feasible region and iteratively refines it:

1. **Initialise** corner nodes from variable bounds (2^d nodes for d first-stage
   variables).  Evaluate Q(x, ω) at each corner for every scenario.
2. **Evaluate all simplices**: for each simplex and each scenario, solve the
   *ms subproblem* (linear surrogate gap) and the *constant-cut c_s* (minimum
   Q on the simplex).  Aggregate across scenarios to get per-simplex LB
   (surrogate lower bound) and UB.
3. **Identify active simplices** whose LB is within tolerance of the global
   UB.  The simplex with the **smallest LB** is selected.
4. **Generate a new point** from the selected simplex: choose from the
   ms-candidate, the c_s-candidate, or an edge/face midpoint fallback.
5. **Update UB** by evaluating Q at the new point, and optionally by solving a
   restricted **Extensive Form (EF)** on the selected simplex.
6. **Subdivide** the simplex containing the new point (star / edge / face
   split).
7. **Repeat** until the gap ≤ tolerance, or the node/time budget is exhausted.

---

## 2. Required Problem Format

To plug a new problem into the solver, you must provide the objects described
below.  The reference implementation is the **PID controller tuning** problem in
`run_pid_case.py` + `modeling.py`.

### 2.1 `model_list` — Pyomo scenario models

```
model_list: list[pyomo.ConcreteModel]     # length S (one per scenario)
```

Each model must expose:

| Attribute       | Type                  | Description |
|-----------------|-----------------------|-------------|
| `obj_expr`      | Pyomo `Expression`    | The scalar second-stage objective expression Q(x, ω). The algorithm reads `pyo.value(model.obj_expr)` after solving. |
| first-stage vars | `pyo.Var`            | The d first-stage decision variables. They must have **finite `lb` and `ub`** (the algorithm calls `corners_from_var_bounds` to generate the initial 2^d corner nodes). |
| second-stage vars/constraints | any    | Free-form; the solver treats each model as a black box to minimise. |

> **Key rule**: the model should NOT have a pre-installed active `Objective`
> named `obj`.  Both `BaseBundle` and `MSBundle` will install their own
> objectives.  If an `obj` component exists, the bundle constructors will
> `del_component('obj')` before adding theirs.

### 2.2 `first_vars_list` — First-stage variable references

```
first_vars_list: list[list[pyomo.core.base.var.VarData]]   # shape [S][d]
```

For each scenario s, `first_vars_list[s]` is a length-d list of references to
the first-stage `Var` objects **inside `model_list[s]`**.  The order must be
**consistent** across scenarios (i.e., `first_vars_list[0][0]` and
`first_vars_list[1][0]` refer to the "same" logical variable).

The `MSBundle` constructor locates these variables by name in a cloned model
(`m.find_component(fv.name)`), so each variable must have a unique,
consistent `.name` attribute.

### 2.3 Variable bounds

Every first-stage variable **must** have finite `lb` and `ub`.  The algorithm
calls `corners_from_var_bounds(first_vars_list[0])` at startup to generate the
initial 2^d corner nodes.  If any bound is `None`, it will raise
`ValueError("... lack UB and LB")`.

### 2.4 `BaseBundle` — True Q evaluation (one per scenario)

```python
from bundles import BaseBundle

base_bundles = [BaseBundle(model, solver_options) for model in model_list]
```

`BaseBundle` wraps a Pyomo model with a `GurobiPersistent` solver and provides:

| Method | Signature | Description |
|--------|-----------|-------------|
| `eval_at` | `(first_vars, first_vals, return_meta=False) → float` or `(float, dict)` | Fix first-stage vars, solve, return Q(x, ω). Returns `Q_max = 1e10` on infeasibility. |

**Solver options dict** (passed to constructor): `{"NonConvex": 2}` is typical.
Add `"MIPGap"`, `"TimeLimit"`, etc. as needed.

### 2.5 `MSBundle` — Surrogate subproblem solver (one per scenario)

```python
from bundles import MSBundle

ms_bundles = [
    MSBundle(model, first_vars, solver_options, scenario_index=s)
    for s, (model, first_vars) in enumerate(zip(model_list, first_stg_vars_list))
]
```

`MSBundle` clones the model, adds barycentric variables `lam[0..d]`, and builds
linking constraints that tie the first-stage variables to a convex combination
of simplex vertices.

| Method | Signature | Description |
|--------|-----------|-------------|
| `update_tetra` | `(tet_vertices, fverts_scene) → None` | Update the simplex geometry. `tet_vertices` is a list of d+1 tuples (vertex coordinates); `fverts_scene` is a list of d+1 floats (Q values at those vertices for this scenario). |
| `solve` | `() → bool` | Solve `min(Q_s − A_s)` (ms subproblem). Returns `True` if optimal. The ms value is read from the dual bound. |
| `get_ms_and_point` | `() → (float, ndarray, tuple)` | Returns `(ms_val, lam_star, new_point)`. |
| `solve_const_cut` | `() → (bool, float, tuple\|None)` | Solve `min Q_s` on the simplex to get `c_s` and corresponding point. |

**Solver options dict**: `{"NonConvex": 2, "MIPGap": 1e-3, "TimeLimit": 60}` is
typical.

**If ms returns +inf**: the algorithm falls back to using the sum of finite c_s
values as the lower bound for that simplex. If both ms and c_s fail, the simplex
gets LB = +inf and is effectively pruned.

**If c_s is not available**: set `solve_const_cut` to return
`(False, float('-inf'), None)`. The algorithm will still function using only the
ms-based surrogates.

### 2.6 Scenario probabilities (optional)

Equal-weight (1/S) is currently hard-coded in `run_pid_simplex_3d` (the UB/LB
histories store **sum** over scenarios; the runner divides by S for reporting).

If you need non-uniform probabilities, modify the EF upper bounder:

```python
SimplexEFUb(..., probabilities=[p1, p2, ..., pS])
```

The `SimplexEFUb.__init__` already accepts a `probabilities` parameter and
uses it to weight the per-scenario objectives in the EF model.

### 2.7 EF Upper Bounder (optional but recommended)

```python
from ef_upper_bounder import SimplexEFUb

ef_ub = SimplexEFUb(
    model_list, first_vars_list,
    probabilities=None,    # or list[float] summing to 1
    time_ub=60.0,          # IPOPT time limit per EF solve
    solver_name="ipopt",
    solver_opts=None,
)
```

This component is created inside `run_pid_simplex_3d` when `enable_ef_ub=True`.
You do **not** need to build it yourself; just pass the flags.

---

## 3. Minimal "Run Script" Template

Create a file named `run_other_case.py` (or any name you prefer).  The template
below is copy-paste ready with placeholder functions for your problem.

```python
"""
run_other_case.py — Runner template for a new problem.

Usage:
    python run_other_case.py
"""

import os, sys, json, csv, time, math
from pathlib import Path
from time import perf_counter

import numpy as np
import pyomo.environ as pyo

# Ensure this script's directory is importable
script_dir = Path(__file__).resolve().parent
os.chdir(script_dir)
sys.path.insert(0, str(script_dir))

# =====================================================================
# Step 1: Build your scenario models
# =====================================================================
def build_models_for_my_problem():
    """
    Build and return:
        model_list:      list[pyo.ConcreteModel]   — one per scenario
        first_vars_list: list[list[pyo.Var]]        — first-stage vars per scenario
    
    Each model must expose model.obj_expr (the Q(x,ω) expression).
    Each first-stage variable must have finite lb and ub.
    """
    model_list = []
    first_vars_list = []

    # --- Example placeholder ---
    scenarios = [...]  # your scenario data
    for scen_data in scenarios:
        m = pyo.ConcreteModel()
        # Define first-stage variables with FINITE bounds
        m.x1 = pyo.Var(bounds=(-10.0, 10.0))
        m.x2 = pyo.Var(bounds=(0.0, 100.0))
        # ... define second-stage variables, constraints, etc. ...
        # Define the objective expression (NOT an Objective component)
        m.obj_expr = ...  # Pyomo Expression for Q(x, ω)
        
        model_list.append(m)
        first_vars_list.append([m.x1, m.x2])   # consistent order!

    return model_list, first_vars_list


# =====================================================================
# Step 2: Configuration
# =====================================================================
UB_SOLVER_OPTS = {"NonConvex": 2}
LB_SOLVER_OPTS = {"NonConvex": 2, "MIPGap": 1e-3, "TimeLimit": 60}

TARGET_NODES  = 200       # max simplex nodes
GAP_STOP_TOL  = 1e-6      # relative gap convergence
TIME_LIMIT    = 7200.0     # wall-clock seconds (None = unlimited)
ENABLE_EF_UB  = True       # solve restricted EF for tighter UB
EF_TIME_UB    = 60.0       # time limit per EF solve
ENABLE_PLOT   = False      # enable 3D Plotly plots
RANDOM_SEED   = 42


# =====================================================================
# Step 3: Main
# =====================================================================
def main():
    np.random.seed(RANDOM_SEED)

    # Build models
    print("[1] Building scenario models...")
    t0 = perf_counter()
    model_list, first_vars_list = build_models_for_my_problem()
    S = len(model_list)
    print(f"    {S} scenarios built in {perf_counter() - t0:.2f}s")

    # Build bundles
    print("[2] Building solver bundles...")
    from bundles import BaseBundle, MSBundle

    base_bundles = [BaseBundle(m, UB_SOLVER_OPTS) for m in model_list]
    ms_bundles = [
        MSBundle(m, fvars, LB_SOLVER_OPTS, scenario_index=s)
        for s, (m, fvars) in enumerate(zip(model_list, first_vars_list))
    ]

    # Run simplex algorithm
    print("[3] Running simplex algorithm...")
    from simplex_specialstart import run_pid_simplex_3d
    from utils import SimplexTracker

    tracker = SimplexTracker()

    result = run_pid_simplex_3d(
        base_bundles=base_bundles,
        ms_bundles=ms_bundles,
        model_list=model_list,
        first_vars_list=first_vars_list,
        target_nodes=TARGET_NODES,
        verbose=True,
        gap_stop_tol=GAP_STOP_TOL,
        tracker=tracker,
        enable_3d_plot=ENABLE_PLOT,
        use_exact_opt=False,
        time_limit=TIME_LIMIT,
        enable_ef_ub=ENABLE_EF_UB,
        ef_time_ub=EF_TIME_UB,
        # axis_labels=("x1", "x2", "x3"),  # custom axis labels for plots
    )

    # Extract and print results
    LB_hist = result["LB_hist"]
    UB_hist = result["UB_hist"]
    n_iters = len(LB_hist)
    final_LB = LB_hist[-1] / S if LB_hist else float("nan")
    final_UB = UB_hist[-1] / S if UB_hist else float("nan")

    print(f"\n  Iterations:  {n_iters}")
    print(f"  Final LB:    {final_LB:.9f}")
    print(f"  Final UB:    {final_UB:.9f}")
    print(f"  Rel gap:     {(final_UB - final_LB) / (abs(final_UB) + 1e-16) * 100:.4f}%")


if __name__ == "__main__":
    main()
```

### Expected outputs

| File / Directory                   | Description |
|------------------------------------|-------------|
| `results/pid_smoke/result.json`    | Full JSON summary (only if you add the writing code from `run_pid_case.py`) |
| `results/pid_smoke/result.txt`     | Human-readable text summary |
| `results/pid_smoke/simplex_result.csv` | Per-iteration CSV (iter, nodes, LB, UB, gap, active_ratio) |
| `simplex_method_debug_log/`       | Debug logs: per-iteration LB decomposition, EF info, timing, split patterns |
| `simplex_result.txt`              | Algorithm console output mirror |
| `simplex_result.csv`              | Quick per-iteration CSV in the script directory |

> **Note**: the `simplex_method_debug_log/` directory and `simplex_result.*`
> files in the script directory are written by the algorithm itself inside
> `run_pid_simplex_3d`.  The `results/` directory files are written by the
> runner script (you copy the pattern from `run_pid_case.py` lines 237–308).

---

## 4. Configuration and Solver Notes

### 4.1 Solver choices

| Component | Default solver | Set via |
|-----------|---------------|---------|
| Q evaluation (`BaseBundle`) | `GurobiPersistent` | `UB_SOLVER_OPTS` dict |
| ms / c_s solve (`MSBundle`) | `GurobiPersistent` | `LB_SOLVER_OPTS` dict |
| c_s warm-start | IPOPT (via IDAES or `SolverFactory`) | Automatic; falls back gracefully |
| EF upper bound | IPOPT | `SimplexEFUb(solver_name="ipopt")` |

### 4.2 Key Gurobi parameters

```python
{
    "NonConvex": 2,       # REQUIRED for nonconvex QCPs
    "MIPGap": 1e-3,       # Relative optimality gap for ms/c_s
    "TimeLimit": 60,      # Per-solve time limit (seconds)
    "NumericFocus": 1,    # Numerical precision (0–3)
    "Presolve": 2,        # Aggressiveness of presolve
}
```

### 4.3 Persistent solver and the reset strategy

The `MSBundle` uses `GurobiPersistent` with incremental coefficient updates via
`chgCoeff` (see `_set_link_coeffs`).  This avoids rebuilding the model from
scratch every iteration.

**Important**: at the end of `update_tetra`, we reset only the **Gurobi model's
solution state** (basis, incumbent, MIP starts) — NOT the Pyomo persistent
mapping:

```python
gmodel = self.gp._solver_model
gmodel.reset()   # clears solution state; keeps model structure + Pyomo mapping
```

Calling `self.gp.reset()` (the Pyomo wrapper) would **invalidate** the
`_pyomo_con_to_solver_con_map` / `_pyomo_var_to_solver_var_map` dictionaries,
breaking all subsequent `chgCoeff` calls.

Set `DEBUG_PERSISTENT_MAPS = True` in `bundles.py` to log map sizes after each
`update_tetra` call (useful when debugging mapping corruption).

### 4.4 Algorithm parameter knobs

These are passed to `run_pid_simplex_3d(...)`:

| Parameter | Type | Description |
|-----------|------|-------------|
| `target_nodes` | `int` | Max number of nodes (budget). |
| `gap_stop_tol` | `float` | Stop when relative gap ≤ this value. |
| `time_limit` | `float\|None` | Wall-clock time limit (seconds). `None` = unlimited. |
| `enable_3d_plot` | `bool` | Master switch for Plotly 3D plots. |
| `plot_every` | `int\|None` | Plot every N iterations (if plotting enabled). |
| `enable_ef_ub` | `bool` | Whether to solve restricted EF for tighter UB. |
| `ef_time_ub` | `float` | Per-EF-solve time limit for IPOPT. |
| `split_mode` | `int` | `1` = standard (corner-start), `2` = custom initial nodes. |
| `initial_nodes` | `list\|None` | Custom starting nodes (used when `split_mode=2`). |
| `axis_labels` | `tuple\|None` | Custom axis labels for 3D plots (default: `("Kp", "Ki", "Kd")`). |

---

## 5. Debugging Checklist

### Common failures

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| `ValueError: ... lack UB and LB` | Missing bounds on first-stage variable | Set finite `lb` and `ub` on all first-stage vars. |
| `RuntimeError: Can't find first-stage variable '...' in cloned model` | Variable name mismatch between original and clone | Ensure variable names are unique and don't change after construction. |
| ms returns `+inf` for all scenarios on some simplex | Subproblem is infeasible in that region | Check variable bounds, constraint feasibility. May indicate bounds are too tight or the model has structural infeasibility. |
| Duplicate point / collision → early termination | New candidate coincides with existing node | Check `min_dist` parameter. Collisions often mean convergence on sub-optimal faces; inspect `simplex_result.txt`. |
| Degenerate simplex volume (skipped) | Near-zero-volume simplex after subdivision | Usually harmless (the algorithm skips these). If too many are skipped, consider checking variable scaling. |
| `LB > UB` (inconsistent bounds) | Bug in ms dual bound extraction or c_s dual bound | Check `simplex_method_debug_log/` for per-scenario dual/primal values. Look for `[Invariant]` warnings in stdout. |
| Variable ordering mismatch across scenarios | `first_vars_list[0]` and `first_vars_list[1]` have different logical ordering | Verify that `first_vars_list[s][i]` refers to the same physical variable for all s. |

### Log files to check

| File | Location | Content |
|------|----------|---------|
| Console output / `simplex_result.txt` | Script dir | Full algorithm trace: per-iter LB, UB, gap, candidate selection |
| `simplex_result.csv` | Script dir | Per-iteration CSV (quick spreadsheet analysis) |
| `simplex_method_debug_log/debug_lb_after_split.txt` | Debug dir | Per-simplex LB decomposition after each split |
| `simplex_method_debug_log/simplex_ef_info.txt` | Debug dir | EF solve details, ms/c_s issue counts |
| `simplex_method_debug_log/debug_reachtimelimit.txt` | Debug dir | Time-limit events for ms/c_s solves |
| `simplex_method_debug_log/debug_cs_timing.txt` | Debug dir | Non-optimal c_s (Q-eval) solve details |
| `simplex_method_debug_log/debug_ms_timing.txt` | Debug dir | Non-optimal ms solve details |
| `crash_log.txt` | Script dir | Full traceback if the runner crashes |
| `dbg_crash_trace.txt` | Script dir | Low-level trace of ms_on_tetra_for_scene calls (survives C-level crashes) |

---

## 6. Example: Toy Quadratic (2 scenarios, 2 first-stage variables)

Consider a simple problem:

```
min_x  (1/2) * [Q(x, ω₁) + Q(x, ω₂)]
s.t.   x = (x₁, x₂) ∈ [0, 10] × [0, 10]
```

where:
- `Q(x, ω₁) = (x₁ − 3)² + (x₂ − 4)²`          (bowl centred at (3, 4))
- `Q(x, ω₂) = (x₁ − 7)² + (x₂ − 6)² + 2*x₁`   (bowl centred at (7, 6), tilted)

### Mapping into the required format

```python
import pyomo.environ as pyo

def build_toy_models():
    model_list = []
    first_vars_list = []

    # Scenario 1
    m1 = pyo.ConcreteModel()
    m1.x1 = pyo.Var(bounds=(0, 10))
    m1.x2 = pyo.Var(bounds=(0, 10))
    m1.obj_expr = (m1.x1 - 3)**2 + (m1.x2 - 4)**2
    model_list.append(m1)
    first_vars_list.append([m1.x1, m1.x2])

    # Scenario 2
    m2 = pyo.ConcreteModel()
    m2.x1 = pyo.Var(bounds=(0, 10))
    m2.x2 = pyo.Var(bounds=(0, 10))
    m2.obj_expr = (m2.x1 - 7)**2 + (m2.x2 - 6)**2 + 2*m2.x1
    model_list.append(m2)
    first_vars_list.append([m2.x1, m2.x2])

    return model_list, first_vars_list
```

Then build bundles and run:

```python
from bundles import BaseBundle, MSBundle

model_list, first_vars_list = build_toy_models()

base_bundles = [BaseBundle(m, {"NonConvex": 2}) for m in model_list]
ms_bundles = [
    MSBundle(m, fv, {"NonConvex": 2, "MIPGap": 1e-4}, scenario_index=s)
    for s, (m, fv) in enumerate(zip(model_list, first_vars_list))
]
```

The algorithm will:
1. Start from 4 corner nodes: (0,0), (0,10), (10,0), (10,10).
2. Evaluate Q₁ and Q₂ at each corner.
3. Build a 2D triangulation (triangles, not tetrahedra — the code automatically
   handles d=2 with 3-vertex simplices).
4. Iteratively refine toward the optimal x ≈ (4.5, 5).

> **Note**: the function is named `run_pid_simplex_3d` for historical reasons,
> but it works for **any dimension d** (the simplex geometry code is fully
> generic).

---

## Quick Reference: Key Files

| File | Role |
|------|------|
| `simplex_specialstart.py` | Core algorithm: `run_pid_simplex_3d()`, `SimplexMesh`, `evaluate_all_tetra()` |
| `bundles.py` | `BaseBundle`, `MSBundle`, `SurrogateLBBundle` |
| `ef_upper_bounder.py` | `SimplexEFUb` — restricted Extensive Form upper bounder |
| `utils.py` | `SimplexTracker`, `corners_from_var_bounds`, `evaluate_Q_at`, plotting, tolerances |
| `modeling.py` | PID-specific model builder (reference implementation) |
| `run_pid_case.py` | PID-specific runner (reference implementation) |
| `simplex_geometry.py` | `simplex_volume`, `barycentric_coordinates`, `snap_to_feature` |
| `iter_logger.py` | Per-iteration CSV/TXT logging |
| `exact_opt.py` | Optional per-scenario exact optimum computation |
| `_safe_linalg.py` | Safe linear algebra wrappers |
