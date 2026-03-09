# Simplex-Based 2 StageStochastic Programming Global Optimization

A spatial branch-and-bound method that partitions the first-stage variable domain using Delaunay simplices, computing lower bounds via McCormick–Style (ms) and constant-cut (cs) relaxations and upper bounds via scenario-wise Q evaluations and an optional extensive-form (EF) solver.

## Core Algorithm Files

| File | Role |
|------|------|
| `simplex_specialstart.py` | **Main entry point.** Contains `run_pid_simplex_3d()` — the outer loop that manages simplex splitting, LB/UB tracking, candidate selection, convergence checks, and all logging. |
| `bundles.py` | `BaseBundle` (Q evaluation) and `MSBundle` (ms/cs lower-bounding) wrappers around Pyomo+Gurobi. Handles solve caching, warm-starts, and metadata capture. |
| `utils.py` | Helper functions: `corners_from_var_bounds`, `evaluate_Q_at`, `tet_quality`, `min_dist_to_nodes`, plotting utilities. |
| `simplex_geometry.py` | `SimplexMesh` — incremental Delaunay mesh management (add node, re-tessellate, simplex lookup). |
| `ef_upper_bounder.py` | Extensive-form UB solver (IPOPT + Gurobi). Solves the full stochastic program for a tight UB. |
| `iter_logger.py` | Per-iteration structured logging (UB provenance, phase timing). |
| `_safe_linalg.py` | Robust linear algebra helpers (barycentric coords, safe `linalg.solve`). |

## Output Files

All outputs are written to `results/<case_name>/`:

| `simplex_result.csv` | Per-iteration: time, node count, LB, UB, gap |
| `simplex_debug.txt` | ms/cs/Q failures (only when issues occur) |
| `simplex_record_split.txt` | Per-iteration split details: parent LB, child LBs |
| `simplex_record_subproblem_runtime.txt` | Per-iteration ms/cs/Q aggregate timing |

## Test-Cases

| Runner | Problem | First-stage dim | Scenarios | Notes |
|--------|---------|:-:|:-:|-------|
| `run_2_1_1_case.py` | 2_1_1 (nonconvex QP) | 2 | 1000 | Plasmo.jl `RandomStochasticModel` |
| `run_2_1_2_case.py` | 2_1_2 (nonconvex QP) | 2 | 1000 | Converges in 1 iter (small domain) |
| `run_2_1_3_case.py` | 2_1_3 (nonconvex QP) | 2 | 1000 | |
| `run_2_1_7_case.py` | 2_1_7 (nonconvex QP) | 5 | 100 | PlasmoOld, `nfirst=5`, `nparam=5` |
| `run_2_1_8_case.py` | 2_1_8 (nonconvex QP) | 5 | 100 | First-stage equality constraint |
| `run_2_1_10_case.py` | 2_1_10 (DC quadratic) | 2 | 1000 | Difference-of-convex objective |
| `run_2_1_10_sngo_case.py` | 2_1_10 (DC quadratic) | 5 | 100 | PlasmoOld/SNGO version |
| `run_14_1_6_case.py` | 14_1_6 (nonconvex QCQP) | 2 | 1000 | Bilinear + quadratic constraints |
| `run_pid_case.py` | PID controller | 3 | varies | Simulation-based Q (black-box) |
| `run_st_fp7*.py` | ST_FP7 variants (nonconvex QP) | 2 | varies | |
| `run_st_rv*.py` | ST_RV variants (nonconvex QP) | 2 | varies | |
