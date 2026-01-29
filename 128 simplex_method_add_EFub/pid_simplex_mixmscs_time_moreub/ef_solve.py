# ef_solve.py
"""
Extensive Form (EF) solve with IPOPT for UB candidate generation.

Features:
- Generic model build (no Kp/Ki/Kd or obj_expr assumptions)
- Barycentric constraints to restrict solution to current simplex
- Timing and diagnostics logging
- IPOPT exit text extraction
"""

from __future__ import annotations
import os
import math
import warnings
from time import perf_counter
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition


def _get_active_objective(model: pyo.ConcreteModel) -> Optional[pyo.Objective]:
    """Get the single active objective from a model, or None."""
    active_objs = [
        obj for obj in model.component_objects(pyo.Objective, active=True)
    ]
    if len(active_objs) == 1:
        return active_objs[0]
    elif len(active_objs) == 0:
        # Try all objectives (including inactive)
        all_objs = list(model.component_objects(pyo.Objective))
        if len(all_objs) == 1:
            return all_objs[0]
    return None


def _build_ef_model(
    model_list: Sequence[pyo.ConcreteModel],
    first_vars_list: Sequence[Sequence[pyo.Var]],
    simplex_verts: Optional[List[List[float]]] = None,
) -> Tuple[pyo.ConcreteModel, List[pyo.Var]]:
    """
    Build an Extensive Form (aggregate) Pyomo model.
    
    Parameters
    ----------
    model_list : list[ConcreteModel]
        Scenario models (will be cloned).
    first_vars_list : list[list[Var]]
        First-stage variable references for each scenario.
    simplex_verts : list[list[float]], optional
        Simplex vertices (d+1 vertices, each of dimension d).
        If provided, solution is constrained to lie in simplex.
    
    Returns
    -------
    (mAgg, x_shared_list)
        mAgg: The aggregate model
        x_shared_list: List of shared first-stage variables
    """
    if not model_list:
        raise ValueError("model_list is empty")
    if not first_vars_list or len(first_vars_list) != len(model_list):
        raise ValueError("first_vars_list must match model_list length")
    
    S = len(model_list)
    d = len(first_vars_list[0])  # dimension of first-stage
    
    mAgg = pyo.ConcreteModel()
    mAgg.S = pyo.RangeSet(0, S - 1)
    mAgg.D = pyo.RangeSet(0, d - 1)
    
    # Get bounds from first scenario's first-stage vars
    def _get_bounds(var):
        lb = float(var.lb) if var.lb is not None else None
        ub = float(var.ub) if var.ub is not None else None
        return (lb, ub)
    
    first_vars_0 = list(first_vars_list[0])
    
    # Shared first-stage variables
    mAgg.x_shared = pyo.Var(
        mAgg.D, 
        bounds=lambda m, i: _get_bounds(first_vars_0[i])
    )
    
    # === Barycentric constraints (if simplex vertices provided) ===
    if simplex_verts is not None:
        n_verts = len(simplex_verts)  # should be d+1 for simplex
        mAgg.V = pyo.RangeSet(0, n_verts - 1)
        
        # Lambda variables (barycentric coordinates)
        mAgg.lam = pyo.Var(mAgg.V, bounds=(0, 1))
        
        # sum(lambda) = 1
        mAgg.lam_sum = pyo.Constraint(
            expr=sum(mAgg.lam[j] for j in mAgg.V) == 1
        )
        
        # x_shared[i] = sum_j lam[j] * verts[j][i]
        def _bary_rule(m, i):
            return m.x_shared[i] == sum(
                m.lam[j] * simplex_verts[j][i] for j in m.V
            )
        mAgg.bary_constr = pyo.Constraint(mAgg.D, rule=_bary_rule)
    
    # === Clone scenario models and link first-stage vars ===
    def _scen_block_rule(b, s_idx):
        m_s = model_list[s_idx].clone()
        
        # Deactivate existing objectives (we'll build aggregate)
        for obj in m_s.component_objects(pyo.Objective, active=True):
            obj.deactivate()
        
        b.m = m_s
    
    mAgg.scen = pyo.Block(mAgg.S, rule=_scen_block_rule)
    
    # Link first-stage vars: scenario_var == x_shared
    def _link_first_stage(m, s, i):
        # Get the corresponding var in cloned model by name
        orig_var = first_vars_list[s][i]
        cloned_var = m.scen[s].m.find_component(orig_var.name)
        if cloned_var is None:
            raise RuntimeError(f"Cannot find {orig_var.name} in cloned model for scenario {s}")
        return cloned_var == m.x_shared[i]
    
    mAgg.link_first_stage = pyo.Constraint(mAgg.S, mAgg.D, rule=_link_first_stage)
    
    # === Aggregate objective ===
    obj_exprs = []
    for s in range(S):
        orig_model = model_list[s]
        orig_obj = _get_active_objective(orig_model)
        if orig_obj is None:
            raise RuntimeError(f"Scenario {s} has no identifiable objective")
        
        # Get the objective expression from the cloned model
        cloned_obj = mAgg.scen[s].m.find_component(orig_obj.name)
        if cloned_obj is not None:
            obj_exprs.append(cloned_obj.expr)
        else:
            # Fallback: try to find any objective
            for obj in mAgg.scen[s].m.component_objects(pyo.Objective):
                obj_exprs.append(obj.expr)
                break
    
    if len(obj_exprs) != S:
        raise RuntimeError(f"Could not find objectives for all {S} scenarios")
    
    # Equal weights (can be extended for probabilities)
    mAgg.obj = pyo.Objective(
        expr=sum(obj_exprs),
        sense=pyo.minimize
    )
    
    x_shared_list = [mAgg.x_shared[i] for i in mAgg.D]
    return mAgg, x_shared_list


def _extract_ipopt_exit(log_path: Path, tail_lines: int = 200) -> Optional[str]:
    """Extract the last IPOPT 'EXIT:' line from log file."""
    if not log_path.exists():
        return None
    
    try:
        with open(log_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        
        for line in reversed(lines[-tail_lines:]):
            if "EXIT:" in line:
                return line.strip()
        return None
    except Exception:
        return None


def solve_ef_with_ipopt(
    model_list: Sequence[pyo.ConcreteModel],
    first_vars_list: Sequence[Sequence[pyo.Var]],
    simplex_verts: Optional[List[List[float]]] = None,
    run_dir: Optional[Path] = None,
    iteration: int = 0,
    time_limit: float = 600.0,
    ipopt_options: Optional[Dict[str, Any]] = None,
) -> Tuple[bool, Optional[List[float]], Optional[float], Dict[str, Any]]:
    """
    Solve the Extensive Form with IPOPT and return diagnostics.
    
    Parameters
    ----------
    model_list : list[ConcreteModel]
        Scenario models.
    first_vars_list : list[list[Var]]
        First-stage variables for each scenario.
    simplex_verts : list[list[float]], optional
        If provided, constrain solution to this simplex via barycentric coords.
    run_dir : Path, optional
        Directory to write IPOPT log file.
    iteration : int
        Current iteration (for log filename).
    time_limit : float
        Time limit in seconds (passed via timelimit= to solve, matching snoglode).
    ipopt_options : dict, optional
        Additional IPOPT options.
    
    Returns
    -------
    (ok, x_sol_list, ub_value, solve_info_dict)
        ok : bool - True if solve succeeded
        x_sol_list : list[float] or None - Solution values
        ub_value : float or None - Objective value
        solve_info_dict : dict with solver diagnostics
    """
    solve_info = {
        "solver_name": "ipopt",
        "options_used": {},
        "time_sec": None,
        "termination_condition": None,
        "status": None,
        "message": None,
        "ipopt_exit": None,
        "simplex_constrained": simplex_verts is not None,
    }
    
    # Build EF model
    try:
        mAgg, x_shared_list = _build_ef_model(model_list, first_vars_list, simplex_verts)
    except Exception as e:
        solve_info["message"] = f"EF build failed: {e}"
        return False, None, None, solve_info
    
    # Configure IPOPT solver
    solver = pyo.SolverFactory("ipopt")
    
    # Default options (matching snoglode/IDAES: gradient-based scaling)
    opts = {
        "nlp_scaling_method": "gradient-based",
    }
    if ipopt_options:
        opts.update(ipopt_options)
    
    for k, v in opts.items():
        solver.options[k] = v
    
    solve_info["options_used"] = dict(opts)
    
    # Configure IPOPT log output
    log_path = None
    if run_dir is not None:
        log_path = Path(run_dir) / f"ipopt_ef_iter_{iteration}.log"
        solver.options["output_file"] = str(log_path)
    
    # Solve with timing (matching snoglode: timelimit= parameter)
    t0 = perf_counter()
    try:
        results = solver.solve(
            mAgg, 
            timelimit=time_limit,  # snoglode uses timelimit= in solve()
            tee=False, 
            load_solutions=True
        )
        time_sec = perf_counter() - t0
    except Exception as e:
        time_sec = perf_counter() - t0
        solve_info["time_sec"] = time_sec
        solve_info["message"] = f"Solve exception: {e}"
        return False, None, None, solve_info
    
    # Record timing and status
    solve_info["time_sec"] = time_sec
    solve_info["termination_condition"] = str(results.solver.termination_condition)
    solve_info["status"] = str(results.solver.status)
    solve_info["message"] = getattr(results.solver, "message", None)
    
    # Extract IPOPT exit text
    if log_path is not None:
        solve_info["ipopt_exit"] = _extract_ipopt_exit(log_path)
    
    # Check success
    ok_terms = {TerminationCondition.optimal}
    if hasattr(TerminationCondition, "locallyOptimal"):
        ok_terms.add(TerminationCondition.locallyOptimal)
    
    success = (
        results.solver.status == SolverStatus.ok and
        results.solver.termination_condition in ok_terms
    )
    
    if success:
        x_sol = [float(pyo.value(v)) for v in x_shared_list]
        ub_value = float(pyo.value(mAgg.obj))
        return True, x_sol, ub_value, solve_info
    
    return False, None, None, solve_info
