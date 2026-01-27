"""
WLSQ Surrogate OBBT (Objective-Based Bounds Tightening)

Uses the WLSQ quadratic lower-bound surrogate to tighten the current
node's first-stage box before branching.

Constraint: phi(x) = x^T Q x + c^T x + d <= UB - delta
"""
import numpy as np
from copy import deepcopy
import pyomo.environ as pyo
from pyomo.opt import SolverFactory

import snoglode.utils.MPI as MPI
from snoglode.utils.supported import SupportedVars

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()


def tighten_node_box_by_wlsq_surrogate(
    node,
    subproblems,
    ub_value: float,
    delta: float,
    epsilon: float,
    only_convex: bool,
    solver_name: str,
    solver_opts: dict
) -> dict:
    """
    Tighten node's first-stage box using WLSQ surrogate constraint: phi(x) <= ub_value - delta.
    
    Parameters
    ----------
    node : Node
        Current node in the BnB tree.
    subproblems : Subproblems
        Subproblems object.
    ub_value : float
        Upper bound value for tightening (should be allreduce'd MIN across ranks beforehand).
    delta : float
        Safety margin for constraint phi(x) <= UB - delta.
    epsilon : float
        Branching epsilon - vars with width <= epsilon are removed from to_branch.
    only_convex : bool
        If True, skip tightening if Hessian is not PSD.
    solver_name : str
        Solver name for QCP solves.
    solver_opts : dict
        Solver options.
    
    Returns
    -------
    dict
        {
            'tightened_count': int,
            'max_shrink_ratio': float,
            'skipped_nonconvex': bool,
            'infeasible': bool,
            'removed_vars': list
        }
    """
    result = {
        'tightened_count': 0,
        'max_shrink_ratio': 0.0,
        'skipped_nonconvex': False,
        'infeasible': False,
        'removed_vars': []
    }
    
    # Check if surrogate is available
    if not hasattr(node.lb_problem, 'wlsq_surrogate') or node.lb_problem.wlsq_surrogate is None:
        return result
    
    surrogate = node.lb_problem.wlsq_surrogate
    Q = surrogate.get('Q')
    c = surrogate.get('c')
    d = surrogate.get('d')
    var_ids = surrogate.get('var_ids')
    
    if Q is None or c is None or d is None or var_ids is None:
        return result
    
    dim = len(var_ids)
    if dim == 0:
        return result
    
    # Ensure Q, c, d are numpy arrays
    Q = np.array(Q, dtype=float)
    c = np.array(c, dtype=float)
    d = float(d)
    
    # Check convexity: H = Q + Q.T should be PSD for phi(x) <= RHS to define a convex feasible set
    H = Q + Q.T
    eigvals = np.linalg.eigvalsh(H)
    min_eig = np.min(eigvals)
    is_convex = min_eig >= -1e-10
    
    if only_convex and not is_convex:
        result['skipped_nonconvex'] = True
        return result
    
    # Root node safety: detach state from shared root_node_state
    if node.id == 0 and not getattr(node, '_state_detached', False):
        node.state = deepcopy(node.state)
        node._state_detached = True
    
    # Get current bounds from node.state
    if SupportedVars.reals not in node.state:
        return result
    
    # Build var_id -> index mapping
    var_id_to_idx = {vid: i for i, vid in enumerate(var_ids)}
    
    # Get list of vars to tighten (those still in to_branch)
    vars_to_tighten = []
    if SupportedVars.reals in node.to_branch:
        for vid in node.to_branch[SupportedVars.reals]:
            if vid in var_id_to_idx and vid in node.state[SupportedVars.reals]:
                vdata = node.state[SupportedVars.reals][vid]
                if not getattr(vdata, 'is_fixed', False):
                    vars_to_tighten.append(vid)
    
    if not vars_to_tighten:
        return result
    
    # Get bounds for all surrogate vars
    bounds = []
    for vid in var_ids:
        if vid in node.state[SupportedVars.reals]:
            vdata = node.state[SupportedVars.reals][vid]
            bounds.append((vdata.lb, vdata.ub))
        else:
            bounds.append((-1e6, 1e6))  # fallback
    
    # Run QCP solves on rank 0 only
    tightened_bounds = {}
    infeasible_flag = False
    
    if rank == 0:
        tightened_bounds, infeasible_flag = _solve_obbt_qcps(
            Q, c, d, var_ids, var_id_to_idx, bounds, vars_to_tighten,
            ub_value, delta, is_convex, solver_name, solver_opts
        )
    
    # Broadcast results to all ranks
    if size > 1:
        MPI.COMM_WORLD.barrier()
        infeasible_flag = MPI.COMM_WORLD.bcast(infeasible_flag, root=0)
        tightened_bounds = MPI.COMM_WORLD.bcast(tightened_bounds, root=0)
    
    if infeasible_flag:
        result['infeasible'] = True
        return result
    
    # Apply tightened bounds to node.state
    max_shrink = 0.0
    tightened_count = 0
    removed_vars = []
    
    for vid, (new_lb, new_ub) in tightened_bounds.items():
        if vid not in node.state[SupportedVars.reals]:
            continue
        
        vdata = node.state[SupportedVars.reals][vid]
        old_lb = vdata.lb
        old_ub = vdata.ub
        old_width = old_ub - old_lb
        
        # Non-expansive: only shrink
        final_lb = max(old_lb, new_lb)
        final_ub = min(old_ub, new_ub)
        
        # Check feasibility
        if final_lb > final_ub + 1e-9:
            result['infeasible'] = True
            return result
        
        # Apply clipping
        final_lb = min(final_lb, final_ub)
        
        # Did we tighten?
        lb_tightened = final_lb > old_lb + 1e-12
        ub_tightened = final_ub < old_ub - 1e-12
        
        if lb_tightened or ub_tightened:
            vdata.lb = final_lb
            vdata.ub = final_ub
            tightened_count += 1
            
            new_width = final_ub - final_lb
            if old_width > 1e-12:
                shrink_ratio = 1.0 - (new_width / old_width)
                max_shrink = max(max_shrink, shrink_ratio)
            
            # Check if var becomes un-branchable
            if new_width <= epsilon:
                if SupportedVars.reals in node.to_branch and vid in node.to_branch[SupportedVars.reals]:
                    node.to_branch[SupportedVars.reals].remove(vid)
                    removed_vars.append(vid)
    
    result['tightened_count'] = tightened_count
    result['max_shrink_ratio'] = max_shrink
    result['removed_vars'] = removed_vars
    
    # Log on rank 0
    if rank == 0 and tightened_count > 0:
        print(f"WLSQ-OBBT tightened {tightened_count} vars; max_shrink={max_shrink:.2%}; skipped_nonconvex={result['skipped_nonconvex']}")
    
    return result


def _solve_obbt_qcps(
    Q, c, d, var_ids, var_id_to_idx, bounds, vars_to_tighten,
    ub_value, delta, is_convex, solver_name, solver_opts
):
    """
    Build Pyomo model and solve min/max QCPs for each var.
    Called on rank 0 only.
    
    Returns (tightened_bounds, infeasible_flag)
    """
    dim = len(var_ids)
    tightened_bounds = {}
    infeasible_flag = False
    
    # Build Pyomo model
    m = pyo.ConcreteModel()
    m.x = pyo.Var(range(dim), bounds=lambda m, i: bounds[i])
    
    # Quadratic constraint: x^T Q x + c^T x + d <= ub_value - delta
    rhs = ub_value - delta
    
    def quad_constraint_rule(m):
        expr = d
        # Linear terms
        for i in range(dim):
            expr += c[i] * m.x[i]
        # Quadratic terms
        for i in range(dim):
            for j in range(dim):
                expr += Q[i, j] * m.x[i] * m.x[j]
        return expr <= rhs
    
    m.quad_con = pyo.Constraint(rule=quad_constraint_rule)
    
    # Configure solver
    solver = SolverFactory(solver_name)
    if not is_convex:
        solver.options['NonConvex'] = 2
    solver.options['TimeLimit'] = 5
    solver.options['OutputFlag'] = 0
    for key, val in solver_opts.items():
        solver.options[key] = val
    
    # Solve min/max for each var
    for vid in vars_to_tighten:
        idx = var_id_to_idx[vid]
        old_lb, old_ub = bounds[idx]
        new_lb = old_lb
        new_ub = old_ub
        
        try:
            # Minimize
            m.obj = pyo.Objective(expr=m.x[idx], sense=pyo.minimize)
            res = solver.solve(m, tee=False)
            if (res.solver.status == pyo.SolverStatus.ok and 
                res.solver.termination_condition in [pyo.TerminationCondition.optimal, 
                                                      pyo.TerminationCondition.locallyOptimal]):
                new_lb = pyo.value(m.x[idx])
            m.del_component(m.obj)
            
            # Maximize
            m.obj = pyo.Objective(expr=m.x[idx], sense=pyo.maximize)
            res = solver.solve(m, tee=False)
            if (res.solver.status == pyo.SolverStatus.ok and 
                res.solver.termination_condition in [pyo.TerminationCondition.optimal,
                                                      pyo.TerminationCondition.locallyOptimal]):
                new_ub = pyo.value(m.x[idx])
            m.del_component(m.obj)
            
        except Exception:
            # On failure, keep original bounds
            new_lb = old_lb
            new_ub = old_ub
        
        # Check for infeasibility from solver (empty feasible region)
        if new_lb > new_ub + 1e-9:
            infeasible_flag = True
            break
        
        tightened_bounds[vid] = (new_lb, new_ub)
    
    return tightened_bounds, infeasible_flag
