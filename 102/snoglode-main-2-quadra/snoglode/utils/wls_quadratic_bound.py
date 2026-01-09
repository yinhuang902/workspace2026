
import numpy as np
from scipy.optimize import minimize
import pyomo.environ as pyo
from snoglode.components.node import Node
from snoglode.components.subproblems import Subproblems
from snoglode.utils.supported import SupportedVars
import snoglode.utils.MPI as MPI
from pyomo.opt import SolverFactory
from snoglode.utils.ms_solver_helper import _solve_ms_dual_bound_gurobi, _solve_true_recourse_primal_gurobi
from typing import Tuple, Optional, List, Dict

def _identify_pid_vars(node: Node, subproblems: Subproblems) -> Optional[List[Dict]]:
    """
    Identifies the 3 PID variables (Kp, Ki, Kd) by name.
    Returns a list of dicts with {'id', 'type', 'lb', 'ub'} in order [Kp, Ki, Kd].
    Returns None if identification fails.
    """
    if not subproblems.names:
        return None
    first_scenario = subproblems.names[0]
    
    # Candidates for each role
    candidates = {'kp': [], 'ki': [], 'kd': []}
    
    # Search patterns (case-insensitive)
    # User requested: name contains "kp" as a whole token, prefer exact token matches.
    # We check if 'kp' or 'k_p' is in the lower-cased name.
    patterns = {
        'kp': ['kp', 'k_p'],
        'ki': ['ki', 'k_i'],
        'kd': ['kd', 'k_d']
    }
    
    for var in subproblems.subproblem_lifted_vars[first_scenario]:
        v_type, v_id, v_stage = subproblems.var_to_data[var]
        
        # Must be in node state
        if v_type not in node.state or v_id not in node.state[v_type]:
            continue
            
        name_lower = var.name.lower()
        
        for role, pats in patterns.items():
            for pat in pats:
                if pat in name_lower:
                    candidates[role].append((var, v_type, v_id, v_stage, name_lower))
                    break
    
    # Select best candidate for each role
    final_vars = []
    for role in ['kp', 'ki', 'kd']:
        role_cands = candidates[role]
        if not role_cands:
            return None
            
        # Filter/Sort
        # 1. Prefer stage == 1
        stage1 = [c for c in role_cands if c[3] == 1]
        if stage1:
            role_cands = stage1
            
        # 2. Shortest name (to prefer "K_p" over "K_p_aux" etc)
        role_cands.sort(key=lambda x: (len(x[4]), x[4]))
        
        best = role_cands[0]
        # Get bounds from node.state
        var_state = node.state[best[1]][best[2]]
        final_vars.append({
            'id': best[2],
            'type': best[1],
            'lb': var_state.lb,
            'ub': var_state.ub
        })
        
    return final_vars

def compute_wls_quadratic_surrogate_bound(
    node: Node, 
    subproblems: Subproblems, 
    solver: SolverFactory, 
    num_samples: int = 100, 
    seed: Optional[int] = None,
    max_retries: int = 1000
) -> float:
    """
    Computes a lower bound using a Weighted Least Squares (WLS) Quadratic Surrogate.
    
    1. Samples 100 UNIQUE, FEASIBLE points inside the node.
    2. Evaluates true recourse for each scenario at these points.
    3. Fits a full quadratic surrogate (10 terms for 3 vars) using WLS (uniform weights) + Ridge.
    4. Computes ms = min(F_s - Q_s) via global optimization.
    5. Computes LB_s = min_box(Q_s) + ms.
    6. Aggregates LB_s across scenarios.
    
    Args:
        seed: Optional seed for reproducible sampling. If provided, uses seed+rank for MPI.
    """
    
    # 1. Identify the box (bounds of lifted variables)
    # ISSUE #1: Select ONLY the three first-stage PID variables (Kp, Ki, Kd) by NAME
    lifted_vars = _identify_pid_vars(node, subproblems)
    if lifted_vars is None:
        return float('-inf')
        
    bounds = []
    for lv in lifted_vars:
        bounds.append((lv['lb'], lv['ub']))
    
    dim = len(lifted_vars)
    if dim != 3:
        return float('-inf')

    # 2. Sample points
    # Create local RNG for reproducibility
    if seed is not None:
        rng_seed = seed + MPI.COMM_WORLD.Get_rank()
        rng = np.random.default_rng(rng_seed)
    else:
        rng = np.random.default_rng()
    
    samples = []
    seen_points = set()
    
    # Save state of ALL subproblems
    # We need to check feasibility for ALL scenarios
    saved_states = {}
    for name in subproblems.names:
        saved_states[name] = {}
        for var in subproblems.subproblem_lifted_vars[name]:
            saved_states[name][id(var)] = (var.lb, var.ub, var.is_fixed(), var.value)
        
    total_attempts = 0
    max_total_attempts = num_samples * max_retries
    
    while len(samples) < num_samples and total_attempts < max_total_attempts:
        total_attempts += 1
        
        # Generate a random point
        point = []
        for i in range(dim):
            lb, ub = bounds[i]
            if lb == float('-inf'): lb = -1e4 
            if ub == float('inf'): ub = 1e4
            
            if lb > ub: 
                # Restore and return failure
                for name in subproblems.names:
                    for var in subproblems.subproblem_lifted_vars[name]:
                        lb_v, ub_v, is_fixed, val = saved_states[name][id(var)]
                        var.lb = lb_v
                        var.ub = ub_v
                        if is_fixed: var.fix(val)
                        else: var.unfix()
                return float('-inf')
            
            val = rng.uniform(lb, ub)
            
            v_type = lifted_vars[i]['type']
            if v_type in [SupportedVars.binary, SupportedVars.integers, SupportedVars.nonnegative_integers]:
                val = round(val)
                val = max(lb, min(ub, val))
            
            point.append(val)
            
        # Check uniqueness
        point_tuple = tuple(np.round(point, 6))
        if point_tuple in seen_points:
            continue
            
        # Test feasibility across ALL scenarios
        # ISSUE #3: Check all scenarios
        all_feasible = True
        
        # ISSUE #2: Use composite keys (type, id)
        id_to_val = {(lifted_vars[i]['type'], lifted_vars[i]['id']): point[i] for i in range(dim)}
        
        for name in subproblems.names:
            model = subproblems.model[name]
            
            # Fix variables for this scenario
            for var in subproblems.subproblem_lifted_vars[name]:
                v_type, v_id, _ = subproblems.var_to_data[var]
                if (v_type, v_id) in id_to_val:
                    var.fix(id_to_val[(v_type, v_id)])
            
            # Solve
            success, _ = _solve_true_recourse_primal_gurobi(model)
            
            # Restore variables for this scenario immediately to be safe? 
            # Or just wait until end of loop? 
            # We need to unfix them before next attempt or before returning.
            # But we are inside the loop, so we should probably just leave them fixed 
            # and restore all at the end of the batch or if we fail.
            # Actually, it's safer to restore them after checking this point 
            # so the model is clean for the next check or next step.
            # But that's expensive. 
            # However, we have saved_states.
            
            if not success:
                all_feasible = False
                break
        
        # Restore ALL variables after the check
        for name in subproblems.names:
            for var in subproblems.subproblem_lifted_vars[name]:
                lb_v, ub_v, is_fixed, val = saved_states[name][id(var)]
                var.lb = lb_v
                var.ub = ub_v
                if is_fixed: var.fix(val)
                else: var.unfix()
                
        if all_feasible:
            samples.append(np.array(point))
            seen_points.add(point_tuple)
            
    if len(samples) < num_samples:
        return float('-inf')
        
    # 3. Build Design Matrix Phi
    # Full quadratic in 3 vars: 1, x1, x2, x3, x1^2, x1x2, x1x3, x2^2, x2x3, x3^2
    Phi = []
    for x in samples:
        row = [1.0]
        # Linear terms
        for i in range(dim):
            row.append(x[i])
        # Quadratic terms (upper triangular)
        for i in range(dim):
            for j in range(i, dim):
                row.append(x[i] * x[j])
        Phi.append(row)
    Phi = np.array(Phi)
    
    # Ridge regularization
    lambda_reg = 1e-8
    n_features = Phi.shape[1]
    reg_matrix = lambda_reg * np.eye(n_features)
    # Don't regularize intercept (first column)
    reg_matrix[0, 0] = 0.0
    
    PhiT_Phi_reg = Phi.T @ Phi + reg_matrix
    
    local_agg_lb = 0.0
    
    # 4. Process each scenario
    for subproblem_name in subproblems.names:
        model = subproblems.model[subproblem_name]
        prob = subproblems.probability[subproblem_name]
        
        Y_vals = []
        
        # Save state
        saved_state = {}
        for var in subproblems.subproblem_lifted_vars[subproblem_name]:
            saved_state[id(var)] = (var.lb, var.ub, var.is_fixed(), var.value)
            
        # Evaluate samples
        for x in samples:
            # ISSUE #2: Use composite keys
            id_to_val = {(lifted_vars[i]['type'], lifted_vars[i]['id']): x[i] for i in range(dim)}
            
            for var in subproblems.subproblem_lifted_vars[subproblem_name]:
                v_type, v_id, _ = subproblems.var_to_data[var]
                if (v_type, v_id) in id_to_val:
                    var.fix(id_to_val[(v_type, v_id)])
            
            # Solve true recourse using strict Gurobi helper
            success, obj_val = _solve_true_recourse_primal_gurobi(model)
            
            if success:
                Y_vals.append(obj_val)
            else:
                Y_vals.append(1e6)
        
        # Restore state
        for var in subproblems.subproblem_lifted_vars[subproblem_name]:
            lb, ub, is_fixed, val = saved_state[id(var)]
            var.lb = lb
            var.ub = ub
            if is_fixed: var.fix(val)
            else: var.unfix()
            
        Y_vals = np.array(Y_vals)
        
        # Fit coefficients
        try:
            beta = np.linalg.solve(PhiT_Phi_reg, Phi.T @ Y_vals)
        except np.linalg.LinAlgError:
            return float('-inf')
            
        # Extract coefficients
        b_s = beta[0]
        c_s = beta[1:dim+1]
        quad_coeffs = beta[dim+1:]
        
        Q_s = np.zeros((dim, dim))
        idx = 0
        for i in range(dim):
            for j in range(i, dim):
                val = quad_coeffs[idx]
                if i == j:
                    Q_s[i, i] = val
                else:
                    Q_s[i, j] = val / 2.0
                    Q_s[j, i] = val / 2.0
                idx += 1
                
        # --- Build Pyomo surrogate expression for ms computation ---
        # ISSUE #2: Use composite keys for map
        var_map = {}
        for var in subproblems.subproblem_lifted_vars[subproblem_name]:
            v_type, v_id, _ = subproblems.var_to_data[var]
            var_map[(v_type, v_id)] = var
        
        pyomo_vars = []
        for lv in lifted_vars:
            pvar = var_map.get((lv['type'], lv['id']))
            if pvar is None:
                return float('-inf')
            pyomo_vars.append(pvar)
            
        # Build Q_s surrogate expression: x^T Q x + c^T x + b
        Q_expr = b_s
        for i in range(dim):
            Q_expr += c_s[i] * pyomo_vars[i]
            Q_expr += Q_s[i, i] * pyomo_vars[i] * pyomo_vars[i]  # diagonal
            for j in range(i+1, dim):
                Q_expr += 2.0 * Q_s[i, j] * pyomo_vars[i] * pyomo_vars[j]
                
        # --- Compute ms = min(F_s - Q_s) via optimization ---
        orig_obj_component = model.component_data_objects(pyo.Objective, active=True).__next__()
        orig_obj_expr = orig_obj_component.expr
        
        ms_val = float('-inf')
        try:
            orig_obj_component.deactivate()
            model.temp_ms_obj = pyo.Objective(expr=orig_obj_expr - Q_expr, sense=pyo.minimize)
            
            # Use strict Gurobi helper
            ms_val = _solve_ms_dual_bound_gurobi(model, 'temp_ms_obj')
            
        except Exception:
            ms_val = float('-inf')
        finally:
            if hasattr(model, 'temp_ms_obj'):
                model.del_component(model.temp_ms_obj)
            orig_obj_component.activate()
            
        if ms_val == float('-inf'):
            return float('-inf')
            
        # --- Compute min_box(Q_s) using scipy ---
        scipy_bounds = []
        for lb, ub in bounds:
            l = lb if lb != float('-inf') else -1e4
            u = ub if ub != float('inf') else 1e4
            scipy_bounds.append((l, u))
            
        x0_list = [(b[0] + b[1]) / 2.0 for b in scipy_bounds]
        x0 = np.array(x0_list)
        
        def make_objective(Q, c, b):
            def objective(x):
                return x @ Q @ x + c @ x + float(b)
            return objective
        
        def make_jacobian(Q, c):
            def jacobian(x):
                return 2 * Q @ x + c
            return jacobian
            
        res_min_q = minimize(make_objective(Q_s, c_s, b_s), x0, method='L-BFGS-B', 
                             jac=make_jacobian(Q_s, c_s), bounds=scipy_bounds)
        min_Q_s = res_min_q.fun
        
        # LB_s = min(Q_s) + ms
        lb_s = min_Q_s + ms_val
        local_agg_lb += prob * lb_s
        
    # Aggregate across ranks
    local_lb_arr = np.array([local_agg_lb])
    global_lb = MPI.COMM_WORLD.allreduce(local_lb_arr, op=MPI.SUM)[0]
    
    return global_lb
