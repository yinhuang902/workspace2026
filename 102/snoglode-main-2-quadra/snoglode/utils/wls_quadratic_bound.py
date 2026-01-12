
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

# WLSQ Method B Tuning Parameters
WLSQ_NUM_SAMPLES_DEFAULT = 500  # Default number of samples per iteration
WLSQ_B_ALPHA = 5.0          # kernel decay alpha for Method B
WLSQ_B_EPS = 0.1            # weight floor epsilon for Method B
WLSQ_B_DIAM2_TINY_THRESH = 1e-9   # threshold for "tiny box" fallback (sets diam2=1.0)
WLSQ_B_DIAM2_SAFE_MIN = 1e-12     # safe denominator threshold for kernel distance

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
    num_samples: int = WLSQ_NUM_SAMPLES_DEFAULT, 
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
                else: 
                    var.unfix()
                    var.value = val
                
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
    local_agg_lb_uniform = 0.0
    local_agg_lb_A = 0.0
    local_agg_lb_B = 0.0
    
    # Pre-scale samples for weighting (Method A)
    # Normalize samples to z in [-1, 1] for weight calculation
    samples = np.array(samples)
    # X_norm [i, j]
    X_norm = np.zeros_like(samples)
    for j in range(dim):
        lb, ub = bounds[j]
        # Handle inf bounds for normalization safety (though samples are finite)
        l = lb if lb != float('-inf') else -1e4
        u = ub if ub != float('inf') else 1e4
        width = u - l
        if width < 1e-9:
             X_norm[:, j] = 0.0
        else:
             # Map [l, u] -> [-1, 1]
             # val = 2*(x - l)/(u -l) - 1
             X_norm[:, j] = 2.0 * (samples[:, j] - l) / width - 1.0
             
    # r2 for center weight
    r2 = np.sum(X_norm**2, axis=1) # (N,)
    w_center = np.exp(-2.0 * r2)   # alpha=2.0
    
    # Calculate box diameter squared for Method B
    diam2 = 0.0
    for j in range(dim):
        lb, ub = bounds[j]
        l = lb if lb != float('-inf') else -1e4
        u = ub if ub != float('inf') else 1e4
        diam2 += (u - l)**2
    if diam2 < WLSQ_B_DIAM2_TINY_THRESH: diam2 = 1.0

    # --- Method B: probability-weighted mixture of kernels over all scenario anchors ---
    anchors = []
    probs = []
    for name in subproblems.names:
        if hasattr(node.lb_problem, 'subproblem_solutions') and name in node.lb_problem.subproblem_solutions:
            sol = node.lb_problem.subproblem_solutions[name].lifted_var_solution
            # Extract x_star = (Kp, Ki, Kd)
            x_star = []
            valid = True
            for lv in lifted_vars:
                 val = sol.get(lv['type'], {}).get(lv['id'])
                 if val is None:
                     valid = False
                     break
                 x_star.append(val)
            
            if valid:
                anchors.append(x_star)
                probs.append(subproblems.probability[name])
    
    w_B_shared = np.ones(num_samples)
    if anchors:
        anchors = np.array(anchors)
        probs = np.array(probs)
        prob_sum = np.sum(probs)
        if prob_sum > 0:
            probs = probs / prob_sum
        else:
             probs = np.ones(len(probs)) / len(probs)
             
        diam2_safe = max(diam2, WLSQ_B_DIAM2_SAFE_MIN)
        
        # d2_it = ||samples[i]-anchors[t]||^2 / diam2_safe
        diffs = samples[:, np.newaxis, :] - anchors[np.newaxis, :, :]
        dists2 = np.sum(diffs**2, axis=2)
        d2_it = dists2 / diam2_safe
        
        alpha = WLSQ_B_ALPHA
        kernel_it = np.exp(-alpha * d2_it)
        mix_i = kernel_it @ probs
        
        eps = WLSQ_B_EPS
        w_B_shared = eps + (1.0 - eps) * mix_i

    for subproblem_name in subproblems.names:
        model = subproblems.model[subproblem_name]
        prob = subproblems.probability[subproblem_name]
        
        Y_vals = []
        valid_indices = [] # Track valid indices for stats
        
        # Save state
        saved_state = {}
        for var in subproblems.subproblem_lifted_vars[subproblem_name]:
            saved_state[id(var)] = (var.lb, var.ub, var.is_fixed(), var.value)
            
        # Evaluate samples
        for i_s, x in enumerate(samples):
            # Use composite keys
            id_to_val = {(lifted_vars[i]['type'], lifted_vars[i]['id']): x[i] for i in range(dim)}
            
            for var in subproblems.subproblem_lifted_vars[subproblem_name]:
                v_type, v_id, _ = subproblems.var_to_data[var]
                if (v_type, v_id) in id_to_val:
                    var.fix(id_to_val[(v_type, v_id)])
            
            # Solve true recourse using strict Gurobi helper
            success, obj_val = _solve_true_recourse_primal_gurobi(model)
            
            if success:
                Y_vals.append(obj_val)
                valid_indices.append(i_s)
            else:
                Y_vals.append(1e6)
        
        # Restore state
        for var in subproblems.subproblem_lifted_vars[subproblem_name]:
            lb, ub, is_fixed, val = saved_state[id(var)]
            var.lb = lb
            var.ub = ub
            if is_fixed: var.fix(val)
            else: 
                var.unfix()
                var.value = val
            
        Y_vals = np.array(Y_vals)
        
        # --- Calculate Weights (Method A) ---
        w_A = np.ones(num_samples)
        
        if len(valid_indices) > 0:
            Y_valid = Y_vals[valid_indices]
            # Median and IQR
            y_ref = np.median(Y_valid)
            iqr = np.percentile(Y_valid, 75) - np.percentile(Y_valid, 25)
            tau = max(iqr, 1e-6)
            
            # Sigmoid Low-Value Preference
            # s_i = 1 / (1 + exp(-(y_ref - y_i)/tau))
            # compute for ALL samples (Y_vals), though failed ones (1e6) will have s_i ~ 0
            # overflow protection for exp
            arg = -(y_ref - Y_vals) / tau
            # clip arg to avoid overflow
            arg = np.clip(arg, -50, 50)
            s = 1.0 / (1.0 + np.exp(arg))
            
            w_low = 1.0 + 2.0 * s # eta=2.0
            
            # Combine
            w_raw = w_center * w_low
            w_A = np.maximum(w_raw, 0.1)
            w_A = np.minimum(w_A, 10.0) # Safety cap
        else:
            w_A = np.ones(num_samples) # Fallback if no valid points
            
        # --- Calculate Weights (Method B) ---
        w_B = w_B_shared

        # --- Fit & Bound: Uniform ---
        # Same as original: beta = (Phi^T Phi + lam I)^-1 Phi^T Y
        try:
            beta_uni = np.linalg.solve(PhiT_Phi_reg, Phi.T @ Y_vals)
            lb_uni = _compute_bound_from_beta(beta_uni, dim, lifted_vars, subproblems, subproblem_name, bounds, model)
        except:
            lb_uni = float('-inf')
        
        local_agg_lb_uniform += prob * lb_uni

        # --- Fit & Bound: Method A (Weighted) ---
        try:
            # Weighted Least Squares
            # sqrt(W) scaling
            S = np.sqrt(w_A) # (N,)
            # Phi_w = S[:, None] * Phi
            Phi_w = Phi * S[:, np.newaxis]
            Y_w = Y_vals * S
            
            # Ridge on weighted data
            PhiT_Phi_w = Phi_w.T @ Phi_w + reg_matrix
            beta_A = np.linalg.solve(PhiT_Phi_w, Phi_w.T @ Y_w)
            
            lb_A = _compute_bound_from_beta(beta_A, dim, lifted_vars, subproblems, subproblem_name, bounds, model)
        except:
            lb_A = float('-inf')
            
        local_agg_lb_A += prob * lb_A
        
        # --- Fit & Bound: Method B (Weighted) ---
        try:
            # Weighted Least Squares
            S_B = np.sqrt(w_B)
            Phi_w_B = Phi * S_B[:, np.newaxis]
            Y_w_B = Y_vals * S_B
            
            PhiT_Phi_w_B = Phi_w_B.T @ Phi_w_B + reg_matrix
            beta_B = np.linalg.solve(PhiT_Phi_w_B, Phi_w_B.T @ Y_w_B)
            
            lb_B = _compute_bound_from_beta(beta_B, dim, lifted_vars, subproblems, subproblem_name, bounds, model)
        except:
            lb_B = float('-inf')
            
        local_agg_lb_B += prob * lb_B
        
    # Aggregate across ranks
    local_arr = np.array([local_agg_lb_uniform, local_agg_lb_A, local_agg_lb_B])
    global_arr = MPI.COMM_WORLD.allreduce(local_arr, op=MPI.SUM)
    
    global_lb_uniform = global_arr[0]
    global_lb_A = global_arr[1]
    global_lb_B = global_arr[2]
    
    # Store on node
    node.lb_problem.wlsq_uniform_bound = global_lb_uniform
    node.lb_problem.wlsq_A_bound = global_lb_A
    node.lb_problem.wlsq_B_bound = global_lb_B
    
    return max(global_lb_uniform, global_lb_A, global_lb_B)

def _compute_bound_from_beta(beta, dim, lifted_vars, subproblems, subproblem_name, bounds, model):
    """
    Helper to compute LB = min_box(Q) + ms given regression coefficients beta.
    Reuses the logic from the massive loop to avoid duplication.
    """
    try:
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
        return lb_s
    except:
        return float('-inf')
