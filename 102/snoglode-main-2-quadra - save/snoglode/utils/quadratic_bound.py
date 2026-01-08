
import numpy as np
from scipy.optimize import minimize
import pyomo.environ as pyo
from snoglode.components.node import Node
from snoglode.components.subproblems import Subproblems
from snoglode.utils.supported import SupportedVars
import snoglode.utils.MPI as MPI
from pyomo.opt import SolverFactory

from typing import Tuple

def compute_quadratic_surrogate_bound(node: Node, subproblems: Subproblems, solver: SolverFactory, num_samples: int = 10) -> Tuple[float, float]:
    """
    Computes an experimental lower bound using a quadratic surrogate.
    
    1. Samples points inside the node.
    2. Evaluates true recourse for each scenario.
    3. Fits a quadratic surrogate for each scenario.
    4. Computes ms = min(F_s - Q_s) via global optimization (NonConvex=2).
    5. Computes LB_s = min_box(Q_s) + ms.
    6. Aggregates LB_s across scenarios.
    """
    
    # 1. Identify the box (bounds of lifted variables)
    lifted_vars = []
    bounds = []
    
    # Iterate through supported var types to collect all lifted vars in a deterministic order
    for var_type in SupportedVars:
        if var_type in node.state:
            # Sort by ID to ensure consistency
            for var_id in sorted(node.state[var_type].keys()):
                var_state = node.state[var_type][var_id]
                lifted_vars.append({
                    'id': var_id,
                    'type': var_type,
                    'lb': var_state.lb,
                    'ub': var_state.ub
                })
                bounds.append((var_state.lb, var_state.ub))
    
    dim = len(lifted_vars)
    if dim == 0:
        return float('-inf'), float('-inf')

    # 2. Sample points
    samples = []
    if num_samples < 1: num_samples = 1
    
    for _ in range(num_samples):
        point = []
        for i in range(dim):
            lb, ub = bounds[i]
            if lb == float('-inf'): lb = -1e4 
            if ub == float('inf'): ub = 1e4
            
            if lb > ub: 
                return float('inf'), float('inf')
            
            val = np.random.uniform(lb, ub)
            
            v_type = lifted_vars[i]['type']
            if v_type in [SupportedVars.binary, SupportedVars.integers, SupportedVars.nonnegative_integers]:
                val = round(val)
                val = max(lb, min(ub, val))
            
            point.append(val)
        samples.append(np.array(point))
    
    # 3. Determine fitting mode based on sample count
    n_params_full = dim * (dim + 1) // 2 + dim + 1
    n_params_diag = 2 * dim + 1
    
    mode = 'linear'
    if num_samples >= n_params_full:
        mode = 'full'
    elif num_samples >= n_params_diag:
        mode = 'diag'
        
    # Build design matrix X_mat
    X_mat = []
    for x in samples:
        row = []
        if mode == 'full':
            for i in range(dim):
                for j in range(i, dim):
                    row.append(x[i] * x[j])
        if mode == 'full' or mode == 'diag':
            if mode == 'diag':
                for i in range(dim):
                    row.append(x[i]**2)
        
        for i in range(dim):
            row.append(x[i])
        
        row.append(1.0)
        X_mat.append(row)
    
    X_mat = np.array(X_mat)
    
    # Bounds for scipy optimizer
    scipy_bounds = []
    for lb, ub in bounds:
        l = lb if lb != float('-inf') else -1e4
        u = ub if ub != float('inf') else 1e4
        scipy_bounds.append((l, u))
        
    def make_objective(Q, c, b):
        def objective(x):
            return x @ Q @ x + c @ x + float(b)
        return objective
    
    def make_jacobian(Q, c):
        def jacobian(x):
            return 2 * Q @ x + c
        return jacobian

    # Initial guess: center of box
    x0_list = []
    for lb, ub in bounds:
        if lb == float('-inf'): lb = -10
        if ub == float('inf'): ub = 10
        x0_list.append((lb + ub) / 2.0)
    x0 = np.array(x0_list)
    
    local_agg_lb = 0.0       # For regular quadratic bound
    local_agg_lb_spec = 0.0  # For spectral quadratic bound
    
    for subproblem_name in subproblems.names:
        model = subproblems.model[subproblem_name]
        prob = subproblems.probability[subproblem_name]
        
        Y_vals = []
        
        # Save state of all variables before fixing
        saved_state = {}
        for var in subproblems.subproblem_lifted_vars[subproblem_name]:
            saved_state[id(var)] = (var.lb, var.ub, var.is_fixed(), var.value)

        for x in samples:
            id_to_val = {lifted_vars[i]['id']: x[i] for i in range(dim)}
            
            for var in subproblems.subproblem_lifted_vars[subproblem_name]:
                _, var_id, _ = subproblems.var_to_data[var]
                if var_id in id_to_val:
                    var.fix(id_to_val[var_id])
            
            results = solver.solve(model, load_solutions=False)
            
            if (results.solver.termination_condition == pyo.TerminationCondition.optimal or
                results.solver.termination_condition == pyo.TerminationCondition.locallyOptimal):
                model.solutions.load_from(results)
                
                try:
                    obj_val = results.problem[0].lower_bound
                    if obj_val is None or obj_val == float('-inf') or obj_val == float('inf'):
                         obj_val = pyo.value(model.component_data_objects(pyo.Objective, active=True).__next__())
                except:
                     obj_val = pyo.value(model.component_data_objects(pyo.Objective, active=True).__next__())
                
                Y_vals.append(obj_val)
            else:
                Y_vals.append(1e6) 
        
        # Restore state of all variables
        for var in subproblems.subproblem_lifted_vars[subproblem_name]:
            lb, ub, is_fixed, val = saved_state[id(var)]
            var.lb = lb
            var.ub = ub
            if is_fixed:
                var.fix(val)
            else:
                var.unfix()

        Y_vals = np.array(Y_vals)
        
        # Fit quadratic surrogate
        if len(Y_vals) == 0:
            continue
            
        coeffs, _, _, _ = np.linalg.lstsq(X_mat, Y_vals, rcond=None)
        
        # Reconstruct Q, c, b from coeffs
        Q_s = np.zeros((dim, dim))
        c_s = np.zeros(dim)
        b_s = 0.0
        
        idx = 0
        if mode == 'full':
            for i in range(dim):
                for j in range(i, dim):
                    val = coeffs[idx]
                    if i == j:
                        Q_s[i, i] = val
                    else:
                        Q_s[i, j] = val / 2.0
                        Q_s[j, i] = val / 2.0
                    idx += 1
        elif mode == 'diag':
            for i in range(dim):
                Q_s[i, i] = coeffs[idx]
                idx += 1
        
        for i in range(dim):
            c_s[i] = coeffs[idx]
            idx += 1
        
        b_s = coeffs[idx]
        
        # --- Build Pyomo surrogate expression for ms computation ---
        # Find Pyomo variables for this subproblem
        var_map = {}
        for var in subproblems.subproblem_lifted_vars[subproblem_name]:
            _, var_id, _ = subproblems.var_to_data[var]
            var_map[var_id] = var
        
        # Get ordered list of Pyomo vars matching lifted_vars order
        pyomo_vars = []
        for lv in lifted_vars:
            pvar = var_map.get(lv['id'])
            if pvar is None:
                # Cannot build surrogate expression without all vars
                return float('-inf'), float('-inf')
            pyomo_vars.append(pvar)
        
        # Build Q_s surrogate expression: x^T Q x + c^T x + b
        Q_expr = b_s
        for i in range(dim):
            Q_expr += c_s[i] * pyomo_vars[i]
            for j in range(dim):
                Q_expr += Q_s[i, j] * pyomo_vars[i] * pyomo_vars[j]
        
        # --- Compute ms = min(F_s - Q_s) via optimization ---
        orig_obj_component = model.component_data_objects(pyo.Objective, active=True).__next__()
        orig_obj_expr = orig_obj_component.expr
        
        ms_val = float('-inf')
        try:
            orig_obj_component.deactivate()
            model.temp_ms_obj = pyo.Objective(expr=orig_obj_expr - Q_expr, sense=pyo.minimize)
            
            res_ms = solver.solve(model, load_solutions=False, options={"NonConvex": 2})
            if (res_ms.solver.termination_condition == pyo.TerminationCondition.optimal or
                res_ms.solver.termination_condition == pyo.TerminationCondition.locallyOptimal):
                try:
                    ms_val = res_ms.problem[0].lower_bound
                    if ms_val is None or ms_val == float('-inf') or ms_val == float('inf'):
                        model.solutions.load_from(res_ms)
                        ms_val = pyo.value(model.temp_ms_obj)
                except:
                    model.solutions.load_from(res_ms)
                    ms_val = pyo.value(model.temp_ms_obj)
            else:
                ms_val = float('-inf')
        except Exception as e:
            ms_val = float('-inf')
        finally:
            # Always restore
            if hasattr(model, 'temp_ms_obj'):
                model.del_component(model.temp_ms_obj)
            orig_obj_component.activate()
        
        if ms_val == float('-inf'):
            return float('-inf'), float('-inf')
        
        # --- Compute min_box(Q_s) using scipy ---
        res_min_q = minimize(make_objective(Q_s, c_s, b_s), x0, method='L-BFGS-B', 
                             jac=make_jacobian(Q_s, c_s), bounds=scipy_bounds)
        min_Q_s = res_min_q.fun
        
        # LB_s = min(Q_s) + ms
        lb_s = min_Q_s + ms_val
        local_agg_lb += prob * lb_s
        
        # --- Spectral-corrected quadratic surrogate ---
        eig_vals, eig_vecs = np.linalg.eigh(Q_s)
        eig_vals[eig_vals < 0] = 0.0
        Q_spec = eig_vecs @ np.diag(eig_vals) @ eig_vecs.T
        
        # Build spectral surrogate Pyomo expression
        Q_spec_expr = b_s
        for i in range(dim):
            Q_spec_expr += c_s[i] * pyomo_vars[i]
            for j in range(dim):
                Q_spec_expr += Q_spec[i, j] * pyomo_vars[i] * pyomo_vars[j]
        
        # Compute ms_spec = min(F_s - Q_spec) via optimization
        ms_spec_val = float('-inf')
        try:
            orig_obj_component.deactivate()
            model.temp_ms_spec_obj = pyo.Objective(expr=orig_obj_expr - Q_spec_expr, sense=pyo.minimize)
            
            res_ms_spec = solver.solve(model, load_solutions=False, options={"NonConvex": 2})
            if (res_ms_spec.solver.termination_condition == pyo.TerminationCondition.optimal or
                res_ms_spec.solver.termination_condition == pyo.TerminationCondition.locallyOptimal):
                try:
                    ms_spec_val = res_ms_spec.problem[0].lower_bound
                    if ms_spec_val is None or ms_spec_val == float('-inf') or ms_spec_val == float('inf'):
                        model.solutions.load_from(res_ms_spec)
                        ms_spec_val = pyo.value(model.temp_ms_spec_obj)
                except:
                    model.solutions.load_from(res_ms_spec)
                    ms_spec_val = pyo.value(model.temp_ms_spec_obj)
            else:
                ms_spec_val = float('-inf')
        except Exception as e:
            ms_spec_val = float('-inf')
        finally:
            if hasattr(model, 'temp_ms_spec_obj'):
                model.del_component(model.temp_ms_spec_obj)
            orig_obj_component.activate()
        
        if ms_spec_val == float('-inf'):
            return float('-inf'), float('-inf')
        
        # min_box(Q_spec)
        res_min_q_spec = minimize(make_objective(Q_spec, c_s, b_s), x0, method='L-BFGS-B',
                                  jac=make_jacobian(Q_spec, c_s), bounds=scipy_bounds)
        min_Q_spec = res_min_q_spec.fun
        
        # LB_s_spec = min(Q_spec) + ms_spec
        lb_s_spec = min_Q_spec + ms_spec_val
        local_agg_lb_spec += prob * lb_s_spec

    # Aggregate across ranks
    local_lb_arr = np.array([local_agg_lb])
    global_lb = MPI.COMM_WORLD.allreduce(local_lb_arr, op=MPI.SUM)[0]
    
    local_lb_spec_arr = np.array([local_agg_lb_spec])
    global_lb_spec = MPI.COMM_WORLD.allreduce(local_lb_spec_arr, op=MPI.SUM)[0]

    return global_lb, global_lb_spec

def compute_random_pid_bound(node: Node, subproblems: Subproblems, solver: SolverFactory, max_retries: int = 100) -> float:
    """
    Computes an independent lower bound using a PI-D separated quadratic surrogate.
    
    1. Samples 9 points.
    2. Fits a PI-D separated quadratic surrogate (8 terms).
       (Kp, Ki): coupled 2D quadratic.
       Kd: independent 1D quadratic.
    3. Computes ms = min (F_s - Q_s) via global optimization.
    4. Computes LB = min (Q_s) + ms.
    """
    
    # 1. Identify variables and box
    lifted_vars = []
    pid_indices = {'K_p': -1, 'K_i': -1, 'K_d': -1}
    
    # Iterate through supported var types to collect all lifted vars in a deterministic order
    idx = 0
    for var_type in SupportedVars:
        if var_type in node.state:
            for var_id in sorted(node.state[var_type].keys()):
                var_state = node.state[var_type][var_id]
                lifted_vars.append({
                    'id': var_id,
                    'type': var_type,
                    'lb': var_state.lb,
                    'ub': var_state.ub
                })
                
                # Check if this is one of our PID vars
                if var_id in pid_indices:
                    pid_indices[var_id] = idx
                
                idx += 1
    
    # If we didn't find all PID vars, we can't use this specific structure
    if any(v == -1 for v in pid_indices.values()):
        return float('-inf')

    dim = len(lifted_vars)
    bounds = [(v['lb'], v['ub']) for v in lifted_vars]
    
    # 2. Sample 9 points
    num_samples = 9
    samples = []
    
    for _ in range(num_samples):
        for _ in range(max_retries):
            point = []
            for i in range(dim):
                lb, ub = bounds[i]
                if lb == float('-inf'): lb = -1e4
                if ub == float('inf'): ub = 1e4
                
                if lb > ub: return float('-inf') 
                
                val = np.random.uniform(lb, ub)
                
                # Enforce integrality
                v_type = lifted_vars[i]['type']
                if v_type in [SupportedVars.binary, SupportedVars.integers, SupportedVars.nonnegative_integers]:
                    val = round(val)
                    val = max(lb, min(ub, val))
                
                point.append(val)
            
            samples.append(np.array(point))
            break
        else:
            # Failed to find sample after retries
            return float('-inf')

    if len(samples) < num_samples:
        return float('-inf')

    # Indices for fitting
    idx_p = pid_indices['K_p']
    idx_i = pid_indices['K_i']
    idx_d = pid_indices['K_d']
    
    local_agg_lb = 0.0
    
    # Design matrix X (N x 8)
    # Terms: Kp^2, Ki^2, Kp*Ki, Kp, Ki, Kd^2, Kd, 1
    X_mat = []
    for x in samples:
        kp = x[idx_p]
        ki = x[idx_i]
        kd = x[idx_d]
        row = [kp**2, ki**2, kp*ki, kp, ki, kd**2, kd, 1.0]
        X_mat.append(row)
    X_mat = np.array(X_mat)
    
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
            id_to_val = {lifted_vars[i]['id']: x[i] for i in range(dim)}
            for var in subproblems.subproblem_lifted_vars[subproblem_name]:
                _, var_id, _ = subproblems.var_to_data[var]
                if var_id in id_to_val:
                    var.fix(id_to_val[var_id])
            
            results = solver.solve(model, load_solutions=False)
            if (results.solver.termination_condition == pyo.TerminationCondition.optimal or
                results.solver.termination_condition == pyo.TerminationCondition.locallyOptimal):
                model.solutions.load_from(results)
                try:
                    obj_val = results.problem[0].lower_bound
                    if obj_val is None or obj_val == float('-inf') or obj_val == float('inf'):
                         obj_val = pyo.value(model.component_data_objects(pyo.Objective, active=True).__next__())
                except:
                     obj_val = pyo.value(model.component_data_objects(pyo.Objective, active=True).__next__())
                
                Y_vals.append(obj_val)
            else:
                # Infeasible sample, use penalty
                Y_vals.append(1e6)
        
        # Restore state
        for var in subproblems.subproblem_lifted_vars[subproblem_name]:
            lb, ub, is_fixed, val = saved_state[id(var)]
            var.lb = lb
            var.ub = ub
            if is_fixed: var.fix(val)
            else: var.unfix()
            
        Y_vals = np.array(Y_vals)
        
        # Fit
        coeffs, _, _, _ = np.linalg.lstsq(X_mat, Y_vals, rcond=None)
        
        # coeffs: [c_p2, c_i2, c_pi, c_p, c_i, c_d2, c_d, c_bias]
        c_p2, c_i2, c_pi, c_p, c_i, c_d2, c_d, c_bias = coeffs
        
        # 4. Compute ms = min (F_s - Q_s) via Global Optimization
        
        # Identify the Pyomo variables for Kp, Ki, Kd in this subproblem
        var_map = {}
        for var in subproblems.subproblem_lifted_vars[subproblem_name]:
            _, var_id, _ = subproblems.var_to_data[var]
            var_map[var_id] = var
            
        Kp_var = var_map.get('K_p')
        Ki_var = var_map.get('K_i')
        Kd_var = var_map.get('K_d')
        
        if Kp_var is None or Ki_var is None or Kd_var is None:
            return float('-inf')
            
        # Construct Q_s expression
        Q_expr = (c_p2 * Kp_var**2 + 
                  c_i2 * Ki_var**2 + 
                  c_pi * Kp_var * Ki_var + 
                  c_p * Kp_var + 
                  c_i * Ki_var + 
                  c_d2 * Kd_var**2 + 
                  c_d * Kd_var + 
                  c_bias)
        
        # Get original objective
        orig_obj_component = model.component_data_objects(pyo.Objective, active=True).__next__()
        orig_obj_expr = orig_obj_component.expr
        
        # Create temporary objective: min (F - Q)
        orig_obj_component.deactivate()
        model.temp_ms_obj = pyo.Objective(expr = orig_obj_expr - Q_expr, sense=pyo.minimize)
        
        # Solve globally
        ms_val = float('-inf')
        try:
            # Explicitly pass NonConvex=2 for Gurobi
            res_ms = solver.solve(model, load_solutions=False, options={"NonConvex": 2})
            if (res_ms.solver.termination_condition == pyo.TerminationCondition.optimal or
                res_ms.solver.termination_condition == pyo.TerminationCondition.locallyOptimal):
                try:
                    ms_val = res_ms.problem[0].lower_bound
                    if ms_val is None or ms_val == float('-inf') or ms_val == float('inf'):
                         ms_val = pyo.value(model.temp_ms_obj)
                except:
                     ms_val = pyo.value(model.temp_ms_obj)
            else:
                ms_val = float('inf') 
        except Exception as e:
            ms_val = float('-inf') 
            
        # Restore model
        model.del_component(model.temp_ms_obj)
        orig_obj_component.activate()
        
        if ms_val == float('inf'):
            return float('inf')
        if ms_val == float('-inf'):
            return float('-inf')

        # 5. Compute min(Q_s) over the box
        def q_func(x):
            kp, ki, kd = x
            return (c_p2 * kp**2 + c_i2 * ki**2 + c_pi * kp * ki + c_p * kp + c_i * ki + 
                    c_d2 * kd**2 + c_d * kd + c_bias)
        
        # Bounds for Kp, Ki, Kd
        b_p = bounds[idx_p]
        b_i = bounds[idx_i]
        b_d = bounds[idx_d]
        
        def fix_b(b):
            l, u = b
            if l == float('-inf'): l = -1e4
            if u == float('inf'): u = 1e4
            return (l, u)
            
        box_bounds = [fix_b(b_p), fix_b(b_i), fix_b(b_d)]
        x0 = [0.5*(b[0]+b[1]) for b in box_bounds]
        
        res_min_q = minimize(q_func, x0, bounds=box_bounds, method='L-BFGS-B')
        min_q_val = res_min_q.fun
        
        # LB_s = min(Q_s) + ms
        lb_s = min_q_val + ms_val
        
        local_agg_lb += prob * lb_s

    # Aggregate across ranks
    local_lb_arr = np.array([local_agg_lb])
    global_lb = MPI.COMM_WORLD.allreduce(local_lb_arr, op=MPI.SUM)[0]
    
    return global_lb
