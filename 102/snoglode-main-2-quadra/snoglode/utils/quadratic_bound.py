
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
    4. Shifts the surrogate to be an underestimator (based on samples).
    5. Aggregates surrogates.
    6. Minimizes the aggregated surrogate over the node box.
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
    # Use random uniform.
    samples = []
    # Ensure we have at least one sample if num_samples > 0
    if num_samples < 1: num_samples = 1
    
    for _ in range(num_samples):
        point = []
        for i in range(dim):
            lb, ub = bounds[i]
            # Handle infinite bounds
            if lb == float('-inf'): lb = -1e4 
            if ub == float('inf'): ub = 1e4
            
            # Avoid sampling from empty range
            if lb > ub: 
                # Infeasible node, should have been caught earlier
                return float('inf'), float('inf')
            
            val = np.random.uniform(lb, ub)
            
            # Enforce integrality if needed
            v_type = lifted_vars[i]['type']
            if v_type in [SupportedVars.binary, SupportedVars.integers, SupportedVars.nonnegative_integers]:
                val = round(val)
                val = max(lb, min(ub, val))
            
            point.append(val)
        samples.append(np.array(point))
    
    # 3. Evaluate recourse & Fit
    local_agg_Q = np.zeros((dim, dim))
    local_agg_c = np.zeros(dim)
    local_agg_b = 0.0
    
    # Prepare feature matrix for fitting
    # Decide on model complexity based on N and dim
    # Full Quadratic: d(d+1)/2 + d + 1 params
    # Diagonal Quadratic: d + d + 1 = 2d + 1 params
    # Linear: d + 1 params
    
    n_params_full = dim * (dim + 1) // 2 + dim + 1
    n_params_diag = 2 * dim + 1
    n_params_lin = dim + 1
    
    mode = 'linear'
    if num_samples >= n_params_full:
        mode = 'full'
    elif num_samples >= n_params_diag:
        mode = 'diag'
    else:
        mode = 'linear'
        
    # Build design matrix X_mat
    X_mat = []
    for x in samples:
        row = []
        if mode == 'full':
            # Quadratic terms (upper triangle)
            for i in range(dim):
                for j in range(i, dim):
                    row.append(x[i] * x[j])
        if mode == 'full' or mode == 'diag':
            if mode == 'diag':
                # Diagonal quadratic terms only
                for i in range(dim):
                    row.append(x[i]**2)
        
        # Linear terms
        for i in range(dim):
            row.append(x[i])
        
        # Bias
        row.append(1.0)
        X_mat.append(row)
    
    X_mat = np.array(X_mat)
    
    # Pre-compute pseudo-inverse if possible, or use lstsq
    # X_mat is (N, n_params)
    
    # Bounds for optimizer
    # scipy requires (min, max) tuples
    scipy_bounds = []
    for lb, ub in bounds:
        l = lb if lb != float('-inf') else None
        u = ub if ub != float('inf') else None
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
    
    local_agg_b_spec = 0.0
    
    for subproblem_name in subproblems.names:
        model = subproblems.model[subproblem_name]
        prob = subproblems.probability[subproblem_name]
        
        b_s_raw = 0.0 # Safety initialization
        Y_vals = []
        
        # Save state of all variables before fixing
        saved_state = {}
        for var in subproblems.subproblem_lifted_vars[subproblem_name]:
            # Store (lb, ub, is_fixed, value)
            saved_state[id(var)] = (var.lb, var.ub, var.is_fixed(), var.value)

        for x in samples:
            # Fix variables to sample point
            id_to_val = {lifted_vars[i]['id']: x[i] for i in range(dim)}
            
            for var in subproblems.subproblem_lifted_vars[subproblem_name]:
                _, var_id, _ = subproblems.var_to_data[var]
                if var_id in id_to_val:
                    var.fix(id_to_val[var_id])
            
            # Solve
            results = solver.solve(model, load_solutions=False)
            
            if (results.solver.termination_condition == pyo.TerminationCondition.optimal or
                results.solver.termination_condition == pyo.TerminationCondition.locallyOptimal):
                model.solutions.load_from(results)
                
                # Use Dual Bound (lower_bound) if available, to match user's precision/validity requirements
                # Fallback to Primal (objective value) if necessary
                try:
                    # Gurobi and others usually populate problem.lower_bound
                    obj_val = results.problem[0].lower_bound
                    # If lower_bound is not a valid number (e.g. not available), use primal
                    if obj_val is None or obj_val == float('-inf') or obj_val == float('inf'):
                         obj_val = pyo.value(model.component_data_objects(pyo.Objective, active=True).__next__())
                except:
                     obj_val = pyo.value(model.component_data_objects(pyo.Objective, active=True).__next__())
                
                Y_vals.append(obj_val)
            else:
                # Infeasible or error. 
                # If infeasible, recourse is infinity.
                # This breaks quadratic fitting.
                # We should probably skip this sample or assign a high value?
                # For now, assign a large penalty.
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
        
        # Fit
        # coeffs = argmin || X_mat * beta - Y ||^2
        if len(Y_vals) > 0:
            # print(f"DEBUG: Fitting for {subproblem_name}, samples={len(Y_vals)}")
            coeffs, residuals, rank, s = np.linalg.lstsq(X_mat, Y_vals, rcond=None)
            
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
            
            b_s_raw = coeffs[idx]
            
            # Compute offset m_s = min (true - surrogate)
            # Predicted values
            Y_pred = X_mat @ coeffs
            diff = Y_vals - Y_pred
            m_s = np.min(diff)
            
            # Shift b_s
            b_s = b_s_raw + m_s
            
            # Accumulate weighted by probability
            local_agg_Q += prob * Q_s
            local_agg_c += prob * c_s
            local_agg_b += prob * b_s

            # --- Spectral Bound Calculation (Decomposed) ---
            # 1. Compute Q_spec (clipped eigenvalues)
            eig_vals, eig_vecs = np.linalg.eigh(Q_s)
            eig_vals[eig_vals < 0] = 0.0 # Clip
            Q_spec = eig_vecs @ np.diag(eig_vals) @ eig_vecs.T

            # 2. Shift Spectral relative to samples
            Y_pred_spec = []
            for k in range(len(samples)):
                x_k = samples[k]
                quad_term = x_k @ Q_spec @ x_k
                lin_term = c_s @ x_k 
                val_k = quad_term + lin_term + b_s_raw
                Y_pred_spec.append(val_k)
            Y_pred_spec = np.array(Y_pred_spec)

            diff_spec = Y_vals - Y_pred_spec
            m_shift_spec = np.min(diff_spec)
            b_spec = b_s_raw + m_shift_spec

            # 3. Minimize LOCALLY
            # ms (as per user) = min value of spectral surrogate for this scene
            res_min = minimize(make_objective(Q_spec, c_s, b_spec), 
                               x0, 
                               method='L-BFGS-B', 
                               jac=make_jacobian(Q_spec, c_s), 
                               bounds=scipy_bounds)
            ms_val = res_min.fun # The minimum value
            
            # 4. Accumulate Sum of Minima
            local_agg_b_spec += prob * ms_val

    # 4. Aggregate across ranks
    # We need to sum local_agg_Q, local_agg_c, local_agg_b across all ranks
    # MPI.COMM_WORLD.allreduce(..., op=MPI.SUM)
    
    # 4. Aggregate across ranks
    # Flatten Q for reduction (Original)
    Q_flat = local_agg_Q.flatten()
    
    global_Q_flat = MPI.COMM_WORLD.allreduce(Q_flat, op=MPI.SUM)
    global_c = MPI.COMM_WORLD.allreduce(local_agg_c, op=MPI.SUM)
    local_b_arr = np.array([local_agg_b])
    global_b = MPI.COMM_WORLD.allreduce(local_b_arr, op=MPI.SUM)
    
    global_Q = global_Q_flat.reshape((dim, dim))
    
    # Aggregate Spectral (Scalar)
    local_val_spec_arr = np.array([local_agg_b_spec])
    global_val_spec = MPI.COMM_WORLD.allreduce(local_val_spec_arr, op=MPI.SUM)[0]
    
    # 5. Minimize aggregated surrogate (Original)
    res_orig = minimize(make_objective(global_Q, global_c, global_b), 
                        x0, 
                        method='L-BFGS-B', 
                        jac=make_jacobian(global_Q, global_c), 
                        bounds=scipy_bounds)
    val_orig = res_orig.fun

    return val_orig, global_val_spec
