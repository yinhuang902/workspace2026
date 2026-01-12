
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
WLSQ_NUM_SAMPLES_DEFAULT = 100  # Default number of samples per iteration
WLSQ_B_ALPHA = 5.0          # kernel decay alpha for Method B
WLSQ_B_EPS = 0.1            # weight floor epsilon for Method B
WLSQ_B_DIAM2_TINY_THRESH = 1e-9   # threshold for "tiny box" fallback (sets diam2=1.0)
WLSQ_B_DIAM2_SAFE_MIN = 1e-12     # safe denominator threshold for kernel distance
WLSQ_C_GRID_POINTS_PER_DIM = 10    # Default grid points per dimension for Method C
WLSQ_D_GAMMA = 5.0        # default, easy to tune
WLSQ_D_EPS = 0.1          # match WLSQ_B_EPS

# Sampling mode switch for all WLSQ variants (except WLSQ_C)
WLSQ_USE_MIXED_SAMPLING = True   # if False -> original pure random sampling

# Mixed sampling parameters
WLSQ_MIXED_FRACTION_UNIFORM = 0.30   # 30% uniform random points
WLSQ_MIXED_FRACTION_LOWBIAS = 0.70   # 70% biased-toward-low-value points
# (Ensure fractions sum to 1.0; if not, normalize in code.)

# Candidate oversampling factor for biased selection (draw more candidates, then keep best)
WLSQ_MIXED_LOWBIAS_OVERSAMPLE = 5    # e.g., draw k*M candidates then pick low-value subset

# If we use rank-based “low-value preference”, define a parameter:
WLSQ_MIXED_RANK_GAMMA = 5.0
WLSQ_MIXED_EPS = 0.1

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

def _rank_weights(y: np.ndarray, gamma: float, eps: float) -> np.ndarray:
    """
    Computes weights based on the rank of y values (smaller y -> higher weight).
    w_i = eps + (1-eps) * exp(-gamma * (rank_i - 1) / max(M-1, 1))
    """
    M = len(y)
    if M == 0: return np.array([])
    
    # argsort of argsort gives the 0-based rank
    # e.g. y=[10, 5, 20] -> argsort=[1, 0, 2] -> argsort=[1, 0, 2] => ranks=[1, 0, 2]
    # 5 is rank 0, 10 is rank 1, 20 is rank 2
    ranks_0based = np.argsort(np.argsort(y))
    
    denom = max(M - 1, 1)
    w = np.exp(-gamma * ranks_0based / denom)
    w = eps + (1.0 - eps) * w
    return w

def _generate_wlsq_samples(
    node: Node, 
    subproblems: Subproblems, 
    lifted_vars: List[Dict], 
    bounds: List[Tuple[float, float]], 
    num_samples: int, 
    rng: np.random.Generator,
    saved_states: Dict,
    max_retries: int = 1000
) -> List[np.ndarray]:
    """
    Generates samples using either pure random or mixed (uniform + low-bias) strategy.
    """
    dim = len(lifted_vars)
    samples = []
    seen_points = set()
    
    # Helper to generate random point
    def get_random_point():
        point = []
        for i in range(dim):
            lb, ub = bounds[i]
            if lb == float('-inf'): lb = -1e4 
            if ub == float('inf'): ub = 1e4
            if lb > ub: return None
            val = rng.uniform(lb, ub)
            v_type = lifted_vars[i]['type']
            if v_type in [SupportedVars.binary, SupportedVars.integers, SupportedVars.nonnegative_integers]:
                val = round(val)
                val = max(lb, min(ub, val))
            point.append(val)
        return point

    # Helper to check feasibility and optionally compute y_bar
    def check_feasibility(point, compute_y_bar=False):
        all_feasible = True
        y_bar = 0.0
        
        id_to_val = {(lifted_vars[i]['type'], lifted_vars[i]['id']): point[i] for i in range(dim)}
        
        for name in subproblems.names:
            model = subproblems.model[name]
            prob = subproblems.probability[name]
            
            # Fix vars
            for var in subproblems.subproblem_lifted_vars[name]:
                v_type, v_id, _ = subproblems.var_to_data[var]
                if (v_type, v_id) in id_to_val:
                    var.fix(id_to_val[(v_type, v_id)])
            
            success, obj = _solve_true_recourse_primal_gurobi(model)
            
            if not success:
                if compute_y_bar:
                    obj = 1e6
                    all_feasible = False 
                else:
                    all_feasible = False
            
            if compute_y_bar:
                y_bar += prob * obj
            
            if not success and not compute_y_bar:
                break

        # Restore vars
        for name in subproblems.names:
            for var in subproblems.subproblem_lifted_vars[name]:
                lb_v, ub_v, is_fixed, val = saved_states[name][id(var)]
                var.lb = lb_v
                var.ub = ub_v
                if is_fixed: var.fix(val)
                else: 
                    var.unfix()
                    var.value = val
                    
        return all_feasible, y_bar

    target_n_u = num_samples
    target_n_b = 0
    
    if WLSQ_USE_MIXED_SAMPLING:
        target_n_u = int(round(WLSQ_MIXED_FRACTION_UNIFORM * num_samples))
        target_n_b = num_samples - target_n_u
    
    # 1. Uniform Samples
    attempts = 0
    while len(samples) < target_n_u and attempts < target_n_u * max_retries:
        attempts += 1
        pt = get_random_point()
        if pt is None: continue
        
        pt_tuple = tuple(np.round(pt, 6))
        if pt_tuple in seen_points: continue
        
        is_feas, _ = check_feasibility(pt, compute_y_bar=False)
        if is_feas:
            samples.append(np.array(pt))
            seen_points.add(pt_tuple)

    # 2. Biased Samples
    if target_n_b > 0:
        pool_size = max(WLSQ_MIXED_LOWBIAS_OVERSAMPLE * target_n_b, target_n_b)
        candidates = []
        
        pool_attempts = 0
        while len(candidates) < pool_size and pool_attempts < pool_size * max_retries:
            pool_attempts += 1
            pt = get_random_point()
            if pt is None: continue
            
            pt_tuple = tuple(np.round(pt, 6))
            if pt_tuple in seen_points: continue
            
            # Evaluate y_bar (and feasibility)
            is_feas, y_bar = check_feasibility(pt, compute_y_bar=True)
            
            # We store it even if infeasible (with high cost) to sort, 
            # but we only pick feasible ones later.
            candidates.append((pt, y_bar, is_feas))
        
        # Filter to only feasible candidates for selection
        feasible_candidates = [c for c in candidates if c[2]]
        
        M = len(feasible_candidates)
        if M > 0:
            # Extract y_bar values
            y_bars = np.array([c[1] for c in feasible_candidates])
            
            # Compute ranks (0-based)
            # argsort gives indices that sort the array
            # argsort of argsort gives the rank
            ranks_0based = np.argsort(np.argsort(y_bars))
            
            # Compute weights
            denom = max(M - 1, 1)
            base_w = np.exp(-WLSQ_MIXED_RANK_GAMMA * ranks_0based / denom)
            w = WLSQ_MIXED_EPS + (1.0 - WLSQ_MIXED_EPS) * base_w
            
            # Normalize
            w_sum = np.sum(w)
            if w_sum > 0 and np.isfinite(w_sum):
                p = w / w_sum
            else:
                p = np.ones(M) / M
                
            # Select indices
            n_select = min(target_n_b, M)
            selected_indices = rng.choice(M, size=n_select, replace=False, p=p)
            
            for idx in selected_indices:
                pt = feasible_candidates[idx][0]
                pt_tuple = tuple(np.round(pt, 6))
                if pt_tuple not in seen_points:
                    samples.append(np.array(pt))
                    seen_points.add(pt_tuple)

    # 3. Fallback (fill up to num_samples)
    attempts = 0
    while len(samples) < num_samples and attempts < num_samples * max_retries:
        attempts += 1
        pt = get_random_point()
        if pt is None: continue
        pt_tuple = tuple(np.round(pt, 6))
        if pt_tuple in seen_points: continue
        is_feas, _ = check_feasibility(pt, compute_y_bar=False)
        if is_feas:
            samples.append(np.array(pt))
            seen_points.add(pt_tuple)
            
    return samples

def _minimize_quadratic(beta, dim, bounds):
    """
    Minimizes the quadratic surrogate defined by beta over the box.
    Returns (min_val, x_min).
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
        return res_min_q.fun, res_min_q.x
    except:
        return float('inf'), None

def _evaluate_true_objective_local(x, subproblems, lifted_vars, saved_states):
    """
    Evaluates sum(prob * true_recourse) for LOCAL subproblems at x.
    """
    dim = len(lifted_vars)
    local_obj = 0.0
    id_to_val = {(lifted_vars[i]['type'], lifted_vars[i]['id']): x[i] for i in range(dim)}
    
    for name in subproblems.names:
        model = subproblems.model[name]
        prob = subproblems.probability[name]
        
        # Fix vars
        for var in subproblems.subproblem_lifted_vars[name]:
            v_type, v_id, _ = subproblems.var_to_data[var]
            if (v_type, v_id) in id_to_val:
                var.fix(id_to_val[(v_type, v_id)])
        
        # Solve
        success, obj = _solve_true_recourse_primal_gurobi(model)
        
        if success:
            local_obj += prob * obj
        else:
            local_obj += prob * 1e6 # Penalty for infeasibility
            
    # Restore vars
    for name in subproblems.names:
        for var in subproblems.subproblem_lifted_vars[name]:
            lb_v, ub_v, is_fixed, val = saved_states[name][id(var)]
            var.lb = lb_v
            var.ub = ub_v
            if is_fixed: var.fix(val)
            else: 
                var.unfix()
                var.value = val
                
    return local_obj

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
    
    # Save state of ALL subproblems
    # We need to check feasibility for ALL scenarios
    saved_states = {}
    for name in subproblems.names:
        saved_states[name] = {}
        for var in subproblems.subproblem_lifted_vars[name]:
            saved_states[name][id(var)] = (var.lb, var.ub, var.is_fixed(), var.value)
            
    # Use mixed sampler
    samples = _generate_wlsq_samples(node, subproblems, lifted_vars, bounds, num_samples, rng, saved_states, max_retries)
    
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
    
    # Calculate box diameter squared for Method B (and C)
    diam2 = 0.0
    for j in range(dim):
        lb, ub = bounds[j]
        l = lb if lb != float('-inf') else -1e4
        u = ub if ub != float('inf') else 1e4
        diam2 += (u - l)**2
    if diam2 < WLSQ_B_DIAM2_TINY_THRESH: diam2 = 1.0
    diam2_safe = max(diam2, WLSQ_B_DIAM2_SAFE_MIN)

    # --- Method B/C: probability-weighted mixture of kernels over all scenario anchors ---
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
    
    if anchors:
        anchors = np.array(anchors)
        probs = np.array(probs)
        prob_sum = np.sum(probs)
        if prob_sum > 0:
            probs = probs / prob_sum
        else:
             probs = np.ones(len(probs)) / len(probs)

    # --- WLSQ_C: Deterministic Grid Sampling ---
    # Generate grid points
    grid_points_per_dim = WLSQ_C_GRID_POINTS_PER_DIM
    grid_axes = []
    for i in range(dim):
        lb, ub = bounds[i]
        if lb == float('-inf'): lb = -1e4
        if ub == float('inf'): ub = 1e4
        
        if lb == ub:
            grid_axes.append(np.array([lb]))
        else:
            grid_axes.append(np.linspace(lb, ub, grid_points_per_dim))
            
    # Cartesian product
    import itertools
    grid_samples_raw = list(itertools.product(*grid_axes))
    
    # Filter feasible grid samples
    samples_C = []
    seen_points_C = set()
    
    # We need to check feasibility for grid points just like random ones
    # Reuse saved_states
    for point_tuple in grid_samples_raw:
        point = list(point_tuple)
        point_tuple_rounded = tuple(np.round(point, 6))
        if point_tuple_rounded in seen_points_C:
            continue
            
        # Check feasibility
        all_feasible = True
        id_to_val = {(lifted_vars[i]['type'], lifted_vars[i]['id']): point[i] for i in range(dim)}
        
        for name in subproblems.names:
            model = subproblems.model[name]
            for var in subproblems.subproblem_lifted_vars[name]:
                v_type, v_id, _ = subproblems.var_to_data[var]
                if (v_type, v_id) in id_to_val:
                    var.fix(id_to_val[(v_type, v_id)])
            
            success, _ = _solve_true_recourse_primal_gurobi(model)
            if not success:
                all_feasible = False
                break
        
        # Restore
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
            samples_C.append(np.array(point))
            seen_points_C.add(point_tuple_rounded)
            
    # Fallback to random if too few points
    if len(samples_C) < 10:
        if WLSQ_USE_MIXED_SAMPLING:
            # Use mixed sampler to fill
            needed = 10 - len(samples_C)
            # Request slightly more to ensure we get enough unique ones
            fill_samples = _generate_wlsq_samples(node, subproblems, lifted_vars, bounds, needed * 2, rng, saved_states, max_retries)
            for pt in fill_samples:
                if len(samples_C) >= 10: break
                pt_tuple = tuple(np.round(pt, 6))
                if pt_tuple not in seen_points_C:
                    samples_C.append(pt)
                    seen_points_C.add(pt_tuple)
        else:
            # Original random fallback
            attempts = 0
            while len(samples_C) < 10 and attempts < max_retries * 10:
                attempts += 1
                point = []
                for i in range(dim):
                    lb, ub = bounds[i]
                    if lb == float('-inf'): lb = -1e4
                    if ub == float('inf'): ub = 1e4
                    val = rng.uniform(lb, ub)
                    v_type = lifted_vars[i]['type']
                    if v_type in [SupportedVars.binary, SupportedVars.integers, SupportedVars.nonnegative_integers]:
                        val = round(val)
                        val = max(lb, min(ub, val))
                    point.append(val)
                
                point_tuple = tuple(np.round(point, 6))
                if point_tuple in seen_points_C:
                    continue
                    
                # Check feasibility
                all_feasible = True
                id_to_val = {(lifted_vars[i]['type'], lifted_vars[i]['id']): point[i] for i in range(dim)}
                for name in subproblems.names:
                    model = subproblems.model[name]
                    for var in subproblems.subproblem_lifted_vars[name]:
                        v_type, v_id, _ = subproblems.var_to_data[var]
                        if (v_type, v_id) in id_to_val:
                            var.fix(id_to_val[(v_type, v_id)])
                    success, _ = _solve_true_recourse_primal_gurobi(model)
                    if not success:
                        all_feasible = False
                        break
                
                # Restore
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
                    samples_C.append(np.array(point))
                    seen_points_C.add(point_tuple)

    # Build Phi_C
    Phi_C = []
    for x in samples_C:
        row = [1.0]
        for i in range(dim):
            row.append(x[i])
        for i in range(dim):
            for j in range(i, dim):
                row.append(x[i] * x[j])
        Phi_C.append(row)
    Phi_C = np.array(Phi_C)
    
    # Pre-calculate weights for C (Method B logic)
    # Reuse anchors and probs from earlier
    w_C_shared = np.ones(len(samples_C))
    if anchors is not None and len(anchors) > 0: # anchors defined above
        # d2_it = ||samples_C[i]-anchors[t]||^2 / diam2_safe
        samples_C_arr = np.array(samples_C)
        diffs_C = samples_C_arr[:, np.newaxis, :] - anchors[np.newaxis, :, :]
        dists2_C = np.sum(diffs_C**2, axis=2)
        d2_it_C = dists2_C / diam2_safe
        
        kernel_it_C = np.exp(-WLSQ_B_ALPHA * d2_it_C)
        mix_i_C = kernel_it_C @ probs
        w_C_shared = WLSQ_B_EPS + (1.0 - WLSQ_B_EPS) * mix_i_C

    local_agg_lb = 0.0
    
    # 4. Process each scenario
    local_agg_lb_uniform = 0.0
    local_agg_lb_A = 0.0
    local_agg_lb_B = 0.0
    local_agg_lb_C = 0.0
    
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
    
    # --- Method B: probability-weighted mixture of kernels over all scenario anchors ---
    # (Anchors logic already done above for C, reusing here for B)
    
    w_B_shared = np.ones(num_samples)
    if anchors is not None and len(anchors) > 0:
        # d2_it = ||samples[i]-anchors[t]||^2 / diam2_safe
        diffs = samples[:, np.newaxis, :] - anchors[np.newaxis, :, :]
        dists2 = np.sum(diffs**2, axis=2)
        d2_it = dists2 / diam2_safe
        
        alpha = WLSQ_B_ALPHA
        kernel_it = np.exp(-alpha * d2_it)
        mix_i = kernel_it @ probs
        
        eps = WLSQ_B_EPS
        w_B_shared = eps + (1.0 - eps) * mix_i

    # --- Phase A: Evaluate all scenarios (collect data) ---
    all_scenario_data = []
    
    # Also accumulate y_bar for D1 (shared weights)
    # y_bar[i] = sum_s p_s * y_s[i]
    y_bar = np.zeros(num_samples)
    
    for subproblem_name in subproblems.names:
        model = subproblems.model[subproblem_name]
        prob = subproblems.probability[subproblem_name]
        
        Y_vals = []
        valid_indices = [] # Track valid indices for stats
        
        # Save state
        saved_state = {}
        for var in subproblems.subproblem_lifted_vars[subproblem_name]:
            saved_state[id(var)] = (var.lb, var.ub, var.is_fixed(), var.value)
            
        # Evaluate samples (Uniform/A/B/D)
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
        
        # Evaluate samples_C (Method C)
        Y_vals_C = []
        for i_s, x in enumerate(samples_C):
            id_to_val = {(lifted_vars[i]['type'], lifted_vars[i]['id']): x[i] for i in range(dim)}
            for var in subproblems.subproblem_lifted_vars[subproblem_name]:
                v_type, v_id, _ = subproblems.var_to_data[var]
                if (v_type, v_id) in id_to_val:
                    var.fix(id_to_val[(v_type, v_id)])
            success, obj_val = _solve_true_recourse_primal_gurobi(model)
            if success:
                Y_vals_C.append(obj_val)
            else:
                Y_vals_C.append(1e6)
        Y_vals_C = np.array(Y_vals_C)
        
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
        
        # Accumulate for D1
        y_bar += prob * Y_vals
        
        # Store for Phase B
        all_scenario_data.append({
            'name': subproblem_name,
            'model': model,
            'prob': prob,
            'Y_vals': Y_vals,
            'Y_vals_C': Y_vals_C,
            'valid_indices': valid_indices
        })

    # --- Compute D1 shared weights ---
    w_D1_shared = _rank_weights(y_bar, WLSQ_D_GAMMA, WLSQ_D_EPS)

    # --- Phase B: Fit and Bound (per scenario) ---
    local_agg_lb_uniform = 0.0
    local_agg_lb_A = 0.0
    local_agg_lb_B = 0.0
    local_agg_lb_C = 0.0
    local_agg_lb_D1 = 0.0
    local_agg_lb_D2 = 0.0
    
    # Beta accumulators for UB_true
    # Size of beta is 1 + dim + dim*(dim+1)/2
    # For dim=3, size is 1 + 3 + 6 = 10
    beta_size = 1 + dim + (dim * (dim + 1)) // 2
    local_beta_agg_uniform = np.zeros(beta_size)
    local_beta_agg_A = np.zeros(beta_size)
    local_beta_agg_B = np.zeros(beta_size)
    local_beta_agg_C = np.zeros(beta_size)
    local_beta_agg_D1 = np.zeros(beta_size)
    local_beta_agg_D2 = np.zeros(beta_size)
    
    for data in all_scenario_data:
        subproblem_name = data['name']
        model = data['model']
        prob = data['prob']
        Y_vals = data['Y_vals']
        Y_vals_C = data['Y_vals_C']
        valid_indices = data['valid_indices']
        
        # --- Calculate Weights (Method A) ---
        w_A = np.ones(num_samples)
        if len(valid_indices) > 0:
            Y_valid = Y_vals[valid_indices]
            y_ref = np.median(Y_valid)
            iqr = np.percentile(Y_valid, 75) - np.percentile(Y_valid, 25)
            tau = max(iqr, 1e-6)
            arg = -(y_ref - Y_vals) / tau
            arg = np.clip(arg, -50, 50)
            s = 1.0 / (1.0 + np.exp(arg))
            w_low = 1.0 + 2.0 * s 
            w_raw = w_center * w_low
            w_A = np.maximum(w_raw, 0.1)
            w_A = np.minimum(w_A, 10.0)
        else:
            w_A = np.ones(num_samples)
            
        # --- Calculate Weights (Method B) ---
        w_B = w_B_shared
        
        # --- Calculate Weights (Method C) ---
        w_C = w_C_shared
        
        # --- Calculate Weights (Method D1) ---
        w_D1 = w_D1_shared
        
        # --- Calculate Weights (Method D2: scenario-specific rank) ---
        w_D2 = _rank_weights(Y_vals, WLSQ_D_GAMMA, WLSQ_D_EPS)

        # --- Fit & Bound: Uniform ---
        try:
            beta_uni = np.linalg.solve(PhiT_Phi_reg, Phi.T @ Y_vals)
            lb_uni = _compute_bound_from_beta(beta_uni, dim, lifted_vars, subproblems, subproblem_name, bounds, model)
            local_beta_agg_uniform += prob * beta_uni
        except:
            lb_uni = float('-inf')
        local_agg_lb_uniform += prob * lb_uni

        # --- Fit & Bound: Method A ---
        try:
            S = np.sqrt(w_A)
            Phi_w = Phi * S[:, np.newaxis]
            Y_w = Y_vals * S
            PhiT_Phi_w = Phi_w.T @ Phi_w + reg_matrix
            beta_A = np.linalg.solve(PhiT_Phi_w, Phi_w.T @ Y_w)
            lb_A = _compute_bound_from_beta(beta_A, dim, lifted_vars, subproblems, subproblem_name, bounds, model)
            local_beta_agg_A += prob * beta_A
        except:
            lb_A = float('-inf')
        local_agg_lb_A += prob * lb_A
        
        # --- Fit & Bound: Method B ---
        try:
            S_B = np.sqrt(w_B)
            Phi_w_B = Phi * S_B[:, np.newaxis]
            Y_w_B = Y_vals * S_B
            PhiT_Phi_w_B = Phi_w_B.T @ Phi_w_B + reg_matrix
            beta_B = np.linalg.solve(PhiT_Phi_w_B, Phi_w_B.T @ Y_w_B)
            lb_B = _compute_bound_from_beta(beta_B, dim, lifted_vars, subproblems, subproblem_name, bounds, model)
            local_beta_agg_B += prob * beta_B
        except:
            lb_B = float('-inf')
        local_agg_lb_B += prob * lb_B
        
        # --- Fit & Bound: Method C ---
        try:
            S_C = np.sqrt(w_C)
            Phi_w_C = Phi_C * S_C[:, np.newaxis]
            Y_w_C = Y_vals_C * S_C
            PhiT_Phi_w_C = Phi_w_C.T @ Phi_w_C + (lambda_reg * np.eye(Phi_C.shape[1]))
            PhiT_Phi_w_C[0, 0] -= lambda_reg
            beta_C = np.linalg.solve(PhiT_Phi_w_C, Phi_w_C.T @ Y_w_C)
            lb_C = _compute_bound_from_beta(beta_C, dim, lifted_vars, subproblems, subproblem_name, bounds, model)
            local_beta_agg_C += prob * beta_C
        except:
            lb_C = float('-inf')
        local_agg_lb_C += prob * lb_C
        
        # --- Fit & Bound: Method D1 ---
        try:
            S_D1 = np.sqrt(w_D1)
            Phi_w_D1 = Phi * S_D1[:, np.newaxis]
            Y_w_D1 = Y_vals * S_D1
            PhiT_Phi_w_D1 = Phi_w_D1.T @ Phi_w_D1 + reg_matrix
            beta_D1 = np.linalg.solve(PhiT_Phi_w_D1, Phi_w_D1.T @ Y_w_D1)
            lb_D1 = _compute_bound_from_beta(beta_D1, dim, lifted_vars, subproblems, subproblem_name, bounds, model)
            local_beta_agg_D1 += prob * beta_D1
        except:
            lb_D1 = float('-inf')
        local_agg_lb_D1 += prob * lb_D1
        
        # --- Fit & Bound: Method D2 ---
        try:
            S_D2 = np.sqrt(w_D2)
            Phi_w_D2 = Phi * S_D2[:, np.newaxis]
            Y_w_D2 = Y_vals * S_D2
            PhiT_Phi_w_D2 = Phi_w_D2.T @ Phi_w_D2 + reg_matrix
            beta_D2 = np.linalg.solve(PhiT_Phi_w_D2, Phi_w_D2.T @ Y_w_D2)
            lb_D2 = _compute_bound_from_beta(beta_D2, dim, lifted_vars, subproblems, subproblem_name, bounds, model)
            local_beta_agg_D2 += prob * beta_D2
        except:
            lb_D2 = float('-inf')
        local_agg_lb_D2 += prob * lb_D2
        
    # Aggregate across ranks
    local_arr = np.array([local_agg_lb_uniform, local_agg_lb_A, local_agg_lb_B, local_agg_lb_C, local_agg_lb_D1, local_agg_lb_D2])
    global_arr = MPI.COMM_WORLD.allreduce(local_arr, op=MPI.SUM)
    
    global_lb_uniform = global_arr[0]
    global_lb_A = global_arr[1]
    global_lb_B = global_arr[2]
    global_lb_C = global_arr[3]
    global_lb_D1 = global_arr[4]
    global_lb_D2 = global_arr[5]
    
    # Store on node
    node.lb_problem.wlsq_uniform_bound = global_lb_uniform
    node.lb_problem.wlsq_A_bound = global_lb_A
    node.lb_problem.wlsq_B_bound = global_lb_B
    node.lb_problem.wlsq_C_bound = global_lb_C
    node.lb_problem.wlsq_D1_bound = global_lb_D1
    node.lb_problem.wlsq_D2_bound = global_lb_D2
    
    # --- Compute UB_true for each method ---
    # 1. Aggregate betas
    local_betas = np.concatenate([
        local_beta_agg_uniform, local_beta_agg_A, local_beta_agg_B, 
        local_beta_agg_C, local_beta_agg_D1, local_beta_agg_D2
    ])
    global_betas = MPI.COMM_WORLD.allreduce(local_betas, op=MPI.SUM)
    
    # Split back
    bs = beta_size
    g_beta_uni = global_betas[0:bs]
    g_beta_A = global_betas[bs:2*bs]
    g_beta_B = global_betas[2*bs:3*bs]
    g_beta_C = global_betas[3*bs:4*bs]
    g_beta_D1 = global_betas[4*bs:5*bs]
    g_beta_D2 = global_betas[5*bs:6*bs]
    
    methods = [
        ('uniform', g_beta_uni), ('A', g_beta_A), ('B', g_beta_B), 
        ('C', g_beta_C), ('D1', g_beta_D1), ('D2', g_beta_D2)
    ]
    
    # Cache evaluations to avoid re-evaluating same point
    eval_cache = {}
    
    for m_name, m_beta in methods:
        # Minimize surrogate
        _, x_star = _minimize_quadratic(m_beta, dim, bounds)
        
        ub_true = float('nan')
        if x_star is not None:
            x_tuple = tuple(np.round(x_star, 6))
            if x_tuple in eval_cache:
                ub_true = eval_cache[x_tuple]
            else:
                # Evaluate true objective
                local_obj = _evaluate_true_objective_local(x_star, subproblems, lifted_vars, saved_states)
                ub_true = MPI.COMM_WORLD.allreduce(local_obj, op=MPI.SUM)
                eval_cache[x_tuple] = ub_true
        
        # Store on node
        setattr(node.lb_problem, f'wlsq_{m_name}_ub', ub_true)

    return max(global_lb_uniform, global_lb_A, global_lb_B, global_lb_C, global_lb_D1, global_lb_D2)

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
