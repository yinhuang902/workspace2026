
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
import math

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
WLSQ_MIXED_FRACTION_UNIFORM = 0.60   # % uniform random points
WLSQ_MIXED_FRACTION_LOWBIAS = 0.40   # 70% biased-toward-low-value points
# (Ensure fractions sum to 1.0; if not, normalize in code.)

# Candidate oversampling factor for biased selection (draw more candidates, then keep best)
WLSQ_MIXED_LOWBIAS_OVERSAMPLE = 5    # e.g., draw k*M candidates then pick low-value subset

# If we use rank-based “low-value preference”, define a parameter:
WLSQ_MIXED_RANK_GAMMA = 5.0
WLSQ_MIXED_EPS = 0.1

# ============ WLSQ Method E: Anchor-Mixed Sampling ============
# Method E uses samples concentrated near per-scenario solution anchors.
# Modify these parameters to tune Method E behavior.

# Fraction of samples near anchors vs uniform random
WLSQ_E_FRACTION_NEAR_ANCHOR = 0.4   # 70% of samples near scenario anchors
WLSQ_E_FRACTION_UNIFORM = 0.6       # 30% of samples uniformly distributed

# Neighborhood size around anchors (Gaussian sigma as fraction of box width)
# sigma_j = WLSQ_E_SIGMA_FRAC * (ub_j - lb_j)
WLSQ_E_SIGMA_FRAC = 0.10             # 10% of box width

# Max rejection attempts for near-anchor sampling
WLSQ_E_MAX_REJECT = 2000

# Anchor selection: weight by scenario probability or uniform
WLSQ_E_PICK_BY_PROB = True           # True = proportional to scenario prob; False = uniform


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

def _sample_near_anchor(rng: np.random.Generator, anchor: np.ndarray, bounds: List[Tuple[float, float]], sigma_frac: float) -> List[float]:
    """
    Sample a point near the given anchor using Gaussian perturbation.
    Returns a list of floats clipped to bounds.
    """
    dim = len(anchor)
    point = []
    for i in range(dim):
        lb, ub = bounds[i]
        # Handle infinite bounds
        if lb == float('-inf'): lb = -1e4
        if ub == float('inf'): ub = 1e4
        width = ub - lb
        
        if width < 1e-9:
            # Tiny or degenerate dimension - keep at anchor
            point.append(anchor[i])
        else:
            sigma = sigma_frac * width
            val = anchor[i] + rng.normal(0, sigma)
            # Clip to bounds
            val = max(lb, min(ub, val))
            point.append(val)
    return point

def _generate_wlsq_samples(
    node: Node, 
    subproblems: Subproblems, 
    lifted_vars: List[Dict], 
    bounds: List[Tuple[float, float]], 
    num_samples: int, 
    rng: np.random.Generator,
    saved_states: Dict,
    max_retries: int = 1000,
    use_mixed_sampling: bool = True,
    anchors: Optional[np.ndarray] = None,
    anchor_probs: Optional[np.ndarray] = None,
    sampling_mode: str = "mixed_lowbias"
) -> List[np.ndarray]:
    """
    Generates samples using one of:
    - "random": pure uniform random
    - "mixed_lowbias": uniform + low-bias based on y_bar (existing)
    - "anchor_mixed": uniform + near-anchor sampling
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

    # Determine sampling strategy based on sampling_mode
    if sampling_mode == "anchor_mixed" and anchors is not None and len(anchors) > 0:
        # ========== ANCHOR-MIXED SAMPLING ==========
        target_n_near = int(round(WLSQ_E_FRACTION_NEAR_ANCHOR * num_samples))
        target_n_uniform = num_samples - target_n_near
        

        
        # 1. Sample near anchors
        near_attempts = 0
        while len(samples) < target_n_near and near_attempts < WLSQ_E_MAX_REJECT:
            near_attempts += 1
            
            # Pick an anchor
            anchor_idx = rng.choice(len(anchors), p=anchor_probs)
            anchor = anchors[anchor_idx]
            
            # Sample near it
            pt = _sample_near_anchor(rng, anchor, bounds, WLSQ_E_SIGMA_FRAC)
            if pt is None: continue
            
            # Round for discrete vars
            for i in range(dim):
                v_type = lifted_vars[i]['type']
                if v_type in [SupportedVars.binary, SupportedVars.integers, SupportedVars.nonnegative_integers]:
                    lb, ub = bounds[i]
                    if lb == float('-inf'): lb = -1e4
                    if ub == float('inf'): ub = 1e4
                    pt[i] = round(pt[i])
                    pt[i] = max(lb, min(ub, pt[i]))
            
            pt_tuple = tuple(np.round(pt, 6))
            if pt_tuple in seen_points: continue
            
            is_feas, _ = check_feasibility(pt, compute_y_bar=False)
            if is_feas:
                samples.append(np.array(pt))
                seen_points.add(pt_tuple)
        
        # 2. Uniform samples
        uniform_attempts = 0
        n_uniform_got = len(samples)
        while len(samples) < n_uniform_got + target_n_uniform and uniform_attempts < target_n_uniform * max_retries:
            uniform_attempts += 1
            pt = get_random_point()
            if pt is None: continue
            
            pt_tuple = tuple(np.round(pt, 6))
            if pt_tuple in seen_points: continue
            
            is_feas, _ = check_feasibility(pt, compute_y_bar=False)
            if is_feas:
                samples.append(np.array(pt))
                seen_points.add(pt_tuple)
        
        # 3. Fallback fill
        fallback_attempts = 0
        while len(samples) < num_samples and fallback_attempts < num_samples * max_retries:
            fallback_attempts += 1
            pt = get_random_point()
            if pt is None: continue
            pt_tuple = tuple(np.round(pt, 6))
            if pt_tuple in seen_points: continue
            is_feas, _ = check_feasibility(pt, compute_y_bar=False)
            if is_feas:
                samples.append(np.array(pt))
                seen_points.add(pt_tuple)
        

        
    elif sampling_mode == "mixed_lowbias" or use_mixed_sampling:
        # ========== EXISTING MIXED LOWBIAS SAMPLING ==========
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
                
                is_feas, y_bar = check_feasibility(pt, compute_y_bar=True)
                candidates.append((pt, y_bar, is_feas))
            
            feasible_candidates = [c for c in candidates if c[2]]
            M = len(feasible_candidates)
            if M > 0:
                y_bars = np.array([c[1] for c in feasible_candidates])
                ranks_0based = np.argsort(np.argsort(y_bars))
                denom = max(M - 1, 1)
                base_w = np.exp(-WLSQ_MIXED_RANK_GAMMA * ranks_0based / denom)
                w = WLSQ_MIXED_EPS + (1.0 - WLSQ_MIXED_EPS) * base_w
                w_sum = np.sum(w)
                if w_sum > 0 and np.isfinite(w_sum):
                    p = w / w_sum
                else:
                    p = np.ones(M) / M
                n_select = min(target_n_b, M)
                selected_indices = rng.choice(M, size=n_select, replace=False, p=p)
                for idx in selected_indices:
                    pt = feasible_candidates[idx][0]
                    pt_tuple = tuple(np.round(pt, 6))
                    if pt_tuple not in seen_points:
                        samples.append(np.array(pt))
                        seen_points.add(pt_tuple)

        # 3. Fallback
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
                
    else:
        # ========== PURE RANDOM SAMPLING ==========
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
    max_retries: int = 1000,
    enabled_methods: Optional[Dict[str, bool]] = None,
    enabled_ub_methods: Optional[Dict[str, bool]] = None
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
        seed: Optional seed for reproducible sampling.
        enabled_methods: Dict mapping {'uniform', 'A', 'B', 'C', 'D1', 'D2'} to bool.
        enabled_ub_methods: Dict mapping same keys for UB computation.
    """
    
    # Defaults (All enabled if not specified, for backward compat)
    if enabled_methods is None:
        enabled_methods = {k: True for k in ['uniform', 'A', 'B', 'C', 'D1', 'D2', 'E']}
    if enabled_ub_methods is None:
        enabled_ub_methods = {k: True for k in ['uniform', 'A', 'B', 'C', 'D1', 'D2', 'E']}
        
    on_U = enabled_methods.get('uniform', False)
    on_A = enabled_methods.get('A', False)
    on_B = enabled_methods.get('B', False)
    on_C = enabled_methods.get('C', False)
    on_D1 = enabled_methods.get('D1', False)
    on_D2 = enabled_methods.get('D2', False)
    on_E = enabled_methods.get('E', False)  # NEW: Anchor-mixed sampling method

    
    # 1. Identify the box (bounds of lifted variables)
    # ISSUE #1: Select ONLY the three first-stage PID variables (Kp, Ki, Kd) by NAME
    lifted_vars = _identify_pid_vars(node, subproblems)
    if lifted_vars is None:
        print("DEBUG: WLSQ failed to identify PID vars")
        return float('-inf')
        
    bounds = []
    for lv in lifted_vars:
        # Sanitize bounds: Handle None as +/- inf
        lb = float(lv['lb']) if lv['lb'] is not None else float('-inf')
        ub = float(lv['ub']) if lv['ub'] is not None else float('inf')
        bounds.append((lb, ub))
    
    dim = len(lifted_vars)
    if dim != 3:
        print(f"DEBUG: WLSQ dim mismatch: {dim}")
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
            
    # Use mixed sampler?
    # Optimization: If ONLY uniform is enabled, we don't need mixed/biased sampling.
    use_mixed = WLSQ_USE_MIXED_SAMPLING
    if not (on_A or on_B or on_C or on_D1 or on_D2) and on_U:
        use_mixed = False
    
    # === ANCHOR EXTRACTION (for Method E) ===
    anchors_list = []
    anchor_weights = []
    
    if on_E:
        for scenario_name in subproblems.names:
            if hasattr(node.lb_problem, 'subproblem_solutions') and scenario_name in node.lb_problem.subproblem_solutions:
                sol = node.lb_problem.subproblem_solutions[scenario_name].lifted_var_solution
                anchor_pt = []
                valid = True
                for lv in lifted_vars:
                    val = sol.get(lv['type'], {}).get(lv['id'])
                    if val is None or not np.isfinite(val):
                        valid = False
                        break
                    anchor_pt.append(val)
                
                if valid:
                    anchors_list.append(np.array(anchor_pt))
                    if WLSQ_E_PICK_BY_PROB:
                        anchor_weights.append(subproblems.probability[scenario_name])
                    else:
                        anchor_weights.append(1.0)
    
    # Convert to arrays and normalize weights for Method E
    anchors_arr = None
    anchor_probs_arr = None
    if len(anchors_list) > 0:
        anchors_arr = np.array(anchors_list)
        anchor_weights = np.array(anchor_weights)
        w_sum = np.sum(anchor_weights)
        if w_sum > 0:
            anchor_probs_arr = anchor_weights / w_sum
        else:
            anchor_probs_arr = np.ones(len(anchors_list)) / len(anchors_list)
    
    # Generate samples for methods uniform/A/B/D1/D2 (original sampling)
    samples = _generate_wlsq_samples(
        node, subproblems, lifted_vars, bounds, num_samples, rng, saved_states, max_retries,
        use_mixed_sampling=use_mixed,
        anchors=None,
        anchor_probs=None,
        sampling_mode="mixed_lowbias" if use_mixed else "random"
    )
    
    if len(samples) < num_samples:
        print(f"DEBUG: WLSQ sampling failed. Got {len(samples)}/{num_samples} samples.")
        return float('-inf')
    
    # Generate SEPARATE samples for Method E (anchor-mixed sampling)
    samples_E = []
    if on_E:
        if anchors_arr is not None and len(anchors_arr) > 0:
            samples_E = _generate_wlsq_samples(
                node, subproblems, lifted_vars, bounds, num_samples, rng, saved_states, max_retries,
                use_mixed_sampling=False,
                anchors=anchors_arr,
                anchor_probs=anchor_probs_arr,
                sampling_mode="anchor_mixed"
            )

        else:
            # Fallback: if no anchors, use the same samples as other methods
            samples_E = samples


    # --- PLOTTING DATA INIT (Per-Method Schema) ---
    if not hasattr(node.lb_problem, 'plot_data_wlsq') or node.lb_problem.plot_data_wlsq is None:
        node.lb_problem.plot_data_wlsq = {}
    
    # Methods uniform/A/B/D1/D2 use 'samples'
    for method_key, is_enabled in [('uniform', on_U), ('A', on_A), ('B', on_B), ('D1', on_D1), ('D2', on_D2)]:
        if is_enabled:
            node.lb_problem.plot_data_wlsq[method_key] = {
                'interp_points': [list(p) for p in samples],
                'interp_truevals': {},
                'ms_point': {},
                'ms_trueval': {}
            }
    # Method C uses samples_C (will be populated after sampling)
    if on_C:
        node.lb_problem.plot_data_wlsq['C'] = {
            'interp_points': [],
            'interp_truevals': {},
            'ms_point': {},
            'ms_trueval': {}
        }
    # Method E uses samples_E
    if on_E and len(samples_E) > 0:
        node.lb_problem.plot_data_wlsq['E'] = {
            'interp_points': [list(p) for p in samples_E],
            'interp_truevals': {},
            'ms_point': {},
            'ms_trueval': {}
        }
    # --------------------------
        
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
    samples_C = []
    w_C_shared = np.array([])
    
    if on_C:
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
            if use_mixed: # Use the local decision variable
                # Use mixed sampler to fill
                needed = 10 - len(samples_C)
                # Request slightly more to ensure we get enough unique ones
                fill_samples = _generate_wlsq_samples(node, subproblems, lifted_vars, bounds, needed * 2, rng, saved_states, max_retries, use_mixed_sampling=True)
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
    if on_C:
        # Store samples_C in plot_data_wlsq['C']
        node.lb_problem.plot_data_wlsq['C']['interp_points'] = [list(p) for p in samples_C]
        
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
        if on_C:
            for i_s, x in enumerate(samples_C):
                id_to_val = {(lifted_vars[i]['type'], lifted_vars[i]['id']): x[i] for i in range(dim)}
                for var in subproblems.subproblem_lifted_vars[subproblem_name]:
                    v_type, v_id, _ = subproblems.var_to_data[var]
                    if (v_type, v_id) in id_to_val:
                        var.fix(id_to_val[(v_type, v_id)])
                success, obj_val = _solve_true_recourse_primal_gurobi(model)
                if success:
                    Y_vals_C.append(obj_val)
                if success:
                    Y_vals_C.append(obj_val)
                else:
                    Y_vals_C.append(1e6)
        Y_vals_C = np.array(Y_vals_C)
        
        # Evaluate samples_E (Method E - anchor-mixed)
        Y_vals_E = []
        if on_E and len(samples_E) > 0:
            for i_s, x in enumerate(samples_E):
                id_to_val = {(lifted_vars[i]['type'], lifted_vars[i]['id']): x[i] for i in range(dim)}
                for var in subproblems.subproblem_lifted_vars[subproblem_name]:
                    v_type, v_id, _ = subproblems.var_to_data[var]
                    if (v_type, v_id) in id_to_val:
                        var.fix(id_to_val[(v_type, v_id)])
                success, obj_val = _solve_true_recourse_primal_gurobi(model)
                if success:
                    Y_vals_E.append(obj_val)
                else:
                    Y_vals_E.append(1e6)
        Y_vals_E = np.array(Y_vals_E)
        
        # --- PLOTTING DATA COLLECTION (True Values Per Method) ---
        # Y_vals corresponds to 'samples' used by uniform/A/B/D1/D2
        # Y_vals_C corresponds to 'samples_C' used by C
        # Y_vals_E corresponds to 'samples_E' used by E
        for method_key in ['uniform', 'A', 'B', 'D1', 'D2']:
            if method_key in node.lb_problem.plot_data_wlsq:
                node.lb_problem.plot_data_wlsq[method_key]['interp_truevals'][subproblem_name] = list(Y_vals)
        if 'C' in node.lb_problem.plot_data_wlsq:
            node.lb_problem.plot_data_wlsq['C']['interp_truevals'][subproblem_name] = list(Y_vals_C)
        if 'E' in node.lb_problem.plot_data_wlsq:
            node.lb_problem.plot_data_wlsq['E']['interp_truevals'][subproblem_name] = list(Y_vals_E)
        # --------------------------------------------
        
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
            'Y_vals_E': Y_vals_E,  # NEW: for Method E
            'valid_indices': valid_indices
        })

    # --- Compute D1 shared weights ---
    w_D1_shared = np.array([])
    if on_D1:
        w_D1_shared = _rank_weights(y_bar, WLSQ_D_GAMMA, WLSQ_D_EPS)

    # --- Phase B: Fit and Bound (per scenario) ---
    local_agg_lb_uniform = 0.0
    local_agg_lb_A = 0.0
    local_agg_lb_B = 0.0
    local_agg_lb_C = 0.0
    local_agg_lb_D1 = 0.0
    local_agg_lb_D2 = 0.0
    local_agg_lb_E = 0.0  # NEW: for Method E
    
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
    local_beta_agg_E = np.zeros(beta_size)  # NEW: for Method E
    
    for data in all_scenario_data:
        subproblem_name = data['name']
        model = data['model']
        prob = data['prob']
        Y_vals = data['Y_vals']
        Y_vals_C = data['Y_vals_C']
        valid_indices = data['valid_indices']
        
        # --- Calculate Weights (Method A) ---
        w_A = np.ones(num_samples)
        if on_A:
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
        w_D2 = np.array([])
        if on_D2:
            w_D2 = _rank_weights(Y_vals, WLSQ_D_GAMMA, WLSQ_D_EPS)

        # --- Fit & Bound: Uniform ---
        if on_U:
            try:
                beta_uni = np.linalg.solve(PhiT_Phi_reg, Phi.T @ Y_vals)
                lb_uni, ms_pt, ms_val = _compute_bound_from_beta(beta_uni, dim, lifted_vars, subproblems, subproblem_name, bounds, model, return_ms_point=True)
                local_beta_agg_uniform += prob * beta_uni
                
                # Store MS point for method 'uniform'
                if 'uniform' in node.lb_problem.plot_data_wlsq:
                    node.lb_problem.plot_data_wlsq['uniform']['ms_point'][subproblem_name] = ms_pt
                    node.lb_problem.plot_data_wlsq['uniform']['ms_trueval'][subproblem_name] = ms_val
            except:
                lb_uni = float('-inf')
        else:
            lb_uni = float('-inf')
        local_agg_lb_uniform += prob * lb_uni

        # --- Fit & Bound: Method A ---
        if on_A:
            try:
                S = np.sqrt(w_A)
                Phi_w = Phi * S[:, np.newaxis]
                Y_w = Y_vals * S
                PhiT_Phi_w = Phi_w.T @ Phi_w + reg_matrix
                beta_A = np.linalg.solve(PhiT_Phi_w, Phi_w.T @ Y_w)
                lb_A, ms_pt, ms_val = _compute_bound_from_beta(beta_A, dim, lifted_vars, subproblems, subproblem_name, bounds, model, return_ms_point=True)
                local_beta_agg_A += prob * beta_A
                
                # Store MS point for method 'A'
                if 'A' in node.lb_problem.plot_data_wlsq:
                    node.lb_problem.plot_data_wlsq['A']['ms_point'][subproblem_name] = ms_pt
                    node.lb_problem.plot_data_wlsq['A']['ms_trueval'][subproblem_name] = ms_val
            except:
                lb_A = float('-inf')
        else:
            lb_A = float('-inf')
        local_agg_lb_A += prob * lb_A
        
        # --- Fit & Bound: Method B ---
        if on_B:
            try:
                S_B = np.sqrt(w_B)
                Phi_w_B = Phi * S_B[:, np.newaxis]
                Y_w_B = Y_vals * S_B
                PhiT_Phi_w_B = Phi_w_B.T @ Phi_w_B + reg_matrix
                beta_B = np.linalg.solve(PhiT_Phi_w_B, Phi_w_B.T @ Y_w_B)
                lb_B, ms_pt, ms_val = _compute_bound_from_beta(beta_B, dim, lifted_vars, subproblems, subproblem_name, bounds, model, return_ms_point=True)
                local_beta_agg_B += prob * beta_B
                
                # Store MS point for method 'B'
                if 'B' in node.lb_problem.plot_data_wlsq:
                    node.lb_problem.plot_data_wlsq['B']['ms_point'][subproblem_name] = ms_pt
                    node.lb_problem.plot_data_wlsq['B']['ms_trueval'][subproblem_name] = ms_val
            except:
                lb_B = float('-inf')
        else:
            lb_B = float('-inf')
        local_agg_lb_B += prob * lb_B
        
        # --- Fit & Bound: Method C ---
        if on_C:
            try:
                S_C = np.sqrt(w_C)
                Phi_w_C = Phi_C * S_C[:, np.newaxis]
                Y_w_C = Y_vals_C * S_C
                PhiT_Phi_w_C = Phi_w_C.T @ Phi_w_C + (lambda_reg * np.eye(Phi_C.shape[1]))
                PhiT_Phi_w_C[0, 0] -= lambda_reg
                beta_C = np.linalg.solve(PhiT_Phi_w_C, Phi_w_C.T @ Y_w_C)
                lb_C, ms_pt, ms_val = _compute_bound_from_beta(beta_C, dim, lifted_vars, subproblems, subproblem_name, bounds, model, return_ms_point=True)
                local_beta_agg_C += prob * beta_C
                
                # Store MS point for method 'C'
                if 'C' in node.lb_problem.plot_data_wlsq:
                    node.lb_problem.plot_data_wlsq['C']['ms_point'][subproblem_name] = ms_pt
                    node.lb_problem.plot_data_wlsq['C']['ms_trueval'][subproblem_name] = ms_val
            except:
                lb_C = float('-inf')
        else:
            lb_C = float('-inf')
        local_agg_lb_C += prob * lb_C
        
        # --- Fit & Bound: Method D1 ---
        if on_D1:
            try:
                S_D1 = np.sqrt(w_D1)
                Phi_w_D1 = Phi * S_D1[:, np.newaxis]
                Y_w_D1 = Y_vals * S_D1
                PhiT_Phi_w_D1 = Phi_w_D1.T @ Phi_w_D1 + reg_matrix
                beta_D1 = np.linalg.solve(PhiT_Phi_w_D1, Phi_w_D1.T @ Y_w_D1)
                lb_D1, ms_pt, ms_val = _compute_bound_from_beta(beta_D1, dim, lifted_vars, subproblems, subproblem_name, bounds, model, return_ms_point=True)
                local_beta_agg_D1 += prob * beta_D1
                
                # Store MS point for method 'D1'
                if 'D1' in node.lb_problem.plot_data_wlsq:
                    node.lb_problem.plot_data_wlsq['D1']['ms_point'][subproblem_name] = ms_pt
                    node.lb_problem.plot_data_wlsq['D1']['ms_trueval'][subproblem_name] = ms_val
            except:
                lb_D1 = float('-inf')
        else:
            lb_D1 = float('-inf')
        local_agg_lb_D1 += prob * lb_D1
        
        # --- Fit & Bound: Method D2 ---
        if on_D2:
            try:
                S_D2 = np.sqrt(w_D2)
                Phi_w_D2 = Phi * S_D2[:, np.newaxis]
                Y_w_D2 = Y_vals * S_D2
                PhiT_Phi_w_D2 = Phi_w_D2.T @ Phi_w_D2 + reg_matrix
                beta_D2 = np.linalg.solve(PhiT_Phi_w_D2, Phi_w_D2.T @ Y_w_D2)
                lb_D2, ms_pt, ms_val = _compute_bound_from_beta(beta_D2, dim, lifted_vars, subproblems, subproblem_name, bounds, model, return_ms_point=True)
                local_beta_agg_D2 += prob * beta_D2
                
                # Store MS point for method 'D2'
                if 'D2' in node.lb_problem.plot_data_wlsq:
                    node.lb_problem.plot_data_wlsq['D2']['ms_point'][subproblem_name] = ms_pt
                    node.lb_problem.plot_data_wlsq['D2']['ms_trueval'][subproblem_name] = ms_val
            except:
                lb_D2 = float('-inf')
        else:
            lb_D2 = float('-inf')
        local_agg_lb_D2 += prob * lb_D2
        
        # --- Fit & Bound: Method E (Anchor-mixed sampling, uniform weights) ---
        Y_vals_E = data.get('Y_vals_E', np.array([]))
        if on_E and len(Y_vals_E) > 0 and len(samples_E) > 0:
            try:
                # Build Phi_E from samples_E
                Phi_E = []
                for x in samples_E:
                    row = [1.0]
                    for i in range(dim):
                        row.append(x[i])
                    for i in range(dim):
                        for j in range(i, dim):
                            row.append(x[i] * x[j])
                    Phi_E.append(row)
                Phi_E = np.array(Phi_E)
                
                # Uniform weighting (like Method Uniform)
                PhiT_Phi_E = Phi_E.T @ Phi_E + reg_matrix
                beta_E = np.linalg.solve(PhiT_Phi_E, Phi_E.T @ Y_vals_E)
                lb_E, ms_pt, ms_val = _compute_bound_from_beta(beta_E, dim, lifted_vars, subproblems, subproblem_name, bounds, model, return_ms_point=True)
                local_beta_agg_E += prob * beta_E
                
                # Store MS point for method 'E'
                if 'E' in node.lb_problem.plot_data_wlsq:
                    node.lb_problem.plot_data_wlsq['E']['ms_point'][subproblem_name] = ms_pt
                    node.lb_problem.plot_data_wlsq['E']['ms_trueval'][subproblem_name] = ms_val
            except:
                lb_E = float('-inf')
        else:
            lb_E = float('-inf')
        local_agg_lb_E += prob * lb_E
        
    # Aggregate across ranks
    local_arr = np.array([local_agg_lb_uniform, local_agg_lb_A, local_agg_lb_B, local_agg_lb_C, local_agg_lb_D1, local_agg_lb_D2, local_agg_lb_E])
    global_arr = MPI.COMM_WORLD.allreduce(local_arr, op=MPI.SUM)
    
    global_lb_uniform = global_arr[0]
    global_lb_A = global_arr[1]
    global_lb_B = global_arr[2]
    global_lb_C = global_arr[3]
    global_lb_D1 = global_arr[4]
    global_lb_D2 = global_arr[5]
    global_lb_E = global_arr[6]  # NEW
    
    # Store on node (Set to None if disabled to match legacy behavior/display)
    node.lb_problem.wlsq_uniform_bound = global_lb_uniform if on_U else None
    node.lb_problem.wlsq_A_bound = global_lb_A if on_A else None
    node.lb_problem.wlsq_B_bound = global_lb_B if on_B else None
    node.lb_problem.wlsq_C_bound = global_lb_C if on_C else None
    node.lb_problem.wlsq_D1_bound = global_lb_D1 if on_D1 else None
    node.lb_problem.wlsq_D2_bound = global_lb_D2 if on_D2 else None
    node.lb_problem.wlsq_E_bound = global_lb_E if on_E else None  # NEW
    
    # --- Compute UB_true for each method ---
    # 1. Aggregate betas
    local_betas = np.concatenate([
        local_beta_agg_uniform, local_beta_agg_A, local_beta_agg_B, 
        local_beta_agg_C, local_beta_agg_D1, local_beta_agg_D2, local_beta_agg_E
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
    g_beta_E = global_betas[6*bs:7*bs]  # NEW
    
    methods = [
        ('uniform', g_beta_uni), ('A', g_beta_A), ('B', g_beta_B), 
        ('C', g_beta_C), ('D1', g_beta_D1), ('D2', g_beta_D2), ('E', g_beta_E)
    ]
    
    # Cache evaluations to avoid re-evaluating same point
    eval_cache = {}
    
    for m_name, m_beta in methods:
        # Check flags (Is method enabled? Is UB enabled for method?)
        if not enabled_methods.get(m_name, False):
            setattr(node.lb_problem, f'wlsq_{m_name}_ub', float('nan'))
            continue
        if not enabled_ub_methods.get(m_name, False):
            setattr(node.lb_problem, f'wlsq_{m_name}_ub', float('nan'))
            continue

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
            
            # --- Store sol_point for plotting ---
            if m_name in node.lb_problem.plot_data_wlsq:
                # Validate x_star is finite
                if len(x_star) >= 3 and all(np.isfinite(x_star[i]) for i in range(3)):
                    node.lb_problem.plot_data_wlsq[m_name]['sol_point'] = [float(x_star[0]), float(x_star[1]), float(x_star[2])]
                    
                    # Compute per-scenario trueval at sol_point for LOCAL scenarios
                    sol_truevals = {}
                    id_to_val = {(lifted_vars[i]['type'], lifted_vars[i]['id']): x_star[i] for i in range(dim)}
                    
                    for subproblem_name in subproblems.names:
                        model = subproblems.model[subproblem_name]
                        # Save state
                        saved_state_local = {}
                        for var in subproblems.subproblem_lifted_vars[subproblem_name]:
                            saved_state_local[id(var)] = (var.lb, var.ub, var.is_fixed(), var.value)
                        
                        # Fix vars to x_star
                        for var in subproblems.subproblem_lifted_vars[subproblem_name]:
                            v_type, v_id, _ = subproblems.var_to_data[var]
                            if (v_type, v_id) in id_to_val:
                                var.fix(id_to_val[(v_type, v_id)])
                        
                        # Solve
                        success, obj_val = _solve_true_recourse_primal_gurobi(model)
                        sol_truevals[subproblem_name] = obj_val if success else 1e6
                        
                        # Restore state
                        for var in subproblems.subproblem_lifted_vars[subproblem_name]:
                            lb_v, ub_v, is_fixed, val = saved_state_local[id(var)]
                            var.lb = lb_v
                            var.ub = ub_v
                            if is_fixed: var.fix(val)
                            else: 
                                var.unfix()
                                var.value = val
                    
                    node.lb_problem.plot_data_wlsq[m_name]['sol_trueval'] = sol_truevals
        
        # Store on node
        setattr(node.lb_problem, f'wlsq_{m_name}_ub', ub_true)


    # --- OBBT Diagnostic for Uniform ---
    # Store aggregated beta and mbar for uniform
    # m_s = lb_s - min_box(Q_s)
    # We need to reconstruct m_s accumulation. 
    # Since we didn't accumulate m_s in the loop, we can infer it:
    # global_lb_uniform = sum(prob * (min_Q_s + m_s)) = sum(prob*min_Q_s) + sum(prob*m_s)
    # So mbar = global_lb_uniform - min_box(Q_bar)
    # This is valid because sum(prob * min(Q_s)) != min(sum prob*Q_s) generally, 
    # BUT the "shift" logic in WLSQ is: Q_s_shifted = Q_s + m_s.
    # The aggregated surrogate is Q_bar = sum(prob * Q_s).
    # The aggregated lower bound function is L(x) = sum(prob * (Q_s(x) + m_s)) = Q_bar(x) + mbar.
    # So we can just define mbar = global_lb_uniform - min_box(Q_bar).
    
    if on_U:
        min_Q_bar, _ = _minimize_quadratic(g_beta_uni, dim, bounds)
        if math.isfinite(min_Q_bar):
            mbar_uniform = global_lb_uniform - min_Q_bar
            node.lb_problem.wlsq_uniform_beta = g_beta_uni
            node.lb_problem.wlsq_uniform_mbar = mbar_uniform
    
    return max(global_lb_uniform if on_U else float('-inf'), 
               global_lb_A if on_A else float('-inf'),
               global_lb_B if on_B else float('-inf'),
               global_lb_C if on_C else float('-inf'),
               global_lb_D1 if on_D1 else float('-inf'),
               global_lb_D2 if on_D2 else float('-inf'),
               global_lb_E if on_E else float('-inf'))

def run_surrogate_obbt_uniform(node: Node, ub: float) -> Optional[float]:
    """
    Runs OBBT on the WLSQ Uniform surrogate: Q_bar(x) + mbar <= UB.
    Returns volume ratio (new_vol / old_vol) or None if failed.
    """
    if not hasattr(node.lb_problem, 'wlsq_uniform_beta') or \
       not hasattr(node.lb_problem, 'wlsq_uniform_mbar'):
        return None
        
    beta = node.lb_problem.wlsq_uniform_beta
    mbar = node.lb_problem.wlsq_uniform_mbar
    
    # Identify vars (re-identify to be safe, or pass in)
    # We need the lifted_vars list to map back to node state
    # For now, we'll assume the node state has the 3 PID vars in SupportedVars.reals
    # and we can find them. But to be precise, we need the exact mapping used in WLSQ.
    # We can re-use _identify_pid_vars if we had subproblems, but we don't have them here easily.
    # However, we know the dimension is 3 and the order is Kp, Ki, Kd (from _identify_pid_vars).
    
    # Let's try to find them in the node state
    # We know they are reals.
    if SupportedVars.reals not in node.state: return None
    
    # We need to match the order used in WLSQ. 
    # WLSQ uses _identify_pid_vars which sorts by name length/etc.
    # We should probably pass lifted_vars or bounds to this function, 
    # but to keep signature simple, we'll try to reconstruct.
    # Actually, we can just use the bounds from the node state directly if we find the 3 vars.
    
    # Heuristic: Find 3 vars with "kp", "ki", "kd" in name
    vars_found = []
    for vid, vdata in node.state[SupportedVars.reals].items():
        name = vdata.name.lower()
        if 'kp' in name or 'k_p' in name: vars_found.append(('kp', vid, vdata))
        elif 'ki' in name or 'k_i' in name: vars_found.append(('ki', vid, vdata))
        elif 'kd' in name or 'k_d' in name: vars_found.append(('kd', vid, vdata))
            
    # Sort by role to match WLSQ order: kp, ki, kd
    vars_found.sort(key=lambda x: {'kp':0, 'ki':1, 'kd':2}.get(x[0], 99))
    
    if len(vars_found) != 3:
        print(f"DEBUG: OBBT var identification failed. Found {len(vars_found)} vars: {[v[0] for v in vars_found]}")
        return None
    
    dim = 3
    bounds = [(v[2].lb, v[2].ub) for v in vars_found]
    
    # Build Pyomo model
    m = pyo.ConcreteModel()
    m.x = pyo.Var(range(dim), bounds=lambda m, i: bounds[i])
    
    # Quadratic constraint: x'Qx + c'x + b + mbar <= UB
    b_s = beta[0]
    c_s = beta[1:dim+1]
    quad_coeffs = beta[dim+1:]
    
    def quad_rule(m):
        expr = b_s + mbar
        # Linear
        for i in range(dim):
            expr += c_s[i] * m.x[i]
        # Quadratic
        idx = 0
        for i in range(dim):
            for j in range(i, dim):
                val = quad_coeffs[idx]
                if i == j:
                    expr += val * m.x[i] * m.x[i]
                else:
                    expr += val * m.x[i] * m.x[j]
                idx += 1
        return expr <= ub
        
    m.q_con = pyo.Constraint(rule=quad_rule)
    
    # Solve min/max for each var
    solver = SolverFactory('gurobi')
    solver.options['NonConvex'] = 2
    solver.options['TimeLimit'] = 2
    solver.options['OutputFlag'] = 0
    
    new_bounds = []
    
    try:
        for i in range(dim):
            # Min
            m.obj = pyo.Objective(expr=m.x[i], sense=pyo.minimize)
            res = solver.solve(m)
            if (res.solver.status == pyo.SolverStatus.ok and 
                res.solver.termination_condition == pyo.TerminationCondition.optimal):
                lb_new = pyo.value(m.x[i])
            else:
                lb_new = bounds[i][0] # Fallback
                
            # Max
            m.obj = pyo.Objective(expr=m.x[i], sense=pyo.maximize)
            res = solver.solve(m)
            if (res.solver.status == pyo.SolverStatus.ok and 
                res.solver.termination_condition == pyo.TerminationCondition.optimal):
                ub_new = pyo.value(m.x[i])
            else:
                ub_new = bounds[i][1] # Fallback
                
            # Intersect with original
            lb_final = max(bounds[i][0], lb_new)
            ub_final = min(bounds[i][1], ub_new)
            
            # Safety
            if lb_final > ub_final + 1e-6: return None # Infeasible
            lb_final = min(lb_final, ub_final)
            
            new_bounds.append((lb_final, ub_final))
            
        # Compute volume ratio
        old_vol = 1.0
        new_vol = 1.0
        for i in range(dim):
            old_vol *= (bounds[i][1] - bounds[i][0])
            new_vol *= (new_bounds[i][1] - new_bounds[i][0])
            
        if old_vol < 1e-12: return 1.0
        return new_vol / old_vol
        
    except Exception as e:
        print(f"DEBUG: OBBT solve failed: {e}")
        return None

def _compute_bound_from_beta(beta, dim, lifted_vars, subproblems, subproblem_name, bounds, model, return_ms_point=False):
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
        
        if return_ms_point:
            # Extract MS point
            ms_point = [pyo.value(v) for v in pyomo_vars]
            
            # Compute true recourse at MS point
            # Fix vars
            id_to_val = {(lifted_vars[i]['type'], lifted_vars[i]['id']): ms_point[i] for i in range(dim)}
            for var in subproblems.subproblem_lifted_vars[subproblem_name]:
                v_type, v_id, _ = subproblems.var_to_data[var]
                if (v_type, v_id) in id_to_val:
                    var.fix(id_to_val[(v_type, v_id)])
            
            success, ms_true_val = _solve_true_recourse_primal_gurobi(model)
            if not success:
                ms_true_val = float('inf')
                
            # Restore vars
            for var in subproblems.subproblem_lifted_vars[subproblem_name]:
                var.unfix()

            return lb_s, ms_point, ms_true_val
            
        return lb_s
    except:
        if return_ms_point:
            return float('-inf'), None, float('inf')
        return float('-inf')
