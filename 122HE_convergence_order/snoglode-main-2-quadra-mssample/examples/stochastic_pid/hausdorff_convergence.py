import os
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import time

# Add repo root to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import snoglode as sno
import snoglode.utils.MPI as MPI
from snoglode.components.node import Node
from snoglode.utils.supported import SupportedVars
from snoglode.utils.wls_quadratic_bound import compute_wls_quadratic_surrogate_bound
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition, SolverStatus
from pyomo.contrib.alternative_solutions.aos_utils import get_active_objective
import pyomo.dae as dae 

# --- Constants from stochastic_pid.py ---
Kp_ref = -9.988319
Ki_ref = -99.987421
Kd_ref = 0.850030
rp = 10
ri = 100
rd = 100

Kp_root = (-10.0, 10.0)
Ki_root = (-100.0, 100.0)
Kd_root = (-100.0, 1000.0)

# Global for scenario count
NUM_SCENARIOS = 5

# WLSQ seed for reproducibility
WLSQ_SEED = 17

# --- Helper Functions ---
def clip_interval(lb, ub, root):
    return (max(lb, root[0]), min(ub, root[1]))

def compute_hausdorff_linf(Kp_star, Ki_star, Kd_star, Kp_l, Kp_u, Ki_l, Ki_u, Kd_l, Kd_u):
    """
    Compute L_infinity Hausdorff distance from singleton {x*} to box B_k.
    H = sup_{b in B_k} ||b - x*||_inf = max distance from x* to any corner.
    Returns (H, inside_box).
    """
    inside_box = (Kp_l <= Kp_star <= Kp_u) and (Ki_l <= Ki_star <= Ki_u) and (Kd_l <= Kd_star <= Kd_u)
    H = max(
        max(abs(Kp_star - Kp_l), abs(Kp_star - Kp_u)),
        max(abs(Ki_star - Ki_l), abs(Ki_star - Ki_u)),
        max(abs(Kd_star - Kd_l), abs(Kd_star - Kd_u)),
    )
    return (H, inside_box)

# --- Model Builder ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "data.csv")
if not os.path.exists(DATA_PATH):
    DATA_PATH = "data.csv"

try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"WARNING: data.csv not found at {DATA_PATH}.")
    sys.exit(1)

sp = 0.5

def build_pid_model(scenario_name):
    _, scen_num = scenario_name.split("_")
    idx = int(scen_num) - 1
    if idx < 0 or idx >= len(df):
        raise ValueError(f"Scenario index {idx} out of range for data with {len(df)} rows")
    row_data = df.iloc[idx]
    tau_xs = float(row_data["tau_xs"])
    tau_us = float(row_data["tau_us"])
    tau_ds = float(row_data["tau_ds"])
    num_disturbances = sum(1 for header in df.columns.tolist() if "disturbance" in header)
    disturbance = [float(row_data[f"disturbance_{i}"]) for i in range(num_disturbances)]
    setpoint_change = sp

    m = pyo.ConcreteModel()
    T = 15
    m.time = pyo.RangeSet(0,T)
    m.t = dae.ContinuousSet(bounds=(0,T))
    m.x_setpoint = pyo.Param(initialize=setpoint_change)
    m.tau_xs = pyo.Param(initialize=tau_xs)
    m.tau_us = pyo.Param(initialize=tau_us)
    m.tau_ds = pyo.Param(initialize=tau_ds)
    m.d_s = pyo.Param(m.t, initialize=0, mutable=True)
    m.K_p = pyo.Var(domain=pyo.Reals, bounds=Kp_root)
    m.K_i = pyo.Var(domain=pyo.Reals, bounds=Ki_root)
    m.K_d = pyo.Var(domain=pyo.Reals, bounds=Kd_root)
    m.x_s = pyo.Var(m.t, domain=pyo.Reals, bounds=[-2.5, 2.5])
    m.e_s = pyo.Var(m.t, domain=pyo.Reals)
    m.u_s = pyo.Var(m.t, domain=pyo.Reals, bounds=[-5.0, 5.0])
    m.dxdt = dae.DerivativeVar(m.x_s, wrt=m.t)
    m.dedt = dae.DerivativeVar(m.e_s, wrt=m.t)

    @m.Constraint(m.t)
    def dxdt_con(m, t):
        if t == m.t.first(): return pyo.Constraint.Skip
        return m.dxdt[t] == -m.tau_xs*m.x_s[t] + m.tau_us*m.u_s[t] + m.tau_ds*m.d_s[t]
    m.x_init_cond = pyo.Constraint(expr=m.x_s[m.t.first()] == 0)

    @m.Constraint(m.t)
    def e_con(m, t):
        return m.e_s[t] == m.x_s[t] - m.x_setpoint

    m.I = pyo.Var(m.t)
    @m.Constraint(m.t)
    def integral(m, t):
        if t == m.t.first(): return m.I[t] == 0
        return m.I[t] == m.I[m.t.prev(t)] + (t - m.t.prev(t)) * m.e_s[t]

    @m.Constraint(m.t)
    def u_con(m, t):
        return m.u_s[t] == m.K_p*m.e_s[t] + m.K_i*m.I[t] + m.K_d*m.dedt[t]

    def e_sq_integral_rule(m, t):
        return 10*m.e_s[t]**2 + 0.01*m.u_s[t]**2
    m.e_sq_integral = dae.Integral(m.t, wrt=m.t, rule=e_sq_integral_rule)
    m.obj = pyo.Objective(sense=pyo.minimize, expr=m.e_sq_integral)

    discretizer = pyo.TransformationFactory('dae.finite_difference')
    discretizer.apply_to(m, nfe=20, wrt=m.t, scheme='BACKWARD')
    index = 0
    for t in m.t:
        m.d_s[t] = disturbance[index]
        index += 1

    first_stage = {"K_p": m.K_p, "K_i": m.K_i, "K_d": m.K_d}
    probability = 1.0 / NUM_SCENARIOS
    return [m, first_stage, probability]

# --- Lower Bounder ---
ipopt = pyo.SolverFactory("ipopt")

class GurobiLBLowerBounder(sno.AbstractLowerBounder):
    def __init__(self, solver, time_ub=600):
        super().__init__(solver=solver, time_ub=time_ub)
        self.iter = 0
        self.current_milp_gap = solver.options.get("MIPGap", 1e-2)

    def solve_a_subproblem(self, subproblem_model, *args, **kwargs):
        self.iter += 1
        
        # Warm start with IPOPT if available
        if ipopt.available():
            try:
                ipopt.solve(subproblem_model, load_solutions=True)
            except:
                pass
        
        # Solve with Gurobi
        results = self.opt.solve(subproblem_model, load_solutions=False, symbolic_solver_labels=True, tee=False)
        
        # If Gurobi hit time limit, fall back to IPOPT
        if results.solver.termination_condition == TerminationCondition.maxTimeLimit:
            if ipopt.available():
                try:
                    ipopt.solve(subproblem_model, load_solutions=True)
                    return True, pyo.value(get_active_objective(subproblem_model))
                except:
                    return False, None
            else:
                return False, None
        
        # Check Gurobi results
        if results.solver.termination_condition == TerminationCondition.optimal and results.solver.status == SolverStatus.ok:
            subproblem_model.solutions.load_from(results)
            if self.current_milp_gap > 0 and hasattr(results.problem, 'lower_bound') and results.problem.lower_bound is not None:
                parent_obj = pyo.value(subproblem_model.successor_obj)
                return True, max(parent_obj, results.problem.lower_bound)
            else:
                return True, pyo.value(get_active_objective(subproblem_model))
        elif results.solver.termination_condition == TerminationCondition.infeasible:
            return False, None
        else:
            return False, None

# --- Main Script ---
def main():
    global NUM_SCENARIOS
    parser = argparse.ArgumentParser(description="Estimate Hausdorff convergence order with WLSQ LB.")
    parser.add_argument("--k_min", type=int, default=0, help="Min k")
    parser.add_argument("--k_max", type=int, default=8, help="Max k")
    parser.add_argument("--scenarios", type=int, default=5, help="Number of scenarios")
    parser.add_argument("--outdir", type=str, default="plots_hausdorff", help="Output directory")
    parser.add_argument("--Kp_star", type=float, default=None, help="True optimal Kp")
    parser.add_argument("--Ki_star", type=float, default=None, help="True optimal Ki")
    parser.add_argument("--Kd_star", type=float, default=None, help="True optimal Kd")
    parser.add_argument("--use_wlsq", action="store_true", help="Also compute WLSQ quadratic bound")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Set x* (true optimal point)
    if args.Kp_star is None or args.Ki_star is None or args.Kd_star is None:
        print("WARNING: --Kp_star, --Ki_star, --Kd_star not provided. Using reference point as proxy for x*.")
        Kp_star, Ki_star, Kd_star = Kp_ref, Ki_ref, Kd_ref
    else:
        Kp_star, Ki_star, Kd_star = args.Kp_star, args.Ki_star, args.Kd_star

    print(f"x* = (Kp={Kp_star}, Ki={Ki_star}, Kd={Kd_star})")
    if args.use_wlsq:
        print("WLSQ quadratic bound computation ENABLED")

    NUM_SCENARIOS = args.scenarios
    scenarios = [f"scen_{i}" for i in range(1, NUM_SCENARIOS+1)]

    nonconvex_gurobi_lb = pyo.SolverFactory("gurobi")
    nonconvex_gurobi_lb.options["NonConvex"] = 2
    nonconvex_gurobi_lb.options["MIPGap"] = 1e-2
    nonconvex_gurobi_lb.options["TimeLimit"] = 15

    nonconvex_gurobi = pyo.SolverFactory("gurobi")
    nonconvex_gurobi.options["NonConvex"] = 2

    params = sno.SolverParameters(subproblem_names=scenarios,
                                  subproblem_creator=build_pid_model,
                                  lb_solver=nonconvex_gurobi_lb,
                                  cg_solver=nonconvex_gurobi,
                                  ub_solver=nonconvex_gurobi)
    params.set_bounders(candidate_solution_finder=sno.SolveExtensiveForm,
                        lower_bounder=GurobiLBLowerBounder)
    params.set_bounds_tightening(fbbt=True, obbt=False)
    solver = sno.Solver(params)
    
    # MS Point Repositories for WLSQ (empty, but required)
    ms_repos = {}

    results_data = []
    print(f"Starting Hausdorff convergence estimation k={args.k_min}..{args.k_max}")

    for k in range(args.k_min, args.k_max + 1):
        scale = 2**(-k)
        Kp_l, Kp_u = clip_interval(Kp_ref - rp*scale, Kp_ref + rp*scale, Kp_root)
        Ki_l, Ki_u = clip_interval(Ki_ref - ri*scale, Ki_ref + ri*scale, Ki_root)
        Kd_l, Kd_u = clip_interval(Kd_ref - rd*scale, Kd_ref + rd*scale, Kd_root)
        diam_k = max(Kp_u - Kp_l, Ki_u - Ki_l, Kd_u - Kd_l)

        print(f"\n--- k={k}, scale={scale:.2e}, diam={diam_k:.2e} ---")

        # Create Node with specific state
        state = copy.deepcopy(solver.subproblems.root_node_state)
        state[SupportedVars.reals]["K_p"].lb = Kp_l
        state[SupportedVars.reals]["K_p"].ub = Kp_u
        state[SupportedVars.reals]["K_i"].lb = Ki_l
        state[SupportedVars.reals]["K_i"].ub = Ki_u
        state[SupportedVars.reals]["K_d"].lb = Kd_l
        state[SupportedVars.reals]["K_d"].ub = Kd_u
        to_branch = {var_type: list(state[var_type]) for var_type in SupportedVars}
        node = Node(state=state, to_branch=to_branch, id=0)
        node.depth = 0
        solver.subproblems.set_all_states(node.state)
        
        # === Compute Primary LB (Gurobi) ===
        t0 = time.time()
        solver.lower_bounder.solve(node=node, subproblems=solver.subproblems)
        lb_time = time.time() - t0
        LB_gurobi = node.lb_problem.objective
        print(f"  LB (Gurobi): {LB_gurobi} (time: {lb_time:.2f}s)")

        # === Compute WLSQ LB (if enabled) ===
        LB_wlsq = float('-inf')
        wlsq_time = 0.0
        if args.use_wlsq:
            t1 = time.time()
            try:
                # Enable only uniform method for simplicity
                enabled_methods = {'uniform': True, 'A': False, 'B': False, 'C': False, 'D1': False, 'D2': False, 'E': False, 'F': False}
                enabled_ub_methods = {'uniform': False, 'A': False, 'B': False, 'C': False, 'D1': False, 'D2': False, 'E': False, 'F': False}
                
                LB_wlsq = compute_wls_quadratic_surrogate_bound(
                    node=node,
                    subproblems=solver.subproblems,
                    solver=solver.lower_bounder.opt,
                    seed=WLSQ_SEED,
                    enabled_methods=enabled_methods,
                    enabled_ub_methods=enabled_ub_methods,
                    ms_repos=ms_repos,
                    iteration=k
                )
                wlsq_time = time.time() - t1
                print(f"  LB (WLSQ):   {LB_wlsq} (time: {wlsq_time:.2f}s)")
            except Exception as e:
                print(f"  LB (WLSQ):   FAILED ({e})")
                LB_wlsq = float('-inf')

        # === Combined LB (max of both) ===
        LB_combined = max(LB_gurobi, LB_wlsq) if args.use_wlsq else LB_gurobi
        if args.use_wlsq:
            print(f"  LB (Max):    {LB_combined}")

        # Compute Hausdorff distance
        H_k, inside_box = compute_hausdorff_linf(Kp_star, Ki_star, Kd_star, Kp_l, Kp_u, Ki_l, Ki_u, Kd_l, Kd_u)
        if not inside_box:
            print(f"  WARNING: x* is outside B_k!")
        print(f"  Hausdorff H_k: {H_k} (inside_box={inside_box})")

        results_data.append({
            "k": k,
            "Kp_lb": Kp_l, "Kp_ub": Kp_u,
            "Ki_lb": Ki_l, "Ki_ub": Ki_u,
            "Kd_lb": Kd_l, "Kd_ub": Kd_u,
            "diam_Linf": diam_k,
            "H": H_k,
            "inside_box": inside_box,
            "LB_gurobi": LB_gurobi,
            "LB_wlsq": LB_wlsq if args.use_wlsq else None,
            "LB_combined": LB_combined,
            "solve_time_seconds": lb_time + wlsq_time
        })

    df_res = pd.DataFrame(results_data)
    csv_path = os.path.join(args.outdir, "hausdorff_convergence.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # Regression & Plotting
    df_valid = df_res[df_res["H"] > 0].copy()
    if len(df_valid) > 1:
        log_diam = np.log(df_valid["diam_Linf"])
        log_H = np.log(df_valid["H"])
        slope, intercept = np.polyfit(log_diam, log_H, 1)
        fit_log_H = intercept + slope * log_diam
        ss_res = np.sum((log_H - fit_log_H)**2)
        ss_tot = np.sum((log_H - np.mean(log_H))**2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
        print(f"\nConvergence Order p: {slope:.4f}")
        print(f"R^2: {r_squared:.4f}")

        plt.figure()
        plt.loglog(df_valid["diam_Linf"], df_valid["H"], 'o-', label='Data')
        fit_y = np.exp(intercept + slope * log_diam)
        plt.loglog(df_valid["diam_Linf"], fit_y, '--', label=f'Fit (p={slope:.2f})')
        plt.xlabel('Diameter (L_inf)')
        plt.ylabel('Hausdorff Distance H')
        plt.title(f'Convergence Order: p={slope:.2f}, R^2={r_squared:.2f}')
        plt.legend()
        plt.grid(True, which="both", ls="-")
        plt.savefig(os.path.join(args.outdir, "hausdorff_vs_diam.png"))
        plt.close()

        plt.figure()
        plt.semilogy(df_res["k"], df_res["H"], 'o-')
        plt.xlabel('k')
        plt.ylabel('Hausdorff Distance H')
        plt.title('Hausdorff vs Iteration k')
        plt.grid(True)
        plt.savefig(os.path.join(args.outdir, "hausdorff_vs_k.png"))
        plt.close()
    else:
        print("Not enough valid data points for regression.")

if __name__ == "__main__":
    main()
