#!/usr/bin/env python
"""
wlsq_convergence.py - Estimate convergence order of WLSQ-based lower bounds.

This script computes various WLSQ lower bounds on shrinking boxes and estimates
the convergence order by fitting log(gap) = a + p*log(diam) where gap = F_TRUE - LB.
"""
import os
os.environ['MKL_THREADING_LAYER'] = 'SEQUENTIAL'

import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import copy
import time
import math

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

# --- Constants ---
Kp_ref = -9.988319
Ki_ref = -99.987421
Kd_ref = 0.850030
rp = 10
ri = 100
rd = 100

Kp_root = (-10.0, 10.0)
Ki_root = (-100.0, 100.0)
Kd_root = (-100.0, 1000.0)

# Trusted true optimal value
F_TRUE = 0.9735317475835165

# Global scenario count
NUM_SCENARIOS = 5

# WLSQ method keys (exact keys used in repo: solver.py and wls_quadratic_bound.py)
WLSQ_METHOD_KEYS = ['uniform', 'A', 'B', 'C', 'D1', 'D2', 'E', 'F']

# Default k values to evaluate (set to None to use CLI args k_min/k_max/k_step)
# Example: DEFAULT_K_LIST = [0, 2, 4, 6, 8, 10]
DEFAULT_K_LIST = [0, 2, 4, 6, 8, 10]

# --- Helper Functions ---
def clip_interval(lb, ub, root):
    """Clip interval to root bounds: (shrinking interval) ∩ (root interval)"""
    return (max(lb, root[0]), min(ub, root[1]))

# --- Model Builder ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(SCRIPT_DIR, "data.csv")
if not os.path.exists(DATA_PATH):
    DATA_PATH = "data.csv"

try:
    df = pd.read_csv(DATA_PATH)
except FileNotFoundError:
    print(f"ERROR: data.csv not found at {DATA_PATH}.")
    sys.exit(1)

sp = 0.5

def build_pid_model(scenario_name):
    _, scen_num = scenario_name.split("_")
    idx = int(scen_num) - 1
    if idx < 0 or idx >= len(df):
        raise ValueError(f"Scenario index {idx} out of range")
    row_data = df.iloc[idx]
    tau_xs = float(row_data["tau_xs"])
    tau_us = float(row_data["tau_us"])
    tau_ds = float(row_data["tau_ds"])
    num_disturbances = sum(1 for h in df.columns if "disturbance" in h)
    disturbance = [float(row_data[f"disturbance_{i}"]) for i in range(num_disturbances)]

    m = pyo.ConcreteModel()
    T = 15
    m.time = pyo.RangeSet(0, T)
    m.t = dae.ContinuousSet(bounds=(0, T))
    m.x_setpoint = pyo.Param(initialize=sp)
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

# --- Lower Bounder (for baseline) ---
ipopt = pyo.SolverFactory("ipopt")

class GurobiLBLowerBounder(sno.AbstractLowerBounder):
    def __init__(self, solver, time_ub=600):
        super().__init__(solver=solver, time_ub=time_ub)
        self.iter = 0
        self.current_milp_gap = solver.options.get("MIPGap", 1e-2)

    def solve_a_subproblem(self, subproblem_model, *args, **kwargs):
        self.iter += 1
        if ipopt.available():
            try:
                ipopt.solve(subproblem_model, load_solutions=True)
            except:
                pass
        results = self.opt.solve(subproblem_model, load_solutions=False, symbolic_solver_labels=True, tee=False)
        if results.solver.termination_condition == TerminationCondition.maxTimeLimit:
            if ipopt.available():
                try:
                    ipopt.solve(subproblem_model, load_solutions=True)
                    return True, pyo.value(get_active_objective(subproblem_model))
                except:
                    return False, None
            return False, None
        if results.solver.termination_condition == TerminationCondition.optimal and results.solver.status == SolverStatus.ok:
            subproblem_model.solutions.load_from(results)
            # Fix: successor_obj may not exist in standalone context
            if self.current_milp_gap > 0 and hasattr(results.problem, 'lower_bound') and results.problem.lower_bound is not None:
                if hasattr(subproblem_model, 'successor_obj'):
                    parent_obj = pyo.value(subproblem_model.successor_obj)
                else:
                    parent_obj = float('-inf')
                return True, max(parent_obj, results.problem.lower_bound)
            return True, pyo.value(get_active_objective(subproblem_model))
        elif results.solver.termination_condition == TerminationCondition.infeasible:
            return False, None
        return False, None

# --- Main Script ---
def main():
    global NUM_SCENARIOS
    parser = argparse.ArgumentParser(description="Estimate WLSQ LB convergence order.")
    parser.add_argument("--k_min", type=int, default=0)
    parser.add_argument("--k_max", type=int, default=8)
    parser.add_argument("--k_step", type=int, default=1, help="Step size for k iteration")
    parser.add_argument("--k_list", type=str, default="",
                        help="Explicit comma-separated list of k values (overrides k_min/k_max/k_step)")
    parser.add_argument("--scenarios", type=int, default=5)
    parser.add_argument("--outdir", type=str, default="plots_wlsq_conv")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--methods", type=str, default="uniform,E,F",
                        help="Comma-separated list of WLSQ methods to enable (from: uniform,A,B,C,D1,D2,E,F)")
    parser.add_argument("--debug_wlsq", action="store_true",
                        help="Print debug info about WLSQ return values and node.lb_problem attributes")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Determine k values to use (priority: DEFAULT_K_LIST > --k_list > k_min/k_max/k_step)
    if DEFAULT_K_LIST is not None:
        k_values = sorted(set(DEFAULT_K_LIST))
    elif args.k_list.strip():
        # Parse explicit k list
        try:
            k_values = sorted(set(int(x.strip()) for x in args.k_list.split(',') if x.strip()))
        except ValueError as e:
            print(f"ERROR: Invalid --k_list format: {e}")
            sys.exit(1)
    else:
        # Use k_min, k_max, k_step
        k_values = list(range(args.k_min, args.k_max + 1, args.k_step))
    
    print(f"k values to evaluate: {k_values}")

    # Parse enabled methods
    enabled_method_list = [m.strip() for m in args.methods.split(',') if m.strip()]
    for m in enabled_method_list:
        if m not in WLSQ_METHOD_KEYS:
            print(f"WARNING: Unknown method '{m}'. Valid keys: {WLSQ_METHOD_KEYS}")
    
    print(f"Enabled WLSQ methods: {enabled_method_list}")

    NUM_SCENARIOS = args.scenarios
    scenarios = [f"scen_{i}" for i in range(1, NUM_SCENARIOS+1)]

    # Solver setup
    nonconvex_gurobi_lb = pyo.SolverFactory("gurobi")
    nonconvex_gurobi_lb.options["NonConvex"] = 2
    nonconvex_gurobi_lb.options["MIPGap"] = 1e-2
    nonconvex_gurobi_lb.options["TimeLimit"] = 15

    nonconvex_gurobi = pyo.SolverFactory("gurobi")
    nonconvex_gurobi.options["NonConvex"] = 2

    params = sno.SolverParameters(
        subproblem_names=scenarios,
        subproblem_creator=build_pid_model,
        lb_solver=nonconvex_gurobi_lb,
        cg_solver=nonconvex_gurobi,
        ub_solver=nonconvex_gurobi
    )
    params.set_bounders(
        candidate_solution_finder=sno.SolveExtensiveForm,
        lower_bounder=GurobiLBLowerBounder
    )
    params.set_bounds_tightening(fbbt=True, obbt=False)
    solver = sno.Solver(params)

    # MS Point Repositories - initialized as empty dict (same as solver.py line 154)
    ms_repos = {}

    # Method names to track (Orig + WLSQ methods)
    all_methods = ['Orig'] + enabled_method_list

    results_data = []
    print(f"Starting WLSQ convergence estimation")
    print(f"F_TRUE = {F_TRUE}, seed = {args.seed}")

    for k in k_values:
        scale = 2**(-k)
        Kp_l, Kp_u = clip_interval(Kp_ref - rp*scale, Kp_ref + rp*scale, Kp_root)
        Ki_l, Ki_u = clip_interval(Ki_ref - ri*scale, Ki_ref + ri*scale, Ki_root)
        Kd_l, Kd_u = clip_interval(Kd_ref - rd*scale, Kd_ref + rd*scale, Kd_root)
        diam_k = max(Kp_u - Kp_l, Ki_u - Ki_l, Kd_u - Kd_l)

        print(f"\n--- k={k}, diam={diam_k:.4e} ---")

        # Create Node
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

        row = {
            "k": k,
            "Kp_lb": Kp_l, "Kp_ub": Kp_u,
            "Ki_lb": Ki_l, "Ki_ub": Ki_u,
            "Kd_lb": Kd_l, "Kd_ub": Kd_u,
            "diam_Linf": diam_k
        }

        # --- Baseline LB (Orig) ---
        t0 = time.time()
        try:
            solver.lower_bounder.solve(node=node, subproblems=solver.subproblems)
            LB_orig = node.lb_problem.objective
        except Exception as e:
            print(f"  Orig LB failed: {e}")
            LB_orig = float('nan')
        time_orig = time.time() - t0
        row["LB_Orig"] = LB_orig
        row["gap_Orig"] = F_TRUE - LB_orig if math.isfinite(LB_orig) else float('nan')
        row["time_Orig"] = time_orig
        print(f"  Orig: LB={LB_orig:.6f}, gap={row['gap_Orig']:.6e}, time={time_orig:.2f}s")

        # --- WLSQ Methods ---
        enabled_methods = {m: (m in enabled_method_list) for m in WLSQ_METHOD_KEYS}
        enabled_ub_methods = {m: False for m in WLSQ_METHOD_KEYS}

        t1 = time.time()
        wlsq_return_val = None
        try:
            wlsq_return_val = compute_wls_quadratic_surrogate_bound(
                node=node,
                subproblems=solver.subproblems,
                solver=solver.lower_bounder.opt,
                seed=args.seed,
                enabled_methods=enabled_methods,
                enabled_ub_methods=enabled_ub_methods,
                ms_repos=ms_repos,
                iteration=k
            )
        except Exception as e:
            print(f"  WLSQ computation failed: {e}")
            import traceback
            traceback.print_exc()
        time_wlsq = time.time() - t1

        if args.debug_wlsq:
            print(f"  [DEBUG] WLSQ return type: {type(wlsq_return_val)}, value: {wlsq_return_val}")
            wlsq_attrs = [a for a in dir(node.lb_problem) if 'wlsq' in a.lower() or 'wls' in a.lower()]
            print(f"  [DEBUG] node.lb_problem WLSQ attrs: {wlsq_attrs}")
            for attr in wlsq_attrs:
                val = getattr(node.lb_problem, attr, None)
                if val is not None and not callable(val):
                    val_str = f"{val:.6f}" if isinstance(val, float) else str(val)[:50]
                    print(f"    {attr} = {val_str}")

        for method in enabled_method_list:
            attr_name = f"wlsq_{method}_bound"
            lb_val = getattr(node.lb_problem, attr_name, None)
            if lb_val is None or not math.isfinite(lb_val):
                lb_val = float('nan')
            row[f"LB_{method}"] = lb_val
            row[f"gap_{method}"] = F_TRUE - lb_val if math.isfinite(lb_val) else float('nan')
            row[f"time_{method}"] = time_wlsq
            if math.isfinite(lb_val):
                print(f"  {method}: LB={lb_val:.6f}, gap={row[f'gap_{method}']:.6e}")
            else:
                print(f"  {method}: FAILED or NaN")

        row["time_wlsq_total"] = time_wlsq
        results_data.append(row)

    # Save CSV
    df_res = pd.DataFrame(results_data)
    csv_path = os.path.join(args.outdir, "wlsq_convergence.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"\nResults saved to {csv_path}")

    # --- Regression & Plotting ---
    colors = {
        'Orig': 'black', 'uniform': 'blue', 'A': 'green', 'B': 'red',
        'C': 'purple', 'D1': 'orange', 'D2': 'brown', 'E': 'cyan', 'F': 'magenta'
    }
    markers = {
        'Orig': 'D', 'uniform': 'o', 'A': 's', 'B': '^',
        'C': 'v', 'D1': '<', 'D2': '>', 'E': 'p', 'F': 'h'
    }

    print("\n=== Convergence Order Summary ===")
    
    fig1, ax1 = plt.subplots(figsize=(10, 7))
    fig2, ax2 = plt.subplots(figsize=(10, 7))

    for method in all_methods:
        gap_col = f"gap_{method}"
        if gap_col not in df_res.columns:
            continue
        
        df_valid = df_res[df_res[gap_col] > 0].copy()
        
        if len(df_valid) > 1:
            log_diam = np.log(df_valid["diam_Linf"].values)
            log_gap = np.log(df_valid[gap_col].values)
            
            slope, intercept = np.polyfit(log_diam, log_gap, 1)
            fit_log_gap = intercept + slope * log_diam
            ss_res = np.sum((log_gap - fit_log_gap)**2)
            ss_tot = np.sum((log_gap - np.mean(log_gap))**2)
            r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0
            
            print(f"  {method}: p={slope:.4f}, R^2={r_squared:.4f}")
            
            ax1.loglog(df_valid["diam_Linf"], df_valid[gap_col], 
                      marker=markers.get(method, 'o'), color=colors.get(method, 'gray'),
                      linestyle='-', label=f'{method} (p={slope:.2f})', markersize=6)
        else:
            print(f"  {method}: Not enough valid points ({len(df_valid)} points with gap>0)")
        
        valid_mask = df_res[gap_col] > 0
        if valid_mask.any():
            ax2.semilogy(df_res.loc[valid_mask, "k"], df_res.loc[valid_mask, gap_col],
                        marker=markers.get(method, 'o'), color=colors.get(method, 'gray'),
                        linestyle='-', label=method, markersize=6)

    ax1.set_xlabel('Diameter (L_inf)', fontsize=12)
    ax1.set_ylabel('Gap = F_TRUE - LB', fontsize=12)
    ax1.set_title('WLSQ Convergence: Gap vs Diameter', fontsize=14)
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, which="both", ls="-", alpha=0.3)
    fig1.tight_layout()
    fig1.savefig(os.path.join(args.outdir, "gap_vs_diam.png"), dpi=150)
    plt.close(fig1)

    ax2.set_xlabel('Iteration k', fontsize=12)
    ax2.set_ylabel('Gap = F_TRUE - LB', fontsize=12)
    ax2.set_title('WLSQ Convergence: Gap vs k', fontsize=14)
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)
    fig2.tight_layout()
    fig2.savefig(os.path.join(args.outdir, "gap_vs_k.png"), dpi=150)
    plt.close(fig2)

    print(f"\nPlots saved to {args.outdir}/")

if __name__ == "__main__":
    main()
