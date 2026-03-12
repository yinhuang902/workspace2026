'''
generating pyomo model for PID example
'''
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition, SolverStatus
from pyomo.common.errors import ApplicationError
from pyomo.contrib.alternative_solutions.aos_utils import get_active_objective
import pyomo.dae as dae 
from typing import Tuple, Optional
from idaes.core.solvers import get_solver
ipopt = get_solver("ipopt")

import os
import math
import time as _time
from pathlib import Path
import numpy as np 
import matplotlib.pyplot as plt
import pandas as pd
np.random.seed(17)


import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import snoglode as sno
import snoglode.utils.MPI as MPI
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

num_scenarios = 10






# ---------------------------------------------------------
# Timeout Debug Logging - shared state
# ---------------------------------------------------------
timeout_records = []          # list of (iteration, subproblem_name, dual_lb)
current_iteration = [0]       # mutable container updated via monkey-patch
sp = 0.5
df = pd.read_csv(os.getcwd() + "/data.csv")
plot_dir =  os.getcwd() + "/plots_snoglode_parallel/"

class GurobiLBLowerBounder(sno.AbstractLowerBounder):
    def __init__(self, 
                 solver: str, 
                 time_ub: int = 600) -> None:
        super().__init__(solver = solver,
                         time_ub = time_ub)
        assert solver.name == "gurobi" or solver.name == "gurobipersistent"
        self.iter = 0
        self.current_milp_gap = solver.options["MIPGap"]
        if self.current_milp_gap == None:
            print("Gurobi's MIP gap is not specified - will revert to classical LB solves.")
        # debug logging plumbing
        self._debug_last_solver_status = None
        self._debug_last_tc = None

    def solve_a_subproblem(self, 
                           subproblem_model: pyo.ConcreteModel,
                           *args, **kwargs) -> Tuple[bool, Optional[float]]:
        
        # how to tell if we are in a new iteration? wb in parallel?
        self.iter += 1
        if self.iter % num_scenarios == 0 \
            and self.current_milp_gap != 1e-2 \
                and self.iter > num_scenarios * 18:
            self.current_milp_gap -= 0.01
            if self.current_milp_gap < 0: 
                self.current_milp_gap = 1e-2
            self.opt.options["MIPGap"] = self.current_milp_gap

        subproblem_name = kwargs.get('subproblem_name', 'Unknown')

        # warm start with the ipopt solution
        try:
            ipopt.solve(subproblem_model,
                        load_solutions = True)
        except Exception as e:
            print(f"WARNING: Ipopt crashed on subproblem '{subproblem_name}'. Proceeding to Gurobi without warm start. Error: {e}")
        
        # solve explicitly to global optimality with gurobi
        results = self.opt.solve(subproblem_model,
                                 load_solutions = False, 
                                 symbolic_solver_labels = True,
                                 tee = False)
        # debug logging plumbing — store solver results for external wrapper
        try:
            self._debug_last_solver_status = str(results.solver.status)
            self._debug_last_tc = str(results.solver.termination_condition)
        except Exception:
            self._debug_last_solver_status = "NA"
            self._debug_last_tc = "NA"
        
        # Handle maxTimeLimit for LOWER BOUNDING:
        # Use Gurobi's dual bound (lower_bound) directly - do NOT use primal/incumbent values.
        # This ensures a valid conservative LB even when Gurobi times out.
        if results.solver.termination_condition == TerminationCondition.maxTimeLimit:
            lb = getattr(results.problem, 'lower_bound', None)
            # Validate lb is a finite float (not None, not NaN, not ±inf)
            if lb is not None and math.isfinite(lb):
                print(f"INFO: Gurobi maxTimeLimit on '{subproblem_name}'. Using dual lower_bound={lb:.8g} as valid LB.")
                timeout_records.append((current_iteration[0], subproblem_name, lb))
                return True, lb
            else:
                # Return (True, -inf) as a conservative LB: this ensures the node is NOT pruned
                # incorrectly. Returning False could be misinterpreted as infeasible upstream.
                print(f"WARNING: Gurobi maxTimeLimit on '{subproblem_name}' but no valid dual lower_bound available "
                      f"(got: {lb}); returning -inf as conservative LB (node will not be pruned).")
                timeout_records.append((current_iteration[0], subproblem_name, lb))
                return True, float('-inf')

        # if the solution is optimal, return objective value
        if (results.solver.termination_condition == TerminationCondition.optimal or \
            results.solver.termination_condition == TerminationCondition.maxIterations) and \
           (results.solver.status == SolverStatus.ok or results.solver.status == SolverStatus.warning):
            
            if results.solver.termination_condition == TerminationCondition.maxIterations:
                print(f"DEBUG: 'maxIterations' reached for subproblem: {subproblem_name}")

            # load in solutions, return [feasibility = True, obj]
            subproblem_model.solutions.load_from(results)
            # gap = (results.problem.upper_bound - results.problem.lower_bound) / results.problem.upper_bound

            # if we do not have a sufficiently small gap, return LB
            if self.current_milp_gap > 0: 
                parent_obj = pyo.value(subproblem_model.successor_obj)
                return True, max(parent_obj, results.problem.lower_bound)
            
            #otw return normal objective
            else: 
                # there should only be one objective, so return that value.
                return True, pyo.value(get_active_objective(subproblem_model))

        # if the solution is not feasible, return None
        elif results.solver.termination_condition == TerminationCondition.infeasible:
            return False, None
        
        elif results.solver.termination_condition == TerminationCondition.unbounded:
            print(f"DEBUG: 'unbounded' condition for subproblem: {subproblem_name}")
            return True, float('-inf')

        else: raise RuntimeError(f"unexpected termination_condition for lower bounding problem: {results.solver.termination_condition}. Subproblem: {subproblem_name}")
    

def build_pid_model(scenario_name):
    '''
    Build instance of pyomo PID model 

    Parameters
    -----------
    setpoint_change: float
        Value for the new setpoint 
    model_uncertainty: list
        List containing the values for the model uncertainty in the form: [tau_xs, tau_us, tau_ds]
    disturbance: float
        Value for the disturbance 

    Returns
    -----------
    m: Concrete Pyomo model 
        Instance of pyomo model with uncertain parameters  
    '''
    # unpack scenario name
    _, scen_num = scenario_name.split("_")

    # retrieve random realizations
    row_data = df.iloc[int(scen_num)]
    tau_xs = float(row_data["tau_xs"])
    tau_us = float(row_data["tau_us"])
    tau_ds = float(row_data["tau_ds"])
    num_disturbances = sum(1 for header in df.columns.tolist() if "disturbance" in header)
    disturbance = [float(row_data[f"disturbance_{i}"]) for i in range(num_disturbances)]
    # setpoint_change = float(row_data["setpoint_change"])
    setpoint_change = sp

    '''''''''''''''
    # create model #
    '''''''''''''''
    m = pyo.ConcreteModel()

    '''''''''''''''
    #### Sets ####
    '''''''''''''''
    # define time set 
    T = 15
    m.time = pyo.RangeSet(0,T)
    m.t = dae.ContinuousSet(bounds=(0,T))

    '''''''''''''''
    # Parameters #
    '''''''''''''''
    # define model parameters 
    m.x_setpoint = pyo.Param(initialize=setpoint_change)        # set-point 
    m.tau_xs = pyo.Param(initialize=tau_xs)                     # model structural uncertainty 
    m.tau_us = pyo.Param(initialize=tau_us)                     # model structural uncertainty
    m.tau_ds = pyo.Param(initialize=tau_ds)                     # model structural uncertainty 
    m.d_s = pyo.Param(m.t, initialize=0, mutable=True)          # disturbances 

    '''''''''''''''
    ## Variables ##
    '''''''''''''''
    # define model variables 
    '''
    m.K_p = pyo.Var(domain=pyo.Reals, bounds=[-10, 10])         # controller gain
    m.K_i = pyo.Var(domain=pyo.Reals, bounds=[-90, -80])       # integral gain 
    m.K_d = pyo.Var(domain=pyo.Reals, bounds=[0, 10])      # dervative gain
    '''
    m.K_p = pyo.Var(domain=pyo.Reals, bounds=[-10, 10])         # controller gain
    m.K_i = pyo.Var(domain=pyo.Reals, bounds=[-100, 100])       # integral gain 
    m.K_d = pyo.Var(domain=pyo.Reals, bounds=[-100, 100])      # dervative gain
    
    m.x_s = pyo.Var(m.t, domain=pyo.Reals, bounds=[-2.5, 2.5])  # state-time trajectories 
    m.e_s = pyo.Var(m.t, domain=pyo.Reals)                      # change in x from set point 
    m.u_s = pyo.Var(m.t, domain=pyo.Reals, bounds=[-5.0, 5.0])  

    # define dervative variable for x_s
    m.dxdt = dae.DerivativeVar(m.x_s, wrt=m.t)      # derivative of state-time trajectory variable
    m.dedt = dae.DerivativeVar(m.e_s, wrt=m.t)      # derivative of 

    '''''''''''''''
    # Constraints #
    '''''''''''''''

    # constraint 1 
    @m.Constraint(m.t)
    def dxdt_con(m, t):
        if t == m.t.first(): return pyo.Constraint.Skip
        else: return m.dxdt[t] == -m.tau_xs*m.x_s[t] + m.tau_us*m.u_s[t] + m.tau_ds*m.d_s[t]
        
    m.x_init_cond = pyo.Constraint(expr=m.x_s[m.t.first()] == 0)

    # constraint 2 
    @m.Constraint(m.t)
    def e_con(m, t):
        return m.e_s[t] == m.x_s[t] - m.x_setpoint 
    
    # constraint 3 
    m.I = pyo.Var(m.t)
    @m.Constraint(m.t)
    def	integral(m,t):
        # at the first time point, we will not have any volume under the curve
        if t ==	m.t.first(): return m.I[t] == 0
        # otherwise, compute the approximation of the integral
        else: return m.I[t] ==  m.I[m.t.prev(t)] + (t-m.t.prev(t))*m.e_s[t]

    @m.Constraint(m.t)
    def u_con(m, t):
        return m.u_s[t] == m.K_p*m.e_s[t] + m.K_i * m.I[t] + m.K_d * m.dedt[t]

        
    '''''''''''''''
    ## Objective ##
    '''''''''''''''
    def e_sq_integral_rule(m, t):
        return 10*m.e_s[t]**2 + 0.01*m.u_s[t]**2
    m.e_sq_integral = dae.Integral(m.t, wrt=m.t, rule=e_sq_integral_rule)
    m.obj = pyo.Objective(sense=pyo.minimize, expr=m.e_sq_integral)

    '''''''''''''''
    # Discretize #
    '''''''''''''''
    discretizer = pyo.TransformationFactory('dae.finite_difference')
    discretizer.apply_to(m, nfe=20, wrt=m.t, scheme='BACKWARD')
 
    # setting disturbance parameters over discretized time
    index = 0
    for t in m.t:  
        m.d_s[t] = disturbance[index]
        index += 1

    first_stage = {
        "K_p": m.K_p,
        "K_i": m.K_i,
        "K_d": m.K_d
    }
    probability = 1/num_scenarios

    return [m,
            first_stage,
            probability]


if __name__ == '__main__':
    nonconvex_gurobi = pyo.SolverFactory("gurobi")
    nonconvex_gurobi.options["NonConvex"] = 2
    
    nonconvex_gurobi_lb = pyo.SolverFactory("gurobi")
    nonconvex_gurobi_lb.options["NonConvex"] = 2
    nonconvex_gurobi_lb.options["MIPGap"] = 1e-3
    nonconvex_gurobi_lb.options["TimeLimit"] = 60
    scenarios = [f"scen_{i}" for i in range(0,num_scenarios)]

    obbt_solver_opts = {
        "NonConvex": 2,
        "MIPGap": 1,
        "TimeLimit": 5
    }

    params = sno.SolverParameters(subproblem_names = scenarios,
                                  subproblem_creator = build_pid_model,
                                  lb_solver = nonconvex_gurobi_lb,
                                  cg_solver = ipopt,
                                  ub_solver = nonconvex_gurobi)
    params.set_bounders(candidate_solution_finder = sno.SolveExtensiveForm,
                        lower_bounder = GurobiLBLowerBounder)
    params.set_bounds_tightening(fbbt=True, 
                                 obbt=True,
                                 obbt_solver_opt=obbt_solver_opts)
    # params.set_branching(selection_strategy = sno.MaximumDisagreement)
    params.set_branching(selection_strategy = sno.HybridBranching,
                         partition_strategy = sno.ExpectedValue)
    
    params.activate_verbose()
    # if (size==1): params.set_logging(fname = os.getcwd() + "/logs/stochastic_pid_log")
    # else: params.set_logging(fname = os.getcwd() + "/logs/stochastic_pid_log_parallel")
    if (rank==0): params.display()

    solver = sno.Solver(params)
    
    # ---------------------------------------------------------
    # CSV Logging Implementation (Monkey Patch)
    # ---------------------------------------------------------
    import csv

    # CSV Logging Setup
    csv_filename = os.getcwd() + "/sp_snog_result.csv"
    csv_header = ["Time (s)", "Nodes Explored", "Pruned by", "Bound Update", "LB", "UB", "Rel. Gap", "Abs. Gap", "# Nodes"]

    # Initialize CSV with header (only on rank 0)
    if rank == 0:
        with open(csv_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)

    # Monkey patch display_status to log to CSV
    original_display_status = solver.display_status

    def csv_logging_display_status(bnb_result):
        # Call original method first to maintain console output
        original_display_status(bnb_result)
        
        # Only log on rank 0
        if rank == 0:
            # Replicate logic to extract formatted data
            pruned = " "
            if "pruned by bound" in bnb_result:
                pruned = "Bound"
            elif "pruned by infeasibility" in bnb_result:
                pruned = "Infeas."
            
            bound_update = " "
            if "ublb" in bnb_result:
                bound_update = "* L U"
            elif "ub" in bnb_result:
                bound_update = "* U  "
            elif "lb" in bnb_result:
                bound_update = "* L  "

            # Extract metrics
            row = [
                round(solver.runtime, 3),
                solver.tree.metrics.nodes.explored,
                pruned,
                bound_update,
                f"{solver.tree.metrics.lb:.8}",
                f"{solver.tree.metrics.ub:.8}",
                f"{round(solver.tree.metrics.relative_gap*100, 4)}%",
                round(solver.tree.metrics.absolute_gap, 6),
                solver.tree.n_nodes()
            ]
            
            with open(csv_filename, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)

    solver.display_status = csv_logging_display_status
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # Timeout Debug Logging (Monkey Patch)
    #   - Writes debug_reachtimelimit.txt INCREMENTALLY so that
    #     records are saved even if the job is killed mid-run.
    # ---------------------------------------------------------
    debug_filename = os.getcwd() + "/debug_reachtimelimit.txt"
    _written_count = [0]  # how many records have already been flushed

    # Initialize the file with a header (rank 0 only)
    if rank == 0:
        with open(debug_filename, mode='w') as f:
            f.write(f"{'Iteration':<12}{'Subproblem':<20}{'DualLB'}\n")
            f.write("-" * 44 + "\n")

    original_dispatch_updates = solver.dispatch_updates

    def timeout_tracking_dispatch_updates(bnb_result):
        # sync the shared iteration counter BEFORE the original call
        current_iteration[0] = solver.iteration
        original_dispatch_updates(bnb_result)

        # flush any NEW timeout records that were appended during this iteration
        if rank == 0 and len(timeout_records) > _written_count[0]:
            with open(debug_filename, mode='a') as f:
                for i in range(_written_count[0], len(timeout_records)):
                    iteration, subproblem_name, dual_lb = timeout_records[i]
                    lb_str = f"{dual_lb:.8g}" if dual_lb is not None and isinstance(dual_lb, (int, float)) else str(dual_lb)
                    f.write(f"{iteration:<12}{subproblem_name:<20}{lb_str}\n")
            _written_count[0] = len(timeout_records)

    solver.dispatch_updates = timeout_tracking_dispatch_updates
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # Debug Logging Setup (3 files in debug_log/)
    # ---------------------------------------------------------
    _base_dir = Path(__file__).resolve().parent
    _log_dir  = _base_dir / "debug_log"
    _log_dir.mkdir(parents=True, exist_ok=True)
    _dbg_paths = {
        "ef":   str(_log_dir / "debug_ef_solver.txt"),
        "dual": str(_log_dir / "debug_dual.txt"),
        "iter": str(_log_dir / "debug_iter_info.txt"),
    }
    # Truncate all three files at start of run
    if rank == 0:
        for _p in _dbg_paths.values():
            open(_p, "w").close()

    # Shared debug context
    class _DebugCtx:
        def __init__(self):
            self.iter = 0
            self.node_id = "NA"
            self.parent_id = "NA"
            self.depth = "NA"
            self.node_state = None
            self.lb_node_obj = "NA"
            self.pruned = 0
            self.prune_reason = "none"
            self.ef_pending = None  # buffered EF log entry
    _dctx = _DebugCtx()

    # --- (1) Wrap dispatch_node_selection: capture iter & node info ---
    _orig_dns = solver.dispatch_node_selection
    def _dbg_dispatch_node_selection():
        _dctx.iter = solver.iteration
        _dctx.ef_pending = None
        node, feasible = _orig_dns()
        try:
            _dctx.node_id = node.id
        except Exception:
            _dctx.node_id = "NA"
        _dctx.parent_id = getattr(node, 'parent_id', 'NA')
        _dctx.depth = getattr(node, 'depth', 'NA')
        _dctx.node_state = getattr(node, 'state', None)
        return node, feasible
    solver.dispatch_node_selection = _dbg_dispatch_node_selection

    # --- (2) Wrap solve_a_subproblem: debug_dual.txt ---
    _orig_solve_sub = solver.lower_bounder.solve_a_subproblem
    def _dbg_solve_a_subproblem(*args, **kwargs):
        subproblem_name = kwargs.get('subproblem_name', 'Unknown')
        t0 = _time.perf_counter()
        result = _orig_solve_sub(*args, **kwargs)
        dt = _time.perf_counter() - t0
        if rank == 0:
            try:
                feasible, obj = result
                s_status = getattr(solver.lower_bounder, '_debug_last_solver_status', 'NA') or 'NA'
                s_tc = getattr(solver.lower_bounder, '_debug_last_tc', 'NA') or 'NA'
                dual_val = obj if obj is not None else "NA"
                obj_val  = obj if obj is not None else "NA"
                line = (
                    f"iter={_dctx.iter} "
                    f"node_id={_dctx.node_id} "
                    f"scenario={subproblem_name} "
                    f"dual_bound={dual_val} "
                    f"obj_value={obj_val} "
                    f"solver_status={s_status} "
                    f"termination_condition={s_tc} "
                    f"solve_time_sec={dt:.6f}\n"
                )
                with open(_dbg_paths["dual"], "a") as f:
                    f.write(line)
            except Exception:
                try:
                    with open(_dbg_paths["dual"], "a") as f:
                        f.write(f"iter={_dctx.iter} scenario={subproblem_name} ERROR\n")
                except Exception:
                    pass
        return result
    solver.lower_bounder.solve_a_subproblem = _dbg_solve_a_subproblem

    # --- (3) Wrap generate: debug_ef_solver.txt (buffered) ---
    _finder = solver.upper_bounder.candidate_solution_finder
    _orig_generate = _finder.generate
    def _dbg_generate(node, subproblems):
        t0 = _time.perf_counter()
        result = _orig_generate(node, subproblems)
        dt = _time.perf_counter() - t0
        try:
            cand_found, _, cand_obj = result
            ef_obj = cand_obj if cand_found else "NA"
        except Exception:
            ef_obj = "NA"
        # Read solver results stored by generate_candidate plumbing
        _ef_ss = getattr(_finder, '_debug_last_solver_status', 'NA') or 'NA'
        _ef_tc = getattr(_finder, '_debug_last_tc', 'NA') or 'NA'
        # Buffer — ub_updated is determined after dispatch_bnb
        _dctx.ef_pending = {
            "iter": _dctx.iter,
            "node_id": _dctx.node_id,
            "solver_name": "ipopt",
            "solver_status": _ef_ss,
            "termination_condition": _ef_tc,
            "ef_time_sec": f"{dt:.6f}",
            "ef_obj": ef_obj,
            "true_obj": "NA",
        }
        return result
    _finder.generate = _dbg_generate

    # --- (4) Wrap dispatch_bnb: capture prune + flush EF log ---
    _orig_dbnb = solver.dispatch_bnb
    def _dbg_dispatch_bnb(current_node):
        try:
            _dctx.lb_node_obj = current_node.lb_problem.objective
        except Exception:
            _dctx.lb_node_obj = "NA"
        ub_before = solver.tree.metrics.ub
        result = _orig_dbnb(current_node)
        ub_after = solver.tree.metrics.ub
        # Prune info
        if "pruned by bound" in result:
            _dctx.pruned = 1; _dctx.prune_reason = "bound"
        elif "pruned by infeasibility" in result:
            _dctx.pruned = 1; _dctx.prune_reason = "infeasible"
        else:
            _dctx.pruned = 0; _dctx.prune_reason = "none"
        # Flush buffered EF log
        if rank == 0 and _dctx.ef_pending is not None:
            try:
                ub_upd = 1 if (ub_after < ub_before) else 0
                d = _dctx.ef_pending
                with open(_dbg_paths["ef"], "a") as f:
                    f.write(f"iter={d['iter']}\n")
                    f.write(f"node_id={d['node_id']}\n")
                    f.write(f"solver_name={d['solver_name']}\n")
                    f.write(f"solver_status={d['solver_status']}\n")
                    f.write(f"termination_condition={d['termination_condition']}\n")
                    f.write(f"ef_time_sec={d['ef_time_sec']}\n")
                    f.write(f"ef_obj={d['ef_obj']}\n")
                    f.write(f"true_obj={d['true_obj']}\n")
                    f.write(f"ub_updated={ub_upd}\n")
                    f.write("---\n")
            except Exception:
                pass
            _dctx.ef_pending = None
        return result
    solver.dispatch_bnb = _dbg_dispatch_bnb

    # --- (5) Wrap dispatch_updates: debug_iter_info.txt ---
    _orig_du = solver.dispatch_updates  # already wrapped for CSV + timeout
    def _dbg_dispatch_updates(bnb_result):
        iter_k = _dctx.iter  # capture before increment
        _orig_du(bnb_result)
        if rank == 0:
            try:
                from snoglode.utils.supported import SupportedVars
                # Box bounds
                box_parts = []
                volume = 1.0
                vol_ok = True
                if _dctx.node_state is not None:
                    for vt in [SupportedVars.reals]:
                        if vt in _dctx.node_state:
                            for vid, vi in _dctx.node_state[vt].items():
                                lb_v = getattr(vi, 'lb', 'NA')
                                ub_v = getattr(vi, 'ub', 'NA')
                                box_parts.append(f"{vid}=({lb_v},{ub_v})")
                                try:
                                    volume *= (ub_v - lb_v)
                                except Exception:
                                    vol_ok = False
                box_str = "; ".join(box_parts) if box_parts else "NA"
                vol_str = f"{volume:.10g}" if vol_ok and box_parts else "NA"
                # UB incumbent point in box
                ub_in_box = "NA"
                try:
                    sol = solver.solution.subproblem_solutions
                    if (sol is not None and _dctx.node_state is not None
                            and solver.tree.metrics.ub < float('inf')):
                        first_sc = list(sol.keys())[0]
                        in_box = True
                        for vt in [SupportedVars.reals]:
                            if vt in _dctx.node_state:
                                for vid, vi in _dctx.node_state[vt].items():
                                    if vid in sol[first_sc]:
                                        val = sol[first_sc][vid]
                                        if val < vi.lb or val > vi.ub:
                                            in_box = False
                        ub_in_box = 1 if in_box else 0
                except Exception:
                    ub_in_box = "NA"
                with open(_dbg_paths["iter"], "a") as f:
                    f.write(f"iter={iter_k}\n")
                    f.write(f"node_id={_dctx.node_id}\n")
                    f.write(f"parent_id={_dctx.parent_id}\n")
                    f.write(f"depth={_dctx.depth}\n")
                    f.write(f"selected_box_bounds={box_str}\n")
                    f.write(f"box_volume={vol_str}\n")
                    f.write(f"vertices=NA\n")
                    f.write(f"current_ub_point_in_box={ub_in_box}\n")
                    f.write(f"LB_value_this_iter={_dctx.lb_node_obj}\n")
                    f.write(f"UB_value_current={solver.tree.metrics.ub}\n")
                    f.write(f"pruned={_dctx.pruned}\n")
                    f.write(f"prune_reason={_dctx.prune_reason}\n")
                    f.write("---\n")
            except Exception:
                try:
                    with open(_dbg_paths["iter"], "a") as f:
                        f.write(f"iter={iter_k} ERROR_LOGGING\n---\n")
                except Exception:
                    pass
    solver.dispatch_updates = _dbg_dispatch_updates
    # ---------------------------------------------------------

    # ef = solver.get_ef()
    # nonconvex_gurobi.solve(ef,
    #                        tee = True)
    # quit()
    timeout_records.clear()  # reset for this run
    solver.solve(max_iter=9000,
                 rel_tolerance = 1e-4,
                 abs_tolerance = 1e-8,
                 time_limit = 60*60*48)

    if rank == 0:
        print(f"\nINFO: Wrote {len(timeout_records)} timeout record(s) to {debug_filename}")

    if (rank==0):
        print("\n====================================================================")
        print("SOLUTION")
        for n in solver.subproblems.names:
            print(f"subproblem = {n}")
            x, u = {}, {}
            for vn in solver.solution.subproblem_solutions[n]:

                # display first stage only (for sanity check)
                if vn=="K_p" or vn=="K_i" or vn=="K_d":
                    var_val = solver.solution.subproblem_solutions[n][vn]
                    print(f"  var name = {vn}, value = {var_val}")

                # collect plot data on x_s, u_s
                if "x_s" in vn:
                    _, half_stripped_time = vn.split("[")
                    stripped_time = half_stripped_time.split("]")[0]
                    time = float(stripped_time)
                    var_val = solver.solution.subproblem_solutions[n][vn]
                    x[time] = var_val
                
                if "u_s" in vn:
                    _, half_stripped_time = vn.split("[")
                    stripped_time = half_stripped_time.split("]")[0]
                    time = float(stripped_time)
                    var_val = solver.solution.subproblem_solutions[n][vn]
                    u[time] = var_val

            # plot
            scen_num = n.split("_")[1]
            plt.suptitle(f"Scenario {scen_num}")
            plt.subplot(1, 2, 1)
            plt.plot(x.keys(), x.values())
            row_data = df.iloc[int(scen_num)]
            # setpoint_change = row_data["setpoint_change"]
            setpoint_change = sp
            plt.axhline(y = setpoint_change, 
                        color='r', 
                        linestyle='dotted', 
                        linewidth=2, 
                        label="set point")
            plt.xlabel('Time')
            plt.ylabel('x')
            plt.legend()

            plt.subplot(1, 2, 2)
            plt.plot(u.keys(), u.values())
            plt.xlabel('Time')
            plt.ylabel('u')

            plt.tight_layout()
            plt.savefig(plot_dir + f'scen_{scen_num}.png',
                        dpi=300)
            plt.clf()

            print()
        print("====================================================================")