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
# MIPGap switching — shared outer-iteration counter
# Updated once per outer iteration via a minimal dispatch_updates wrapper.
# ---------------------------------------------------------
current_iteration = [0]

sp = 0.5
df = pd.read_csv(os.getcwd() + "/data.csv")
plot_dir =  os.getcwd() + "/plots_snoglode_parallel/"

# ---------------------------------------------------------
# MIPGap switching parameter
#   For the first `small_gap_iter_num` outer iterations the Gurobi
#   LB subproblem solver uses a loose MIPGap (1e-1) to speed up
#   early exploration.  After that the gap reverts to the value
#   configured on the solver (e.g. 1e-3).
#   Set to 0 to disable (solver always uses the configured gap).
# ---------------------------------------------------------
small_gap_iter_num = 100

class GurobiLBLowerBounder(sno.AbstractLowerBounder):
    def __init__(self, 
                 solver: str, 
                 time_ub: int = 600) -> None:
        super().__init__(solver = solver,
                         time_ub = time_ub)
        assert solver.name == "gurobi" or solver.name == "gurobipersistent"
        self.iter = 0
        self._mipgap_restored = False  # one-shot flag for the restore print
        # Store the original configured MIPGap for later restoration (may be None)
        self.original_milp_gap = solver.options.get("MIPGap", None)
        # If small_gap_iter_num > 0, start with loose MIPGap; otherwise keep original
        if small_gap_iter_num > 0:
            self.current_milp_gap = 1e-1
            solver.options["MIPGap"] = self.current_milp_gap
        else:
            self.current_milp_gap = self.original_milp_gap

    def solve_a_subproblem(self, 
                           subproblem_model: pyo.ConcreteModel,
                           *args, **kwargs) -> Tuple[bool, Optional[float]]:
        
        self.iter += 1

        # ----- MIPGap two-phase switching (uses outer iteration counter) -----
        # Phase 1 (iter < small_gap_iter_num): loose gap (1e-1) for faster solves
        # Phase 2 (iter >= small_gap_iter_num): revert to configured gap (e.g. 1e-3)
        outer_iter = current_iteration[0]
        if small_gap_iter_num > 0:
            if outer_iter < small_gap_iter_num:
                if self.current_milp_gap != 1e-1:
                    self.current_milp_gap = 1e-1
                    self.opt.options["MIPGap"] = self.current_milp_gap
            else:
                if self.current_milp_gap != self.original_milp_gap:
                    if not self._mipgap_restored:
                        print(f"[Iter {outer_iter}] MIPGap restored from {self.current_milp_gap} to {self.original_milp_gap}")
                        self._mipgap_restored = True
                    self.current_milp_gap = self.original_milp_gap
                    if self.original_milp_gap is not None:
                        self.opt.options["MIPGap"] = self.current_milp_gap
                    else:
                        self.opt.options.pop("MIPGap", None)

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
        
        # if the solution is optimal, return objective value
        if (results.solver.termination_condition == TerminationCondition.optimal or \
            results.solver.termination_condition == TerminationCondition.maxIterations or \
            results.solver.termination_condition == TerminationCondition.maxTimeLimit) and \
           (results.solver.status == SolverStatus.ok or results.solver.status == SolverStatus.warning or results.solver.status == SolverStatus.aborted):
            
            if results.solver.termination_condition == TerminationCondition.maxIterations:
                print(f"DEBUG: 'maxIterations' reached for subproblem: {subproblem_name}")
            if results.solver.termination_condition == TerminationCondition.maxTimeLimit:
                print(f"DEBUG: 'maxTimeLimit' reached for subproblem: {subproblem_name}")

            # load in solutions, return [feasibility = True, obj]
            subproblem_model.solutions.load_from(results)

            # if we do not have a sufficiently small gap, return LB
            if self.current_milp_gap is not None and self.current_milp_gap > 0: 
                parent_obj = pyo.value(subproblem_model.successor_obj)
                return True, max(parent_obj, results.problem.lower_bound)
            
            #otw return normal objective
            else: 
                # there should only be one objective, so return that value.
                return True, pyo.value(get_active_objective(subproblem_model))

        # if the solution is not feasible, return None
        elif results.solver.termination_condition == TerminationCondition.infeasible:
            return False, None

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
    # Minimal monkey-patch: track the outer iteration number
    # so that GurobiLBLowerBounder can switch MIPGap at the
    # right iteration.  This does NOT alter any solver logic.
    # ---------------------------------------------------------
    _orig_dispatch_updates = solver.dispatch_updates
    def _iter_tracking_dispatch_updates(bnb_result):
        current_iteration[0] = solver.iteration
        _orig_dispatch_updates(bnb_result)
    solver.dispatch_updates = _iter_tracking_dispatch_updates

    # ef = solver.get_ef()
    # nonconvex_gurobi.solve(ef,
    #                        tee = True)
    # quit()
    solver.solve(max_iter=9000,
                 rel_tolerance = 1e-4,
                 abs_tolerance = 1e-8,
                 time_limit = 60*60*48)

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