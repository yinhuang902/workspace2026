import os
import sys
import json
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition
from idaes.core.solvers import get_solver

# 1. Set working directory to script's directory (Robustness requirement)
# This ensures stochastic_pid.py finds data.csv and creates plot paths correctly.
script_dir = os.path.dirname(os.path.abspath(__file__))
os.chdir(script_dir)

# 2. Setup paths for imports
# Add script dir to sys.path to allow importing stochastic_pid
if script_dir not in sys.path:
    sys.path.append(script_dir)

# Add repository root to sys.path to allow importing snoglode
# Assumes repo structure: repo_root/examples/stochastic_pid/solve_pid_ef.py
repo_root = os.path.abspath(os.path.join(script_dir, '..', '..'))
if repo_root not in sys.path:
    sys.path.append(repo_root)

import snoglode as sno
import stochastic_pid

def main():
    # 3. Setup SolverParameters
    # Reuse model construction from stochastic_pid
    num_scenarios = 10
    # CRITICAL: Update the module's global variable so that build_pid_model calculates 
    # the correct probability (1/num_scenarios). Otherwise it uses the value in the file.
    stochastic_pid.num_scenarios = num_scenarios
    scenarios = [f"scen_{i}" for i in range(1, num_scenarios + 1)]
    
    # Define solvers for initialization (required by SolverParameters)
    # We use Gurobi for LB/UB and IPOPT for CG (consistent with stochastic_pid)
    # Note: These are just for initialization; the EF solve uses its own Gurobi instance.
    lb_solver = pyo.SolverFactory("gurobi")
    ub_solver = pyo.SolverFactory("gurobi")
    cg_solver = get_solver("ipopt") # Use IDAES get_solver for robustness on Windows
    
    params = sno.SolverParameters(
        subproblem_names=scenarios,
        subproblem_creator=stochastic_pid.build_pid_model,
        lb_solver=lb_solver,
        cg_solver=cg_solver,
        ub_solver=ub_solver
    )
    
    # 4. Instantiate Solver and Build EF
    print("Initializing SNoGloDe Solver...")
    solver = sno.Solver(params)
    
    print("Building Extensive Form (EF) model...")
    ef = solver.get_ef()
    
    # 5. Solve EF directly with Gurobi
    print("Solving EF with Gurobi...")
    opt = pyo.SolverFactory("gurobi")
    opt.options["NonConvex"] = 2
    opt.options["MIPGap"] = 1e-8
    opt.options["TimeLimit"] = 60*5
    
    results = opt.solve(ef, tee=True, load_solutions=False)
    if results.solver.termination_condition == TerminationCondition.optimal:
        ef.solutions.load_from(results)
    else:
        print(f"Solver termination condition: {results.solver.termination_condition}")
        ef.solutions.load_from(results)
    
    # 6. Output Results
    print("\n" + "="*50)
    print("SOLVE COMPLETE")
    print("="*50)
    
    # Print Objective
    ef_obj_val = pyo.value(ef.obj)
    print(f"EF Objective Value: {ef_obj_val}")
    
    # Print and Collect First-Stage Solution
    first_stage_solution = {}
    print("\nFirst-Stage Solution (Lifted Variables):")
    
    # ef.lifted_vars is a Pyomo Var component indexed by the names of the first-stage variables
    # Iterate over its indices to get the values
    for var_name in ef.lifted_vars:
        val = pyo.value(ef.lifted_vars[var_name])
        print(f"  {var_name}: {val}")
        first_stage_solution[var_name] = val
        
    # 7. Save Solution to JSON
    output_file = "ef_solution.json"
    with open(output_file, "w") as f:
        json.dump(first_stage_solution, f, indent=4)
    print(f"\nSolution saved to {output_file}")

if __name__ == "__main__":
    main()
