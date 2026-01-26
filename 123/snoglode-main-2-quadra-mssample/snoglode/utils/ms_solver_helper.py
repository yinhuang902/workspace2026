
import pyomo.environ as pyo
from pyomo.opt import SolverFactory
from typing import Tuple, Optional

def _solve_ms_dual_bound_gurobi(model: pyo.ConcreteModel, temp_obj_name: str, load_solutions: bool = False) -> float:
    """
    Solves the ms subproblem (min F - Q) using Gurobi with strict settings.
    Returns the dual lower bound (best bound) if optimal, else -inf.
    
    Settings:
    - MIPGap = 1e-1
    - NonConvex = 2
    - TimeLimit = 15
    - Only accepts TerminationCondition.optimal
    """
    
    # Create Gurobi solver instance
    ms_solver = SolverFactory("gurobi")
    
    # Set strict options
    ms_solver.options["MIPGap"] = 1e-1
    ms_solver.options["NonConvex"] = 2
    ms_solver.options["TimeLimit"] = 15
    
    try:
        # Solve
        results = ms_solver.solve(model, load_solutions=load_solutions)
        
        # Check strict termination condition
        if results.solver.termination_condition == pyo.TerminationCondition.optimal:
            # Try to get the dual lower bound
            try:
                # Access the lower bound from the first problem solution
                ms_lb = results.problem[0].lower_bound
                
                # Validate the bound is a finite float
                if ms_lb is None or ms_lb == float('inf') or ms_lb == float('-inf'):
                    return float('-inf')
                
                return float(ms_lb)
            except (AttributeError, IndexError, ValueError):
                # Fallback: if bound access fails, return -inf
                return float('-inf')
        else:
            # Not optimal (including locallyOptimal, maxTimeLimit, etc.)
            return float('-inf')
            
    except Exception:
        # Any solver error -> failure
        return float('-inf')

def _solve_true_recourse_primal_gurobi(model: pyo.ConcreteModel) -> Tuple[bool, float]:
    """
    Solves the true recourse scenario subproblem using Gurobi.
    Returns (success, obj_val).
    
    Settings:
    - NonConvex = 2
    - MIPGap = 1e-1
    - TimeLimit = 15
    - Only accepts TerminationCondition.optimal
    
    Returns:
    - success: True if optimal
    - obj_val: Primal objective value (pyo.value(model.obj)) if success, else None
    """
    solver = SolverFactory("gurobi")
    solver.options["NonConvex"] = 2
    solver.options["MIPGap"] = 1e-1
    solver.options["TimeLimit"] = 15
    
    try:
        # Load solutions to get primal values
        results = solver.solve(model, load_solutions=True)
        
        if results.solver.termination_condition == pyo.TerminationCondition.optimal:
            # Get primal objective value
            obj_comp = model.component_data_objects(pyo.Objective, active=True).__next__()
            return True, float(pyo.value(obj_comp.expr))
        else:
            return False, 0.0
    except Exception:
        return False, 0.0
