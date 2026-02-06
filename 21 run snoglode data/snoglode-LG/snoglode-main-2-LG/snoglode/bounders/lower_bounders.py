"""
Different methods for generating the solutions to a lower bounding problem.

There are many different ways a LB can be generated
Options here could be to simply solve as is, perform OBBT / FBBT, 
generate a convex relaxation & solve, etc.
"""
from typing import Tuple, Optional
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition, SolverStatus
from pyomo.contrib.alternative_solutions.aos_utils import get_active_objective

# suppress warnings when loading infeasible models
import logging
logging.getLogger('pyomo.core').setLevel(logging.ERROR)

from snoglode.bounders.base import BoundingProblemBase
from snoglode.utils.solve_stats import OneLowerBoundSolve
from snoglode.components.subproblems import Subproblems
from snoglode.components.node import Node
import numpy as np
import math
import copy as cp
import snoglode.utils.MPI as MPI
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

# ================================================================================================ #

class AbstractLowerBounder(BoundingProblemBase):
    """
    Abstract base class for the lower bounding problem.

    This is not intended to be used directly -> a child class must be
    defined from this abstract parent to be used to solve for a lwoer bound
    within the broaded solver.
    """
    perform_fbbt = True
    
    def __init__(self, 
                 solver, 
                 inherit_solutions: bool = True) -> None:
        """
        Initializes the solver information.

        Parameters
        -----------
        solver : pyo.SolverFactory
            initialized Pyomo solver factory object 
            to be used for the LOWER bounding problem solves.
        inherit_solutions : bool
            if we want to check for inheritable solutions from parent nodes.
        """

        # see snoglode.bounders.base.BoundingProblemBase
        super().__init__(solver = solver)
        
        # indicate if we want to check for inheritable solutions from parent nodes
        self.inhert_solutions = inherit_solutions


    def solve(self, 
              node: Node, 
              subproblems: Subproblems) -> None:
        """
        This solves each of the subproblems for the overall lower bound.

        The Base class does not define the solve_a_subproblem.
        Must define within the child, or an error is raised.

        Parameters
        -----------
        node : Node
            Current node in the spatial BnB tree.
        subproblems : Subproblems
            Subproblems objective containing this ranks subproblems.
        """
        assert type(node)==Node
        assert type(subproblems)==Subproblems

        statistics = OneLowerBoundSolve(subproblems.names)
        
        # for each subproblems's model
        for subproblem_name in subproblems.names:

            # cannot inherit the solution if we are at the root node / do not want to inherit
            if (node.id > 0 and self.inhert_solutions):
                inheritable_solution = self.inherit_parent_solution(node = node,
                                                                    subproblems = subproblems,
                                                                    subproblem_name = subproblem_name)
            else: inheritable_solution = False

            # if we can validly inherit the solution, update statistics and move on
            if inheritable_solution:

                # update statistics and move on
                statistics.update_to_parent(subproblem_name = subproblem_name,
                                            subproblems = subproblems,
                                            subproblem_objective = node.lb_problem.subproblem_solutions[subproblem_name].objective,
                                            parent_complicating_var_solution = node.lb_problem.subproblem_solutions[subproblem_name].complicating_var_solution)
            
            # if we cannot inhert the solution, then solve
            if not inheritable_solution:

                # relax the binaries, if there are any
                if subproblems.relax_binaries: subproblems.relax_all_binaries()
                if subproblems.relax_integers: subproblems.relax_all_integers() 

                # update & activate objective feasibility cuts (if not at root)
                if node.id > 0:
                    self.activate_bound_cuts(node = node,
                                             subproblem_model = subproblems.model[subproblem_name])

                # solve the current model representing this scenario - returns bool (feasible) & obj value (float)
                subproblem_is_feasible, subproblem_objective = \
                    self.solve_a_subproblem(subproblem_name = subproblem_name,
                                            subproblem_model = subproblems.model[subproblem_name],
                                            subproblem_complicating_vars = subproblems.subproblem_complicating_vars[subproblem_name])
                
                # deactivate bound cuts
                self.deactivate_bound_cuts(subproblems.model[subproblem_name])

                # if we have one infeasible scenario, the entire node is infeasible
                if not subproblem_is_feasible:
                    
                    # if we are infeasible, both UB/LB are infeasible -> add appropriate stats
                    node.lb_problem.is_infeasible()
                    node.ub_problem.is_infeasible()
                    
                    return False
            
                # if we are feasible, add statistics
                statistics.update(subproblem_name = subproblem_name,
                                  subproblem_objective = subproblem_objective,
                                  subproblems = subproblems)
            
        # if we were successful, add statistics to node
        node.lb_problem.is_feasible(statistics)

        return True


    def activate_bound_cuts(self,
                            node: Node,
                            subproblem_model: pyo.ConcreteModel) -> None:
        """
        Based on the current LB_parent value,
        this subproblem must be able to yield a larger
        LB than the successors.

        Parameters
        ----------
        node : Node
            current node in the spatial branch and bound tree
        subproblem_model : pyo.ConcreteModel
            pyomo model corresponding to this subproblem
        """
        # the subproblem obj is bounded by below from successor LB 
        # if we do not have the solution we want - set to -inf
        try:
            subproblem_model.successor_obj = \
                node.lb_problem.subproblem_solutions[subproblem_model.name].objective
        except:
            subproblem_model.successor_obj = float("-inf")

        # add constraint to the model
        subproblem_model.successor_lb_cut.activate()


    def inherit_parent_solution(self,
                                node: Node,
                                subproblems: Subproblems,
                                subproblem_name: str) -> bool:
        """
        Because the LB problems are solved to global optimality,
        we should only have to solve a node if the bounds of the 
        new node overlap that of the original solution.
        
        Parameters
        ----------
        node : Node
            current node in the spatial branch and bound tree
        subproblems : Subproblems
            Subproblems object for this rank.
        subproblem_name : str
            which subproblem we are trying to see if we can inherit solutions for.

        Returns
        ----------
        bool
            if we can inherit the parent solution for this subproblem.
        """
        # do all of the complicating variables for the parent solution 
        # fall within the bounds of the current node?
        for complicating_var in subproblems.subproblem_complicating_vars[subproblem_name]:

            # determine the domain 
            var_type, complicating_var_id, _ = subproblems.var_to_data[complicating_var]
            subproblem_model = subproblems.model[subproblem_name]

            # current bounds
            node_lb = node.state[var_type][complicating_var_id].lb
            node_ub = node.state[var_type][complicating_var_id].ub

            # solution
            parent_solution = \
                node.lb_problem.subproblem_solutions[subproblem_model.name].complicating_var_solution[var_type][complicating_var_id]

            # if any solution falls outside the bounds or there is no solution, return false
            if (parent_solution == None) or (parent_solution < node_lb) or (parent_solution > node_ub):
                return False
        
        # all solutions fall within the bounds
        return True
    

    def deactivate_bound_cuts(self,
                              subproblem_model: pyo.ConcreteModel) -> None:
        """
        Deactivates the successor_lb_cut.
        (do not want this active for the UB problem)

        Parameters
        ----------
        subproblem_model : pyo.ConcreteModel
            pyomo model corresponding to this subproblem
        """
        subproblem_model.successor_lb_cut.deactivate()

    
    def evaluate_termination(self,
                             results, 
                             subproblem_model: pyo.ConcreteModel) -> None:
        """
        After a model is solved in solve_a_subproblem(), need to evaluate what
        the termination status means.

        Parameters
        ----------
        results: pyo.SolverResults
            solver results object
        subproblem_model : pyo.ConcreteModel
            pyomo model corresponding to this subproblem
        root_node: bool
            indicates if we are at the root node or not.
        """
        
        # check again if locally optimal, raise error
        if results.solver.termination_condition==TerminationCondition.locallyOptimal:
            raise RuntimeError(f"While solving a subproblem at the lower bound, found a locally optimal solution. We *must* have global solutions to subproblems at the lower bound.")
            
        # if the solution is OPTIMAL: load solutions & retrieve objective value
        if results.solver.termination_condition in [TerminationCondition.optimal, 
                                                    TerminationCondition.globallyOptimal] \
                and results.solver.status==SolverStatus.ok:

            # load in solutions, return [feasibility = True, obj, results]
            subproblem_model.solutions.load_from(results)

            # return the value of the singular active objective.
            # PATCH 1.3: Store reason in side channel for LG, return 2-tuple for compatibility
            self._last_term_reason = "optimal"
            return True, pyo.value(get_active_objective(subproblem_model))
        
        # if the solution is STOPPED SHORT: try to load incumbent, then retrieve lb
        if results.solver.termination_condition in [TerminationCondition.maxTimeLimit, 
                                                    TerminationCondition.maxIterations,
                                                    TerminationCondition.maxEvaluations,
                                                    TerminationCondition.feasible]:
            # PATCH 1.1: Must verify solution was actually loaded before using variable values.
            # Without this, Pyomo vars may retain stale values from prior solves when 
            # load_from fails under time limit or when results contains no incumbent.
            solution_loaded = False
            # Check if results contains at least one solution before attempting load
            if hasattr(results, 'solution') and len(results.solution) > 0:
                try:
                    subproblem_model.solutions.load_from(results)
                    solution_loaded = True
                except:
                    pass  # load_from failed; values remain stale
            
            if not solution_loaded:
                # PATCH 1.3: No incumbent available but NOT infeasible - just unusable for this iteration
                # Store reason in side channel; return 2-tuple for compatibility
                self._last_term_reason = "no_incumbent"
                return False, None
            
            subproblem_lb = self.retrieve_solver_lb(results)            # returns -inf if not retrievable
            parent_obj = pyo.value(subproblem_model.successor_obj)      # returns -inf if parent did not have a solution
            self._last_term_reason = "nonoptimal_loaded"
            return True, max(subproblem_lb, parent_obj)
        
        # if the solution is not feasible, return None
        # PATCH 1.3: "infeasible" reason = TRUE infeasibility evidence
        elif results.solver.termination_condition == TerminationCondition.infeasible:
            self._last_term_reason = "infeasible"
            return False, None

        else:
            self._last_term_reason = "error"
            raise RuntimeError(f"unexpected termination_condition for lower bounding problem: {results.solver.termination_condition}")


    def retrieve_solver_lb(self,
                           results) -> float:
        """
        When we do not find an incumbent, or we fail to reach the optimality gap 
        (due to time limit, for example) we can retrieve the lb for the subproblem
        and use this.

        Parameters
        ----------
        results: pyo.SolverResults
            solver results object
        """
        # PATCH 1.2: Safe solver detection to prevent TypeError/crashes
        # self.solver is a SolverFactory object; use safe introspection
        solver_name = getattr(self.solver, 'name', None) or getattr(self.solver, 'type', None) or str(self.solver)
        solver_name_lower = solver_name.lower() if solver_name else ""
        
        if "gurobi" in solver_name_lower:
            try: # retrieve via pyomo's default interface
                return results.problem.lower_bound
            except:
                pass
            try: # retrieve via another method
                return results.problem[0]["Lower bound"]
            except:
                pass
            try: # access gurobipy model, return lower bound directly
                gurobi_model = getattr(self.solver, '_solver_model', None)
                if gurobi_model is not None:
                    return gurobi_model.ObjBound
            except:
                pass
            return float("-inf")
        
        elif "baron" in solver_name_lower:
            try:
                return results.problem.lower_bound
            except:
                return float("-inf")
            
        else:
            # PATCH 1.2: Don't crash for unknown solvers; return -inf as fallback
            try:
                return results.problem.lower_bound
            except:
                return float("-inf")


    def solve_a_subproblem(self, 
                           subproblem_name: str, 
                           subproblem_model: pyo.ConcreteModel, 
                           subproblem_complicating_vars: dict) -> Tuple[bool, Optional[float]]:
        """
        This must be defined in the child class.
        It must always take these inputs, to maintain fluidity within the solver.
        
        Options here could be to simply solve as is, perform OBBT / FBBT, 
        generate a convex relaxation & solve, etc.

        Parameters
        -----------
        subproblem_name : str
            String corresponding to this subproblems name
        subproblem_model : pyo.ConcreteModel
            pyomo model corresponding to this subproblem
        subproblem_complicating_vars : dict
            dictionary corresponding to this current subproblems complicating variables

        Returns
        -----------
        feasible : bool
            if the solve of the subproblem model was feasible or not
        objective : float
            objective value of the subproblem model; None if infeasible.
        """

        print( "The child LB class must have a method called " + \
                "solve_a_subproblem(subproblem_name: str, subproblem_model: pyo.ConcreteModel, " + \
                    "subproblem_complicating_vars: dict{(var_id): pyo.Var}) -> feasible: bool, obj: float")
        raise NotImplementedError

# ================================================================================================ #

class DropNonants(AbstractLowerBounder):
    """
    Most basic lower bounder - drop nonanticipativty and solve each subproblem to global optimality.
    """

    def __init__(self, 
                 solver,
                 inherit_solutions: bool = True) -> None:
        """
        Initializes solver information (via Parent class).

        Parameters
        -----------
        solver : pyo.SolverFactory
            initialized Pyomo solver factory object 
            to be used for the LOWER bounding problem solves.
        inherit_solutions : bool
            if we want to check for inheritable solutions from parent nodes.
        """

        super().__init__(solver = solver, 
                         inherit_solutions = inherit_solutions)
        
    def solve_a_subproblem(self, 
                           subproblem_model: pyo.ConcreteModel, 
                           *args, **kwargs) -> Tuple[bool, Optional[float]]:
        """
        Given a subproblems model, name, and list of listed vars,
        solve the subproblem.
 
        Parameters
        -----------
        subproblem_model : pyo.ConcreteModel
            subproblem's Pyomo model.
        
        Returns
        -----------
        feasible : bool
            if the solve of the subproblem model was feasible or not
        objective : float
            objective value of the subproblem model; None if infeasible.
        """
        
        # solve model
        results = self.opt.solve(subproblem_model,
                                 load_solutions = False, 
                                 symbolic_solver_labels=True,
                                 tee = False)

        return self.evaluate_termination(results = results,
                                         subproblem_model = subproblem_model)


# ================================================================================================ #

class LGLowerBounder(AbstractLowerBounder):
    """
    Lower bounder using Li-Grossmann Lagrangian relaxation.
    Dualizes Non-Anticipativity Constraints (NAC).
    """

    def __init__(self, 
                 solver,
                 inherit_solutions: bool = True,
                 initial_ub_estimate: float = None) -> None:
        """
        Initializes solver information (via Parent class).
        
        Parameters
        ----------
        solver : Solver
            The parent solver instance
        inherit_solutions : bool
            Whether to inherit solutions from parent nodes
        initial_ub_estimate : float, optional
            User-provided UB estimate for K-G step-size when tree UB is inf.
            Set this to a rough estimate of the optimal value (e.g., 1.0 if 
            you expect optimal ~0.9). If None and tree UB is inf, uses fallback.
        """
        super().__init__(solver = solver, 
                         inherit_solutions = inherit_solutions)
        self.K = 10  # Fixed number of iterations
        self._cut_signatures = set()  # For duplicate detection
        self.initial_ub_estimate = initial_ub_estimate  # For K-G step when UB=inf
    
    def _evaluate_lg_subproblem(self, results, model, subproblem_name: str, subproblems):
        """
        Evaluate LG subproblem termination and extract:
        - dual_bound: Lower bound on optimal value (for cut validity)
        - y_vals: Primal solution values (for subgradient updates)
        - reason: Termination condition string
        
        CRITICAL: For cuts to be valid, v_val must be a LOWER BOUND on the 
        subproblem optimal value. For optimal solves, obj = dual_bound. For 
        non-optimal solves, we MUST use the solver's dual bound (ObjBound), 
        NOT the incumbent objective (which is an upper bound).
        
        Returns: (dual_bound, y_vals, reason)
          - dual_bound: float or None (None = cannot generate valid cut)
          - y_vals: dict {var_id: value} or None (None = no primal solution)
          - reason: str
        """
        from pyomo.opt import TerminationCondition, SolverStatus
        
        term_cond = results.solver.termination_condition
        
        # OPTIMAL: primal = dual, both are valid
        if term_cond in [TerminationCondition.optimal, TerminationCondition.globallyOptimal] \
                and results.solver.status == SolverStatus.ok:
            try:
                model.solutions.load_from(results)
            except:
                return None, None, "optimal_load_failed"
            
            obj_val = pyo.value(get_active_objective(model))
            y_vals = self._extract_y_vals(model, subproblem_name, subproblems)
            return obj_val, y_vals, "optimal"
        
        # NON-OPTIMAL with potential incumbent
        if term_cond in [TerminationCondition.maxTimeLimit, 
                         TerminationCondition.maxIterations,
                         TerminationCondition.maxEvaluations,
                         TerminationCondition.feasible]:
            # Get DUAL BOUND (required for valid cuts)
            dual_bound = self.retrieve_solver_lb(results)
            
            # Try to load primal solution for subgradient updates
            y_vals = None
            if hasattr(results, 'solution') and len(results.solution) > 0:
                try:
                    model.solutions.load_from(results)
                    y_vals = self._extract_y_vals(model, subproblem_name, subproblems)
                except:
                    pass  # No primal solution available
            
            if math.isfinite(dual_bound):
                return dual_bound, y_vals, "nonoptimal_with_bound"
            else:
                return None, y_vals, "nonoptimal_no_bound"
        
        # INFEASIBLE
        if term_cond == TerminationCondition.infeasible:
            return None, None, "infeasible"
        
        # OTHER (error)
        return None, None, f"error_{term_cond}"
    
    def _extract_y_vals(self, model, subproblem_name: str, subproblems):
        """Extract y values (complicating vars) from loaded model solution."""
        y_vals = {}
        for vid_var in subproblems.subproblem_complicating_vars[subproblem_name]:
            _, var_id, _ = subproblems.var_to_data[vid_var]
            val = pyo.value(vid_var)
            if val is None or not math.isfinite(val):
                return None  # Invalid values
            y_vals[var_id] = val
        return y_vals

    def solve(self, 
              node: Node, 
              subproblems: Subproblems,
              tree_ub: float = None) -> None:
        """
        Solves for the LB using the LG method.
        
        Parameters
        ----------
        node : Node
            Current tree node
        subproblems : Subproblems
            Subproblem collection
        tree_ub : float, optional
            Best known upper bound from the tree (for K-G step-size)
        
        NOTE: Requires a global solver (Gurobi NonConvex=2 or BARON) to ensure
        valid lower bounds. Local solvers will produce unsafe cuts.
        """
        # 0. Initialization
        # SAFETY PATCH: Validate global solver capability
        import warnings
        solver_name = getattr(self.opt, 'name', None) or getattr(self.opt, 'type', None) or str(self.opt)
        solver_name_lower = solver_name.lower() if solver_name else ""
        
        if 'gurobi' in solver_name_lower:
            # Check if NonConvex=2 is set (required for global optimality on nonconvex problems)
            nc_val = self.opt.options.get('NonConvex', self.opt.options.get('nonconvex', 0))
            if nc_val != 2:
                warnings.warn(
                    f"LGLowerBounder: Gurobi NonConvex={nc_val}. For nonconvex problems, "
                    "set NonConvex=2 to ensure valid global lower bounds. "
                    "Without this, cuts may be based on local optima.",
                    RuntimeWarning
                )
        elif 'baron' not in solver_name_lower and 'scip' not in solver_name_lower:
            warnings.warn(
                f"LGLowerBounder: Solver '{solver_name}' may not provide global optimality. "
                "Consider using Gurobi (NonConvex=2) or BARON for guaranteed lower bounds.",
                RuntimeWarning
            )
        
        statistics = OneLowerBoundSolve(subproblems.names)
        
        # We need a consistent list of ALL complicating variable IDs across all scenarios
        all_var_ids = sorted(subproblems.complicating_var_ids)
        all_scenario_names = subproblems.all_names
        num_scenarios = len(all_scenario_names)

        # Initialize or retrieve multipliers
        # Structure: dict[scenario_name][var_id] -> float
        if not hasattr(node, "lg_multipliers") or not node.lg_multipliers:
             current_mu = {sname: {vid: 0.0 for vid in all_var_ids} for sname in all_scenario_names}
        else:
             current_mu = cp.deepcopy(node.lg_multipliers)
             # Ensure completeness
             for sname in all_scenario_names:
                 if sname not in current_mu:
                     current_mu[sname] = {vid: 0.0 for vid in all_var_ids}
                 else:
                     for vid in all_var_ids:
                         if vid not in current_mu[sname]:
                             current_mu[sname][vid] = 0.0

        # Storage for NEW Cuts generated THIS node solve (local list)
        # These will be added to global cut_pool after generation
        new_cuts = []  
        
        # Extract node bounds for cut pool domain tracking
        # Format: {var_id: (lb, ub)}
        node_var_bounds = {}
        for var_type in node.state:
            for var_id, comp_var in node.state[var_type].items():
                node_var_bounds[var_id] = (comp_var.lb, comp_var.ub)
        
        # ============ K-G STEP-SIZE STATE INITIALIZATION ============
        # Karuppiah-Grossmann adaptive step-size rule
        alpha_kg = 1.0  # Initial alpha in (0, 2]
        best_zLB = float('-inf')  # Best Lagrangian dual bound seen so far
        ub_warning_printed = False  # Only warn once about missing UB
        
        # Use tree_ub, or fall back to initial_ub_estimate if tree_ub is inf
        if tree_ub is not None and math.isfinite(tree_ub):
            global_UB = tree_ub
            ub_source = "tree"
        elif self.initial_ub_estimate is not None:
            global_UB = self.initial_ub_estimate
            ub_source = "estimate"
        else:
            global_UB = float('inf')
            ub_source = "inf"
        
        # Gather probabilities EARLY for zLB computation in inner loop
        local_probs_early = {name: subproblems.probability[name] for name in subproblems.names}
        all_probs_early_list = MPI.COMM_WORLD.gather(local_probs_early, root=0)
        prob_by_scenario_early = {}
        if rank == 0:
            for prob_dict in all_probs_early_list:
                prob_by_scenario_early.update(prob_dict)
        prob_by_scenario_early = MPI.COMM_WORLD.bcast(prob_by_scenario_early, root=0)
        
        # 1. Inner Loop
        for k in range(self.K):
            
            # (A) Broadcast Multipliers
            current_mu = MPI.COMM_WORLD.bcast(current_mu, root=0)

            # (B) Parallel Scenario Solves using new LG-specific evaluator
            # local_results: name -> (dual_bound, y_vals) or None
            local_results = {}
            local_reason = {}
            
            for subproblem_name in subproblems.names:
                model = subproblems.model[subproblem_name]
                
                # Update mu for THIS scenario
                scenario_mu = current_mu[subproblem_name]
                for vid in model.lg_mu:
                    model.lg_mu[vid].set_value(scenario_mu.get(vid, 0.0))

                # Relaxations
                if subproblems.relax_binaries: subproblems.relax_all_binaries()
                if subproblems.relax_integers: subproblems.relax_all_integers() 
                
                # Solve
                results = self.opt.solve(model, load_solutions=False, tee=False)
                
                # Use new LG-specific evaluator that returns dual bound for cuts
                try:
                    dual_bound, y_vals, reason = self._evaluate_lg_subproblem(
                        results, model, subproblem_name, subproblems)
                except Exception as e:
                    dual_bound, y_vals, reason = None, None, f"exception_{e}"
                
                local_results[subproblem_name] = (dual_bound, y_vals)
                local_reason[subproblem_name] = reason

            # (C) Gather Results
            all_local_results = MPI.COMM_WORLD.gather(local_results, root=0)
            all_local_reasons = MPI.COMM_WORLD.gather(local_reason, root=0)
            
            # Process results on Rank 0
            node_infeasible = False
            global_results = {}  # name -> (dual_bound, y_vals)
            global_reason = {}
            scenarios_with_cuts = []  # Scenarios that can generate valid cuts
            scenarios_with_primal = []  # Scenarios with primal solutions for subgradient

            if rank == 0:
                # Merge from all ranks
                for reason_dict in all_local_reasons:
                    global_reason.update(reason_dict)
                
                for res_dict in all_local_results:
                    for name, (dual_bound, y_vals) in res_dict.items():
                        reason = global_reason.get(name, "unknown")
                        
                        if reason == "infeasible":
                            node_infeasible = True
                            break
                        
                        # Store result
                        global_results[name] = (dual_bound, y_vals)
                        
                        if dual_bound is not None and math.isfinite(dual_bound):
                            scenarios_with_cuts.append(name)
                        if y_vals is not None:
                            scenarios_with_primal.append(name)
                    
                    if node_infeasible:
                        break
            
            node_infeasible = MPI.COMM_WORLD.bcast(node_infeasible, root=0)
            if node_infeasible:
                node.lb_problem.is_infeasible()
                node.ub_problem.is_infeasible()
                return
            
            # Broadcast categorization
            scenarios_with_cuts = MPI.COMM_WORLD.bcast(scenarios_with_cuts, root=0)
            scenarios_with_primal = MPI.COMM_WORLD.bcast(scenarios_with_primal, root=0)

            # PATCH 1.4: (A) Fix MPI deadlock - synchronize early-break across ALL ranks
            # Compute break condition on rank0, broadcast to all ranks
            should_break = False
            if rank == 0:
                should_break = (len(global_results) == 0)
            should_break = MPI.COMM_WORLD.bcast(should_break, root=0)
            
            if should_break:
                # No valid scenario data this iteration; break inner loop on ALL ranks
                break 

            # (D) Rank 0 Updates (Cuts & Multipliers) with DIAGNOSTICS
            if rank == 0:
                # ============ LG DIAGNOSTICS START ============
                # Compute mu statistics
                all_mu_values = []
                for sname in all_scenario_names:
                    for vid in all_var_ids:
                        all_mu_values.append(current_mu[sname][vid])
                
                if all_mu_values:
                    mu_norm = math.sqrt(sum(m**2 for m in all_mu_values))
                    mu_max = max(abs(m) for m in all_mu_values)
                else:
                    mu_norm, mu_max = 0.0, 0.0
                
                # ============ K-G STEP-SIZE COMPUTATION (FIXED) ============
                
                # (C) Compute zLB_k = sum(prob_s * v_s) from valid dual bounds only
                zLB_k = 0.0
                num_valid_dual = 0
                total_prob_valid = 0.0
                for name in scenarios_with_cuts:
                    dual_bound, y_vals = global_results[name]
                    if dual_bound is not None and math.isfinite(dual_bound):
                        prob_s = prob_by_scenario_early.get(name, 1.0 / num_scenarios)
                        zLB_k += prob_s * dual_bound
                        total_prob_valid += prob_s
                        num_valid_dual += 1
                
                # Scale zLB_k if not all scenarios contributed (optional, for consistency)
                # Keep raw sum for now as it represents partial Lagrangian bound
                
                # (D) Stable alpha-halving: ONLY when UB is finite and zLB reliable
                alpha_halved = False
                alpha_halve_reason = ""
                fraction_valid = num_valid_dual / max(1, num_scenarios)
                
                # Only halve alpha if: (1) UB is finite, (2) enough scenarios valid, (3) no improvement
                if math.isfinite(global_UB) and fraction_valid >= 0.8 and math.isfinite(zLB_k):
                    if zLB_k <= best_zLB + 1e-6:  # Tolerance for improvement
                        alpha_kg = max(0.001, alpha_kg * 0.5)  # Floor at 0.001 to prevent numerical zero
                        alpha_halved = True
                        alpha_halve_reason = "no_improve"
                    else:
                        best_zLB = zLB_k
                elif math.isfinite(zLB_k) and zLB_k > best_zLB:
                    # Even if UB inf, update best_zLB if it improved
                    best_zLB = zLB_k
                # If UB is inf, don't halve alpha (prevents runaway halving)
                
                # (B) Compute y_bar and denom using PROBABILITY WEIGHTS (once)
                # y_bar[var] = sum_s prob[s] * y_s[var]
                y_bar_weighted = {vid: 0.0 for vid in all_var_ids}
                total_prob_primal = 0.0
                for name in scenarios_with_primal:
                    dual_bound, y_vals = global_results[name]
                    if y_vals:
                        prob_s = prob_by_scenario_early.get(name, 1.0 / num_scenarios)
                        for vid in all_var_ids:
                            y_bar_weighted[vid] += prob_s * y_vals.get(vid, 0.0)
                        total_prob_primal += prob_s
                
                # Normalize y_bar by total probability used
                if total_prob_primal > 1e-12:
                    for vid in all_var_ids:
                        y_bar_weighted[vid] /= total_prob_primal
                
                # denom = sum_s prob[s] * sum_var (y_s[var] - y_bar[var])^2
                denom = 0.0
                for name in scenarios_with_primal:
                    dual_bound, y_vals = global_results[name]
                    if y_vals:
                        prob_s = prob_by_scenario_early.get(name, 1.0 / num_scenarios)
                        scenario_sq_diff = 0.0
                        for vid in all_var_ids:
                            diff = y_vals.get(vid, 0.0) - y_bar_weighted[vid]
                            scenario_sq_diff += diff ** 2
                        denom += prob_s * scenario_sq_diff
                
                denom = max(1e-12, denom)
                
                # (A) Compute gap using EFFECTIVE UB (use initial_ub_estimate if tree UB is inf)
                # Use best_zLB (monotonic) for stable gap computation
                UB_eff = global_UB if math.isfinite(global_UB) else (self.initial_ub_estimate if self.initial_ub_estimate else 1.0)
                gap_term = max(0.0, UB_eff - best_zLB)
                
                if not math.isfinite(global_UB) and not ub_warning_printed:
                    print(f"WARNING: K-G using UB_eff={UB_eff:.6f} (from initial_ub_estimate) since tree UB=inf")
                    ub_warning_printed = True
                
                # (E) K-G step-size formula: t^k = alpha * (UB_eff - zLB_best) / ||g||^2
                step_unclamped = alpha_kg * gap_term / denom
                
                # TIGHT step clamp to prevent first-step explosion (use 0.1, not 10.0)
                MAX_STEP = 0.1
                MIN_STEP = 1e-8  # Prevent complete stall
                step_size = max(MIN_STEP, min(step_unclamped, MAX_STEP))
                step_clamped = (step_size != step_unclamped)
                
                # (F) K-G Diagnostics (always print for debugging)
                halved_str = f" (alpha HALVED: {alpha_halve_reason})" if alpha_halved else ""
                clamp_str = " [CLAMPED]" if step_clamped else ""
                ub_label = "tree" if math.isfinite(global_UB) else ("estimate" if self.initial_ub_estimate else "default")
                print(f"LG K-G [k={k}]: alpha={alpha_kg:.4f}, zLB_k={zLB_k:.6f}, zLB_best={best_zLB:.6f}, "
                      f"UB_eff={UB_eff:.6f}[{ub_label}], gap={gap_term:.6f}, denom={denom:.6f}, "
                      f"step={step_unclamped:.6f}->{step_size:.6f}{clamp_str}, valid={num_valid_dual}/{num_scenarios}{halved_str}")
                
                print(f"LG DIAG [k={k}]: ||mu||={mu_norm:.4f}, max|mu|={mu_max:.4f}")
                print(f"LG DIAG [k={k}]: scenarios_with_cuts={len(scenarios_with_cuts)}/{num_scenarios}, "
                      f"scenarios_with_primal={len(scenarios_with_primal)}/{num_scenarios}")
                
                # Compute max disagreement |y_s - y_bar| if we have primal solutions
                # First compute y_bar from scenarios_with_primal
                sums_diag = {vid: 0.0 for vid in all_var_ids}
                counts_diag = {vid: 0 for vid in all_var_ids}
                for name in scenarios_with_primal:
                    dual_bound, y_vals = global_results[name]
                    if y_vals:
                        for vid, val in y_vals.items():
                            sums_diag[vid] += val
                            counts_diag[vid] += 1
                
                y_bar_diag = {vid: (sums_diag[vid] / counts_diag[vid] if counts_diag[vid] > 0 else 0.0) 
                              for vid in all_var_ids}
                
                max_disagreement = 0.0
                for name in scenarios_with_primal:
                    dual_bound, y_vals = global_results[name]
                    if y_vals:
                        for vid in all_var_ids:
                            diff = abs(y_vals.get(vid, 0.0) - y_bar_diag[vid])
                            max_disagreement = max(max_disagreement, diff)
                
                print(f"LG DIAG [k={k}]: max_disagreement={max_disagreement:.6f}")
                # ============ LG DIAGNOSTICS END ============
                
                # 1. Generate Cuts (ONLY for scenarios with valid dual bounds)
                num_cuts_added = 0
                for name in scenarios_with_cuts:
                    dual_bound, y_vals = global_results[name]
                    # v_val = dual_bound (NOT primal objective for non-optimal cases!)
                    v_val = dual_bound
                    # Store cut info: (name, mu_vector (copy), v_val)
                    new_cuts.append((name, cp.deepcopy(current_mu[name]), v_val))
                    num_cuts_added += 1
                
                print(f"LG DIAG [k={k}]: cuts_added_this_iter={num_cuts_added}")
                
                # ============ SIGN CONVENTION DEBUG (one scenario, k=0 only) ============
                if k == 0 and len(scenarios_with_cuts) > 0:
                    debug_s = list(scenarios_with_cuts)[0]
                    debug_dual, debug_y = global_results[debug_s]
                    debug_mu = current_mu[debug_s]
                    # Pick first var for illustration
                    debug_vid = list(all_var_ids)[0] if all_var_ids else None
                    if debug_vid is not None and debug_y:
                        mu_i = debug_mu.get(debug_vid, 0.0)
                        y_i = debug_y.get(debug_vid, 0.0)
                        ybar_i = y_bar_weighted.get(debug_vid, 0.0)
                        
                        # Subproblem: min f - mu^T y = v_val
                        # Cut: eta >= v_val + mu^T y
                        # At y_s: cut_rhs_at_y_s = v_val + sum(mu_i * y_s_i)
                        # At y_bar: cut_rhs_at_y_bar = v_val + sum(mu_i * y_bar_i)
                        mu_dot_y_s = sum(debug_mu.get(v, 0) * debug_y.get(v, 0) for v in all_var_ids)
                        mu_dot_y_bar = sum(debug_mu.get(v, 0) * y_bar_weighted.get(v, 0) for v in all_var_ids)
                        cut_rhs_at_y_s = debug_dual + mu_dot_y_s
                        cut_rhs_at_y_bar = debug_dual + mu_dot_y_bar
                        
                        print("=" * 70)
                        print(f"SIGN DEBUG [{debug_s}] at k={k}:")
                        print(f"  Subproblem obj: min f - mu^T y  (code uses MINUS)")
                        print(f"  v_val (dual_bound) = {debug_dual:.6f}")
                        print(f"  First var [{debug_vid}]: mu={mu_i:.6f}, y_s={y_i:.6f}, y_bar={ybar_i:.6f}")
                        print(f"  mu dot y_s = {mu_dot_y_s:.6f}")
                        print(f"  mu dot y_bar = {mu_dot_y_bar:.6f}")
                        print(f"  Cut RHS at y_s: v_val + mu.y_s = {cut_rhs_at_y_s:.6f}")
                        print(f"  Cut RHS at y_bar: v_val + mu.y_bar = {cut_rhs_at_y_bar:.6f}")
                        print(f"  Subgradient (ascent): g_s[{debug_vid}] = y_bar - y_s = {ybar_i - y_i:.6f}")
                        print(f"  Update direction: mu += step * (y_bar - y_s)  [ASCENT]")
                        print("=" * 70)
                
                # 1b. Store new cuts in global cut pool (if available)
                if hasattr(self, 'cut_pool') and self.cut_pool is not None:
                    self.cut_pool.add_cuts_from_iteration(
                        cuts_data=[(name, cp.deepcopy(current_mu[name]), global_results[name][0]) 
                                   for name in scenarios_with_cuts],
                        iteration=k,
                        node_id=node.id,
                        y_bounds=node_var_bounds
                    )

                # 2. Update Multipliers (using PRIMAL solutions, not dual bounds)
                # -----------------------------------------------------------------
                # DERIVATION (Lagrangian Relaxation with consensus constraint):
                #   Subproblem: v_s(mu) = min_y [f_s(y) - mu_s^T y]
                #   Dual function: D(mu) = sum_s p_s * v_s(mu_s)  (we MAXIMIZE this)
                #   Subgradient: dD/d(mu_s) = p_s * (-y_s^*)  (envelope theorem)
                #   With projection to sum_s mu_s = 0, the ascent direction becomes:
                #       g_s = y_bar - y_s   (NOT y_s - y_bar!)
                # -----------------------------------------------------------------
                subgradients = {}
                for name in scenarios_with_primal:
                    dual_bound, y_vals = global_results[name]
                    if y_vals:
                        subgradients[name] = {}
                        for vid in all_var_ids:
                            y_val = y_vals.get(vid, 0.0)
                            # CORRECT SIGN: ascent direction is (y_bar - y_s)
                            subgradients[name][vid] = y_bar_weighted[vid] - y_val
                
                # Apply subgradient ASCENT update: mu += step_size * (y_bar - y_s)
                for name in subgradients:
                    for vid in subgradients[name]:
                        current_mu[name][vid] += step_size * subgradients[name][vid]
                
                # Project over ALL scenarios to enforce Σω μω = 0
                for vid in all_var_ids:
                    sum_mu = sum(current_mu[s][vid] for s in all_scenario_names)
                    avg_mu = sum_mu / num_scenarios
                    for s in all_scenario_names:
                        current_mu[s][vid] -= avg_mu
                
                # STABILIZATION 2: Clip mu to prevent blow-up
                MU_CAP = 100.0  # Safety cap on multiplier magnitude
                for s in all_scenario_names:
                    for vid in all_var_ids:
                        current_mu[s][vid] = max(-MU_CAP, min(MU_CAP, current_mu[s][vid]))
        
        # End Inner Loop
        
        # 2. Solve Relaxed Master Problem (RMP) on Rank 0
        rmp_obj = None
        rmp_feasible = False
        use_conservative_lb = False  # PATCH 1.3: Flag for missing cuts scenario
        
        # PATCH 3: Gather scenario probabilities from ALL ranks BEFORE rank0-only block (avoid MPI deadlock)
        local_probs = {name: subproblems.probability[name] for name in subproblems.names}
        all_probs_list = MPI.COMM_WORLD.gather(local_probs, root=0)
        
        # Merge gathered probabilities on rank 0
        prob_by_scenario = {}
        if rank == 0:
            for prob_dict in all_probs_list:
                prob_by_scenario.update(prob_dict)
            # Sanity check: all scenarios have probability and sum ≈ 1
            missing = [s for s in all_scenario_names if s not in prob_by_scenario]
            assert len(missing) == 0, f"Missing probabilities for scenarios: {missing}"
            prob_sum = sum(prob_by_scenario.values())
            assert abs(prob_sum - 1.0) < 1e-6, f"Probabilities sum to {prob_sum}, expected ~1.0"
        
        if rank == 0:
            # Combine new cuts with valid cuts from global pool
            all_cuts = list(new_cuts)  # Start with new cuts from this solve
            
            # Get valid cuts from global pool (if available)
            if hasattr(self, 'cut_pool') and self.cut_pool is not None:
                pool_cuts = self.cut_pool.get_cuts_for_rmp(node_var_bounds, all_scenario_names)
                # Avoid duplicates: pool_cuts may include cuts just added, but that's OK
                # since RMP handles duplicate cuts gracefully
                all_cuts.extend(pool_cuts)
                
                # DEBUG: Print cut statistics and domain filtering info
                total_in_pool = len(self.cut_pool)
                valid_for_domain = len(pool_cuts)
                filtered_out = total_in_pool - valid_for_domain
                print(f"DEBUG RMP: Node {node.id}, new_cuts={len(new_cuts)}, pool_cuts={valid_for_domain}, "
                      f"total={len(all_cuts)}, pool_size={total_in_pool}, filtered_by_domain={filtered_out}")
            
            # PATCH 1.3: (C) Guard against RMP unboundedness when some scenarios have no cuts
            # Count cuts per scenario
            cut_count = {s: 0 for s in all_scenario_names}
            for (s_name, _, _) in all_cuts:
                cut_count[s_name] += 1
            
            scenarios_no_cuts = [s for s in all_scenario_names if cut_count[s] == 0]
            
            if len(scenarios_no_cuts) > 0:
                # Cannot solve RMP: some scenarios have no cuts -> eta would be unbounded
                # PATCH 1.4: (B) Use robust conservative LB with priority:
                # 1) Parent node's LB if available and finite
                # 2) Fall back to -inf
                use_conservative_lb = True
                conservative_lb = float('-inf')
                
                # Try to get parent LB from node.parent.lb_problem if available
                parent_node = getattr(node, 'parent', None)
                if parent_node is not None:
                    parent_lb_problem = getattr(parent_node, 'lb_problem', None)
                    if parent_lb_problem is not None:
                        parent_lb_obj = getattr(parent_lb_problem, 'objective', None)
                        if parent_lb_obj is not None and math.isfinite(parent_lb_obj):
                            conservative_lb = parent_lb_obj
                
                rmp_obj = conservative_lb
                rmp_feasible = True  # We have a valid (conservative) LB, not infeasible
                print(f"DEBUG RMP: Node {node.id} - no cuts for scenarios {scenarios_no_cuts}, using conservative LB={conservative_lb}")
            else:
                rmp_feasible, rmp_obj = self._solve_rmp(node, subproblems, all_cuts, all_var_ids, all_scenario_names, prob_by_scenario)
                print(f"DEBUG RMP: Node {node.id}, RMP objective = {rmp_obj}")
                
                # ============ RMP > UB CRITICAL WARNING ============
                # Get current global UB from tree metrics
                global_ub = getattr(node, '_solver_ub', None)  # May not be available
                if global_ub is None:
                    # Try to get from tree if available
                    try:
                        global_ub = self.solver.tree.metrics.ub if hasattr(self, 'solver') and hasattr(self.solver, 'tree') else None
                    except:
                        global_ub = None
                
                if rmp_obj is not None and global_ub is not None and math.isfinite(global_ub):
                    if rmp_obj > global_ub + 1e-8:
                        print("=" * 80)
                        print("!!! CRITICAL WARNING: RMP objective > UB !!!")
                        print(f"    Node {node.id}: RMP_obj={rmp_obj:.8f}, UB={global_ub:.8f}")
                        print(f"    This indicates INVALID CUTS were generated!")
                        print("")
                        print(f"    Cuts per scenario in RMP:")
                        for s in all_scenario_names:
                            print(f"      {s}: {cut_count.get(s, 0)} cuts")
                        print("")
                        print(f"    Scenario termination reasons this node:")
                        for s in all_scenario_names:
                            reason = global_reason.get(s, "not_solved")
                            db, yv = global_results.get(s, (None, None))
                            print(f"      {s}: reason={reason}, dual_bound={db}")
                        print("=" * 80)
        
        # Broadcast Result
        rmp_info = (rmp_feasible, rmp_obj, use_conservative_lb)
        rmp_feasible, rmp_obj, use_conservative_lb = MPI.COMM_WORLD.bcast(rmp_info, root=0)
        
        if rmp_feasible:
            # PATCH 1: FIX MPI AGGREGATION BUG
            # Only rank 0 contributes objective value to SUM allreduce in Solver.dispatch_lb_solve().
            # Non-root ranks contribute 0.0 so SUM yields correct value (not rmp_obj * size).
            if rank == 0:
                statistics.aggregated_objective = rmp_obj
            else:
                statistics.aggregated_objective = 0.0  # Non-root contributes 0 for SUM
            node.lb_problem.is_feasible(statistics)
            
            # Update Node LG State (Rank 0 sends best/last mu?)
            # We use current_mu (last updated).
            node.lg_multipliers = MPI.COMM_WORLD.bcast(current_mu, root=0)
            
        else:
            node.lb_problem.is_infeasible()
            node.ub_problem.is_infeasible()


    def _solve_rmp(self, node, subproblems, cuts, all_var_ids, all_scenario_names, prob_by_scenario):
        """
        Builds and solves the RMP on Rank 0.
        
        Parameters:
            prob_by_scenario: Dict[str, float] - probability for each scenario (gathered from all ranks)
        
        Returns (feasible, objective_value)
        """
        m = pyo.ConcreteModel()
        
        # Variables
        # y: common decision variables (bounded by node)
        m.y = pyo.Var(all_var_ids, domain=pyo.Reals)
        
        # Set bounds on y from Node state
        # Note: Node bounds are usually stored by type. We need to find type for each ID.
        # But we can look at subproblems.root_node_state? No, current node `state`.
        # Accessing `node.state` requires mapping `vid` to `type`.
        # `subproblems.var_to_data` is [var_obj] -> (type, id, name).
        # We need ID -> Type.
        # We can build a map efficiently or search.
        # Efficient: `subproblems` has `id_to_vars` -> gets var objs.
        
        for vid in all_var_ids:
            # Get variable limits from node state
            # Find domain/type
            # We can pick the first var object associated with this ID
            var_obj = subproblems.id_to_vars[vid][0]
            var_type, _, _ = subproblems.var_to_data[var_obj]
            
            lb = node.state[var_type][vid].lb
            ub = node.state[var_type][vid].ub
            m.y[vid].setlb(lb)
            m.y[vid].setub(ub)

        # eta: scenario approximation
        m.eta = pyo.Var(all_scenario_names, domain=pyo.Reals)
        
        # Objective: sum(p_s * eta_s) using actual probabilities (passed in from gather)
        m.obj = pyo.Objective(
            expr=sum(prob_by_scenario[s] * m.eta[s] for s in all_scenario_names), 
            sense=pyo.minimize
        )
        
        # Constraints (Cuts)
        m.cuts = pyo.ConstraintList()
        for (s_name, mu_vec, v_val) in cuts:
            # Derivation: v_val = min_y [f_ω(y) - μ^T y], so for any y: f_ω(y) - μ^T y ≥ v_val
            # Rearranging: f_ω(y) ≥ v_val + μ^T y, hence the cut η_ω ≥ v_val + μ^T y
            term = sum(mu_vec[v] * m.y[v] for v in all_var_ids)
            m.cuts.add(m.eta[s_name] >= v_val + term)
            
        # PATCH 2: Use configured solver (self.opt) instead of hardcoded 'gurobi'
        # This respects user's solver configuration and works on systems without gurobi
        res = self.opt.solve(m, tee=False, load_solutions=False)
        
        if res.solver.termination_condition in [TerminationCondition.optimal, TerminationCondition.globallyOptimal]:
            # Load solution for diagnostics
            try:
                m.solutions.load_from(res)
            except:
                return True, pyo.value(m.obj)
            
            obj_val = pyo.value(m.obj)
            
            # ============ RMP DIAGNOSTIC 1: Y SATURATION ============
            num_y_at_bound = 0
            total_y = len(all_var_ids)
            min_distance_all = float('inf')
            y_diag_lines = []
            
            for vid in all_var_ids:
                y_val = pyo.value(m.y[vid])
                lb = m.y[vid].lb
                ub = m.y[vid].ub
                
                # Compute distance to bounds safely
                dist_to_lb = (y_val - lb) if lb is not None else float('inf')
                dist_to_ub = (ub - y_val) if ub is not None else float('inf')
                dist_to_bounds = min(dist_to_lb, dist_to_ub)
                is_at_bound = (dist_to_bounds <= 1e-6)
                
                if is_at_bound:
                    num_y_at_bound += 1
                
                if dist_to_bounds < min_distance_all:
                    min_distance_all = dist_to_bounds
                
                # Only log key vars or first few (Kp, Ki, Kd, or first 3)
                if len(y_diag_lines) < 5 or 'Kp' in str(vid) or 'Ki' in str(vid) or 'Kd' in str(vid):
                    bound_flag = "AT_BOUND" if is_at_bound else ""
                    y_diag_lines.append(f"      {vid}: y={y_val:.6f}, dist_to_bounds={dist_to_bounds:.6f} {bound_flag}")
            
            print("=" * 60)
            print("RMP DIAGNOSTIC 1: Y Variable Saturation")
            print(f"  y_at_bound: {num_y_at_bound}/{total_y}")
            print(f"  min_dist_to_bounds: {min_distance_all:.6f}")
            for line in y_diag_lines[:5]:
                print(line)
            
            # ============ RMP DIAGNOSTIC 2: CUT SLACK STATISTICS ============
            total_cuts_in_rmp = len(cuts)
            total_tight = 0
            total_slack = 0
            total_violated = 0
            cuts_by_scenario = {}  # scenario -> list of slacks
            
            for idx, (s_name, mu_vec, v_val) in enumerate(cuts):
                # Compute RHS: v_val + sum(mu_i * y_i)
                eta_val = pyo.value(m.eta[s_name])
                mu_y_term = sum(mu_vec[v] * pyo.value(m.y[v]) for v in all_var_ids)
                rhs_val = v_val + mu_y_term
                slack = eta_val - rhs_val
                
                if s_name not in cuts_by_scenario:
                    cuts_by_scenario[s_name] = {'total': 0, 'tight': 0, 'violated': 0}
                cuts_by_scenario[s_name]['total'] += 1
                
                if abs(slack) <= 1e-6:
                    total_tight += 1
                    cuts_by_scenario[s_name]['tight'] += 1
                elif slack < -1e-7:
                    total_violated += 1
                    cuts_by_scenario[s_name]['violated'] += 1
                    print(f"  !! VIOLATED CUT #{idx}: scenario={s_name}, slack={slack:.8f}")
                else:
                    total_slack += 1
            
            print("")
            print("RMP DIAGNOSTIC 2: Cut Slack Statistics")
            print(f"  total_cuts={total_cuts_in_rmp}, tight={total_tight}, slack_positive={total_slack}, VIOLATED={total_violated}")
            
            # Show per-scenario breakdown (top 3 with most tight cuts)
            scenario_ranked = sorted(cuts_by_scenario.items(), 
                                     key=lambda x: x[1]['tight'], reverse=True)[:3]
            print("  Per-scenario (top 3 by tight cuts):")
            for s_name, stats in scenario_ranked:
                print(f"    {s_name}: tight={stats['tight']}/{stats['total']}, violated={stats['violated']}")
            print("=" * 60)
            
            return True, obj_val
        else:
            return False, None


    def solve_a_subproblem(self, *args, **kwargs):
        raise NotImplementedError("Use solve() directly for LG.")