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
                 inherit_solutions: bool = True) -> None:
        """
        Initializes solver information (via Parent class).
        """
        super().__init__(solver = solver, 
                         inherit_solutions = inherit_solutions)
        self.K = 5  # Fixed number of iterations

    def solve(self, 
              node: Node, 
              subproblems: Subproblems) -> None:
        """
        Solves for the LB using the LG method.
        
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
        
        # 1. Inner Loop
        for k in range(self.K):
            
            # (A) Broadcast Multipliers
            current_mu = MPI.COMM_WORLD.bcast(current_mu, root=0)

            # (B) Parallel Scenario Solves
            local_results = {} # name -> (obj, y_vals_dict) or None for failure
            local_reason = {}  # PATCH 1.3: Separate dict for reasons to keep local_results format clean
            
            for subproblem_name in subproblems.names:
                model = subproblems.model[subproblem_name]
                
                # Update mu for THIS scenario
                scenario_mu = current_mu[subproblem_name]
                for vid in model.lg_mu:
                    model.lg_mu[vid].set_value(scenario_mu.get(vid, 0.0))

                # Relaxations
                if subproblems.relax_binaries: subproblems.relax_all_binaries()
                if subproblems.relax_integers: subproblems.relax_all_integers() 
                
                # Activate bound cuts if needed (LB monotonicity) - Optional for LG, usually standard LB logic
                # For strict LG, we trust the RMP. But we can keep local cuts if they help (and don't conflict with dual)
                # Let's Skip standard bound cuts to rely on LG pure logic first.
                
                # Solve
                results = self.opt.solve(model, load_solutions=False, tee=False)
                
                termination_ok = False
                obj_val = float("inf")
                reason = "unknown"
                
                try:
                    # PATCH 1.3: evaluate_termination returns 2-tuple, reason in side channel
                    status_bool, obj = self.evaluate_termination(results, model)
                    reason = getattr(self, '_last_term_reason', 'unknown')
                    if status_bool:
                        termination_ok = True
                        obj_val = obj
                except Exception:
                    reason = "exception"
                
                # PATCH 1.3: Standardize local_results format
                # Success: (obj_val, y_vals) where obj_val is float and y_vals is dict
                # Failure: None (store reason separately in local_reason)
                if not termination_ok:
                    local_results[subproblem_name] = None
                    local_reason[subproblem_name] = reason
                else:
                    # Extract y values (linking vars)
                    # Validate values are finite to avoid corrupted subgradient updates
                    y_vals = {}
                    values_valid = True
                    for vid_var in subproblems.subproblem_complicating_vars[subproblem_name]:
                         # subproblems.subproblem_complicating_vars[name] is a list of VAR OBJECTS
                         _, var_id, _ = subproblems.var_to_data[vid_var]
                         val = pyo.value(vid_var)
                         # Check if value is a finite number
                         if val is None or not math.isfinite(val):
                             values_valid = False
                             break
                         y_vals[var_id] = val
                    
                    if values_valid:
                        local_results[subproblem_name] = (obj_val, y_vals)
                        local_reason[subproblem_name] = reason
                    else:
                        # Invalid values: treat as unusable for this iteration (not infeasible)
                        local_results[subproblem_name] = None
                        local_reason[subproblem_name] = "invalid_values"

            # (C) Gather Results
            all_local_results = MPI.COMM_WORLD.gather(local_results, root=0)
            all_local_reasons = MPI.COMM_WORLD.gather(local_reason, root=0)
            
            # PATCH 1.3: Distinguish true infeasibility from missing incumbent
            # global_reason stores reasons for ALL scenarios for this iteration
            node_infeasible = False
            missing_scenarios = []  # Scenarios with no usable data this iteration
            global_results = {} # name -> (obj, y_vals)
            global_reason = {}  # name -> reason string

            if rank == 0:
                # Merge reasons from all ranks
                for reason_dict in all_local_reasons:
                    global_reason.update(reason_dict)
                
                # Process results
                for res_dict in all_local_results:
                    for name, val in res_dict.items():
                        if val is None:
                            # Failed scenario - check reason
                            reason = global_reason.get(name, "unknown")
                            if reason == "infeasible":
                                # TRUE infeasibility: node is infeasible
                                node_infeasible = True
                                break
                            else:
                                # no_incumbent, invalid_values, exception, etc.
                                # NOT infeasible, just unusable this iteration
                                missing_scenarios.append(name)
                        else:
                            # Success: val is (obj, y_vals)
                            global_results[name] = val
                    if node_infeasible: 
                        break
            
            node_infeasible = MPI.COMM_WORLD.bcast(node_infeasible, root=0)
            if node_infeasible:
                node.lb_problem.is_infeasible()
                node.ub_problem.is_infeasible()
                return
            
            # PATCH 1.4: (A) Fix MPI deadlock - synchronize early-break across ALL ranks
            # Compute break condition on rank0, broadcast to all ranks
            should_break = False
            if rank == 0:
                should_break = (len(global_results) == 0)
            should_break = MPI.COMM_WORLD.bcast(should_break, root=0)
            
            if should_break:
                # No valid scenario data this iteration; break inner loop on ALL ranks
                break 

            # (D) Rank 0 Updates (Cuts & Multipliers)
            if rank == 0:
                # 1. Generate Cuts (store locally for this iteration)
                for name in global_results:
                    v_val, y_vals = global_results[name]
                    # Store cut info: (name, mu_vector (copy), v_val)
                    new_cuts.append( (name, cp.deepcopy(current_mu[name]), v_val) )
                
                # 1b. Store new cuts in global cut pool (if available)
                if hasattr(self, 'cut_pool') and self.cut_pool is not None:
                    self.cut_pool.add_cuts_from_iteration(
                        cuts_data=[(name, cp.deepcopy(current_mu[name]), v_val) 
                                   for name, (v_val, _) in global_results.items()],
                        iteration=k,
                        node_id=node.id,
                        y_bounds=node_var_bounds
                    )

                # 2. Update Multipliers
                # Calculate y_bar (Consensus)
                sums = {vid: 0.0 for vid in all_var_ids}
                counts = {vid: 0 for vid in all_var_ids}
                
                for name in global_results:
                    _, y_vals = global_results[name]
                    for vid, val in y_vals.items():
                        sums[vid] += val
                        counts[vid] += 1
                
                y_bar = {}
                for vid in all_var_ids:
                    if counts[vid] > 0:
                        y_bar[vid] = sums[vid] / counts[vid]
                    else:
                        y_bar[vid] = 0.0

                # Subgradient Check & Step Size
                # Calculate subgradient norm? 
                # g_omega = y_omega - y_bar
                # Let's just use simple step size rule provided in requirements
                # t_k = t0 / sqrt(k+1)
                t0 = 1.0 # Default
                step_size = t0 / math.sqrt(k + 1)

                # Update Unprojected mu (only for scenarios with valid y*)
                # PATCH 1.3: Missing scenarios keep μ unchanged (no subgradient update)
                for name in global_results:
                    _, y_vals = global_results[name]
                    for vid in all_var_ids:
                        y_val = y_vals.get(vid, 0.0) # Should exist if linking var
                        g = y_val - y_bar[vid]
                        current_mu[name][vid] += step_size * g
                
                # PATCH 1.3: Project over ALL scenarios to enforce Σω μω = 0
                # This is critical: projection must include ALL scenarios, not just those with valid data
                for vid in all_var_ids:
                    sum_mu = sum(current_mu[s][vid] for s in all_scenario_names)
                    avg_mu = sum_mu / num_scenarios
                    for s in all_scenario_names:
                        current_mu[s][vid] -= avg_mu
        
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
            else:
                rmp_feasible, rmp_obj = self._solve_rmp(node, subproblems, all_cuts, all_var_ids, all_scenario_names, prob_by_scenario)
        
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
        res = self.opt.solve(m, tee=False)
        
        if res.solver.termination_condition in [TerminationCondition.optimal, TerminationCondition.globallyOptimal]:
            return True, pyo.value(m.obj)
        else:
            return False, None


    def solve_a_subproblem(self, *args, **kwargs):
        raise NotImplementedError("Use solve() directly for LG.")