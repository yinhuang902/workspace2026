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
    Lower bounder using Karuppiah-Grossmann (2008) Appendix Lagrangian relaxation.
    Dualizes Non-Anticipativity Constraints (NAC) using chain-based adjacent
    difference multipliers (NOT consensus/PH-style).
    """

    def __init__(self, 
                 solver,
                 inherit_solutions: bool = True,
                 initial_ub_estimate: float = None):
        """
        Parameters
        ----------
        solver : Solver
            The parent solver instance
        inherit_solutions : bool
            Whether to inherit solutions from parent nodes
        initial_ub_estimate : float, optional
            User-provided UB estimate for step-size when tree UB is inf.
        """
        super().__init__(solver = solver, 
                         inherit_solutions = inherit_solutions)
        self.K = 10              # Fixed number of inner-loop iterations
        self._cut_signatures = set()
        self.initial_ub_estimate = initial_ub_estimate
        # Appendix step-size parameters
        self.alpha_init = 2.0    # alpha^0 in (0, 2]
        self.alpha_min = 1e-6    # floor for alpha halving
        self.improvement_tol = 1e-6  # zLB improvement tolerance
    
    # ------------------------------------------------------------------
    # Subproblem evaluation helpers (unchanged from previous version)
    # ------------------------------------------------------------------
    def _evaluate_lg_subproblem(self, results, model, subproblem_name: str, subproblems):
        """
        Evaluate LG subproblem termination and extract dual_bound + primal y.
        
        Returns: (dual_bound, y_vals, reason)
        """
        from pyomo.opt import TerminationCondition, SolverStatus
        
        term_cond = results.solver.termination_condition
        
        if term_cond in [TerminationCondition.optimal, TerminationCondition.globallyOptimal] \
                and results.solver.status == SolverStatus.ok:
            try:
                model.solutions.load_from(results)
            except:
                return None, None, "optimal_load_failed"
            obj_val = pyo.value(get_active_objective(model))
            y_vals = self._extract_y_vals(model, subproblem_name, subproblems)
            return obj_val, y_vals, "optimal"
        
        if term_cond in [TerminationCondition.maxTimeLimit, 
                         TerminationCondition.maxIterations,
                         TerminationCondition.maxEvaluations,
                         TerminationCondition.feasible]:
            dual_bound = self.retrieve_solver_lb(results)
            y_vals = None
            if hasattr(results, 'solution') and len(results.solution) > 0:
                try:
                    model.solutions.load_from(results)
                    y_vals = self._extract_y_vals(model, subproblem_name, subproblems)
                except:
                    pass
            if math.isfinite(dual_bound):
                return dual_bound, y_vals, "nonoptimal_with_bound"
            else:
                return None, y_vals, "nonoptimal_no_bound"
        
        if term_cond == TerminationCondition.infeasible:
            return None, None, "infeasible"
        
        return None, None, f"error_{term_cond}"
    
    def _extract_y_vals(self, model, subproblem_name: str, subproblems):
        """Extract y values (complicating vars) from loaded model solution."""
        y_vals = {}
        for vid_var in subproblems.subproblem_complicating_vars[subproblem_name]:
            _, var_id, _ = subproblems.var_to_data[vid_var]
            val = pyo.value(vid_var)
            if val is None or not math.isfinite(val):
                return None
            y_vals[var_id] = val
        return y_vals

    # ------------------------------------------------------------------
    # lam -> mu  incidence mapping (chain graph)
    # ------------------------------------------------------------------
    @staticmethod
    def _compute_effective_mu(lambda_y, scenario_order, all_var_ids):
        """
        Convert chain-edge multipliers lam to per-scenario effective mu.

        For a chain  0 --lam[0]-- 1 --lam[1]-- 2 -- ... -- (N-1),
        the NAC constraints are  y_n - y_{n+1} = 0  for n = 0..N-2.
        The Lagrangian adds  Sum_n lam[n]^T (y_n - y_{n+1}).
        Collecting terms per scenario index n:
            mu(0)   =  +lam[0]
            mu(n)   =  +lam[n]  - lam[n-1]     for 1 <= n <= N-2
            mu(N-1) =           - lam[N-2]

        Parameters
        ----------
        lambda_y : list[dict]
            Length N-1.  lambda_y[n][vid] is the multiplier on edge (n, n+1).
        scenario_order : list[str]
            Deterministic ordering of scenarios, length N.
        all_var_ids : list
            Sorted list of complicating variable IDs.

        Returns
        -------
        effective_mu : dict[str, dict]
            {scenario_name: {vid: float}}
        """
        N = len(scenario_order)
        num_edges = N - 1
        effective_mu = {}
        for n, sname in enumerate(scenario_order):
            mu_n = {}
            for vid in all_var_ids:
                val = 0.0
                if n <= num_edges - 1:          # outgoing edge n -> n+1
                    val += lambda_y[n][vid]
                if n >= 1:                      # incoming edge n-1 -> n
                    val -= lambda_y[n - 1][vid]
                mu_n[vid] = val
            effective_mu[sname] = mu_n
        return effective_mu

    # ------------------------------------------------------------------
    # Main solve  (Appendix algorithm)
    # ------------------------------------------------------------------
    def solve(self, 
              node: Node, 
              subproblems: Subproblems,
              tree_ub: float = None) -> None:
        """
        Solves for the LB using the K-G (2008) Appendix Lagrangian method.

        Algorithm per iteration k
        -------------------------
        1. Broadcast lam^k  (rank 0 -> all).
        2. Each rank locally computes mu(s) from lam  (no communication).
        3. Each rank solves its assigned SP_s(mu(s)) independently.
        4. Gather (v_s, y_s) to rank 0.
        5. Rank 0 computes  zLB, step t^k,  updates lam^{k+1}.
        6. Back to 1.
        """
        # 0. Initialization  -----------------------------------------------
        import warnings
        solver_name = getattr(self.opt, 'name', None) or getattr(self.opt, 'type', None) or str(self.opt)
        solver_name_lower = solver_name.lower() if solver_name else ""
        
        if 'gurobi' in solver_name_lower:
            nc_val = self.opt.options.get('NonConvex', self.opt.options.get('nonconvex', 0))
            if nc_val != 2:
                warnings.warn(
                    f"LGLowerBounder: Gurobi NonConvex={nc_val}. "
                    "Set NonConvex=2 for guaranteed global lower bounds.",
                    RuntimeWarning)
        elif 'baron' not in solver_name_lower and 'scip' not in solver_name_lower:
            warnings.warn(
                f"LGLowerBounder: Solver '{solver_name}' may not provide global optimality.",
                RuntimeWarning)
        
        statistics = OneLowerBoundSolve(subproblems.names)
        
        all_var_ids = sorted(subproblems.complicating_var_ids)
        all_scenario_names = subproblems.all_names
        num_scenarios = len(all_scenario_names)

        # Deterministic chain ordering  (fixed across iterations / ranks)
        scenario_order = list(all_scenario_names)
        N = len(scenario_order)
        num_edges = N - 1       # adjacent pairs 0..N-2

        # ---- Initialise / recover chain-edge lam multipliers ----------------
        lambda_y = None
        if hasattr(node, "lg_multipliers") and node.lg_multipliers:
            stored = node.lg_multipliers
            if isinstance(stored, dict) and "lambda_y" in stored:
                stored_ly = stored["lambda_y"]
                if isinstance(stored_ly, list) and len(stored_ly) == num_edges:
                    lambda_y = [
                        {vid: edge_dict.get(vid, 0.0) for vid in all_var_ids}
                        for edge_dict in stored_ly
                    ]
        if lambda_y is None:
            lambda_y = [{vid: 0.0 for vid in all_var_ids} for _ in range(num_edges)]

        new_cuts = []           # cuts generated in THIS node solve

        # Node bounds for cut-pool domain tracking
        node_var_bounds = {}
        for var_type in node.state:
            for var_id, comp_var in node.state[var_type].items():
                node_var_bounds[var_id] = (comp_var.lb, comp_var.ub)

        # ---- Step-size state ------------------------------------------------
        alpha_kg = self.alpha_init
        best_zLB = float('-inf')
        ub_warning_printed = False

        if tree_ub is not None and math.isfinite(tree_ub):
            global_UB = tree_ub
        elif self.initial_ub_estimate is not None:
            global_UB = self.initial_ub_estimate
        else:
            global_UB = float('inf')

        # Gather scenario probabilities (all ranks, to avoid deadlocks)
        local_probs_early = {name: subproblems.probability[name]
                             for name in subproblems.names}
        all_probs_early_list = MPI.COMM_WORLD.gather(local_probs_early, root=0)
        prob_by_scenario_early = {}
        if rank == 0:
            for d in all_probs_early_list:
                prob_by_scenario_early.update(d)
        prob_by_scenario_early = MPI.COMM_WORLD.bcast(prob_by_scenario_early, root=0)

        # ==================================================================
        #                       INNER  LOOP
        # ==================================================================
        for k in range(self.K):

            # (1) Broadcast lam  ------------------------------------------------
            lambda_y = MPI.COMM_WORLD.bcast(lambda_y, root=0)

            # (2) Local  lam -> mu(s)  mapping (no communication) ----------------
            effective_mu = self._compute_effective_mu(
                lambda_y, scenario_order, all_var_ids)

            # (3) Scenario solves -- FULLY INDEPENDENT -------------------------
            local_results = {}   # name -> (dual_bound, y_vals)
            local_reason  = {}   # name -> str

            for subproblem_name in subproblems.names:
                model = subproblems.model[subproblem_name]

                # Set mu(s) on subproblem Lagrangian Param
                scenario_mu = effective_mu[subproblem_name]
                for vid in model.lg_mu:
                    model.lg_mu[vid].set_value(scenario_mu.get(vid, 0.0))

                if subproblems.relax_binaries:
                    subproblems.relax_all_binaries()
                if subproblems.relax_integers:
                    subproblems.relax_all_integers()

                results = self.opt.solve(model, load_solutions=False, tee=False)

                try:
                    dual_bound, y_vals, reason = self._evaluate_lg_subproblem(
                        results, model, subproblem_name, subproblems)
                except Exception as e:
                    dual_bound, y_vals, reason = None, None, f"exception_{e}"

                local_results[subproblem_name] = (dual_bound, y_vals)
                local_reason[subproblem_name]  = reason

            # (4) Gather to rank 0 (AFTER all local solves) -------------------
            all_local_results = MPI.COMM_WORLD.gather(local_results, root=0)
            all_local_reasons = MPI.COMM_WORLD.gather(local_reason,  root=0)

            node_infeasible     = False
            global_results      = {}
            global_reason       = {}
            scenarios_with_cuts = []
            scenarios_with_primal = []

            if rank == 0:
                for rd in all_local_reasons:
                    global_reason.update(rd)
                for rd in all_local_results:
                    for name, (db, yv) in rd.items():
                        if global_reason.get(name) == "infeasible":
                            node_infeasible = True
                            break
                        global_results[name] = (db, yv)
                        if db is not None and math.isfinite(db):
                            scenarios_with_cuts.append(name)
                        if yv is not None:
                            scenarios_with_primal.append(name)
                    if node_infeasible:
                        break

            node_infeasible       = MPI.COMM_WORLD.bcast(node_infeasible, root=0)
            if node_infeasible:
                node.lb_problem.is_infeasible()
                node.ub_problem.is_infeasible()
                return

            scenarios_with_cuts   = MPI.COMM_WORLD.bcast(scenarios_with_cuts,   root=0)
            scenarios_with_primal = MPI.COMM_WORLD.bcast(scenarios_with_primal, root=0)

            should_break = MPI.COMM_WORLD.bcast(
                (len(global_results) == 0) if rank == 0 else False, root=0)
            if should_break:
                break

            # (5) Rank 0: step-size, update, cuts, diagnostics ----------------
            if rank == 0:
                # ---- zLB_k = Sum p_s * v_s -----------------------------------
                zLB_k = 0.0
                num_valid_dual = 0
                for name in scenarios_with_cuts:
                    db, _ = global_results[name]
                    if db is not None and math.isfinite(db):
                        prob_s = prob_by_scenario_early.get(name, 1.0/num_scenarios)
                        zLB_k += prob_s * db
                        num_valid_dual += 1

                # ---- Alpha-halving (Appendix rule) --------------------------
                alpha_halved = False
                if math.isfinite(zLB_k):
                    if zLB_k <= best_zLB + self.improvement_tol:
                        alpha_kg = max(self.alpha_min, alpha_kg * 0.5)
                        alpha_halved = True
                    else:
                        best_zLB = zLB_k

                # ---- Chain-adjacent denominator -----------------------------
                # g_y[n] = y_n - y_{n+1}  ;  denom = Sum ||g_y[n]||**2
                denom = 0.0
                valid_edge_count = 0
                max_adj_disagree = 0.0

                for n in range(num_edges):
                    s_n   = scenario_order[n]
                    s_np1 = scenario_order[n + 1]
                    if s_n not in global_results or s_np1 not in global_results:
                        continue
                    _, y_n   = global_results[s_n]
                    _, y_np1 = global_results[s_np1]
                    if y_n is None or y_np1 is None:
                        continue

                    edge_sq = 0.0
                    for vid in all_var_ids:
                        d = y_n.get(vid, 0.0) - y_np1.get(vid, 0.0)
                        edge_sq += d * d
                    denom += edge_sq
                    valid_edge_count += 1
                    edge_norm = math.sqrt(edge_sq)
                    if edge_norm > max_adj_disagree:
                        max_adj_disagree = edge_norm

                # ---- Step t^k (Appendix formula) ----------------------------
                DENOM_TOL = 1e-16
                UB_eff = global_UB
                if not math.isfinite(global_UB):
                    if self.initial_ub_estimate is not None:
                        UB_eff = self.initial_ub_estimate
                    else:
                        UB_eff = (best_zLB + 1.0) if math.isfinite(best_zLB) else 1.0
                    if not ub_warning_printed:
                        print(f"WARNING: LG APPX UB_eff={UB_eff:.6f} (tree UB=inf)")
                        ub_warning_printed = True

                gap_term = max(0.0, UB_eff - zLB_k) if math.isfinite(zLB_k) else 1.0

                step_size = (alpha_kg * gap_term / denom) if denom > DENOM_TOL else 0.0

                # ---- lam statistics -------------------------------------------
                all_lam = [ed[vid] for ed in lambda_y for vid in all_var_ids]
                lam_norm = math.sqrt(sum(v*v for v in all_lam)) if all_lam else 0.0
                lam_max  = max((abs(v) for v in all_lam), default=0.0)

                # ---- Diagnostics -------------------------------------------
                h = " (alpha HALVED)" if alpha_halved else ""
                ub_label = ("tree" if math.isfinite(global_UB)
                            else ("estimate" if self.initial_ub_estimate else "fallback"))
                print(f"LG APPX [k={k}]: alpha={alpha_kg:.6f}, "
                      f"zLB_k={zLB_k:.6f}, zLB_best={best_zLB:.6f}, "
                      f"UB_eff={UB_eff:.6f}[{ub_label}], gap={gap_term:.6f}, "
                      f"denom={denom:.6f}, step={step_size:.6f}, "
                      f"valid_dual={num_valid_dual}/{num_scenarios}, "
                      f"edges={valid_edge_count}/{num_edges}{h}")
                print(f"LG APPX [k={k}]: ||lam||={lam_norm:.4f}, "
                      f"max|lam|={lam_max:.4f}, "
                      f"max_adj_disagree={max_adj_disagree:.6f}")

                # ---- Generate cuts  eta_s >= v_val + mu_s^T y ------------------
                num_cuts_added = 0
                for name in scenarios_with_cuts:
                    db, _ = global_results[name]
                    # Store effective mu(s) with the cut (NOT raw lam)
                    new_cuts.append((name, cp.deepcopy(effective_mu[name]), db))
                    num_cuts_added += 1
                print(f"LG APPX [k={k}]: cuts_added={num_cuts_added}")

                if hasattr(self, 'cut_pool') and self.cut_pool is not None:
                    self.cut_pool.add_cuts_from_iteration(
                        cuts_data=[
                            (name, cp.deepcopy(effective_mu[name]),
                             global_results[name][0])
                            for name in scenarios_with_cuts],
                        iteration=k, node_id=node.id,
                        y_bounds=node_var_bounds)

                # ---- Appendix lam update  ------------------------------------
                # lam_y[n][vid] += t * (y_n[vid] - y_{n+1}[vid])
                if step_size > 0.0:
                    for n in range(num_edges):
                        s_n   = scenario_order[n]
                        s_np1 = scenario_order[n + 1]
                        if s_n not in global_results or s_np1 not in global_results:
                            continue
                        _, y_n   = global_results[s_n]
                        _, y_np1 = global_results[s_np1]
                        if y_n is None or y_np1 is None:
                            continue
                        for vid in all_var_ids:
                            d = y_n.get(vid, 0.0) - y_np1.get(vid, 0.0)
                            lambda_y[n][vid] += step_size * d

        # ==================================================================
        #            END INNER LOOP  --  Solve RMP on Rank 0
        # ==================================================================
        rmp_obj = None
        rmp_feasible = False
        use_conservative_lb = False

        # Gather probabilities (all ranks, avoid MPI deadlock)
        local_probs = {name: subproblems.probability[name]
                       for name in subproblems.names}
        all_probs_list = MPI.COMM_WORLD.gather(local_probs, root=0)

        prob_by_scenario = {}
        if rank == 0:
            for d in all_probs_list:
                prob_by_scenario.update(d)
            missing = [s for s in all_scenario_names if s not in prob_by_scenario]
            assert not missing, f"Missing probabilities: {missing}"
            assert abs(sum(prob_by_scenario.values()) - 1.0) < 1e-6

        if rank == 0:
            all_cuts = list(new_cuts)
            if hasattr(self, 'cut_pool') and self.cut_pool is not None:
                pool_cuts = self.cut_pool.get_cuts_for_rmp(
                    node_var_bounds, all_scenario_names)
                all_cuts.extend(pool_cuts)
                print(f"DEBUG RMP: Node {node.id}, new={len(new_cuts)}, "
                      f"pool={len(pool_cuts)}, total={len(all_cuts)}")

            cut_count = {s: 0 for s in all_scenario_names}
            for (s_name, _, _) in all_cuts:
                cut_count[s_name] += 1

            scenarios_no_cuts = [s for s in all_scenario_names
                                 if cut_count[s] == 0]

            if scenarios_no_cuts:
                use_conservative_lb = True
                conservative_lb = float('-inf')
                pn = getattr(node, 'parent', None)
                if pn is not None:
                    plb = getattr(pn, 'lb_problem', None)
                    if plb is not None:
                        po = getattr(plb, 'objective', None)
                        if po is not None and math.isfinite(po):
                            conservative_lb = po
                rmp_obj = conservative_lb
                rmp_feasible = True
                print(f"DEBUG RMP: Node {node.id} - no cuts for "
                      f"{scenarios_no_cuts}, conservative LB={conservative_lb}")
            else:
                rmp_feasible, rmp_obj = self._solve_rmp(
                    node, subproblems, all_cuts,
                    all_var_ids, all_scenario_names, prob_by_scenario)
                print(f"DEBUG RMP: Node {node.id}, obj={rmp_obj}")

        rmp_info = (rmp_feasible, rmp_obj, use_conservative_lb)
        rmp_feasible, rmp_obj, use_conservative_lb = \
            MPI.COMM_WORLD.bcast(rmp_info, root=0)

        if rmp_feasible:
            if rank == 0:
                statistics.aggregated_objective = rmp_obj
            else:
                statistics.aggregated_objective = 0.0
            node.lb_problem.is_feasible(statistics)

            # Persist chain-edge lam on node for child propagation
            stored_lambda = {"lambda_y": lambda_y}
            node.lg_multipliers = MPI.COMM_WORLD.bcast(stored_lambda, root=0)
        else:
            node.lb_problem.is_infeasible()
            node.ub_problem.is_infeasible()


    def _solve_rmp(self, node, subproblems, cuts,
                   all_var_ids, all_scenario_names, prob_by_scenario):
        """Build and solve the RMP on Rank 0.  Returns (feasible, obj)."""
        m = pyo.ConcreteModel()

        m.y   = pyo.Var(all_var_ids,        domain=pyo.Reals)
        m.eta = pyo.Var(all_scenario_names,  domain=pyo.Reals)

        # y-bounds from node state
        for vid in all_var_ids:
            var_obj = subproblems.id_to_vars[vid][0]
            vt, _, _ = subproblems.var_to_data[var_obj]
            m.y[vid].setlb(node.state[vt][vid].lb)
            m.y[vid].setub(node.state[vt][vid].ub)

        m.obj = pyo.Objective(
            expr=sum(prob_by_scenario[s] * m.eta[s]
                     for s in all_scenario_names),
            sense=pyo.minimize)

        m.cuts = pyo.ConstraintList()
        for (s_name, mu_vec, v_val) in cuts:
            term = sum(mu_vec[v] * m.y[v] for v in all_var_ids)
            m.cuts.add(m.eta[s_name] >= v_val + term)

        res = self.opt.solve(m, tee=False, load_solutions=False)

        if res.solver.termination_condition in [
                TerminationCondition.optimal,
                TerminationCondition.globallyOptimal]:
            try:
                m.solutions.load_from(res)
            except:
                return True, pyo.value(m.obj)
            obj_val = pyo.value(m.obj)

            # Compact RMP diagnostic
            n_at_bound = sum(
                1 for vid in all_var_ids
                if min((pyo.value(m.y[vid]) - (m.y[vid].lb or -1e30)),
                       ((m.y[vid].ub or 1e30) - pyo.value(m.y[vid]))) <= 1e-6)
            print(f"RMP DIAG: y_at_bound={n_at_bound}/{len(all_var_ids)}, "
                  f"obj={obj_val:.6f}")
            return True, obj_val
        else:
            return False, None

    def solve_a_subproblem(self, *args, **kwargs):
        raise NotImplementedError("Use solve() directly for LG.")
