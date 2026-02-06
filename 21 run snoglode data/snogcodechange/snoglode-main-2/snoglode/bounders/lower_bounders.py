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
            return True, pyo.value(get_active_objective(subproblem_model))
        
        # if the solution is STOPPED SHORT:, we can retrieve the lb solution & return that / parent
        if results.solver.termination_condition in [TerminationCondition.maxTimeLimit, 
                                                    TerminationCondition.maxIterations,
                                                    TerminationCondition.maxEvaluations,
                                                    TerminationCondition.feasible]:
            subproblem_lb = self.retrieve_solver_lb(results)            # returns -inf if not retrievable
            parent_obj = pyo.value(subproblem_model.successor_obj)      # returns -inf if parent did not have a solution
            return True, max(subproblem_lb, parent_obj)
        
        # if the solution is not feasible, return None
        elif results.solver.termination_condition == TerminationCondition.infeasible:
            return False, None

        else:
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
        if "gurobi" in self.solver:
            try: # retrieve via pyomo's default interface
                return results.problem.lower_bound
            except:
                pass
            try: # retrieve via another method
                return results.problem[0]["Lower bound"]
            except:
                pass
            try: # access gurobipy model, compute gap manually
                gurobi_model = self.solver._solver_model
                lb = gurobi_model.ObjBound
                ub = gurobi_model.ObjVal
                return (ub - lb) / max(1.0, abs(ub))
            except:
                return float("-inf")
        
        elif "baron" == self.solver:
            try:
                return results.problem.lower_bound
            except:
                return float("-inf")

            
        else:
            raise RuntimeError(f"Attemping to retrieve a lower bound for a subproblem solve, but did not anticipate solver = {self.solver.name}")


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