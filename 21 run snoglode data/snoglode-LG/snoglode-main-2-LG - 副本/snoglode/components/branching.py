"""
Tailored branching strategies for the tree.

This was written intentionally to be abstract such that
the user can easily define their own or plug in their choices.

There are two general architectures:
    (1) variable selection
    (2) paritioning
"""
import numpy as np
np.random.seed(17)
import math
import random
random.seed(42)
from typing import Tuple

import pyomo.environ as pyo
from pyomo.opt import TerminationCondition, SolverStatus
from pyomo.contrib.alternative_solutions.aos_utils import get_active_objective

from snoglode.components.node import Node, NodeDirection
from snoglode.components.subproblems import Subproblems
from snoglode.utils.supported import SupportedVars
import snoglode.utils.compute as compute

import snoglode.utils.MPI as MPI
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()


class SelectionStrategy():
    def __init__(self, *args, **kwargs) -> None:
        pass

    def select_variable(self, 
                        node: Node, 
                        subproblems: Subproblems) -> Tuple[SupportedVars, str]:
        """
        returns a complicating var name to branch on & var_type
        this is a derived function of a child class.
        
        Parameters
        -------------
        node : Node
            Current node in the spatial BnB tree.
        subproblems : Subproblems
            Subproblems objective containing this ranks subproblems.

        Returns
        --------------
        var_type : str
            type of variable - should be in SupportedVars
        var_name : str
            ID corresponding to the complicating variable that was selected.
        """
        raise NotImplementedError('should be implemented by derived classes')
    

class PartitionStrategy():

    def __init__(self, *args, **kwargs) -> None:
        pass

    def split_point(self, 
                    varID: str,
                    var_lb: float, 
                    var_ub: float,
                    node: Node,
                    subproblems: Subproblems) -> float:
        """
        given a variable UB / LB, determine a split point.

        NOTE: this is not called if the selected variable is binary.

        Parameters
        -------------
        varID : str
            ID corresponding to the complicating variable that was selected.
        var_lb : float
            current lowerbound of the variables.
        var_ub : float
            current upperbound of the variables.
        node : Node
            current node in the BnB tree
        subproblems : Subproblems
            Subproblems objective containing this ranks subproblems.

        Returns
        --------------
        split_point : float
            point at which to split the continuous domain
            var_lb <= split_point <= var_ub
        """
        raise NotImplementedError('should be implemented by derived classes')
    

    def new_bounds(self, 
                   var_name: str, 
                   var_type: str, 
                   node: Node,
                   subproblems: Subproblems) -> Tuple[float, float, float]:
        """
        Uses the abstract split_point() function and the current variable
        to return the LB / UB / split of the variable

        Parameters
        --------------
        var_type : str
            type of variable - should be in SupportedVars
        var_name : str
            ID corresponding to the complicating variable that was selected.
        node : Node
            Current node in the spatial BnB tree.

        Returns
        --------------
        var_lb : float
            current lowerbound of the variables.
        split_point : float
            point at which to split the continuous domain
            var_lb <= split_point <= var_ub
        var_ub : float
            current upperbound of the variables.
        """
        # get ub / lb from model
        var_ub = node.state[var_type][var_name].ub
        var_lb = node.state[var_type][var_name].lb

        # determine split point
        split_point = self.split_point(varID = var_name,
                                       var_lb = var_lb,
                                       var_ub = var_ub, 
                                       node = node, 
                                       subproblems=subproblems)

        left_child_lb = var_lb
        left_child_ub = split_point
        right_child_lb = left_child_ub
        right_child_ub = var_ub

        # arithmetic checks
        assert left_child_lb <= left_child_ub
        assert right_child_lb <= right_child_ub
        assert left_child_ub == right_child_lb

        # ensure branching point keeps a min. distance from var. bounds
        return left_child_lb, left_child_ub, right_child_ub


# =================== SELECTION STRATEGIES ================================= #

class RandomSelection(SelectionStrategy):
    """
    Randomly selects a variable from all those available
    No descriminations made.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()
    
    def name(self):
        return "RandomSelection"

    def select_variable(self, 
                        node: Node, 
                        subproblems: Subproblems) -> Tuple[SupportedVars, str]:
        """
        randomly select any variable.

        Parameters
        -------------
        node : Node
            Current node in the spatial BnB tree.
        subproblems : Subproblems
            Subproblems objective containing this ranks subproblems.
        
        Returns
        --------------
        var_type : str
            type of variable - should be in SupportedVars
        var_name : str
            ID corresponding to the complicating variable that was selected.
        """
        # get lists of all variables left to branch on 
        # NOTE: sorting is very important for coordination over ranks 
        binaries = sorted(list(node.to_branch[SupportedVars.binary]))
        reals = sorted(list(node.to_branch[SupportedVars.reals]))
        integers = sorted(list(node.to_branch[SupportedVars.integers]))
        nonnegative_integers = sorted(list(node.to_branch[SupportedVars.nonnegative_integers]))
        vars = binaries + reals + integers + nonnegative_integers
        
        # randomly decide
        index = MPI.COMM_WORLD.bcast(np.random.randint(0, len(vars)), root=0)
        var_name = vars[index]

        # determine which var type we selected based on index value
        if binaries and (index in range(0, len(binaries))): 
            var_type = SupportedVars.binary
        elif reals and (index in range(len(binaries), (len(binaries) + len(reals)))):
            var_type = SupportedVars.reals
        elif integers and (index in range((len(binaries) + len(reals)), (len(binaries) + len(reals) + len(integers)))):
            var_type = SupportedVars.integers
        elif nonnegative_integers and (index in range((len(binaries) + len(reals) + len(integers)), len(vars))): 
            var_type = SupportedVars.nonnegative_integers

        return var_type, var_name
    

class MostInfeasibleBinary(SelectionStrategy):
    """
    Selects the binary variable that is the most violated;
    
    Once there are no binaries left, randomly selects
    the variables with continuous domain.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def name(self):
        return "MostInfeasibleBinary"
    
    def select_variable(self, 
                        node: Node, 
                        subproblems: Subproblems) -> Tuple[SupportedVars, str]:
        """
        prioritize binaries - select most infeasible.

        Parameters
        -------------
        node : Node
            Current node in the spatial BnB tree.
        subproblems : Subproblems
            Subproblems objective containing this ranks subproblems.
        
        Returns
        --------------
        var_type : str
            type of variable - should be in SupportedVars
        var_name : str
            ID corresponding to the complicating variable that was selected.
        """
        # get lists of all variables left to branch on 
        # NOTE: sorting is very important for coordination over ranks 
        binaries = sorted(list(node.to_branch[SupportedVars.binary]))
        
        # prioritize binary branching first
        if (binaries):

            # get the averaged solution across all of the subrpoblems
            averaged_solution = compute.average_lb_solution(node = node,
                                                            subproblems = subproblems,
                                                            round_binaries = False)
            
            # check that we have solutions to actually compute things with
            has_none_value = any(value is None for value in averaged_solution.values())

            # select the binary variable that is closest to 0.5 (i.e. "most infeasible")
            if not has_none_value:
                var_name = None
                var_distance = 1
                for var_id in binaries:

                    # compute how far away it is from 0.5
                    distance = abs(averaged_solution[var_id] - 0.5)

                    # want the closest to 0.5 as possible (i.e. smallest distance)
                    if distance < var_distance:
                        var_name = var_id
                        var_distance = distance
            
                return SupportedVars.binary, var_name
            
        # if we have run out of binaries, move on to branching the rest
        else: 
            # randomly decide on any other var
            reals = sorted(list(node.to_branch[SupportedVars.reals]))
            integers = sorted(list(node.to_branch[SupportedVars.integers]))
            nonnegative_integers = sorted(list(node.to_branch[SupportedVars.nonnegative_integers]))
            vars = reals + integers + nonnegative_integers

            index = MPI.COMM_WORLD.bcast(np.random.randint(0, len(vars)), root=0)
            var_name = reals[index]

            # determine which var type we selected based on index value
            if reals and (index in range(0, len(reals))):
                var_type = SupportedVars.reals
            elif integers and (index <= range(len(reals), (len(reals) + len(integers)))):
                var_type = SupportedVars.integers
            if nonnegative_integers and (index <= range((len(reals) + len(integers)), len(vars))): 
                var_type = SupportedVars.nonnegative_integers

            return var_type, var_name


class MaximumDisagreement(SelectionStrategy):
    """
    Selects the variable that is the most violated based on
    the highest variance across subproblem solutions (applies to cont/int/binary)
    
    In this case, all domains can be considered because we will normalize.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

    def name(self):
        return "MaximumDisagreement"
    
    def select_variable(self, 
                        node: Node, 
                        subproblems: Subproblems) -> Tuple[SupportedVars, str]:
        """
        Computes the variance of all variables (standardizing,
        when they are within a continuous/integer domain)

        Selects the variable with the maximum variance.

        Parameters
        -------------
        node : Node
            Current node in the spatial BnB tree.
        subproblems : Subproblems
            Subproblems objective containing this ranks subproblems.
        
        Returns
        --------------
        var_type : str
            type of variable - should be in SupportedVars
        var_name : str
            ID corresponding to the complicating variable that was selected.
        """
        # compute the variance across all subproblem LB solutions
        variance = compute.variance_lb_solution(node = node,
                                                subproblems = subproblems,
                                                normalize = True)
        
        # determine which (normalized) variance is the largest
        varID = max(variance, 
                    key = variance.get)
        
        for var_type in SupportedVars:
            if varID in node.state[var_type]: break
        
        return var_type, varID


class Pseudocost(SelectionStrategy):
    """
    Compute the pseudocost of branching on this variable, based
    on the progression of the tree / behavior of the variable in previous
    solutions.

    see: Achterberg, Tobias, Thorsten Koch, and Alexander Martin. 
        "Branching rules revisited." Operations Research Letters 33.1 (2005): 42-54.
    """
    def __init__(self,
                 id_to_var: dict,
                 var_to_data, # pyo.ComponentMap
                 *args, **kwargs) -> None:
        """
        Pseudocost branching strategy initialization (slightly more involved
        than other branching strategies).

        Parameters
        --------------
        id_to_var : dict
            mapping from complicating var ID to (pyomo var, var_type, list of subproblem IDs)
        var_to_data : pyo.ComponentMap
            mapping from pyomo var to (var_type, complicating var ID, list of subproblem IDs)
        """
        super().__init__()

        # can update this dynamically because it depends on full tree
        # (iow, do not have to recompute everything* at each node)
        self.pseudocost = {varID: {NodeDirection.upward:   0,
                                   NodeDirection.downward: 0} 
                                        for varID in id_to_var.keys()}
        self.explored   = {varID: {NodeDirection.upward:   0,
                                   NodeDirection.downward: 0} 
                                        for varID in id_to_var.keys()}
        
        # score parameter [0,1]
        self.mu = 1/6

        # unitialized pseudocost default score
        self.average_pseudocost = 1
        self.num_vars = len(id_to_var.keys())

        # keeps track of the scores / types for each of the variables
        self.scores = {}
        self.var_types = {}
        for varID in id_to_var.keys():
            self.scores[varID] = self.average_pseudocost
            var = id_to_var[varID][0] # just grab the first one - only need var_type
            var_type, _, _ = var_to_data[var]
            self.var_types[varID] = var_type
        
        # initialized indicates if we have solved both directions at least once for this var
        self.intialized = {varID: False for varID in id_to_var.keys()}

        # scoring function also relies on most recent variable change
        self.most_recent_var_delta = {varID: {NodeDirection.upward:   0,
                                              NodeDirection.downward: 0} 
                                                for varID in id_to_var.keys()}

    def name(self):
        return "Pseudocost"

    def update_data(self, 
                    node: Node,
                    subproblems: Subproblems) -> None:
        """
        given the current node / solutions,
        update the solved_nodes data and
        the branching_on_var_at_node data.

        Parameters
        -------------
        node : Node
            Current node in the spatial BnB tree.
        subproblems : Subproblems
            Subproblems objective containing this ranks subproblems.
        """
        # only possible if we have a feasible problem
        if not node.lb_problem.feasible: return

        # compute the new average at this node (across all subproblems/ranks)
        node_var_avg = compute.average_var_lb_solution(node = node,
                                                       subproblems = subproblems,
                                                       var_ID = node.branched_on)
        
        # compute the delta for this variable & normalize
        var_avg_delta = abs(node.var_delta - node_var_avg)
        normalized_var_avg_delta = var_avg_delta / (node.parent_var_ub - node.parent_var_lb)
        if var_avg_delta == 0: var_avg_delta = 1 # avoid division by zero...
        if normalized_var_avg_delta == 0: normalized_var_avg_delta = 1 # avoid division by zero...

        # update most recent var delta (needed for computing score)
        # self.most_recent_var_delta[node.branched_on] = var_avg_delta
        self.most_recent_var_delta[node.branched_on] = normalized_var_avg_delta
        
        # compute the change in objective (parent obj <= child by def); avoid numerical issues
        obj_change = max(node.lb_problem.objective - node.parent_obj, 0)
        assert obj_change >= 0, \
            f"obj_change = {obj_change}, node.lb_problem.objective = {node.lb_problem.objective}, node.parent_obj = {node.parent_obj}"
        
        # compute pseudocost of this node
        # pseudocost = obj_change / var_avg_delta
        pseudocost = obj_change / normalized_var_avg_delta

        # update the solution for this node
        self.pseudocost[node.branched_on][node.dir] += pseudocost
        self.explored[node.branched_on][node.dir]   += 1

        # update the score
        self.update_scores(node.branched_on)


    def update_scores(self, 
                      varID: str) -> None:
        """
        Computes the score of the variable, based
        on the current upward and downward direction.

        Parameters
        -------------
        varID : str
            ID corresponding to the complicating variable that was selected.
        """

        # if we have initialized costs in both directions, we can compute a score
        if self.explored[varID][NodeDirection.upward] > 0.0 \
                and self.explored[varID][NodeDirection.downward] > 0.0:
            
            # reset flag (not always needed, but let's be careful anyways)
            self.intialized[varID] = True

            # compute the scores of both directions
            upward_score = self.pseudocost[varID][NodeDirection.upward] * self.most_recent_var_delta[varID] \
                                / self.explored[varID][NodeDirection.upward]
            downward_score = self.pseudocost[varID][NodeDirection.downward] * self.most_recent_var_delta[varID] \
                                / self.explored[varID][NodeDirection.downward]
            
            # store old score & change average
            prev_score = self.scores[varID]
            self.scores[varID] = \
                (1 - self.mu) * min(upward_score, downward_score) + \
                    self.mu * max(upward_score, downward_score)

            # update new average and update uninitialized scores 
            self.update_avg_pseudocosts(varID, prev_score)

    def update_avg_pseudocosts(self,
                               varID: str,
                               prev_score: float) -> None:
        """
        When we do not have an initial score,
        take the average psuedocost score and set that
        as the score for all of the variables.

        Parameters
        -------------
        varID : str
            ID corresponding to the complicating variable that was selected.
        prev_score : float
            previous score for this variable.
        """
        # recompute the average
        self.average_pseudocost = ((self.average_pseudocost * self.num_vars) \
            - prev_score + self.scores[varID]) / self.num_vars

        # update scores for uninitialized vars, if we still have some
        for varID in self.scores:
            if self.intialized[varID] == False: 
                self.scores[varID] = self.average_pseudocost

    def best_scoring_var(self,
                         node: Node) -> Tuple[SupportedVars, str]:
        """
        returns which variables is currently scored as the hightest.

        Paremeters
        -----------
        node : Node
            current node of the BnB tree
        
        Returns
        -----------
        var_type : str
            type of variable - should be in SupportedVars
        var_name : str
            ID corresponding to the complicating variable that was selected.
        """
        best_var = ""
        best_score = float("-inf")
        tied = []

        # check each of the scores
        for varID in self.scores:
            var_type = self.var_types[varID]

            if self.scores[varID] > best_score \
                    and varID in node.to_branch[var_type]:
                
                # update best scores / var
                best_score = self.scores[varID]
                best_var = varID
                
                # if we have updated the best score, reset tied list
                tied = [varID]

            # if it is the same, update tied variables
            if self.scores[varID] == best_score \
                    and varID in node.to_branch[var_type]:
                tied.append(varID)
        
        # if we are tied, return a random selection
        if tied: 

            # randomly decide between binary or continuous
            index = MPI.COMM_WORLD.bcast(np.random.randint(0, len(tied)), root=0)
            best_var = tied[index]
            var_type = self.var_types[best_var]

        # otw, return best variable
        return var_type, best_var

    def select_variable(self, 
                        node: Node, 
                        subproblems: Subproblems) -> Tuple[SupportedVars, str]:
        """
        computes the pseduocosts based on previous
        branching results & selects variable in this manner.

        Parameters
        -------------
        node : Node
            Current node in the spatial BnB tree.
        subproblems : Subproblems
            Subproblems objective containing this ranks subproblems.

        Returns
        --------------
        var_type : str
            type of variable - should be in SupportedVars
        var_name : str
            ID corresponding to the complicating variable that was selected.
        """
        # first, update the current data (if we are not the root & feas)
        if node.dir != NodeDirection.root and node.lb_problem.feasible:
            self.update_data(node = node,
                             subproblems = subproblems)
    
        # select the variable with the highest pseudocost
        return self.best_scoring_var(node)


class HybridBranching(Pseudocost):
    """
    Pseudocost branching is highly effective after the algorithm
    has progressed enough to have accumulated some data and generated
    "good" pseudocost estimates.

    Hybrid branching uses a different strategy (MaximumDisagreement)
    for the first several iterations, and then switches to pseudocost
    branching afterwards (i..e. once we have some data to work with).
    """
    def __init__(self, 
                 hybrid_switch_iter: int,
                 hybrid_starting_method: SelectionStrategy,
                 *args, **kwargs) -> None:
        """
        Hybrid branching strategy initialization.

        Parameters
        --------------
        hybrid_switch_iter : int
            number of iterations to use the starting method before switching
            to pseudocost branching.
        hybrid_starting_method : SelectionStrategy
            branching strategy to use for the first hybrid_switch_iter iterations.
        """
        super().__init__(*args, **kwargs)
        assert type(hybrid_switch_iter) == int and hybrid_switch_iter >= 0, \
            "hybrid_switch_iter must be a non-negative integer."
        self.method_switch_iter = hybrid_switch_iter
        self.iter = 0

        assert issubclass(hybrid_starting_method, SelectionStrategy), \
            "Hybrid starting strategy should be a derived class of SelectionStrategy."
        assert type(hybrid_starting_method) is not HybridBranching, \
            "Can't exactly use HybridBranching as the start of HybridBranching... that's not quite the point, is it?\n" +\
            " -> hint: just specify the branching strategy you want to use the whole time, avoiding hybrid altogether."
        assert type(hybrid_starting_method) is not Pseudocost, \
            "Can't exactly use Pseudocost as the start of HybridBranching that reverts to Pseudocost... that's not quite the point, is it?\n" +\
            " -> hint: just use Pseudocost branching directly."
        self.first_method = hybrid_starting_method()
    
    def name(self):
        return "HybridBranching"

    def select_variable(self, 
                        node: Node, 
                        subproblems: Subproblems) -> Tuple[SupportedVars, str]:
        """
        computes the pseduocosts based on previous
        branching results & selects variable in this manner.

        Parameters
        -------------
        node : Node
            Current node in the spatial BnB tree.
        subproblems : Subproblems
            Subproblems objective containing this ranks subproblems.

        Returns
        --------------
        var_type : str
            type of variable - should be in SupportedVars
        var_name : str
            ID corresponding to the complicating variable that was selected.
        """
        self.iter += 1

        # first, update the current data (if we are not the root & feas)
        if node.dir != NodeDirection.root and node.lb_problem.feasible:
            self.update_data(node = node,
                             subproblems = subproblems)
    
        # in the beggining, default to MaximumDisagreement
        if self.iter <= self.method_switch_iter:
            return self.first_method.select_variable(node = node,
                                                     subproblems = subproblems)
        else:
            return self.best_scoring_var(node)


class StrongBranching(SelectionStrategy):
    """
    The idea behind strong branching is to test progress for all
    possible variables before selecting one to branch on.  This is done
    by temporarily branching on each variable in both directions,
    solving the resulting lower bound problems (a linear relaxation), and then selecting
    the variable that results in the best improvement in the lower bound.
    
    This is very expensive, but can be very effective in reducing the
    size of the tree. To mitigate this, we limit the number of simplex iterations
    and can perform this on a subset of the branching variables, rather than all of them.

    see: Achterberg, Tobias, Thorsten Koch, and Alexander Martin. 
    "Branching rules revisited." Operations Research Letters 33.1 (2005): 42-54.
    """
    def __init__(self, 
                 max_simplex_iterations: int,
                 subset_branching_variables: int,
                 *args, **kwargs) -> None:
        """
        Strong branching strategy initialization.

        Parameters
        --------------
        max_simplex_iterations : int
            maximum number of simplex iterations to allow per subproblem
            when performing strong branching.
        subset_branching_variables : int or None
            number of variables to randomly select from the current
            set of branching variables to perform strong branching on.
            If None, strong branching is performed on all variables.
        """
        super().__init__(*args, **kwargs)
        
        self.opt = pyo.SolverFactory('gurobi')
        assert self.opt.available(), \
            "To use FullStrongBranching, you must have Gurobi installed."
        
        # limit the number of simplex iterations
        self.opt.options['Method'] = 0     # 0 = Primal Simplex, 1 = Dual Simplex
        self.opt.options['IterationLimit'] = max_simplex_iterations  # limits number of iterations for simplex

        assert (type(subset_branching_variables) == int and subset_branching_variables > 0) or \
                    (subset_branching_variables is None), print("subset_branching_variables must be a positive integer or None.")
        self.subset_branching_variables = subset_branching_variables

        # for the first iter, need to check that we have a fully linear relaxation
        self.first_iter = True

    def name(self):
        return "StrongBranching"
    
    def get_vars_to_evaluate(self, node: Node) -> list:
        """
        Based on how many variables we were indicated to evaluate,
        randomly select a subset of the current set of branching variables
        that we will perform strong branching on this iteration.

        Parameters
        -------------
        node : Node
            Current node.
        
        Returns
        --------------
        vars_to_evaluate : list
            list of varID's corresponding to the variables we should check
            on this iteration of strong branching.
        """
        # check that number of variables to branch on is not too large
        total_branching_vars =  len(node.to_branch[SupportedVars.binary]) + \
                                len(node.to_branch[SupportedVars.integers]) + \
                                len(node.to_branch[SupportedVars.nonnegative_integers]) + \
                                len(node.to_branch[SupportedVars.reals])

        # if not specified, default to using all
        if self.subset_branching_variables is None: self.subset_branching_variables = total_branching_vars
        # if specified and too large, adjust (this could happen later in the tree, when binaries/integers are eliminated)
        elif self.subset_branching_variables > total_branching_vars:
            self.subset_branching_variables = total_branching_vars

        # select set of variables to evaluate
        binaries = sorted(list(node.to_branch[SupportedVars.binary]))        
        integers = sorted(list(node.to_branch[SupportedVars.integers]))
        nonnegative_integers = sorted(list(node.to_branch[SupportedVars.nonnegative_integers]))
        reals = sorted(list(node.to_branch[SupportedVars.reals]))
        all_vars = binaries + integers + nonnegative_integers + reals
        
        if self.subset_branching_variables < total_branching_vars:
            return random.sample(all_vars, self.subset_branching_variables)
        else: return all_vars

    def is_linear_model(self, model: pyo.ConcreteModel) -> bool:
        """
        Checks that a Pyomo model is linear.
        FullStrongBranching only works for linear relaxations.
        Checks *after* relaxing integrality constraints.

        Parameters
        -------------
        model : pyo.ConcreteModel
            Pyomo model to check.   
        
        Returns
        --------------
        is_linear : bool
            True if the model is linear, False otherwise.
        """
        for obj in model.component_objects(pyo.Objective, active=True):
            if obj.expr.polynomial_degree() not in (0, 1):
                return False
        for c in model.component_objects(pyo.Constraint, active=True):
            for idx in c:
                expr = c[idx].body
                if expr.polynomial_degree() not in (0, 1):
                    return False
        return True

    def get_var_type(self, 
                     varID: str,
                     node: Node) -> SupportedVars:
        """
        Determines the variable type.

        Parameters
        -----------
        varID : str
            var we plan to pseudo branch on.
        node : Node
            current node in the tree.
        """
        if varID in node.to_branch[SupportedVars.binary]:
            return SupportedVars.binary
        elif varID in node.to_branch[SupportedVars.integers]:
            return SupportedVars.integers
        elif varID in node.to_branch[SupportedVars.nonnegative_integers]:
            return SupportedVars.nonnegative_integers
        elif varID in node.to_branch[SupportedVars.reals]:
            return SupportedVars.reals
        else:
            raise TypeError(f"Variable ID does not correspond to a proper type.")

    def branch_var(self,
                   varID: str,
                   var_type: SupportedVars,
                   subproblems: Subproblems,
                   node: Node,
                   direction: str) -> None:
        """
        Temporarily branches a variable, based on it's domain, 
        up or down across all subproblems.

        Parameters
        -----------
        varID : str
            var we plan to pseudo branch on.
        var_type : SupportedVars
            type of varID.
        subproblems : Subproblems
            initialized Subproblems object.
        node : Node
            current node in the tree.
        direction : str
            branch "up" or "down"?
        """
        # binary branching - fix to 0 or 1
        if var_type is SupportedVars.binary:
            vars = subproblems.id_to_vars[varID]
            if direction=="up": 
                for var in vars: var.fix(1)
            if direction=="down": 
                for var in vars: var.fix(0)
        
        # integer branching - fix to floor(average) or cieling(average)
        elif var_type in [SupportedVars.integers, SupportedVars.nonnegative_integers]:
            avg_lb_solution = compute.average_var_lb_solution(node = node,
                                                              subproblems = subproblems,
                                                              var_ID = varID)
            vars = subproblems.id_to_vars[varID]
            integer_var_lb = node.state[var_type][varID].lb
            integer_var_ub = node.state[var_type][varID].ub
            assert integer_var_lb <= avg_lb_solution <= integer_var_ub, \
                "Problem: the average solution should lie between the current LB and UB.\n" + \
                        f"  LB = {integer_var_lb}, UB = {integer_var_ub}, avg = {avg_lb_solution}"
            
            if direction=="up": 
                for var in vars: var.fix(math.ceil(avg_lb_solution))
            if direction=="down":
                for var in vars: var.fix(math.floor(avg_lb_solution))
        
        # continuous branching - update LB/UB to the average
        elif var_type is SupportedVars.reals: 
            avg_lb_solution = compute.average_var_lb_solution(node = node,
                                                                subproblems = subproblems,
                                                                var_ID = varID)
            vars = subproblems.id_to_vars[varID]

            continuous_var_lb = node.state[SupportedVars.reals][varID].lb
            continuous_var_ub = node.state[SupportedVars.reals][varID].ub
            assert continuous_var_lb <= avg_lb_solution <= continuous_var_ub, \
                "Problem: the average solution should lie between the current LB and UB.\n" + \
                        f"  LB = {continuous_var_lb}, UB = {continuous_var_ub}, avg = {avg_lb_solution}"
            
            if direction=="up":
                for var in vars: 
                    var.setlb(avg_lb_solution)
                    var.setub(continuous_var_ub)
            if direction=="down":
                for var in vars: 
                    var.setlb(continuous_var_lb)
                    var.setub(avg_lb_solution)

    def reset_var(self,
                  varID: str,
                  var_type: SupportedVars,
                  subproblems: Subproblems,
                  node: Node) -> None:
        """
        After we have finished our pseudo branching, reset.

        Parameters
        -----------
        varID : str
            var we plan to pseudo branch on.
        var_type : SupportedVars
            type of varID.
        subproblems : Subproblems
            initialized Subproblems object.
        node : Node
            current node in the tree.
        """
        vars = subproblems.id_to_vars[varID]
        if var_type is SupportedVars.binary:
            for var in vars: 
                var.unfix()
                var.setlb(0)
                var.setub(1)
        
        elif var_type in [SupportedVars.integers, SupportedVars.nonnegative_integers]:
            for var in vars: 
                var.unfix()
                var.setlb(node.state[var_type][varID].lb)
                var.setub(node.state[var_type][varID].ub)

        elif var_type is SupportedVars.reals:
            for var in vars: 
                var.setlb(node.state[SupportedVars.reals][varID].lb)
                var.setub(node.state[SupportedVars.reals][varID].ub)

    def solve_a_subproblem(self,
                           subproblem_model: pyo.ConcreteModel) -> Tuple[bool, float]:
        """
        Takes the relaxed model, solves it, and returns
        whether it was feasible and the objective value.

        Parameters
        -------------
        subproblem_model : pyo.ConcreteModel
            Pyomo model corresponding to the subproblem.

        Returns
        --------------
        feasible : bool
            whether the subproblem was feasible.
        objective : float
            objective value of the subproblem (if feasible),
            -inf otherwise.
        """
        # solve model
        results = self.opt.solve(subproblem_model,
                                 load_solutions = False, 
                                 symbolic_solver_labels = True,
                                 tee = False)

        # if the solution is optimal, return objective value
        if results.solver.termination_condition==TerminationCondition.optimal and \
                    results.solver.status==SolverStatus.ok:

            # load in solutions, return [feasibility = True, obj, results]
            subproblem_model.solutions.load_from(results)
            return True, pyo.value(get_active_objective(subproblem_model))
        
        # if we reached the max # of primal simplex iterations, check for feasible solutions
        elif results.solver.termination_condition==TerminationCondition.maxIterations:

            # if we can load solutions, return [feasibility = True, obj, results]
            try: 
                subproblem_model.solutions.load_from(results)
                return True, pyo.value(get_active_objective(subproblem_model))
            
            # if we cannot load solutions, did not find a feasible solution before Gurobi terminated
            except ValueError as e:
                if "Cannot load a SolverResults object with bad status: aborted" in str(e): return False, float('-inf')
                else: raise RuntimeError(f"unexpected behavior when loading results during a strong branching subroutine problem:\n\t{e}")

        # if the solution is not feasible, return None
        elif results.solver.termination_condition == TerminationCondition.infeasible:
            return False, float('-inf')

        else:
            raise RuntimeError("unexpected termination_condition for a strong branching subroutine problem: " + \
                               f"{results.solver.termination_condition}")

    def select_variable(self, 
                        node: Node, 
                        subproblems: Subproblems) -> Tuple[SupportedVars, str]:
        """
        Performs full strong branching on *all* branching (i.e. complicating) variables
        and selects the one that results in the best improvement in the lower bound.

        This requires solving 2 * (num. branching variables) * (num. subproblems) linear programs,
        so it is very expensive.  However, it can be very effective in reducing the size of the tree.

        Parameters
        -------------
        node : Node
            Current node in the spatial BnB tree.
        subproblems : Subproblems
            Subproblems objective containing this ranks subproblems.

        Returns
        --------------
        var_type : str
            type of variable - should be in SupportedVars
        var_name : str
            ID corresponding to the complicating variable that was selected.
        """
        # check model is fully linear (only need to do this once)
        if self.first_iter:
            for subproblem_name in subproblems.names:
                assert self.is_linear_model(subproblems.model[subproblem_name]), \
                    "StrongBranching only works for fully linear models.\n" + \
                    f"Subproblem {subproblem_name} is not fully linear, when binaries/integers are relaxed."
            self.first_iter = False

        # if we have a feasible LB solution from this node, we can perform full strong branching
        if node.lb_problem.feasible is True:
            best_var = None
            best_var_type = None
            best_improvement = float('-inf')

            # relax everything
            subproblems.relax_all_binaries()
            subproblems.relax_all_integers()

            # determine which variables we will evaluate this iteration
            vars_to_evaluate = self.get_vars_to_evaluate(node)

            # evaluate the vars selected (method changes based on binary/integer/continuous domain)
            for varID in vars_to_evaluate:
                var_type = self.get_var_type(varID = varID, node = node)

                # branching up
                self.branch_var(varID, var_type, subproblems, node, direction = "up")
                upward_branching_obj = 0
                for subproblem_name in subproblems.names:
                    subproblem_is_feasible, subproblem_objective = self.solve_a_subproblem(subproblems.model[subproblem_name])
                    if subproblem_is_feasible is False: 
                        upward_branching_obj = float('-inf')
                        break
                    else: upward_branching_obj += subproblem_objective * subproblems.probability[subproblem_name]
                
                MPI.COMM_WORLD.barrier()
                upward_branching_obj = MPI.COMM_WORLD.allreduce(upward_branching_obj, op=MPI.SUM)
        
                # branching down
                self.branch_var(varID, var_type, subproblems, node, direction = "down")
                downward_branching_obj = 0
                for subproblem_name in subproblems.names:
                    subproblem_is_feasible, subproblem_objective = self.solve_a_subproblem(subproblems.model[subproblem_name])
                    if subproblem_is_feasible is False: 
                        downward_branching_obj = float('-inf')
                        break
                    else: downward_branching_obj += subproblem_objective * subproblems.probability[subproblem_name]
                
                MPI.COMM_WORLD.barrier()
                downward_branching_obj = MPI.COMM_WORLD.allreduce(downward_branching_obj, op=MPI.SUM)
                
                # reset var to original bounds / unfix
                self.reset_var(varID, var_type, subproblems, node)

                # compare improvements
                improvement = max(upward_branching_obj - node.lb_problem.objective, downward_branching_obj - node.lb_problem.objective)
                if improvement > best_improvement:
                    best_improvement = improvement
                    best_var = varID
                    best_var_type = var_type

            # reset all of the states / unfix
            subproblems.unrelax_all_binaries()
            subproblems.unrelax_all_integers()

            # if we found a candidate, return it
            if (best_var is not None) and (best_var_type is not None):
                assert best_var in node.state[best_var_type]
                return best_var_type, best_var
        
        # if LB is infeasible or we didn't find a solution for any var, select at random...
        if (node.lb_problem.feasible is False) or (best_var is None): 

            # NOTE: sorting is very important for coordination over ranks 
            binaries = sorted(list(node.to_branch[SupportedVars.binary]))
            reals = sorted(list(node.to_branch[SupportedVars.reals]))
            integers = sorted(list(node.to_branch[SupportedVars.integers]))
            nonnegative_integers = sorted(list(node.to_branch[SupportedVars.nonnegative_integers]))
            vars = binaries + reals + integers + nonnegative_integers
            
            # randomly decide
            index = MPI.COMM_WORLD.bcast(np.random.randint(0, len(vars)), root=0)
            var_name = vars[index]

            # determine which var type we selected based on index value
            if binaries and (index in range(0, len(binaries))): 
                var_type = SupportedVars.binary
            elif reals and (index in range(len(binaries), (len(binaries) + len(reals)))):
                var_type = SupportedVars.reals
            elif integers and (index in range((len(binaries) + len(reals)), (len(binaries) + len(reals) + len(integers)))):
                var_type = SupportedVars.integers
            elif nonnegative_integers and (index in range((len(binaries) + len(reals) + len(integers)), len(vars))): 
                var_type = SupportedVars.nonnegative_integers

            return var_type, var_name


class FullStrongBranching(StrongBranching):
    """
    Full strong branching is the most extreme case of strong branching,
    where we will consider ALL of the branching varaibles at each solve.

    see: Achterberg, Tobias, Thorsten Koch, and Alexander Martin. 
    "Branching rules revisited." Operations Research Letters 33.1 (2005): 42-54.
    """
    def __init__(self, 
                 max_simplex_iterations: int,
                 subset_branching_variables: int,
                 *args, **kwargs) -> None:
        """
        Full strong branching strategy initialization.

        Parameters
        --------------
        max_simplex_iterations : int
            maximum number of simplex iterations to allow per subproblem
            when performing strong branching.
        """
        super().__init__(max_simplex_iterations = max_simplex_iterations,
                         subset_branching_variables = None, # defaults to all - which indicates "full"
                         *args, **kwargs)


# TODO: reliability branching

# =================== PARTITIONING STRATEGIES ============================= #

class Midpoint(PartitionStrategy):
    """
    Based on the lower bound and the upper bound,
    simply return the midpoint as the split point.
    """
    def __init__(self) -> None:
        super().__init__()
        self.epsilon = 1e-3

    def split_point(self, 
                    var_lb: float, 
                    var_ub: float,
                    *args, **kwards) -> float:
        """
        Split variable at the midpoint.

        Parameters
        -------------
        var_lb : float
            current lower bound of the variable.
        var_ub : float
            current upper bound of the variable.
        
        Returns
        --------------
        split_point : float
            point at which to split the variable.
        """
        
        # branch by splitting the space in half
        return var_lb + round(0.5 * (var_ub - var_lb), ndigits = 3)


class ExpectedValue(PartitionStrategy):
    """
    Return the average value of all solutions.

    Make sure that we do not get too close to one
    particular bound, by considering a theta tolerance.
    """
    def __init__(self) -> None:
        super().__init__()
        self.epsilon = 1e-3

        # tolerance for partition vicinity to bounds
        self.theta = 0.1

    def split_point(self,
                    varID: str,
                    var_lb: float,
                    var_ub: float,
                    node: Node,
                    subproblems: Subproblems) -> float:
        """
        Split variable at the average across all subproblem solutions.

        Parameters
        -------------
        varID : str
            ID corresponding to the complicating variable that was selected.
        var_lb : float
            current lower bound of the variable.
        var_ub : float
            current upper bound of the variable.
        node : Node 
            current node in the BnB tree.
        subproblems : Subproblems
            initialized Subproblems object.
        
        Returns
        --------------
        split_point : float
            point at which to split the variable.
        """


        # compute the expected value
        ev_var = compute.average_var_lb_solution(node = node,
                                                 subproblems = subproblems,
                                                 var_ID = varID)
        
        # compute bounds that would maintain a safe distance from current branching
        safe_lb = var_lb + self.theta * (var_ub - var_lb)
        safe_ub = var_ub - self.theta * (var_ub - var_lb)
        assert safe_lb <= safe_ub

        # reset EV if we are violating safe bounds
        if ev_var < safe_lb: return safe_lb
        if ev_var > safe_ub: return safe_ub
        
        # otw, return the EV 
        return ev_var