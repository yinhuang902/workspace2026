"""
These are methods than can help inform information
on branching / candidate generator steps. They take in,
for the most part, the node and the subproblems, and 
pass back information about the solution currently stored.
"""
from typing import Tuple, Optional
from snoglode.components.subproblems import Subproblems
from snoglode.components.node import Node
from snoglode.utils.supported import SupportedVars

import snoglode.utils.MPI as MPI
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

def average_lb_solution(node: Node,
                        subproblems: Subproblems,
                        round_binaries: bool = True,
                        round_integers: bool = True,
                        normalize: bool = False,
                        return_frequencies: bool = False) -> Tuple[dict, Optional[dict]]:
    """
    Given a node, collect the LB solutions and average.

    Method: 
        (1) get complicating var solutions from each LB subproblem solution.
        (2) average 

    NOTE: this method will always return a "feasible" candidate - i.e.,
            as long as we have a solution to the LB subproblems,
            we can generate a candidate solution and return it.
            ** as long as the round_binaries flag is enabled
    
    NOTE: the candidate solution objective will be None, because we do not
            solve an optimization in this case.

    NOTE: in the context of booleans, we round:
        - if > 0.5  -> 1
        - if <= 0.5 -> 0

    Parameters
    ----------
    node : Node
        node object representing the current node we are exploring in the branch 
        and bound tree. Contains all bounding information.
    subproblems : Subproblems
        initialized subproblem manager.
        contains all subproblem names, models, probabilities, and complicating var lists/
    round_binaries : bool, optional
        this ensures a feasible solution from the point of view of satisfaction
        of integrality constraints
    normalize : bool, optional
        when computing the averagee of the continuously bounded variables,
        normalize their solutions.
    return_frequencies: bool, optional
        optional return flag to also return the frequency of each variable.

    Returns
    ----------
    average_solution : dict
        each complicating_var_id is a key, each corresponding value is the average
        of that variable across all subproblems.
    frequency_of_complicating_vars : dict, optional
        the number of times each variable was present in each subproblem
    """
    # must have a feasible LB solution (though UB problem shouldn't be called if that is the case...)
    assert node.lb_problem.feasible

    # running sum / frequency
    # these **MUST** be in the same order across ranks!!!
    complicating_vars = sorted(subproblems.complicating_var_ids)
    aggregated_complicating_vars = {varID:0 for varID in complicating_vars}
    frequency_of_complicating_vars = {varID:0 for varID in complicating_vars}

    # access the solutions for each subproblem to add to sums / freqencies
    for subproblem_name in subproblems.names:

        # go through each of the complicating var IDs
        for var in subproblems.subproblem_complicating_vars[subproblem_name]:

            # extract varID
            var_type, varID, _ = subproblems.var_to_data[var]

            # if we have this variable in the subproblem, can compute average otw pass
            if varID in node.lb_problem.subproblem_solutions[subproblem_name].complicating_var_solution[var_type]:
                
                # retrieve solution
                LB_solution = node.lb_problem.subproblem_solutions[subproblem_name].complicating_var_solution[var_type][varID]

                # if we have a solution: normalize, if needed, and add to running sum
                if LB_solution != None:
                    if normalize and var_type != SupportedVars.binary:
                        var_lb = node.state[var_type][varID].lb
                        var_ub = node.state[var_type][varID].ub
                        normalized_LB_solution = (LB_solution - var_lb) / (var_ub - var_lb)
                        aggregated_complicating_vars[varID] += normalized_LB_solution
                    else:
                        aggregated_complicating_vars[varID] += LB_solution

                    # aggregate frequency
                    frequency_of_complicating_vars[varID] += 1
    
    # make sure all the ranks catch up first
    MPI.COMM_WORLD.barrier()

    # aggregate all of the information across ranks
    candidate_solution_state = {varID: None for varID in complicating_vars}
    for varID in complicating_vars:
        aggregated_complicating_vars[varID] = MPI.COMM_WORLD.allreduce(aggregated_complicating_vars[varID], op=MPI.SUM)
        frequency_of_complicating_vars[varID] = MPI.COMM_WORLD.allreduce(frequency_of_complicating_vars[varID], op=MPI.SUM)

        # if this variable does occur & has solutions, compute average
        if frequency_of_complicating_vars[varID] != 0:
            average_complicating_var = aggregated_complicating_vars[varID] / frequency_of_complicating_vars[varID]
            
            # round if this is a binary variable
            if varID in node.state[SupportedVars.binary] and round_binaries:
                average_complicating_var = round(average_complicating_var, ndigits=0)
                assert (average_complicating_var >= 0 and average_complicating_var <= 1)
            
            # round if this is an integer variable
            if (varID in node.state[SupportedVars.integers] or \
                varID in node.state[SupportedVars.nonnegative_integers]) and round_integers:
                average_complicating_var = round(average_complicating_var, ndigits=0)
                if varID in node.state[SupportedVars.nonnegative_integers]: 
                    assert average_complicating_var >= 0
            
            # grab value, save under var_id
            candidate_solution_state[varID] = average_complicating_var

        # if this variable does not have an average, return None
        else:
            candidate_solution_state[varID] = None

    if return_frequencies: 
        return candidate_solution_state, frequency_of_complicating_vars
    else:
        return candidate_solution_state


def average_var_lb_solution(node: Node,
                            subproblems: Subproblems,
                            var_ID: str) -> float:
    """
    Given a node & specific variable, collect the LB solutions and averages
    across the solutions of *all* subproblems

    Method: 
        (1) get complicating var solution from each LB subproblem solution.
        (2) average 

    Parameters
    ----------
    node : Node
        node object representing the current node we are exploring in the branch 
        and bound tree. Contains all bounding information.
    subproblems : Subproblems
        initialized subproblem manager.
        contains all subproblem names, models, probabilities, and complicating var lists/
    var_ID : str
        string corresponding to the complicating variable ID we want the averge of.

    Returns
    ----------
    avg_var_value : float
        averge of this variable across all of the subproblems
    """

    # must have a feasible LB solution (though UB problem shouldn't be called if that is the case...)
    assert node.lb_problem.feasible

    # determine the var type
    for var_type in SupportedVars:
        if var_ID in node.state[var_type]: break

    # running sum / frequency
    aggregated_complicating_var = 0
    frequency_of_complicating_var = 0

    # access the solutions for each subproblem to add to sums / freqencies
    for subproblem_name in subproblems.names:

        # if we have this variable in the subproblem, can compute average otw pass
        if var_ID in node.lb_problem.subproblem_solutions[subproblem_name].complicating_var_solution[var_type]:
            
            # retriew solution
            LB_solution = node.lb_problem.subproblem_solutions[subproblem_name].complicating_var_solution[var_type][var_ID]
            
            # add to running sum / frequency
            if LB_solution != None:
                aggregated_complicating_var   += LB_solution
                frequency_of_complicating_var += 1
    
    # make sure all the ranks catch up first
    MPI.COMM_WORLD.barrier()

    # compute averages
    aggregated_complicating_var = MPI.COMM_WORLD.allreduce(aggregated_complicating_var, op=MPI.SUM)
    frequency_of_complicating_var = MPI.COMM_WORLD.allreduce(frequency_of_complicating_var, op=MPI.SUM)                            
    
    # if for some reason this variable never appears, return None
    if frequency_of_complicating_var == 0: 
        return None
    
    average_complicating_var = aggregated_complicating_var / frequency_of_complicating_var
    return average_complicating_var


def variance_lb_solution(node: Node,
                         subproblems: Subproblems,
                         normalize: bool = True) -> dict:
    """
    Given a node, collect the LB solutions and find the variance across
    subproblem solutions for each of the first stage variables.

    Method: 
        (1) get complicating var solutions from each LB subproblem solution.
        (2) normalize solution, if indicated
        (3) compute the variance across the solutions 

    NOTE: In the case of continuous domains, we normalize by default such that we can
    easily compare the variance against binary domains as well.

    We define the variance as:

        var = 1/frequency * (sum(x_i - avg(x)) ^ 2 for i in subproblems containing x)
    
    where
        var:        variance
        frequency:  number of subproblem x appears in
        x_i:        solution value of x in subproblem i containing x
        avg(x):     average solution of x across all subproblems containing x

    We first compute the sum for the subproblems containing x on this rank,
    and then we aggregated using MPI.allreduce sum operation.

    Parameters
    ----------
    node : Node
        node object representing the current node we are exploring in the branch 
        and bound tree. Contains all bounding information.
    subproblems : Subproblems
        initialized subproblem manager.
        contains all subproblem names, models, probabilities, and complicating var lists/
    normalize : bool
        when computing the variance of the continuously bounded variables,
        normalize such that they can be validly compared to the variance of binary vars.

    Returns
    ----------
    variance : dict
        each complicating_var_id is a key, each corresponding value is the variance
        of that variable across all subproblems.
    """
    # must have a feasible LB solution (though UB problem shouldn't be called if that is the case...)
    assert node.lb_problem.feasible

    # running sum / frequency
    # these **MUST** be in the same order across ranks!!!
    complicating_vars = sorted(subproblems.complicating_var_ids)
    average, frequency \
            = average_lb_solution(node = node, 
                                  subproblems = subproblems,
                                  round_binaries = False,
                                  return_frequencies = True)

    variance = {varID:0 for varID in complicating_vars}
    
    # access the solutions for each subproblem to add to sums / freqencies
    for subproblem_name in subproblems.names:

        # go through each of the complicating var IDs
        for var in subproblems.subproblem_complicating_vars[subproblem_name]:

            # extract varID
            var_type, varID, _ = subproblems.var_to_data[var]

            # if we have this variable in the subproblem, can compute average otw pass
            if varID in node.lb_problem.subproblem_solutions[subproblem_name].complicating_var_solution[var_type]:

                # retriew solution
                x_i = node.lb_problem.subproblem_solutions[subproblem_name].complicating_var_solution[var_type][varID]
                if x_i != None:
                    # normalize, if needed, and add to running sum
                    if normalize and var_type != SupportedVars.binary:
                        var_lb = node.state[var_type][varID].lb
                        var_ub = node.state[var_type][varID].ub
                        variance[varID] += ((x_i - average[varID])/(var_ub - var_lb))**2

                    # if we are normalizing, update x_i
                    else:
                        variance[varID] += (x_i - average[varID])**2
    
    MPI.COMM_WORLD.barrier()

    # aggregate all of the information across ranks & compute final variance
    for varID in complicating_vars:
        if frequency[varID] != 0:
            variance[varID] = MPI.COMM_WORLD.allreduce(variance[varID], op=MPI.SUM) / frequency[varID]
        else:
            variance[varID] = None
    return variance


def frequency_of_var(node: Node,
                     subproblems: Subproblems,
                     varID: str,
                     var_type: SupportedVars,
                     with_var_solution: bool = False) -> int:
    """
    Counts the number of times a variable appears across
    all subproblems, all ranks (relevant when we have asymmetric structure...)

    Parameters
    ----------
    node : Node
        node object representing the current node we are exploring in the branch 
        and bound tree. Contains all bounding information.
    subproblems : Subproblems
        initialized subproblem manager.
        contains all subproblem names, models, probabilities, and complicating var lists/
    varID : str
        string corresponding to the complicating variable ID we want the averge of.
    var_type : SupportedVars
        corresponds to the type of var varID is
    with_var_solution : bool
        indicates if we want to only tally the variables that have a solution
        (i.e., their stored solution is anyting other than None)

    Returns
    ----------
    frequency : int
        count number of times the varID appears across all subproblems/ranks
    """
    rank_frequency = 0
    for subproblem_name in subproblems.names:

        # if we have this variable in the subproblem, can compute average otw pass
        if varID in node.lb_problem.subproblem_solutions[subproblem_name].complicating_var_solution[var_type]:
            
            # if we want to guarantee the var has a solution, check the value before incrementing
            if with_var_solution:
                x_i = node.lb_problem.subproblem_solutions[subproblem_name].complicating_var_solution[var_type][varID]
                if x_i != None:
                    rank_frequency += 1 
            
            # otw incremenet based on appearance alone
            else:
                rank_frequency += 1
    
    # make sure all the ranks catch up first
    MPI.COMM_WORLD.barrier()

    # compute global frequency
    frequency = MPI.COMM_WORLD.allreduce(rank_frequency, op=MPI.SUM)
    return frequency


def variance_var_lb_solution(node: Node,
                             subproblems: Subproblems,
                             varID: str,
                             normalize: bool = True) -> float:
    """
    Given a node, collect the LB solutions and find the variance across
    subproblem solutions for the complicating variable, varID

    Method: 
        (1) get complicating var solutions from each LB subproblem solution.
        (2) normalize solution, if indicated
        (3) compute the variance across the solutions 

    NOTE: In the case of continuous domains, we normalize by default such that we can
    easily compare the variance against binary domains as well.

    We define the variance as:

        var = 1/frequency * (sum(x_i - avg(x)) ^ 2 for i in subproblems containing x)
    
    where
        var:        variance
        frequency:  number of subproblem x appears in
        x_i:        solution value of x in subproblem i containing x
        avg(x):     average solution of x across all subproblems containing x

    We first compute the sum for the subproblems containing x on this rank,
    and then we aggregated using MPI.allreduce sum operation.

    Parameters
    ----------
    node : Node
        node object representing the current node we are exploring in the branch 
        and bound tree. Contains all bounding information.
    subproblems : Subproblems
        initialized subproblem manager.
        contains all subproblem names, models, probabilities, and complicating var lists/
    varID : str
        string corresponding to the complicating variable ID we want the averge of.
    normalize : bool
        when computing the variance of the continuously bounded variables,
        normalize such that they can be validly compared to the variance of binary vars.

    Returns
    ----------
    variance : float
        variance of the variable specified.
    """
    # must have a feasible LB solution (though UB problem shouldn't be called if that is the case...)
    assert node.lb_problem.feasible

    # determine the var type
    for var_type in SupportedVars:
        if varID in node.state[var_type]: break

    average = average_var_lb_solution(node = node, subproblems = subproblems, var_ID = varID)
    frequency = frequency_of_var(node=node, subproblems=subproblems, varID = varID, var_type=var_type, with_var_solution=True)
    
    # if there are no solutions, return None
    if frequency==0:
        return None
    
    # access the solutions for each subproblem to add to sums / freqencies
    variance = 0
    for subproblem_name in subproblems.names:

        # if we have this variable in the subproblem, can compute average otw pass
        if varID in node.lb_problem.subproblem_solutions[subproblem_name].complicating_var_solution[var_type]:

            # retriew solution
            x_i = node.lb_problem.subproblem_solutions[subproblem_name].complicating_var_solution[var_type][varID]
            if x_i != None:
                # normalize, if needed, and add to running sum
                if normalize and var_type != SupportedVars.binary:
                    var_lb = node.state[var_type][varID].lb
                    var_ub = node.state[var_type][varID].ub
                    variance += ((x_i - average)/(var_ub - var_lb))**2

                # if we are normalizing, update x_i
                else:
                    variance += (x_i - average)**2
        
    MPI.COMM_WORLD.barrier()

    # aggregate all of the information across ranks & compute final variance
    variance = MPI.COMM_WORLD.allreduce(variance, op=MPI.SUM) / frequency
    return variance