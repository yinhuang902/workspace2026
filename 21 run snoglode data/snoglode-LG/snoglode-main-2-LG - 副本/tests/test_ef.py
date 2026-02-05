"""
Testing auto-build of the Extensive Form (EF) capability
"""
import pytest as pytest
import snoglode as sno
import pyomo.environ as pyo

import snoglode.utils.MPI as MPI
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()


@pytest.mark.requires_gurobi
@pytest.mark.skipif(size > 3, reason="test can run with at most 3 ranks.")
def test_ef_generation_farmer_skew(farmer_skew):
    """
    When we generate the skewed version of the farmer problem,
    we want to ensure that we only are building the correct
    number of constraints to represent the problem
    """
    subproblem_names = ["good", "fair", "bad"]
    params = sno.SolverParameters(subproblem_names=subproblem_names,
                                  subproblem_creator=farmer_skew)
    solver = sno.Solver(params)
    
    # should be three complicating vars
    assert len(solver.subproblems.ef.model.complicating_vars) == 3

    # we expected 6 constraints (because not all complicating variables are linked)
    solver.subproblems.ef.activate()
    assert len(solver.subproblems.ef.model.nonants) == 6


@pytest.mark.requires_gurobi
@pytest.mark.skipif(size > 3, reason="test can run with at most 3 ranks.")
def test_ef_generation_farmer_classic(farmer_classic, gurobi):
    """
    When we generate the classic version of the farmer problem,
    we want to ensure that we only are building the correct
    number of constraints to represent the problem
    """
    subproblem_names = ["good", "fair", "bad"]
    params = sno.SolverParameters(subproblem_names=subproblem_names,
                                  subproblem_creator=farmer_classic,
                                  cg_solver = gurobi)
    solver = sno.Solver(params)
    
    # should be three complicating vars
    assert len(solver.subproblems.ef.model.complicating_vars) == 3

    # we expected 9 constraints (because not all complicating variables are linked across all periods)
    solver.subproblems.ef.activate()
    assert len(solver.subproblems.ef.model.nonants) == 9