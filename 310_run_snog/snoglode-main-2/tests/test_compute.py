import pytest as pytest
import pyomo.environ as pyo
from snoglode import SupportedVars as SupportedVars
import snoglode.utils.compute as compute

import snoglode.utils.MPI as MPI
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

@pytest.mark.skipif(size > 3, reason="test can run with at most 3 ranks.")
def test_node_feasibility(sample_symmetric_node, sample_asymmetric_node):
    assert sample_symmetric_node.lb_problem is not None
    assert sample_asymmetric_node.lb_problem is not None

@pytest.mark.skipif(size > 3, reason="test can run with at most 3 ranks.")
def test_symmetric_average_lb_solution(sample_symmetric_node, sample_symmetric_subproblems):
    avg = compute.average_lb_solution(sample_symmetric_node, sample_symmetric_subproblems)
    assert avg["x1"] == pytest.approx(4/3) # (1+2+1)/3
    assert avg["x2"] == pytest.approx(8/3) # (2+3+3)/3

@pytest.mark.skipif(size > 3, reason="test can run with at most 3 ranks.")
def test_asymmetric_average_lb_solution(sample_asymmetric_node, sample_asymmetric_subproblems):
    avg = compute.average_lb_solution(sample_asymmetric_node, sample_asymmetric_subproblems)
    assert avg["x1"] == pytest.approx(3/2) # (2+1)/2
    assert avg["x2"] == pytest.approx(3)   # (3+3)/2

@pytest.mark.skipif(size > 3, reason="test can run with at most 3 ranks.")
def test_symmetric_average_var_lb_solution(sample_symmetric_node, sample_symmetric_subproblems):
    x1_avg = compute.average_var_lb_solution(sample_symmetric_node, sample_symmetric_subproblems, "x1")
    assert x1_avg == pytest.approx(4/3) # (1+2+1)/3
    x2_avg = compute.average_var_lb_solution(sample_symmetric_node, sample_symmetric_subproblems, "x2")
    assert x2_avg == pytest.approx(8/3) # (2+3+3)/3

@pytest.mark.skipif(size > 3, reason="test can run with at most 3 ranks.")
def test_asymmetric_average_var_lb_solution(sample_asymmetric_node, sample_asymmetric_subproblems):
    x1_avg = compute.average_var_lb_solution(sample_asymmetric_node, sample_asymmetric_subproblems, "x1")
    assert x1_avg == pytest.approx(3/2) # (2+1)/2
    x2_avg = compute.average_var_lb_solution(sample_asymmetric_node, sample_asymmetric_subproblems, "x2")
    assert x2_avg == pytest.approx(3)   # (3+3)/2

@pytest.mark.skipif(size > 3, reason="test can run with at most 3 ranks.")
def test_symmetric_variance_lb_solution(sample_symmetric_node, sample_symmetric_subproblems):
    var = compute.variance_lb_solution(sample_symmetric_node, sample_symmetric_subproblems)
    assert var["x1"] == pytest.approx(0.2222222222) # ((1-4/3)^2 + (2-4/3)^2) + (1-4/3)^2) / 3
    assert var["x2"] == pytest.approx(0.2222222222) # ((2-8/3)^2 + (3-8/3)^2) + (3-8/3)^2)) / 3

@pytest.mark.skipif(size > 3, reason="test can run with at most 3 ranks.")
def test_asymmetric_variance_lb_solution(sample_asymmetric_node, sample_asymmetric_subproblems):
    var = compute.variance_lb_solution(sample_asymmetric_node, sample_asymmetric_subproblems)
    assert var["x1"] == pytest.approx(0.25) # ((2-3/2)^2 + (1-3/2)^2) / 2
    assert var["x2"] == pytest.approx(0.0)  # ((3-3)^2 + (3-3)^2) / 2

@pytest.mark.skipif(size > 3, reason="test can run with at most 3 ranks.")
def test_symmetric_frequency_of_var(sample_symmetric_node, sample_symmetric_subproblems):
    x1_freq = compute.frequency_of_var(sample_symmetric_node, sample_symmetric_subproblems, "x1", SupportedVars.reals)
    assert x1_freq == 3 # appears in all subproblems
    x2_freq = compute.frequency_of_var(sample_symmetric_node, sample_symmetric_subproblems, "x2", SupportedVars.reals)
    assert x2_freq == 3 # appears in all subproblems

@pytest.mark.skipif(size > 3, reason="test can run with at most 3 ranks.")
def test_asymmetric_frequency_of_var(sample_asymmetric_node, sample_asymmetric_subproblems):
    x1_freq = compute.frequency_of_var(sample_asymmetric_node, sample_asymmetric_subproblems, "x1", SupportedVars.reals)
    assert x1_freq == 2 # appears in both subproblems
    x2_freq = compute.frequency_of_var(sample_asymmetric_node, sample_asymmetric_subproblems, "x2", SupportedVars.reals)
    assert x2_freq == 2 # appears in both subproblems

@pytest.mark.skipif(size > 3, reason="test can run with at most 3 ranks.")
def test_symmetric_variance_var_lb_solution(sample_symmetric_node, sample_symmetric_subproblems):
    x1_var = compute.variance_var_lb_solution(sample_symmetric_node, sample_symmetric_subproblems, "x1")
    assert x1_var == pytest.approx(0.2222222222) # ((1-4/3)^2 + (2-4/3)^2) + (1-4/3)^2) / 3
    x2_var = compute.variance_var_lb_solution(sample_symmetric_node, sample_symmetric_subproblems, "x2")
    assert x2_var == pytest.approx(0.2222222222) # ((2-8/3)^2 + (3-8/3)^2) + (3-8/3)^2)) / 3

@pytest.mark.skipif(size > 3, reason="test can run with at most 3 ranks.")
def test_asymmetric_variance_var_lb_solution(sample_asymmetric_node, sample_asymmetric_subproblems):
    x1_var = compute.variance_var_lb_solution(sample_asymmetric_node, sample_asymmetric_subproblems, "x1")
    assert x1_var == pytest.approx(0.25) # ((2-3/2)^2 + (1-3/2)^2) / 2
    x2_var = compute.variance_var_lb_solution(sample_asymmetric_node, sample_asymmetric_subproblems, "x2")
    assert x2_var == pytest.approx(0.0)  # ((3-3)^2 + (3-3)^2) / 2

# ================================================================================================
# Test compute functions when we have None as some of the variable solutions
# ================================================================================================

@pytest.mark.skipif(size > 3, reason="test can run with at most 3 ranks.")
def test_None_sol_symmetric_average_lb_solution(sample_None_sol_symmetric_node, sample_symmetric_subproblems):
    avg = compute.average_lb_solution(sample_None_sol_symmetric_node, sample_symmetric_subproblems)
    assert avg["x1"] == pytest.approx(3/2) # (2+1)/2
    assert avg["x2"] == pytest.approx(8/3) # (2+3+3)/3

@pytest.mark.skipif(size > 3, reason="test can run with at most 3 ranks.")
def test_None_sol_asymmetric_average_lb_solution(sample_None_sol_asymmetric_node, sample_asymmetric_subproblems):
    avg = compute.average_lb_solution(sample_None_sol_asymmetric_node, sample_asymmetric_subproblems)
    assert avg["x1"] == pytest.approx(2)   # (2)/1
    assert avg["x2"] == pytest.approx(3)   # (3+3)/2

@pytest.mark.skipif(size > 3, reason="test can run with at most 3 ranks.")
def test_None_sol_symmetric_average_var_lb_solution(sample_None_sol_symmetric_node, sample_symmetric_subproblems):
    x1_avg = compute.average_var_lb_solution(sample_None_sol_symmetric_node, sample_symmetric_subproblems, "x1")
    assert x1_avg == pytest.approx(3/2) # (2+1)/2
    x2_avg = compute.average_var_lb_solution(sample_None_sol_symmetric_node, sample_symmetric_subproblems, "x2")
    assert x2_avg == pytest.approx(8/3) # (2+3+3)/3

@pytest.mark.skipif(size > 3, reason="test can run with at most 3 ranks.")
def test_None_sol_asymmetric_average_var_lb_solution(sample_None_sol_asymmetric_node, sample_asymmetric_subproblems):
    x1_avg = compute.average_var_lb_solution(sample_None_sol_asymmetric_node, sample_asymmetric_subproblems, "x1")
    assert x1_avg == pytest.approx(2)   # (2)/1
    x2_avg = compute.average_var_lb_solution(sample_None_sol_asymmetric_node, sample_asymmetric_subproblems, "x2")
    assert x2_avg == pytest.approx(3)   # (3+3)/2

@pytest.mark.skipif(size > 3, reason="test can run with at most 3 ranks.")
def test_None_sol_symmetric_variance_lb_solution(sample_None_sol_symmetric_node, sample_symmetric_subproblems):
    var = compute.variance_lb_solution(sample_None_sol_symmetric_node, sample_symmetric_subproblems)
    assert var["x1"] == pytest.approx(0.25)         # ((2-3/2)^2) + (1-3/2)^2) / 2
    assert var["x2"] == pytest.approx(0.2222222222) # ((2-8/3)^2 + (3-8/3)^2) + (3-8/3)^2)) / 3

@pytest.mark.skipif(size > 3, reason="test can run with at most 3 ranks.")
def test_None_sol_asymmetric_variance_lb_solution(sample_None_sol_asymmetric_node, sample_asymmetric_subproblems):
    var = compute.variance_lb_solution(sample_None_sol_asymmetric_node, sample_asymmetric_subproblems)
    assert var["x1"] == pytest.approx(0.0)  # ((2-2)^2) / 1
    assert var["x2"] == pytest.approx(0.0)  # ((3-3)^2 + (3-3)^2) / 2

@pytest.mark.skipif(size > 3, reason="test can run with at most 3 ranks.")
def test_None_sol_symmetric_frequency_of_var(sample_symmetric_node, sample_symmetric_subproblems):
    # inject one None solution; should reduce incremement by 1
    sample_symmetric_node.lb_problem.subproblem_solutions["sub1"].complicating_var_solution[SupportedVars.reals]["x1"] = None
    x1_freq = compute.frequency_of_var(sample_symmetric_node, sample_symmetric_subproblems, "x1", SupportedVars.reals, with_var_solution=True)
    assert x1_freq == 2 # appears in all subproblems, with 1 None solution

@pytest.mark.skipif(size > 3, reason="test can run with at most 3 ranks.")
def test_None_sol_asymmetric_frequency_of_var(sample_asymmetric_node, sample_asymmetric_subproblems):
    # inject one None solution; should reduce incremement by 1
    sample_asymmetric_node.lb_problem.subproblem_solutions["sub1"].complicating_var_solution[SupportedVars.reals]["x1"] = None
    x1_freq = compute.frequency_of_var(sample_asymmetric_node, sample_asymmetric_subproblems, "x1", SupportedVars.reals, with_var_solution=True)
    assert x1_freq == 1 # appears in both subproblems, with 1 None solution

@pytest.mark.skipif(size > 3, reason="test can run with at most 3 ranks.")
def test_all_None_sol_average_lb_solution(sample_all_None_sol_node, sample_symmetric_subproblems):
    avg = compute.average_lb_solution(sample_all_None_sol_node, sample_symmetric_subproblems)
    assert avg["x1"] == None
    assert avg["x2"] == None

@pytest.mark.skipif(size > 3, reason="test can run with at most 3 ranks.")
def test_all_None_sol_average_var_lb_solution(sample_all_None_sol_node, sample_symmetric_subproblems):
    x1_avg = compute.average_var_lb_solution(sample_all_None_sol_node, sample_symmetric_subproblems, "x1")
    assert x1_avg == None
    x2_avg = compute.average_var_lb_solution(sample_all_None_sol_node, sample_symmetric_subproblems, "x2")
    assert x2_avg == None

@pytest.mark.skipif(size > 3, reason="test can run with at most 3 ranks.")
def test_all_None_sol_variance_lb_solution(sample_all_None_sol_node, sample_symmetric_subproblems):
    var = compute.variance_lb_solution(sample_all_None_sol_node, sample_symmetric_subproblems)
    assert var["x1"] == None
    assert var["x2"] == None

@pytest.mark.skipif(size > 3, reason="test can run with at most 3 ranks.")
def test_all_None_sol_frequency_of_var(sample_all_None_sol_node, sample_symmetric_subproblems):
    x1_freq = compute.frequency_of_var(sample_all_None_sol_node, sample_symmetric_subproblems, "x1", SupportedVars.reals, with_var_solution=True)
    assert x1_freq == 0
    x1_freq = compute.frequency_of_var(sample_all_None_sol_node, sample_symmetric_subproblems, "x1", SupportedVars.reals, with_var_solution=False)
    assert x1_freq == 3
    x2_freq = compute.frequency_of_var(sample_all_None_sol_node, sample_symmetric_subproblems, "x2", SupportedVars.reals, with_var_solution=True)
    assert x2_freq == 0
    x2_freq = compute.frequency_of_var(sample_all_None_sol_node, sample_symmetric_subproblems, "x1", SupportedVars.reals, with_var_solution=False)
    assert x2_freq == 3