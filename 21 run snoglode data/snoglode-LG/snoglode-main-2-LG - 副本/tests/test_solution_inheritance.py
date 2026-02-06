"""
In the case of lower bounders, we can sometimes reuse solutions from a parent
lower bound solution (saving significant computational time). 
"""
import pytest as pytest

@pytest.mark.mpi_skip()
def test_can_inherit_parent_solution_symmetric(sample_abstract_lowerbounder, sample_symmetric_child_nodes, sample_symmetric_subproblems):
    """
    Test that our lower bounder can inherit parent solutions correctly.

    child nodes are branched on x1 in [1,1.5] and [1.5,2]
        sub1, x1 = 1
        sub2, x1 = 2
        sub3, x1 = 1

    child1 can inherit: sub1 and sub3
    child2 can inherit: sub2
    """
    child1_node, child2_node = sample_symmetric_child_nodes    
    
    # check child1 (can inherit sub1 and sub3)
    inheritable = sample_abstract_lowerbounder.inherit_parent_solution(node = child1_node, 
                                                                       subproblems = sample_symmetric_subproblems,
                                                                       subproblem_name = "sub1")
    assert inheritable is True
    inheritable = sample_abstract_lowerbounder.inherit_parent_solution(node = child1_node, 
                                                                       subproblems = sample_symmetric_subproblems,
                                                                       subproblem_name = "sub2")
    assert inheritable is False
    inheritable = sample_abstract_lowerbounder.inherit_parent_solution(node = child1_node, 
                                                                       subproblems = sample_symmetric_subproblems,
                                                                       subproblem_name = "sub3")
    assert inheritable is True

    # check child2 (can inherit sub2)
    inheritable = sample_abstract_lowerbounder.inherit_parent_solution(node = child2_node, 
                                                                       subproblems = sample_symmetric_subproblems,
                                                                       subproblem_name = "sub1")
    assert inheritable is False
    inheritable = sample_abstract_lowerbounder.inherit_parent_solution(node = child2_node, 
                                                                       subproblems = sample_symmetric_subproblems,
                                                                       subproblem_name = "sub2")
    assert inheritable is True
    inheritable = sample_abstract_lowerbounder.inherit_parent_solution(node = child2_node, 
                                                                       subproblems = sample_symmetric_subproblems,
                                                                       subproblem_name = "sub3")
    assert inheritable is False


@pytest.mark.mpi_skip()
def test_can_inherit_parent_solution_asymmetric(sample_abstract_lowerbounder, sample_asymmetric_child_nodes, sample_asymmetric_subproblems):
    """
    Test that our lower bounder can inherit parent solutions correctly.

    child nodes are branched on x1 in [1,1.5] and [1.5,2], x2 remains [2,3]
        sub1, x1 = 1 (complicating)
              x2 = 2 (NOT complicating)
        
        sub2, x1 = 2 (complicating)
              x2 = 3 (complicating)
        
        sub3, x1 = 1 (NOT complicating)
              x2 = 3 (complicating)

    child1 can inherit: sub1 and sub3
    child2 can inherit: sub2 and sub3
    """
    child1_node, child2_node = sample_asymmetric_child_nodes    
    
    # check child1 (can inherit sub1 and sub3)
    inheritable = sample_abstract_lowerbounder.inherit_parent_solution(node = child1_node, 
                                                                       subproblems = sample_asymmetric_subproblems,
                                                                       subproblem_name = "sub1")
    assert inheritable is True
    inheritable = sample_abstract_lowerbounder.inherit_parent_solution(node = child1_node, 
                                                                       subproblems = sample_asymmetric_subproblems,
                                                                       subproblem_name = "sub2")
    assert inheritable is False
    inheritable = sample_abstract_lowerbounder.inherit_parent_solution(node = child1_node, 
                                                                       subproblems = sample_asymmetric_subproblems,
                                                                       subproblem_name = "sub3")
    assert inheritable is True

    # check child2 (can inherit sub2 and sub3)
    inheritable = sample_abstract_lowerbounder.inherit_parent_solution(node = child2_node, 
                                                                       subproblems = sample_asymmetric_subproblems,
                                                                       subproblem_name = "sub1")
    assert inheritable is False
    inheritable = sample_abstract_lowerbounder.inherit_parent_solution(node = child2_node, 
                                                                       subproblems = sample_asymmetric_subproblems,
                                                                       subproblem_name = "sub2")
    assert inheritable is True
    inheritable = sample_abstract_lowerbounder.inherit_parent_solution(node = child2_node, 
                                                                       subproblems = sample_asymmetric_subproblems,
                                                                       subproblem_name = "sub3")
    assert inheritable is True


@pytest.mark.mpi_skip()
def test_cannot_inherit_none(sample_abstract_lowerbounder, sample_None_sol_symmetric_node, sample_symmetric_subproblems):
    """
    Test that we cannot inherit solutions containing missing variable solutions.
    """
    inheritable = sample_abstract_lowerbounder.inherit_parent_solution(node = sample_None_sol_symmetric_node, 
                                                                       subproblems = sample_symmetric_subproblems,
                                                                       subproblem_name = "sub1")
    assert inheritable is False