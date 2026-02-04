# tests/test_node.py

import time
import pytest
import numpy as np

from snoglode.components.node import (
    Node,
    NodeDirection,
    LowerBoundNodeMetrics,
    UpperBoundNodeMetrics,
)
from snoglode.utils.supported import SupportedVars
from snoglode.utils.solve_stats import OneLowerBoundSolve

@pytest.fixture
def simple_state():
    """
    Minimal fake state that satisfies Node.display() access patterns.
    We do not test variable behavior here, only structure.
    """
    class FakeVar:
        def __init__(self, lb=0.0, ub=1.0, is_fixed=False, value=None):
            self.lb = lb
            self.ub = ub
            self.is_fixed = is_fixed
            self.value = value

    return {
        SupportedVars.binary: {
            "x1": FakeVar(),
        }
    }


@pytest.fixture
def simple_to_branch():
    return {
        SupportedVars.binary: ["x1"]
    }


@pytest.fixture
def terminal_to_branch():
    return {
        SupportedVars.binary: []
    }


@pytest.fixture
def sample_node(simple_to_branch, simple_state):
    return Node(simple_to_branch, simple_state, id=0)


@pytest.fixture
def terminal_node(terminal_to_branch, simple_state):
    return Node(terminal_to_branch, simple_state, id=1)


@pytest.fixture
def sample_lbsolve():
    """
    Minimal OneLowerBoundSolve stub.
    """
    lbsolve = OneLowerBoundSolve(["s1"])
    lbsolve.aggregated_objective = 5.0
    lbsolve.subproblem_solutions = {"s1": 5.0}
    return lbsolve


# ---------------------------------------------------------------------------
# Constructor & validation tests
# ---------------------------------------------------------------------------

def test_node_initializes(sample_node):
    assert sample_node.id == 0
    assert sample_node.to_branch is not None
    assert sample_node.state is not None
    assert sample_node.lb_problem is not None
    assert sample_node.ub_problem is not None


def test_node_rejects_invalid_logical_keys(simple_state):
    bad_to_branch = {"BAD": ["x1"]}

    with pytest.raises(AssertionError):
        Node(bad_to_branch, simple_state, 0)


def test_node_rejects_invalid_id(simple_to_branch, simple_state):
    with pytest.raises(AssertionError):
        Node(simple_to_branch, simple_state, "bad_id")


# ---------------------------------------------------------------------------
# Terminal logic
# ---------------------------------------------------------------------------

def test_non_terminal_node(sample_node):
    assert sample_node.terminal is False


def test_terminal_node(terminal_node):
    assert terminal_node.terminal is True


# ---------------------------------------------------------------------------
# Comparison operators
# ---------------------------------------------------------------------------

def test_node_lt_uses_lb_objective(simple_to_branch, simple_state):
    n1 = Node(simple_to_branch, simple_state, 1)
    n2 = Node(simple_to_branch, simple_state, 2)

    n1.lb_problem.objective = 1.0
    n2.lb_problem.objective = 2.0

    assert n1 < n2


def test_node_le_allows_equal_objective(simple_to_branch, simple_state):
    n1 = Node(simple_to_branch, simple_state, 1)
    n2 = Node(simple_to_branch, simple_state, 2)

    n1.lb_problem.objective = 2.0
    n2.lb_problem.objective = 2.0

    assert n1 <= n2


# ---------------------------------------------------------------------------
# Pseudocost initialization
# ---------------------------------------------------------------------------

def test_pseudocost_initialization(sample_node):
    sample_node._init_psuedocost_data(
        dir=NodeDirection.upward,
        parent_id=10,
        parent_obj=100.0,
        branched_on="x1",
        branched_on_avg=0.5,
        branched_var_lb=0.0,
        branched_var_ub=1.0,
    )

    assert sample_node.parent_id == 10
    assert sample_node.parent_obj == 100.0
    assert sample_node.dir == NodeDirection.upward
    assert sample_node.branched_on == "x1"
    assert sample_node.var_delta == 0.5


def test_pseudocost_rejects_invalid_direction(sample_node):
    with pytest.raises(AssertionError):
        sample_node._init_psuedocost_data(
            dir="up",
            parent_id=1,
            parent_obj=0.0,
            branched_on="x1",
            branched_on_avg=0.5,
            branched_var_lb=0.0,
            branched_var_ub=1.0,
        )


# ---------------------------------------------------------------------------
# LowerBoundNodeMetrics
# ---------------------------------------------------------------------------

def test_lb_feasible_sets_values(sample_lbsolve):
    lb = LowerBoundNodeMetrics()
    lb.is_feasible(sample_lbsolve)

    assert lb.feasible is True
    assert lb.objective == sample_lbsolve.aggregated_objective
    assert lb.subproblem_solutions == sample_lbsolve.subproblem_solutions


def test_lb_infeasible_sets_infinity():
    lb = LowerBoundNodeMetrics()
    lb.is_infeasible()

    assert lb.feasible is False
    assert lb.objective == float("inf")


# ---------------------------------------------------------------------------
# UpperBoundNodeMetrics
# ---------------------------------------------------------------------------

def test_ub_feasible_sets_values():
    ub = UpperBoundNodeMetrics()
    candidate = {"x1": 1}

    ub.is_feasible(10.0, candidate)

    assert ub.feasible is True
    assert ub.objective == 10.0
    assert ub.candidate == candidate


def test_ub_infeasible_sets_infinity():
    ub = UpperBoundNodeMetrics()
    ub.is_infeasible()

    assert ub.feasible is False
    assert ub.objective == float("inf")


# ---------------------------------------------------------------------------
# Display (smoke test only)
# ---------------------------------------------------------------------------

def test_display_does_not_crash(sample_node, capsys):
    sample_node.display()
    captured = capsys.readouterr()
    assert "NODE DATA" in captured.out