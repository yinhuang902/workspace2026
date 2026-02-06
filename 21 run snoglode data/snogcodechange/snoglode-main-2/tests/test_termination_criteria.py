# tests/unit/test_evaluate_termination.py

import pytest
import pyomo.environ as pyo
from pyomo.opt import TerminationCondition, SolverStatus
from pyomo.contrib.alternative_solutions.aos_utils import get_active_objective

# ============================================================
# FAKES (Pyomo-like minimal objects)
# ============================================================

class FakeProblem:
    def __init__(self, lb=None, ub=None):
        self.lower_bound = lb
        self.upper_bound = ub


class FakeSolverInfo:
    def __init__(self, term_cond, status=SolverStatus.ok):
        self.termination_condition = term_cond
        self.status = status


class FakeResults:
    def __init__(self, term_cond, status=SolverStatus.ok, lb=None, ub=None):
        self.solver = FakeSolverInfo(term_cond, status)
        self.problem = FakeProblem(lb, ub)


class FakeSolutions:
    def __init__(self):
        self.loaded = False

    def load_from(self, results):
        self.loaded = True


class FakeModel:
    def __init__(self, successor_obj=float("-inf")):
        self.solutions = FakeSolutions()
        self.successor_obj = successor_obj


# ============================================================
# OBJECT UNDER TEST (minimal stand-in)
# ============================================================

class DummySolverWrapper:
    """
    Minimal object that owns evaluate_termination.
    Only implements what the method actually touches.
    """

    def __init__(self, solver_name="baron", lb_return=None):
        self.solver = type("Solver", (), {"name": solver_name})
        self._lb_return = lb_return

    def retrieve_solver_lb(self, results):
        return self._lb_return

    # ---- METHOD UNDER TEST (copied verbatim) ----
    def evaluate_termination(self, results, subproblem_model):
        if results.solver.termination_condition == TerminationCondition.locallyOptimal \
                and "baron" in self.solver.name:

            subproblem_model.solutions.load_from(results)

            LB = results.problem.lower_bound
            UB = results.problem.upper_bound

            if LB is not None and UB is not None:
                rel_gap = abs(UB - LB) / (abs(UB) + 1e-10)
                if rel_gap <= 0.00011:
                    results.solver.termination_condition = pyo.TerminationCondition.optimal

        if results.solver.termination_condition == TerminationCondition.locallyOptimal:
            raise RuntimeError(
                "While solving a subproblem at the lower bound, found a locally optimal solution."
            )

        if results.solver.termination_condition in [
            TerminationCondition.optimal,
            TerminationCondition.globallyOptimal
        ] and results.solver.status == SolverStatus.ok:

            subproblem_model.solutions.load_from(results)
            return True, pyo.value(get_active_objective(subproblem_model))

        if results.solver.termination_condition in [
            TerminationCondition.maxTimeLimit,
            TerminationCondition.maxIterations,
            TerminationCondition.maxEvaluations,
            TerminationCondition.feasible
        ]:
            subproblem_lb = self.retrieve_solver_lb(results)
            parent_obj = pyo.value(subproblem_model.successor_obj)

            if (subproblem_lb is None) and (parent_obj == float("-inf")):
                raise RuntimeError("Could not access a lower bound")

            return True, max(subproblem_lb, parent_obj)

        elif results.solver.termination_condition == TerminationCondition.infeasible:
            return False, None

        else:
            raise RuntimeError(
                f"unexpected termination_condition: "
                f"{results.solver.termination_condition}"
            )


# ============================================================
# FIXTURES
# ============================================================

@pytest.fixture
def patch_active_objective(monkeypatch):
    monkeypatch.setattr(
        __name__ + ".get_active_objective",
        lambda model: 10.0
    )


# ============================================================
# TESTS
# ============================================================

def test_baron_locally_optimal_promoted_to_optimal(patch_active_objective):
    solver = DummySolverWrapper(solver_name="baron")
    model = FakeModel()

    results = FakeResults(
        term_cond=TerminationCondition.locallyOptimal,
        lb=9.999,
        ub=10.0
    )

    feasible, obj = solver.evaluate_termination(results, model)

    assert feasible is True
    assert obj == 10.0
    assert model.solutions.loaded
    assert results.solver.termination_condition == TerminationCondition.optimal


def test_locally_optimal_raises_error():
    solver = DummySolverWrapper(solver_name="ipopt")
    model = FakeModel()

    results = FakeResults(TerminationCondition.locallyOptimal)

    with pytest.raises(RuntimeError, match="locally optimal"):
        solver.evaluate_termination(results, model)


def test_optimal_solution_returns_objective(patch_active_objective):
    solver = DummySolverWrapper()
    model = FakeModel()

    results = FakeResults(TerminationCondition.optimal)

    feasible, obj = solver.evaluate_termination(results, model)

    assert feasible is True
    assert obj == 10.0
    assert model.solutions.loaded


def test_time_limit_returns_best_bound():
    solver = DummySolverWrapper(lb_return=8.0)
    model = FakeModel(successor_obj=7.0)

    results = FakeResults(TerminationCondition.maxTimeLimit)

    feasible, val = solver.evaluate_termination(results, model)

    assert feasible is True
    assert val == 8.0


def test_time_limit_without_bounds_raises():
    solver = DummySolverWrapper(lb_return=None)
    model = FakeModel(successor_obj=float("-inf"))

    results = FakeResults(TerminationCondition.maxTimeLimit)

    with pytest.raises(RuntimeError):
        solver.evaluate_termination(results, model)


def test_infeasible_returns_false():
    solver = DummySolverWrapper()
    model = FakeModel()

    results = FakeResults(TerminationCondition.infeasible)

    feasible, obj = solver.evaluate_termination(results, model)

    assert feasible is False
    assert obj is None


def test_unexpected_termination_raises():
    solver = DummySolverWrapper()
    model = FakeModel()

    results = FakeResults(TerminationCondition.unbounded)

    with pytest.raises(RuntimeError, match="unexpected termination_condition"):
        solver.evaluate_termination(results, model)
