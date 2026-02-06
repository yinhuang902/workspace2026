import pytest
import pyomo.environ as pyo
import copy as cp

# ---------------------------------------------------------------------------
# Gurobi license checks (avoids GitHub action CI failures)
# ---------------------------------------------------------------------------

def _has_gurobi_license() -> bool:
    try:
        import gurobipy as gp
        env = gp.Env(empty=True)
        env.start()   # license check
        env.dispose()
        return True
    except Exception:
        return False


@pytest.fixture(scope="session")
def has_gurobi():
    """Session-wide Gurobi availability + license check."""
    return _has_gurobi_license()


def pytest_runtest_setup(item):
    if "requires_gurobi" in item.keywords:
        has_gurobi = item.session._has_gurobi = getattr(
            item.session, "_has_gurobi", _has_gurobi_license()
        )
        if not has_gurobi:
            pytest.skip("Gurobi license not available")

@pytest.fixture(scope="session")
def gurobi():
    return pyo.SolverFactory("gurobi")

@pytest.fixture(scope="session")
def nonconvex_gurobi():
    return pyo.SolverFactory("gurobi", options={"nonconvex": 2})

# ---------------------------------------------------------------------------
# Problem instances fixture configuration
# ---------------------------------------------------------------------------

try: from problems import farmer_classic_subproblem_creator, \
                            bilinear_subproblem_creator, \
                            pmedian_subproblem_creator, \
                            farmer_skew_subproblem_creator, \
                            MockCandidateGenerator, IntegerProgram, \
                            continuous_1var_subproblem_creator, \
                            integer_knapsack_subproblem_creator, \
                            quadradic_subproblem_creator
except: from .problems import farmer_classic_subproblem_creator, \
                                bilinear_subproblem_creator, \
                                pmedian_subproblem_creator, \
                                farmer_skew_subproblem_creator, \
                                MockCandidateGenerator, IntegerProgram, \
                                continuous_1var_subproblem_creator, \
                                integer_knapsack_subproblem_creator, \
                                quadradic_subproblem_creator

@pytest.fixture
def farmer_classic():
    return farmer_classic_subproblem_creator

@pytest.fixture
def bilinear():
    return bilinear_subproblem_creator

@pytest.fixture
def pmedian():
    return pmedian_subproblem_creator

@pytest.fixture
def farmer_skew():
    return farmer_skew_subproblem_creator

@pytest.fixture
def mock_cg():
    return MockCandidateGenerator

@pytest.fixture
def integer_program():
    return IntegerProgram

@pytest.fixture
def continuous_1var():
    return continuous_1var_subproblem_creator

@pytest.fixture
def integer_knapsack():
 return integer_knapsack_subproblem_creator

@pytest.fixture
def quadradic():
    quadradic_subproblem_creator

# ---------------------------------------------------------------------------
# Simplified symmetric/asymmetric problem fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_subproblem_names():
    return ["sub1", "sub2", "sub3"]

@pytest.fixture
def sample_symmetric_subproblem_creator():
    def subproblem_creator(name):
        model = pyo.ConcreteModel()
        if name == "sub1":
            model.x1 = pyo.Var(initialize=1, domain = pyo.Reals, bounds=(1,2))
            model.x2 = pyo.Var(initialize=2, domain = pyo.Reals, bounds=(2,3))
            model.obj = pyo.Objective(expr=0, sense=pyo.minimize)
        elif name == "sub2":
            model.x1 = pyo.Var(initialize=2, domain = pyo.Reals, bounds=(1,2))
            model.x2 = pyo.Var(initialize=3, domain = pyo.Reals, bounds=(2,3))
            model.obj = pyo.Objective(expr=0, sense=pyo.minimize)
        elif name == "sub3":
            model.x1 = pyo.Var(initialize=1, domain = pyo.Reals, bounds=(1,2))
            model.x2 = pyo.Var(initialize=3, domain = pyo.Reals, bounds=(2,3))
            model.obj = pyo.Objective(expr=0, sense=pyo.minimize)
        return model, {"x1": model.x1, "x2": model.x2}, 1/3
    return subproblem_creator

@pytest.fixture
def sample_asymmetric_subproblem_creator():
    def subproblem_creator(name):
        model = pyo.ConcreteModel()
        if name == "sub1":
            model.x1 = pyo.Var(initialize=1, domain = pyo.Reals, bounds=(1,2))
            model.x2 = pyo.Var(initialize=2, domain = pyo.Reals, bounds=(2,3))
            model.obj = pyo.Objective(expr=0, sense=pyo.minimize)
            complicating_vars = {"x1": model.x1}
        elif name == "sub2":
            model.x1 = pyo.Var(initialize=2, domain = pyo.Reals, bounds=(1,2))
            model.x2 = pyo.Var(initialize=3, domain = pyo.Reals, bounds=(2,3))
            model.obj = pyo.Objective(expr=0, sense=pyo.minimize)
            complicating_vars = {"x1": model.x1, "x2": model.x2}
        elif name == "sub3":
            model.x1 = pyo.Var(initialize=1, domain = pyo.Reals, bounds=(1,2))
            model.x2 = pyo.Var(initialize=3, domain = pyo.Reals, bounds=(2,3))
            model.obj = pyo.Objective(expr=0, sense=pyo.minimize)
            complicating_vars = {"x2": model.x2}
        return model, complicating_vars, 1/3
    return subproblem_creator

@pytest.fixture
def sample_symmetric_subproblems(sample_symmetric_subproblem_creator):
    from snoglode import Subproblems as Subproblems
    return Subproblems(subproblem_names=["sub1", "sub2", "sub3"],
                       subproblem_creator=sample_symmetric_subproblem_creator,
                       use_fbbt=False,
                       use_obbt=False,
                       obbt_solver_name=None,
                       obbt_solver_opts=None,
                       relax_binaries=False,
                       relax_integers=False)

@pytest.fixture
def sample_asymmetric_subproblems(sample_asymmetric_subproblem_creator):
    from snoglode import Subproblems as Subproblems
    return Subproblems(subproblem_names=["sub1", "sub2", "sub3"],
                       subproblem_creator=sample_asymmetric_subproblem_creator,
                       use_fbbt=False,
                       use_obbt=False,
                       obbt_solver_name=None,
                       obbt_solver_opts=None,
                       relax_binaries=False,
                       relax_integers=False)

@pytest.fixture
def sample_to_branch(sample_subproblem_names):
    from snoglode import SupportedVars as SupportedVars
    to_branch = {var_type: {} for var_type in SupportedVars}
    to_branch[SupportedVars.reals] = sample_subproblem_names
    return to_branch

@pytest.fixture
def sample_state():
    from snoglode import SupportedVars
    from snoglode.components.subproblems import ComplicatingVariable as ComplicatingVariable
    state = {var_type: {} for var_type in SupportedVars}
    state[SupportedVars.reals]["x1"] = ComplicatingVariable(SupportedVars.reals, 1, 2, "x1")
    state[SupportedVars.reals]["x2"] = ComplicatingVariable(SupportedVars.reals, 2, 3, "x2")
    return state

@pytest.fixture
def sample_symmetric_lbsolve(sample_subproblem_names, sample_symmetric_subproblems):
    from snoglode.components.node import OneLowerBoundSolve as OneLowerBoundSolve
    lbsolve = OneLowerBoundSolve(sample_subproblem_names)
    lbsolve.update(subproblem_name="sub1", subproblem_objective=0,subproblems=sample_symmetric_subproblems)
    lbsolve.update(subproblem_name="sub2", subproblem_objective=0,subproblems=sample_symmetric_subproblems)
    lbsolve.update(subproblem_name="sub3", subproblem_objective=0,subproblems=sample_symmetric_subproblems)
    return lbsolve

@pytest.fixture
def sample_asymmetric_lbsolve(sample_subproblem_names, sample_asymmetric_subproblems):
    from snoglode.components.node import OneLowerBoundSolve as OneLowerBoundSolve
    lbsolve = OneLowerBoundSolve(sample_subproblem_names)
    lbsolve.update(subproblem_name="sub1", subproblem_objective=0,subproblems=sample_asymmetric_subproblems)
    lbsolve.update(subproblem_name="sub2", subproblem_objective=0,subproblems=sample_asymmetric_subproblems)
    lbsolve.update(subproblem_name="sub3", subproblem_objective=0,subproblems=sample_asymmetric_subproblems)
    return lbsolve

@pytest.fixture
def sample_symmetric_node(sample_to_branch, sample_state, sample_symmetric_lbsolve):
    from snoglode import Node as NodeClass
    node = NodeClass(sample_to_branch, sample_state, 0)
    node.lb_problem.is_feasible(sample_symmetric_lbsolve)
    return node

@pytest.fixture
def sample_asymmetric_node(sample_to_branch, sample_state, sample_asymmetric_lbsolve):
    from snoglode import Node as NodeClass
    node = NodeClass(sample_to_branch, sample_state, 0)
    node.lb_problem.is_feasible(sample_asymmetric_lbsolve)
    return node

@pytest.fixture
def sample_symmetric_child_nodes(sample_to_branch, sample_state, sample_symmetric_lbsolve):
    from snoglode import Node as NodeClass
    from snoglode import SupportedVars
    # assume we branch on x1 in [1,1.5] and [1.5,2]
    child1_state = cp.deepcopy(sample_state)
    child1_state[SupportedVars.reals]["x1"].lb = 1
    child1_state[SupportedVars.reals]["x1"].ub = 1.5
    child2_state = cp.deepcopy(sample_state)
    child2_state[SupportedVars.reals]["x1"].lb = 1.5
    child2_state[SupportedVars.reals]["x1"].ub = 2
    child1_node = NodeClass(sample_to_branch, child1_state, 1)
    child1_node.lb_problem.is_feasible(sample_symmetric_lbsolve)
    child2_node = NodeClass(sample_to_branch, child2_state, 2)
    child2_node.lb_problem.is_feasible(sample_symmetric_lbsolve)
    return child1_node, child2_node 

@pytest.fixture
def sample_asymmetric_child_nodes(sample_to_branch, sample_state, sample_asymmetric_lbsolve):
    from snoglode import Node as NodeClass
    from snoglode import SupportedVars
    # assume we branch on x1 in [1,1.5] and [1.5,2]
    child1_state = cp.deepcopy(sample_state)
    child1_state[SupportedVars.reals]["x1"].lb = 1
    child1_state[SupportedVars.reals]["x1"].ub = 1.5
    child2_state = cp.deepcopy(sample_state)
    child2_state[SupportedVars.reals]["x1"].lb = 1.5
    child2_state[SupportedVars.reals]["x1"].ub = 2
    child1_node = NodeClass(sample_to_branch, child1_state, 1)
    child1_node.lb_problem.is_feasible(sample_asymmetric_lbsolve)
    child2_node = NodeClass(sample_to_branch, child2_state, 2)
    child2_node.lb_problem.is_feasible(sample_asymmetric_lbsolve)
    return child1_node, child2_node 

@pytest.fixture
def sample_None_sol_symmetric_node(sample_to_branch, sample_state, sample_symmetric_lbsolve):
    from snoglode import Node as NodeClass
    from snoglode import SupportedVars
    node = NodeClass(sample_to_branch, sample_state, 0)
    node.lb_problem.is_feasible(sample_symmetric_lbsolve)
    # inject one None solution at subproblem 1, for x1
    node.lb_problem.subproblem_solutions["sub1"].complicating_var_solution[SupportedVars.reals]["x1"] = None
    return node

@pytest.fixture
def sample_None_sol_asymmetric_node(sample_to_branch, sample_state, sample_asymmetric_lbsolve):
    from snoglode import Node as NodeClass
    from snoglode import SupportedVars
    node = NodeClass(sample_to_branch, sample_state, 0)
    node.lb_problem.is_feasible(sample_asymmetric_lbsolve)
    # inject one None solution at subproblem 1, for x1
    node.lb_problem.subproblem_solutions["sub1"].complicating_var_solution[SupportedVars.reals]["x1"] = None
    return node

@pytest.fixture
def sample_all_None_sol_node(sample_to_branch, sample_state, sample_symmetric_lbsolve):
    from snoglode import Node as NodeClass
    from snoglode import SupportedVars
    node = NodeClass(sample_to_branch, sample_state, 0)
    node.lb_problem.is_feasible(sample_symmetric_lbsolve)
    node.lb_problem.subproblem_solutions["sub1"].complicating_var_solution[SupportedVars.reals]["x1"] = None
    node.lb_problem.subproblem_solutions["sub2"].complicating_var_solution[SupportedVars.reals]["x1"] = None
    node.lb_problem.subproblem_solutions["sub3"].complicating_var_solution[SupportedVars.reals]["x1"] = None
    node.lb_problem.subproblem_solutions["sub1"].complicating_var_solution[SupportedVars.reals]["x2"] = None
    node.lb_problem.subproblem_solutions["sub2"].complicating_var_solution[SupportedVars.reals]["x2"] = None
    node.lb_problem.subproblem_solutions["sub3"].complicating_var_solution[SupportedVars.reals]["x2"] = None
    return node

# ---------------------------------------------------------------------------
# Additional misc fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_abstract_lowerbounder(gurobi):
    from snoglode import AbstractLowerBounder as AbstractLowerBounder
    lbounder = AbstractLowerBounder(gurobi)
    return lbounder

@pytest.fixture
def dummy_results():
    class DummySolver:
        def __init__(self):
            self.termination_condition = None
            self.name = None

    class DummyResults:
        def __init__(self, solver):
            self.solver = solver

@pytest.fixture
def dummy_subproblem_model():
    class DummySolutions:
        def __init__(self):
            pass

        
    class SubproblemModel:
        def __init__(self, solutions):
            self.solutions = solutions