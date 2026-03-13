"""
problem_2_1_7.py -- Snoglode solver for SNGO-master/Global/2_1_7

Julia reference (PlasmoOld):
  RandomStochasticModel(createModel, NS=100) => nscen=100, nfirst=5, nparam=5
  srand(1234), scenario 1 unperturbed

Model: 20 variables (xi >= 0), 10 <= constraints, concave quadratic objective
  Objective: Min -0.5 * sum(i*(xi-2)^2, i=1..20)

Stage split (nfirst=5):
  First-stage:  x1..x5
  Second-stage: x6..x20

Stochastic perturbation (PlasmoOld addnoise for <= constraints):
  addnoise(ub, 0, 10, 0, 2) = ub + |ub| * U(0, 2)  if ub != 0
                             = ub + U(0, 10)          if ub == 0
  nparam=5 => c1-c5 perturbed. Scenario 0 unperturbed.
"""
import pyomo.environ as pyo
import os
import sys
import csv

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import snoglode as sno
import snoglode.utils.MPI as MPI

rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()

# ============================================================================
# Parameters matching Julia setup.jl
# ============================================================================
NUM_SCENARIOS = 100
NFIRST = 5
NPARAM = 5
SEED = 1234

# ============================================================================
# Julia-compatible RNG (MT19937 seeded with 1234) + PlasmoOld addnoise
# ============================================================================

class JuliaMT19937:
    N = 624; M = 397
    MATRIX_A = 0x9908B0DF; UPPER_MASK = 0x80000000; LOWER_MASK = 0x7FFFFFFF

    def __init__(self, seed: int = 1234):
        self.mt = [0] * self.N
        self.mti = self.N + 1
        self._seed(seed)

    def _seed(self, seed: int):
        seed &= 0xFFFFFFFF
        self.mt[0] = seed
        for i in range(1, self.N):
            self.mt[i] = (1812433253 * (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) + i) & 0xFFFFFFFF
        self.mti = self.N

    def rand_uint32(self) -> int:
        mag01 = [0x0, self.MATRIX_A]
        if self.mti >= self.N:
            for kk in range(self.N - self.M):
                y = (self.mt[kk] & self.UPPER_MASK) | (self.mt[kk + 1] & self.LOWER_MASK)
                self.mt[kk] = self.mt[kk + self.M] ^ (y >> 1) ^ mag01[y & 0x1]
            for kk in range(self.N - self.M, self.N - 1):
                y = (self.mt[kk] & self.UPPER_MASK) | (self.mt[kk + 1] & self.LOWER_MASK)
                self.mt[kk] = self.mt[kk + (self.M - self.N)] ^ (y >> 1) ^ mag01[y & 0x1]
            y = (self.mt[self.N - 1] & self.UPPER_MASK) | (self.mt[0] & self.LOWER_MASK)
            self.mt[self.N - 1] = self.mt[self.M - 1] ^ (y >> 1) ^ mag01[y & 0x1]
            self.mti = 0
        y = self.mt[self.mti]; self.mti += 1
        y ^= (y >> 11); y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000; y ^= (y >> 18)
        return y & 0xFFFFFFFF

    def rand_uint64(self) -> int:
        return ((self.rand_uint32() << 32) | self.rand_uint32()) & 0xFFFFFFFFFFFFFFFF

    def rand_float64(self) -> float:
        return (self.rand_uint64() >> 12) * (1.0 / (1 << 52))

    def rand_uniform(self, a: float, b: float) -> float:
        return a + (b - a) * self.rand_float64()


def addnoise_le(a: float, rng: JuliaMT19937) -> float:
    """PlasmoOld addnoise for <= constraint UB: addnoise(a, 0, 10, 0, 2)"""
    if a == 0.0:
        return a + rng.rand_uniform(0.0, 10.0)
    return a + abs(a) * rng.rand_uniform(0.0, 2.0)


# ============================================================================
# Pre-compute scenario RHS values (RNG must be consumed sequentially)
# ============================================================================
# Base RHS values for c1-c5 (in constraint declaration order)
_RHS_NAMES = ["c1_rhs", "c2_rhs", "c3_rhs", "c4_rhs", "c5_rhs"]
_RHS_BASE  = [-5.0, 2.0, -1.0, -3.0, 5.0]

def _precompute_scenario_data():
    """Pre-compute perturbed RHS for all scenarios so subproblem_creator is order-independent."""
    rng = JuliaMT19937(SEED)
    data = {}
    for s in range(NUM_SCENARIOS):
        name = f"scen_{s}"
        if s == 0:
            # Scenario 0: unperturbed
            data[name] = {_RHS_NAMES[i]: _RHS_BASE[i] for i in range(NPARAM)}
        else:
            # Scenarios 1-99: perturb first NPARAM constraint RHS
            rhs = {}
            for i in range(NPARAM):
                rhs[_RHS_NAMES[i]] = addnoise_le(_RHS_BASE[i], rng)
            data[name] = rhs
    return data

SCENARIO_DATA = _precompute_scenario_data()
scenarios = [f"scen_{i}" for i in range(NUM_SCENARIOS)]


# ============================================================================
# Scenario model builder
# ============================================================================

def build_scenario_model(scenario_name):
    """
    Build a Pyomo model for the given scenario.
    Returns [model, first_stage_vars_dict, probability].
    """
    rhs = SCENARIO_DATA[scenario_name]
    m = pyo.ConcreteModel()

    # Variables: all xi >= 0 (matching Julia: @variable(m, xi >= 0))
    for i in range(1, 21):
        setattr(m, f'x{i}', pyo.Var(bounds=(0, None), initialize=0))

    x = [getattr(m, f'x{i}') for i in range(1, 21)]

    # Constraints (10 <= constraints, identical to Julia)
    m.c1  = pyo.Constraint(expr=-3*x[0] + 7*x[1] - 5*x[3] + x[4] + x[5] + 2*x[7] - x[8] - x[9] - 9*x[10] + 3*x[11] + 5*x[12] + x[15] + 7*x[16] - 7*x[17] - 4*x[18] - 6*x[19] <= rhs["c1_rhs"])
    m.c2  = pyo.Constraint(expr=7*x[0] - 5*x[2] + x[3] + x[4] + 2*x[6] - x[7] - x[8] - 9*x[9] + 3*x[10] + 5*x[11] + x[14] + 7*x[15] - 7*x[16] - 4*x[17] - 6*x[18] - 3*x[19] <= rhs["c2_rhs"])
    m.c3  = pyo.Constraint(expr=-5*x[1] + x[2] + x[3] + 2*x[5] - x[6] - x[7] - 9*x[8] + 3*x[9] + 5*x[10] + x[13] + 7*x[14] - 7*x[15] - 4*x[16] - 6*x[17] - 3*x[18] + 7*x[19] <= rhs["c3_rhs"])
    m.c4  = pyo.Constraint(expr=-5*x[0] + x[1] + x[2] + 2*x[4] - x[5] - x[6] - 9*x[7] + 3*x[8] + 5*x[9] + x[12] + 7*x[13] - 7*x[14] - 4*x[15] - 6*x[16] - 3*x[17] + 7*x[18] <= rhs["c4_rhs"])
    m.c5  = pyo.Constraint(expr=x[0] + x[1] + 2*x[3] - x[4] - x[5] - 9*x[6] + 3*x[7] + 5*x[8] + x[11] + 7*x[12] - 7*x[13] - 4*x[14] - 6*x[15] - 3*x[16] + 7*x[17] - 5*x[19] <= rhs["c5_rhs"])
    m.c6  = pyo.Constraint(expr=x[0] + 2*x[2] - x[3] - x[4] - 9*x[5] + 3*x[6] + 5*x[7] + x[10] + 7*x[11] - 7*x[12] - 4*x[13] - 6*x[14] - 3*x[15] + 7*x[16] - 5*x[18] + x[19] <= 4)
    m.c7  = pyo.Constraint(expr=2*x[1] - x[2] - x[3] - 9*x[4] + 3*x[5] + 5*x[6] + x[9] + 7*x[10] - 7*x[11] - 4*x[12] - 6*x[13] - 3*x[14] + 7*x[15] - 5*x[17] + x[18] + x[19] <= -1)
    m.c8  = pyo.Constraint(expr=2*x[0] - x[1] - x[2] - 9*x[3] + 3*x[4] + 5*x[5] + x[8] + 7*x[9] - 7*x[10] - 4*x[11] - 6*x[12] - 3*x[13] + 7*x[14] - 5*x[16] + x[17] + x[18] <= 0)
    m.c9  = pyo.Constraint(expr=-x[0] - x[1] - 9*x[2] + 3*x[3] + 5*x[4] + x[7] + 7*x[8] - 7*x[9] - 4*x[10] - 6*x[11] - 3*x[12] + 7*x[13] - 5*x[15] + x[16] + x[17] + 2*x[19] <= 9)
    m.c10 = pyo.Constraint(expr=sum(x) <= 40)

    # Objective: Min -0.5 * sum(i*(xi-2)^2, i=1..20)
    m.obj = pyo.Objective(expr=-0.5 * sum((i + 1) * (x[i] - 2)**2 for i in range(20)),
                          sense=pyo.minimize)

    # First-stage variables: x1..x5
    first_stage = {f"x{i}": getattr(m, f"x{i}") for i in range(1, NFIRST + 1)}
    probability = 1.0 / NUM_SCENARIOS

    return [m, first_stage, probability]


# ============================================================================
# Main
# ============================================================================

if __name__ == '__main__':
    # Solvers
    lb_solver = pyo.SolverFactory("gurobi")
    lb_solver.options["NonConvex"] = 2
    lb_solver.options["MIPGap"] = 1e-2
    lb_solver.options["TimeLimit"] = 15

    cg_solver = pyo.SolverFactory("gurobi")
    cg_solver.options["NonConvex"] = 2
    cg_solver.options["TimeLimit"] = 15

    ub_solver = pyo.SolverFactory("gurobi")
    ub_solver.options["NonConvex"] = 2
    ub_solver.options["TimeLimit"] = 15

    # Solver parameters
    _script_dir = os.path.dirname(os.path.abspath(__file__))
    _log_dir = os.path.join(_script_dir, "logs")
    os.makedirs(_log_dir, exist_ok=True)

    params = sno.SolverParameters(subproblem_names=scenarios,
                                  subproblem_creator=build_scenario_model,
                                  lb_solver=lb_solver,
                                  cg_solver=cg_solver,
                                  ub_solver=ub_solver)
    params.set_bounders(candidate_solution_finder=sno.SolveExtensiveForm)
    params.guarantee_global_convergence()
    params.set_bounds_tightening(fbbt=True, obbt=True)
    params.activate_verbose()
    if (size == 1):
        params.set_logging(fname=os.path.join(_log_dir, "problem_2_1_7_log"))
    else:
        params.set_logging(fname=os.path.join(_log_dir, "problem_2_1_7_log_parallel"))
    if (rank == 0):
        params.display()

    # Solve
    solver = sno.Solver(params)

    # ---------------------------------------------------------
    # CSV Logging Implementation (Monkey Patch)
    # ---------------------------------------------------------
    csv_filename = os.path.join(_log_dir, "problem_2_1_7_result.csv")
    csv_header = ["Time (s)", "Nodes Explored", "Pruned by", "Bound Update", "LB", "UB", "Rel. Gap", "Abs. Gap", "# Nodes"]
    if rank == 0:
        with open(csv_filename, mode='w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(csv_header)

    original_display_status = solver.display_status

    def csv_logging_display_status(bnb_result):
        original_display_status(bnb_result)
        if rank == 0:
            pruned = " "
            if "pruned by bound" in bnb_result:
                pruned = "Bound"
            elif "pruned by infeasibility" in bnb_result:
                pruned = "Infeas."
            bound_update = " "
            if "ublb" in bnb_result:
                bound_update = "* L U"
            elif "ub" in bnb_result:
                bound_update = "* U  "
            elif "lb" in bnb_result:
                bound_update = "* L  "
            row = [
                round(solver.runtime, 3),
                solver.tree.metrics.nodes.explored,
                pruned,
                bound_update,
                f"{solver.tree.metrics.lb:.8}",
                f"{solver.tree.metrics.ub:.8}",
                f"{round(solver.tree.metrics.relative_gap*100, 4)}%",
                round(solver.tree.metrics.absolute_gap, 6),
                solver.tree.n_nodes()
            ]
            with open(csv_filename, mode='a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(row)

    solver.display_status = csv_logging_display_status
    # ---------------------------------------------------------

    solver.solve(max_iter=9000,
                 rel_tolerance=1e-4,
                 abs_tolerance=1e-8,
                 time_limit=60 * 60 * 12)

    # Print solution
    if (rank == 0):
        print("\n====================================================================")
        print("SOLUTION")
        print(f"Obj: {solver.tree.metrics.ub}")
        for name in solver.subproblems.names:
            print(f"subproblem = {name}")
            for var_name in solver.solution.subproblem_solutions[name]:
                var_val = solver.solution.subproblem_solutions[name][var_name]
                print(f"  var name = {var_name}, value = {round(var_val, 5)}")
            print()
        print("====================================================================")
