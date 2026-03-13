"""
problem_2_1_8.py -- Snoglode solver for SNGO-master/Global/2_1_8

Julia reference (PlasmoOld):
  RandomStochasticModel(createModel, NS=100) => nscen=100, nfirst=5, nparam=5
  srand(1234), scenario 1 unperturbed

Model: 24 variables (0 <= xi <= 100), 10 equality constraints,
  concave quadratic objective: Min sum(lin_i*xi - quad_i*xi^2)

Stage split (nfirst=5):
  First-stage:  x1..x5
  Second-stage: x6..x24

Stochastic perturbation (PlasmoOld equality constraint splitting):
  c1: x1+x2+x3+x4 == 8       (all first-stage -> SKIP)
  c2: x5+x6+x7+x8 == 24      (has 2nd-stage -> modified)
  c3: x9+x10+x11+x12 == 20   (all 2nd-stage -> modified)
  c4: x13+x14+x15+x16 == 24  (all 2nd-stage -> modified)
  c5: x17+x18+x19+x20 == 16  (all 2nd-stage -> modified)
  c6: x21+x22+x23+x24 == 12  (all 2nd-stage -> modified, nmodified=5 -> break)
  c7-c10: not reached

  Each equality (lb==ub) is split into:
    <= constraint: ub_new = addnoise_le(ub)
    >= constraint: lb_new = addnoise_ge(lb)
  Each equality consumes 2 RNG draws.
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

# Objective coefficients from Julia: lin_coeff*xi - quad_coeff*xi^2
OBJ_LIN  = [300, 270, 460, 800, 740, 600, 540, 380, 300, 490,
            380, 760, 430, 250, 390, 600, 210, 830, 470, 680, 360, 290, 400, 310]
OBJ_QUAD = [7, 4, 6, 8, 12, 9, 14, 7, 13, 12,
            8, 4, 7, 9, 16, 8, 4, 10, 21, 13, 17, 9, 8, 4]

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
    """PlasmoOld addnoise for <= / upper bound: addnoise(a, 0, 10, 0, 2)"""
    if a == 0.0:
        return a + rng.rand_uniform(0.0, 10.0)
    return a + abs(a) * rng.rand_uniform(0.0, 2.0)


def addnoise_ge(a: float, rng: JuliaMT19937) -> float:
    """PlasmoOld addnoise for >= / lower bound: addnoise(a, -10, 0, -2, 0)"""
    if a == 0.0:
        return a + rng.rand_uniform(-10.0, 0.0)
    return a + abs(a) * rng.rand_uniform(-2.0, 0.0)


# ============================================================================
# Pre-compute scenario data
# ============================================================================
# Base RHS values for c2-c6 equality constraints
_BASE_EQ = [24.0, 20.0, 24.0, 16.0, 12.0]  # c2, c3, c4, c5, c6


def _precompute_scenario_data():
    """Pre-compute perturbed bounds for all scenarios."""
    rng = JuliaMT19937(SEED)
    data = {}
    for s in range(NUM_SCENARIOS):
        name = f"scen_{s}"
        if s == 0:
            # Unperturbed: equalities kept as ==
            data[name] = {
                "c2_ub": 24.0, "c2_lb": 24.0,
                "c3_ub": 20.0, "c3_lb": 20.0,
                "c4_ub": 24.0, "c4_lb": 24.0,
                "c5_ub": 16.0, "c5_lb": 16.0,
                "c6_ub": 12.0, "c6_lb": 12.0,
            }
        else:
            # PlasmoOld: for each == constraint, UB noise then LB noise (in order)
            d = {}
            for i, val in enumerate(_BASE_EQ):
                cname = f"c{i + 2}"
                d[f"{cname}_ub"] = addnoise_le(val, rng)
                d[f"{cname}_lb"] = addnoise_ge(val, rng)
            data[name] = d
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
    sd = SCENARIO_DATA[scenario_name]
    m = pyo.ConcreteModel()

    # Variables: all in [0, 100] (matching Julia: @variable(m, 0<=xi<=100))
    for i in range(1, 25):
        setattr(m, f'x{i}', pyo.Var(bounds=(0, 100), initialize=0))

    xs = [getattr(m, f'x{i}') for i in range(1, 25)]

    # c1: x1+x2+x3+x4 == 8 (all first-stage, never modified)
    m.c1 = pyo.Constraint(expr=xs[0] + xs[1] + xs[2] + xs[3] == 8)

    # c2-c6: equality constraints split into <= and >= with perturbed bounds
    m.c2_le = pyo.Constraint(expr=xs[4] + xs[5] + xs[6] + xs[7] <= sd["c2_ub"])
    m.c2_ge = pyo.Constraint(expr=xs[4] + xs[5] + xs[6] + xs[7] >= sd["c2_lb"])
    m.c3_le = pyo.Constraint(expr=xs[8] + xs[9] + xs[10] + xs[11] <= sd["c3_ub"])
    m.c3_ge = pyo.Constraint(expr=xs[8] + xs[9] + xs[10] + xs[11] >= sd["c3_lb"])
    m.c4_le = pyo.Constraint(expr=xs[12] + xs[13] + xs[14] + xs[15] <= sd["c4_ub"])
    m.c4_ge = pyo.Constraint(expr=xs[12] + xs[13] + xs[14] + xs[15] >= sd["c4_lb"])
    m.c5_le = pyo.Constraint(expr=xs[16] + xs[17] + xs[18] + xs[19] <= sd["c5_ub"])
    m.c5_ge = pyo.Constraint(expr=xs[16] + xs[17] + xs[18] + xs[19] >= sd["c5_lb"])
    m.c6_le = pyo.Constraint(expr=xs[20] + xs[21] + xs[22] + xs[23] <= sd["c6_ub"])
    m.c6_ge = pyo.Constraint(expr=xs[20] + xs[21] + xs[22] + xs[23] >= sd["c6_lb"])

    # c7-c10: deterministic equalities (unchanged)
    m.c7  = pyo.Constraint(expr=xs[0] + xs[4]  + xs[8]  + xs[12] + xs[16] + xs[20] == 29)
    m.c8  = pyo.Constraint(expr=xs[1] + xs[5]  + xs[9]  + xs[13] + xs[17] + xs[21] == 41)
    m.c9  = pyo.Constraint(expr=xs[2] + xs[6]  + xs[10] + xs[14] + xs[18] + xs[22] == 13)
    m.c10 = pyo.Constraint(expr=xs[3] + xs[7]  + xs[11] + xs[15] + xs[19] + xs[23] == 21)

    # Objective: Min sum(lin_i*xi - quad_i*xi^2)
    m.obj = pyo.Objective(
        expr=sum(OBJ_LIN[i] * xs[i] - OBJ_QUAD[i] * xs[i]**2 for i in range(24)),
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
    lb_solver.options["TimeLimit"] = 60

    cg_solver = pyo.SolverFactory("gurobi")
    cg_solver.options["NonConvex"] = 2
    cg_solver.options["TimeLimit"] = 60

    ub_solver = pyo.SolverFactory("gurobi")
    ub_solver.options["NonConvex"] = 2
    ub_solver.options["TimeLimit"] = 60

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
        params.set_logging(fname=os.path.join(_log_dir, "problem_2_1_8_log"))
    else:
        params.set_logging(fname=os.path.join(_log_dir, "problem_2_1_8_log_parallel"))
    if (rank == 0):
        params.display()

    # Solve
    solver = sno.Solver(params)

    # ---------------------------------------------------------
    # CSV Logging Implementation (Monkey Patch)
    # ---------------------------------------------------------
    csv_filename = os.path.join(_log_dir, "problem_2_1_8_result.csv")
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
