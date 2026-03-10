"""
singo_14_1_6.py — Snoglode example for Julia SINGO Global/14_1_6

Julia reference:
  P = RandomStochasticModel(createModel, 1000, 2, 2)
  srand(1234), nscen=1000, nfirst=2, nparam=2

Stage split (nfirst=2, JuMP column order):
  First-stage:  x1 (col 1), x2 (col 2)   — both in [-1, 1]
  Second-stage: x3..x8 (cols 3-8) in [-1,1], x9 (col 9) free

Constraint categorization in JuMP (old, pre-0.19):
  linconstr:  c7 only  (-0.7623*x1+0.2238*x2 == -0.3461, all first-stage -> SKIP)
  quadconstr: c1-c6 (bilinear), c8-c15 (quadratic xi^2)

Stochastic perturbation:
  Linear scan: c7 has only first-stage vars -> skip. nmodified=0 < nparam=2.
  Quadratic scan: c1 (quadconstr[1]) has 2nd-stage vars -> eligible.
    nmodified=1, 1 < 2 -> MODIFY: aff.constant = addnoise(-0.3571)
    Julia stores "<= ub" as "body - ub <= 0", so aff.constant = -0.3571
    addnoise(-0.3571) = -0.3571 * U(0.5, 2.0)
    Effective RHS: body <= 0.3571 * U(0.5, 2.0)
  In Pyomo: set c1 RHS = addnoise_julia(0.3571, rng) = 0.3571 * U(0.5, 2.0)
    (same RNG draw consumed, same effective constraint)

Objective:
  Min x9
"""
import os
import sys
import pyomo.environ as pyo

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))
import snoglode as sno
import snoglode.utils.MPI as MPI
rank = MPI.COMM_WORLD.Get_rank()
size = MPI.COMM_WORLD.Get_size()


# =============================================================================
# Julia-compatible RNG
# =============================================================================

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
            self.mt[i] = (1812433253 * (self.mt[i-1] ^ (self.mt[i-1] >> 30)) + i) & 0xFFFFFFFF
        self.mti = self.N

    def _uint32(self) -> int:
        mag01 = [0x0, self.MATRIX_A]
        if self.mti >= self.N:
            for kk in range(self.N - self.M):
                y = (self.mt[kk] & self.UPPER_MASK) | (self.mt[kk+1] & self.LOWER_MASK)
                self.mt[kk] = self.mt[kk + self.M] ^ (y >> 1) ^ mag01[y & 0x1]
            for kk in range(self.N - self.M, self.N - 1):
                y = (self.mt[kk] & self.UPPER_MASK) | (self.mt[kk+1] & self.LOWER_MASK)
                self.mt[kk] = self.mt[kk + (self.M - self.N)] ^ (y >> 1) ^ mag01[y & 0x1]
            y = (self.mt[self.N-1] & self.UPPER_MASK) | (self.mt[0] & self.LOWER_MASK)
            self.mt[self.N-1] = self.mt[self.M-1] ^ (y >> 1) ^ mag01[y & 0x1]
            self.mti = 0
        y = self.mt[self.mti]; self.mti += 1
        y ^= (y >> 11); y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000; y ^= (y >> 18)
        return y & 0xFFFFFFFF

    def _uint64(self) -> int:
        return ((self._uint32() << 32) | self._uint32()) & 0xFFFFFFFFFFFFFFFF

    def _float64(self) -> float:
        return (self._uint64() >> 12) * (1.0 / (1 << 52))

    def uniform(self, a: float, b: float) -> float:
        return a + (b - a) * self._float64()


def addnoise(a: float, rng: JuliaMT19937) -> float:
    if a == 0.0:
        return a + rng.uniform(-10.0, 10.0)
    return a * rng.uniform(0.5, 2.0)


# =============================================================================
# Pre-compute scenario RHS values
# Julia: addnoise(-0.3571) on aff.constant; Pyomo equivalent: RHS = 0.3571 * U(0.5,2)
# We compute addnoise(0.3571, rng) which consumes the same RNG draw and equals |addnoise(-0.3571)|.
# =============================================================================

num_scenarios = 1000

_rng = JuliaMT19937(1234)
scenario_c1_rhs = [addnoise(0.3571, _rng) for _ in range(num_scenarios)]


# =============================================================================
# Subproblem creator
# =============================================================================

def subproblem_creator(scenario_name):
    scen_idx = int(scenario_name.split("_")[1])

    m = pyo.ConcreteModel()

    # Variables (match Julia exactly)
    m.x1 = pyo.Var(bounds=(-1, 1))    # col 1, first-stage
    m.x2 = pyo.Var(bounds=(-1, 1))    # col 2, first-stage
    m.x3 = pyo.Var(bounds=(-1, 1))    # col 3
    m.x4 = pyo.Var(bounds=(-1, 1))    # col 4
    m.x5 = pyo.Var(bounds=(-1, 1))    # col 5
    m.x6 = pyo.Var(bounds=(-1, 1))    # col 6
    m.x7 = pyo.Var(bounds=(-1, 1))    # col 7
    m.x8 = pyo.Var(bounds=(-1, 1))    # col 8
    m.x9 = pyo.Var()                  # col 9, free (no bounds in Julia)

    # Constraints (c1 RHS is stochastic)
    c1_rhs = scenario_c1_rhs[scen_idx]

    m.c1  = pyo.Constraint(expr=0.004731*m.x1*m.x3 - 0.1238*m.x1 - 0.3578*m.x2*m.x3 - 0.001637*m.x2 - 0.9338*m.x4 + m.x7 - m.x9 <= c1_rhs)
    m.c2  = pyo.Constraint(expr=0.1238*m.x1 - 0.004731*m.x1*m.x3 + 0.3578*m.x2*m.x3 + 0.001637*m.x2 + 0.9338*m.x4 - m.x7 - m.x9 <= -0.3571)
    m.c3  = pyo.Constraint(expr=0.2238*m.x1*m.x3 + 0.2638*m.x1 + 0.7623*m.x2*m.x3 - 0.07745*m.x2 - 0.6734*m.x4 - m.x7 - m.x9 <= 0.6022)
    m.c4  = pyo.Constraint(expr=-0.2238*m.x1*m.x3 - 0.2638*m.x1 - 0.7623*m.x2*m.x3 + 0.07745*m.x2 + 0.6734*m.x4 + m.x7 - m.x9 <= -0.6022)
    m.c5  = pyo.Constraint(expr=m.x6*m.x8 + 0.3578*m.x1 + 0.004731*m.x2 - m.x9 <= 0.0)
    m.c6  = pyo.Constraint(expr=-m.x6*m.x8 - 0.3578*m.x1 - 0.004731*m.x2 - m.x9 <= 0.0)
    m.c7  = pyo.Constraint(expr=-0.7623*m.x1 + 0.2238*m.x2 == -0.3461)
    m.c8  = pyo.Constraint(expr=m.x1**2 + m.x2**2 - m.x9 <= 1.0)
    m.c9  = pyo.Constraint(expr=-(m.x1**2) - (m.x2**2) - m.x9 <= -1.0)
    m.c10 = pyo.Constraint(expr=m.x3**2 + m.x4**2 - m.x9 <= 1.0)
    m.c11 = pyo.Constraint(expr=-(m.x3**2) - (m.x4**2) - m.x9 <= -1.0)
    m.c12 = pyo.Constraint(expr=m.x5**2 + m.x6**2 - m.x9 <= 1.0)
    m.c13 = pyo.Constraint(expr=-(m.x5**2) - (m.x6**2) - m.x9 <= -1.0)
    m.c14 = pyo.Constraint(expr=m.x7**2 + m.x8**2 - m.x9 <= 1.0)
    m.c15 = pyo.Constraint(expr=-(m.x7**2) - (m.x8**2) - m.x9 <= -1.0)

    # Objective: Min x9
    m.obj = pyo.Objective(expr=m.x9, sense=pyo.minimize)

    first_stage = {"x1": m.x1, "x2": m.x2}
    probability = 1.0 / num_scenarios

    return [m, first_stage, probability]


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    scenarios = [f"scen_{i}" for i in range(num_scenarios)]

    lb_solver = pyo.SolverFactory("gurobi")
    lb_solver.options["NonConvex"] = 2
    lb_solver.options["MIPGap"] = 1e-1
    cg_solver = pyo.SolverFactory("gurobi")
    cg_solver.options["NonConvex"] = 2
    ub_solver = pyo.SolverFactory("gurobi")
    ub_solver.options["NonConvex"] = 2

    params = sno.SolverParameters(
        subproblem_names=scenarios,
        subproblem_creator=subproblem_creator,
        lb_solver=lb_solver,
        cg_solver=cg_solver,
        ub_solver=ub_solver,
    )
    params.set_bounders(candidate_solution_finder=sno.SolveExtensiveForm)
    params.set_bounds_tightening(fbbt=True, obbt=True)
    params.activate_verbose()
    if size == 1:
        os.makedirs(os.getcwd() + "/logs", exist_ok=True)
        params.set_logging(fname=os.getcwd() + "/logs/singo_14_1_6_log")
    if rank == 0:
        params.display()

    solver = sno.Solver(params)
    solver.solve(max_iter=500,
                 rel_tolerance=1e-2,
                 abs_tolerance=1e-6,
                 time_limit=43200)

    if rank == 0:
        print("\n" + "=" * 68)
        print("SOLUTION")
        print(f"Obj: {solver.tree.metrics.ub}")
        for n in solver.subproblems.names[:3]:
            print(f"subproblem = {n}")
            for vn in solver.solution.subproblem_solutions[n]:
                var_val = solver.solution.subproblem_solutions[n][vn]
                print(f"  {vn} = {round(var_val, 6)}")
            print()
        print("=" * 68)
