"""
singo_2_1_3.py — Snoglode example for Julia SINGO Global/2_1_3

Julia reference:
  P = RandomStochasticModel(createModel, 1000, 2, 2)
  srand(1234), nscen=1000, nfirst=2, nparam=2

Stage split (nfirst=2, JuMP column order):
  First-stage:  x1 (col 1), x2 (col 2)
  Second-stage: x3..x13 (cols 3..13)

Stochastic perturbation:
  linconstr[1]: 2x1+2x2+x10+x11 <= 10  (has 2nd-stage x10,x11)
  -> nmodified=1 < nparam=2 -> MODIFY: addnoise(10.0) -> 10 * U(0.5, 2.0)
  linconstr[2]: 2x1+2x3+x10+x12 <= 10  -> nmodified=2 >= 2 -> break

Objective:
  Min 5x1+5x2+5x3+5x4 - 5*(x1^2+x2^2+x3^2+x4^2) - x5-x6-x7-x8-x9-x10-x11-x12-x13

Verification at x=(0,...,0): obj=0, all constraints satisfied.
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
# =============================================================================

num_scenarios = 1000

_rng = JuliaMT19937(1234)
scenario_c1_rhs = [addnoise(10.0, _rng) for _ in range(num_scenarios)]


# =============================================================================
# Subproblem creator
# =============================================================================

def subproblem_creator(scenario_name):
    scen_idx = int(scenario_name.split("_")[1])

    m = pyo.ConcreteModel()

    # Variables (match Julia: x1-x9 in [0,1] start=1; x10-x12 >= 0 start=3; x13 in [0,1] start=1)
    m.x1  = pyo.Var(bounds=(0, 1), initialize=1)
    m.x2  = pyo.Var(bounds=(0, 1), initialize=1)
    m.x3  = pyo.Var(bounds=(0, 1), initialize=1)
    m.x4  = pyo.Var(bounds=(0, 1), initialize=1)
    m.x5  = pyo.Var(bounds=(0, 1), initialize=1)
    m.x6  = pyo.Var(bounds=(0, 1), initialize=1)
    m.x7  = pyo.Var(bounds=(0, 1), initialize=1)
    m.x8  = pyo.Var(bounds=(0, 1), initialize=1)
    m.x9  = pyo.Var(bounds=(0, 1), initialize=1)
    m.x10 = pyo.Var(bounds=(0, None), initialize=3)
    m.x11 = pyo.Var(bounds=(0, None), initialize=3)
    m.x12 = pyo.Var(bounds=(0, None), initialize=3)
    m.x13 = pyo.Var(bounds=(0, 1), initialize=1)

    # Constraints (c1 RHS is stochastic)
    rhs = scenario_c1_rhs[scen_idx]
    m.c1 = pyo.Constraint(expr=2*m.x1 + 2*m.x2 + m.x10 + m.x11 <= rhs)
    m.c2 = pyo.Constraint(expr=2*m.x1 + 2*m.x3 + m.x10 + m.x12 <= 10)
    m.c3 = pyo.Constraint(expr=2*m.x2 + 2*m.x3 + m.x11 + m.x12 <= 10)
    m.c4 = pyo.Constraint(expr=-8*m.x1 + m.x10 <= 0)
    m.c5 = pyo.Constraint(expr=-8*m.x2 + m.x11 <= 0)
    m.c6 = pyo.Constraint(expr=-8*m.x3 + m.x12 <= 0)
    m.c7 = pyo.Constraint(expr=-2*m.x4 - m.x5 + m.x10 <= 0)
    m.c8 = pyo.Constraint(expr=-2*m.x6 - m.x7 + m.x11 <= 0)
    m.c9 = pyo.Constraint(expr=-2*m.x8 - m.x9 + m.x12 <= 0)

    # Objective: 5(x1+x2+x3+x4) - 5(x1^2+x2^2+x3^2+x4^2) - (x5+...+x13)
    quad = 5.0 * (m.x1**2 + m.x2**2 + m.x3**2 + m.x4**2)
    m.obj = pyo.Objective(
        expr=5*m.x1 + 5*m.x2 + 5*m.x3 + 5*m.x4 - quad
             - (m.x5 + m.x6 + m.x7 + m.x8 + m.x9 + m.x10 + m.x11 + m.x12 + m.x13),
        sense=pyo.minimize)

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
        params.set_logging(fname=os.getcwd() + "/logs/singo_2_1_3_log")
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
