"""
singo_2_1_1.py — Snoglode example for Julia SINGO Global/2_1_1

Julia reference:
  P = RandomStochasticModel(createModel, 1000, 2, 2)
  srand(1234), nscen=1000, nfirst=2, nparam=2

Stage split (from Julia nfirst=2, column order):
  First-stage:  x1 (col 1), x2 (col 2)
  Second-stage: x3 (col 3), x4 (col 4), x5 (col 5)

Stochastic perturbation:
  Only constraint: 20x1+12x2+11x3+7x4+4x5 <= 40
  This is linconstr[1], involves second-stage vars (x3,x4,x5).
  With nparam=2: 1 modification per scenario.
  RHS perturbed: addnoise(40.0) -> 40.0 * U(0.5, 2.0)

Objective:
  Min 42*x1 + 44*x2 + 45*x3 + 47*x4 + 47.5*x5
      - 50*(x1^2 + x2^2 + x3^2 + x4^2 + x5^2)

Verification at x=(0,...,0): obj=0, constraint=0<=40 (feasible)
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
# Julia-compatible RNG (MT19937 seeded with 1234)
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
    """Plasmo.jl addnoise: if a==0 -> a+U(-10,10); else -> a*U(0.5,2.0)"""
    if a == 0.0:
        return a + rng.uniform(-10.0, 10.0)
    return a * rng.uniform(0.5, 2.0)


# =============================================================================
# Pre-compute scenario RHS values (matches Julia srand(1234))
# =============================================================================

num_scenarios = 1000

_rng = JuliaMT19937(1234)
scenario_c1_rhs = [addnoise(40.0, _rng) for _ in range(num_scenarios)]
# Each scenario consumes exactly 1 RNG draw: 40.0 * U(0.5, 2.0)


# =============================================================================
# Subproblem creator (Snoglode API)
# =============================================================================

def subproblem_creator(scenario_name):
    """
    Build one scenario subproblem for SINGO 2_1_1.

    Returns [pyomo_model, first_stage_vars_dict, probability]
    """
    scen_idx = int(scenario_name.split("_")[1])

    m = pyo.ConcreteModel()

    # Variables (Julia: all in [0,1])
    m.x1 = pyo.Var(bounds=(0, 1), initialize=1)    # col 1, start=1
    m.x2 = pyo.Var(bounds=(0, 1), initialize=1)    # col 2, start=1
    m.x3 = pyo.Var(bounds=(0, 1), initialize=0)    # col 3
    m.x4 = pyo.Var(bounds=(0, 1), initialize=1)    # col 4, start=1
    m.x5 = pyo.Var(bounds=(0, 1), initialize=0)    # col 5

    # Constraint (RHS perturbed per scenario)
    rhs = scenario_c1_rhs[scen_idx]
    m.c1 = pyo.Constraint(expr=20*m.x1 + 12*m.x2 + 11*m.x3 + 7*m.x4 + 4*m.x5 <= rhs)

    # Objective: 42x1+44x2+45x3+47x4+47.5x5 - 50*(x1^2+...+x5^2)
    lin = 42*m.x1 + 44*m.x2 + 45*m.x3 + 47*m.x4 + 47.5*m.x5
    quad = 50*(m.x1**2 + m.x2**2 + m.x3**2 + m.x4**2 + m.x5**2)
    m.obj = pyo.Objective(expr=lin - quad, sense=pyo.minimize)

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
        params.set_logging(fname=os.getcwd() + "/logs/singo_2_1_1_log")
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
        for n in solver.subproblems.names[:3]:  # print first 3 scenarios
            print(f"subproblem = {n}")
            for vn in solver.solution.subproblem_solutions[n]:
                var_val = solver.solution.subproblem_solutions[n][vn]
                print(f"  {vn} = {round(var_val, 6)}")
            print()
        print("=" * 68)
