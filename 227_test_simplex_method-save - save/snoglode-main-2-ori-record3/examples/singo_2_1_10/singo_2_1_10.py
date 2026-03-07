"""
singo_2_1_10.py — Snoglode example for Julia SINGO Global/2_1_10

Julia reference:
  P = RandomStochasticModel(createModel, 1000, 2, 2)
  srand(1234), nscen=1000, nfirst=2, nparam=2

Stage split (nfirst=2, JuMP column order):
  First-stage:  x1 (col 1), x2 (col 2)
  Second-stage: x3..x20 (cols 3..20)

Variable bounds (Julia):
  All xi >= 0 (no upper bound in Julia). x6 start=4.348, x14 start=62.609.
  NOTE: x1,x2 bounded to [0,200] for Snoglode B&B (implied by constraint c10: sum <= 200).

Stochastic perturbation:
  linconstr[1] (c1): 3x1+5x2+5x3+...+3x20 <= 380  (has 2nd-stage vars)
  -> nmodified=1 < nparam=2 -> MODIFY: addnoise(380.0) -> 380 * U(0.5, 2.0)
  linconstr[2]: nmodified=2 >= 2 -> break

Objective:
  Min 0.5*[42(52+x11)^2 + 98(3+x12)^2 + 48(x13-81)^2 + 91(x14-30)^2
         + 11(85+x15)^2 + 63(x16-68)^2 + 61(x17-27)^2 + 61(81+x18)^2
         + 38(x19-97)^2 + 26(73+x20)^2]
    - 0.5*[63(19+x1)^2 + 15(27+x2)^2 + 44(23+x3)^2 + 91(53+x4)^2
         + 45(42+x5)^2 + 50(x6-26)^2 + 89(33+x7)^2 + 58(23+x8)^2
         + 86(x9-41)^2 + 82(x10-19)^2]
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
scenario_c1_rhs = [addnoise(380.0, _rng) for _ in range(num_scenarios)]


# =============================================================================
# Subproblem creator
# =============================================================================

def subproblem_creator(scenario_name):
    scen_idx = int(scenario_name.split("_")[1])

    m = pyo.ConcreteModel()

    # Variables (Julia: all >= 0, no upper bound except implicit from constraints)
    # x1,x2 bounded [0,200] for Snoglode B&B (from c10: sum xi <= 200)
    m.x1  = pyo.Var(bounds=(0, 200), initialize=0)
    m.x2  = pyo.Var(bounds=(0, 200), initialize=0)
    m.x3  = pyo.Var(bounds=(0, None), initialize=0)
    m.x4  = pyo.Var(bounds=(0, None), initialize=0)
    m.x5  = pyo.Var(bounds=(0, None), initialize=0)
    m.x6  = pyo.Var(bounds=(0, None), initialize=4.348)    # start=4.348
    m.x7  = pyo.Var(bounds=(0, None), initialize=0)
    m.x8  = pyo.Var(bounds=(0, None), initialize=0)
    m.x9  = pyo.Var(bounds=(0, None), initialize=0)
    m.x10 = pyo.Var(bounds=(0, None), initialize=0)
    m.x11 = pyo.Var(bounds=(0, None), initialize=0)
    m.x12 = pyo.Var(bounds=(0, None), initialize=0)
    m.x13 = pyo.Var(bounds=(0, None), initialize=0)
    m.x14 = pyo.Var(bounds=(0, None), initialize=62.609)   # start=62.609
    m.x15 = pyo.Var(bounds=(0, None), initialize=0)
    m.x16 = pyo.Var(bounds=(0, None), initialize=0)
    m.x17 = pyo.Var(bounds=(0, None), initialize=0)
    m.x18 = pyo.Var(bounds=(0, None), initialize=0)
    m.x19 = pyo.Var(bounds=(0, None), initialize=0)
    m.x20 = pyo.Var(bounds=(0, None), initialize=0)

    # Constraints (c1 RHS is stochastic, c2-c10 deterministic)
    rhs = scenario_c1_rhs[scen_idx]
    m.c1  = pyo.Constraint(expr=3*m.x1+5*m.x2+5*m.x3+6*m.x4+4*m.x5+4*m.x6+5*m.x7+6*m.x8+4*m.x9+4*m.x10+8*m.x11+4*m.x12+2*m.x13+1*m.x14+1*m.x15+1*m.x16+2*m.x17+1*m.x18+7*m.x19+3*m.x20 <= rhs)
    m.c2  = pyo.Constraint(expr=5*m.x1+4*m.x2+5*m.x3+4*m.x4+1*m.x5+4*m.x6+4*m.x7+2*m.x8+5*m.x9+2*m.x10+3*m.x11+6*m.x12+1*m.x13+7*m.x14+7*m.x15+5*m.x16+8*m.x17+7*m.x18+2*m.x19+1*m.x20 <= 415)
    m.c3  = pyo.Constraint(expr=1*m.x1+5*m.x2+2*m.x3+4*m.x4+7*m.x5+3*m.x6+1*m.x7+5*m.x8+7*m.x9+6*m.x10+1*m.x11+7*m.x12+2*m.x13+4*m.x14+7*m.x15+5*m.x16+3*m.x17+4*m.x18+1*m.x19+2*m.x20 <= 385)
    m.c4  = pyo.Constraint(expr=3*m.x1+2*m.x2+6*m.x3+3*m.x4+2*m.x5+1*m.x6+6*m.x7+1*m.x8+7*m.x9+3*m.x10+7*m.x11+7*m.x12+8*m.x13+2*m.x14+3*m.x15+4*m.x16+5*m.x17+8*m.x18+1*m.x19+2*m.x20 <= 405)
    m.c5  = pyo.Constraint(expr=6*m.x1+6*m.x2+6*m.x3+4*m.x4+5*m.x5+2*m.x6+2*m.x7+4*m.x8+3*m.x9+2*m.x10+7*m.x11+5*m.x12+3*m.x13+6*m.x14+7*m.x15+5*m.x16+8*m.x17+4*m.x18+6*m.x19+3*m.x20 <= 470)
    m.c6  = pyo.Constraint(expr=5*m.x1+5*m.x2+2*m.x3+1*m.x4+3*m.x5+5*m.x6+5*m.x7+7*m.x8+4*m.x9+3*m.x10+4*m.x11+1*m.x12+7*m.x13+3*m.x14+8*m.x15+3*m.x16+1*m.x17+6*m.x18+2*m.x19+8*m.x20 <= 415)
    m.c7  = pyo.Constraint(expr=3*m.x1+6*m.x2+6*m.x3+3*m.x4+1*m.x5+6*m.x6+1*m.x7+6*m.x8+7*m.x9+1*m.x10+4*m.x11+3*m.x12+1*m.x13+4*m.x14+3*m.x15+6*m.x16+4*m.x17+6*m.x18+5*m.x19+4*m.x20 <= 400)
    m.c8  = pyo.Constraint(expr=1*m.x1+2*m.x2+1*m.x3+7*m.x4+8*m.x5+7*m.x6+6*m.x7+5*m.x8+8*m.x9+7*m.x10+2*m.x11+3*m.x12+5*m.x13+5*m.x14+4*m.x15+5*m.x16+4*m.x17+2*m.x18+2*m.x19+8*m.x20 <= 460)
    m.c9  = pyo.Constraint(expr=8*m.x1+5*m.x2+2*m.x3+5*m.x4+3*m.x5+8*m.x6+1*m.x7+3*m.x8+3*m.x9+5*m.x10+4*m.x11+5*m.x12+5*m.x13+6*m.x14+1*m.x15+7*m.x16+1*m.x17+2*m.x18+2*m.x19+4*m.x20 <= 400)
    m.c10 = pyo.Constraint(expr=m.x1+m.x2+m.x3+m.x4+m.x5+m.x6+m.x7+m.x8+m.x9+m.x10+m.x11+m.x12+m.x13+m.x14+m.x15+m.x16+m.x17+m.x18+m.x19+m.x20 <= 200)

    # Objective: 0.5*pos - 0.5*neg  (nonconvex DC quadratic)
    pos = (42*(52+m.x11)**2 + 98*(3+m.x12)**2 + 48*(m.x13-81)**2 + 91*(m.x14-30)**2
         + 11*(85+m.x15)**2 + 63*(m.x16-68)**2 + 61*(m.x17-27)**2 + 61*(81+m.x18)**2
         + 38*(m.x19-97)**2 + 26*(73+m.x20)**2)
    neg = (63*(19+m.x1)**2 + 15*(27+m.x2)**2 + 44*(23+m.x3)**2 + 91*(53+m.x4)**2
         + 45*(42+m.x5)**2 + 50*(m.x6-26)**2 + 89*(33+m.x7)**2 + 58*(23+m.x8)**2
         + 86*(m.x9-41)**2 + 82*(m.x10-19)**2)
    m.obj = pyo.Objective(expr=0.5*pos - 0.5*neg, sense=pyo.minimize)

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
        params.set_logging(fname=os.getcwd() + "/logs/singo_2_1_10_log")
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
