import math
from dataclasses import dataclass
from typing import Callable, List, Tuple, Optional

import pyomo.environ as pyo
from pyomo.core.expr.visitor import identify_variables


# ============================================================
# A) Julia-style MT19937 + rand(Float64)  (to match srand/rand)
# ============================================================

class JuliaMT19937:
    """
    Minimal MT19937 RNG with a Julia-style rand(Float64) construction:
      u = rand(UInt64)
      x = (u >> 12) * 2^-52   # in [0,1)
    This matches the common Julia implementation pattern:
      reinterpret(Float64, 0x3ff0000000000000 | (u >> 12)) - 1.0
    """
    # MT19937 parameters
    N = 624
    M = 397
    MATRIX_A = 0x9908B0DF
    UPPER_MASK = 0x80000000
    LOWER_MASK = 0x7FFFFFFF

    def __init__(self, seed: int = 1234):
        self.mt = [0] * self.N
        self.mti = self.N + 1
        self.seed(seed)

    def seed(self, seed: int):
        seed &= 0xFFFFFFFF
        self.mt[0] = seed
        for i in range(1, self.N):
            self.mt[i] = (1812433253 * (self.mt[i - 1] ^ (self.mt[i - 1] >> 30)) + i) & 0xFFFFFFFF
        self.mti = self.N

    def rand_uint32(self) -> int:
        mag01 = [0x0, self.MATRIX_A]
        if self.mti >= self.N:
            # generate N words
            for kk in range(self.N - self.M):
                y = (self.mt[kk] & self.UPPER_MASK) | (self.mt[kk + 1] & self.LOWER_MASK)
                self.mt[kk] = self.mt[kk + self.M] ^ (y >> 1) ^ mag01[y & 0x1]
            for kk in range(self.N - self.M, self.N - 1):
                y = (self.mt[kk] & self.UPPER_MASK) | (self.mt[kk + 1] & self.LOWER_MASK)
                self.mt[kk] = self.mt[kk + (self.M - self.N)] ^ (y >> 1) ^ mag01[y & 0x1]
            y = (self.mt[self.N - 1] & self.UPPER_MASK) | (self.mt[0] & self.LOWER_MASK)
            self.mt[self.N - 1] = self.mt[self.M - 1] ^ (y >> 1) ^ mag01[y & 0x1]
            self.mti = 0

        y = self.mt[self.mti]
        self.mti += 1

        # tempering
        y ^= (y >> 11)
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= (y >> 18)
        return y & 0xFFFFFFFF

    def rand_uint64(self) -> int:
        # combine two uint32s
        hi = self.rand_uint32()
        lo = self.rand_uint32()
        return ((hi << 32) | lo) & 0xFFFFFFFFFFFFFFFF

    def rand_float64(self) -> float:
        # Julia-style: use top 52 bits
        u = self.rand_uint64()
        return ((u >> 12) * (1.0 / (1 << 52)))

    def rand_uniform(self, a: float, b: float) -> float:
        return a + (b - a) * self.rand_float64()


# ============================================================
# B) Plasmo.jl addnoise(a)
# ============================================================

def addnoise_julia(a: float, rng: JuliaMT19937) -> float:
    # Plasmo.jl:
    # if a == 0: a += Uniform(-10,10)
    # else:      a *= Uniform(0.5,2.0)
    if a == 0:
        return a + rng.rand_uniform(-10.0, 10.0)
    else:
        return a * rng.rand_uniform(0.5, 2.0)


# ============================================================
# C) RandomStochasticModel (Python version)
# ============================================================

def _collect_constraints_in_order(m: pyo.ConcreteModel) -> List[pyo.Constraint]:
    # Deterministic order (Pyomo creation order is usually preserved)
    return list(m.component_data_objects(pyo.Constraint, active=True, descend_into=True))

def _constraint_involves_any(vars_in_body, target_vars_set) -> bool:
    for v in vars_in_body:
        if v in target_vars_set:
            return True
    return False

def random_stochastic_models_pyomo(
    create_model_fn: Callable[[], pyo.ConcreteModel],
    all_vars_fn: Callable[[pyo.ConcreteModel], List[pyo.Var]],
    nscen: int = 10,
    nfirst: int = 2,
    nparam: int = 2,
    seed: int = 1234,
) -> Tuple[List[pyo.ConcreteModel], List[List[pyo.Var]]]:
    """
    Python replica of Plasmo.jl RandomStochasticModel(createModel, nscen, nfirst, nparam)
    for Pyomo models.

    Key behaviors (match Plasmo.jl):
    - rng seeded with seed=1234 at the beginning (like srand(1234))
    - for each scenario:
        * create a fresh model
        * identify first-stage vars as the first nfirst vars in all_vars_fn(model)
        * identify second-stage vars as the rest
        * scan linear constraints in order; for each constraint involving any second-stage var:
              modify its bound by addnoise
          stop after modifying (nparam-1) constraints
        * if not enough modified, would scan quadratic constraints (Plasmo.jl does),
          but for these examples it's not needed; we keep a placeholder.
    Returns:
      model_list, first_vars_list
    """
    rng = JuliaMT19937(seed)

    model_list: List[pyo.ConcreteModel] = []
    first_vars_list: List[List[pyo.Var]] = []

    max_mods = max(nparam - 1, 0)

    for s in range(nscen):
        m = create_model_fn()
        all_vars = all_vars_fn(m)
        assert len(all_vars) >= nfirst, f"Need at least nfirst={nfirst} vars, got {len(all_vars)}"

        first_vars = all_vars[:nfirst]
        second_vars = all_vars[nfirst:]
        second_set = set(second_vars)

        cons = _collect_constraints_in_order(m)

        nmodified = 0

        # 1) linear constraints first (degree 1 or 0)
        for con in cons:
            if nmodified >= max_mods:
                break

            # Determine polynomial degree (None means nonlinear; 0 constant; 1 linear; 2 quadratic)
            deg = con.body.polynomial_degree()
            if deg is None or deg > 1:
                continue

            vars_in_body = list(identify_variables(con.body, include_fixed=False))
            if not _constraint_involves_any(vars_in_body, second_set):
                continue

            # Modify bound similarly to JuMP's con.lb / con.ub logic
            lb = con.lower
            ub = con.upper

            # equality
            if lb is not None and ub is not None and abs(float(lb) - float(ub)) <= 0.0:
                newb = addnoise_julia(float(lb), rng)
                con.setlb(newb)
                con.setub(newb)
                nmodified += 1
                continue

            # Plasmo.jl logic:
            # if lb == -Inf -> modify ub
            # else          -> modify lb
            if lb is None:  # -Inf
                if ub is None:
                    # no bounds to modify (rare)
                    continue
                newub = addnoise_julia(float(ub), rng)
                con.setub(newub)
            else:
                newlb = addnoise_julia(float(lb), rng)
                con.setlb(newlb)

            nmodified += 1

        # 2) quadratic constraints (Plasmo.jl also perturbs affine constant term)
        # For your 2_1_* examples, this is typically not used; left as a no-op placeholder.

        # Store
        model_list.append(m)
        first_vars_list.append(first_vars)

    return model_list, first_vars_list