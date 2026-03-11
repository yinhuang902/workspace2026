"""
run_8_4_1_sngo_case.py  -- Simplex runner for SNGO-master/Global/8_4_1

Julia reference (PlasmoOld.jl):
  RandomStochasticModel(createModel, NS=100) => nscen=100, nfirst=5, nparam=5
  srand(1234), scenario 1 unperturbed

Stage split (nfirst=5):
  First-stage:  x1..x5   (cols 1-5)
  Second-stage: x6..x22  (cols 6-22)

Variables (22 total, with non-zero LBs):
  x1 in [-0.5, 0.5],  x2 in [5.4, 6.4],  x3 in [0.4, 1.4],
  x4 in [4.9, 5.9],   x5 in [1.3, 2.3],  x6 in [3.9, 4.9],
  x7 in [2.1, 3.1],   x8 in [4.1, 5.1],  x9 in [2.8, 3.8],
  x10 in [3, 4],      x11 in [3.9, 4.9], x12 in [3.2, 4.2],
  x13 in [4.7, 5.7],  x14 in [2.3, 3.3], x15 in [5.6, 6.6],
  x16 in [2.3, 3.3],  x17 in [6, 7],     x18 in [1.9, 2.9],
  x19 in [6.9, 7.9],  x20 in [1, 2],     x21 in [0, 10],
  x22 in [-2, 2]

Constraints (10 bilinear equalities):
  x22*x_odd - x_even + x21 == 0  for pairs (x1,x2), (x3,x4), ..., (x19,x20)
  All involve x21, x22 (second-stage). RHS=0 for all.

Stochastic perturbation: first 5 constraints get addnoise(0,...) = U(-10,10)

Objective:
  Sum of squared deviations (convex quadratic)
"""
import argparse
from pathlib import Path
from time import perf_counter
from typing import List, Tuple

import pyomo.environ as pyo

from bundles import BaseBundle, MSBundle
from simplex_specialstart import run_pid_simplex_3d


# =============================================================================
# Julia-compatible RNG + Plasmo.jl addnoise
# =============================================================================

class JuliaMT19937:
    N = 624; M = 397
    MATRIX_A = 0x9908B0DF; UPPER_MASK = 0x80000000; LOWER_MASK = 0x7FFFFFFF

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

        y ^= (y >> 11)
        y ^= (y << 7) & 0x9D2C5680
        y ^= (y << 15) & 0xEFC60000
        y ^= (y >> 18)
        return y & 0xFFFFFFFF

    def rand_uint64(self) -> int:
        hi = self.rand_uint32()
        lo = self.rand_uint32()
        return ((hi << 32) | lo) & 0xFFFFFFFFFFFFFFFF

    def rand_float64(self) -> float:
        # Julia-style [0,1)
        u = self.rand_uint64()
        return ((u >> 12) * (1.0 / (1 << 52)))

    def rand_uniform(self, a: float, b: float) -> float:
        return a + (b - a) * self.rand_float64()


def addnoise_julia(a: float, rng: JuliaMT19937) -> float:
    # Plasmo.jl: if a==0 -> +Uniform(-10,10), else *Uniform(0.5,2.0)
    if a == 0.0:
        return a + rng.rand_uniform(-10.0, 10.0)
    return a * rng.rand_uniform(0.5, 2.0)


# =============================================================================
# 8_4_1 deterministic base model: exact translation of Julia createModel()
# =============================================================================

def create_model_8_4_1() -> pyo.ConcreteModel:
    m = pyo.ConcreteModel()

    m.x1  = pyo.Var(bounds=(-0.5, 0.5), initialize=-0.328252868)
    m.x2  = pyo.Var(bounds=(5.4, 6.4),  initialize=6.243266708)
    m.x3  = pyo.Var(bounds=(0.4, 1.4),  initialize=0.950375356)
    m.x4  = pyo.Var(bounds=(4.9, 5.9),  initialize=5.201137904)
    m.x5  = pyo.Var(bounds=(1.3, 2.3),  initialize=1.592212117)
    m.x6  = pyo.Var(bounds=(3.9, 4.9),  initialize=4.124052867)
    m.x7  = pyo.Var(bounds=(2.1, 3.1),  initialize=2.449830504)
    m.x8  = pyo.Var(bounds=(4.1, 5.1),  initialize=4.956270347)
    m.x9  = pyo.Var(bounds=(2.8, 3.8),  initialize=2.867113723)
    m.x10 = pyo.Var(bounds=(3.0, 4.0),  initialize=3.500210669)
    m.x11 = pyo.Var(bounds=(3.9, 4.9),  initialize=4.898117627)
    m.x12 = pyo.Var(bounds=(3.2, 4.2),  initialize=3.778733378)
    m.x13 = pyo.Var(bounds=(4.7, 5.7),  initialize=5.691133039)
    m.x14 = pyo.Var(bounds=(2.3, 3.3),  initialize=3.062250467)
    m.x15 = pyo.Var(bounds=(5.6, 6.6),  initialize=5.730692483)
    m.x16 = pyo.Var(bounds=(2.3, 3.3),  initialize=2.939718759)
    m.x17 = pyo.Var(bounds=(6.0, 7.0),  initialize=6.159517864)
    m.x18 = pyo.Var(bounds=(1.9, 2.9),  initialize=2.150080533)
    m.x19 = pyo.Var(bounds=(6.9, 7.9),  initialize=7.568928609)
    m.x20 = pyo.Var(bounds=(1.0, 2.0),  initialize=1.435356381)
    m.x21 = pyo.Var(bounds=(0.0, 10.0), initialize=3.59700266)
    m.x22 = pyo.Var(bounds=(-2.0, 2.0), initialize=-0.594234528)

    # Mutable RHS params for stochastic perturbation
    # All 10 constraints have RHS=0 and are bilinear (quadratic) involving x21,x22 (second-stage)
    # First 5 get perturbed (nparam=5)
    m.c1_rhs = pyo.Param(mutable=True, initialize=0.0)
    m.c2_rhs = pyo.Param(mutable=True, initialize=0.0)
    m.c3_rhs = pyo.Param(mutable=True, initialize=0.0)
    m.c4_rhs = pyo.Param(mutable=True, initialize=0.0)
    m.c5_rhs = pyo.Param(mutable=True, initialize=0.0)

    # Constraints: x22*x_odd - x_even + x21 == RHS
    m.c1  = pyo.Constraint(expr=m.x22*m.x1  - m.x2  + m.x21 == m.c1_rhs)
    m.c2  = pyo.Constraint(expr=m.x22*m.x3  - m.x4  + m.x21 == m.c2_rhs)
    m.c3  = pyo.Constraint(expr=m.x22*m.x5  - m.x6  + m.x21 == m.c3_rhs)
    m.c4  = pyo.Constraint(expr=m.x22*m.x7  - m.x8  + m.x21 == m.c4_rhs)
    m.c5  = pyo.Constraint(expr=m.x22*m.x9  - m.x10 + m.x21 == m.c5_rhs)
    m.c6  = pyo.Constraint(expr=m.x22*m.x11 - m.x12 + m.x21 == 0)
    m.c7  = pyo.Constraint(expr=m.x22*m.x13 - m.x14 + m.x21 == 0)
    m.c8  = pyo.Constraint(expr=m.x22*m.x15 - m.x16 + m.x21 == 0)
    m.c9  = pyo.Constraint(expr=m.x22*m.x17 - m.x18 + m.x21 == 0)
    m.c10 = pyo.Constraint(expr=m.x22*m.x19 - m.x20 + m.x21 == 0)

    # Objective: sum of squared deviations (convex quadratic)
    m.obj_expr = (
        (m.x1)**2 + (m.x2 - 5.9)**2 + (m.x3 - 0.9)**2 + (m.x4 - 5.4)**2
      + (m.x5 - 1.8)**2 + (m.x6 - 4.4)**2 + (m.x7 - 2.6)**2 + (m.x8 - 4.6)**2
      + (m.x9 - 3.3)**2 + (m.x10 - 3.5)**2 + (m.x11 - 4.4)**2 + (m.x12 - 3.7)**2
      + (m.x13 - 5.2)**2 + (m.x14 - 2.8)**2 + (m.x15 - 6.1)**2 + (m.x16 - 2.8)**2
      + (m.x17 - 6.5)**2 + (m.x18 - 2.4)**2 + (m.x19 - 7.4)**2 + (m.x20 - 1.5)**2
    )
    return m


def all_vars_8_4_1(m: pyo.ConcreteModel) -> List[pyo.Var]:
    return [m.x1, m.x2, m.x3, m.x4, m.x5, m.x6, m.x7, m.x8, m.x9, m.x10,
            m.x11, m.x12, m.x13, m.x14, m.x15, m.x16, m.x17, m.x18,
            m.x19, m.x20, m.x21, m.x22]


# =============================================================================
# Scenario generator
# =============================================================================

def build_models_8_4_1(
    nscen: int,
    nfirst: int = 5,
    nparam: int = 5,
    seed: int = 1234,
    print_first_k_rhs: int = 0,
) -> Tuple[List[pyo.ConcreteModel], List[List[pyo.Var]]]:
    rng = JuliaMT19937(seed)

    model_list: List[pyo.ConcreteModel] = []
    first_vars_list: List[List[pyo.Var]] = []

    for s in range(nscen):
        m = create_model_8_4_1()
        allv = all_vars_8_4_1(m)
        first = allv[:nfirst]

        # Perturb first nparam=5 constraint RHSs (all == 0, so addnoise(0) = U(-10,10))
        # Julia PlasmoOld: scenario 1 (i==1) is unperturbed (continue before noise)
        if s > 0:
            m.c1_rhs.set_value(addnoise_julia(0.0, rng))
            m.c2_rhs.set_value(addnoise_julia(0.0, rng))
            m.c3_rhs.set_value(addnoise_julia(0.0, rng))
            m.c4_rhs.set_value(addnoise_julia(0.0, rng))
            m.c5_rhs.set_value(addnoise_julia(0.0, rng))

        if print_first_k_rhs > 0 and s < print_first_k_rhs:
            print(f"[SCEN {s:04d}] c1_rhs={float(pyo.value(m.c1_rhs)):.4f}  "
                  f"c2_rhs={float(pyo.value(m.c2_rhs)):.4f}  "
                  f"c3_rhs={float(pyo.value(m.c3_rhs)):.4f}  "
                  f"c4_rhs={float(pyo.value(m.c4_rhs)):.4f}  "
                  f"c5_rhs={float(pyo.value(m.c5_rhs)):.4f}")

        model_list.append(m)
        first_vars_list.append(first)

    return model_list, first_vars_list


# =============================================================================
# Runner config
# =============================================================================

MODE_PARAMS = {
    "smoke": {
        "nscen": 10,
        "target_nodes": 60,
        "gap_stop_tol": 1e-6,
        "time_limit": 300,
        "enable_ef_ub": True,
        "ef_time_ub": 30.0,
        "plot_every": None,
        "plot_output_dir": "results/8_4_1_smoke/plots",
        "output_csv_path": "results/8_4_1_smoke/simplex_result.csv",
    },
    "full": {
        "nscen": 100,
        "target_nodes": 900,
        "gap_stop_tol": 1e-2,
        "time_limit": 60*60*12,
        "enable_ef_ub": True,
        "ef_time_ub": 60,
        "plot_every": None,
        "plot_output_dir": "results/8_4_1_full/plots",
        "output_csv_path": "results/8_4_1_full/simplex_result.csv",
    },
}

BUNDLE_OPTIONS = {
    "NonConvex": 2,
    "MIPGap": 1e-2,
    "TimeLimit": 60
}
Q_MAX = 1e2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--print_first_k_rhs", type=int, default=0)
    args = ap.parse_args()

    cfg = dict(MODE_PARAMS[args.mode])

    out_csv = Path(cfg["output_csv_path"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if cfg["plot_output_dir"] is not None:
        Path(cfg["plot_output_dir"]).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("8_4_1 (Python) -- scenario generation matches Plasmo.jl RandomStochasticModel")
    print(f"Mode: {args.mode}")
    print(f"nscen={cfg['nscen']}, seed={args.seed}, target_nodes={cfg['target_nodes']}")
    print(f"gap_stop_tol={cfg['gap_stop_tol']}, time_limit={cfg['time_limit']}")
    print(f"EF UB enabled={cfg['enable_ef_ub']}, ef_time_ub={cfg['ef_time_ub']}")
    print(f"Bundle options: {BUNDLE_OPTIONS}")
    print("=" * 60)

    t0 = perf_counter()

    model_list, first_vars_list = build_models_8_4_1(
        nscen=cfg["nscen"], nfirst=5, nparam=5,
        seed=args.seed, print_first_k_rhs=args.print_first_k_rhs
    )
    S = len(model_list)

    base_bundles = [BaseBundle(model_list[s], options=BUNDLE_OPTIONS, q_max=Q_MAX) for s in range(S)]
    ms_bundles   = [MSBundle(model_list[s], first_vars_list[s], options=BUNDLE_OPTIONS) for s in range(S)]

    res = run_pid_simplex_3d(
        model_list=model_list,
        first_vars_list=first_vars_list,
        base_bundles=base_bundles,
        ms_bundles=ms_bundles,
        target_nodes=cfg["target_nodes"],
        min_dist=1e-3,        gap_stop_tol=cfg["gap_stop_tol"],
        time_limit=cfg["time_limit"],
        enable_ef_ub=cfg["enable_ef_ub"],
        ef_time_ub=cfg["ef_time_ub"],
        plot_every=cfg["plot_every"],
        plot_output_dir=cfg["plot_output_dir"],
        output_csv_path=str(out_csv),
        enable_3d_plot=False,
        axis_labels=("x1", "x2", "x3", "x4", "x5"),
    )

    t1 = perf_counter()

    LB_hist = res.get("LB_hist", [])
    UB_hist = res.get("UB_hist", [])
    if LB_hist and UB_hist:
        final_LB_sum = float(LB_hist[-1])
        final_UB_sum = float(UB_hist[-1])
        print("\n=== Final (sum over scenarios) ===")
        print(f"LB_sum = {final_LB_sum:.12f}")
        print(f"UB_sum = {final_UB_sum:.12f}")
        print("\n=== Final (per-scenario / expectation) ===")
        print(f"LB_per_scen = {final_LB_sum / S:.12f}")
        print(f"UB_per_scen = {final_UB_sum / S:.12f}")

    print("=" * 60)
    print(f"Done. Wall time: {t1 - t0:.2f} sec")
    print(f"CSV: {out_csv}")
    print("=" * 60)


if __name__ == "__main__":
    main()
