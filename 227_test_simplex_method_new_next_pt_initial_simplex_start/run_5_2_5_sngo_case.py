"""
run_5_2_5_sngo_case.py  -- Simplex runner for SNGO-master/Global/5_2_5

Julia reference (PlasmoOld.jl):
  RandomStochasticModel(createModel, NS=100) => nscen=100, nfirst=5, nparam=5
  srand(1234), scenario 1 unperturbed

Stage split (nfirst=5):
  First-stage:  x1..x5   (cols 1-5)
  Second-stage: x6..x32  (cols 6-32)

Variables:
  x1-x12  in [0,1]
  x13     in [0,100], x14 in [0,200], x15-x18 in [0,100]
  x19     in [0,200], x20-x23 in [0,100]
  x24     in [0,200], x25-x28 in [0,100]
  x29     in [0,200], x30-x32 in [0,100]

Constraints:
  19 constraints total (1 bilinear capacity, 5 capacity, 10 bilinear coupling, 3 equality)

Stochastic perturbation (PlasmoOld addnoise for <= constraints):
  First 5 constraints involving second-stage vars get perturbed.
  c1 (bilinear, <= 50) is quadratic involving x7-x9 (second-stage).
  c2-c6 (linear capacity, <= 100/200) involve second-stage vars.

Objective:
  Nonconvex bilinear (Min of negated terms)
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
# 5_2_5 deterministic base model: exact translation of Julia createModel()
# =============================================================================

def create_model_5_2_5() -> pyo.ConcreteModel:
    m = pyo.ConcreteModel()

    # Variables x1-x12 in [0,1]
    m.x1  = pyo.Var(bounds=(0, 1), initialize=0)
    m.x2  = pyo.Var(bounds=(0, 1), initialize=0)
    m.x3  = pyo.Var(bounds=(0, 1), initialize=0)
    m.x4  = pyo.Var(bounds=(0, 1), initialize=0)
    m.x5  = pyo.Var(bounds=(0, 1), initialize=0)
    m.x6  = pyo.Var(bounds=(0, 1), initialize=0)
    m.x7  = pyo.Var(bounds=(0, 1), initialize=0)
    m.x8  = pyo.Var(bounds=(0, 1), initialize=0)
    m.x9  = pyo.Var(bounds=(0, 1), initialize=0)
    m.x10 = pyo.Var(bounds=(0, 1), initialize=0)
    m.x11 = pyo.Var(bounds=(0, 1), initialize=0)
    m.x12 = pyo.Var(bounds=(0, 1), initialize=0)

    # Variables x13-x32 with various bounds
    m.x13 = pyo.Var(bounds=(0, 100), initialize=0)
    m.x14 = pyo.Var(bounds=(0, 200), initialize=0)
    m.x15 = pyo.Var(bounds=(0, 100), initialize=0)
    m.x16 = pyo.Var(bounds=(0, 100), initialize=0)
    m.x17 = pyo.Var(bounds=(0, 100), initialize=0)
    m.x18 = pyo.Var(bounds=(0, 100), initialize=0)
    m.x19 = pyo.Var(bounds=(0, 200), initialize=0)
    m.x20 = pyo.Var(bounds=(0, 100), initialize=0)
    m.x21 = pyo.Var(bounds=(0, 100), initialize=0)
    m.x22 = pyo.Var(bounds=(0, 100), initialize=0)
    m.x23 = pyo.Var(bounds=(0, 100), initialize=0)
    m.x24 = pyo.Var(bounds=(0, 200), initialize=0)
    m.x25 = pyo.Var(bounds=(0, 100), initialize=0)
    m.x26 = pyo.Var(bounds=(0, 100), initialize=0)
    m.x27 = pyo.Var(bounds=(0, 100), initialize=0)
    m.x28 = pyo.Var(bounds=(0, 100), initialize=0)
    m.x29 = pyo.Var(bounds=(0, 200), initialize=0)
    m.x30 = pyo.Var(bounds=(0, 100), initialize=0)
    m.x31 = pyo.Var(bounds=(0, 100), initialize=0)
    m.x32 = pyo.Var(bounds=(0, 100), initialize=0)

    # Mutable RHS params for stochastic perturbation
    # c1 is quadratic (bilinear) involving second-stage vars x7-x9
    m.c1_rhs = pyo.Param(mutable=True, initialize=50.0)
    # c2-c6 are linear involving second-stage vars
    m.c2_rhs = pyo.Param(mutable=True, initialize=100.0)
    m.c3_rhs = pyo.Param(mutable=True, initialize=200.0)
    m.c4_rhs = pyo.Param(mutable=True, initialize=100.0)
    m.c5_rhs = pyo.Param(mutable=True, initialize=100.0)
    m.c6_rhs = pyo.Param(mutable=True, initialize=100.0)

    # Constraints
    # c1: bilinear capacity (quadratic, involves second-stage x7,x8,x9)
    m.c1 = pyo.Constraint(expr=
        m.x7*m.x18 + m.x7*m.x19 + m.x7*m.x20 + m.x7*m.x21 + m.x7*m.x22
      + m.x8*m.x23 + m.x8*m.x24 + m.x8*m.x25 + m.x8*m.x26 + m.x8*m.x27
      + m.x9*m.x28 + m.x9*m.x29 + m.x9*m.x30 + m.x9*m.x31 + m.x9*m.x32
      <= m.c1_rhs)

    # c2-c6: linear capacity constraints (involve second-stage vars)
    m.c2 = pyo.Constraint(expr=m.x13 + m.x18 + m.x23 + m.x28 <= m.c2_rhs)
    m.c3 = pyo.Constraint(expr=m.x14 + m.x19 + m.x24 + m.x29 <= m.c3_rhs)
    m.c4 = pyo.Constraint(expr=m.x15 + m.x20 + m.x25 + m.x30 <= m.c4_rhs)
    m.c5 = pyo.Constraint(expr=m.x16 + m.x21 + m.x26 + m.x31 <= m.c5_rhs)
    m.c6 = pyo.Constraint(expr=m.x17 + m.x22 + m.x27 + m.x32 <= m.c6_rhs)

    # c7-c16: bilinear coupling constraints (no stochastic perturbation beyond nparam=5)
    m.c7 = pyo.Constraint(expr=
        (3*m.x1 + m.x4 + m.x7 + 1.5*m.x10 - 2.5)*m.x18
      + (3*m.x2 + m.x5 + m.x8 + 1.5*m.x11 - 2.5)*m.x23
      + (3*m.x3 + m.x6 + m.x9 + 1.5*m.x12 - 2.5)*m.x28
      - 0.5*m.x13 <= 0)
    m.c8 = pyo.Constraint(expr=
        (m.x1 + 3*m.x4 + 2.5*m.x7 + 2.5*m.x10 - 2)*m.x18
      + (m.x2 + 3*m.x5 + 2.5*m.x8 + 2.5*m.x11 - 2)*m.x23
      + (m.x3 + 3*m.x6 + 2.5*m.x9 + 2.5*m.x12 - 2)*m.x28
      + 0.5*m.x13 <= 0)
    m.c9 = pyo.Constraint(expr=
        (3*m.x1 + m.x4 + m.x7 + 1.5*m.x10 - 1.5)*m.x19
      + (3*m.x2 + m.x5 + m.x8 + 1.5*m.x11 - 1.5)*m.x24
      + (3*m.x3 + m.x6 + m.x9 + 1.5*m.x12 - 1.5)*m.x29
      + 0.5*m.x14 <= 0)
    m.c10 = pyo.Constraint(expr=
        (m.x1 + 3*m.x4 + 2.5*m.x7 + 2.5*m.x10 - 2.5)*m.x19
      + (m.x2 + 3*m.x5 + 2.5*m.x8 + 2.5*m.x11 - 2.5)*m.x24
      + (m.x3 + 3*m.x6 + 2.5*m.x9 + 2.5*m.x12 - 2.5)*m.x29
      <= 0)
    m.c11 = pyo.Constraint(expr=
        (3*m.x1 + m.x4 + m.x7 + 1.5*m.x10 - 2)*m.x20
      + (3*m.x2 + m.x5 + m.x8 + 1.5*m.x11 - 2)*m.x25
      + (3*m.x3 + m.x6 + m.x9 + 1.5*m.x12 - 2)*m.x30
      <= 0)
    m.c12 = pyo.Constraint(expr=
        (m.x1 + 3*m.x4 + 2.5*m.x7 + 2.5*m.x10 - 2.6)*m.x20
      + (m.x2 + 3*m.x5 + 2.5*m.x8 + 2.5*m.x11 - 2.6)*m.x25
      + (m.x3 + 3*m.x6 + 2.5*m.x9 + 2.5*m.x12 - 2.6)*m.x30
      - 0.1*m.x15 <= 0)
    m.c13 = pyo.Constraint(expr=
        (3*m.x1 + m.x4 + m.x7 + 1.5*m.x10 - 2)*m.x21
      + (3*m.x2 + m.x5 + m.x8 + 1.5*m.x11 - 2)*m.x26
      + (3*m.x3 + m.x6 + m.x9 + 1.5*m.x12 - 2)*m.x31
      <= 0)
    m.c14 = pyo.Constraint(expr=
        (m.x1 + 3*m.x4 + 2.5*m.x7 + 2.5*m.x10 - 2)*m.x21
      + (m.x2 + 3*m.x5 + 2.5*m.x8 + 2.5*m.x11 - 2)*m.x26
      + (m.x3 + 3*m.x6 + 2.5*m.x9 + 2.5*m.x12 - 2)*m.x31
      + 0.5*m.x16 <= 0)
    m.c15 = pyo.Constraint(expr=
        (3*m.x1 + m.x4 + m.x7 + 1.5*m.x10 - 2)*m.x22
      + (3*m.x2 + m.x5 + m.x8 + 1.5*m.x11 - 2)*m.x27
      + (3*m.x3 + m.x6 + m.x9 + 1.5*m.x12 - 2)*m.x32
      <= 0)
    m.c16 = pyo.Constraint(expr=
        (m.x1 + 3*m.x4 + 2.5*m.x7 + 2.5*m.x10 - 2)*m.x22
      + (m.x2 + 3*m.x5 + 2.5*m.x8 + 2.5*m.x11 - 2)*m.x27
      + (m.x3 + 3*m.x6 + 2.5*m.x9 + 2.5*m.x12 - 2)*m.x32
      + 0.5*m.x17 <= 0)

    # c17-c19: equality constraints (only first-stage vars x1-x12)
    m.c17 = pyo.Constraint(expr=m.x1 + m.x4 + m.x7 + m.x10 == 1)
    m.c18 = pyo.Constraint(expr=m.x2 + m.x5 + m.x8 + m.x11 == 1)
    m.c19 = pyo.Constraint(expr=m.x3 + m.x6 + m.x9 + m.x12 == 1)

    # Objective: Min -(...) - 8*x13 - 5*x14 - 9*x15 - 6*x16 - 4*x17
    bilinear_sum = (
        (18 - 6*m.x1 - 16*m.x4 - 15*m.x7 - 12*m.x10)*m.x18
      + (18 - 6*m.x2 - 16*m.x5 - 15*m.x8 - 12*m.x11)*m.x23
      + (18 - 6*m.x3 - 16*m.x6 - 15*m.x9 - 12*m.x12)*m.x28
      + (15 - 6*m.x1 - 16*m.x4 - 15*m.x7 - 12*m.x10)*m.x19
      + (15 - 6*m.x2 - 16*m.x5 - 15*m.x8 - 12*m.x11)*m.x24
      + (15 - 6*m.x3 - 16*m.x6 - 15*m.x9 - 12*m.x12)*m.x29
      + (19 - 6*m.x1 - 16*m.x4 - 15*m.x7 - 12*m.x10)*m.x20
      + (19 - 6*m.x2 - 16*m.x5 - 15*m.x8 - 12*m.x11)*m.x25
      + (19 - 6*m.x3 - 16*m.x6 - 15*m.x9 - 12*m.x12)*m.x30
      + (16 - 6*m.x1 - 16*m.x4 - 15*m.x7 - 12*m.x10)*m.x21
      + (16 - 6*m.x2 - 16*m.x5 - 15*m.x8 - 12*m.x11)*m.x26
      + (16 - 6*m.x3 - 16*m.x6 - 15*m.x9 - 12*m.x12)*m.x31
      + (14 - 6*m.x1 - 16*m.x4 - 15*m.x7 - 12*m.x10)*m.x22
      + (14 - 6*m.x2 - 16*m.x5 - 15*m.x8 - 12*m.x11)*m.x27
      + (14 - 6*m.x3 - 16*m.x6 - 15*m.x9 - 12*m.x12)*m.x32
    )
    m.obj_expr = -(bilinear_sum) - 8*m.x13 - 5*m.x14 - 9*m.x15 - 6*m.x16 - 4*m.x17
    return m


def all_vars_5_2_5(m: pyo.ConcreteModel) -> List[pyo.Var]:
    return [m.x1, m.x2, m.x3, m.x4, m.x5, m.x6, m.x7, m.x8, m.x9, m.x10,
            m.x11, m.x12, m.x13, m.x14, m.x15, m.x16, m.x17, m.x18, m.x19, m.x20,
            m.x21, m.x22, m.x23, m.x24, m.x25, m.x26, m.x27, m.x28, m.x29, m.x30,
            m.x31, m.x32]


# =============================================================================
# Scenario generator
# =============================================================================

def build_models_5_2_5(
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
        m = create_model_5_2_5()
        allv = all_vars_5_2_5(m)
        first = allv[:nfirst]

        # Perturb first nparam constraint RHSs involving second-stage vars
        # c1 (quadratic, RHS=50), c2-c6 (linear, RHS=100/200/100/100/100)
        # Only 5 modifications total (nparam=5): c1 + c2 + c3 + c4 + c5
        # Julia PlasmoOld: scenario 1 (i==1) is unperturbed (continue before noise)
        if s > 0:
            m.c1_rhs.set_value(addnoise_julia(50.0, rng))
            m.c2_rhs.set_value(addnoise_julia(100.0, rng))
            m.c3_rhs.set_value(addnoise_julia(200.0, rng))
            m.c4_rhs.set_value(addnoise_julia(100.0, rng))
            m.c5_rhs.set_value(addnoise_julia(100.0, rng))

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
        "nscen": 30,
        "target_nodes": 60,
        "gap_stop_tol": 1e-6,
        "time_limit": 60*10,
        "enable_ef_ub": True,
        "ef_time_ub": 30.0,
        "plot_every": None,
        "plot_output_dir": "results/5_2_5_smoke/plots",
        "output_csv_path": "results/5_2_5_smoke/simplex_result.csv",
    },
    "full": {
        "nscen": 100,
        "target_nodes": 900,
        "gap_stop_tol": 1e-2,
        "time_limit": 60*60*12,
        "enable_ef_ub": True,
        "ef_time_ub": 60,
        "plot_every": None,
        "plot_output_dir": "results/5_2_5_full/plots",
        "output_csv_path": "results/5_2_5_full/simplex_result.csv",
    },
}

BUNDLE_OPTIONS = {
    "NonConvex": 2,
    "MIPGap": 1e-2,
    "TimeLimit": 60
}
Q_MAX = -1e3


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="full")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--print_first_k_rhs", type=int, default=0)
    args = ap.parse_args()

    cfg = dict(MODE_PARAMS[args.mode])

    out_csv = Path(cfg["output_csv_path"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if cfg["plot_output_dir"] is not None:
        Path(cfg["plot_output_dir"]).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("5_2_5 (Python) -- scenario generation matches Plasmo.jl RandomStochasticModel")
    print(f"Mode: {args.mode}")
    print(f"nscen={cfg['nscen']}, seed={args.seed}, target_nodes={cfg['target_nodes']}")
    print(f"gap_stop_tol={cfg['gap_stop_tol']}, time_limit={cfg['time_limit']}")
    print(f"EF UB enabled={cfg['enable_ef_ub']}, ef_time_ub={cfg['ef_time_ub']}")
    print(f"Bundle options: {BUNDLE_OPTIONS}")
    print("=" * 60)

    t0 = perf_counter()

    model_list, first_vars_list = build_models_5_2_5(
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
