"""
run_8_4_8_sngo_case.py  -- Simplex runner for SNGO-master/Global/8_4_8

Julia reference (PlasmoOld.jl):
  RandomStochasticModel(createModel, NS=100) => nscen=100, nfirst=5, nparam=5
  srand(1234), scenario 1 unperturbed

Stage split (nfirst=5):
  First-stage:  x1..x5   (cols 1-5)
  Second-stage: x6..x42  (cols 6-42)

Variables (42 total):
  x1-x20: tight bounds (see below)
  x21,x22: [1,2]
  x23-x32: free (unbounded), start=1
  x33-x42: free (unbounded)

Constraints (20 NL constraints):
  10 product constraints:  x_{2k-1}*x_{2k+21}*x_{2k+31} - x_{2k}*x_{2k+2} == 0
  10 log constraints:      x21/x_gamma/(1+...)^2 - log(x_alpha) == 0

  All involve x21,x22 (second-stage). RHS=0 for all.

Stochastic perturbation: first 5 constraints get addnoise(0,...) = U(-10,10)

Objective:
  Sum of weighted squared deviations (convex quadratic in x1-x20)

NOTE: This problem has log() and division in constraints. Gurobi cannot handle
      these; IPOPT or other NLP solver may be needed.
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
# 8_4_8 deterministic base model: exact translation of Julia createModel()
# =============================================================================

def create_model_8_4_8() -> pyo.ConcreteModel:
    m = pyo.ConcreteModel()

    # Variables x1-x20 with tight bounds
    m.x1  = pyo.Var(bounds=(0.285, 0.315), initialize=0.29015241396)
    m.x2  = pyo.Var(bounds=(0.546, 0.636), initialize=0.62189400372)
    m.x3  = pyo.Var(bounds=(0.999071638557945, 1.00092836144205), initialize=1.00009353307628)
    m.x4  = pyo.Var(bounds=(481.55, 486.05), initialize=482.905120568)
    m.x5  = pyo.Var(bounds=(0.385, 0.415), initialize=0.39376636351)
    m.x6  = pyo.Var(bounds=(0.557, 0.647), initialize=0.57716475803)
    m.x7  = pyo.Var(bounds=(0.999071638557945, 1.00092836144205), initialize=0.999721176860282)
    m.x8  = pyo.Var(bounds=(490.95, 495.45), initialize=494.8032165615)
    m.x9  = pyo.Var(bounds=(0.485, 0.515), initialize=0.48701341169)
    m.x10 = pyo.Var(bounds=(0.567, 0.657), initialize=0.61201896021)
    m.x11 = pyo.Var(bounds=(0.999071638557945, 1.00092836144205), initialize=1.00092486639703)
    m.x12 = pyo.Var(bounds=(497.65, 502.15), initialize=500.254300201)
    m.x13 = pyo.Var(bounds=(0.685, 0.715), initialize=0.71473399117)
    m.x14 = pyo.Var(bounds=(0.612, 0.702), initialize=0.68060254203)
    m.x15 = pyo.Var(bounds=(0.999071638557945, 1.00092836144205), initialize=0.999314298281912)
    m.x16 = pyo.Var(bounds=(499.15, 503.65), initialize=502.0287344155)
    m.x17 = pyo.Var(bounds=(0.885, 0.915), initialize=0.88978553592)
    m.x18 = pyo.Var(bounds=(0.769, 0.859), initialize=0.79150724797)
    m.x19 = pyo.Var(bounds=(0.999071638557945, 1.00092836144205), initialize=1.00031365361411)
    m.x20 = pyo.Var(bounds=(467.45, 471.95), initialize=469.4091037145)

    # x21, x22 in [1,2]
    m.x21 = pyo.Var(bounds=(1, 2), initialize=1.9)
    m.x22 = pyo.Var(bounds=(1, 2), initialize=1.6)

    # x23-x32: free variables with start=1
    m.x23 = pyo.Var(bounds=(None, None), initialize=1)
    m.x24 = pyo.Var(bounds=(None, None), initialize=1)
    m.x25 = pyo.Var(bounds=(None, None), initialize=1)
    m.x26 = pyo.Var(bounds=(None, None), initialize=1)
    m.x27 = pyo.Var(bounds=(None, None), initialize=1)
    m.x28 = pyo.Var(bounds=(None, None), initialize=1)
    m.x29 = pyo.Var(bounds=(None, None), initialize=1)
    m.x30 = pyo.Var(bounds=(None, None), initialize=1)
    m.x31 = pyo.Var(bounds=(None, None), initialize=1)
    m.x32 = pyo.Var(bounds=(None, None), initialize=1)

    # x33-x42: free variables
    m.x33 = pyo.Var(bounds=(None, None), initialize=0)
    m.x34 = pyo.Var(bounds=(None, None), initialize=0)
    m.x35 = pyo.Var(bounds=(None, None), initialize=0)
    m.x36 = pyo.Var(bounds=(None, None), initialize=0)
    m.x37 = pyo.Var(bounds=(None, None), initialize=0)
    m.x38 = pyo.Var(bounds=(None, None), initialize=0)
    m.x39 = pyo.Var(bounds=(None, None), initialize=0)
    m.x40 = pyo.Var(bounds=(None, None), initialize=0)
    m.x41 = pyo.Var(bounds=(None, None), initialize=0)
    m.x42 = pyo.Var(bounds=(None, None), initialize=0)

    # Mutable RHS params for stochastic perturbation (first 5 NL constraints)
    m.c1_rhs = pyo.Param(mutable=True, initialize=0.0)
    m.c2_rhs = pyo.Param(mutable=True, initialize=0.0)
    m.c3_rhs = pyo.Param(mutable=True, initialize=0.0)
    m.c4_rhs = pyo.Param(mutable=True, initialize=0.0)
    m.c5_rhs = pyo.Param(mutable=True, initialize=0.0)

    # Objective: sum of weighted squared deviations
    m.obj_expr = (
        (200*m.x1 - 60)**2
      + (66.6666666666667*m.x2 - 39.4)**2
      + (3231.5*m.x3 - 3231.5)**2
      + (1.33333333333333*m.x4 - 645.066666666667)**2
      + (200*m.x5 - 80)**2
      + (66.6666666666667*m.x6 - 40.1333333333333)**2
      + (3231.5*m.x7 - 3231.5)**2
      + (1.33333333333333*m.x8 - 657.6)**2
      + (200*m.x9 - 100)**2
      + (66.6666666666667*m.x10 - 40.8)**2
      + (3231.5*m.x11 - 3231.5)**2
      + (1.33333333333333*m.x12 - 666.533333333333)**2
      + (200*m.x13 - 140)**2
      + (66.6666666666667*m.x14 - 43.8)**2
      + (3231.5*m.x15 - 3231.5)**2
      + (1.33333333333333*m.x16 - 668.533333333333)**2
      + (200*m.x17 - 180)**2
      + (66.6666666666667*m.x18 - 54.2666666666667)**2
      + (3231.5*m.x19 - 3231.5)**2
      + (1.33333333333333*m.x20 - 626.266666666667)**2
    )

    # Product constraints (10): x_{2k-1}*x_{2k+21}*x_{2k+31} - x_{2k}*x_{2k+2} == 0
    m.c1  = pyo.Constraint(expr=m.x23*m.x1*m.x33 - m.x2*m.x4 == m.c1_rhs)
    m.c2  = pyo.Constraint(expr=m.x25*m.x5*m.x35 - m.x6*m.x8 == m.c2_rhs)
    m.c3  = pyo.Constraint(expr=m.x27*m.x9*m.x37 - m.x10*m.x12 == m.c3_rhs)
    m.c4  = pyo.Constraint(expr=m.x29*m.x13*m.x39 - m.x14*m.x16 == m.c4_rhs)
    m.c5  = pyo.Constraint(expr=m.x31*m.x17*m.x41 - m.x18*m.x20 == m.c5_rhs)

    m.c6  = pyo.Constraint(expr=m.x24*(1-m.x1)*m.x34 - (1-m.x2)*m.x4 == 0)
    m.c7  = pyo.Constraint(expr=m.x26*(1-m.x5)*m.x36 - (1-m.x6)*m.x8 == 0)
    m.c8  = pyo.Constraint(expr=m.x28*(1-m.x9)*m.x38 - (1-m.x10)*m.x12 == 0)
    m.c9  = pyo.Constraint(expr=m.x30*(1-m.x13)*m.x40 - (1-m.x14)*m.x16 == 0)
    m.c10 = pyo.Constraint(expr=m.x32*(1-m.x17)*m.x42 - (1-m.x18)*m.x20 == 0)

    # Log constraints (10): x21/x_gamma/(1+ratio)^2 - log(x_alpha) == 0
    m.c11 = pyo.Constraint(expr=m.x21/m.x3/(1 + m.x21/m.x22*m.x1/(1-m.x1))**2 - pyo.log(m.x23) == 0)
    m.c12 = pyo.Constraint(expr=m.x21/m.x7/(1 + m.x21/m.x22*m.x5/(1-m.x5))**2 - pyo.log(m.x25) == 0)
    m.c13 = pyo.Constraint(expr=m.x21/m.x11/(1 + m.x21/m.x22*m.x9/(1-m.x9))**2 - pyo.log(m.x27) == 0)
    m.c14 = pyo.Constraint(expr=m.x21/m.x15/(1 + m.x21/m.x22*m.x13/(1-m.x13))**2 - pyo.log(m.x29) == 0)
    m.c15 = pyo.Constraint(expr=m.x21/m.x19/(1 + m.x21/m.x22*m.x17/(1-m.x17))**2 - pyo.log(m.x31) == 0)

    m.c16 = pyo.Constraint(expr=m.x22/m.x3/(1 + m.x22/m.x21*(1-m.x1)/m.x1)**2 - pyo.log(m.x24) == 0)
    m.c17 = pyo.Constraint(expr=m.x22/m.x7/(1 + m.x22/m.x21*(1-m.x5)/m.x5)**2 - pyo.log(m.x26) == 0)
    m.c18 = pyo.Constraint(expr=m.x22/m.x11/(1 + m.x22/m.x21*(1-m.x9)/m.x9)**2 - pyo.log(m.x28) == 0)
    m.c19 = pyo.Constraint(expr=m.x22/m.x15/(1 + m.x22/m.x21*(1-m.x13)/m.x13)**2 - pyo.log(m.x30) == 0)
    m.c20 = pyo.Constraint(expr=m.x22/m.x19/(1 + m.x22/m.x21*(1-m.x17)/m.x17)**2 - pyo.log(m.x32) == 0)

    return m


def all_vars_8_4_8(m: pyo.ConcreteModel) -> List[pyo.Var]:
    return [m.x1, m.x2, m.x3, m.x4, m.x5, m.x6, m.x7, m.x8, m.x9, m.x10,
            m.x11, m.x12, m.x13, m.x14, m.x15, m.x16, m.x17, m.x18,
            m.x19, m.x20, m.x21, m.x22,
            m.x23, m.x24, m.x25, m.x26, m.x27, m.x28, m.x29, m.x30,
            m.x31, m.x32, m.x33, m.x34, m.x35, m.x36, m.x37, m.x38,
            m.x39, m.x40, m.x41, m.x42]


# =============================================================================
# Scenario generator
# =============================================================================

def build_models_8_4_8(
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
        m = create_model_8_4_8()
        allv = all_vars_8_4_8(m)
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
        "nscen": 5,
        "target_nodes": 60,
        "gap_stop_tol": 1e-6,
        "time_limit": 300,
        "enable_ef_ub": True,
        "ef_time_ub": 30.0,
        "plot_every": None,
        "plot_output_dir": "results/8_4_8_smoke/plots",
        "output_csv_path": "results/8_4_8_smoke/simplex_result.csv",
    },
    "full": {
        "nscen": 100,
        "target_nodes": 300,
        "gap_stop_tol": 1e-2,
        "time_limit": None,
        "enable_ef_ub": True,
        "ef_time_ub": 43200.0,
        "plot_every": None,
        "plot_output_dir": "results/8_4_8_full/plots",
        "output_csv_path": "results/8_4_8_full/simplex_result.csv",
    },
}

BUNDLE_OPTIONS = {
    "NonConvex": 2,
    "MIPGap": 1e-1,
}
Q_MAX = 1e10


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
    print("8_4_8 (Python) -- scenario generation matches Plasmo.jl RandomStochasticModel")
    print(f"Mode: {args.mode}")
    print(f"nscen={cfg['nscen']}, seed={args.seed}, target_nodes={cfg['target_nodes']}")
    print(f"gap_stop_tol={cfg['gap_stop_tol']}, time_limit={cfg['time_limit']}")
    print(f"EF UB enabled={cfg['enable_ef_ub']}, ef_time_ub={cfg['ef_time_ub']}")
    print(f"Bundle options: {BUNDLE_OPTIONS}")
    print("=" * 60)

    t0 = perf_counter()

    model_list, first_vars_list = build_models_8_4_8(
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
