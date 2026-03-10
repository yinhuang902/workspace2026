"""
run_st_fp8_case.py -- Simplex runner for SNGO-master/Global/st_fp8

24 variables (xi >= 0), 20 <= constraints (10 equality-band pairs),
concave quadratic objective with per-variable coefficients.

PlasmoOld: RandomStochasticModel(createModel, NS=100) => nfirst=5, nparam=5
  Scenario 1 unperturbed. First 5 constraint UBs perturbed via addnoise_le.
"""
import argparse
from pathlib import Path
from time import perf_counter
from typing import List, Tuple

import pyomo.environ as pyo

from bundles import BaseBundle, MSBundle
from simplex_specialstart import run_pid_simplex_3d


# =============================================================================
# Julia-compatible RNG + PlasmoOld addnoise
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


# =============================================================================
# Deterministic base model -- st_fp8
# =============================================================================

def create_model_st_fp8() -> pyo.ConcreteModel:
    m = pyo.ConcreteModel()

    # 24 variables, all >= 0
    m.x1  = pyo.Var(bounds=(0, 8), initialize=0)
    m.x2  = pyo.Var(bounds=(0, 8), initialize=0)
    m.x3  = pyo.Var(bounds=(0, 8), initialize=0)
    m.x4  = pyo.Var(bounds=(0, 8), initialize=0)
    m.x5  = pyo.Var(bounds=(0, 24), initialize=0)
    m.x6  = pyo.Var(bounds=(0, None), initialize=0)
    m.x7  = pyo.Var(bounds=(0, None), initialize=0)
    m.x8  = pyo.Var(bounds=(0, None), initialize=0)
    m.x9  = pyo.Var(bounds=(0, None), initialize=0)
    m.x10 = pyo.Var(bounds=(0, None), initialize=0)
    m.x11 = pyo.Var(bounds=(0, None), initialize=0)
    m.x12 = pyo.Var(bounds=(0, None), initialize=0)
    m.x13 = pyo.Var(bounds=(0, None), initialize=0)
    m.x14 = pyo.Var(bounds=(0, None), initialize=0)
    m.x15 = pyo.Var(bounds=(0, None), initialize=0)
    m.x16 = pyo.Var(bounds=(0, None), initialize=0)
    m.x17 = pyo.Var(bounds=(0, None), initialize=0)
    m.x18 = pyo.Var(bounds=(0, None), initialize=0)
    m.x19 = pyo.Var(bounds=(0, None), initialize=0)
    m.x20 = pyo.Var(bounds=(0, None), initialize=0)
    m.x21 = pyo.Var(bounds=(0, None), initialize=0)
    m.x22 = pyo.Var(bounds=(0, None), initialize=0)
    m.x23 = pyo.Var(bounds=(0, None), initialize=0)
    m.x24 = pyo.Var(bounds=(0, None), initialize=0)

    # Mutable RHS params for first 5 constraints (stochastic perturbation)
    m.c1_rhs  = pyo.Param(mutable=True, initialize=-29.0)
    m.c2_rhs  = pyo.Param(mutable=True, initialize=29.0)
    m.c3_rhs  = pyo.Param(mutable=True, initialize=-41.0)
    m.c4_rhs  = pyo.Param(mutable=True, initialize=41.0)
    m.c5_rhs  = pyo.Param(mutable=True, initialize=-13.0)

    # 20 constraints (10 equality-band pairs)
    # Column-sum constraints (6 groups of 4 columns each)
    m.c1  = pyo.Constraint(expr=-m.x1 - m.x5 - m.x9 - m.x13 - m.x17 - m.x21 <= m.c1_rhs)
    m.c2  = pyo.Constraint(expr= m.x1 + m.x5 + m.x9 + m.x13 + m.x17 + m.x21 <= m.c2_rhs)
    m.c3  = pyo.Constraint(expr=-m.x2 - m.x6 - m.x10 - m.x14 - m.x18 - m.x22 <= m.c3_rhs)
    m.c4  = pyo.Constraint(expr= m.x2 + m.x6 + m.x10 + m.x14 + m.x18 + m.x22 <= m.c4_rhs)
    m.c5  = pyo.Constraint(expr=-m.x3 - m.x7 - m.x11 - m.x15 - m.x19 - m.x23 <= m.c5_rhs)
    m.c6  = pyo.Constraint(expr= m.x3 + m.x7 + m.x11 + m.x15 + m.x19 + m.x23 <= 13)
    m.c7  = pyo.Constraint(expr=-m.x4 - m.x8 - m.x12 - m.x16 - m.x20 - m.x24 <= -21)
    m.c8  = pyo.Constraint(expr= m.x4 + m.x8 + m.x12 + m.x16 + m.x20 + m.x24 <= 21)
    # Row-sum constraints (4 columns per group)
    m.c9  = pyo.Constraint(expr=-m.x1 - m.x2 - m.x3 - m.x4 <= -8)
    m.c10 = pyo.Constraint(expr= m.x1 + m.x2 + m.x3 + m.x4 <= 8)
    m.c11 = pyo.Constraint(expr=-m.x5 - m.x6 - m.x7 - m.x8 <= -24)
    m.c12 = pyo.Constraint(expr= m.x5 + m.x6 + m.x7 + m.x8 <= 24)
    m.c13 = pyo.Constraint(expr=-m.x9 - m.x10 - m.x11 - m.x12 <= -20)
    m.c14 = pyo.Constraint(expr= m.x9 + m.x10 + m.x11 + m.x12 <= 20)
    m.c15 = pyo.Constraint(expr=-m.x13 - m.x14 - m.x15 - m.x16 <= -24)
    m.c16 = pyo.Constraint(expr= m.x13 + m.x14 + m.x15 + m.x16 <= 24)
    m.c17 = pyo.Constraint(expr=-m.x17 - m.x18 - m.x19 - m.x20 <= -16)
    m.c18 = pyo.Constraint(expr= m.x17 + m.x18 + m.x19 + m.x20 <= 16)
    m.c19 = pyo.Constraint(expr=-m.x21 - m.x22 - m.x23 - m.x24 <= -12)
    m.c20 = pyo.Constraint(expr= m.x21 + m.x22 + m.x23 + m.x24 <= 12)

    # Objective: concave quadratic with per-variable coefficients
    m.obj_expr = (
        300*m.x1  -  7*m.x1**2  + 270*m.x2  -  4*m.x2**2  + 460*m.x3  -  6*m.x3**2
      + 800*m.x4  -  8*m.x4**2  + 740*m.x5  - 12*m.x5**2  + 600*m.x6  -  9*m.x6**2
      + 540*m.x7  - 14*m.x7**2  + 380*m.x8  -  7*m.x8**2  + 300*m.x9  - 13*m.x9**2
      + 490*m.x10 - 12*m.x10**2 + 380*m.x11 -  8*m.x11**2 + 760*m.x12 -  4*m.x12**2
      + 430*m.x13 -  7*m.x13**2 + 250*m.x14 -  9*m.x14**2 + 390*m.x15 - 16*m.x15**2
      + 600*m.x16 -  8*m.x16**2 + 210*m.x17 -  4*m.x17**2 + 830*m.x18 - 10*m.x18**2
      + 470*m.x19 - 21*m.x19**2 + 680*m.x20 - 13*m.x20**2 + 360*m.x21 - 17*m.x21**2
      + 290*m.x22 -  9*m.x22**2 + 400*m.x23 -  8*m.x23**2 + 310*m.x24 -  4*m.x24**2
    )
    return m


def all_vars_st_fp8(m: pyo.ConcreteModel) -> List[pyo.Var]:
    return [m.x1, m.x2, m.x3, m.x4, m.x5, m.x6, m.x7, m.x8, m.x9, m.x10,
            m.x11, m.x12, m.x13, m.x14, m.x15, m.x16, m.x17, m.x18, m.x19, m.x20,
            m.x21, m.x22, m.x23, m.x24]


# =============================================================================
# Scenario generator
# =============================================================================

# RHS values for constraints in JuMP declaration order
_RHS_BASE = [
    ("c1_rhs", -29.0),
    ("c2_rhs",  29.0),
    ("c3_rhs", -41.0),
    ("c4_rhs",  41.0),
    ("c5_rhs", -13.0),
]


def build_models_st_fp8(
    nscen: int,
    nfirst: int = 5,
    nparam: int = 5,
    seed: int = 1234,
    print_first_k_rhs: int = 0,
) -> Tuple[List[pyo.ConcreteModel], List[List[pyo.Var]]]:
    rng = JuliaMT19937(seed)

    model_list: List[pyo.ConcreteModel] = []
    first_vars_list: List[List[pyo.Var]] = []

    max_mods = nparam  # PlasmoOld: cap is nparam, not nparam-1

    for s in range(nscen):
        m = create_model_st_fp8()
        allv = all_vars_st_fp8(m)
        first = allv[:nfirst]

        # Scenario 0 unperturbed (Julia: if i==1; continue; end)
        if s > 0:
            for idx in range(min(max_mods, len(_RHS_BASE))):
                pname, base_val = _RHS_BASE[idx]
                getattr(m, pname).set_value(addnoise_le(base_val, rng))

        if print_first_k_rhs > 0 and s < print_first_k_rhs:
            vals = " ".join(f"{p}={float(pyo.value(getattr(m, p))):.4f}" for p, _ in _RHS_BASE[:max_mods])
            print(f"[SCEN {s:04d}] {vals}")

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
        "time_limit": 60*3,
        "enable_ef_ub": True,
        "ef_time_ub": 30.0,
        "plot_every": None,
        "plot_output_dir": "results/st_fp8_smoke/plots",
        "output_csv_path": "results/st_fp8_smoke/simplex_result.csv",
    },
    "full": {
        "nscen": 100,
        "target_nodes": 900,
        "gap_stop_tol": 1e-3,
        "time_limit": 60*60*12,
        "enable_ef_ub": True,
        "ef_time_ub": 60,
        "plot_every": None,
        "plot_output_dir": "results/st_fp8_full/plots",
        "output_csv_path": "results/st_fp8_full/simplex_result.csv",
    },
}

BUNDLE_OPTIONS = {"NonConvex": 2, "MIPGap": 1e-2, "TimeLimit": 60}
Q_MAX = 1e7


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="full")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--nfirst", type=int, default=5)
    ap.add_argument("--print_first_k_rhs", type=int, default=0)
    args = ap.parse_args()

    nfirst = args.nfirst
    nparam = nfirst  # PlasmoOld default: nparam == nfirst

    cfg = dict(MODE_PARAMS[args.mode])
    out_csv = Path(cfg["output_csv_path"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if cfg["plot_output_dir"] is not None:
        Path(cfg["plot_output_dir"]).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"st_fp8 (Python) -- PlasmoOld RandomStochasticModel")
    print(f"Mode: {args.mode}, nscen={cfg['nscen']}, nfirst={nfirst}, nparam={nparam}, seed={args.seed}")
    print(f"Bundle options: {BUNDLE_OPTIONS}")
    print("=" * 60)

    t0 = perf_counter()

    model_list, first_vars_list = build_models_st_fp8(
        nscen=cfg["nscen"], nfirst=nfirst, nparam=nparam,
        seed=args.seed, print_first_k_rhs=args.print_first_k_rhs
    )
    S = len(model_list)

    base_bundles = [BaseBundle(model_list[s], options=BUNDLE_OPTIONS, q_max=Q_MAX) for s in range(S)]
    ms_bundles   = [MSBundle(model_list[s], first_vars_list[s], options=BUNDLE_OPTIONS) for s in range(S)]

    axis_labels = tuple(f"x{i+1}" for i in range(nfirst))

    res = run_pid_simplex_3d(
        model_list=model_list,
        first_vars_list=first_vars_list,
        base_bundles=base_bundles,
        ms_bundles=ms_bundles,
        target_nodes=cfg["target_nodes"],
        gap_stop_tol=cfg["gap_stop_tol"],
        time_limit=cfg["time_limit"],
        enable_ef_ub=cfg["enable_ef_ub"],
        ef_time_ub=cfg["ef_time_ub"],
        plot_every=cfg["plot_every"],
        plot_output_dir=cfg["plot_output_dir"],
        output_csv_path=str(out_csv),
        enable_3d_plot=False,
        axis_labels=axis_labels,
    )

    t1 = perf_counter()

    LB_hist = res.get("LB_hist", [])
    UB_hist = res.get("UB_hist", [])
    if LB_hist and UB_hist:
        print(f"\n=== Final (sum) ===\nLB_sum = {float(LB_hist[-1]):.12f}\nUB_sum = {float(UB_hist[-1]):.12f}")
        print(f"\n=== Final (per-scenario) ===\nLB = {float(LB_hist[-1])/S:.12f}\nUB = {float(UB_hist[-1])/S:.12f}")

    print(f"{'='*60}\nDone. Wall time: {t1-t0:.2f} sec\nCSV: {out_csv}\n{'='*60}")


if __name__ == "__main__":
    main()
