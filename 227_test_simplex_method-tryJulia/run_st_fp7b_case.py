"""
run_st_fp7b_case.py �?Simplex runner for SNGO-master/Global/st_fp7b

Same constraints as st_fp7a (identical to 2_1_7).
Objective: Min Σ (�?.5*xi² �?5*xi) for i=1..20  (concave quadratic)
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
# Deterministic base model �?st_fp7b
# =============================================================================

def create_model_st_fp7b() -> pyo.ConcreteModel:
    m = pyo.ConcreteModel()

    m.x1  = pyo.Var(bounds=(0, 18.22), initialize=0)
    m.x2  = pyo.Var(bounds=(0, 17.40), initialize=0)
    m.x3  = pyo.Var(bounds=(0, 28.81), initialize=0)
    m.x4  = pyo.Var(bounds=(0, 25.78), initialize=0)
    m.x5  = pyo.Var(bounds=(0, 19.14), initialize=0)
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

    m.c1_rhs  = pyo.Param(mutable=True, initialize=-5.0)
    m.c2_rhs  = pyo.Param(mutable=True, initialize=2.0)
    m.c3_rhs  = pyo.Param(mutable=True, initialize=-1.0)
    m.c4_rhs  = pyo.Param(mutable=True, initialize=-3.0)
    m.c5_rhs  = pyo.Param(mutable=True, initialize=5.0)
    m.c6_rhs  = pyo.Param(mutable=True, initialize=4.0)
    m.c7_rhs  = pyo.Param(mutable=True, initialize=-1.0)
    m.c8_rhs  = pyo.Param(mutable=True, initialize=0.0)
    m.c9_rhs  = pyo.Param(mutable=True, initialize=9.0)
    m.c10_rhs = pyo.Param(mutable=True, initialize=40.0)

    m.c1  = pyo.Constraint(expr=-3*m.x1 + 7*m.x2 - 5*m.x4 + m.x5 + m.x6 + 2*m.x8 - m.x9 - m.x10 - 9*m.x11 + 3*m.x12 + 5*m.x13 + m.x16 + 7*m.x17 - 7*m.x18 - 4*m.x19 - 6*m.x20 <= m.c1_rhs)
    m.c2  = pyo.Constraint(expr=7*m.x1 - 5*m.x3 + m.x4 + m.x5 + 2*m.x7 - m.x8 - m.x9 - 9*m.x10 + 3*m.x11 + 5*m.x12 + m.x15 + 7*m.x16 - 7*m.x17 - 4*m.x18 - 6*m.x19 - 3*m.x20 <= m.c2_rhs)
    m.c3  = pyo.Constraint(expr=-5*m.x2 + m.x3 + m.x4 + 2*m.x6 - m.x7 - m.x8 - 9*m.x9 + 3*m.x10 + 5*m.x11 + m.x14 + 7*m.x15 - 7*m.x16 - 4*m.x17 - 6*m.x18 - 3*m.x19 + 7*m.x20 <= m.c3_rhs)
    m.c4  = pyo.Constraint(expr=-5*m.x1 + m.x2 + m.x3 + 2*m.x5 - m.x6 - m.x7 - 9*m.x8 + 3*m.x9 + 5*m.x10 + m.x13 + 7*m.x14 - 7*m.x15 - 4*m.x16 - 6*m.x17 - 3*m.x18 + 7*m.x19 <= m.c4_rhs)
    m.c5  = pyo.Constraint(expr=m.x1 + m.x2 + 2*m.x4 - m.x5 - m.x6 - 9*m.x7 + 3*m.x8 + 5*m.x9 + m.x12 + 7*m.x13 - 7*m.x14 - 4*m.x15 - 6*m.x16 - 3*m.x17 + 7*m.x18 - 5*m.x20 <= m.c5_rhs)
    m.c6  = pyo.Constraint(expr=m.x1 + 2*m.x3 - m.x4 - m.x5 - 9*m.x6 + 3*m.x7 + 5*m.x8 + m.x11 + 7*m.x12 - 7*m.x13 - 4*m.x14 - 6*m.x15 - 3*m.x16 + 7*m.x17 - 5*m.x19 + m.x20 <= m.c6_rhs)
    m.c7  = pyo.Constraint(expr=2*m.x2 - m.x3 - m.x4 - 9*m.x5 + 3*m.x6 + 5*m.x7 + m.x10 + 7*m.x11 - 7*m.x12 - 4*m.x13 - 6*m.x14 - 3*m.x15 + 7*m.x16 - 5*m.x18 + m.x19 + m.x20 <= m.c7_rhs)
    m.c8  = pyo.Constraint(expr=2*m.x1 - m.x2 - m.x3 - 9*m.x4 + 3*m.x5 + 5*m.x6 + m.x9 + 7*m.x10 - 7*m.x11 - 4*m.x12 - 6*m.x13 - 3*m.x14 + 7*m.x15 - 5*m.x17 + m.x18 + m.x19 <= m.c8_rhs)
    m.c9  = pyo.Constraint(expr=-m.x1 - m.x2 - 9*m.x3 + 3*m.x4 + 5*m.x5 + m.x8 + 7*m.x9 - 7*m.x10 - 4*m.x11 - 6*m.x12 - 3*m.x13 + 7*m.x14 - 5*m.x16 + m.x17 + m.x18 + 2*m.x20 <= m.c9_rhs)
    m.c10 = pyo.Constraint(expr=m.x1+m.x2+m.x3+m.x4+m.x5+m.x6+m.x7+m.x8+m.x9+m.x10+m.x11+m.x12+m.x13+m.x14+m.x15+m.x16+m.x17+m.x18+m.x19+m.x20 <= m.c10_rhs)

    # Objective: Min Σ (�?.5*xi² �?5*xi) for i=1..20
    m.obj_expr = sum(-0.5*v**2 - 5*v for v in [
        m.x1, m.x2, m.x3, m.x4, m.x5, m.x6, m.x7, m.x8, m.x9, m.x10,
        m.x11, m.x12, m.x13, m.x14, m.x15, m.x16, m.x17, m.x18, m.x19, m.x20])
    return m


def all_vars_st_fp7b(m): return [getattr(m, f"x{i}") for i in range(1, 21)]

_RHS_BASE = [("c1_rhs",-5.),("c2_rhs",2.),("c3_rhs",-1.),("c4_rhs",-3.),("c5_rhs",5.),
             ("c6_rhs",4.),("c7_rhs",-1.),("c8_rhs",0.),("c9_rhs",9.),("c10_rhs",40.)]

def build_models_st_fp7b(nscen, nfirst=2, nparam=2, seed=1234, print_first_k_rhs=0):
    rng = JuliaMT19937(seed)
    model_list, first_vars_list = [], []
    max_mods = nparam  # PlasmoOld: nmodified >= nparam (cap is nparam, not nparam-1)
    for s in range(nscen):
        m = create_model_st_fp7b()
        allv = all_vars_st_fp7b(m)
        first = allv[:nfirst]
        if s > 0:
            for idx in range(min(max_mods, len(_RHS_BASE))):
                pname, base_val = _RHS_BASE[idx]
                getattr(m, pname).set_value(addnoise_le(base_val, rng))
        if print_first_k_rhs > 0 and s < print_first_k_rhs:
            vals = " ".join(f"{p}={float(pyo.value(getattr(m, p))):.4f}" for p, _ in _RHS_BASE[:max_mods])
            print(f"[SCEN {s:04d}] {vals}")
        model_list.append(m); first_vars_list.append(first)
    return model_list, first_vars_list

MODE_PARAMS = {
    "smoke": {"nscen":5,"target_nodes":60,"gap_stop_tol":1e-3,"time_limit":60*3,"enable_ef_ub":True,"ef_time_ub":30.0,"plot_every":None,"plot_output_dir":"results/st_fp7b_smoke/plots","output_csv_path":"results/st_fp7b_smoke/simplex_result.csv"},
    "full":  {"nscen":100,"target_nodes":900,"gap_stop_tol":1e-4,"time_limit":60*60*12,"enable_ef_ub":True,"ef_time_ub":60,"plot_every":None,"plot_output_dir":"results/st_fp7b_full/plots","output_csv_path":"results/st_fp7b_full/simplex_result.csv"},
}
BUNDLE_OPTIONS = {"NonConvex": 2, "MIPGap": 1e-2, "TimeLimit": 60}
Q_MAX = -1e2

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke","full"), default="full")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--nfirst", type=int, default=5)
    ap.add_argument("--print_first_k_rhs", type=int, default=0)
    args = ap.parse_args()
    nfirst = args.nfirst; nparam = nfirst
    cfg = dict(MODE_PARAMS[args.mode])
    out_csv = Path(cfg["output_csv_path"]); out_csv.parent.mkdir(parents=True, exist_ok=True)
    if cfg["plot_output_dir"]: Path(cfg["plot_output_dir"]).mkdir(parents=True, exist_ok=True)
    print("="*60); print(f"st_fp7b (Python) �?nfirst={nfirst}, nparam={nparam}, seed={args.seed}"); print("="*60)
    t0 = perf_counter()
    model_list, first_vars_list = build_models_st_fp7b(nscen=cfg["nscen"], nfirst=nfirst, nparam=nparam, seed=args.seed, print_first_k_rhs=args.print_first_k_rhs)
    S = len(model_list)
    base_bundles = [BaseBundle(model_list[s], options=BUNDLE_OPTIONS, q_max=Q_MAX) for s in range(S)]
    ms_bundles   = [MSBundle(model_list[s], first_vars_list[s], options=BUNDLE_OPTIONS) for s in range(S)]
    res = run_pid_simplex_3d(model_list=model_list, first_vars_list=first_vars_list, base_bundles=base_bundles, ms_bundles=ms_bundles,
        target_nodes=cfg["target_nodes"], gap_stop_tol=cfg["gap_stop_tol"], time_limit=cfg["time_limit"],
        enable_ef_ub=cfg["enable_ef_ub"], ef_time_ub=cfg["ef_time_ub"], plot_every=cfg["plot_every"],
        plot_output_dir=cfg["plot_output_dir"], output_csv_path=str(out_csv), enable_3d_plot=False, axis_labels=tuple(f"x{i+1}" for i in range(nfirst)))
    t1 = perf_counter()
    LB_hist = res.get("LB_hist",[]); UB_hist = res.get("UB_hist",[])
    if LB_hist and UB_hist:
        print(f"\n=== Final ===\nLB_sum={float(LB_hist[-1]):.12f}  UB_sum={float(UB_hist[-1]):.12f}")
        print(f"LB/S={float(LB_hist[-1])/S:.12f}  UB/S={float(UB_hist[-1])/S:.12f}")
    print(f"{'='*60}\nDone. {t1-t0:.2f}s  CSV: {out_csv}\n{'='*60}")

if __name__ == "__main__": main()
