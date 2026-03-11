"""
run_2_1_8_case.py �?Simplex runner for SNGO-master/Global/2_1_8

Julia reference (PlasmoOld.jl):
  RandomStochasticModel(createModel, NS=100)  =>  nscen=100, nfirst=5, nparam=5
  srand(1234), scenario 1 unperturbed

Stage split (nfirst=5):
  First-stage:  x1..x5   (cols 1-5)
  Second-stage: x6..x24  (cols 6-24)

All 24 variables: 0 <= xi <= 100

Constraints: 10 equality constraints
  c1: x1+x2+x3+x4 == 8          (all first-stage -> SKIP)
  c2: x5+x6+x7+x8 == 24         (has 2nd-stage -> modified)
  c3: x9+x10+x11+x12 == 20      (all 2nd-stage -> modified)
  c4: x13+x14+x15+x16 == 24     (all 2nd-stage -> modified)
  c5: x17+x18+x19+x20 == 16     (all 2nd-stage -> modified)
  c6: x21+x22+x23+x24 == 12     (all 2nd-stage -> modified, nmodified=5 -> break)
  c7-c10: not reached

PlasmoOld equality constraint splitting:
  Each equality (lb==ub) is split into:
    - Original becomes <=: con.ub = addnoise(ub, 0, 10, 0, 2), con.lb = -Inf
    - New copy becomes >=: connew.lb = addnoise(lb, -10, 0, -2, 0), connew.ub = Inf
  Each equality consumes 2 RNG draws.

  addnoise(a, 0, 10, 0, 2):  a + |a| * U(0, 2)    if a != 0
  addnoise(a, -10, 0, -2, 0):  a + |a| * U(-2, 0)  if a != 0

Objective:
  Min sum_i (lin_i*xi - coeff_i*xi^2)
  = (300x1-7x1^2) + (270x2-4x2^2) + (460x3-6x3^2) + (800x4-8x4^2) + ...
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
    """PlasmoOld addnoise for <= / upper bound: addnoise(a, 0, 10, 0, 2)"""
    if a == 0.0:
        return a + rng.rand_uniform(0.0, 10.0)
    return a + abs(a) * rng.rand_uniform(0.0, 2.0)


def addnoise_ge(a: float, rng: JuliaMT19937) -> float:
    """PlasmoOld addnoise for >= / lower bound: addnoise(a, -10, 0, -2, 0)"""
    if a == 0.0:
        return a + rng.rand_uniform(-10.0, 0.0)
    return a + abs(a) * rng.rand_uniform(-2.0, 0.0)


# =============================================================================
# Deterministic base model
# =============================================================================

# Objective coefficients from Julia: lin_coeff*xi - quad_coeff*xi^2
# Julia: 300*x1 - 7*x1*x1 - 4*x2*x2 + 270*x2 - 6*x3*x3 + 460*x3 ...
OBJ_LIN = [300, 270, 460, 800, 740, 600, 540, 380, 300, 490,
           380, 760, 430, 250, 390, 600, 210, 830, 470, 680, 360, 290, 400, 310]
OBJ_QUAD = [7, 4, 6, 8, 12, 9, 14, 7, 13, 12,
            8, 4, 7, 9, 16, 8, 4, 10, 21, 13, 17, 9, 8, 4]

STARTS = {1: 6, 2: 2, 6: 3, 8: 21, 9: 20, 13: 24, 17: 3, 19: 13, 22: 12}  # 1-indexed


def create_model_2_1_8(c2_ub=24.0, c2_lb=24.0,
                       c3_ub=20.0, c3_lb=20.0,
                       c4_ub=24.0, c4_lb=24.0,
                       c5_ub=16.0, c5_lb=16.0,
                       c6_ub=12.0, c6_lb=12.0) -> pyo.ConcreteModel:
    m = pyo.ConcreteModel()

    # Variables: all in [0, 100]. x1-x5 first-stage.
# Variables: x1-x5 use tight first-stage bounds; x6-x24 keep [0, 100].
    for i in range(1, 25):
        start_val = STARTS.get(i, 0)
        if i in [1, 2, 3, 4]:
            bds = (0, 8)
        elif i == 5:
            bds = (0, 24)
        else:
            bds = (0, 100)
        setattr(m, f'x{i}', pyo.Var(bounds=bds, initialize=start_val))

    xs = [getattr(m, f'x{i}') for i in range(1, 25)]

    # c1: x1+x2+x3+x4 == 8  (all first-stage, never modified)
    m.c1 = pyo.Constraint(expr=xs[0] + xs[1] + xs[2] + xs[3] == 8)

    # c2-c6: equality constraints split into <= and >=
    m.c2_le = pyo.Constraint(expr=xs[4] + xs[5] + xs[6] + xs[7] <= c2_ub)
    m.c2_ge = pyo.Constraint(expr=xs[4] + xs[5] + xs[6] + xs[7] >= c2_lb)
    m.c3_le = pyo.Constraint(expr=xs[8] + xs[9] + xs[10] + xs[11] <= c3_ub)
    m.c3_ge = pyo.Constraint(expr=xs[8] + xs[9] + xs[10] + xs[11] >= c3_lb)
    m.c4_le = pyo.Constraint(expr=xs[12] + xs[13] + xs[14] + xs[15] <= c4_ub)
    m.c4_ge = pyo.Constraint(expr=xs[12] + xs[13] + xs[14] + xs[15] >= c4_lb)
    m.c5_le = pyo.Constraint(expr=xs[16] + xs[17] + xs[18] + xs[19] <= c5_ub)
    m.c5_ge = pyo.Constraint(expr=xs[16] + xs[17] + xs[18] + xs[19] >= c5_lb)
    m.c6_le = pyo.Constraint(expr=xs[20] + xs[21] + xs[22] + xs[23] <= c6_ub)
    m.c6_ge = pyo.Constraint(expr=xs[20] + xs[21] + xs[22] + xs[23] >= c6_lb)

    # c7-c10: deterministic equality (unchanged)
    m.c7  = pyo.Constraint(expr=xs[0]+xs[4]+xs[8]+xs[12]+xs[16]+xs[20] == 29)
    m.c8  = pyo.Constraint(expr=xs[1]+xs[5]+xs[9]+xs[13]+xs[17]+xs[21] == 41)
    m.c9  = pyo.Constraint(expr=xs[2]+xs[6]+xs[10]+xs[14]+xs[18]+xs[22] == 13)
    m.c10 = pyo.Constraint(expr=xs[3]+xs[7]+xs[11]+xs[15]+xs[19]+xs[23] == 21)

    # Objective: sum(lin_i*xi - quad_i*xi^2)
    obj_expr = sum(OBJ_LIN[i]*xs[i] - OBJ_QUAD[i]*xs[i]**2 for i in range(24))
    m.obj_expr = obj_expr
    return m


def all_vars_2_1_8(m: pyo.ConcreteModel) -> List[pyo.Var]:
    return [getattr(m, f'x{i}') for i in range(1, 25)]


# =============================================================================
# Scenario generator
# =============================================================================

def build_models_2_1_8(
    nscen: int,
    nfirst: int = 5,
    nparam: int = 5,
    seed: int = 1234,
    print_first_k_rhs: int = 0,
) -> Tuple[List[pyo.ConcreteModel], List[List[pyo.Var]]]:
    rng = JuliaMT19937(seed)

    model_list: List[pyo.ConcreteModel] = []
    first_vars_list: List[List[pyo.Var]] = []

    # Base RHS values for c2-c6 equality constraints
    base_eq = [24.0, 20.0, 24.0, 16.0, 12.0]  # c2, c3, c4, c5, c6

    for s in range(nscen):
        if s == 0:
            # Scenario 1: unperturbed (equality constraints kept as ==)
            m = create_model_2_1_8()
        else:
            # Compute perturbed UB and LB for c2-c6
            # PlasmoOld: for each == constraint, UB noise then LB noise (in that order)
            ubs = []
            lbs = []
            for val in base_eq:
                ubs.append(addnoise_le(val, rng))  # upper bound noise
                lbs.append(addnoise_ge(val, rng))  # lower bound noise
            m = create_model_2_1_8(
                c2_ub=ubs[0], c2_lb=lbs[0],
                c3_ub=ubs[1], c3_lb=lbs[1],
                c4_ub=ubs[2], c4_lb=lbs[2],
                c5_ub=ubs[3], c5_lb=lbs[3],
                c6_ub=ubs[4], c6_lb=lbs[4],
            )

        allv = all_vars_2_1_8(m)
        first = allv[:nfirst]  # x1..x5

        if print_first_k_rhs > 0 and s < print_first_k_rhs:
            if s == 0:
                print(f"[SCEN {s:04d}] c2-c6 == (unperturbed equalities)")
            else:
                print(f"[SCEN {s:04d}] c2=[{lbs[0]:.4f},{ubs[0]:.4f}] c3=[{lbs[1]:.4f},{ubs[1]:.4f}] c4=[{lbs[2]:.4f},{ubs[2]:.4f}] c5=[{lbs[3]:.4f},{ubs[3]:.4f}] c6=[{lbs[4]:.4f},{ubs[4]:.4f}]")

        model_list.append(m)
        first_vars_list.append(first)

    return model_list, first_vars_list


# =============================================================================
# Runner config
# =============================================================================

MODE_PARAMS = {
    "smoke": {
        "nscen": 5,
        "target_nodes": 900,
        "gap_stop_tol": 1e-6,
        "time_limit": 60*10,
        "enable_ef_ub": True,
        "ef_time_ub": 30.0,
        "plot_every": None,
        "plot_output_dir": "results/2_1_8_smoke/plots",
        "output_csv_path": "results/2_1_8_smoke/simplex_result.csv",
    },
    "full": {
        "nscen": 100,
        "target_nodes": 900,
        "gap_stop_tol": 1e-3,
        "time_limit": 60*60*12,
        "enable_ef_ub": True,
        "ef_time_ub": 60,
        "plot_every": None,
        "plot_output_dir": "results/2_1_8_full/plots",
        "output_csv_path": "results/2_1_8_full/simplex_result.csv",
    },
}

BUNDLE_OPTIONS = {"NonConvex": 2, "MIPGap": 1e-2, "TimeLimit": 60}
Q_MAX = 1e4


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
    print("2_1_8 (Python) �?PlasmoOld RandomStochasticModel")
    print(f"Mode: {args.mode}, nscen={cfg['nscen']}, nfirst=5, nparam=5, seed={args.seed}")
    print(f"Bundle options: {BUNDLE_OPTIONS}")
    print("=" * 60)

    t0 = perf_counter()
    model_list, first_vars_list = build_models_2_1_8(
        nscen=cfg["nscen"], seed=args.seed, print_first_k_rhs=args.print_first_k_rhs)
    S = len(model_list)

    base_bundles = [BaseBundle(model_list[s], options=BUNDLE_OPTIONS, q_max=Q_MAX) for s in range(S)]
    ms_bundles   = [MSBundle(model_list[s], first_vars_list[s], options=BUNDLE_OPTIONS) for s in range(S)]

    # === Initial simplex for split_mode=2 ===
    # 6 vertices for d=5, spanning full 5D. Vertices do NOT need to satisfy
    # constraint c1 (x1+x2+x3+x4=8) — that is enforced by the solver
    # during Q/ms evaluations.  The simplex just defines the search domain.
    # Bounds: x1-x4 ∈ [0,8], x5 ∈ [0,24].
    #                       x1    x2    x3    x4    x5
    INITIAL_VERTICES = [
        (0.0,  0.0,  0.0,  0.0,   0.0),   # origin corner
        (8.0,  0.0,  0.0,  0.0,   0.0),   # max x1
        (0.0,  8.0,  0.0,  0.0,   0.0),   # max x2
        (0.0,  0.0,  8.0,  0.0,   0.0),   # max x3
        (0.0,  0.0,  0.0,  8.0,   0.0),   # max x4
        (0.0,  0.0,  0.0,  0.0,  24.0),   # max x5
    ]

    res = run_pid_simplex_3d(
        model_list=model_list,
        first_vars_list=first_vars_list,
        base_bundles=base_bundles,
        ms_bundles=ms_bundles,
        target_nodes=cfg["target_nodes"],
        min_dist=1e-3,
        gap_stop_tol=cfg["gap_stop_tol"],
        time_limit=cfg["time_limit"],
        enable_ef_ub=cfg["enable_ef_ub"],
        ef_time_ub=cfg["ef_time_ub"],
        plot_every=cfg["plot_every"],
        plot_output_dir=cfg["plot_output_dir"],
        output_csv_path=str(out_csv),
        enable_3d_plot=False,
        axis_labels=("x1", "x2", "x3", "x4", "x5"),
        initial_nodes=INITIAL_VERTICES,
        split_mode=2,
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
