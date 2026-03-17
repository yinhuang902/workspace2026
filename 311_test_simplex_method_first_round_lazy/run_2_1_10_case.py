import argparse
from pathlib import Path
from time import perf_counter
from typing import List, Tuple

import pyomo.environ as pyo

from bundles import BaseBundle, MSBundle
from simplex_specialstart import run_pid_simplex_3d


# =============================================================================
# Julia-equivalent RNG + noise (Plasmo.jl RandomStochasticModel)
# =============================================================================

class JuliaMT19937:
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
# 2_1_10 deterministic base model (exact translation of Julia createModel())
# =============================================================================

def create_model_2_1_10() -> pyo.ConcreteModel:
    m = pyo.ConcreteModel()

    # Variables x1..x20 >= 0
    # Julia has starts for x6 and x14
    m.x1 = pyo.Var(bounds=(0, 200.0), initialize=0)
    m.x2 = pyo.Var(bounds=(0, 200.0), initialize=0)
    m.x3 = pyo.Var(bounds=(0, None), initialize=0)
    m.x4 = pyo.Var(bounds=(0, None), initialize=0)
    m.x5 = pyo.Var(bounds=(0, None), initialize=0)
    m.x6 = pyo.Var(bounds=(0, None), initialize=4.348)
    m.x7 = pyo.Var(bounds=(0, None), initialize=0)
    m.x8  = pyo.Var(bounds=(0, None), initialize=0)
    m.x9  = pyo.Var(bounds=(0, None), initialize=0)
    m.x10 = pyo.Var(bounds=(0, None), initialize=0)
    m.x11 = pyo.Var(bounds=(0, None), initialize=0)
    m.x12 = pyo.Var(bounds=(0, None), initialize=0)
    m.x13 = pyo.Var(bounds=(0, None), initialize=0)
    m.x14 = pyo.Var(bounds=(0, None), initialize=62.609)
    m.x15 = pyo.Var(bounds=(0, None), initialize=0)
    m.x16 = pyo.Var(bounds=(0, None), initialize=0)
    m.x17 = pyo.Var(bounds=(0, None), initialize=0)
    m.x18 = pyo.Var(bounds=(0, None), initialize=0)
    m.x19 = pyo.Var(bounds=(0, None), initialize=0)
    m.x20 = pyo.Var(bounds=(0, None), initialize=0)

    # Constraint 1 RHS is the one modified by RandomStochasticModel (nparam=2 => only 1 modification)
    m.c1_rhs = pyo.Param(mutable=True, initialize=380.0)
    m.c1 = pyo.Constraint(expr=
        3*m.x1 + 5*m.x2 + 5*m.x3 + 6*m.x4 + 4*m.x5 + 4*m.x6 + 5*m.x7 + 6*m.x8 + 4*m.x9 + 4*m.x10
        + 8*m.x11 + 4*m.x12 + 2*m.x13 + 1*m.x14 + 1*m.x15 + 1*m.x16 + 2*m.x17 + 1*m.x18 + 7*m.x19 + 3*m.x20
        <= m.c1_rhs
    )

    m.c2 = pyo.Constraint(expr=
        5*m.x1 + 4*m.x2 + 5*m.x3 + 4*m.x4 + 1*m.x5 + 4*m.x6 + 4*m.x7 + 2*m.x8 + 5*m.x9 + 2*m.x10
        + 3*m.x11 + 6*m.x12 + 1*m.x13 + 7*m.x14 + 7*m.x15 + 5*m.x16 + 8*m.x17 + 7*m.x18 + 2*m.x19 + 1*m.x20
        <= 415
    )

    m.c3 = pyo.Constraint(expr=
        1*m.x1 + 5*m.x2 + 2*m.x3 + 4*m.x4 + 7*m.x5 + 3*m.x6 + 1*m.x7 + 5*m.x8 + 7*m.x9 + 6*m.x10
        + 1*m.x11 + 7*m.x12 + 2*m.x13 + 4*m.x14 + 7*m.x15 + 5*m.x16 + 3*m.x17 + 4*m.x18 + 1*m.x19 + 2*m.x20
        <= 385
    )

    m.c4 = pyo.Constraint(expr=
        3*m.x1 + 2*m.x2 + 6*m.x3 + 3*m.x4 + 2*m.x5 + 1*m.x6 + 6*m.x7 + 1*m.x8 + 7*m.x9 + 3*m.x10
        + 7*m.x11 + 7*m.x12 + 8*m.x13 + 2*m.x14 + 3*m.x15 + 4*m.x16 + 5*m.x17 + 8*m.x18 + 1*m.x19 + 2*m.x20
        <= 405
    )

    m.c5 = pyo.Constraint(expr=
        6*m.x1 + 6*m.x2 + 6*m.x3 + 4*m.x4 + 5*m.x5 + 2*m.x6 + 2*m.x7 + 4*m.x8 + 3*m.x9 + 2*m.x10
        + 7*m.x11 + 5*m.x12 + 3*m.x13 + 6*m.x14 + 7*m.x15 + 5*m.x16 + 8*m.x17 + 4*m.x18 + 6*m.x19 + 3*m.x20
        <= 470
    )

    m.c6 = pyo.Constraint(expr=
        5*m.x1 + 5*m.x2 + 2*m.x3 + 1*m.x4 + 3*m.x5 + 5*m.x6 + 5*m.x7 + 7*m.x8 + 4*m.x9 + 3*m.x10
        + 4*m.x11 + 1*m.x12 + 7*m.x13 + 3*m.x14 + 8*m.x15 + 3*m.x16 + 1*m.x17 + 6*m.x18 + 2*m.x19 + 8*m.x20
        <= 415
    )

    m.c7 = pyo.Constraint(expr=
        3*m.x1 + 6*m.x2 + 6*m.x3 + 3*m.x4 + 1*m.x5 + 6*m.x6 + 1*m.x7 + 6*m.x8 + 7*m.x9 + 1*m.x10
        + 4*m.x11 + 3*m.x12 + 1*m.x13 + 4*m.x14 + 3*m.x15 + 6*m.x16 + 4*m.x17 + 6*m.x18 + 5*m.x19 + 4*m.x20
        <= 400
    )

    m.c8 = pyo.Constraint(expr=
        1*m.x1 + 2*m.x2 + 1*m.x3 + 7*m.x4 + 8*m.x5 + 7*m.x6 + 6*m.x7 + 5*m.x8 + 8*m.x9 + 7*m.x10
        + 2*m.x11 + 3*m.x12 + 5*m.x13 + 5*m.x14 + 4*m.x15 + 5*m.x16 + 4*m.x17 + 2*m.x18 + 2*m.x19 + 8*m.x20
        <= 460
    )

    m.c9 = pyo.Constraint(expr=
        8*m.x1 + 5*m.x2 + 2*m.x3 + 5*m.x4 + 3*m.x5 + 8*m.x6 + 1*m.x7 + 3*m.x8 + 3*m.x9 + 5*m.x10
        + 4*m.x11 + 5*m.x12 + 5*m.x13 + 6*m.x14 + 1*m.x15 + 7*m.x16 + 1*m.x17 + 2*m.x18 + 2*m.x19 + 4*m.x20
        <= 400
    )

    m.c10 = pyo.Constraint(expr=
        m.x1 + m.x2 + m.x3 + m.x4 + m.x5 + m.x6 + m.x7 + m.x8 + m.x9 + m.x10
        + m.x11 + m.x12 + m.x13 + m.x14 + m.x15 + m.x16 + m.x17 + m.x18 + m.x19 + m.x20
        <= 200
    )

    # Objective (nonconvex difference of convex quadratics)
    pos = (
        42*(52 + m.x11)**2 + 98*(3 + m.x12)**2 + 48*(m.x13 - 81)**2 + 91*(m.x14 - 30)**2
        + 11*(85 + m.x15)**2 + 63*(m.x16 - 68)**2 + 61*(m.x17 - 27)**2 + 61*(81 + m.x18)**2
        + 38*(m.x19 - 97)**2 + 26*(73 + m.x20)**2
    )
    neg = (
        63*(19 + m.x1)**2 + 15*(27 + m.x2)**2 + 44*(23 + m.x3)**2 + 91*(53 + m.x4)**2
        + 45*(42 + m.x5)**2 + 50*(m.x6 - 26)**2 + 89*(33 + m.x7)**2 + 58*(23 + m.x8)**2
        + 86*(m.x9 - 41)**2 + 82*(m.x10 - 19)**2
    )
    m.obj_expr = 0.5*pos - 0.5*neg

    return m


def all_vars_2_1_10(m: pyo.ConcreteModel) -> List[pyo.Var]:
    return [
        m.x1, m.x2, m.x3, m.x4, m.x5, m.x6, m.x7, m.x8, m.x9, m.x10,
        m.x11, m.x12, m.x13, m.x14, m.x15, m.x16, m.x17, m.x18, m.x19, m.x20
    ]


# =============================================================================
# Scenario generator matching Plasmo.jl RandomStochasticModel(createModel, 1000, 2, 2)
# For 2_1_10: only the first eligible linear constraint bound is modified: c1_rhs (380 -> 380*U(0.5,2.0))
# =============================================================================

def build_models_2_1_10(
    nscen: int,
    nfirst: int = 2,
    nparam: int = 2,
    seed: int = 1234,
    print_first_k_rhs: int = 0,
) -> Tuple[List[pyo.ConcreteModel], List[List[pyo.Var]]]:
    rng = JuliaMT19937(seed)

    model_list: List[pyo.ConcreteModel] = []
    first_vars_list: List[List[pyo.Var]] = []

    max_mods = max(nparam - 1, 0)  # nparam=2 => 1 modification per scenario

    for s in range(nscen):
        m = create_model_2_1_10()
        allv = all_vars_2_1_10(m)
        first = allv[:nfirst]  # x1, x2

        if max_mods >= 1:
            base_ub = float(pyo.value(m.c1_rhs))  # 380.0
            m.c1_rhs.set_value(addnoise_julia(base_ub, rng))

        if print_first_k_rhs > 0 and s < print_first_k_rhs:
            print(f"[SCEN {s:04d}] c1_rhs = {float(pyo.value(m.c1_rhs)):.12f}")

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
        "plot_output_dir": "results/2_1_10_smoke/plots",
        "output_csv_path": "results/2_1_10_smoke/simplex_result.csv",
    },
    "full": {
        "nscen": 1000,
        "target_nodes": 300,
        "gap_stop_tol": 1e-2,
        "time_limit": None,
        "enable_ef_ub": True,
        "ef_time_ub": 43200.0,
        "plot_every": None,
        "plot_output_dir": "results/2_1_10_full/plots",
        "output_csv_path": "results/2_1_10_full/simplex_result.csv",
    },
}

# Nonconvex QP => Gurobi NonConvex=2
BUNDLE_OPTIONS = {
    "NonConvex": 2,
    "MIPGap": 1e-1,
    # "TimeLimit": 60,  # optional per-solve cap
}
Q_MAX = 1e10


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
    print("2_1_10 (Python) â€?scenario generation matches Plasmo.jl RandomStochasticModel")
    print(f"Mode: {args.mode}")
    print(f"nscen={cfg['nscen']}, seed={args.seed}, target_nodes={cfg['target_nodes']}")
    print(f"gap_stop_tol={cfg['gap_stop_tol']}, time_limit={cfg['time_limit']}")
    print(f"EF UB enabled={cfg['enable_ef_ub']}, ef_time_ub={cfg['ef_time_ub']}")
    print(f"Bundle options: {BUNDLE_OPTIONS}")
    print("=" * 60)

    t0 = perf_counter()

    model_list, first_vars_list = build_models_2_1_10(
        nscen=cfg["nscen"], nfirst=2, nparam=2,
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
        enable_3d_plot=False,  # first-stage dim = 2
        axis_labels=("x1", "x2"),
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