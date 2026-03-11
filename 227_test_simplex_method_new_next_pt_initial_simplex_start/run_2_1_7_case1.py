import argparse
from pathlib import Path
from time import perf_counter
from typing import List, Tuple

import pyomo.environ as pyo

from bundles import BaseBundle, MSBundle
from simplex_specialstart import run_pid_simplex_3d


# =============================================================================
# Julia-equivalent RNG (MT19937) — same class as run_2_1_1_case.py
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


# =============================================================================
# PlasmoOld.jl addnoise(a, adl, adu, rdl, rdu)
#   if a == 0: a + Uniform(adl, adu)
#   else:      a + abs(a) * Uniform(rdl, rdu)
#
# Defaults from PlasmoOld.RandomStochasticModel:
#   rdl=0.0, rdu=2.0, adl=-10, adu=10
# =============================================================================

def addnoise_plasmoold(a: float, rng: JuliaMT19937,
                       adl: float = -10.0, adu: float = 10.0,
                       rdl: float = 0.0, rdu: float = 2.0) -> float:
    if a == 0.0:
        return a + rng.rand_uniform(adl, adu)
    return a + abs(a) * rng.rand_uniform(rdl, rdu)


# =============================================================================
# 2_1_7 deterministic base model: exact translation of Julia createModel()
# =============================================================================

def create_model_2_1_7() -> pyo.ConcreteModel:
    m = pyo.ConcreteModel()

    # NOTE: Julia only has x>=0, but constraint sum(x)<=40 implies each x<=40.
    # We set ub=40 explicitly so simplex can form initial corner nodes.
    UB = 40.0

    m.x1  = pyo.Var(bounds=(0, UB))
    m.x2  = pyo.Var(bounds=(0, UB))
    m.x3  = pyo.Var(bounds=(0, UB), initialize=1.04289)
    m.x4  = pyo.Var(bounds=(0, UB))
    m.x5  = pyo.Var(bounds=(0, UB))
    m.x6  = pyo.Var(bounds=(0, UB))
    m.x7  = pyo.Var(bounds=(0, UB))
    m.x8  = pyo.Var(bounds=(0, UB))
    m.x9  = pyo.Var(bounds=(0, UB))
    m.x10 = pyo.Var(bounds=(0, UB))
    m.x11 = pyo.Var(bounds=(0, UB), initialize=1.74674)
    m.x12 = pyo.Var(bounds=(0, UB))
    m.x13 = pyo.Var(bounds=(0, UB), initialize=0.43147)
    m.x14 = pyo.Var(bounds=(0, UB))
    m.x15 = pyo.Var(bounds=(0, UB))
    m.x16 = pyo.Var(bounds=(0, UB), initialize=4.43305)
    m.x17 = pyo.Var(bounds=(0, UB))
    m.x18 = pyo.Var(bounds=(0, UB), initialize=15.85893)
    m.x19 = pyo.Var(bounds=(0, UB))
    m.x20 = pyo.Var(bounds=(0, UB), initialize=16.4889)

    # Make RHS mutable params so scenario generator can perturb them
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

    m.c1 = pyo.Constraint(expr=(
        -3*m.x1 + 7*m.x2 - 5*m.x4 + m.x5 + m.x6 + 2*m.x8 - m.x9 - m.x10
        - 9*m.x11 + 3*m.x12 + 5*m.x13 + m.x16 + 7*m.x17 - 7*m.x18 - 4*m.x19 - 6*m.x20
        <= m.c1_rhs
    ))
    m.c2 = pyo.Constraint(expr=(
        7*m.x1 - 5*m.x3 + m.x4 + m.x5 + 2*m.x7 - m.x8 - m.x9 - 9*m.x10
        + 3*m.x11 + 5*m.x12 + m.x15 + 7*m.x16 - 7*m.x17 - 4*m.x18 - 6*m.x19 - 3*m.x20
        <= m.c2_rhs
    ))
    m.c3 = pyo.Constraint(expr=(
        -5*m.x2 + m.x3 + m.x4 + 2*m.x6 - m.x7 - m.x8 - 9*m.x9 + 3*m.x10 + 5*m.x11
        + m.x14 + 7*m.x15 - 7*m.x16 - 4*m.x17 - 6*m.x18 - 3*m.x19 + 7*m.x20
        <= m.c3_rhs
    ))
    m.c4 = pyo.Constraint(expr=(
        -5*m.x1 + m.x2 + m.x3 + 2*m.x5 - m.x6 - m.x7 - 9*m.x8 + 3*m.x9 + 5*m.x10
        + m.x13 + 7*m.x14 - 7*m.x15 - 4*m.x16 - 6*m.x17 - 3*m.x18 + 7*m.x19
        <= m.c4_rhs
    ))
    m.c5 = pyo.Constraint(expr=(
        m.x1 + m.x2 + 2*m.x4 - m.x5 - m.x6 - 9*m.x7 + 3*m.x8 + 5*m.x9 + m.x12
        + 7*m.x13 - 7*m.x14 - 4*m.x15 - 6*m.x16 - 3*m.x17 + 7*m.x18 - 5*m.x20
        <= m.c5_rhs
    ))
    m.c6 = pyo.Constraint(expr=(
        m.x1 + 2*m.x3 - m.x4 - m.x5 - 9*m.x6 + 3*m.x7 + 5*m.x8 + m.x11
        + 7*m.x12 - 7*m.x13 - 4*m.x14 - 6*m.x15 - 3*m.x16 + 7*m.x17 - 5*m.x19 + m.x20
        <= m.c6_rhs
    ))
    m.c7 = pyo.Constraint(expr=(
        2*m.x2 - m.x3 - m.x4 - 9*m.x5 + 3*m.x6 + 5*m.x7 + m.x10 + 7*m.x11
        - 7*m.x12 - 4*m.x13 - 6*m.x14 - 3*m.x15 + 7*m.x16 - 5*m.x18 + m.x19 + m.x20
        <= m.c7_rhs
    ))
    m.c8 = pyo.Constraint(expr=(
        2*m.x1 - m.x2 - m.x3 - 9*m.x4 + 3*m.x5 + 5*m.x6 + m.x9 + 7*m.x10
        - 7*m.x11 - 4*m.x12 - 6*m.x13 - 3*m.x14 + 7*m.x15 - 5*m.x17 + m.x18 + m.x19
        <= m.c8_rhs
    ))
    m.c9 = pyo.Constraint(expr=(
        -m.x1 - m.x2 - 9*m.x3 + 3*m.x4 + 5*m.x5 + m.x8 + 7*m.x9 - 7*m.x10
        - 4*m.x11 - 6*m.x12 - 3*m.x13 + 7*m.x14 - 5*m.x16 + m.x17 + m.x18 + 2*m.x20
        <= m.c9_rhs
    ))
    m.c10 = pyo.Constraint(expr=(
        m.x1 + m.x2 + m.x3 + m.x4 + m.x5 + m.x6 + m.x7 + m.x8 + m.x9 + m.x10
        + m.x11 + m.x12 + m.x13 + m.x14 + m.x15 + m.x16 + m.x17 + m.x18 + m.x19 + m.x20
        <= m.c10_rhs
    ))

    xs = [
        m.x1, m.x2, m.x3, m.x4, m.x5, m.x6, m.x7, m.x8, m.x9, m.x10,
        m.x11, m.x12, m.x13, m.x14, m.x15, m.x16, m.x17, m.x18, m.x19, m.x20
    ]
    quad_sum = sum((i + 1) * (xs[i] - 2.0) ** 2 for i in range(20))

    # IMPORTANT:
    # bundles.BaseBundle expects model.obj_expr to exist and will build model.obj itself.
    # So do NOT create an Objective here; just provide the expression.
    m.obj_expr = -0.5 * quad_sum

    return m


def all_vars_2_1_7(m: pyo.ConcreteModel) -> List[pyo.Var]:
    return [
        m.x1, m.x2, m.x3, m.x4, m.x5, m.x6, m.x7, m.x8, m.x9, m.x10,
        m.x11, m.x12, m.x13, m.x14, m.x15, m.x16, m.x17, m.x18, m.x19, m.x20
    ]


# =============================================================================
# Scenario generator matching PlasmoOld.RandomStochasticModel(createModel, NS)
# From PlasmoOld.jl:
#   srand(1234)
#   for i=1:nscen:
#     node=createModel()
#     if i==1: continue
#     iterate linear constraints, skip those using only firstVars
#     perturb bounds until nmodified >= nparam
#
# Here: firstVars = x1..x5 (nfirst=5), nparam=5, so we perturb c1..c5 RHS for scen>=2.
# =============================================================================

def build_models_2_1_7(
    nscen: int,
    nfirst: int = 5,
    nparam: int = 5,
    seed: int = 1234,
    rdl: float = 0.0,
    rdu: float = 2.0,
    adl: float = -10.0,
    adu: float = 10.0,
    print_first_k_rhs: int = 0,
) -> Tuple[List[pyo.ConcreteModel], List[List[pyo.Var]]]:
    rng = JuliaMT19937(seed)

    model_list: List[pyo.ConcreteModel] = []
    first_vars_list: List[List[pyo.Var]] = []

    for s in range(nscen):
        m = create_model_2_1_7()
        allv = all_vars_2_1_7(m)
        first = allv[:nfirst]  # x1..x5

        # PlasmoOld: scen 1 (i==1) unchanged
        if s >= 1:
            nmodified = 0

            # Constraints are all <= ub, and all involve vars beyond firstVars => eligible.
            # We perturb in order until nmodified >= nparam.
            rhs_params = [m.c1_rhs, m.c2_rhs, m.c3_rhs, m.c4_rhs, m.c5_rhs,
                          m.c6_rhs, m.c7_rhs, m.c8_rhs, m.c9_rhs, m.c10_rhs]
            for p in rhs_params:
                if nmodified >= nparam:
                    break
                base = float(pyo.value(p))
                p.set_value(addnoise_plasmoold(base, rng, adl=adl, adu=adu, rdl=rdl, rdu=rdu))
                nmodified += 1

        if print_first_k_rhs > 0 and s < print_first_k_rhs:
            print(
                f"[SCEN {s:04d}] "
                f"c1={float(pyo.value(m.c1_rhs)):.8f} "
                f"c2={float(pyo.value(m.c2_rhs)):.8f} "
                f"c3={float(pyo.value(m.c3_rhs)):.8f} "
                f"c4={float(pyo.value(m.c4_rhs)):.8f} "
                f"c5={float(pyo.value(m.c5_rhs)):.8f}"
            )

        model_list.append(m)
        first_vars_list.append(first)

    return model_list, first_vars_list


# =============================================================================
# Runner config — same style as run_2_1_1_case.py
# setup.jl: NS=100
# solver.jl: RandomStochasticModel(createModel, NS)  (defaults nfirst=5, nparam=5, seed=1234 inside)
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
        "plot_output_dir": "results/2_1_7_smoke/plots",
        "output_csv_path": "results/2_1_7_smoke/simplex_result.csv",
    },
    "full": {
        "nscen": 100,          # NS=100 from setup.jl
        "target_nodes": 300,
        "gap_stop_tol": 1e-2,
        "time_limit": None,
        "enable_ef_ub": True,
        "ef_time_ub": 43200.0,  # aligns with other Julia scripts
        "plot_every": None,
        "plot_output_dir": "results/2_1_7_full/plots",
        "output_csv_path": "results/2_1_7_full/simplex_result.csv",
    },
}

BUNDLE_OPTIONS = {
    "NonConvex": 2,   # Gurobi nonconvex QP
    "MIPGap": 1e-1,
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke", "full"), default="smoke")
    ap.add_argument("--seed", type=int, default=1234,
                    help="PlasmoOld.RandomStochasticModel uses srand(1234) internally; default matches Julia.")
    ap.add_argument("--print_first_k_rhs", type=int, default=0)
    args = ap.parse_args()

    cfg = dict(MODE_PARAMS[args.mode])

    out_csv = Path(cfg["output_csv_path"])
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    if cfg["plot_output_dir"] is not None:
        Path(cfg["plot_output_dir"]).mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("2_1_7 (Python) — scenario generation matches PlasmoOld.RandomStochasticModel(createModel, NS)")
    print(f"Mode: {args.mode}")
    print(f"nscen={cfg['nscen']}, seed={args.seed}, target_nodes={cfg['target_nodes']}")
    print(f"gap_stop_tol={cfg['gap_stop_tol']}, time_limit={cfg['time_limit']}")
    print(f"EF UB enabled={cfg['enable_ef_ub']}, ef_time_ub={cfg['ef_time_ub']}")
    print(f"Bundle options: {BUNDLE_OPTIONS}")
    print("PlasmoOld defaults used: nfirst=5, nparam=5, rdl=0.0, rdu=2.0, adl=-10, adu=10")
    print("=" * 60)

    t0 = perf_counter()

    model_list, first_vars_list = build_models_2_1_7(
        nscen=cfg["nscen"],
        nfirst=5,
        nparam=5,
        seed=args.seed,
        rdl=0.0, rdu=2.0, adl=-10.0, adu=10.0,
        print_first_k_rhs=args.print_first_k_rhs,
    )
    S = len(model_list)

    base_bundles = [BaseBundle(model_list[s], options=BUNDLE_OPTIONS) for s in range(S)]
    ms_bundles = [MSBundle(model_list[s], first_vars_list[s], options=BUNDLE_OPTIONS) for s in range(S)]

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
        enable_3d_plot=False,  # dim=5, no 3D plot
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