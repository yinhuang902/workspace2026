"""
run_abel_sngo_case.py  -- Simplex runner for SNGO-master/Global/abel

29 JuMP variables (x2-x8, x10-x30; x1=387.9, x9=85.3 are fixed constants).
14 equality constraints (==). Quadratic least-squares objective.

Stage split: nfirst=5, first=[x2,x3,x4,x5,x6].
Perturbed: first 4 == constraints (RHS=-59.4), addnoise_julia.
"""
import argparse
from pathlib import Path
from time import perf_counter
from typing import List, Tuple
import pyomo.environ as pyo
from bundles import BaseBundle, MSBundle
from simplex_specialstart import run_pid_simplex_3d
from run_st_fp7a_case import JuliaMT19937

# -- addnoise for == constraints (from PlasmoOld) --
def addnoise_julia(a: float, rng: JuliaMT19937) -> float:
    if a == 0.0:
        return a + rng.rand_uniform(-10.0, 10.0)
    return a * rng.rand_uniform(0.5, 2.0)


# Fixed constants from Julia
_X1 = 387.9
_X9 = 85.3

_RHS_BASE = [
    ("c1_rhs", -59.4), ("c2_rhs", -59.4), ("c3_rhs", -59.4), ("c4_rhs", -59.4),
    ("c5_rhs", -59.4), ("c6_rhs", -59.4), ("c7_rhs", -59.4),
    ("c8_rhs", -184.7), ("c9_rhs", -184.7), ("c10_rhs", -184.7), ("c11_rhs", -184.7),
    ("c12_rhs", -184.7), ("c13_rhs", -184.7), ("c14_rhs", -184.7),
]

def create_model():
    m = pyo.ConcreteModel()
    # Variables (x1, x9 are fixed constants, not Pyomo Vars)
    for i in [2,3,4,5,6,7,8]:
        if i <= 6:
            setattr(m, f"x{i}", pyo.Var(bounds=(0, 1000), initialize=387.9))
        else:
            setattr(m, f"x{i}", pyo.Var(initialize=387.9))
    for i in range(10, 17):
        setattr(m, f"x{i}", pyo.Var(initialize=85.3))
    for i in range(17, 24):
        setattr(m, f"x{i}", pyo.Var(initialize=110.5))
    for i in range(24, 31):
        setattr(m, f"x{i}", pyo.Var(initialize=147.1))

    for name, val in _RHS_BASE:
        setattr(m, name, pyo.Param(mutable=True, initialize=val))

    # Shorthand
    x = {i: getattr(m, f"x{i}") for i in list(range(2,9))+list(range(10,31))}

    # 7 constraints of form: -0.914*x_{k} + x_{k+1} + 0.016*x_{k+8} - 0.305*x_{k+15} - 0.424*x_{k+22} == RHS
    # (first block uses fixed x1, x9)
    m.c1  = pyo.Constraint(expr=-0.914*_X1 + x[2] + 0.016*_X9 - 0.305*x[17] - 0.424*x[24] == m.c1_rhs)
    m.c2  = pyo.Constraint(expr=-0.914*x[2] + x[3] + 0.016*x[10] - 0.305*x[18] - 0.424*x[25] == m.c2_rhs)
    m.c3  = pyo.Constraint(expr=-0.914*x[3] + x[4] + 0.016*x[11] - 0.305*x[19] - 0.424*x[26] == m.c3_rhs)
    m.c4  = pyo.Constraint(expr=-0.914*x[4] + x[5] + 0.016*x[12] - 0.305*x[20] - 0.424*x[27] == m.c4_rhs)
    m.c5  = pyo.Constraint(expr=-0.914*x[5] + x[6] + 0.016*x[13] - 0.305*x[21] - 0.424*x[28] == m.c5_rhs)
    m.c6  = pyo.Constraint(expr=-0.914*x[6] + x[7] + 0.016*x[14] - 0.305*x[22] - 0.424*x[29] == m.c6_rhs)
    m.c7  = pyo.Constraint(expr=-0.914*x[7] + x[8] + 0.016*x[15] - 0.305*x[23] - 0.424*x[30] == m.c7_rhs)

    # 7 constraints of form: -0.097*x_{k} - 0.424*x_{k+8} + x_{k+9} + 0.101*x_{k+15} - 1.459*x_{k+22} == -184.7
    m.c8  = pyo.Constraint(expr=-0.097*_X1 - 0.424*_X9 + x[10] + 0.101*x[17] - 1.459*x[24] == m.c8_rhs)
    m.c9  = pyo.Constraint(expr=-0.097*x[2] - 0.424*x[10] + x[11] + 0.101*x[18] - 1.459*x[25] == m.c9_rhs)
    m.c10 = pyo.Constraint(expr=-0.097*x[3] - 0.424*x[11] + x[12] + 0.101*x[19] - 1.459*x[26] == m.c10_rhs)
    m.c11 = pyo.Constraint(expr=-0.097*x[4] - 0.424*x[12] + x[13] + 0.101*x[20] - 1.459*x[27] == m.c11_rhs)
    m.c12 = pyo.Constraint(expr=-0.097*x[5] - 0.424*x[13] + x[14] + 0.101*x[21] - 1.459*x[28] == m.c12_rhs)
    m.c13 = pyo.Constraint(expr=-0.097*x[6] - 0.424*x[14] + x[15] + 0.101*x[22] - 1.459*x[29] == m.c13_rhs)
    m.c14 = pyo.Constraint(expr=-0.097*x[7] - 0.424*x[15] + x[16] + 0.101*x[23] - 1.459*x[30] == m.c14_rhs)

    # Objective (from Julia): quadratic least-squares
    # The Julia objective is a complex sum of squared differences; translating exactly:
    m.obj_expr = (
        0.5*((0.0625*_X1 - 24.24375)*(_X1 - 387.9) + (_X9 - 85.3)*(_X9 - 85.3)
        + (0.0625*x[2] - 24.425578125)*(x[2] - 390.80925) + (x[10] - 85.93975)*(x[10] - 85.93975)
        + (0.0625*x[3] - 24.6087699609375)*(x[3] - 393.740319375) + (x[11] - 86.584298125)*(x[11] - 86.584298125)
        + (0.0625*x[4] - 24.7933357356445)*(x[4] - 396.693371770313) + (x[12] - 87.2336803609375)*(x[12] - 87.2336803609375)
        + (0.0625*x[5] - 24.9792857536619)*(x[5] - 399.66857205859) + (x[13] - 87.8879329636445)*(x[13] - 87.8879329636445)
        + (0.0625*x[6] - 25.1666303968143)*(x[6] - 402.666086349029) + (x[14] - 88.5470924608719)*(x[14] - 88.5470924608719)
        + (0.0625*x[7] - 25.3553801247904)*(x[7] - 405.686081996647) + (x[15] - 89.2111956543284)*(x[15] - 89.2111956543284)
        + (6.25*x[8] - 2554.55454757264)*(x[8] - 408.728727611622) + (100*x[16] - 8988.02796217359)*(x[16] - 89.8802796217359))
        + 0.5*((x[17] - 110.5)*(x[17] - 110.5) + (0.444*x[24] - 65.3124)*(x[24] - 147.1)
        + (x[18] - 111.32875)*(x[18] - 111.32875) + (0.444*x[25] - 65.802243)*(x[25] - 148.20325)
        + (x[19] - 112.163715625)*(x[19] - 112.163715625) + (0.444*x[26] - 66.2957598225)*(x[26] - 149.314774375)
        + (x[20] - 113.004943492188)*(x[20] - 113.004943492188) + (0.444*x[27] - 66.7929780211688)*(x[27] - 150.434635182813)
        + (x[21] - 113.852480568379)*(x[21] - 113.852480568379) + (0.444*x[28] - 67.2939253563275)*(x[28] - 151.562894946684)
        + (x[22] - 114.706374172642)*(x[22] - 114.706374172642) + (0.444*x[29] - 67.7986297965)*(x[29] - 152.699616658784)
        + (x[23] - 115.566671978937)*(x[23] - 115.566671978937) + (0.444*x[30] - 68.3071195199738)*(x[30] - 153.844863783725))
    )
    return m


def all_vars(m):
    return [getattr(m, f"x{i}") for i in list(range(2,9))+list(range(10,31))]


def build_models(nscen, nfirst=5, nparam=5, seed=1234, **kw):
    rng = JuliaMT19937(seed); ml = []; fl = []; mx = nparam
    for s in range(nscen):
        m = create_model(); av = all_vars(m); f = av[:nfirst]
        if s > 0:
            for idx in range(min(mx, len(_RHS_BASE))):
                p, bv = _RHS_BASE[idx]
                getattr(m, p).set_value(addnoise_julia(bv, rng))
        ml.append(m); fl.append(f)
    return ml, fl


MODE_PARAMS = {
    "smoke": {"nscen":10,"target_nodes":60,"gap_stop_tol":1e-6,"time_limit":300,"enable_ef_ub":True,"ef_time_ub":30.,"plot_every":None,
              "plot_output_dir":"results/abel_smoke/plots","output_csv_path":"results/abel_smoke/simplex_result.csv"},
    "full":  {"nscen":100,"target_nodes":300,"gap_stop_tol":1e-2,"time_limit":None,"enable_ef_ub":True,"ef_time_ub":43200.,"plot_every":None,
              "plot_output_dir":"results/abel_full/plots","output_csv_path":"results/abel_full/simplex_result.csv"},
}
BUNDLE_OPTIONS = {"NonConvex": 2, "MIPGap": 1e-1}
Q_MAX = 1e10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke","full"), default="smoke")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--nfirst", type=int, default=5)
    args = ap.parse_args(); nf = args.nfirst; cfg = dict(MODE_PARAMS[args.mode])
    out = Path(cfg["output_csv_path"]); out.parent.mkdir(parents=True, exist_ok=True)
    if cfg["plot_output_dir"]: Path(cfg["plot_output_dir"]).mkdir(parents=True, exist_ok=True)
    print("="*60 + f"\nabel -- nfirst={nf}, seed={args.seed}\n" + "="*60)
    t0 = perf_counter()
    ml, fl = build_models(cfg["nscen"], nf, nf, args.seed); S = len(ml)
    bb = [BaseBundle(ml[s], options=BUNDLE_OPTIONS, q_max=Q_MAX) for s in range(S)]
    mb = [MSBundle(ml[s], fl[s], options=BUNDLE_OPTIONS) for s in range(S)]
    res = run_pid_simplex_3d(model_list=ml, first_vars_list=fl, base_bundles=bb, ms_bundles=mb,
        target_nodes=cfg["target_nodes"], min_dist=1e-3, gap_stop_tol=cfg["gap_stop_tol"], time_limit=cfg["time_limit"],
        enable_ef_ub=cfg["enable_ef_ub"], ef_time_ub=cfg["ef_time_ub"], plot_every=cfg["plot_every"],
        plot_output_dir=cfg["plot_output_dir"], output_csv_path=str(out), enable_3d_plot=False,
        axis_labels=tuple(f"x{i}" for i in [2,3,4,5,6][:nf]))
    t1 = perf_counter(); LB = res.get("LB_hist",[]); UB = res.get("UB_hist",[])
    if LB and UB: print(f"\nLB_sum={float(LB[-1]):.12f} UB_sum={float(UB[-1]):.12f}\nLB/S={float(LB[-1])/S:.12f} UB/S={float(UB[-1])/S:.12f}")
    print(f"{'='*60}\nDone. {t1-t0:.2f}s CSV: {out}\n{'='*60}")


if __name__ == "__main__": main()
