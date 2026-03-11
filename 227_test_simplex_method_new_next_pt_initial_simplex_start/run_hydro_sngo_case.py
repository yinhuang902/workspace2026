"""
run_hydro_sngo_case.py  -- Simplex runner for SNGO-master/Global/hydro

30 JuMP variables (x1-x24, x26-x31; x25=100000 fixed).
6 >= constraints, 6 == constraints, 6 quadratic == constraints, 6 == constraints.
Quadratic objective.

Stage split: nfirst=5, first=[x1,x2,x3,x4,x5].
Perturbed: first 4 >= constraints (RHS=1200,1500,1100,1800), addnoise_ge.
"""
import argparse
from pathlib import Path
from time import perf_counter
import pyomo.environ as pyo
from bundles import BaseBundle, MSBundle
from simplex_specialstart import run_pid_simplex_3d
from run_st_fp7a_case import JuliaMT19937

def addnoise_ge(a: float, rng: JuliaMT19937) -> float:
    """PlasmoOld addnoise for >= / lower bound: addnoise(a, -10, 0, -2, 0)"""
    if a == 0.0:
        return a + rng.rand_uniform(-10.0, 0.0)
    return a + abs(a) * rng.rand_uniform(-2.0, 0.0)

_X25 = 100000.0  # fixed constant

_RHS_BASE = [
    ("c1_rhs", 1200., "ge"), ("c2_rhs", 1500., "ge"),
    ("c3_rhs", 1100., "ge"), ("c4_rhs", 1800., "ge"),
    ("c5_rhs",  950., "ge"), ("c6_rhs", 1300., "ge"),
    ("c7_rhs", 24000., "eq"), ("c8_rhs", 24000., "eq"),
    ("c9_rhs", 24000., "eq"), ("c10_rhs", 24000., "eq"),
    ("c11_rhs", 24000., "eq"), ("c12_rhs", 24000., "eq"),
]

def create_model():
    m = pyo.ConcreteModel()
    for i in range(1, 7):
        setattr(m, f"x{i}", pyo.Var(bounds=(150, 1500), initialize=150))
    for i in range(7, 13):
        setattr(m, f"x{i}", pyo.Var(bounds=(0, 1000), initialize=0))
    for i in range(13, 25):
        setattr(m, f"x{i}", pyo.Var(bounds=(0, None), initialize=0))
    # x25 is fixed = 100000
    for i in range(26, 32):
        setattr(m, f"x{i}", pyo.Var(bounds=(60000, 120000), initialize=60000))

    for name, val, _ in _RHS_BASE:
        setattr(m, name, pyo.Param(mutable=True, initialize=val))

    x = {i: getattr(m, f"x{i}") for i in list(range(1,25)) + list(range(26,32))}

    # >= constraints
    m.c1  = pyo.Constraint(expr=x[1] + x[7] - x[13] >= m.c1_rhs)
    m.c2  = pyo.Constraint(expr=x[2] + x[8] - x[14] >= m.c2_rhs)
    m.c3  = pyo.Constraint(expr=x[3] + x[9] - x[15] >= m.c3_rhs)
    m.c4  = pyo.Constraint(expr=x[4] + x[10] - x[16] >= m.c4_rhs)
    m.c5  = pyo.Constraint(expr=x[5] + x[11] - x[17] >= m.c5_rhs)
    m.c6  = pyo.Constraint(expr=x[6] + x[12] - x[18] >= m.c6_rhs)

    # == constraints (x25 = 100000 is a constant)
    m.c7  = pyo.Constraint(expr=12*x[19] - _X25 + x[26] == m.c7_rhs)
    m.c8  = pyo.Constraint(expr=12*x[20] - x[26] + x[27] == m.c8_rhs)
    m.c9  = pyo.Constraint(expr=12*x[21] - x[27] + x[28] == m.c9_rhs)
    m.c10 = pyo.Constraint(expr=12*x[22] - x[28] + x[29] == m.c10_rhs)
    m.c11 = pyo.Constraint(expr=12*x[23] - x[29] + x[30] == m.c11_rhs)
    m.c12 = pyo.Constraint(expr=12*x[24] - x[30] + x[31] == m.c12_rhs)

    # Quadratic == constraints (declared with @constraint in Julia, so quadratic)
    m.c13 = pyo.Constraint(expr=-8e-5*x[7]**2 + x[13] == 0)
    m.c14 = pyo.Constraint(expr=-8e-5*x[8]**2 + x[14] == 0)
    m.c15 = pyo.Constraint(expr=-8e-5*x[9]**2 + x[15] == 0)
    m.c16 = pyo.Constraint(expr=-8e-5*x[10]**2 + x[16] == 0)
    m.c17 = pyo.Constraint(expr=-8e-5*x[11]**2 + x[17] == 0)
    m.c18 = pyo.Constraint(expr=-8e-5*x[12]**2 + x[18] == 0)

    # Linear == constraints
    m.c19 = pyo.Constraint(expr=-4.97*x[7] + x[19] == 330)
    m.c20 = pyo.Constraint(expr=-4.97*x[8] + x[20] == 330)
    m.c21 = pyo.Constraint(expr=-4.97*x[9] + x[21] == 330)
    m.c22 = pyo.Constraint(expr=-4.97*x[10] + x[22] == 330)
    m.c23 = pyo.Constraint(expr=-4.97*x[11] + x[23] == 330)
    m.c24 = pyo.Constraint(expr=-4.97*x[12] + x[24] == 330)

    # Objective
    xv = [x[i] for i in range(1,7)]
    m.obj_expr = 82.8 * sum(0.0016*xv[i]**2 + 8*xv[i] for i in range(6)) + 248400
    return m


def all_vars(m):
    return [getattr(m, f"x{i}") for i in list(range(1,25)) + list(range(26,32))]


def build_models(nscen, nfirst=5, nparam=5, seed=1234, **kw):
    rng = JuliaMT19937(seed); ml = []; fl = []; mx = nparam
    for s in range(nscen):
        m = create_model(); av = all_vars(m); f = av[:nfirst]
        if s > 0:
            for idx in range(min(mx, len(_RHS_BASE))):
                p, bv, ct = _RHS_BASE[idx]
                if ct == "ge":
                    getattr(m, p).set_value(addnoise_ge(bv, rng))
                else:
                    # == constraint: use Julia-style addnoise
                    if bv == 0.0:
                        getattr(m, p).set_value(rng.rand_uniform(-10.0, 10.0))
                    else:
                        getattr(m, p).set_value(bv * rng.rand_uniform(0.5, 2.0))
        ml.append(m); fl.append(f)
    return ml, fl


MODE_PARAMS = {
    "smoke": {"nscen":5,"target_nodes":60,"gap_stop_tol":1e-6,"time_limit":120,"enable_ef_ub":True,"ef_time_ub":30.,"plot_every":None,
              "plot_output_dir":"results/hydro_smoke/plots","output_csv_path":"results/hydro_smoke/simplex_result.csv"},
    "full":  {"nscen":100,"target_nodes":300,"gap_stop_tol":1e-2,"time_limit":None,"enable_ef_ub":True,"ef_time_ub":43200.,"plot_every":None,
              "plot_output_dir":"results/hydro_full/plots","output_csv_path":"results/hydro_full/simplex_result.csv"},
}
BUNDLE_OPTIONS = {"NonConvex": 2, "MIPGap": 1e-1,"TimeLimit": 60}
Q_MAX = 1e10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke","full"), default="smoke")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--nfirst", type=int, default=5)
    args = ap.parse_args(); nf = args.nfirst; cfg = dict(MODE_PARAMS[args.mode])
    out = Path(cfg["output_csv_path"]); out.parent.mkdir(parents=True, exist_ok=True)
    if cfg["plot_output_dir"]: Path(cfg["plot_output_dir"]).mkdir(parents=True, exist_ok=True)
    print("="*60 + f"\nhydro -- nfirst={nf}, seed={args.seed}\n" + "="*60)
    t0 = perf_counter()
    ml, fl = build_models(cfg["nscen"], nf, nf, args.seed); S = len(ml)
    bb = [BaseBundle(ml[s], options=BUNDLE_OPTIONS, q_max=Q_MAX) for s in range(S)]
    mb = [MSBundle(ml[s], fl[s], options=BUNDLE_OPTIONS) for s in range(S)]
    labels = ["x1","x2","x3","x4","x5"][:nf]
    res = run_pid_simplex_3d(model_list=ml, first_vars_list=fl, base_bundles=bb, ms_bundles=mb,
        target_nodes=cfg["target_nodes"], min_dist=1e-3, gap_stop_tol=cfg["gap_stop_tol"], time_limit=cfg["time_limit"],
        enable_ef_ub=cfg["enable_ef_ub"], ef_time_ub=cfg["ef_time_ub"], plot_every=cfg["plot_every"],
        plot_output_dir=cfg["plot_output_dir"], output_csv_path=str(out), enable_3d_plot=False,
        axis_labels=tuple(labels))
    t1 = perf_counter(); LB = res.get("LB_hist",[]); UB = res.get("UB_hist",[])
    if LB and UB: print(f"\nLB_sum={float(LB[-1]):.12f} UB_sum={float(UB[-1]):.12f}\nLB/S={float(LB[-1])/S:.12f} UB/S={float(UB[-1])/S:.12f}")
    print(f"{'='*60}\nDone. {t1-t0:.2f}s CSV: {out}\n{'='*60}")


if __name__ == "__main__": main()
