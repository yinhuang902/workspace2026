"""
run_srcpm_sngo_case.py  -- Simplex runner for SNGO-master/Global/srcpm

38 JuMP variables (x1, x3-x39, objvar; x2=0 fixed).
21 linear constraints + 1 NL objective constraint via objvar.
NL terms: x^(-4), x^(-2.333).

Stage split: nfirst=5, first=[x1,x3,x4,x5,x6].
Perturbed: first 4 >= constraints (RHS=0,0,0,-0.7), addnoise_ge.
"""
import argparse
from pathlib import Path
from time import perf_counter
import pyomo.environ as pyo
from bundles import BaseBundle, MSBundle
from simplex_specialstart import run_pid_simplex_3d
from run_st_fp7a_case import JuliaMT19937

def addnoise_ge(a: float, rng: JuliaMT19937) -> float:
    if a == 0.0:
        return a + rng.rand_uniform(-10.0, 0.0)
    return a + abs(a) * rng.rand_uniform(-2.0, 0.0)

_X2 = 0.0  # fixed constant

_RHS = [("c1_rhs",0.),("c2_rhs",0.),("c3_rhs",0.),("c4_rhs",-0.7),("c5_rhs",0.)]

def create_model():
    m = pyo.ConcreteModel()
    m.x1  = pyo.Var(bounds=(0, 3.4), initialize=3.1)
    # x2 = 0 is fixed
    m.x3  = pyo.Var(bounds=(0, None), initialize=13.6)
    m.x4  = pyo.Var(bounds=(0, None), initialize=0)
    m.x5  = pyo.Var(bounds=(0, None), initialize=1.1)
    m.x6  = pyo.Var(bounds=(0, None), initialize=0)
    m.x7  = pyo.Var(bounds=(0, None), initialize=1)
    m.x8  = pyo.Var(bounds=(0, None), initialize=0)
    m.x9  = pyo.Var(bounds=(0, None), initialize=16.4244058299284)
    m.x10 = pyo.Var(bounds=(0, None), initialize=0)
    m.x11 = pyo.Var(bounds=(0, None), initialize=8.9)
    m.x12 = pyo.Var(bounds=(0, None), initialize=0)
    m.x13 = pyo.Var(bounds=(0, None), initialize=4.4)
    m.x14 = pyo.Var(bounds=(0, None), initialize=0)
    m.x15 = pyo.Var(bounds=(0, 7.1), initialize=7.1)
    m.x16 = pyo.Var(bounds=(0, 0.8), initialize=0.8)
    m.x17 = pyo.Var(bounds=(0, None), initialize=5.56103683518173)
    m.x18 = pyo.Var(bounds=(0, None), initialize=0.312071787775987)
    m.x19 = pyo.Var(bounds=(0, None), initialize=1.73896316481828)
    m.x20 = pyo.Var(bounds=(0, 2.5), initialize=2.5)
    m.x21 = pyo.Var(bounds=(0, 2.7), initialize=2.7)
    m.x22 = pyo.Var(bounds=(0, 0.3), initialize=0)
    m.x23 = pyo.Var(bounds=(0, 13.6), initialize=13.6)
    m.x24 = pyo.Var(bounds=(0, 1.1), initialize=1.1)
    m.x25 = pyo.Var(bounds=(0, 1), initialize=1)
    m.x26 = pyo.Var(bounds=(0, 16.2), initialize=15.7244058299284)
    m.x27 = pyo.Var(bounds=(0, 8.9), initialize=8.9)
    m.x28 = pyo.Var(bounds=(0, 4.4), initialize=4.4)
    m.x29 = pyo.Var(bounds=(0, 3.1), initialize=3.1)
    m.x30 = pyo.Var(bounds=(0, 1.7), initialize=0.928008053710258)
    m.x31 = pyo.Var(bounds=(0, 1.9), initialize=0.268195340806014)
    m.x32 = pyo.Var(bounds=(0, None), initialize=2.78989137704229)
    m.x33 = pyo.Var(bounds=(0, None), initialize=6.47831105055452)
    m.x34 = pyo.Var(bounds=(2, None), initialize=12.8)
    m.x35 = pyo.Var(bounds=(2, None), initialize=13.8)
    m.x36 = pyo.Var(bounds=(2, None), initialize=8.3)
    m.x37 = pyo.Var(bounds=(2, None), initialize=4.2)
    m.x38 = pyo.Var(bounds=(2, None), initialize=8.6)
    m.x39 = pyo.Var(initialize=1560.6691675193)
    m.objvar = pyo.Var(initialize=0)

    for name, val in _RHS:
        setattr(m, name, pyo.Param(mutable=True, initialize=val))

    m.obj_expr = m.objvar

    # >= constraints (first 4 are perturbed)
    m.c1  = pyo.Constraint(expr=-m.x3 - m.x4 + m.x23 >= m.c1_rhs)
    m.c2  = pyo.Constraint(expr=-m.x5 - m.x6 + m.x24 >= m.c2_rhs)
    m.c3  = pyo.Constraint(expr=-m.x7 - m.x8 + m.x25 >= m.c3_rhs)
    m.c4  = pyo.Constraint(expr=-m.x9 - m.x10 + m.x26 >= m.c4_rhs)

    # Remaining >= constraints (not perturbed)
    m.c5  = pyo.Constraint(expr=-m.x11 - m.x12 + m.x27 >= m.c5_rhs)
    m.c6  = pyo.Constraint(expr=-m.x13 - m.x14 + m.x28 >= 0)
    m.c7  = pyo.Constraint(expr=-m.x1 - _X2 + m.x29 >= 0)

    # Blending >= constraints
    m.c8  = pyo.Constraint(expr=0.35*m.x3 + 0.34*m.x4 + 0.5*m.x5 + 0.49*m.x6
        + 0.68*m.x7 + 0.67*m.x8 - m.x17 - m.x18 + 0.99*m.x21 + 0.99*m.x22 - m.x32 >= 0)
    m.c9  = pyo.Constraint(expr=0.38*m.x9 + 0.38*m.x10 + 0.48*m.x11 + 0.47*m.x12
        + 0.66*m.x13 + 0.65*m.x14 - m.x19 - m.x20 - m.x21 - m.x22 - m.x33 >= 0)
    m.c10 = pyo.Constraint(expr=0.2*m.x1 + 0.2*_X2 + 0.96*m.x15 + 0.96*m.x16
        + 0.67*m.x17 + 0.36*m.x18 + 0.61*m.x19 + 0.25*m.x20 - m.x30 - m.x34 >= 0)
    m.c11 = pyo.Constraint(expr=0.28*m.x3 + 0.28*m.x4 + 0.25*m.x5 + 0.25*m.x6
        + 0.2*m.x7 + 0.2*m.x8 + 0.26*m.x9 + 0.26*m.x10 + 0.23*m.x11
        + 0.23*m.x12 + 0.18*m.x13 + 0.18*m.x14 + 0.07*m.x17 + 0.18*m.x18
        + 0.02*m.x19 + 0.1*m.x20 + m.x30 + 0.93*m.x31 - m.x35 >= -0.5)
    m.c12 = pyo.Constraint(expr=0.8*m.x1 + 0.8*_X2 + 0.35*m.x3 + 0.35*m.x4
        + 0.23*m.x5 + 0.23*m.x6 + 0.1*m.x7 + 0.1*m.x8 + 0.33*m.x9
        + 0.33*m.x10 + 0.27*m.x11 + 0.27*m.x12 + 0.14*m.x13 + 0.14*m.x14
        - m.x15 - m.x16 + 0.04*m.x17 + 0.03*m.x18 + 0.06*m.x19 + 0.04*m.x20
        - m.x31 - m.x36 >= 0)
    m.c13 = pyo.Constraint(expr=0.23*m.x17 + 0.42*m.x18 + m.x32 - m.x37 >= 0)
    m.c14 = pyo.Constraint(expr=0.3*m.x19 + 0.6*m.x20 + m.x33 - m.x38 >= -0.1)

    # <= constraints
    m.c15 = pyo.Constraint(expr=m.x3 + m.x5 + m.x7 + m.x9 + m.x11 + m.x13 <= 50.5)
    m.c16 = pyo.Constraint(expr=m.x4 + m.x6 + m.x8 + m.x10 + m.x12 + m.x14 <= 7.5)
    m.c17 = pyo.Constraint(expr=m.x17 + m.x19 <= 7.3)
    m.c18 = pyo.Constraint(expr=m.x18 + m.x20 <= 2.9)
    m.c19 = pyo.Constraint(expr=-0.83*m.x17 + m.x19 <= 3.9)

    # Cost == constraint
    m.c20 = pyo.Constraint(expr=-0.45*m.x3 - 0.5*m.x4 - 0.45*m.x5 - 0.5*m.x6
        - 0.45*m.x7 - 0.5*m.x8 - 0.5*m.x9 - 0.55*m.x10 - 0.5*m.x11
        - 0.55*m.x12 - 0.5*m.x13 - 0.55*m.x14 - 0.41*m.x15 - 0.5*m.x16
        - 0.27*m.x17 - 0.45*m.x18 - 0.32*m.x19 - 0.28*m.x20 - 0.9*m.x21
        - m.x22 - 32*m.x23 - 32*m.x24 - 32*m.x25 - 32*m.x26 - 32*m.x27
        - 32*m.x28 - 32*m.x29 + 0.3*m.x30 + m.x39 == 0)

    # NL objective constraint
    m.c21 = pyo.Constraint(expr=(
        -3865470.56640001*m.x34**(-4) - 5130022.82472*m.x35**(-4)
        - 423446.8691225*m.x36**(-4) - 1808.40439881057*m.x37**(-2.33333333333333)
        - 17313.2956782741*m.x38**(-2.33333333333333) - m.x39 + m.objvar == 0
    ))

    return m


def all_vars(m):
    return [m.x1] + [getattr(m, f"x{i}") for i in range(3, 40)] + [m.objvar]


def build_models(nscen, nfirst=5, nparam=5, seed=1234, **kw):
    rng = JuliaMT19937(seed); ml = []; fl = []; mx = nparam
    for s in range(nscen):
        m = create_model(); av = all_vars(m); f = av[:nfirst]
        if s > 0:
            for idx in range(min(mx, len(_RHS))):
                p, bv = _RHS[idx]
                getattr(m, p).set_value(addnoise_ge(bv, rng))
        ml.append(m); fl.append(f)
    return ml, fl


MODE_PARAMS = {
    "smoke": {"nscen":10,"target_nodes":60,"gap_stop_tol":1e-6,"time_limit":300,"enable_ef_ub":True,"ef_time_ub":30.,"plot_every":None,
              "plot_output_dir":"results/srcpm_smoke/plots","output_csv_path":"results/srcpm_smoke/simplex_result.csv"},
    "full":  {"nscen":100,"target_nodes":300,"gap_stop_tol":1e-2,"time_limit":None,"enable_ef_ub":True,"ef_time_ub":43200.,"plot_every":None,
              "plot_output_dir":"results/srcpm_full/plots","output_csv_path":"results/srcpm_full/simplex_result.csv"},
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
    print("="*60 + f"\nsrcpm -- nfirst={nf}, seed={args.seed}\n" + "="*60)
    t0 = perf_counter()
    ml, fl = build_models(cfg["nscen"], nf, nf, args.seed); S = len(ml)
    bb = [BaseBundle(ml[s], options=BUNDLE_OPTIONS, q_max=Q_MAX) for s in range(S)]
    mb = [MSBundle(ml[s], fl[s], options=BUNDLE_OPTIONS) for s in range(S)]
    res = run_pid_simplex_3d(model_list=ml, first_vars_list=fl, base_bundles=bb, ms_bundles=mb,
        target_nodes=cfg["target_nodes"], min_dist=1e-3, gap_stop_tol=cfg["gap_stop_tol"], time_limit=cfg["time_limit"],
        enable_ef_ub=cfg["enable_ef_ub"], ef_time_ub=cfg["ef_time_ub"], plot_every=cfg["plot_every"],
        plot_output_dir=cfg["plot_output_dir"], output_csv_path=str(out), enable_3d_plot=False,
        axis_labels=tuple(f"x{i}" for i in [1,3,4,5,6][:nf]))
    t1 = perf_counter(); LB = res.get("LB_hist",[]); UB = res.get("UB_hist",[])
    if LB and UB: print(f"\nLB_sum={float(LB[-1]):.12f} UB_sum={float(UB[-1]):.12f}\nLB/S={float(LB[-1])/S:.12f} UB/S={float(UB[-1])/S:.12f}")
    print(f"{'='*60}\nDone. {t1-t0:.2f}s CSV: {out}\n{'='*60}")


if __name__ == "__main__": main()
