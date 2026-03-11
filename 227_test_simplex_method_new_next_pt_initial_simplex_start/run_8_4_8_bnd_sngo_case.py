"""
run_8_4_8_bnd_sngo_case.py  -- Simplex runner for SNGO-master/Global/8_4_8_bnd

Same as 8_4_8 but with bounds on x23-x32 (>=0.1 instead of free).
42 variables (x1-x42). 20 NL constraints (10 product + 10 log).
Quadratic least-squares objective.

Stage split: nfirst=5, first=[x1,x2,x3,x4,x5].
Perturbed: first 4 product constraints (RHS=0), addnoise_julia.
"""
import argparse
from pathlib import Path
from time import perf_counter
import pyomo.environ as pyo
from bundles import BaseBundle, MSBundle
from simplex_specialstart import run_pid_simplex_3d
from run_st_fp7a_case import JuliaMT19937

def addnoise_julia(a: float, rng: JuliaMT19937) -> float:
    if a == 0.0:
        return a + rng.rand_uniform(-10.0, 10.0)
    return a * rng.rand_uniform(0.5, 2.0)

_RHS = [("c1_rhs",0.),("c2_rhs",0.),("c3_rhs",0.),("c4_rhs",0.),("c5_rhs",0.)]

def create_model():
    m = pyo.ConcreteModel()
    # x1-x20: same bounds as 8_4_8
    _bnds = {1:(0.285,0.315),2:(0.546,0.636),3:(0.999071638557945,1.00092836144205),4:(481.55,486.05),
             5:(0.385,0.415),6:(0.557,0.647),7:(0.999071638557945,1.00092836144205),8:(490.95,495.45),
             9:(0.485,0.515),10:(0.567,0.657),11:(0.999071638557945,1.00092836144205),12:(497.65,502.15),
             13:(0.685,0.715),14:(0.612,0.702),15:(0.999071638557945,1.00092836144205),16:(499.15,503.65),
             17:(0.885,0.915),18:(0.769,0.859),19:(0.999071638557945,1.00092836144205),20:(467.45,471.95),
             21:(1,2),22:(1,2)}
    _starts = {1:0.29015241396,2:0.62189400372,3:1.00009353307628,4:482.905120568,
               5:0.39376636351,6:0.57716475803,7:0.999721176860282,8:494.8032165615,
               9:0.48701341169,10:0.61201896021,11:1.00092486639703,12:500.254300201,
               13:0.71473399117,14:0.68060254203,15:0.999314298281912,16:502.0287344155,
               17:0.88978553592,18:0.79150724797,19:1.00031365361411,20:469.4091037145,
               21:1.9,22:1.6}
    for i in range(1,23):
        setattr(m, f"x{i}", pyo.Var(bounds=_bnds[i], initialize=_starts[i]))
    # x23-x32: bounded >= 0.1 (difference from 8_4_8 which has no bounds)
    for i in range(23,33):
        setattr(m, f"x{i}", pyo.Var(bounds=(0.1, None), initialize=1))
    # x33-x42: free variables
    for i in range(33,43):
        setattr(m, f"x{i}", pyo.Var(initialize=1))

    for name, val in _RHS:
        setattr(m, name, pyo.Param(mutable=True, initialize=val))

    # Objective
    m.obj_expr = (
        (200*m.x1 - 60)**2 + (66.6666666666667*m.x2 - 39.4)**2
        + (3231.5*m.x3 - 3231.5)**2 + (1.33333333333333*m.x4 - 645.066666666667)**2
        + (200*m.x5 - 80)**2 + (66.6666666666667*m.x6 - 40.1333333333333)**2
        + (3231.5*m.x7 - 3231.5)**2 + (1.33333333333333*m.x8 - 657.6)**2
        + (200*m.x9 - 100)**2 + (66.6666666666667*m.x10 - 40.8)**2
        + (3231.5*m.x11 - 3231.5)**2 + (1.33333333333333*m.x12 - 666.533333333333)**2
        + (200*m.x13 - 140)**2 + (66.6666666666667*m.x14 - 43.8)**2
        + (3231.5*m.x15 - 3231.5)**2 + (1.33333333333333*m.x16 - 668.533333333333)**2
        + (200*m.x17 - 180)**2 + (66.6666666666667*m.x18 - 54.2666666666667)**2
        + (3231.5*m.x19 - 3231.5)**2 + (1.33333333333333*m.x20 - 626.266666666667)**2
    )

    # Product constraints (10)
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

    # Log constraints (10)
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


def all_vars(m):
    return [getattr(m, f"x{i}") for i in range(1, 43)]


def build_models(nscen, nfirst=5, nparam=5, seed=1234, **kw):
    rng = JuliaMT19937(seed); ml = []; fl = []; mx = nparam
    for s in range(nscen):
        m = create_model(); av = all_vars(m); f = av[:nfirst]
        if s > 0:
            for idx in range(min(mx, len(_RHS))):
                p, bv = _RHS[idx]
                getattr(m, p).set_value(addnoise_julia(bv, rng))
        ml.append(m); fl.append(f)
    return ml, fl


MODE_PARAMS = {
    "smoke": {"nscen":10,"target_nodes":60,"gap_stop_tol":1e-6,"time_limit":300,"enable_ef_ub":True,"ef_time_ub":30.,"plot_every":None,
              "plot_output_dir":"results/8_4_8_bnd_smoke/plots","output_csv_path":"results/8_4_8_bnd_smoke/simplex_result.csv"},
    "full":  {"nscen":100,"target_nodes":300,"gap_stop_tol":1e-2,"time_limit":None,"enable_ef_ub":True,"ef_time_ub":43200.,"plot_every":None,
              "plot_output_dir":"results/8_4_8_bnd_full/plots","output_csv_path":"results/8_4_8_bnd_full/simplex_result.csv"},
}
BUNDLE_OPTIONS = {"NonConvex": 2, "MIPGap": 1e-1}
Q_MAX = 1e10


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=("smoke","full"), default="smoke")
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args(); cfg = dict(MODE_PARAMS[args.mode])
    out = Path(cfg["output_csv_path"]); out.parent.mkdir(parents=True, exist_ok=True)
    if cfg["plot_output_dir"]: Path(cfg["plot_output_dir"]).mkdir(parents=True, exist_ok=True)
    print("="*60 + f"\n8_4_8_bnd -- seed={args.seed}\n" + "="*60)
    t0 = perf_counter()
    ml, fl = build_models(cfg["nscen"], 5, 5, args.seed); S = len(ml)
    bb = [BaseBundle(ml[s], options=BUNDLE_OPTIONS, q_max=Q_MAX) for s in range(S)]
    mb = [MSBundle(ml[s], fl[s], options=BUNDLE_OPTIONS) for s in range(S)]
    res = run_pid_simplex_3d(model_list=ml, first_vars_list=fl, base_bundles=bb, ms_bundles=mb,
        target_nodes=cfg["target_nodes"], min_dist=1e-3, gap_stop_tol=cfg["gap_stop_tol"], time_limit=cfg["time_limit"],
        enable_ef_ub=cfg["enable_ef_ub"], ef_time_ub=cfg["ef_time_ub"], plot_every=cfg["plot_every"],
        plot_output_dir=cfg["plot_output_dir"], output_csv_path=str(out), enable_3d_plot=False,
        axis_labels=("x1","x2","x3","x4","x5"))
    t1 = perf_counter(); LB = res.get("LB_hist",[]); UB = res.get("UB_hist",[])
    if LB and UB: print(f"\nLB_sum={float(LB[-1]):.12f} UB_sum={float(UB[-1]):.12f}\nLB/S={float(LB[-1])/S:.12f} UB/S={float(UB[-1])/S:.12f}")
    print(f"{'='*60}\nDone. {t1-t0:.2f}s CSV: {out}\n{'='*60}")


if __name__ == "__main__": main()
