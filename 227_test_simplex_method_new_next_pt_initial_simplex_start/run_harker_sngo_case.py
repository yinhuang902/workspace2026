"""
run_harker_sngo_case.py  -- Simplex runner for SNGO-master/Global/harker

21 variables (x1-x20 >= 0, objvar free).
7 flow-balance == constraints + 1 NL epigraph constraint.
Objective: Min objvar (epigraph of concave function with cubic terms).

Stage split: nfirst=5, first=[x1,x2,x3,x4,x5].
Perturbed: first 4 == constraints (all RHS=0), addnoise_julia.
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

_RHS = [("c1_rhs",0.),("c2_rhs",0.),("c3_rhs",0.),("c4_rhs",0.),
        ("c5_rhs",0.),("c6_rhs",0.),("c7_rhs",0.)]

def create_model():
    m = pyo.ConcreteModel()
    for i in range(1, 15):
        setattr(m, f"x{i}", pyo.Var(bounds=(0, 200 if i <= 5 else None), initialize=0))
    for i in range(15, 21):
        setattr(m, f"x{i}", pyo.Var(bounds=(0, None), initialize=25))
    m.objvar = pyo.Var(initialize=0)

    for name, val in _RHS:
        setattr(m, name, pyo.Param(mutable=True, initialize=val))

    # Flow balance constraints
    m.c1 = pyo.Constraint(expr=m.x15 + m.x16 + m.x17 - m.x18 - m.x19 - m.x20 == m.c1_rhs)
    m.c2 = pyo.Constraint(expr=-m.x1 - m.x2 + m.x5 + m.x8 - m.x15 + m.x18 == m.c2_rhs)
    m.c3 = pyo.Constraint(expr=-m.x3 + m.x11 - m.x16 + m.x19 == m.c3_rhs)
    m.c4 = pyo.Constraint(expr=-m.x4 + m.x12 - m.x17 + m.x20 == m.c4_rhs)
    m.c5 = pyo.Constraint(expr=m.x1 - m.x5 - m.x6 - m.x7 + m.x9 + m.x13 == m.c5_rhs)
    m.c6 = pyo.Constraint(expr=m.x2 + m.x6 - m.x8 - m.x9 - m.x10 + m.x14 == m.c6_rhs)
    m.c7 = pyo.Constraint(expr=m.x3 + m.x4 + m.x7 + m.x10 - m.x11 - m.x12 - m.x13 - m.x14 == m.c7_rhs)

    # Objective via epigraph
    m.obj_expr = m.objvar

    # NL epigraph constraint:
    # objvar >= -(revenue - cost)
    # revenue = 19*x15 - 0.1*x15^2 + 27*x16 - 0.005*x16^2 + 30*x17 - 0.15*x17^2
    #         - 0.5*x18^2 - x18 - 0.4*x19^2 - 2*x19 - 0.3*x20^2 - 1.5*x20
    # cost = sum of cubic + linear terms
    m.c8 = pyo.Constraint(expr=m.objvar >= -(
        19*m.x15 - 0.1*m.x15**2 - 0.5*m.x18**2
        - m.x18 - 0.005*m.x16**2 + 27*m.x16 - 0.4*m.x19**2
        - 2*m.x19 - 0.15*m.x17**2 + 30*m.x17 - 0.3*m.x20**2
        - 1.5*m.x20
        - (0.166666666666667*m.x1**3 + m.x1
           + 0.0666666666666667*m.x2**3 + 2*m.x2
           + 0.1*m.x3**3 + 3*m.x3
           + 0.133333333333333*m.x4**3 + m.x4
           + 0.1*m.x5**3 + 2*m.x5
           + 0.0333333333333333*m.x6**3 + m.x6
           + 0.0333333333333333*m.x7**3 + m.x7
           + 0.166666666666667*m.x8**3 + 3*m.x8
           + 0.0666666666666667*m.x9**3 + 2*m.x9
           + 0.333333333333333*m.x10**3 + m.x10
           + 0.0833333333333333*m.x11**3 + 2*m.x11
           + 0.0666666666666667*m.x12**3 + 2*m.x12
           + 0.3*m.x13**3 + m.x13
           + 0.266666666666667*m.x14**3 + 3*m.x14)
    ))
    return m


def all_vars(m):
    return [getattr(m, f"x{i}") for i in range(1, 21)] + [m.objvar]


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
              "plot_output_dir":"results/harker_smoke/plots","output_csv_path":"results/harker_smoke/simplex_result.csv"},
    "full":  {"nscen":100,"target_nodes":300,"gap_stop_tol":1e-2,"time_limit":None,"enable_ef_ub":True,"ef_time_ub":43200.,"plot_every":None,
              "plot_output_dir":"results/harker_full/plots","output_csv_path":"results/harker_full/simplex_result.csv"},
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
    print("="*60 + f"\nharker -- nfirst={nf}, seed={args.seed}\n" + "="*60)
    t0 = perf_counter()
    ml, fl = build_models(cfg["nscen"], nf, nf, args.seed); S = len(ml)
    bb = [BaseBundle(ml[s], options=BUNDLE_OPTIONS, q_max=Q_MAX) for s in range(S)]
    mb = [MSBundle(ml[s], fl[s], options=BUNDLE_OPTIONS) for s in range(S)]
    res = run_pid_simplex_3d(model_list=ml, first_vars_list=fl, base_bundles=bb, ms_bundles=mb,
        target_nodes=cfg["target_nodes"], min_dist=1e-3, gap_stop_tol=cfg["gap_stop_tol"], time_limit=cfg["time_limit"],
        enable_ef_ub=cfg["enable_ef_ub"], ef_time_ub=cfg["ef_time_ub"], plot_every=cfg["plot_every"],
        plot_output_dir=cfg["plot_output_dir"], output_csv_path=str(out), enable_3d_plot=False,
        axis_labels=tuple(f"x{i+1}" for i in range(nf)))
    t1 = perf_counter(); LB = res.get("LB_hist",[]); UB = res.get("UB_hist",[])
    if LB and UB: print(f"\nLB_sum={float(LB[-1]):.12f} UB_sum={float(UB[-1]):.12f}\nLB/S={float(LB[-1])/S:.12f} UB/S={float(UB[-1])/S:.12f}")
    print(f"{'='*60}\nDone. {t1-t0:.2f}s CSV: {out}\n{'='*60}")


if __name__ == "__main__": main()
