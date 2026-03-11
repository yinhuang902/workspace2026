"""
run_chenery_sngo_case.py  -- Simplex runner for SNGO-master/Global/chenery

43 JuMP variables (x1-x39, x41-x44; no x40). Complex NL model.
Linear objective: Min -x9 - x10 - x11 - x12.
Linear and NL constraints including bilinear products, power laws, and log.

Stage split: nfirst=5, first=[x1,x2,x3,x4,x5].
Perturbed: first 4 >= constraints (all RHS=0), addnoise_ge.
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

_RHS = [("c1_rhs",0.),("c2_rhs",0.),("c3_rhs",0.),("c4_rhs",0.)]

def create_model():
    m = pyo.ConcreteModel()
    # Variables with bounds and starts from Julia
    _specs = {
        1:(0,2000,200), 2:(0,2000,200), 3:(0,812,200), 4:(0,1169,200),
        5:(0,100,1.08002386572984), 6:(0,100,1.25850763714561),
        7:(0,100,2.47224270643972), 8:(0,100,2.08174548233022),
        9:(0,2000,250), 10:(0,2000,250), 11:(0,2000,250), 12:(0,2000,250),
        13:(0.1,100,3), 14:(0.1,100,3), 15:(0.1,100,3), 16:(0.1,100,3),
        17:(0,1,0.283078383128534), 18:(0,1,0.383990781960791),
        19:(0,1,0.309951359679435), 20:(0,1,0.580992426342466),
        21:(0,1,0.22769870931466), 22:(0,1,0.249861958624235),
        23:(0,1,0.617797527645794), 24:(0,1,0.428786587425074),
        25:(0,400,None), 26:(0,400,None), 27:(0,400,None), 28:(0,400,None),
        29:(0,400,None), 30:(0,400,None),
        31:(0,4,1), 32:(0,4,1), 33:(0,4,1), 34:(0,4,1),
        35:(0,4,1.1), 36:(0,4,1),
        37:(0.25,4,3.5), 38:(0.25,4,3.5),
        39:(0.01,None,0.3),
    }
    for i, (lb, ub, st) in _specs.items():
        setattr(m, f"x{i}", pyo.Var(bounds=(lb, ub), initialize=st))
    # x41-x44 (no x40!)
    m.x41 = pyo.Var(bounds=(0.001, None), initialize=0.171804999139287)
    m.x42 = pyo.Var(bounds=(0.001, None), initialize=0.349221638418406)
    m.x43 = pyo.Var(bounds=(0.001, None), initialize=15.7837604335036)
    m.x44 = pyo.Var(bounds=(0.001, None), initialize=0.00311417990544524)

    for name, val in _RHS:
        setattr(m, name, pyo.Param(mutable=True, initialize=val))

    # Objective
    m.obj_expr = -m.x9 - m.x10 - m.x11 - m.x12

    # Linear >= constraints (first 4 are perturbed)
    m.c1  = pyo.Constraint(expr=m.x1 - m.x9 - m.x25 + m.x28 >= m.c1_rhs)
    m.c2  = pyo.Constraint(expr=-0.1*m.x1 + m.x2 - m.x10 - m.x26 + m.x29 >= m.c2_rhs)
    m.c3  = pyo.Constraint(expr=-0.2*m.x1 - 0.1*m.x2 + m.x3 - m.x11 - m.x27 + m.x30 >= m.c3_rhs)
    m.c4  = pyo.Constraint(expr=-0.2*m.x1 - 0.3*m.x2 - 0.1*m.x3 + m.x4 - m.x12 >= m.c4_rhs)

    # Bilinear <= constraint
    m.c5  = pyo.Constraint(expr=m.x31*m.x28 - m.x34*m.x25 + m.x32*m.x29 - m.x35*m.x26 + m.x33*m.x30 - m.x36*m.x27 <= 0)

    # Linear == constraints
    m.c6  = pyo.Constraint(expr=-0.005*m.x28 + m.x31 == 1)
    m.c7  = pyo.Constraint(expr=-0.0157*m.x29 + m.x32 == 1)
    m.c8  = pyo.Constraint(expr=-0.00178*m.x30 + m.x33 == 1)
    m.c9  = pyo.Constraint(expr=0.005*m.x25 + m.x34 == 1)
    m.c10 = pyo.Constraint(expr=0.001*m.x26 + m.x35 == 1.1)
    m.c11 = pyo.Constraint(expr=0.01*m.x27 + m.x36 == 1)

    # NL constraints: power law production functions
    m.c12 = pyo.Constraint(expr=-100*(m.x39*m.x13)**(-0.674) + m.x9 == 0)
    m.c13 = pyo.Constraint(expr=-230*(m.x39*m.x14)**(-0.246) + m.x10 == 0)
    m.c14 = pyo.Constraint(expr=-220*(m.x39*m.x15)**(-0.587) + m.x11 == 0)
    m.c15 = pyo.Constraint(expr=-450*(m.x39*m.x16)**(-0.352) + m.x12 == 0)

    # Bilinear constraints
    m.c16 = pyo.Constraint(expr=m.x17*m.x1 + m.x18*m.x2 + m.x19*m.x3 + m.x20*m.x4 <= 750)
    m.c17 = pyo.Constraint(expr=m.x21*m.x1 + m.x22*m.x2 + m.x23*m.x3 + m.x24*m.x4 == 500)

    # Input-output constraints
    m.c18 = pyo.Constraint(expr=-m.x5 + m.x13 - 0.1*m.x14 - 0.2*m.x15 - 0.2*m.x16 == 0)
    m.c19 = pyo.Constraint(expr=-m.x6 + m.x14 - 0.1*m.x15 - 0.3*m.x16 == 0)
    m.c20 = pyo.Constraint(expr=-m.x7 + m.x15 - 0.1*m.x16 == 0)
    m.c21 = pyo.Constraint(expr=-m.x8 + m.x16 == 0)

    m.c22 = pyo.Constraint(expr=-m.x37 + m.x38 == 0)

    # NL constraints: activity coefficients
    m.c23 = pyo.Constraint(expr=-(2.06748466257669*m.x38)**(-0.89) + m.x41 == 0)
    m.c24 = pyo.Constraint(expr=-(1.25733634311512*m.x38)**(-0.71) + m.x42 == 0)
    m.c25 = pyo.Constraint(expr=-(0.00908173562058528*m.x38)**(-0.8) + m.x43 == 0)
    m.c26 = pyo.Constraint(expr=-(124.31328320802*m.x38)**(-0.95) + m.x44 == 0)

    # NL constraints: cost functions
    m.c27 = pyo.Constraint(expr=-(0.674 + 0.326/m.x41)**0.123595505617978 + 3.97*m.x17 == 0)
    m.c28 = pyo.Constraint(expr=-(0.557 + 0.443/m.x42)**0.408450704225352 + 3.33*m.x18 == 0)
    m.c29 = pyo.Constraint(expr=-(0.00900000000000001 + 0.991/m.x43)**0.25 + 1.67*m.x19 == 0)
    m.c30 = pyo.Constraint(expr=-(0.99202 + 0.00798/m.x44)**0.0526315789473684 + 1.84*m.x20 == 0)

    m.c31 = pyo.Constraint(expr=-(0.326 + 0.674*m.x41)**0.123595505617978 + 3.97*m.x21 == 0)
    m.c32 = pyo.Constraint(expr=-(0.443 + 0.557*m.x42)**0.408450704225352 + 3.33*m.x22 == 0)
    m.c33 = pyo.Constraint(expr=-(0.991 + 0.00900000000000001*m.x43)**0.25 + 1.67*m.x23 == 0)
    m.c34 = pyo.Constraint(expr=-(0.00798 + 0.99202*m.x44)**0.0526315789473684 + 1.84*m.x24 == 0)

    # Linking constraints
    m.c35 = pyo.Constraint(expr=-m.x37*m.x21 + m.x5 - m.x17 == 0)
    m.c36 = pyo.Constraint(expr=-m.x37*m.x22 + m.x6 - m.x18 == 0)
    m.c37 = pyo.Constraint(expr=-m.x37*m.x23 + m.x7 - m.x19 == 0)
    m.c38 = pyo.Constraint(expr=-m.x37*m.x24 + m.x8 - m.x20 == 0)

    return m


def all_vars(m):
    return [getattr(m, f"x{i}") for i in list(range(1,40)) + [41,42,43,44]]


def build_models(nscen, nfirst=5, nparam=5, seed=1234, **kw):
    rng = JuliaMT19937(seed); ml = []; fl = []; mx = min(nparam, len(_RHS))
    for s in range(nscen):
        m = create_model(); av = all_vars(m); f = av[:nfirst]
        if s > 0:
            for idx in range(mx):
                p, bv = _RHS[idx]
                getattr(m, p).set_value(addnoise_ge(bv, rng))
        ml.append(m); fl.append(f)
    return ml, fl


MODE_PARAMS = {
    "smoke": {"nscen":10,"target_nodes":60,"gap_stop_tol":1e-6,"time_limit":300,"enable_ef_ub":True,"ef_time_ub":30.,"plot_every":None,
              "plot_output_dir":"results/chenery_smoke/plots","output_csv_path":"results/chenery_smoke/simplex_result.csv"},
    "full":  {"nscen":100,"target_nodes":300,"gap_stop_tol":1e-2,"time_limit":None,"enable_ef_ub":True,"ef_time_ub":43200.,"plot_every":None,
              "plot_output_dir":"results/chenery_full/plots","output_csv_path":"results/chenery_full/simplex_result.csv"},
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
    print("="*60 + f"\nchenery -- seed={args.seed}\n" + "="*60)
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