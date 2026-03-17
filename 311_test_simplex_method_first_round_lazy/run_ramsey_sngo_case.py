"""
run_ramsey_sngo_case.py  -- Simplex runner for SNGO-master/Global/ramsey

32 JuMP variables (x2-x11, x12-x22, x24-x33, objvar; x1=3, x23=0.05 fixed).
11 NL constraints (x^0.25), 11 linear ==, 1 <=, 1 NL log objective via objvar.

Stage split: nfirst=5, first=[x2,x3,x4,x5,x6].
Perturbed: first 4 NL constraints (all RHS=0), addnoise_julia.
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

_X1 = 3.0
_X23 = 0.05

_RHS = [("c1_rhs",0.),("c2_rhs",0.),("c3_rhs",0.),("c4_rhs",0.),
        ("c5_rhs",0.),("c6_rhs",0.),("c7_rhs",0.),("c8_rhs",0.),
        ("c9_rhs",0.),("c10_rhs",0.),("c11_rhs",0.)]

def create_model():
    m = pyo.ConcreteModel()
    # x2-x11: capital stock, >= 3
    for i in range(2, 12):
        setattr(m, f"x{i}", pyo.Var(bounds=(3, 100 if i <= 6 else None), initialize=3))
    # x12-x22: consumption, >= 0.95
    for i in range(12, 23):
        setattr(m, f"x{i}", pyo.Var(bounds=(0.95, None), initialize=0.95))
    # x24-x33: investment, bounded
    _inv_ubs = {24:0.0575, 25:0.066125, 26:0.07604375, 27:0.0874503125,
                28:0.100567859375, 29:0.11565303828125, 30:0.133000994023437,
                31:0.152951143126953, 32:0.175893814595996, 33:0.202277886785395}
    for i in range(24, 34):
        setattr(m, f"x{i}", pyo.Var(bounds=(0.05, _inv_ubs[i]), initialize=0.05))
    m.objvar = pyo.Var(initialize=0)

    for name, val in _RHS:
        setattr(m, name, pyo.Param(mutable=True, initialize=val))

    m.obj_expr = m.objvar

    # x1=3 and x23=0.05 are constants. JuMP variable order: x2,...,x11,x12,...,x22,x24,...,x33,objvar
    # 11 NL constraints: coeff * x_k^0.25 - x_{k+10} - x_{k+22} == 0
    coeffs = [0.759835685651593, 0.77686866556676, 0.794283468039448,
              0.812088652256959, 0.830292969275008, 0.848905366318769,
              0.867934991180342, 0.88739119671479, 0.907283545436972,
              0.92762181422141, 0.948415999107521]

    # c1: 0.7598*x1^0.25 - x12 - x23 == 0  (x1=3, x23=0.05 constants)
    m.c1 = pyo.Constraint(expr=coeffs[0]*_X1**0.25 - m.x12 - _X23 == m.c1_rhs)
    # c2-c11: similar pattern with JuMP vars
    m.c2  = pyo.Constraint(expr=coeffs[1]*m.x2**0.25 - m.x13 - m.x24 == m.c2_rhs)
    m.c3  = pyo.Constraint(expr=coeffs[2]*m.x3**0.25 - m.x14 - m.x25 == m.c3_rhs)
    m.c4  = pyo.Constraint(expr=coeffs[3]*m.x4**0.25 - m.x15 - m.x26 == m.c4_rhs)
    m.c5  = pyo.Constraint(expr=coeffs[4]*m.x5**0.25 - m.x16 - m.x27 == m.c5_rhs)
    m.c6  = pyo.Constraint(expr=coeffs[5]*m.x6**0.25 - m.x17 - m.x28 == m.c6_rhs)
    m.c7  = pyo.Constraint(expr=coeffs[6]*m.x7**0.25 - m.x18 - m.x29 == m.c7_rhs)
    m.c8  = pyo.Constraint(expr=coeffs[7]*m.x8**0.25 - m.x19 - m.x30 == m.c8_rhs)
    m.c9  = pyo.Constraint(expr=coeffs[8]*m.x9**0.25 - m.x20 - m.x31 == m.c9_rhs)
    m.c10 = pyo.Constraint(expr=coeffs[9]*m.x10**0.25 - m.x21 - m.x32 == m.c10_rhs)
    m.c11 = pyo.Constraint(expr=coeffs[10]*m.x11**0.25 - m.x22 - m.x33 == m.c11_rhs)

    # Linear capital accumulation constraints (x1=3 → -x1 + x2 - x23 == 0 → -3 + x2 - 0.05 == 0)
    m.c12 = pyo.Constraint(expr=-_X1 + m.x2 - _X23 == 0)
    m.c13 = pyo.Constraint(expr=-m.x2 + m.x3 - m.x24 == 0)
    m.c14 = pyo.Constraint(expr=-m.x3 + m.x4 - m.x25 == 0)
    m.c15 = pyo.Constraint(expr=-m.x4 + m.x5 - m.x26 == 0)
    m.c16 = pyo.Constraint(expr=-m.x5 + m.x6 - m.x27 == 0)
    m.c17 = pyo.Constraint(expr=-m.x6 + m.x7 - m.x28 == 0)
    m.c18 = pyo.Constraint(expr=-m.x7 + m.x8 - m.x29 == 0)
    m.c19 = pyo.Constraint(expr=-m.x8 + m.x9 - m.x30 == 0)
    m.c20 = pyo.Constraint(expr=-m.x9 + m.x10 - m.x31 == 0)
    m.c21 = pyo.Constraint(expr=-m.x10 + m.x11 - m.x32 == 0)

    # Terminal constraint
    m.c22 = pyo.Constraint(expr=0.03*m.x11 - m.x33 <= 0)

    # NL objective constraint (log utility)
    m.c23 = pyo.Constraint(expr=(
        0.95*pyo.log(m.x12) + 0.9025*pyo.log(m.x13) + 0.857375*pyo.log(m.x14)
        + 0.81450625*pyo.log(m.x15) + 0.7737809375*pyo.log(m.x16)
        + 0.735091890625*pyo.log(m.x17) + 0.69833729609375*pyo.log(m.x18)
        + 0.663420431289062*pyo.log(m.x19) + 0.630249409724609*pyo.log(m.x20)
        + 0.598736939238379*pyo.log(m.x21) + 11.3760018455292*pyo.log(m.x22)
        + m.objvar == 0
    ))

    return m


def all_vars(m):
    return ([getattr(m, f"x{i}") for i in range(2, 12)]
            + [getattr(m, f"x{i}") for i in range(12, 23)]
            + [getattr(m, f"x{i}") for i in range(24, 34)]
            + [m.objvar])


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
              "plot_output_dir":"results/ramsey_smoke/plots","output_csv_path":"results/ramsey_smoke/simplex_result.csv"},
    "full":  {"nscen":100,"target_nodes":300,"gap_stop_tol":1e-2,"time_limit":None,"enable_ef_ub":True,"ef_time_ub":43200.,"plot_every":None,
              "plot_output_dir":"results/ramsey_full/plots","output_csv_path":"results/ramsey_full/simplex_result.csv"},
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
    print("="*60 + f"\nramsey -- nfirst={nf}, seed={args.seed}\n" + "="*60)
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
