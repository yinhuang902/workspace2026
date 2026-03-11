"""
run_immun_sngo_case.py  -- Simplex runner for SNGO-master/Global/immun

21 variables (x1-x21, all >= 0; x1 has UB=187217.3).
7 equality constraints (==). Quadratic objective (sum of squared differences).

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

_RHS_BASE = [
    ("c1_rhs", 0.), ("c2_rhs", 0.), ("c3_rhs", 0.), ("c4_rhs", 0.),
    ("c5_rhs", 0.), ("c6_rhs", 0.), ("c7_rhs", 0.),
]

def create_model():
    m = pyo.ConcreteModel()
    m.x1  = pyo.Var(bounds=(0, 187217.324724184), initialize=187217.324724184)
    m.x2  = pyo.Var(bounds=(0, 5000), initialize=956.904106888036)
    m.x3  = pyo.Var(bounds=(0, 5000), initialize=0)
    m.x4  = pyo.Var(bounds=(0, 5000), initialize=45.5987315339227)
    m.x5  = pyo.Var(bounds=(0, 5000), initialize=0)
    for i in range(6, 22):
        init = {6:0, 7:0, 8:0, 9:40.6641597628654, 10:0,
                11:66834.2651808549, 12:33347.8176607291, 13:0,
                14:18186.5732712855, 15:27099.21721716, 16:0,
                17:938765.155199853, 18:42000, 19:42000, 20:42000, 21:0}.get(i, 0)
        setattr(m, f"x{i}", pyo.Var(bounds=(0, None), initialize=init))

    for name, val in _RHS_BASE:
        setattr(m, name, pyo.Param(mutable=True, initialize=val))

    x = [getattr(m, f"x{i}") for i in range(1, 22)]  # 0-indexed

    m.c1 = pyo.Constraint(expr=-x[9] - x[15] == m.c1_rhs)  # -x10 - x16 == 0
    m.c2 = pyo.Constraint(expr=1044.80727456326*x[1] + 1079.40354193291*x[2]
        + 74.5442033113223*x[3] + 36.3324688408125*x[4] + 41.3438438533384*x[5]
        + 43.2231094830356*x[6] + 43.8495313596014*x[7] + 59.5100782737447*x[8]
        + 1.00940093153723*x[9] - x[10] - x[16] == m.c2_rhs)
    m.c3 = pyo.Constraint(expr=75.57763951196*x[3] + 36.8361604344007*x[4]
        + 41.9170101494904*x[5] + 43.8223287926491*x[6] + 44.4574350070353*x[7]
        + 60.3350903666908*x[8] + 1.0391091639109*x[10] - x[11] - x[17] == m.c3_rhs)
    m.c4 = pyo.Constraint(expr=75.456505608033*x[3] + 36.7771203803858*x[4]
        + 41.8498266397494*x[5] + 43.7520914870108*x[6] + 44.3861797694312*x[7]
        + 60.2383868299423*x[8] + 1.02284761238063*x[11] - x[12] - x[18] == m.c4_rhs)
    m.c5 = pyo.Constraint(expr=1167.30216560492*x[3] + 74.4548991299823*x[4]
        + 84.7245403892903*x[5] + 88.5756558615307*x[6] + 89.8593610189442*x[7]
        + 121.951989954281*x[8] + 1.05*x[12] - x[13] - x[19] == m.c5_rhs)
    m.c6 = pyo.Constraint(expr=1115.8195763046*x[4] + 1126.3428356729*x[5]
        + 134.503508270593*x[6] + 136.452834477414*x[7] + 185.185989647919*x[8]
        + 1.07600174350434*x[13] - x[14] - x[20] == m.c6_rhs)
    m.c7 = pyo.Constraint(expr=x[0] - 40.9351218608642*x[1] - 43.2018652628815*x[2]
        - 45.3473311101868*x[3] - 39.805625287987*x[4] - 41.3125769494053*x[5]
        - 41.8781498541141*x[6] - 42.1403213448084*x[7] - 46.6038914670337*x[8] == m.c7_rhs)

    # Objective: sum of squared differences
    m.obj_expr = (
        (-x[15])**2 + (50000 - x[16])**2 + (42000 - x[17])**2
        + (40000 - x[18])**2 + (40000 - x[19])**2 + (45000 - x[20])**2
    )
    return m


def all_vars(m):
    return [getattr(m, f"x{i}") for i in range(1, 22)]


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
              "plot_output_dir":"results/immun_smoke/plots","output_csv_path":"results/immun_smoke/simplex_result.csv"},
    "full":  {"nscen":100,"target_nodes":300,"gap_stop_tol":1e-2,"time_limit":None,"enable_ef_ub":True,"ef_time_ub":43200.,"plot_every":None,
              "plot_output_dir":"results/immun_full/plots","output_csv_path":"results/immun_full/simplex_result.csv"},
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
    print("="*60 + f"\nimmun -- nfirst={nf}, seed={args.seed}\n" + "="*60)
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
