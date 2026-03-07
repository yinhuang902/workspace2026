"""
run_st_rv3_case.py â€?Simplex runner for SNGO-master/Global/st_rv3

20 variables, 20 <= constraints (19 structural + sum<=200).
Concave quadratic objective with varying coefficients per variable.
More constraints than st_rv2 â†?richer stochastic perturbation.

Stage split: nfirst=2 (default), x1-x2 bounded [0,200].
PlasmoOld addnoise: ub + |ub|*U(0,2) for nonzero, U(0,10) for zero.
"""
import argparse
from pathlib import Path
from time import perf_counter
import pyomo.environ as pyo
from bundles import BaseBundle, MSBundle
from simplex_specialstart import run_pid_simplex_3d
from run_st_fp7a_case import JuliaMT19937, addnoise_le

_RHS_BASE = [
    ("c1_rhs",220.),("c2_rhs",175.),("c3_rhs",215.),("c4_rhs",195.),
    ("c5_rhs",145.),("c6_rhs",185.),("c7_rhs",225.),("c8_rhs",215.),
    ("c9_rhs",175.),("c10_rhs",155.),("c11_rhs",210.),("c12_rhs",190.),
    ("c13_rhs",205.),("c14_rhs",245.),("c15_rhs",160.),("c16_rhs",225.),
    ("c17_rhs",195.),("c18_rhs",240.),("c19_rhs",215.),("c20_rhs",200.),
]

def create_model():
    m = pyo.ConcreteModel()
    m.x1=pyo.Var(bounds=(0,200),initialize=0); m.x2=pyo.Var(bounds=(0,200),initialize=0)
    for i in range(3,21): setattr(m,f"x{i}",pyo.Var(bounds=(0,None),initialize=0))
    for name,val in _RHS_BASE:
        setattr(m, name, pyo.Param(mutable=True, initialize=val))
    x=[getattr(m,f"x{i}") for i in range(1,21)]  # 0-indexed: x[0]=x1, x[19]=x20

    # 19 structural constraints + sum constraint (from Julia st_rv3)
    m.c1 =pyo.Constraint(expr=8*x[0]+5*x[5]+4*x[6]+6*x[11]+6*x[12]+9*x[13]+5*x[18]+x[19]<=m.c1_rhs)
    m.c2 =pyo.Constraint(expr=3*x[0]+4*x[1]+3*x[6]+7*x[7]+4*x[12]+9*x[13]+3*x[14]+2*x[19]<=m.c2_rhs)
    m.c3 =pyo.Constraint(expr=2*x[1]+x[2]+6*x[7]+8*x[8]+9*x[13]+9*x[14]+8*x[15]<=m.c3_rhs)
    m.c4 =pyo.Constraint(expr=7*x[2]+x[3]+7*x[8]+9*x[9]+2*x[14]+4*x[15]+9*x[16]<=m.c4_rhs)
    m.c5 =pyo.Constraint(expr=4*x[3]+4*x[4]+x[9]+3*x[10]+7*x[15]+2*x[16]+8*x[17]<=m.c5_rhs)
    m.c6 =pyo.Constraint(expr=9*x[4]+5*x[5]+5*x[10]+7*x[11]+x[16]+4*x[17]+6*x[18]<=m.c6_rhs)
    m.c7 =pyo.Constraint(expr=5*x[0]+5*x[5]+3*x[6]+8*x[11]+5*x[12]+9*x[17]+9*x[18]+x[19]<=m.c7_rhs)
    m.c8 =pyo.Constraint(expr=x[0]+9*x[1]+9*x[6]+3*x[7]+9*x[12]+7*x[13]+4*x[18]+x[19]<=m.c8_rhs)
    m.c9 =pyo.Constraint(expr=3*x[0]+6*x[1]+3*x[2]+4*x[7]+2*x[8]+6*x[13]+3*x[14]+8*x[18]+x[19]<=m.c9_rhs)
    m.c10=pyo.Constraint(expr=x[1]+2*x[2]+8*x[3]+4*x[8]+x[9]+9*x[14]+6*x[15]<=m.c10_rhs)
    m.c11=pyo.Constraint(expr=9*x[2]+3*x[3]+6*x[4]+x[9]+6*x[10]+9*x[15]+8*x[16]<=m.c11_rhs)
    m.c12=pyo.Constraint(expr=6*x[3]+3*x[4]+3*x[5]+6*x[10]+3*x[11]+8*x[16]+9*x[17]<=m.c12_rhs)
    m.c13=pyo.Constraint(expr=9*x[4]+8*x[5]+2*x[6]+7*x[11]+8*x[12]+4*x[17]+3*x[18]<=m.c13_rhs)
    m.c14=pyo.Constraint(expr=4*x[0]+6*x[5]+9*x[6]+x[7]+6*x[12]+9*x[13]+8*x[18]+6*x[19]<=m.c14_rhs)
    m.c15=pyo.Constraint(expr=7*x[0]+3*x[1]+7*x[6]+4*x[7]+2*x[8]+x[13]+3*x[14]+5*x[19]<=m.c15_rhs)
    m.c16=pyo.Constraint(expr=7*x[1]+9*x[2]+7*x[7]+9*x[8]+5*x[9]+2*x[14]+6*x[15]<=m.c16_rhs)
    m.c17=pyo.Constraint(expr=6*x[2]+9*x[3]+8*x[8]+4*x[9]+2*x[10]+6*x[15]+4*x[16]<=m.c17_rhs)
    m.c18=pyo.Constraint(expr=5*x[3]+5*x[4]+7*x[9]+8*x[10]+9*x[11]+8*x[16]+6*x[17]<=m.c18_rhs)
    m.c19=pyo.Constraint(expr=7*x[4]+5*x[5]+6*x[10]+2*x[11]+8*x[12]+6*x[17]+9*x[18]<=m.c19_rhs)
    m.c20=pyo.Constraint(expr=sum(x)<=m.c20_rhs)

    # Objective: concave quadratic with varying coefficients (from Julia)
    a = [0.00055,0.0019,0.0002,0.00095,0.0046,0.0035,0.00315,0.00475,0.0048,0.003,
         0.00265,0.0017,0.0012,0.00295,0.00315,0.0021,0.00225,0.0034,0.001,0.00305]
    b = [0.0583,0.2318,0.0108,0.1634,0.138,0.357,0.1953,0.361,0.1824,0.162,
         0.4346,0.1054,0.2376,0.0059,0.189,0.0252,0.099,0.3604,0.022,0.3294]
    m.obj_expr = sum(-a[i]*x[i]**2 - b[i]*x[i] for i in range(20))
    return m

def build_models(nscen,nfirst=2,nparam=2,seed=1234,**kw):
    rng=JuliaMT19937(seed); ml=[]; fl=[]; mx=max(nparam-1,0)
    for s in range(nscen):
        m=create_model(); av=[getattr(m,f"x{i}") for i in range(1,21)]; f=av[:nfirst]
        if s>0:
            for idx in range(min(mx,len(_RHS_BASE))):
                p,bv=_RHS_BASE[idx]; getattr(m,p).set_value(addnoise_le(bv,rng))
        ml.append(m); fl.append(f)
    return ml,fl

MODE_PARAMS={"smoke":{"nscen":10,"target_nodes":60,"gap_stop_tol":1e-6,"time_limit":300,"enable_ef_ub":True,"ef_time_ub":30.,"plot_every":None,"plot_output_dir":"results/st_rv3_smoke/plots","output_csv_path":"results/st_rv3_smoke/simplex_result.csv"},
 "full":{"nscen":100,"target_nodes":300,"gap_stop_tol":1e-2,"time_limit":None,"enable_ef_ub":True,"ef_time_ub":43200.,"plot_every":None,"plot_output_dir":"results/st_rv3_full/plots","output_csv_path":"results/st_rv3_full/simplex_result.csv"}}
BUNDLE_OPTIONS={"NonConvex":2,"MIPGap":1e-1}
Q_MAX = 1e10

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--mode",choices=("smoke","full"),default="smoke"); ap.add_argument("--seed",type=int,default=1234); ap.add_argument("--nfirst",type=int,default=2)
    args=ap.parse_args(); nf=args.nfirst; cfg=dict(MODE_PARAMS[args.mode])
    out=Path(cfg["output_csv_path"]); out.parent.mkdir(parents=True,exist_ok=True)
    if cfg["plot_output_dir"]: Path(cfg["plot_output_dir"]).mkdir(parents=True,exist_ok=True)
    print("="*60+f"\nst_rv3 â€?nfirst={nf}, seed={args.seed}\n"+"="*60)
    t0=perf_counter(); ml,fl=build_models(cfg["nscen"],nf,nf,args.seed); S=len(ml)
    bb=[BaseBundle(ml[s], options=BUNDLE_OPTIONS, q_max=Q_MAX) for s in range(S)]
    mb=[MSBundle(ml[s],fl[s],options=BUNDLE_OPTIONS) for s in range(S)]
    res=run_pid_simplex_3d(model_list=ml,first_vars_list=fl,base_bundles=bb,ms_bundles=mb,target_nodes=cfg["target_nodes"],gap_stop_tol=cfg["gap_stop_tol"],time_limit=cfg["time_limit"],enable_ef_ub=cfg["enable_ef_ub"],ef_time_ub=cfg["ef_time_ub"],plot_every=cfg["plot_every"],plot_output_dir=cfg["plot_output_dir"],output_csv_path=str(out),enable_3d_plot=False,axis_labels=tuple(f"x{i+1}" for i in range(nf)))
    t1=perf_counter(); LB=res.get("LB_hist",[]); UB=res.get("UB_hist",[])
    if LB and UB: print(f"\nLB_sum={float(LB[-1]):.12f} UB_sum={float(UB[-1]):.12f}\nLB/S={float(LB[-1])/S:.12f} UB/S={float(UB[-1])/S:.12f}")
    print(f"{'='*60}\nDone. {t1-t0:.2f}s CSV: {out}\n{'='*60}")

if __name__=="__main__": main()
