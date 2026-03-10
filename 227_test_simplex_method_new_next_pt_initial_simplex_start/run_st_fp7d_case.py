"""
run_st_fp7d_case.py �?Simplex runner for SNGO-master/Global/st_fp7d
Same constraints as st_fp7a. Objective: Min Σ(8xi �?0.5xi²) i=1..20
"""
import argparse
from pathlib import Path
from time import perf_counter
import pyomo.environ as pyo
from bundles import BaseBundle, MSBundle
from simplex_specialstart import run_pid_simplex_3d
from run_st_fp7a_case import JuliaMT19937, addnoise_le

_RHS_BASE = [("c1_rhs",-5.),("c2_rhs",2.),("c3_rhs",-1.),("c4_rhs",-3.),("c5_rhs",5.),
             ("c6_rhs",4.),("c7_rhs",-1.),("c8_rhs",0.),("c9_rhs",9.),("c10_rhs",40.)]

def _add_constraints(m, x):
    m.c1_rhs=pyo.Param(mutable=True,initialize=-5.); m.c2_rhs=pyo.Param(mutable=True,initialize=2.)
    m.c3_rhs=pyo.Param(mutable=True,initialize=-1.); m.c4_rhs=pyo.Param(mutable=True,initialize=-3.)
    m.c5_rhs=pyo.Param(mutable=True,initialize=5.); m.c6_rhs=pyo.Param(mutable=True,initialize=4.)
    m.c7_rhs=pyo.Param(mutable=True,initialize=-1.); m.c8_rhs=pyo.Param(mutable=True,initialize=0.)
    m.c9_rhs=pyo.Param(mutable=True,initialize=9.); m.c10_rhs=pyo.Param(mutable=True,initialize=40.)
    m.c1 =pyo.Constraint(expr=-3*x[0]+7*x[1]-5*x[3]+x[4]+x[5]+2*x[7]-x[8]-x[9]-9*x[10]+3*x[11]+5*x[12]+x[15]+7*x[16]-7*x[17]-4*x[18]-6*x[19]<=m.c1_rhs)
    m.c2 =pyo.Constraint(expr=7*x[0]-5*x[2]+x[3]+x[4]+2*x[6]-x[7]-x[8]-9*x[9]+3*x[10]+5*x[11]+x[14]+7*x[15]-7*x[16]-4*x[17]-6*x[18]-3*x[19]<=m.c2_rhs)
    m.c3 =pyo.Constraint(expr=-5*x[1]+x[2]+x[3]+2*x[5]-x[6]-x[7]-9*x[8]+3*x[9]+5*x[10]+x[13]+7*x[14]-7*x[15]-4*x[16]-6*x[17]-3*x[18]+7*x[19]<=m.c3_rhs)
    m.c4 =pyo.Constraint(expr=-5*x[0]+x[1]+x[2]+2*x[4]-x[5]-x[6]-9*x[7]+3*x[8]+5*x[9]+x[12]+7*x[13]-7*x[14]-4*x[15]-6*x[16]-3*x[17]+7*x[18]<=m.c4_rhs)
    m.c5 =pyo.Constraint(expr=x[0]+x[1]+2*x[3]-x[4]-x[5]-9*x[6]+3*x[7]+5*x[8]+x[11]+7*x[12]-7*x[13]-4*x[14]-6*x[15]-3*x[16]+7*x[17]-5*x[19]<=m.c5_rhs)
    m.c6 =pyo.Constraint(expr=x[0]+2*x[2]-x[3]-x[4]-9*x[5]+3*x[6]+5*x[7]+x[10]+7*x[11]-7*x[12]-4*x[13]-6*x[14]-3*x[15]+7*x[16]-5*x[18]+x[19]<=m.c6_rhs)
    m.c7 =pyo.Constraint(expr=2*x[1]-x[2]-x[3]-9*x[4]+3*x[5]+5*x[6]+x[9]+7*x[10]-7*x[11]-4*x[12]-6*x[13]-3*x[14]+7*x[15]-5*x[17]+x[18]+x[19]<=m.c7_rhs)
    m.c8 =pyo.Constraint(expr=2*x[0]-x[1]-x[2]-9*x[3]+3*x[4]+5*x[5]+x[8]+7*x[9]-7*x[10]-4*x[11]-6*x[12]-3*x[13]+7*x[14]-5*x[16]+x[17]+x[18]<=m.c8_rhs)
    m.c9 =pyo.Constraint(expr=-x[0]-x[1]-9*x[2]+3*x[3]+5*x[4]+x[7]+7*x[8]-7*x[9]-4*x[10]-6*x[11]-3*x[12]+7*x[13]-5*x[15]+x[16]+x[17]+2*x[19]<=m.c9_rhs)
    m.c10=pyo.Constraint(expr=sum(x)<=m.c10_rhs)

def create_model():
    m = pyo.ConcreteModel()
    m.x1=pyo.Var(bounds=(0,18.22),initialize=0); m.x2=pyo.Var(bounds=(0,17.40),initialize=0)
    m.x3=pyo.Var(bounds=(0,28.81),initialize=0); m.x4=pyo.Var(bounds=(0,25.78),initialize=0); m.x5=pyo.Var(bounds=(0,19.14),initialize=0)
    for i in range(6,21): setattr(m,f"x{i}",pyo.Var(bounds=(0,None),initialize=0))
    x=[getattr(m,f"x{i}") for i in range(1,21)]
    _add_constraints(m, x)
    m.obj_expr = sum(8*v - 0.5*v**2 for v in x)
    return m

def build_models(nscen,nfirst=2,nparam=2,seed=1234,print_first_k_rhs=0):
    rng=JuliaMT19937(seed); ml=[]; fl=[]; mx=nparam  # PlasmoOld cap is nparam, not nparam-1
    for s in range(nscen):
        m=create_model(); av=[getattr(m,f"x{i}") for i in range(1,21)]; f=av[:nfirst]
        if s>0:
            for idx in range(min(mx,len(_RHS_BASE))):
                p,b=_RHS_BASE[idx]; getattr(m,p).set_value(addnoise_le(b,rng))
        ml.append(m); fl.append(f)
    return ml,fl

MODE_PARAMS={"smoke":{"nscen":10,"target_nodes":60,"gap_stop_tol":1e-6,"time_limit":300,"enable_ef_ub":True,"ef_time_ub":30.,"plot_every":None,"plot_output_dir":"results/st_fp7d_smoke/plots","output_csv_path":"results/st_fp7d_smoke/simplex_result.csv"},
 "full":{"nscen":100,"target_nodes":900,"gap_stop_tol":1e-3,"time_limit": 60*60*12,"enable_ef_ub":True,"ef_time_ub":60,"plot_every":None,"plot_output_dir":"results/st_fp7d_full/plots","output_csv_path":"results/st_fp7d_full/simplex_result.csv"}}
BUNDLE_OPTIONS={"NonConvex":2,"MIPGap":1e-2, "TimeLimit":60}
Q_MAX = -1e1

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--mode",choices=("smoke","full"),default="full"); ap.add_argument("--seed",type=int,default=1234); ap.add_argument("--nfirst",type=int,default=5)
    args=ap.parse_args(); nf=args.nfirst; cfg=dict(MODE_PARAMS[args.mode])
    out=Path(cfg["output_csv_path"]); out.parent.mkdir(parents=True,exist_ok=True)
    if cfg["plot_output_dir"]: Path(cfg["plot_output_dir"]).mkdir(parents=True,exist_ok=True)
    print("="*60+f"\nst_fp7d �?nfirst={nf}, seed={args.seed}\n"+"="*60)
    t0=perf_counter(); ml,fl=build_models(cfg["nscen"],nf,nf,args.seed); S=len(ml)
    bb=[BaseBundle(ml[s], options=BUNDLE_OPTIONS, q_max=Q_MAX) for s in range(S)]
    mb=[MSBundle(ml[s],fl[s],options=BUNDLE_OPTIONS) for s in range(S)]
    res=run_pid_simplex_3d(model_list=ml,first_vars_list=fl,base_bundles=bb,ms_bundles=mb,target_nodes=cfg["target_nodes"],gap_stop_tol=cfg["gap_stop_tol"],time_limit=cfg["time_limit"],enable_ef_ub=cfg["enable_ef_ub"],ef_time_ub=cfg["ef_time_ub"],plot_every=cfg["plot_every"],plot_output_dir=cfg["plot_output_dir"],output_csv_path=str(out),enable_3d_plot=False,axis_labels=tuple(f"x{i+1}" for i in range(nf)))
    t1=perf_counter(); LB=res.get("LB_hist",[]); UB=res.get("UB_hist",[])
    if LB and UB: print(f"\nLB_sum={float(LB[-1]):.12f} UB_sum={float(UB[-1]):.12f}\nLB/S={float(LB[-1])/S:.12f} UB/S={float(UB[-1])/S:.12f}")
    print(f"{'='*60}\nDone. {t1-t0:.2f}s CSV: {out}\n{'='*60}")

if __name__=="__main__": main()
