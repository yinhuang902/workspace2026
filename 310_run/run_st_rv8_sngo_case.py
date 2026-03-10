"""
run_st_rv8_sngo_case.py  -- Simplex runner for SNGO-master/Global/st_rv8

40 variables, 20 <= constraints (19 structural + sum<=400).
Concave quadratic objective with varying coefficients per variable.

Stage split: nfirst=5 (default), x1-x5 bounded [0,400] from sum constraint.
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
    ("c1_rhs",330.),("c2_rhs",425.),("c3_rhs",430.),("c4_rhs",405.),
    ("c5_rhs",355.),("c6_rhs",275.),("c7_rhs",345.),("c8_rhs",345.),
    ("c9_rhs",350.),("c10_rhs",350.),("c11_rhs",345.),("c12_rhs",335.),
    ("c13_rhs",375.),("c14_rhs",380.),("c15_rhs",425.),("c16_rhs",455.),
    ("c17_rhs",365.),("c18_rhs",365.),("c19_rhs",385.),("c20_rhs",400.),
]

def create_model():
    m = pyo.ConcreteModel()

    # Tighter first-stage bounds (do not cut off feasible region)
    first_stage_ubs = {
        1: 38.3334,
        2: 47.2223,
        3: 40.5556,
        4: 38.8889,
        5: 46.8751,
    }
    for i in range(1, 6):
        setattr(m, f"x{i}", pyo.Var(bounds=(0, first_stage_ubs[i]), initialize=0))

    for i in range(6,41):
        setattr(m,f"x{i}",pyo.Var(bounds=(0,None),initialize=0))
    for name,val in _RHS_BASE:
        setattr(m, name, pyo.Param(mutable=True, initialize=val))
    x=[getattr(m,f"x{i}") for i in range(1,41)]  # 0-indexed

    m.c1 =pyo.Constraint(expr=7*x[0]+4*x[5]+7*x[6]+6*x[11]+9*x[12]+2*x[13]+x[18]+5*x[19]+x[24]+5*x[25]+3*x[30]+9*x[31]+5*x[32]+x[37]+x[38]<=m.c1_rhs)
    m.c2 =pyo.Constraint(expr=4*x[0]+7*x[1]+7*x[6]+8*x[7]+9*x[12]+3*x[13]+6*x[14]+2*x[19]+6*x[20]+5*x[25]+3*x[26]+4*x[31]+6*x[32]+6*x[33]+6*x[38]+3*x[39]<=m.c2_rhs)
    m.c3 =pyo.Constraint(expr=x[1]+6*x[2]+8*x[7]+7*x[8]+9*x[13]+8*x[14]+8*x[15]+6*x[20]+5*x[21]+4*x[26]+2*x[27]+4*x[32]+7*x[33]+9*x[34]+2*x[39]<=m.c3_rhs)
    m.c4 =pyo.Constraint(expr=x[2]+5*x[3]+9*x[8]+6*x[9]+4*x[14]+9*x[15]+6*x[16]+7*x[21]+9*x[22]+8*x[27]+3*x[28]+7*x[33]+4*x[34]+3*x[35]<=m.c4_rhs)
    m.c5 =pyo.Constraint(expr=4*x[3]+7*x[4]+3*x[9]+6*x[10]+2*x[15]+8*x[16]+5*x[17]+2*x[22]+9*x[23]+6*x[28]+4*x[29]+3*x[34]+6*x[35]+6*x[36]<=m.c5_rhs)
    m.c6 =pyo.Constraint(expr=5*x[4]+5*x[5]+7*x[10]+4*x[11]+4*x[16]+6*x[17]+2*x[18]+4*x[23]+2*x[24]+x[29]+4*x[30]+4*x[35]+3*x[36]+4*x[37]<=m.c6_rhs)
    m.c7 =pyo.Constraint(expr=2*x[0]+3*x[5]+3*x[6]+5*x[11]+9*x[12]+9*x[17]+x[18]+4*x[19]+6*x[24]+5*x[25]+3*x[30]+7*x[31]+3*x[36]+5*x[37]+4*x[38]<=m.c7_rhs)
    m.c8 =pyo.Constraint(expr=9*x[0]+7*x[1]+3*x[6]+6*x[7]+7*x[12]+2*x[13]+x[18]+x[19]+4*x[20]+5*x[25]+2*x[26]+6*x[31]+5*x[32]+4*x[37]+4*x[38]+3*x[39]<=m.c8_rhs)
    m.c9 =pyo.Constraint(expr=6*x[0]+3*x[1]+4*x[2]+2*x[7]+7*x[8]+3*x[13]+7*x[14]+2*x[19]+3*x[20]+2*x[21]+6*x[26]+x[27]+6*x[32]+7*x[33]+9*x[38]+2*x[39]<=m.c9_rhs)
    m.c10=pyo.Constraint(expr=2*x[1]+8*x[2]+9*x[3]+x[8]+x[9]+6*x[14]+x[15]+6*x[20]+7*x[21]+6*x[22]+3*x[27]+2*x[28]+7*x[33]+6*x[34]+5*x[39]<=m.c10_rhs)
    m.c11=pyo.Constraint(expr=3*x[2]+6*x[3]+5*x[4]+6*x[9]+5*x[10]+8*x[15]+9*x[16]+6*x[21]+4*x[22]+x[23]+5*x[28]+2*x[29]+5*x[34]+4*x[35]<=m.c11_rhs)
    m.c12=pyo.Constraint(expr=3*x[3]+3*x[4]+9*x[5]+3*x[10]+8*x[11]+9*x[16]+4*x[17]+4*x[22]+3*x[23]+6*x[24]+6*x[29]+x[30]+6*x[35]+2*x[36]<=m.c12_rhs)
    m.c13=pyo.Constraint(expr=8*x[4]+2*x[5]+4*x[6]+8*x[11]+9*x[12]+3*x[17]+8*x[18]+x[23]+8*x[24]+8*x[25]+3*x[30]+x[31]+5*x[36]+7*x[37]<=m.c13_rhs)
    m.c14=pyo.Constraint(expr=x[0]+9*x[5]+x[6]+4*x[7]+9*x[12]+6*x[13]+6*x[18]+7*x[19]+x[24]+5*x[25]+7*x[26]+x[31]+8*x[32]+9*x[37]+2*x[38]<=m.c14_rhs)
    m.c15=pyo.Constraint(expr=3*x[0]+9*x[1]+4*x[6]+2*x[7]+x[8]+3*x[13]+9*x[14]+7*x[19]+7*x[20]+8*x[25]+7*x[26]+5*x[27]+4*x[32]+x[33]+6*x[38]+9*x[39]<=m.c15_rhs)
    m.c16=pyo.Constraint(expr=9*x[1]+6*x[2]+9*x[7]+5*x[8]+6*x[9]+6*x[14]+9*x[15]+5*x[20]+7*x[21]+8*x[26]+7*x[27]+x[28]+x[33]+8*x[34]+4*x[39]<=m.c16_rhs)
    m.c17=pyo.Constraint(expr=9*x[2]+9*x[3]+4*x[8]+2*x[9]+7*x[10]+4*x[15]+8*x[16]+3*x[21]+2*x[22]+2*x[27]+7*x[28]+3*x[29]+4*x[34]+9*x[35]<=m.c17_rhs)
    m.c18=pyo.Constraint(expr=5*x[3]+6*x[4]+8*x[9]+9*x[10]+6*x[11]+6*x[16]+4*x[17]+3*x[22]+3*x[23]+x[28]+9*x[29]+2*x[30]+4*x[35]+7*x[36]<=m.c18_rhs)
    m.c19=pyo.Constraint(expr=5*x[4]+7*x[5]+2*x[10]+8*x[11]+x[12]+9*x[17]+8*x[18]+6*x[23]+x[24]+4*x[29]+9*x[30]+7*x[31]+4*x[36]+6*x[37]<=m.c19_rhs)
    m.c20=pyo.Constraint(expr=sum(x)<=m.c20_rhs)

    a = [0.0004,0.00285,0.00155,0.0038,0.0044,0.0046,0.00085,0.00165,0.0025,0.00385,
         0.00355,0.0015,0.0037,0.00125,0.00095,0.0048,0.0015,0.0037,0.00125,0.00095,
         0.0045,0.00245,0.0004,0.0048,0.00485,0.00025,0.00435,0.00365,0.0002,0.00205,
         0.00165,0.00175,0.0048,5e-5,0.00155,0.00015,0.00245,0.00095,0.0038,0.0029]
    b = [0.0384,0.3876,0.1116,0.4636,0.044,0.3588,0.0272,0.231,0.27,0.308,
         0.3692,0.288,0.407,0.1175,0.1045,0.1632,0.135,0.0666,0.21,0.1425,
         0.882,0.3283,0.0648,0.0864,0.4753,0.046,0.7917,0.7008,0.0384,0.0164,
         0.0891,0.0945,0.7296,0.0023,0.1488,0.0189,0.0343,0.1045,0.608,0.0174]
    m.obj_expr = sum(-a[i]*x[i]**2 - b[i]*x[i] for i in range(40))
    return m

def build_models(nscen,nfirst=5,nparam=5,seed=1234,**kw):
    rng=JuliaMT19937(seed); ml=[]; fl=[]; mx=nparam
    for s in range(nscen):
        m=create_model(); av=[getattr(m,f"x{i}") for i in range(1,41)]; f=av[:nfirst]
        if s>0:
            for idx in range(min(mx,len(_RHS_BASE))):
                p,bv=_RHS_BASE[idx]; getattr(m,p).set_value(addnoise_le(bv,rng))
        ml.append(m); fl.append(f)
    return ml,fl

MODE_PARAMS={"smoke":{"nscen":5,"target_nodes":60,"gap_stop_tol":1e-6,"time_limit":300,"enable_ef_ub":True,"ef_time_ub":30.,"plot_every":None,"plot_output_dir":"results/st_rv8_smoke/plots","output_csv_path":"results/st_rv8_smoke/simplex_result.csv"},
 "full":{"nscen":100,"target_nodes":900,"gap_stop_tol":1e-3,"time_limit":60*60*12,"enable_ef_ub":True,"ef_time_ub":60.,"plot_every":None,"plot_output_dir":"results/st_rv8_full/plots","output_csv_path":"results/st_rv8_full/simplex_result.csv"}}
BUNDLE_OPTIONS={"NonConvex":2,"MIPGap":1e-1,"TimeLimit":60}
Q_MAX = -1e1

def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--mode",choices=("smoke","full"),default="smoke"); ap.add_argument("--seed",type=int,default=1234); ap.add_argument("--nfirst",type=int,default=5)
    args=ap.parse_args(); nf=args.nfirst; cfg=dict(MODE_PARAMS[args.mode])
    out=Path(cfg["output_csv_path"]); out.parent.mkdir(parents=True,exist_ok=True)
    if cfg["plot_output_dir"]: Path(cfg["plot_output_dir"]).mkdir(parents=True,exist_ok=True)
    print("="*60+f"\nst_rv8 -- nfirst={nf}, seed={args.seed}\n"+"="*60)
    t0=perf_counter(); ml,fl=build_models(cfg["nscen"],nf,nf,args.seed); S=len(ml)
    bb=[BaseBundle(ml[s], options=BUNDLE_OPTIONS, q_max=Q_MAX) for s in range(S)]
    mb=[MSBundle(ml[s],fl[s],options=BUNDLE_OPTIONS) for s in range(S)]
    res=run_pid_simplex_3d(model_list=ml,first_vars_list=fl,base_bundles=bb,ms_bundles=mb,target_nodes=cfg["target_nodes"],gap_stop_tol=cfg["gap_stop_tol"],time_limit=cfg["time_limit"],enable_ef_ub=cfg["enable_ef_ub"],ef_time_ub=cfg["ef_time_ub"],plot_every=cfg["plot_every"],plot_output_dir=cfg["plot_output_dir"],output_csv_path=str(out),enable_3d_plot=False,axis_labels=tuple(f"x{i+1}" for i in range(nf)))
    t1=perf_counter(); LB=res.get("LB_hist",[]); UB=res.get("UB_hist",[])
    if LB and UB: print(f"\nLB_sum={float(LB[-1]):.12f} UB_sum={float(UB[-1]):.12f}\nLB/S={float(LB[-1])/S:.12f} UB/S={float(UB[-1])/S:.12f}")
    print(f"{'='*60}\nDone. {t1-t0:.2f}s CSV: {out}\n{'='*60}")

if __name__=="__main__": main()
