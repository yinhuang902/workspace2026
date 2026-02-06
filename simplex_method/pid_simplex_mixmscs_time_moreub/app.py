from time import perf_counter  

from modeling import build_models_from_csv
from bundles import BaseBundle, MSBundle
from utils import (
    tighten_bounds_one_model,
    MIN_DIST, ACTIVE_TOL, GAP_STOP_TOL,
)
from simplex_specialstart import run_pid_simplex_3d

# setup scenario number and target_nodes
csv_path       = "data.csv"
max_scenarios  = 2
target_nodes   = 100


bounds={
    "Kp": (-1, 0),
    "Ki": (-101, -99),
    "Kd": (0, 1),
    "x": (-2.5, 2.5),
    "u": (-5.0, 5.0),
    "e": (None, None),
    "I": (None, None),
}
bounds={
    "Kp": (-10.0, 10.0),
    "Ki": (-100.0, 100.0),
    "Kd": (-100.0, 1000.0),
    "x": (-2.5, 2.5),
    "u": (-5.0, 5.0),
    "e": (None, None),
    "I": (None, None),
}

weights = (10.0, 0.01)


# ====== stage 1：load data and generate models ======
T   = 15.0          
nfe = 20            

t_load0 = perf_counter()
model_list, first_stg_vars_list, m_tmpl_list, nfe = build_models_from_csv(
    csv_path=csv_path,
    T=T,
    nfe=nfe,
    weights=weights,
    bounds=bounds,
    sp0=0.0,
    sp1=0.5,
    tau_xs_col="tau_xs",
    tau_us_col="tau_us",
    tau_ds_col="tau_ds",          
    disturb_prefix="disturbance_",
    setpoint_change_col="setpoint_change",
    max_scenarios=max_scenarios,
    skip=0,
)
t_load1 = perf_counter()
print(f"[Time] Data load & scenario build: {t_load1 - t_load0:.3f}s")


# ====== stage 1.5：FBBT / OBBT ======
# same as snog: FBBT open,OBBT open
obbt_solver_opts = {
    "NonConvex": 2,
    "MIPGap": 1,     
    "TimeLimit": 15   
}
for m, yvars in zip(model_list, first_stg_vars_list):
    tighten_bounds_one_model(m, yvars,
                             use_fbbt=False,
                             use_obbt=False,
                             obbt_solver_name="gurobi",
                             obbt_solver_opts=obbt_solver_opts,
                             max_rounds=3, tol=1e-6, verbose=True)

# ====== stage 2: Persistent Solver Packaging ======
ub_options = {
    'NonConvex': 2,        
}
lb_options = {
    'NonConvex': 2,
    'MIPGap': 1e-1,            
    'TimeLimit': 15    
}

t_wrap0 = perf_counter()
base_bundles = [BaseBundle(m, ub_options) for m in model_list]  
ms_bundles   = [MSBundle(m, yvars, lb_options) for m, yvars in zip(model_list, first_stg_vars_list)]  # LB 侧
t_wrap1 = perf_counter()
print(f"[Time] Persistent wrapper (GurobiPersistent) setup: {t_wrap1 - t_wrap0:.3f}s")

# ====== stage 3：main loop ======
agg_bundle = None  

t_run0 = perf_counter()
hist = run_pid_simplex_3d(
    base_bundles=base_bundles,
    ms_bundles=ms_bundles,
    model_list=model_list,
    first_vars_list=first_stg_vars_list,
    target_nodes=target_nodes,
    min_dist=0,
    active_tol=ACTIVE_TOL,
    verbose=True,
    agg_bundle=agg_bundle,
    gap_stop_tol=1e-4,
    plot_every=1,
    use_exact_opt=False,
    exact_solver_name="gurobi",
    exact_solver_opts={"NonConvex": 2, "TimeLimit": 60},
    time_limit=60*10*1
)
t_run1 = perf_counter()
print(f"[Time] Main loop total: {t_run1 - t_run0:.3f}s")

# ====== print output ======
print("\n==== Done ====")
print(f"Total nodes: {len(hist['nodes'])}")
print(f"Best UB: {min(hist['UB_hist']) if hist['UB_hist'] else None}")
print(f"Last LB: {hist['LB_hist'][-1] if hist['LB_hist'] else None}")