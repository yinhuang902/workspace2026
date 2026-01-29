# modeling.py
import numpy as np
import itertools as it
import csv
from tqdm import tqdm
import pyomo.environ as pyo
import pyomo.dae as dae
from pyomo.opt import SolverStatus, TerminationCondition
from scipy.spatial import Delaunay
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from time import perf_counter
from pyomo.contrib.fbbt.fbbt import fbbt
from pyomo.core import Objective
from pyomo.contrib.alternative_solutions.obbt import obbt_analysis

# ------------------------- PID scenario model -------------------------
def build_pid_model(T=15, h=0.75, scen=None, weights=(10.0, 0.01),
                    bounds=None, use_cvar=False, alpha=0.95, nfe=20):
    """
    Build PID model using Pyomo DAE (Continuous time formulation).
    Matches structure of pidspmodel_snog.py and test_ms_dae_model.py EXACTLY.
    """
    assert scen is not None, "Please provide a scen dict"
    
    # Map input scen dict to what build_pid_model_dae expects
    # Input scen has: tau_xs, tau_us, tau_ds, d (list), sp
    tau_xs = scen["tau_xs"]
    tau_us = scen["tau_us"]
    tau_ds = scen["tau_ds"]
    disturbances = scen["d"]
    setpoint = scen["sp"]
    
    # We expect nfe+1 data points for disturbances
    assert len(disturbances) == nfe+1, f"Expected {nfe+1} disturbance data points, got d={len(disturbances)}"

    if bounds is None:
        bounds = {}
    
    # Bounds matching stochastic_pid.py / pidspmodel_snog.py
    # Note: test_ms_dae_model.py uses hardcoded bounds, we keep flexibility here but default to same
    bx = bounds.get("x",  (-2.5, 2.5))
    bu = bounds.get("u",  (-5.0, 5.0))
    bKp= bounds.get("Kp", (-1, 0))      # Updated to match test script
    bKi= bounds.get("Ki", (-101, -99))  # Updated to match test script
    bKd= bounds.get("Kd", (0, 1))       # Updated to match test script
    be = bounds.get("e",  (None, None))
    bI = bounds.get("I",  (None, None))

    m = pyo.ConcreteModel()
    
    # Time set
    m.time = pyo.RangeSet(0, T)
    m.t = dae.ContinuousSet(bounds=(0, T))
    
    # Parameters
    m.x_setpoint = pyo.Param(initialize=setpoint)
    m.tau_xs = pyo.Param(initialize=tau_xs)
    m.tau_us = pyo.Param(initialize=tau_us)
    m.tau_ds = pyo.Param(initialize=tau_ds)
    m.d_s = pyo.Param(m.t, initialize=0, mutable=True)
    
    # Variables
    m.K_p = pyo.Var(domain=pyo.Reals, bounds=bKp)
    m.K_i = pyo.Var(domain=pyo.Reals, bounds=bKi)
    m.K_d = pyo.Var(domain=pyo.Reals, bounds=bKd)
    
    m.x_s = pyo.Var(m.t, domain=pyo.Reals, bounds=bx)
    m.e_s = pyo.Var(m.t, domain=pyo.Reals, bounds=be)
    m.u_s = pyo.Var(m.t, domain=pyo.Reals, bounds=bu)
    
    # Derivative variables
    m.dxdt = dae.DerivativeVar(m.x_s, wrt=m.t)
    m.dedt = dae.DerivativeVar(m.e_s, wrt=m.t)
    
    # Constraints
    @m.Constraint(m.t)
    def dxdt_con(m, t):
        if t == m.t.first():
            return pyo.Constraint.Skip
        else:
            return m.dxdt[t] == -m.tau_xs*m.x_s[t] + m.tau_us*m.u_s[t] + m.tau_ds*m.d_s[t]
    
    m.x_init_cond = pyo.Constraint(expr=m.x_s[m.t.first()] == 0)
    
    @m.Constraint(m.t)
    def e_con(m, t):
        return m.e_s[t] == m.x_s[t] - m.x_setpoint
    
    # Integral (like in paper)
    m.I = pyo.Var(m.t, bounds=bI)
    @m.Constraint(m.t)
    def integral(m, t):
        if t == m.t.first():
            return m.I[t] == 0
        else:
            return m.I[t] == m.I[m.t.prev(t)] + (t - m.t.prev(t)) * m.e_s[t]
    
    @m.Constraint(m.t)
    def u_con(m, t):
        return m.u_s[t] == m.K_p*m.e_s[t] + m.K_i*m.I[t] + m.K_d*m.dedt[t]
    
    # Objective (using DAE Integral)
    w_e, w_u = weights
    def e_sq_integral_rule(m, t):
        return w_e*m.e_s[t]**2 + w_u*m.u_s[t]**2
    m.e_sq_integral = dae.Integral(m.t, wrt=m.t, rule=e_sq_integral_rule)
    
    # Create obj_expr as Expression for compatibility with MS problem code
    m.obj_expr = pyo.Expression(expr=m.e_sq_integral)
    
    # Discretize
    discretizer = pyo.TransformationFactory('dae.finite_difference')
    discretizer.apply_to(m, nfe=nfe, wrt=m.t, scheme='BACKWARD')
    
    # Set disturbance parameters
    # Note: disturbances is a list of nfe+1 values
    # After discretization, m.t has nfe+1 points
    
    # Get sorted time points
    time_points = sorted(m.t)
    
    for i, t in enumerate(time_points):
        if i < len(disturbances):
            m.d_s[t] = disturbances[i]
            
    return m, [m.K_p, m.K_i, m.K_d]


def load_scenarios_from_csv(csv_path: str, nfe=None,
                            sp0: float = 0.0, sp1: float = 0.5,
                            tau_xs_col: str = "tau_xs", 
                            tau_us_col: str = "tau_us",
                            tau_ds_col: str = "tau_ds",
                            disturb_prefix: str = "disturbance_",
                            setpoint_change_col: str = "setpoint_change"):
    """
    Load scenarios from CSV file.
    
    Parameters:
    -----------
    nfe : int
        Number of finite elements (data should have nfe+1 points)
    """
    scens = []
    if nfe is None:
        with open(csv_path, "r", newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            fields  = reader.fieldnames or []
            max_idx = -1
            for name in fields:
                if name.startswith(disturb_prefix):
                    try:
                        k = int(name[len(disturb_prefix):])
                        max_idx = max(max_idx, k)
                    except:
                        pass
            if max_idx < 0:
                raise ValueError(f"No perturbation column prefix found {disturb_prefix}k")
            nfe = max_idx  # max_idx is the last index, so nfe = max_idx gives nfe+1 points

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            tau_xs = float(row[tau_xs_col])
            tau_us = float(row[tau_us_col])
            tau_ds = float(row[tau_ds_col])

            d = []
            for t in range(nfe+1):  # nfe+1 data points
                col = f"{disturb_prefix}{t}"
                d.append(float(row[col]))

            # Use constant setpoint (sp1) like stochastic_pid.py
            # Ignore setpoint_change_col to match stochastic_pid behavior
            sp = sp1  # Scalar constant setpoint

            scens.append({"tau_xs": tau_xs, "tau_us": tau_us, "tau_ds": tau_ds, "d": d, "sp": sp})
    return scens, nfe

def build_models_from_csv(csv_path: str, T: float = 15, nfe: int = 20,
                          weights=(1.0, 0.01), bounds=None,
                          sp0: float = 0.0, sp1: float = 0.5,
                          tau_xs_col: str = "tau_xs", 
                          tau_us_col: str = "tau_us",
                          tau_ds_col: str = "tau_ds",
                          disturb_prefix: str = "disturbance_",
                          setpoint_change_col: str = "setpoint_change",
                          max_scenarios=None, skip=0):
    """
    Build PID models from CSV data.
    
    Parameters:
    -----------
    T : float
        Total time horizon (default 15)
    nfe : int
        Number of finite elements (default 20), creates nfe+1 time points
    """
    h = T / nfe  # Calculate step size
    
    scens, nfe_data = load_scenarios_from_csv(
        csv_path=csv_path, nfe=nfe, sp0=sp0, sp1=sp1,
        tau_xs_col=tau_xs_col, tau_us_col=tau_us_col, tau_ds_col=tau_ds_col,
        disturb_prefix=disturb_prefix,
        setpoint_change_col=setpoint_change_col,
    )
    if skip or max_scenarios:
        scens = scens[skip: (skip + max_scenarios) if max_scenarios else None]

    model_list, first_stg_vars_list = [], []
    for scen in scens:
        m, yvars = build_pid_model(T=T, h=h, scen=scen, weights=weights, bounds=bounds, nfe=nfe)
        model_list.append(m)
        first_stg_vars_list.append(yvars)

    m_tmpl_list = [model_list[0], first_stg_vars_list[0]]
    return model_list, first_stg_vars_list, m_tmpl_list, nfe
