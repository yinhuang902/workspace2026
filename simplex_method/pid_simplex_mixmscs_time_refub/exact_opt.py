# exact_opt.py
"""
Compute "exact" optimal Kp, Ki, Kd

- Per-scenario exact optimum: solve each scenario model alone.
- Aggregate exact optimum: build a big model that shares (Kp,Ki,Kd)
  across all scenarios and minimizes the sum of scenario costs.

This is only for validation / visualization, so we do not try to be
super-optimized. We assume the number of scenarios is small (2-3).
"""

from __future__ import annotations
import pyomo.environ as pyo
from pyomo.opt import SolverStatus, TerminationCondition
from typing import Sequence, Dict, Any, Tuple, List


def _solve_single_scenario_exact(
    model: pyo.ConcreteModel,
    first_vars: Sequence[pyo.Var],
    solver_name: str = "gurobi",
    solver_opts: Dict[str, Any] | None = None,
) -> Tuple[Tuple[float, float, float], float]:
    """
    Solve one scenario model 'exactly' with free Kp,Ki,Kd.

    We clone the model to avoid messing with the persistent solver and
    existing BaseBundle. The clone uses obj_expr as the objective.
    """
    # clone
    m = model.clone()

    # locate first-stage vars in the clone by name
    first_vars_clone = []
    for v in first_vars:
        w = m.find_component(v.name)
        if w is None:
            raise RuntimeError(f"Cannot find first-stage var {v.name} in cloned model")
        first_vars_clone.append(w)

    # ensure we have a single active objective: use obj_expr
    if hasattr(m, "obj"):
        # de-activate existing objective added by BaseBundle
        try:
            m.obj.deactivate()
        except Exception:
            # if it's not an Objective or already inactive, ignore
            pass

    # create our temporary objective
    m._obj_exact = pyo.Objective(expr=m.obj_expr, sense=pyo.minimize)

    solver = pyo.SolverFactory(solver_name)
    if solver_opts:
        for k, v in solver_opts.items():
            solver.options[k] = v

    res = solver.solve(m, load_solutions=True)
    status = res.solver.status
    term = res.solver.termination_condition

    ok_terms = {TerminationCondition.optimal}
    # Some Pyomo versions also have locallyOptimal
    if hasattr(TerminationCondition, "locallyOptimal"):
        ok_terms.add(TerminationCondition.locallyOptimal)

    ok = (status == SolverStatus.ok) and (term in ok_terms)
    if not ok:
        print(f"[WARN] per-scenario exact solve failed: status={status}, term={term}")
        return (float("nan"), float("nan"), float("nan")), float("inf")

    K_vals = tuple(float(pyo.value(w)) for w in first_vars_clone)
    obj_val = float(pyo.value(m.obj_expr))
    return K_vals, obj_val


def _build_aggregate_model_from_models(
    model_list: Sequence[pyo.ConcreteModel],
) -> Tuple[pyo.ConcreteModel, Tuple[pyo.Var, pyo.Var, pyo.Var]]:
    """
    Build an aggregate Pyomo model that shares (Kp,Ki,Kd) across all
    scenarios and sums their obj_expr.

    Structure:
        mAgg.Kp, mAgg.Ki, mAgg.Kd  (shared first-stage vars)
        mAgg.scen[s].m             (clone of scenario s model)
        link constraints: mAgg.scen[s].m.Kp == mAgg.Kp, etc.
        objective: sum_s scen[s].m.obj_expr
    """
    if not model_list:
        raise ValueError("model_list is empty")

    import itertools as it

    base = model_list[0]
    # try to get bounds from the first model's K variables
    Kp0 = getattr(base, "Kp", None)
    Ki0 = getattr(base, "Ki", None)
    Kd0 = getattr(base, "Kd", None)
    if any(v is None for v in (Kp0, Ki0, Kd0)):
        raise RuntimeError("Cannot find (Kp,Ki,Kd) on the first model")

    def _b(v):
        lb = float(v.lb) if v.lb is not None else None
        ub = float(v.ub) if v.ub is not None else None
        return (lb, ub)

    mAgg = pyo.ConcreteModel()
    S = len(model_list)
    mAgg.S = pyo.RangeSet(0, S - 1)

    # shared first-stage variables
    mAgg.Kp = pyo.Var(bounds=_b(Kp0))
    mAgg.Ki = pyo.Var(bounds=_b(Ki0))
    mAgg.Kd = pyo.Var(bounds=_b(Kd0))

    # scenario blocks
    def _scen_block_rule(b, s_idx):
        # clone the scenario model
        m_s = model_list[s_idx].clone()
        # deactivate any existing objective (if BaseBundle already added one)
        if hasattr(m_s, "obj"):
            try:
                m_s.obj.deactivate()
            except Exception:
                pass
        b.m = m_s
        return

    mAgg.scen = pyo.Block(mAgg.S, rule=_scen_block_rule)

    # linking constraints: Kp_s == Kp_shared, etc.
    def _link_kp_rule(m, s):
        return m.scen[s].m.Kp == m.Kp

    def _link_ki_rule(m, s):
        return m.scen[s].m.Ki == m.Ki

    def _link_kd_rule(m, s):
        return m.scen[s].m.Kd == m.Kd

    mAgg.link_kp = pyo.Constraint(mAgg.S, rule=_link_kp_rule)
    mAgg.link_ki = pyo.Constraint(mAgg.S, rule=_link_ki_rule)
    mAgg.link_kd = pyo.Constraint(mAgg.S, rule=_link_kd_rule)

    # objective: sum of scenario obj_expr
    def _obj_rule(m):
        return sum(m.scen[s].m.obj_expr for s in m.S)

    mAgg.obj = pyo.Objective(rule=_obj_rule, sense=pyo.minimize)

    return mAgg, (mAgg.Kp, mAgg.Ki, mAgg.Kd)


def _solve_aggregate_exact(
    model_list: Sequence[pyo.ConcreteModel],
    solver_name: str = "gurobi",
    solver_opts: Dict[str, Any] | None = None,
) -> Tuple[Tuple[float, float, float], float]:
    """
    Solve the aggregate model that minimizes the sum of scenario costs.
    """
    mAgg, Kvars = _build_aggregate_model_from_models(model_list)
    solver = pyo.SolverFactory(solver_name)
    if solver_opts:
        for k, v in solver_opts.items():
            solver.options[k] = v

    res = solver.solve(mAgg, load_solutions=True)
    status = res.solver.status
    term = res.solver.termination_condition

    ok_terms = {TerminationCondition.optimal}
    if hasattr(TerminationCondition, "locallyOptimal"):
        ok_terms.add(TerminationCondition.locallyOptimal)

    ok = (status == SolverStatus.ok) and (term in ok_terms)
    if not ok:
        print(f"[WARN] aggregate exact solve failed: status={status}, term={term}")
        return (float("nan"), float("nan"), float("nan")), float("inf")

    K_vals = tuple(float(pyo.value(v)) for v in Kvars)
    obj_val = float(pyo.value(mAgg.obj))
    return K_vals, obj_val


def compute_exact_optima(
    model_list: Sequence[pyo.ConcreteModel],
    first_vars_list: Sequence[Sequence[pyo.Var]],
    solver_name: str = "gurobi",
    solver_opts: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """
    Public entry:

    Returns
    -------
    dict with keys:
        - 'per_scenario': list of dicts:
              [{'K': (Kp,Ki,Kd), 'obj': value}, ...]
        - 'aggregate': dict:
              {'K': (Kp,Ki,Kd), 'obj': value}
    """
    S = len(model_list)
    if len(first_vars_list) != S:
        raise ValueError("model_list and first_vars_list must have same length")

    per_scen = []
    for s in range(S):
        K_s, obj_s = _solve_single_scenario_exact(
            model_list[s],
            first_vars_list[s],
            solver_name=solver_name,
            solver_opts=solver_opts,
        )
        per_scen.append({"K": K_s, "obj": obj_s})

    return {
        "per_scenario": per_scen,
        "aggregate": {"K": None, "obj": float("nan")},
    }

