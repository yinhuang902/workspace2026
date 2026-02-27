#!/usr/bin/env python
"""
run_farmer_single_scenario_simplex.py
=====================================
Single-scenario version of the 3D simplex-style tetrahedral refinement
on the Farmer feasible region  xw >= 0, xc >= 0, xb >= 0, xw+xc+xb <= TOTAL.

Uses ONLY the first (good) scenario (yield=1.2) instead of the
full 3-scenario expected value.  All other features are identical to
run_farmer_visual_simplex_with_boundaries.py.
"""

import argparse
import io
import json
import math
import os
import shutil
import sys
import time as _time
from itertools import combinations
from pathlib import Path
from time import perf_counter

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import pyomo.environ as pyo
from pyomo.opt import SolverFactory, TerminationCondition

# ---------------------------------------------------------------------------
# Windows console encoding fix
# ---------------------------------------------------------------------------
if sys.stdout.encoding and sys.stdout.encoding.lower() not in ("utf-8", "utf8"):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding=sys.stdout.encoding, errors="replace")
    sys.stderr = io.TextIOWrapper(
        sys.stderr.buffer, encoding=sys.stderr.encoding, errors="replace")

# ---------------------------------------------------------------------------
# Problem parameters
# ---------------------------------------------------------------------------
TOTAL = 500.0

SCENARIOS = {
    "good": {"yield": 1.2, "prob": 1.0},
}
S = len(SCENARIOS)   # 1
PROBS = [SCENARIOS[k]["prob"] for k in SCENARIOS]
YIELDS = [SCENARIOS[k]["yield"] for k in SCENARIOS]

# Single-scenario optimal is not the same as the 3-scenario expected optimum.
# Set to None to skip the reference line on plots.
KNOWN_OPTIMAL = None

# Global constant-cut underestimator (set to a finite value if known).
# Used as fallback global LB when all simplex LBs are invalid.
CONSTANT_CUT = float('-inf')

# Max consecutive iterations with ALL tet LBs invalid before exit
MAX_CONSECUTIVE_ALL_INVALID = 3
# Max fraction of invalid tets before exit
MAX_INVALID_TET_FRAC = 0.8

SOLVER_NAME = "gurobi"
SOLVER_OPTS = {"TimeLimit": 30}

# Options for LB-side Gurobi solves (ms-LP, simplex-LB LP)
LB_GUROBI_OPTIONS = {"MIPGap": 1e-2, "TimeLimit": 30}


def _make_lb_solver():
    """Create a Gurobi solver configured for LB-side LP solves."""
    s = SolverFactory(SOLVER_NAME)
    for k, v in LB_GUROBI_OPTIONS.items():
        s.options[k] = v
    return s


# ---------------------------------------------------------------------------
# Build per-scenario models (for point evaluations)
# ---------------------------------------------------------------------------
def build_farmer_models():
    from farmer_problem import TwoStageFarmer
    scenario_names = list(SCENARIOS.keys())
    model_list = []
    for sname in scenario_names:
        farmer = TwoStageFarmer(SCENARIOS[sname]["yield"])
        m = farmer.model
        m.obj_expr = pyo.Expression(expr=m.obj.expr)
        model_list.append(m)
    return scenario_names, model_list


def make_solver():
    solver = SolverFactory(SOLVER_NAME)
    for k, v in SOLVER_OPTS.items():
        solver.options[k] = v
    return solver


def set_first_stage_x(m, w, c, b):
    m.x["wheat"].fix(w)
    m.x["corn"].fix(c)
    m.x["beets"].fix(b)


def unfix_first_stage_x(m):
    m.x["wheat"].unfix()
    m.x["corn"].unfix()
    m.x["beets"].unfix()


def solve_scenario_model(solver, m):
    solver.solve(m, tee=False)
    return float(pyo.value(m.obj_expr))


# ---------------------------------------------------------------------------
# Evaluate Q_s(x) for all scenarios at a point
# ---------------------------------------------------------------------------
def eval_Qs_at_point(w, c, b, solver, model_list):
    """Return (qs_dict, f_expected).

    qs_dict: {s_idx: Q_s(x)} for s=0..S-1
    f_expected: sum_s p_s * Q_s(x)
    """
    qs = {}
    for s_idx in range(S):
        m = model_list[s_idx]
        set_first_stage_x(m, w, c, b)
        qs[s_idx] = solve_scenario_model(solver, m)
        unfix_first_stage_x(m)
    f_exp = sum(PROBS[s] * qs[s] for s in range(S))
    return qs, f_exp


# ---------------------------------------------------------------------------
# Per-scenario affine surrogate
# ---------------------------------------------------------------------------
def build_affine_surrogate_scenario(tet_verts, nodes, qs_at_node, s_idx):
    """Compute (a_s, g_s) such that a_s + g_s.vi ≈ Q_s(vi) for vertices.

    Returns (a_s, g_s) where a_s is scalar, g_s is np.array(3).

    When the 4 vertices are coplanar (e.g., all on xw+xc+xb=500),
    the 4×4 system is singular. In this case, we project to 2D
    and fit a reduced affine function, then lift back to 3D.
    """
    V = np.array([nodes[vi] for vi in tet_verts])   # 4x3
    f = np.array([qs_at_node[vi][s_idx] for vi in tet_verts])  # 4
    A_mat = np.column_stack([np.ones(4), V])  # 4x4

    # Check condition number to detect (near-)singularity
    cond = np.linalg.cond(A_mat)
    if cond < 1e12:
        # Well-conditioned: standard 3D affine fit
        params = np.linalg.solve(A_mat, f)
        return float(params[0]), params[1:].copy()

    # --- Coplanar / near-singular case ---
    # Project to 2D on the plane containing the 4 points.
    # Use two edge vectors as basis for the 2D coordinate system.
    origin = V[0]
    e1 = V[1] - V[0]
    e1_norm = np.linalg.norm(e1)
    if e1_norm < 1e-14:
        e1 = V[2] - V[0]
        e1_norm = np.linalg.norm(e1)
    e1 = e1 / e1_norm

    # Second basis vector: orthogonal to e1, in the plane
    e2_raw = V[2] - V[0]
    e2_raw = e2_raw - np.dot(e2_raw, e1) * e1
    e2_norm = np.linalg.norm(e2_raw)
    if e2_norm < 1e-14:
        e2_raw = V[3] - V[0]
        e2_raw = e2_raw - np.dot(e2_raw, e1) * e1
        e2_norm = np.linalg.norm(e2_raw)
    e2 = e2_raw / e2_norm

    # Project to 2D (u, v) coordinates
    U = np.array([
        [np.dot(V[i] - origin, e1), np.dot(V[i] - origin, e2)]
        for i in range(4)
    ])  # 4x2

    # Fit A_2d(u,v) = c0 + c1*u + c2*v via least squares (4 eqs, 3 unknowns)
    A2d = np.column_stack([np.ones(4), U])  # 4x3
    params_2d, residuals, _, _ = np.linalg.lstsq(A2d, f, rcond=None)

    # Check interpolation quality
    f_pred = A2d @ params_2d
    max_resid = np.max(np.abs(f_pred - f))
    if max_resid > 1e-2:
        print(f"    [WARN] coplanar surrogate s={s_idx}: "
              f"max residual={max_resid:.4f} (Q not affine on this tet)")

    # Lift back to 3D: A(x) = c0 + c1*(x-origin).e1 + c2*(x-origin).e2
    # = c0 - c1*(origin.e1) - c2*(origin.e2)
    #   + c1*e1[0]*xw + c1*e1[1]*xc + c1*e1[2]*xb
    #   + c2*e2[0]*xw + c2*e2[1]*xc + c2*e2[2]*xb
    c0, c1, c2 = params_2d
    g_3d = c1 * e1 + c2 * e2
    a_3d = c0 - np.dot(g_3d, origin)

    return float(a_3d), g_3d.copy()


# ---------------------------------------------------------------------------
# Per-scenario exact ms LP
# ---------------------------------------------------------------------------
def solve_ms_lp_scenario(tet_verts_3d, a_s, g_s, s_idx,
                         solver_name=SOLVER_NAME):
    """Solve  ms_{Δ,s} = min_{x in Δ} [ Q_s(x) - A_s(x) ].

    Single-scenario EF LP with barycentric tet-membership constraints.
    Returns (ms_value, argmin_point, status):
      - ms_value: float or None
      - argmin_point: tuple (xw*, xc*, xb*) where ms is achieved, or None
      - status: 'optimal' or 'failed'
    """
    yld = YIELDS[s_idx]
    cy_w = 2.5 * yld
    cy_c = 3.0 * yld
    cy_b = 20.0 * yld

    m = pyo.ConcreteModel()

    # First-stage variables (free within tet)
    m.xw = pyo.Var(within=pyo.NonNegativeReals)
    m.xc = pyo.Var(within=pyo.NonNegativeReals)
    m.xb = pyo.Var(within=pyo.NonNegativeReals)

    # Barycentric: x = sum lam_i * v_i,  lam >= 0, sum = 1
    m.lam = pyo.Var(range(4), within=pyo.NonNegativeReals)
    m.lam_sum = pyo.Constraint(
        expr=sum(m.lam[i] for i in range(4)) == 1)
    m.xw_bary = pyo.Constraint(
        expr=m.xw == sum(
            m.lam[i] * tet_verts_3d[i][0] for i in range(4)))
    m.xc_bary = pyo.Constraint(
        expr=m.xc == sum(
            m.lam[i] * tet_verts_3d[i][1] for i in range(4)))
    m.xb_bary = pyo.Constraint(
        expr=m.xb == sum(
            m.lam[i] * tet_verts_3d[i][2] for i in range(4)))

    # Recourse variables for scenario s
    m.ww = pyo.Var(within=pyo.NonNegativeReals)   # sell wheat
    m.wc = pyo.Var(within=pyo.NonNegativeReals)   # sell corn
    m.bf = pyo.Var(within=pyo.NonNegativeReals)   # sell beets favorable
    m.bu = pyo.Var(within=pyo.NonNegativeReals)   # sell beets unfavorable
    m.yw = pyo.Var(within=pyo.NonNegativeReals)   # buy wheat
    m.yc = pyo.Var(within=pyo.NonNegativeReals)   # buy corn

    # Scenario constraints
    m.wheat_req = pyo.Constraint(
        expr=cy_w * m.xw + m.yw - m.ww >= 200)
    m.corn_req = pyo.Constraint(
        expr=cy_c * m.xc + m.yc - m.wc >= 240)
    m.beets_bal = pyo.Constraint(
        expr=m.bf + m.bu <= cy_b * m.xb)
    m.beets_q = pyo.Constraint(expr=m.bf <= 6000)

    # Q_s(x) = planting + purchasing - selling
    q_expr = (150 * m.xw + 230 * m.xc + 260 * m.xb
              - 170 * m.ww - 150 * m.wc
              - 36 * m.bf - 10 * m.bu
              + 238 * m.yw + 210 * m.yc)

    # A_s(x) = a_s + g_s . x
    a_expr = a_s + g_s[0] * m.xw + g_s[1] * m.xc + g_s[2] * m.xb

    # Objective: minimise Q_s(x) - A_s(x)
    m.obj = pyo.Objective(expr=q_expr - a_expr, sense=pyo.minimize)

    slvr = _make_lb_solver()
    results = slvr.solve(m, tee=False)
    tc = results.solver.termination_condition
    if tc == TerminationCondition.optimal:
        ms_val = float(pyo.value(m.obj))
        argmin_pt = (float(pyo.value(m.xw)),
                     float(pyo.value(m.xc)),
                     float(pyo.value(m.xb)))
        return ms_val, argmin_pt, 'optimal'
    else:
        print(f"[LB SOLVE WARN] ms-LP scenario {s_idx} failed: "
              f"termination={tc} (invalid)")
        return None, None, 'failed'


# ---------------------------------------------------------------------------
# Per-tet dual bound: min_{x in Δ} Q_s(x)
# ---------------------------------------------------------------------------
def solve_tet_dual_bound(tet_verts_3d, s_idx, solver_name=SOLVER_NAME):
    """Compute the dual bound (constant cut) for scenario s over a tet.

    Solves:  min_{x in Δ} Q_s(x)
    Returns (dual_bound, status) where status is 'optimal' or 'failed'.
    """
    yld = YIELDS[s_idx]
    cy_w = 2.5 * yld
    cy_c = 3.0 * yld
    cy_b = 20.0 * yld

    m = pyo.ConcreteModel()
    m.xw = pyo.Var(within=pyo.NonNegativeReals)
    m.xc = pyo.Var(within=pyo.NonNegativeReals)
    m.xb = pyo.Var(within=pyo.NonNegativeReals)

    m.lam = pyo.Var(range(4), within=pyo.NonNegativeReals)
    m.lam_sum = pyo.Constraint(
        expr=sum(m.lam[i] for i in range(4)) == 1)
    m.xw_bary = pyo.Constraint(
        expr=m.xw == sum(
            m.lam[i] * tet_verts_3d[i][0] for i in range(4)))
    m.xc_bary = pyo.Constraint(
        expr=m.xc == sum(
            m.lam[i] * tet_verts_3d[i][1] for i in range(4)))
    m.xb_bary = pyo.Constraint(
        expr=m.xb == sum(
            m.lam[i] * tet_verts_3d[i][2] for i in range(4)))

    m.ww = pyo.Var(within=pyo.NonNegativeReals)
    m.wc = pyo.Var(within=pyo.NonNegativeReals)
    m.bf = pyo.Var(within=pyo.NonNegativeReals)
    m.bu = pyo.Var(within=pyo.NonNegativeReals)
    m.yw = pyo.Var(within=pyo.NonNegativeReals)
    m.yc = pyo.Var(within=pyo.NonNegativeReals)

    m.wheat_req = pyo.Constraint(
        expr=cy_w * m.xw + m.yw - m.ww >= 200)
    m.corn_req = pyo.Constraint(
        expr=cy_c * m.xc + m.yc - m.wc >= 240)
    m.beets_bal = pyo.Constraint(
        expr=m.bf + m.bu <= cy_b * m.xb)
    m.beets_q = pyo.Constraint(expr=m.bf <= 6000)

    q_expr = (150 * m.xw + 230 * m.xc + 260 * m.xb
              - 170 * m.ww - 150 * m.wc
              - 36 * m.bf - 10 * m.bu
              + 238 * m.yw + 210 * m.yc)

    m.obj = pyo.Objective(expr=q_expr, sense=pyo.minimize)

    slvr = _make_lb_solver()
    results = slvr.solve(m, tee=False)
    tc = results.solver.termination_condition
    if tc == TerminationCondition.optimal:
        return float(pyo.value(m.obj)), 'optimal'
    else:
        return None, 'failed'


# ---------------------------------------------------------------------------
# Simplex LB LP (with constant cut)
# ---------------------------------------------------------------------------
def solve_simplex_lb_lp(tet_verts_3d, scen_surrogates, scen_ms,
                        constant_cut=float('-inf'),
                        solver_name=SOLVER_NAME):
    """Compute LB_Δ = min_{x in Δ} max(U_Δ(x), C(x)).

    U_Δ(x) = sum_s p_s * (a_s + g_s.x + ms_s)   (affine)
    C(x) = constant_cut                           (scalar)

    Parameters
    ----------
    scen_surrogates : list of (a_s, g_s) for each scenario
    scen_ms : list of ms_s for each scenario

    Returns LB_Δ (float).
    """
    # U_Δ(x) is affine: combine
    a_u = sum(PROBS[s] * (scen_surrogates[s][0] + scen_ms[s])
              for s in range(S))
    g_u = sum(PROBS[s] * scen_surrogates[s][1] for s in range(S))

    # If no constant cut active, LB = min over vertices of U(v)
    if not math.isfinite(constant_cut) or constant_cut <= -1e30:
        vals = []
        for v in tet_verts_3d:
            vals.append(a_u + np.dot(g_u, v))
        return min(vals)

    # With constant cut: solve LP
    m = pyo.ConcreteModel()
    m.xw = pyo.Var(within=pyo.NonNegativeReals)
    m.xc = pyo.Var(within=pyo.NonNegativeReals)
    m.xb = pyo.Var(within=pyo.NonNegativeReals)
    m.lam = pyo.Var(range(4), within=pyo.NonNegativeReals)
    m.lam_sum = pyo.Constraint(
        expr=sum(m.lam[i] for i in range(4)) == 1)
    m.xw_bary = pyo.Constraint(
        expr=m.xw == sum(
            m.lam[i] * tet_verts_3d[i][0] for i in range(4)))
    m.xc_bary = pyo.Constraint(
        expr=m.xc == sum(
            m.lam[i] * tet_verts_3d[i][1] for i in range(4)))
    m.xb_bary = pyo.Constraint(
        expr=m.xb == sum(
            m.lam[i] * tet_verts_3d[i][2] for i in range(4)))
    m.t = pyo.Var()
    u_expr = a_u + g_u[0] * m.xw + g_u[1] * m.xc + g_u[2] * m.xb
    m.cut_u = pyo.Constraint(expr=m.t >= u_expr)
    m.cut_c = pyo.Constraint(expr=m.t >= constant_cut)
    m.obj = pyo.Objective(expr=m.t, sense=pyo.minimize)
    slvr = _make_lb_solver()
    results = slvr.solve(m, tee=False)
    tc = results.solver.termination_condition
    if tc == TerminationCondition.optimal:
        return float(pyo.value(m.t))
    else:
        print(f"[LB SOLVE WARN] simplex-LB LP failed: "
              f"termination={tc} (invalid)")
        return None


# ---------------------------------------------------------------------------
# Candidate point generation within a tet
# ---------------------------------------------------------------------------
def _tet_candidates(tet_verts, nodes):
    """Generate candidate points for insertion within a tet.

    Returns list of (point_3d, point_type, extra_info):
      point_type: 'centroid', 'edge_mid', 'face_centroid'
      extra_info: edge (u,v) for edge_mid, face tuple for face_centroid
    """
    vi = [np.array(nodes[v]) for v in tet_verts]
    idxs = list(tet_verts)
    candidates = []

    # 1) Centroid
    centroid = sum(vi) / 4.0
    candidates.append((tuple(centroid), 'centroid', None))

    # 2) 6 edge midpoints
    for a, b in combinations(range(4), 2):
        mid = 0.5 * (vi[a] + vi[b])
        edge = (idxs[a], idxs[b])
        candidates.append((tuple(mid), 'edge_mid', edge))

    # 3) 4 face centroids
    for face_local in combinations(range(4), 3):
        fc = sum(vi[j] for j in face_local) / 3.0
        face_ids = tuple(idxs[j] for j in face_local)
        candidates.append((tuple(fc), 'face_centroid', face_ids))

    return candidates


# ---------------------------------------------------------------------------
# Affine surrogate from raw coordinate/value arrays
# ---------------------------------------------------------------------------
def _build_surrogate_from_values(vert_coords, q_values):
    """Build affine surrogate from 4 vertex coordinates and 4 Q values.

    Returns (a, g) where a is scalar, g is np.array(3).
    """
    V = np.array(vert_coords)
    f = np.array(q_values)
    A_mat = np.column_stack([np.ones(4), V])
    try:
        params = np.linalg.solve(A_mat, f)
    except np.linalg.LinAlgError:
        params, _, _, _ = np.linalg.lstsq(A_mat, f, rcond=None)
    return float(params[0]), params[1:].copy()


# ---------------------------------------------------------------------------
# Hypothetical child tets after candidate insertion
# ---------------------------------------------------------------------------
def _hypothetical_children(tet_verts, cand_type, cand_info):
    """Return child tet vertex lists for a hypothetical split.

    Each child is a list of 4 elements: either an int (existing node index)
    or the string 'new' (the candidate point).
    """
    a, b, c, d = tet_verts
    if cand_type == 'edge_mid':
        u, v = cand_info
        others = [x for x in tet_verts if x != u and x != v]
        return [
            [u, 'new'] + others,
            [v, 'new'] + others,
        ]
    elif cand_type == 'face_centroid':
        face_set = set(cand_info)
        dd = (set(tet_verts) - face_set).pop()
        fa, fb, fc = list(cand_info)
        return [
            ['new', fa, fb, dd],
            ['new', fb, fc, dd],
            ['new', fc, fa, dd],
        ]
    else:  # centroid
        return [
            ['new', b, c, d],
            [a, 'new', c, d],
            [a, b, 'new', d],
            [a, b, c, 'new'],
        ]


# ---------------------------------------------------------------------------
# 2-stage insertion point selection (min shared ms)
# ---------------------------------------------------------------------------
CANDIDATE_K = 3  # Stage-1 keeps top K candidates for Stage-2


def select_insertion_point(tet_verts, nodes, scen_surrogates, scen_ms,
                           qs_at_node, solver, model_list):
    """Select insertion point by minimising hypothetical-split ms.

    Stage 1 (cheap proxy): evaluate Q_s(x_cand), compute
       proxy(x) = min_s [Q_s(x) - A_{parent,s}(x)].
       Keep top K most negative.
    Stage 2 (exact): for each kept candidate, hypothetically split tet,
       build child-tet surrogates, solve per-scenario ms-LPs, compute
       ms(x) = min_child min_s ms_{child,s}.
       Pick candidate with smallest ms(x).

    Returns (best_cand, best_ms) where
       best_cand = (pt, type, info, qs_dict)
       best_ms = float (the winning shared ms).
    """
    candidates = _tet_candidates(tet_verts, nodes)

    # ── Stage 1: cheap proxy filter ──────────────────────────────────
    stage1 = []
    for pt, ptype, info in candidates:
        w, c, b = pt
        if w < -1e-8 or c < -1e-8 or b < -1e-8:
            continue
        if w + c + b > TOTAL + 1e-6:
            continue

        qs, _ = eval_Qs_at_point(w, c, b, solver, model_list)
        x_arr = np.array(pt)
        proxy = min(
            qs[s] - (scen_surrogates[s][0]
                     + np.dot(scen_surrogates[s][1], x_arr))
            for s in range(S))
        stage1.append((proxy, pt, ptype, info, qs))

    if not stage1:
        return None, float('inf')

    # Sort by proxy (most negative first), keep top K
    stage1.sort(key=lambda x: x[0])
    kept = stage1[:CANDIDATE_K]

    # ── Stage 2: exact hypothetical-split ms ─────────────────────────
    best_ms = float('inf')
    best_cand = None

    for proxy_val, pt, ptype, info, qs_cand in kept:
        children = _hypothetical_children(tet_verts, ptype, info)
        cand_ms = float('inf')
        cand_valid = True

        for child in children:
            child_coords = []
            for v in child:
                if v == 'new':
                    child_coords.append(pt)
                else:
                    child_coords.append(nodes[v])

            # Skip degenerate children (repeated vertices)
            if len(set(tuple(c) for c in child_coords)) < 4:
                continue

            ms_child_shared = float('inf')
            for s in range(S):
                child_qs = []
                for v in child:
                    if v == 'new':
                        child_qs.append(qs_cand[s])
                    else:
                        child_qs.append(qs_at_node[v][s])

                a_cs, g_cs = _build_surrogate_from_values(
                    child_coords, child_qs)
                ms_cs, status = solve_ms_lp_scenario(
                    child_coords, a_cs, g_cs, s)

                if status != 'optimal':
                    cand_valid = False
                    break

                ms_child_shared = min(ms_child_shared, ms_cs)

            if not cand_valid:
                break

            cand_ms = min(cand_ms, ms_child_shared)

        if not cand_valid:
            continue

        if cand_ms < best_ms:
            best_ms = cand_ms
            best_cand = (pt, ptype, info, qs_cand)

    # Fallback: if all Stage-2 failed, use Stage-1 best
    if best_cand is None and stage1:
        print("    [SELECT WARN] Stage-2 exact ms evaluation failed for "
              "all candidates; falling back to proxy selection")
        _, pt, ptype, info, qs_cand = stage1[0]
        best_cand = (pt, ptype, info, qs_cand)
        best_ms = float('inf')

    return best_cand, best_ms


# ---------------------------------------------------------------------------
# Mesh operations
# ---------------------------------------------------------------------------
def _edge_key(u, v):
    return (min(u, v), max(u, v))


def _classify_point_in_tet(pt, tet_verts, nodes, tol=1e-6):
    """Classify a point relative to a tet using barycentric coordinates.

    Returns (location, face_or_edge, projected_pt):
      location: 'interior', 'face', 'edge', 'vertex'
      face_or_edge:
        - for 'face': tuple of 3 vertex indices forming the face
        - for 'edge': tuple of 2 vertex indices forming the edge
        - for 'vertex': single vertex index
        - for 'interior': None
      projected_pt: the point snapped to the detected feature
    """
    # Build the barycentric coordinate system
    # x = lam0*v0 + lam1*v1 + lam2*v2 + lam3*v3, sum(lam) = 1
    v = [np.array(nodes[vi]) for vi in tet_verts]
    p = np.array(pt)

    # Solve via the standard method: T * lam[1:4] = p - v0
    T = np.column_stack([v[1] - v[0], v[2] - v[0], v[3] - v[0]])
    try:
        lam_123 = np.linalg.solve(T, p - v[0])
    except np.linalg.LinAlgError:
        # Degenerate tet — use lstsq
        lam_123, _, _, _ = np.linalg.lstsq(T, p - v[0], rcond=None)
    lam = np.array([1.0 - lam_123.sum(), lam_123[0], lam_123[1], lam_123[2]])

    # Count how many barycentric coords are ~0
    near_zero = [i for i in range(4) if abs(lam[i]) < tol]

    if len(near_zero) >= 3:
        # On a vertex
        non_zero = [i for i in range(4) if i not in near_zero]
        vi = non_zero[0] if non_zero else 0
        return 'vertex', tet_verts[vi], tuple(nodes[tet_verts[vi]])

    elif len(near_zero) == 2:
        # On an edge (2 bary coords are ~0 => point is on the edge
        # connecting the 2 vertices with non-zero bary coords)
        on_edge = [i for i in range(4) if i not in near_zero]
        edge_vis = (tet_verts[on_edge[0]], tet_verts[on_edge[1]])
        # Project: snap the zero coords to exactly 0
        lam_proj = lam.copy()
        for i in near_zero:
            lam_proj[i] = 0.0
        lam_proj /= lam_proj.sum()
        proj_pt = sum(lam_proj[i] * v[i] for i in range(4))
        return 'edge', edge_vis, tuple(proj_pt.tolist())

    elif len(near_zero) == 1:
        # On a face (1 bary coord ~0 => point is on the face
        # opposite the vertex with zero bary coord)
        face_vis = tuple(tet_verts[i] for i in range(4) if i not in near_zero)
        # Project: snap the zero coord to exactly 0
        lam_proj = lam.copy()
        lam_proj[near_zero[0]] = 0.0
        lam_proj /= lam_proj.sum()
        proj_pt = sum(lam_proj[i] * v[i] for i in range(4))
        return 'face', face_vis, tuple(proj_pt.tolist())

    else:
        # Interior
        return 'interior', None, pt


def _tet_longest_edge(nodes, tet):
    best_d2, best_e = -1.0, (tet[0], tet[1])
    for a, b in combinations(tet, 2):
        d2 = np.sum((np.array(nodes[a]) - np.array(nodes[b])) ** 2)
        if d2 > best_d2:
            best_d2, best_e = d2, (a, b)
    return best_e


def split_edge_consistent(nodes, tets, edge, edge_mid_cache):
    """Mesh-consistent edge bisection.
    Insert midpoint of edge (u,v) and split ALL tets sharing it.
    Returns (new_tets, mid_idx).
    """
    u, v = edge
    ekey = _edge_key(u, v)
    if ekey in edge_mid_cache:
        mid_idx = edge_mid_cache[ekey]
    else:
        mid = 0.5 * (np.asarray(nodes[u]) + np.asarray(nodes[v]))
        mid_idx = len(nodes)
        nodes.append(tuple(mid.tolist()))
        edge_mid_cache[ekey] = mid_idx

    new_tets = []
    for t in tets:
        if u in t and v in t:
            others = [x for x in t if x != u and x != v]
            new_tets.append(tuple([u, mid_idx] + others))
            new_tets.append(tuple([v, mid_idx] + others))
        else:
            new_tets.append(t)
    return new_tets, mid_idx


def split_interior_point(nodes, tets, sel_tet_idx, new_node_idx):
    """Split a tet by inserting an interior point, creating 4 sub-tets.
    Only the selected tet is split.
    (a,b,c,d) -> (p,b,c,d), (a,p,c,d), (a,b,p,d), (a,b,c,p)
    """
    a, b, c, d = tets[sel_tet_idx]
    p = new_node_idx
    new_sub = [
        (p, b, c, d),
        (a, p, c, d),
        (a, b, p, d),
        (a, b, c, p),
    ]
    new_tets = []
    for ti, t in enumerate(tets):
        if ti == sel_tet_idx:
            new_tets.extend(new_sub)
        else:
            new_tets.append(t)
    return new_tets


def split_face_point(nodes, tets, face_ids, new_node_idx):
    """Split all tets sharing the given face by inserting a face centroid.
    Each tet (a,b,c,d) containing face (a,b,c) -> 3 sub-tets.
    """
    face_set = set(face_ids)
    new_tets = []
    p = new_node_idx
    for t in tets:
        t_set = set(t)
        if face_set.issubset(t_set):
            # d is the vertex not on the face
            d = (t_set - face_set).pop()
            fa, fb, fc = list(face_ids)
            new_tets.append((p, fa, fb, d))
            new_tets.append((p, fb, fc, d))
            new_tets.append((p, fc, fa, d))
        else:
            new_tets.append(t)
    return new_tets


def _validate_mesh(nodes, tets, total=TOTAL):
    for ti, t in enumerate(tets):
        if len(set(t)) != 4:
            print(f"  [MESH ERR] tet {ti} repeated verts: {t}")
            return False
        for vi in t:
            w, c, b = nodes[vi]
            if w < -1e-8 or c < -1e-8 or b < -1e-8:
                return False
            if w + c + b > total + 1e-6:
                return False
    return True


# ---------------------------------------------------------------------------
# ms validation
# ---------------------------------------------------------------------------
def validate_ms_at_vertices(tet_verts, nodes, qs_at_node, s_idx,
                            a_s, g_s, ms_s, tol=1e-4):
    """Check A_s(vi) + ms_s <= Q_s(vi) + tol for all vertices."""
    ok = True
    for vi in tet_verts:
        v = np.array(nodes[vi])
        u_val = a_s + np.dot(g_s, v) + ms_s
        q_val = qs_at_node[vi][s_idx]
        if u_val > q_val + tol:
            print(f"    [MS WARN] s={s_idx} v{vi}: "
                  f"A+ms={u_val:.4f} > Q={q_val:.4f} by {u_val-q_val:.2e}")
            ok = False
    return ok


# ---------------------------------------------------------------------------
# Kink planes (3D visualisation)
# ---------------------------------------------------------------------------
_KINK_YIELDS = (1.2, 1.0, 0.8)
_KINK_COLORS = {"wheat": "#e67e22", "corn": "#27ae60", "beets": "#8e44ad"}


def _kink_plane_triangles(total=TOTAL, ys=_KINK_YIELDS):
    result = {"wheat": [], "corn": [], "beets": []}
    for y in ys:
        tw = 80.0 / y
        tb = 300.0 / y
        if tw <= total:
            rem = total - tw
            result["wheat"].append(
                (np.array([[tw,0,0],[tw,rem,0],[tw,0,rem]]), f"xw={tw:.1f}"))
        if tw <= total:
            rem = total - tw
            result["corn"].append(
                (np.array([[0,tw,0],[rem,tw,0],[0,tw,rem]]), f"xc={tw:.1f}"))
        if tb <= total:
            rem = total - tb
            result["beets"].append(
                (np.array([[0,0,tb],[rem,0,tb],[0,rem,tb]]), f"xb={tb:.1f}"))
    return result


# ---------------------------------------------------------------------------
# 3D Plotly iteration plot
# ---------------------------------------------------------------------------
def _render_iteration_plot(it, nodes, tets, sel_idx, best_idx,
                           new_pt, true_ub, global_lb,
                           f_at_node, tet_lb_map,
                           *, show_inline, save_plots, save_html, plot_dir):
    fig = go.Figure()

    # Feasible tetrahedron wireframe
    tv = [(0,0,0),(TOTAL,0,0),(0,TOTAL,0),(0,0,TOTAL)]
    te = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]
    tx, ty, tz = [], [], []
    for i, j in te:
        tx += [tv[i][0], tv[j][0], None]
        ty += [tv[i][1], tv[j][1], None]
        tz += [tv[i][2], tv[j][2], None]
    fig.add_trace(go.Scatter3d(
        x=tx, y=ty, z=tz, mode='lines',
        line=dict(color='rgba(180,180,180,0.5)', width=2, dash='dot'),
        name='feasible tet', hoverinfo='skip'))

    # Kink planes
    for crop, planes in _kink_plane_triangles().items():
        for verts, _ in planes:
            fig.add_trace(go.Mesh3d(
                x=verts[:,0], y=verts[:,1], z=verts[:,2],
                i=[0], j=[1], k=[2],
                color=_KINK_COLORS[crop], opacity=0.08,
                showlegend=False, hoverinfo='skip'))
    for crop, color in _KINK_COLORS.items():
        fig.add_trace(go.Scatter3d(
            x=[None], y=[None], z=[None], mode='lines',
            line=dict(color=color, width=3), name=f'{crop} kinks'))

    # Mesh edges
    edges_set = set()
    for t in tets:
        for a, b in combinations(t, 2):
            edges_set.add(_edge_key(a, b))
    ex, ey, ez = [], [], []
    for a, b in edges_set:
        pa, pb = nodes[a], nodes[b]
        ex += [pa[0], pb[0], None]
        ey += [pa[1], pb[1], None]
        ez += [pa[2], pb[2], None]
    fig.add_trace(go.Scatter3d(
        x=ex, y=ey, z=ez, mode='lines',
        line=dict(color='steelblue', width=2),
        name='mesh edges', hoverinfo='skip'))

    # Selected tet
    if sel_idx is not None and sel_idx < len(tets):
        st = tets[sel_idx]
        sx, sy, sz = [], [], []
        for a, b in combinations(st, 2):
            pa, pb = nodes[a], nodes[b]
            sx += [pa[0], pb[0], None]
            sy += [pa[1], pb[1], None]
            sz += [pa[2], pb[2], None]
        fig.add_trace(go.Scatter3d(
            x=sx, y=sy, z=sz, mode='lines',
            line=dict(color='red', width=5), name='selected tet'))

    # Evaluated nodes
    eval_idxs = sorted(f_at_node.keys())
    nx = [nodes[i][0] for i in eval_idxs]
    ny = [nodes[i][1] for i in eval_idxs]
    nz = [nodes[i][2] for i in eval_idxs]
    nvals = [f_at_node[i] for i in eval_idxs]
    fig.add_trace(go.Scatter3d(
        x=nx, y=ny, z=nz, mode='markers',
        marker=dict(size=5, color=nvals, colorscale='Viridis',
                    colorbar=dict(title='F(x)', x=1.05), showscale=True),
        text=[f'n{i}: F={f_at_node[i]:.0f}' for i in eval_idxs],
        name=f'nodes ({len(eval_idxs)})', hoverinfo='text'))

    # Best node
    bp = nodes[best_idx]
    fig.add_trace(go.Scatter3d(
        x=[bp[0]], y=[bp[1]], z=[bp[2]], mode='markers',
        marker=dict(size=10, color='red', symbol='diamond'),
        name='best node'))

    # New point
    if new_pt is not None:
        fig.add_trace(go.Scatter3d(
            x=[new_pt[0]], y=[new_pt[1]], z=[new_pt[2]], mode='markers',
            marker=dict(size=10, color='lime',
                        line=dict(color='darkgreen', width=2)),
            name='new point'))

    # Per-tet LB text
    if len(tets) <= 30:
        for ti, t in enumerate(tets):
            lb_val = tet_lb_map.get(ti, float('inf'))
            cx = sum(nodes[t[j]][0] for j in range(4)) / 4
            cy = sum(nodes[t[j]][1] for j in range(4)) / 4
            cz = sum(nodes[t[j]][2] for j in range(4)) / 4
            fig.add_trace(go.Scatter3d(
                x=[cx], y=[cy], z=[cz], mode='text',
                text=[f'{lb_val:.0f}'],
                textfont=dict(size=8, color='gray'),
                showlegend=False, hoverinfo='skip'))

    gap_str = (f'{true_ub - global_lb:.1f}'
               if math.isfinite(global_lb) else 'inf')
    fig.update_layout(
        title=dict(
            text=(f'Iter {it} | {len(f_at_node)} nodes  {len(tets)} tets | '
                  f'UB={true_ub:.1f}  LB={global_lb:.1f}  gap={gap_str}'),
            x=0.5),
        scene=dict(
            xaxis=dict(title='wheat', range=[-10, TOTAL+10]),
            yaxis=dict(title='corn',  range=[-10, TOTAL+10]),
            zaxis=dict(title='beets', range=[-10, TOTAL+10]),
            aspectmode='data',
            camera=dict(eye=dict(x=1.6, y=1.6, z=1.0)),
            bgcolor='white'),
        legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.7)'),
        margin=dict(l=0, r=0, t=50, b=0), width=900, height=750)

    if save_html:
        html_path = plot_dir / f'iter_{it:03d}.html'
        fig.write_html(str(html_path), include_plotlyjs='cdn',
                       full_html=True, auto_open=False)
        print(f"    [HTML saved] {html_path}")
    if show_inline:
        fig.show()
    if save_plots:
        fig.write_image(str(plot_dir / f'iter_{it:03d}.png'),
                        width=900, height=750, scale=2)


# ---------------------------------------------------------------------------
# Convergence plot
# ---------------------------------------------------------------------------
def _render_convergence_plot(ub_hist, lb_hist, node_hist,
                             *, show_inline, save_plots, plot_dir):
    if not ub_hist:
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 4.5))
    ax1.plot(node_hist, ub_hist, "ro-", ms=4, label="UB (best F)")
    ax1.plot(node_hist, lb_hist, "bs-", ms=4,
             label="Simplex LB (As+ms)")
    if KNOWN_OPTIMAL is not None:
        ax1.axhline(KNOWN_OPTIMAL, color="green", ls="--", lw=1.5,
                    label=f"F*={KNOWN_OPTIMAL:.0f}")
    ax1.set_xlabel("# Nodes"); ax1.set_ylabel("Objective")
    ax1.set_title("Convergence"); ax1.legend(fontsize=8); ax1.grid(alpha=0.3)

    gaps = [ub - lb for ub, lb in zip(ub_hist, lb_hist)]
    ax2.semilogy(node_hist, [max(g, 1e-16) for g in gaps], "mo-", ms=4)
    ax2.set_xlabel("# Nodes"); ax2.set_ylabel("UB - LB")
    ax2.set_title("UB-LB Gap (Simplex)"); ax2.grid(alpha=0.3)

    plt.tight_layout()
    if save_plots:
        fig.savefig(str(plot_dir / "convergence.png"),
                    dpi=150, bbox_inches='tight')
    plt.show(); plt.close(fig)


# ---------------------------------------------------------------------------
# Core callable
# ---------------------------------------------------------------------------
def run_visual_simplex(max_iters=50,
                       target_nodes=50,
                       show_inline=False,
                       save_plots=False,
                       save_html=True,
                       plot_every=1,
                       plot_dir=None,
                       pause=0.0,
                       gap_tol=None,
                       time_limit=600.0,
                       constant_cut=float('-inf')):
    """Run 3D simplex refinement with exact per-scenario ms-LP LB."""
    global CONSTANT_CUT
    if math.isfinite(constant_cut):
        CONSTANT_CUT = constant_cut

    if plot_dir is None:
        plot_dir = Path.cwd() / "farmer_single_scenario_plots"
    else:
        plot_dir = Path(plot_dir)
    if save_plots or save_html:
        if plot_dir.exists():
            shutil.rmtree(plot_dir)
        plot_dir.mkdir(parents=True, exist_ok=True)

    try:
        script_dir = Path(__file__).resolve().parent
    except NameError:
        script_dir = Path.cwd()
    if str(script_dir) not in sys.path:
        sys.path.insert(0, str(script_dir))

    print("=" * 60)
    print("  Farmer 3D Simplex — Single Scenario (good, y=1.2)")
    print("=" * 60)

    print("\n[1] Building scenario models ...")
    _, model_list = build_farmer_models()
    print(f"    {S} scenarios")

    print("\n[2] Creating solver ...")
    solver = make_solver()

    # -- Nodes --
    nodes = [
        (0.0, 0.0, 0.0),
        (TOTAL, 0.0, 0.0),
        (0.0, TOTAL, 0.0),
        (0.0, 0.0, TOTAL),
    ]
    tets = [(0, 1, 2, 3)]
    edge_mid_cache = {}
    qs_at_node = {}   # {node_id: {s: Q_s}}
    f_at_node = {}    # {node_id: expected F}

    # ms cache: key = frozenset(tet verts) -> {s: (a_s, g_s, ms_s)}
    ms_cache = {}

    print(f"    LB solver options: {LB_GUROBI_OPTIONS}")

    print("\n[3] Evaluating initial vertices ...")
    for i in range(len(nodes)):
        w, c, b = nodes[i]
        qs, f_exp = eval_Qs_at_point(w, c, b, solver, model_list)
        qs_at_node[i] = qs
        f_at_node[i] = f_exp
        print(f"    node {i}: ({w:.0f},{c:.0f},{b:.0f})  F={f_exp:.2f}  "
              f"Q=[{', '.join(f'{qs[s]:.1f}' for s in range(S))}]")

    # -- Compute ms for initial tet --
    def _compute_tet_ms(tet_tuple):
        """Compute per-scenario surrogates + ms for a tet.

        Returns dict {s: (a_s, g_s, ms_s, argmin_pt, status)}.
        status is 'optimal' or 'failed'.
        """
        key = frozenset(tet_tuple)
        if key in ms_cache:
            return ms_cache[key]
        if not all(vi in qs_at_node for vi in tet_tuple):
            return None
        result = {}
        verts_3d = [nodes[vi] for vi in tet_tuple]
        for s in range(S):
            a_s, g_s = build_affine_surrogate_scenario(
                tet_tuple, nodes, qs_at_node, s)
            ms_s, argmin_pt, status = solve_ms_lp_scenario(
                verts_3d, a_s, g_s, s)
            if status == 'optimal':
                validate_ms_at_vertices(
                    tet_tuple, nodes, qs_at_node, s, a_s, g_s, ms_s)
            result[s] = (a_s, g_s, ms_s, argmin_pt, status)
        ms_cache[key] = result
        return result

    print("\n[4] Computing ms for initial tet ...")
    init_ms = _compute_tet_ms(tets[0])
    for s in range(S):
        a_s, g_s, ms_s, argmin_pt, st = init_ms[s]
        ms_str = f"{ms_s:.4f}" if ms_s is not None else "None"
        arg_str = (f"({argmin_pt[0]:.1f},{argmin_pt[1]:.1f},{argmin_pt[2]:.1f})"
                   if argmin_pt else "None")
        print(f"    scenario {s}: ms={ms_str}  argmin={arg_str}  status={st}")

    # -- Helpers --
    def _should_plot(iteration):
        return plot_every > 0 and iteration % plot_every == 0

    def _do_plot(iteration, sel_idx_, new_pt_, true_ub_, global_lb_,
                 tet_lb_map_):
        _render_iteration_plot(
            iteration, nodes, tets, sel_idx_, best_idx, new_pt_,
            true_ub_, global_lb_, f_at_node, tet_lb_map_,
            show_inline=show_inline, save_plots=save_plots,
            save_html=save_html, plot_dir=plot_dir)
        if pause > 0:
            _time.sleep(pause)

    # -- Main loop --
    ub_hist, lb_hist, node_hist = [], [], []
    selection_hist = []  # per-iter: (tet_idx, ins_type, ins_pt, shared_ms)
    consecutive_all_invalid = 0   # exit counter
    t_start = perf_counter()
    it = 0

    print(f"\n{'=' * 60}")
    print(f"  Running (target={target_nodes}, max_iters={max_iters}, "
          f"gap_tol={gap_tol})")
    print(f"  CONSTANT_CUT = {CONSTANT_CUT}")
    print(f"{'=' * 60}\n")

    # Open iteration log file
    log_path = plot_dir / "iteration_details.txt"
    log_file = open(str(log_path), 'w', encoding='utf-8')
    log_file.write("Simplex Iteration Log\n")
    log_file.write("=" * 80 + "\n\n")

    while len(f_at_node) < target_nodes and it < max_iters:
        N = len(f_at_node)
        N_tet = len(tets)

        # -- UB --
        true_ub = min(f_at_node.values())
        best_idx = min(f_at_node, key=f_at_node.get)

        # -- Compute per-tet ms + LB (valid/invalid separation) --
        tet_lb_map = {}
        tet_ub_map = {}
        tet_gap_map = {}
        tet_ms_data = {}   # ti -> {s: (a_s, g_s, ms_s, argmin_pt, status)}
        valid_tet_lbs = []
        n_invalid = 0

        for ti, t in enumerate(tets):
            if not all(vi in qs_at_node for vi in t):
                tet_lb_map[ti] = float('-inf')
                tet_ub_map[ti] = float('inf')
                tet_gap_map[ti] = float('inf')
                n_invalid += 1
                if it < 20:
                    missing = [vi for vi in t if vi not in qs_at_node]
                    print(f"    [DEBUG] tet {ti} {list(t)}: "
                          f"missing Q at nodes {missing}")
                continue

            ms_data = _compute_tet_ms(t)
            tet_ms_data[ti] = ms_data

            # If any scenario ms failed, mark tet as invalid
            any_failed = any(ms_data[s][4] != 'optimal' for s in range(S))
            if any_failed:
                tet_lb_map[ti] = float('-inf')
                ub = min(f_at_node[vi] for vi in t)
                tet_ub_map[ti] = ub
                tet_gap_map[ti] = float('inf')
                n_invalid += 1
                if it < 20:
                    for s in range(S):
                        print(f"    [DEBUG] tet {ti} {list(t)}: "
                              f"ms s={s} status={ms_data[s][4]} "
                              f"ms_val={ms_data[s][2]}")
                continue

            # Build surrogate params for LB LP
            scen_surr = [(ms_data[s][0], ms_data[s][1]) for s in range(S)]
            scen_ms_vals = [ms_data[s][2] for s in range(S)]
            verts_3d = [nodes[vi] for vi in t]

            lb = solve_simplex_lb_lp(verts_3d, scen_surr, scen_ms_vals,
                                     CONSTANT_CUT)
            if lb is None:
                # simplex-LB LP itself failed
                tet_lb_map[ti] = float('-inf')
                ub = min(f_at_node[vi] for vi in t)
                tet_ub_map[ti] = ub
                tet_gap_map[ti] = float('inf')
                n_invalid += 1
                if it < 20:
                    print(f"    [DEBUG] tet {ti} {list(t)}: LB LP failed")
                continue

            ub = min(f_at_node[vi] for vi in t)
            tet_lb_map[ti] = lb
            tet_ub_map[ti] = ub
            tet_gap_map[ti] = ub - lb
            valid_tet_lbs.append(lb)

        # -- Compute global LB (only from valid tets + constant cut) --
        if valid_tet_lbs:
            min_valid_lb = min(valid_tet_lbs)
            if math.isfinite(CONSTANT_CUT):
                global_lb = max(min_valid_lb, CONSTANT_CUT)
            else:
                global_lb = min_valid_lb
            consecutive_all_invalid = 0
        elif math.isfinite(CONSTANT_CUT):
            global_lb = CONSTANT_CUT
            consecutive_all_invalid += 1
        else:
            global_lb = float('-inf')
            consecutive_all_invalid += 1

        gap = true_ub - global_lb

        ub_hist.append(true_ub)
        lb_hist.append(global_lb)
        node_hist.append(N)

        print(f"[Iter {it:3d}] nodes={N}  tets={N_tet}  "
              f"UB={true_ub:.2f}  LB={global_lb:.2f}  gap={gap:.2f}  "
              f"invalid_tets={n_invalid}/{N_tet}")

        if gap < -1e-4:
            print(f"  [WARN] Negative gap: {gap:.4e}")

        if N_tet <= 10:
            for ti, t in enumerate(tets):
                ms_str = ""
                if ti in tet_ms_data:
                    ms_vals = [tet_ms_data[ti][s][2] for s in range(S)]
                    statuses = [tet_ms_data[ti][s][4] for s in range(S)]
                    ms_fmts = [f'{v:.1f}' if v is not None else 'None'
                               for v in ms_vals]
                    ms_str = (f"  ms=[{', '.join(ms_fmts)}]"
                              f"  st={statuses}")
                print(f"    t{ti} {list(t)}  "
                      f"LB={tet_lb_map[ti]:.1f}  "
                      f"UB={tet_ub_map[ti]:.1f}  "
                      f"gap={tet_gap_map[ti]:.1f}{ms_str}")

        # -- Stop? --
        if gap_tol is not None and gap <= gap_tol:
            if _should_plot(it):
                _do_plot(it, None, None, true_ub, global_lb, tet_lb_map)
            print(f"  STOP: gap converged ({gap:.4e} <= {gap_tol}).")
            break
        if time_limit and (perf_counter() - t_start) >= time_limit:
            if _should_plot(it):
                _do_plot(it, None, None, true_ub, global_lb, tet_lb_map)
            print("  STOP: time limit.")
            break

        # -- Exit on failure domination --
        if consecutive_all_invalid >= MAX_CONSECUTIVE_ALL_INVALID:
            print(f"  STOP: all tet LBs invalid for "
                  f"{consecutive_all_invalid} consecutive iterations.")
            break
        if N_tet > 0 and (n_invalid / N_tet) > MAX_INVALID_TET_FRAC:
            print(f"  STOP: invalid tet fraction "
                  f"{n_invalid}/{N_tet} = {n_invalid/N_tet:.1%} "
                  f"> {MAX_INVALID_TET_FRAC:.0%} threshold.")
            break

        # -- Select tet with smallest LB (min simplex LB) --
        valid_tis = [ti for ti in range(N_tet)
                     if math.isfinite(tet_lb_map.get(ti, float('-inf')))
                     and tet_lb_map.get(ti, float('-inf')) > -1e30]
        if not valid_tis:
            print("  STOP: no tet with a valid LB; cannot select.")
            break
        sel_idx = min(valid_tis, key=lambda ti: tet_lb_map[ti])
        sel_tet = tets[sel_idx]
        print(f"    -> selected tet {sel_idx}  {list(sel_tet)}  "
              f"LB={tet_lb_map[sel_idx]:.4f}")

        # -- Select insertion point: ms-LP argmin --
        # Use the argmin x* from the ms-LP of the worst scenario
        # (in single-scenario mode, there is only s=0).
        # Fallback to centroid if x* is too close to ANY existing node.
        CLOSENESS_TOL = 1e-6  # absolute tolerance for node deduplication

        def _find_existing_node(pt, tol=CLOSENESS_TOL):
            """Return index of existing node at pt, or None."""
            pt_arr = np.array(pt)
            for ni in range(len(nodes)):
                if np.linalg.norm(pt_arr - np.array(nodes[ni])) < tol:
                    return ni
            return None

        sel_ms_data = tet_ms_data.get(sel_idx)
        ins_pt = None
        ins_type = 'centroid'
        ins_ms_val = float('inf')

        if sel_ms_data is not None:
            # Find scenario with most negative ms (worst approximation)
            worst_s = min(range(S),
                          key=lambda s: (sel_ms_data[s][2]
                                         if sel_ms_data[s][2] is not None
                                         else float('inf')))
            a_s, g_s, ms_s, argmin_pt, status = sel_ms_data[worst_s]
            ins_ms_val = ms_s if ms_s is not None else float('inf')

            if argmin_pt is not None and status == 'optimal':
                # Check if argmin is at one of the tet's own 4 vertices
                argmin_arr = np.array(argmin_pt)
                at_tet_vertex = None
                for vi in sel_tet:
                    if np.linalg.norm(argmin_arr - np.array(nodes[vi])) < 1e-6:
                        at_tet_vertex = vi
                        break

                if at_tet_vertex is not None:
                    print(f"    [INFO] ms argmin is AT tet vertex "
                          f"node {at_tet_vertex} "
                          f"({nodes[at_tet_vertex][0]:.1f},"
                          f"{nodes[at_tet_vertex][1]:.1f},"
                          f"{nodes[at_tet_vertex][2]:.1f}); "
                          f"ms should be ~0 but is {ms_s:.4f}")
                    # Tet is fully resolved at this vertex — skip
                else:
                    existing = _find_existing_node(argmin_pt)
                    if existing is None:
                        ins_pt = argmin_pt
                        ins_type = 'ms_argmin'
                    else:
                        # Key fix: the argmin is at an existing mesh node
                        # that's NOT a vertex of this tet.  Classify it
                        # within the tet: if it's on an edge or face,
                        # we can split using this existing node directly.
                        loc_ex, feat_ex, _ = _classify_point_in_tet(
                            nodes[existing], sel_tet, nodes, tol=1e-4)
                        print(f"    [INFO] ms argmin at existing node "
                              f"{existing} "
                              f"({nodes[existing][0]:.1f},"
                              f"{nodes[existing][1]:.1f},"
                              f"{nodes[existing][2]:.1f}); "
                              f"loc={loc_ex} feat={feat_ex}")
                        if loc_ex in ('face', 'edge'):
                            # Use this existing node to split the tet
                            ins_pt = nodes[existing]
                            ins_type = 'existing_node'
                            # We'll use the existing node index directly
                            # (no new node needed)
                        else:
                            print(f"    [INFO] existing node is {loc_ex} "
                                  f"in tet; falling back to centroid")

        if ins_pt is None:
            # Fallback: tet centroid
            verts = [np.array(nodes[vi]) for vi in sel_tet]
            centroid_pt = tuple((sum(verts) / 4.0).tolist())
            existing = _find_existing_node(centroid_pt)
            if existing is None:
                ins_pt = centroid_pt
                ins_type = 'centroid'
            else:
                print(f"    [WARN] centroid also coincides with existing "
                      f"node {existing}; skipping this tet")
                # Mark this tet as converged by setting its LB high
                tet_lb_map[sel_idx] = float('inf')
                it += 1
                continue

        # Log per-scenario ms values
        if sel_ms_data is not None:
            for s in range(S):
                ms_v = sel_ms_data[s][2]
                arg_v = sel_ms_data[s][3]
                st_v = sel_ms_data[s][4]
                ms_s_str = f"{ms_v:.6f}" if ms_v is not None else "None"
                arg_s_str = (f"({arg_v[0]:.4f},{arg_v[1]:.4f},{arg_v[2]:.4f})"
                             if arg_v else "None")
                print(f"      s={s}: ms={ms_s_str}  argmin={arg_s_str}  "
                      f"status={st_v}")

        ms_disp = (f"{ins_ms_val:.6f}"
                   if ins_ms_val is not None and math.isfinite(ins_ms_val)
                   else "N/A")
        print(f"    -> insertion: {ins_type}  "
              f"x=({ins_pt[0]:.6f}, {ins_pt[1]:.6f}, {ins_pt[2]:.6f})  "
              f"ms={ms_disp}")
        selection_hist.append((sel_idx, ins_type,
                              f"({ins_pt[0]:.1f},{ins_pt[1]:.1f},"
                              f"{ins_pt[2]:.1f})",
                              ins_ms_val))

        # -- Write iteration log --
        verts_3d = [nodes[vi] for vi in sel_tet]
        # Compute dual bound (min Q(x) over the tet)
        tet_dual_bounds = []
        for s in range(S):
            db, db_st = solve_tet_dual_bound(verts_3d, s)
            tet_dual_bounds.append((db, db_st))

        log_file.write(f"--- Iteration {it} ---\n")
        log_file.write(f"Selected tet: {sel_idx}\n")
        log_file.write(f"Vertices (idx -> coords, Q values):\n")
        for j, vi in enumerate(sel_tet):
            coord = nodes[vi]
            qs_vi = qs_at_node.get(vi, {})
            q_strs = [f"Q_{s}={qs_vi.get(s, '?'):.4f}"
                      if isinstance(qs_vi.get(s), (int, float)) else f"Q_{s}=?"
                      for s in range(S)]
            log_file.write(f"  v{j} (node {vi}): "
                          f"({coord[0]:.6f}, {coord[1]:.6f}, {coord[2]:.6f})  "
                          f"{', '.join(q_strs)}\n")
        log_file.write(f"Per-scenario ms:\n")
        for s in range(S):
            if sel_ms_data is not None:
                ms_v = sel_ms_data[s][2]
                arg_v = sel_ms_data[s][3]
                st_v = sel_ms_data[s][4]
                ms_str = f"{ms_v:.6f}" if ms_v is not None else "None"
                arg_str = (f"({arg_v[0]:.6f}, {arg_v[1]:.6f}, {arg_v[2]:.6f})"
                           if arg_v else "None")
            else:
                ms_str, arg_str, st_v = "N/A", "N/A", "N/A"
            log_file.write(f"  s={s}: ms={ms_str}  "
                          f"argmin={arg_str}  status={st_v}\n")
        log_file.write(f"Dual bound (min Q(x) over tet):\n")
        for s in range(S):
            db, db_st = tet_dual_bounds[s]
            db_str = f"{db:.6f}" if db is not None else "failed"
            log_file.write(f"  s={s}: dual_bound={db_str}  "
                          f"status={db_st}\n")
        log_file.write(f"Simplex LB (A+ms): {tet_lb_map.get(sel_idx, 'N/A')}\n")
        log_file.write(f"Insertion: type={ins_type}  "
                      f"point=({ins_pt[0]:.6f}, {ins_pt[1]:.6f}, "
                      f"{ins_pt[2]:.6f})\n")
        log_file.write(f"Global: UB={true_ub:.4f}  LB={global_lb:.4f}  "
                      f"gap={gap:.4f}\n")
        log_file.write("\n")
        log_file.flush()

        # -- Plot BEFORE split --
        if _should_plot(it):
            _do_plot(it, sel_idx, ins_pt, true_ub, global_lb, tet_lb_map)

        # -- Classify and insert --
        if ins_type == 'existing_node':
            # The argmin coincides with an existing mesh node on
            # an edge/face of this tet.  Use it directly (no new node).
            existing_idx = _find_existing_node(ins_pt)
            new_idx = existing_idx
            loc_ex, feat_ex, _ = _classify_point_in_tet(
                ins_pt, sel_tet, nodes, tol=1e-4)
            if loc_ex == 'edge':
                edge_set = set(feat_ex)
                other_verts = [v for v in sel_tet if v not in edge_set]
                feat_ex = tuple(sorted(edge_set | {other_verts[0]}))
                loc_ex = 'face'
                print(f"    [INFO] existing node on edge -> face-split "
                      f"via face {feat_ex}")
            elif loc_ex == 'face':
                print(f"    [INFO] existing node on face {feat_ex} "
                      f"-> 3-split")
            else:
                # Shouldn't happen, but fall back to interior
                print(f"    [INFO] existing node is {loc_ex}; "
                      f"using interior 4-split")
                loc_ex = 'interior'

            if loc_ex == 'face':
                tets = split_face_point(nodes, tets, feat_ex, new_idx)
            else:
                tets = split_interior_point(nodes, tets, sel_idx, new_idx)
        else:
            # New point: classify, snap, insert
            loc, feat, proj_pt = _classify_point_in_tet(
                ins_pt, sel_tet, nodes, tol=1e-6)
            if loc == 'vertex':
                print(f"    [INFO] argmin is at vertex {feat}; skipping")
                tet_lb_map[sel_idx] = float('inf')
                it += 1
                continue
            if loc == 'face':
                ins_pt = proj_pt
                print(f"    [INFO] point on face {feat} -> 3-split")
            elif loc == 'edge':
                ins_pt = proj_pt
                edge_set = set(feat)
                other_verts = [v for v in sel_tet if v not in edge_set]
                feat = tuple(sorted(edge_set | {other_verts[0]}))
                loc = 'face'
                print(f"    [INFO] point on edge {edge_set} -> face-split "
                      f"via face {feat}")
            else:
                print(f"    [INFO] interior point -> 4-split")

            # Check for existing node at snapped location
            existing = _find_existing_node(ins_pt)
            if existing is not None:
                print(f"    [WARN] projected point coincides with node "
                      f"{existing}; using centroid")
                verts = [np.array(nodes[vi]) for vi in sel_tet]
                ins_pt = tuple((sum(verts) / 4.0).tolist())
                ins_type = 'centroid'
                loc = 'interior'
                existing2 = _find_existing_node(ins_pt)
                if existing2 is not None:
                    print(f"    [WARN] centroid also at node "
                          f"{existing2}; skip")
                    tet_lb_map[sel_idx] = float('inf')
                    it += 1
                    continue

            new_idx = len(nodes)
            nodes.append(ins_pt)

            if loc == 'face':
                tets = split_face_point(nodes, tets, feat, new_idx)
            else:
                tets = split_interior_point(
                    nodes, tets, sel_idx, new_idx)

        if not _validate_mesh(nodes, tets):
            print("  [FATAL] Mesh validation failed!")
            break

        # -- Evaluate ALL unevaluated nodes --
        all_node_ids = set()
        for t in tets:
            all_node_ids.update(t)
        for ni in all_node_ids:
            if ni not in qs_at_node:
                pt = nodes[ni]
                qs, f_exp = eval_Qs_at_point(pt[0], pt[1], pt[2],
                                              solver, model_list)
                qs_at_node[ni] = qs
                f_at_node[ni] = f_exp
                print(f"    [eval] node {ni}: "
                      f"({pt[0]:.1f},{pt[1]:.1f},{pt[2]:.1f})  "
                      f"F={f_exp:.2f}")

        it += 1

    dt_total = perf_counter() - t_start

    # Close iteration log
    log_file.close()
    print(f"  Iteration log saved: {log_path}")

    # -- Results --
    final_ub = min(f_at_node.values())
    final_lb = lb_hist[-1] if lb_hist else float('nan')
    final_gap = final_ub - final_lb

    print(f"\n{'=' * 60}")
    print(f"  RESULTS - {len(f_at_node)} nodes, {len(tets)} tets")
    print(f"{'=' * 60}")
    if KNOWN_OPTIMAL is not None:
        print(f"  Known F*:      {KNOWN_OPTIMAL:.2f}")
    print(f"  Best UB:       {final_ub:.2f}")
    print(f"  Simplex LB:    {final_lb:.2f}")
    print(f"  Gap (UB-LB):   {final_gap:.2e}")
    if KNOWN_OPTIMAL is not None:
        if abs(final_ub - KNOWN_OPTIMAL) < 100:
            print(f"  Near optimum (delta = {final_ub - KNOWN_OPTIMAL:.2f})")
        else:
            print(f"  Far from optimum (delta = {final_ub - KNOWN_OPTIMAL:.2f})")
    print(f"  Iterations:    {it}")
    print(f"  Wall time:     {dt_total:.1f}s")
    if save_plots or save_html:
        print(f"  Plots saved:   {plot_dir}")
    print(f"{'=' * 60}")

    # -- Per-iteration summary table --
    print(f"\n{'=' * 110}")
    print(f"  {'Iter':>4}  {'Nodes':>5}  {'UB':>12}  {'LB':>12}  "
          f"{'AbsGap':>12}  {'RelGap%':>8}  "
          f"{'Tet':>4}  {'Type':>14}  {'Point':>20}")
    print(f"{'-' * 110}")
    n_rows = min(len(ub_hist), len(selection_hist))
    for r in range(n_rows):
        ub_r = ub_hist[r]
        lb_r = lb_hist[r]
        abs_gap = ub_r - lb_r
        rel_gap = (abs_gap / max(abs(ub_r), 1e-12)) * 100.0
        ti, itype, ipt, _ = selection_hist[r]
        print(f"  {r:4d}  {node_hist[r]:5d}  {ub_r:12.2f}  {lb_r:12.2f}  "
              f"{abs_gap:12.2f}  {rel_gap:7.3f}%  "
              f"{ti:4d}  {itype:>14}  {ipt:>20}")
    # Print final row if there's one more UB/LB entry after last selection
    if len(ub_hist) > n_rows:
        r = n_rows
        ub_r = ub_hist[r]
        lb_r = lb_hist[r]
        abs_gap = ub_r - lb_r
        rel_gap = (abs_gap / max(abs(ub_r), 1e-12)) * 100.0
        print(f"  {r:4d}  {node_hist[r]:5d}  {ub_r:12.2f}  {lb_r:12.2f}  "
              f"{abs_gap:12.2f}  {rel_gap:7.3f}%  "
              f"{'':>4}  {'(converged)':>14}  {'':>20}")
    print(f"{'=' * 110}")

    _render_convergence_plot(ub_hist, lb_hist, node_hist,
                            show_inline=show_inline, save_plots=save_plots,
                            plot_dir=plot_dir)

    return {
        "nodes": [tuple(n) for n in nodes],
        "tets": list(tets),
        "history": {"ub": list(ub_hist), "lb": list(lb_hist),
                    "n_nodes": list(node_hist)},
        "f_at_node": dict(f_at_node),
        "qs_at_node": {k: dict(v) for k, v in qs_at_node.items()},
        "best_ub": final_ub,
        "best_lb": final_lb,
        "iterations": it,
        "wall_time": dt_total,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="3D simplex + per-scenario ms-LP LB (Farmer)")
    parser.add_argument("--target_nodes", type=int, default=50)
    parser.add_argument("--max_iters", type=int, default=50)
    parser.add_argument("--gap_tol", type=float, default=None)
    parser.add_argument("--time_limit", type=float, default=600.0)
    parser.add_argument("--show_inline", action="store_true", default=False)
    parser.add_argument("--save_plots", action="store_true", default=True)
    parser.add_argument("--no_save_plots", dest="save_plots",
                        action="store_false")
    parser.add_argument("--plot_every", type=int, default=1)
    parser.add_argument("--plot_dir", type=str, default=None)
    parser.add_argument("--pause", type=float, default=0.0)
    parser.add_argument("--save_html", action="store_true", default=True)
    parser.add_argument("--no_save_html", dest="save_html",
                        action="store_false")
    args = parser.parse_args()
    if not args.show_inline:
        matplotlib.use("Agg")
    run_visual_simplex(
        max_iters=args.max_iters,
        target_nodes=args.target_nodes,
        show_inline=args.show_inline,
        save_plots=args.save_plots,
        save_html=args.save_html,
        plot_every=args.plot_every,
        plot_dir=args.plot_dir,
        pause=args.pause,
        gap_tol=args.gap_tol,
        time_limit=args.time_limit,
    )
    return 0


def _in_notebook():
    try:
        from IPython import get_ipython
        return get_ipython() is not None
    except ImportError:
        return False


if __name__ == "__main__":
    if _in_notebook():
        results = run_visual_simplex(
            show_inline=False,
            save_html=True,
            save_plots=False,
        )
    else:
        try:
            sys.exit(main())
        except Exception as exc:
            import traceback
            print("\n" + "=" * 60)
            print("FATAL ERROR:")
            print("=" * 60)
            traceback.print_exc()
            sys.exit(1)
