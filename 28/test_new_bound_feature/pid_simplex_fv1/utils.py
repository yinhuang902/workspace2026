# utils.py
import numpy as np
from itertools import combinations, product
import pyomo.environ as pyo
from pyomo.contrib.fbbt.fbbt import fbbt
from pyomo.core import Objective
from pyomo.contrib.alternative_solutions.obbt import obbt_analysis
import plotly.graph_objects as go
import numpy as np

# ===== debug bucket =====
LAST_DEBUG = None   # If next_node collides with an existing vertex, the context will be added for debug

# ------------------------- Config knobs -------------------------
MIN_DIST   = 1e-8      # minimum distance between nodes
ACTIVE_TOL = 1e-8      # Tolerance for determining whether a simplex is active
MS_AGG     = "sum"     # 'sum' or 'mean' to process ms in different scenario
MS_CACHE_ENABLE = True  # Enable caching of simplex ms evaluations (True by default)
                        # when enabled, previously evaluated (simplex, scenario) pairs are not recomputed
GAP_STOP_TOL = 1e-4     # End iteration if the optimality rel-gap reaches this threshold

# Bad-shape inactivation tolerances (absolute thresholds)
SMALL_VOL_TOL_ABS = 1e-4   # volume below this triggers aspect-ratio check
ASPECT_BAD_TOL    = 1e5    # aspect ratio >= this marks simplex inactive
EDGE_EPS          = 1e-18  # epsilon for edge-length division

# ------------------------- Basic utils -------------------------
def corners_from_var_bounds(vars_3):
    bnds = []
    for v in vars_3:
        lb, ub = v.lb, v.ub
        if lb is None or ub is None:
            raise ValueError(f"{v.name} lack UB and LB")
        bnds.append((float(lb), float(ub)))
    return [tuple(p) for p in product(*[(lo, hi) for (lo,hi) in bnds])]

def too_close(p, nodes, tol=MIN_DIST):
    return any(np.linalg.norm(np.asarray(p)-np.asarray(q)) < tol for q in nodes)

def evaluate_Q_at(base_bundle, first_stg_vars, first_stg_vals, return_meta=False):
    return base_bundle.eval_at(first_stg_vars, first_stg_vals, return_meta=return_meta)


def evaluate_true_expected_obj(base_bundles, first_vars_list, K_tuple):
    """
    Evaluate the true expected objective (sum over all scenarios)
    at the given first-stage point K_tuple = (Kp, Ki, Kd).

    Returns the **total** (sum, not average) so it is directly comparable
    to UB_incumbent which is also stored as a sum.
    """
    total = 0.0
    for s in range(len(base_bundles)):
        total += evaluate_Q_at(base_bundles[s], first_vars_list[s], K_tuple)
    return total


# ---- EF per-iteration log file ----
_EF_LOG_HEADER_WRITTEN = {}          # path -> bool

_EF_LOG_FIELDS = [
    "iter", "simplex_id", "ef_attempted", "ef_ok",
    "solver_status", "termination_condition", "ef_time_sec",
    "K_ef", "ef_obj", "true_obj",
    "ub_updated", "UB_incumbent",
    # MS / dual-bound diagnostic columns (added for LB>UB investigation)
    "ms_issue_count", "ms_issue_scenarios", "ms_issue_reasons",
    "ms_issue_details",
    "dual_bound_fail_count", "dual_bound_fail_details",
    # Cached-scenario visibility
    "ms_cached_count", "ms_cached_scenarios",
    "cs_cached_count", "cs_cached_scenarios",
    # LB construction components (added for LB>UB investigation)
    "lb_simp_LB_sur", "lb_simp_LB_linear",
    "lb_simp_c_total_finite", "lb_simp_c_total_all",
    "lb_simp_ms_total", "lb_simp_min_fverts_sum",
    "lb_simp_lb_case",
    "LB_global_end_sum", "UB_global_end_sum",
    # Per-scenario bound invariant counts
    "ms_lb_gt_primal_count", "cs_lb_gt_primal_count",
]

_MAX_DETAIL_LEN = 500   # truncate detail strings to avoid over-long lines


def append_ef_info(log_path: str, fields: dict):
    """
    Append one tab-separated line to *log_path*.
    Creates the file with a header row on first call.
    If an existing file has a different header (schema change), the old file
    is backed up with a timestamp suffix and a fresh file is started.
    """
    import os
    from datetime import datetime as _dt

    if not _EF_LOG_HEADER_WRITTEN.get(log_path, False):
        # First call this process — check if file exists and header matches
        need_fresh = True
        if os.path.exists(log_path):
            try:
                with open(log_path, "r", encoding="utf-8") as fh:
                    first_line = fh.readline().rstrip("\n\r")
                existing_cols = first_line.split("\t")
                if existing_cols == list(_EF_LOG_FIELDS):
                    # Header matches — keep appending
                    need_fresh = False
                else:
                    # Header mismatch — backup old file
                    ts = _dt.now().strftime("%Y%m%d_%H%M%S")
                    base, ext = os.path.splitext(log_path)
                    backup = f"{base}.old.{ts}{ext}"
                    try:
                        os.rename(log_path, backup)
                    except OSError:
                        pass  # if rename fails, overwrite
            except Exception:
                pass  # can't read existing file — will overwrite

        if need_fresh:
            with open(log_path, "w", encoding="utf-8") as fh:
                fh.write("\t".join(_EF_LOG_FIELDS) + "\n")
        _EF_LOG_HEADER_WRITTEN[log_path] = True

    with open(log_path, "a", encoding="utf-8") as fh:
        vals = []
        for k in _EF_LOG_FIELDS:
            v = fields.get(k, "")
            vals.append(str(v))
        fh.write("\t".join(vals) + "\n")
        fh.flush()


def _truncate(s, maxlen=_MAX_DETAIL_LEN):
    """Truncate string *s* to *maxlen*, appending '...' if truncated."""
    s = str(s)
    if len(s) <= maxlen:
        return s
    return s[:maxlen - 3] + "..."


def collect_ms_cs_issues(lb_simp_rec):
    """
    Scan the per-scene MS and c_s metadata stored in *lb_simp_rec* and
    return a dict with diagnostic fields for simplex_ef_info.txt.

    Optimality is determined by **termination_condition** (not status).
    A solve is "optimal" only when ``termination_condition`` lowered equals
    ``"optimal"`` (matching Pyomo's ``TerminationCondition.optimal``).

    When metadata is ``None`` (cached solve), the scenario is counted as
    cached and recorded in ``ms_cached_count`` / ``ms_cached_scenarios``
    so hidden issues remain visible.

    Parameters
    ----------
    lb_simp_rec : dict
        The per_tet record for the selected (LB) simplex.  Must contain
        ``ms_meta_per_scene`` and ``cs_meta_per_scene``.

    Returns
    -------
    dict  with keys:
        ms_issue_count, ms_issue_scenarios, ms_issue_reasons,
        ms_issue_details, dual_bound_fail_count, dual_bound_fail_details,
        ms_cached_count, ms_cached_scenarios,
        cs_cached_count, cs_cached_scenarios
    """
    import json as _json
    import math as _math

    ms_meta_list = lb_simp_rec.get("ms_meta_per_scene", []) or []
    cs_meta_list = lb_simp_rec.get("cs_meta_per_scene", []) or []

    ms_issues = {}        # scen_id -> compact dict
    dual_fails = {}       # scen_id -> compact dict
    ms_cached = []        # list of scen_id strings
    cs_cached = []        # list of scen_id strings

    # --- scan MS metadata ---
    for s, meta in enumerate(ms_meta_list):
        scen_key = f"scen_{s+1}"
        if meta is None:
            ms_cached.append(scen_key)
            continue

        # Use termination_condition (not status) for optimality
        term = str(meta.get("termination_condition", "")).lower()
        is_opt = (term == "optimal")

        is_issue = False
        if meta.get("used_fallback", False):
            is_issue = True
        if not is_opt:
            is_issue = True
        if not meta.get("ok", True):
            is_issue = True

        if is_issue:
            ms_issues[scen_key] = {
                "term": meta.get("termination_condition", "?"),
                "fallback": meta.get("fallback_reason"),
                "status": meta.get("status", "?"),
                "ok": meta.get("ok"),
                "t": round(meta.get("time_sec", 0.0), 3),
            }

        # Dual bound failures (MS path)
        fb_reason = meta.get("fallback_reason") or ""
        is_dual_fail = False

        # Explicit fallback reason mentioning bounds
        if meta.get("used_fallback", False) and (
            "dual" in fb_reason.lower()
            or "bound" in fb_reason.lower()
            or "no_lower" in fb_reason.lower()
        ):
            is_dual_fail = True

        # Also flag when dual_bound key is None/NaN (if it exists)
        db = meta.get("dual_bound")
        if db is not None:
            try:
                if not _math.isfinite(float(db)):
                    is_dual_fail = True
            except (ValueError, TypeError):
                is_dual_fail = True
        elif meta.get("used_fallback", False):
            # dual_bound key absent AND fallback used -> suspicious
            is_dual_fail = True

        # Non-optimal + any fallback -> dual bound likely bad
        if not is_opt and meta.get("used_fallback", False):
            is_dual_fail = True

        if is_dual_fail:
            dual_fails[scen_key] = {
                "what": f"ms_{fb_reason}" if fb_reason else "ms_nonoptimal",
                "term": meta.get("termination_condition", "?"),
            }

    # --- scan c_s metadata ---
    for s, meta in enumerate(cs_meta_list):
        scen_key = f"scen_{s+1}"
        if meta is None:
            cs_cached.append(scen_key)
            continue

        term = str(meta.get("termination_condition", "")).lower()
        is_opt = (term == "optimal")
        fb_reason = meta.get("fallback_reason") or ""
        is_dual_fail = False

        if meta.get("used_fallback", False) and (
            "dual" in fb_reason.lower()
            or "bound" in fb_reason.lower()
            or "no_lower" in fb_reason.lower()
        ):
            is_dual_fail = True

        db = meta.get("dual_bound")
        if db is not None:
            try:
                if not _math.isfinite(float(db)):
                    is_dual_fail = True
            except (ValueError, TypeError):
                is_dual_fail = True
        elif meta.get("used_fallback", False):
            is_dual_fail = True

        if not is_opt and meta.get("used_fallback", False):
            is_dual_fail = True

        if is_dual_fail:
            entry = {
                "what": f"cs_{fb_reason}" if fb_reason else "cs_nonoptimal",
                "term": meta.get("termination_condition", "?"),
            }
            if scen_key in dual_fails:
                existing = dual_fails[scen_key]
                if isinstance(existing, list):
                    existing.append(entry)
                else:
                    dual_fails[scen_key] = [existing, entry]
            else:
                dual_fails[scen_key] = entry

    # --- assemble output ---
    ms_issue_scenarios = ",".join(ms_issues.keys()) if ms_issues else ""
    ms_issue_reasons = ",".join(
        (v.get("fallback") or v.get("status", "?"))
        for v in ms_issues.values()
    ) if ms_issues else ""

    try:
        # Embed cached count inside detail string for visibility
        detail_obj = {"issues": ms_issues, "cached": len(ms_cached)}
        ms_detail_str = _json.dumps(detail_obj, separators=(",", ":"))
    except Exception:
        ms_detail_str = str(ms_issues)

    try:
        dual_detail_str = _json.dumps(dual_fails, separators=(",", ":"))
    except Exception:
        dual_detail_str = str(dual_fails)

    # --- count dual > primal invariant violations ---
    _DGP_EPS = 1e-6
    ms_lb_gt_count = 0
    cs_lb_gt_count = 0
    for meta in ms_meta_list:
        if meta is None:
            continue
        db = meta.get("dual_bound")
        pr = meta.get("primal_obj")
        if (db is not None and pr is not None):
            try:
                dbf, prf = float(db), float(pr)
                if _math.isfinite(dbf) and _math.isfinite(prf) and dbf > prf + _DGP_EPS:
                    ms_lb_gt_count += 1
            except (ValueError, TypeError):
                pass
    for meta in cs_meta_list:
        if meta is None:
            continue
        db = meta.get("dual_bound")
        pr = meta.get("primal_obj")
        if (db is not None and pr is not None):
            try:
                dbf, prf = float(db), float(pr)
                if _math.isfinite(dbf) and _math.isfinite(prf) and dbf > prf + _DGP_EPS:
                    cs_lb_gt_count += 1
            except (ValueError, TypeError):
                pass

    return {
        "ms_issue_count": len(ms_issues),
        "ms_issue_scenarios": ms_issue_scenarios,
        "ms_issue_reasons": ms_issue_reasons,
        "ms_issue_details": _truncate(ms_detail_str),
        "dual_bound_fail_count": len(dual_fails),
        "dual_bound_fail_details": _truncate(dual_detail_str),
        "ms_cached_count": len(ms_cached),
        "ms_cached_scenarios": ",".join(ms_cached) if ms_cached else "",
        "cs_cached_count": len(cs_cached),
        "cs_cached_scenarios": ",".join(cs_cached) if cs_cached else "",
        "ms_lb_gt_primal_count": ms_lb_gt_count,
        "cs_lb_gt_primal_count": cs_lb_gt_count,
    }


def tet_volume(verts):
    V = np.array(verts, float)
    v0, v1, v2, v3 = V
    return float(abs(np.linalg.det(np.stack([v1 - v0, v2 - v0, v3 - v0], axis=1))) / 6.0)

def tet_quality(verts):
    V = np.array(verts, float)
    edges = [np.linalg.norm(V[i] - V[j]) for (i, j) in combinations(range(4), 2)]
    denom = float(np.sum(np.power(edges, 3))) + 1e-16
    vol = tet_volume(verts)
    return float(6.0 * vol / denom)


def compute_edge_aspect(verts):
    """Return (max_edge, min_edge, aspect) for a simplex.

    Parameters
    ----------
    verts : list of array-like
        d+1 vertices of the simplex in R^d.

    Returns
    -------
    max_edge : float
    min_edge : float
    aspect   : float   (max_edge / (min_edge + EDGE_EPS)), or inf if min_edge <= EDGE_EPS
    """
    V = np.array(verts, float)
    n = len(V)
    edges = [float(np.linalg.norm(V[i] - V[j]))
             for (i, j) in combinations(range(n), 2)]
    max_e = max(edges)
    min_e = min(edges)
    if min_e <= EDGE_EPS:
        return max_e, min_e, float("inf")
    return max_e, min_e, max_e / (min_e + EDGE_EPS)

def min_dist_to_nodes(pt, nodes):
    P = np.asarray(pt, float)
    X = np.asarray(nodes, float)
    return float(np.min(np.linalg.norm(X - P, axis=1)))

# ----------------- print tables in each iteration ----------------------
def _print_candidates_table(cands_sorted, nodes, topN=10):
    W = {"rank":4, "simp":6, "scene":7, "ms":12, "mind":12, "pt":30}

    def header_line():
        return (f"{'rank':>{W['rank']}} "
                f"{'simp':>{W['simp']}} "
                f"{'scene':>{W['scene']}} "
                f"{'ms':>{W['ms']}} "
                f"{'mind(all)':>{W['mind']}} "
                f"{'pt':>{W['pt']}}")

    print("== ms candidates (sorted by (ms, -dist)) ==")
    head = header_line()
    print(head)
    print("-" * len(head))

    for rnk, ci in enumerate(cands_sorted[:topN], start=1):
        pt = ci["cand_pt"]
        d  = float('nan') if pt is None else min_dist_to_nodes(pt, nodes)
        simp = f"T{ci['simplex_index']}"
        pt_str = "None" if pt is None else f"({pt[0]:.4f}, {pt[1]:.4f}, {pt[2]:.4f})"
        print(f"{rnk:>{W['rank']}} "
              f"{simp:>{W['simp']}} "
              f"{ci['scene']:>{W['scene']}} "
              f"{ci['cand_ms']:>{W['ms']}.4e} "
              f"{d:>{W['mind']}.2e} "
              f"{pt_str:>{W['pt']}}")

def print_tetra_table(per_tet, active_mask, purple_set=None, prec=6):
    purple_set = set() if purple_set is None else set(purple_set)
    per_tet = sorted(per_tet, key=lambda r: r["simplex_index"])
    tet_ids = [r["simplex_index"] for r in per_tet]
    active_set = {tid for tid in tet_ids if active_mask.get(tid, False)}

    def _mark(tid):
        s = f"T{tid}"
        flags = []
        if tid in active_set:  flags.append("*")
        if tid in purple_set:  flags.append("^")
        return s + ("".join(flags) if flags else "")

    header = ["row\\simp"] + [_mark(tid) for tid in tet_ids]
    rows = [
        ["UB"] + [f"{r['UB']:.{prec}f}" for r in per_tet],
        ["LB"] + [f"{r['LB']:.{prec}f}" for r in per_tet],
        ["ms"] + [f"{r['ms']:.3e}"       for r in per_tet],
    ]
    table = [header] + rows
    colw = [max(len(str(row[c])) for row in table) + 2 for c in range(len(header))]

    RED, PURPLE, RESET = "\033[31m", "\033[35m", "\033[0m"
    def colorize(col_idx, s):
        if col_idx == 0:
            return s
        tid = tet_ids[col_idx-1]
        if tid in purple_set:
            return f"{PURPLE}{s}{RESET}"
        elif tid in active_set:
            return f"{RED}{s}{RESET}"
        return s

    print("\n== Per-tetra summary ==")
    print("".join(colorize(c, str(header[c]).ljust(colw[c])) for c in range(len(header))))
    print("-"*sum(colw))
    for r in rows:
        line = []
        for c in range(len(header)):
            cell = str(r[c])
            pad  = cell.ljust(colw[c]) if c==0 else cell.rjust(colw[c])
            line.append(colorize(c, pad))
        print("".join(line))
    print("(Red column = active; Purple column = simplex containing UB; Row 1 = UB, Row 2 = LB, Row 3 = ms)\n")

def print_per_scenario_ms(per_tet, max_scenarios_to_print=10, prec=3):
    per_tet = sorted(per_tet, key=lambda r: r["simplex_index"])
    if not per_tet or "ms_per_scene" not in per_tet[0]:
        return
    S = len(per_tet[0]["ms_per_scene"])
    show = min(S, max_scenarios_to_print)
    head = "simp | " + " ".join([f"s{j}".rjust(10) for j in range(show)])
    print("== Per-tetra per-scenario ms (showing first", show, "of", S, "scenes) ==")
    print(head); print("-"*len(head))
    for r in per_tet:
        arr = r["ms_per_scene"][:show]
        sline = " ".join([f"{v:.{prec}e}".rjust(10) for v in arr])
        print(f"{r['simplex_index']:>4d} | {sline}")
    if show < S:
        print(f"... ({S-show} scenes omitted)")
    print()

# ------------------------- Plotly visualization -------------------------
def plot_iteration_plotly(iter_id, nodes, tri, active_mask,
                          ub_node, next_node, per_tet,
                          highlight_simplices=None,
                          true_opt_points=None,
                          UB_global=None,
                          LB_global=None):


    import numpy as np
    import plotly.graph_objects as go

    if highlight_simplices is None:
        highlight_simplices = set()
    else:
        highlight_simplices = set(highlight_simplices)

    fig = go.Figure()
    nodes = np.asarray(nodes, float)

    # --- NEW: precompute per-node sum Q (across scenarios) for hover ---
    node_q_sum = None
    node_colors = ["black"] * len(nodes)  # Default color

    try:
        if per_tet and len(nodes) > 0:
            n_nodes = len(nodes)
            # node_q_sum[i] = sum_s Q_s(node i)
            node_q_sum = [None] * n_nodes
            for r in per_tet:
                # The per_tet record contains the global vertex index and fverts_sum
                if "vert_idx" not in r or "fverts_sum" not in r:
                    continue
                idxs = list(r["vert_idx"])       # Global node index
                fsum = list(r["fverts_sum"])     # The sum_s and Q_s of the 4 vertices on this tet
                for j, gi in enumerate(idxs):
                    gi = int(gi)
                    if gi < 0 or gi >= n_nodes:
                        continue
                    val = float(fsum[j])
                    if node_q_sum[gi] is None:
                        node_q_sum[gi] = val
                    else:
                        # The values ​​at the same point in multiple simplexes 
                        # should be consistent; here, we conservatively use an average
                        node_q_sum[gi] = 0.5 * (node_q_sum[gi] + val)
            
            # Determine colors based on Q values
            for i in range(n_nodes):
                val = node_q_sum[i]
                if val is not None and val >= 1e5 - 1e-9:
                    node_colors[i] = "orange"
                # If val is None, it might be an unused node or something, keep black or handle if needed
                
    except Exception:
        node_q_sum = None
        # Fallback to black if something fails


    if true_opt_points is not None:
        true_opt_points = np.asarray(true_opt_points, float)


    if len(nodes) > 0:
        fig.add_trace(go.Scatter3d(
            x=nodes[:, 0], y=nodes[:, 1], z=nodes[:, 2],
            mode='markers',
            marker=dict(size=4, color=node_colors),
            name='nodes',
            # NEW: Each node corresponds to a sum_s Q_s, used for hover display.
            customdata=node_q_sum
        ))


    if ub_node is not None:
        fig.add_trace(go.Scatter3d(
            x=[ub_node[0]], y=[ub_node[1]], z=[ub_node[2]],
            mode='markers',
            marker=dict(size=7, symbol="circle", color="blue"),
            name='UB node'
        ))

    if next_node is not None:
        fig.add_trace(go.Scatter3d(
            x=[next_node[0]], y=[next_node[1]], z=[next_node[2]],
            mode='markers',
            marker=dict(size=7, symbol="circle", color="red"),
            name='LB node / next node'
        ))

    if true_opt_points is not None and len(true_opt_points) > 0:
        fig.add_trace(go.Scatter3d(
            x=true_opt_points[:, 0],
            y=true_opt_points[:, 1],
            z=true_opt_points[:, 2],
            mode="markers",
            marker=dict(size=6, symbol="circle", color="lightgreen", opacity=0.9),
            name="true opt (per scen)",
        ))




    def _is_same_point(a, b, atol=1e-6):
        if a is None or b is None:
            return False
        return np.linalg.norm(np.asarray(a, float) - np.asarray(b, float)) <= float(atol)

    if tri is not None:
        legend_mesh_added = False
        legend_edge_added = False

        for r in per_tet:
            sid = r["simplex_index"]
            if not active_mask.get(sid, False):
                continue

            verts = np.array(r["verts"], dtype=float)

            # current simplex is the global lb simplex
            is_lb_simp = (sid in highlight_simplices)

            if is_lb_simp:
                # global lb simplex color: lightcoral
                mesh_color   = "lightcoral"   
                edge_color   = "lightcoral"
                edge_width   = 5
                mesh_opacity = 0.65
            else:
                # other active simplex color: light grey
                mesh_color   = "lightgray"
                edge_color   = "gray"
                edge_width   = 1.0
                mesh_opacity = 0.25


            I = [0, 0, 0, 1]
            J = [1, 1, 2, 2]
            K = [2, 3, 3, 3]

            qtxt = ""
            if "quality" in r and r["quality"] is not None:
                try:
                    qtxt = f"<br>q={float(r['quality']):.3e}"
                except Exception:
                    qtxt = ""
            txt = (f"simp={sid}"
                   f"<br>LB={float(r['LB']):.6f}"
                   f"<br>UB={float(r['UB']):.6f}"
                   f"<br>ms={float(r['ms']):.3e}"
                   f"<br>vol={float(r['volume']):.3e}"
                   f"{qtxt}")
            
            fig.add_trace(go.Mesh3d(
                x=verts[:, 0], y=verts[:, 1], z=verts[:, 2],
                i=I, j=J, k=K,
                color=mesh_color,
                opacity=mesh_opacity,
                showscale=False,
                name="active simplex",
                showlegend=(not legend_mesh_added),
                hoverinfo="text",
                hovertext=txt,
                # Optional: Use hovertemplate to control the format.
                # hovertemplate=txt + "<extra></extra>",
            ))
            legend_mesh_added = True


            edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3)]
            for (a, b) in edges:
                pa, pb = verts[a], verts[b]
                fig.add_trace(go.Scatter3d(
                    x=[pa[0], pb[0]],
                    y=[pa[1], pb[1]],
                    z=[pa[2], pb[2]],
                    mode='lines',
                    line=dict(width=edge_width, color=edge_color),
                    name='active edge',
                    showlegend=(not legend_edge_added)
                ))
            legend_edge_added = True

            
            '''
            cx, cy, cz = np.mean(verts, axis=0)
            qtxt = ""
            if "quality" in r and r["quality"] is not None:
                try:
                    qtxt = f"<br>q={float(r['quality']):.3e}"
                except Exception:
                    qtxt = ""
            txt = (f"simp={sid}"
                   f"<br>LB={float(r['LB']):.6f}"
                   f"<br>UB={float(r['UB']):.6f}"
                   f"<br>ms={float(r['ms']):.3e}"
                   f"<br>vol={float(r['volume']):.3e}"
                   f"{qtxt}")

            fig.add_trace(go.Scatter3d(
                x=[cx], y=[cy], z=[cz],
                mode='markers',
                marker=dict(size=1, opacity=0.0),
                text=[txt], hoverinfo="text",
                name="tetra info",
                showlegend=False
            ))
            '''
            

    fig.update_layout(
        title=f"Iteration {iter_id}",
        scene=dict(
            xaxis_title="Kp",
            yaxis_title="Ki",
            zaxis_title="Kd",
            aspectmode="cube",
            zaxis=dict(tickformat=".2f"),
        ),
        width=980,
        height=720,
        legend=dict(itemsizing="constant")
    )

    # show Kp,Ki,Kd
    fig.update_traces(
        hovertemplate="Kp: %{x:.6f}<br>Ki: %{y:.6f}<br>Kd: %{z:.6f}",
        selector=dict(type='scatter3d')
    )
    # Node trace: Displays sum Q
    if node_q_sum is not None:
        fig.update_traces(
            hovertemplate=(
                "Kp: %{x:.6f}<br>"
                "Ki: %{y:.6f}<br>"
                "Kd: %{z:.6f}<br>"
                "sum Q: %{customdata:.6e}"
            ),
            selector=dict(type='scatter3d', name='nodes')
        )

    # === Write UB/LB on the chart ===
    text_lines = []
    if UB_global is not None:
        text_lines.append(f"UB (sum Q) = {UB_global:.6e}")
    if LB_global is not None:
        text_lines.append(f"LB (surrogate) = {LB_global:.6e}")

    if text_lines:
        fig.add_annotation(
            x=0.02, y=0.98, xref="paper", yref="paper",
            text="<br>".join(text_lines),
            showarrow=False,
            align="left",
            bgcolor="rgba(255,255,255,0.8)",
            bordercolor="black",
            borderwidth=1,
            font=dict(size=11)
        )

    fig.show()
    return fig



# ------------------------- Tightening -------------------------
def tighten_bounds_one_model(model, first_stage_vars,
                             use_fbbt=True,
                             use_obbt=True,
                             obbt_solver_name="gurobi",
                             obbt_solver_opts=None,
                             max_rounds=3,
                             tol=1e-6,
                             verbose=True):
    if obbt_solver_opts is None:
        obbt_solver_opts = {}

    def _snapshot_bounds(vs):
        return [(float(v.lb) if v.lb is not None else -float("inf"),
                 float(v.ub) if v.ub is not None else  float("inf")) for v in vs]

    def _max_change(old, vs):
        mx = 0.0
        for (olb, oub), v in zip(old, vs):
            nlb = float(v.lb) if v.lb is not None else -float("inf")
            nub = float(v.ub) if v.ub is not None else  float("inf")
            mx = max(mx, abs(nlb - olb), abs(nub - oub))
        return mx

    rounds = 0
    while True:
        changed = False
        if use_fbbt:
            fbbt(model)

        if use_obbt and len(first_stage_vars) > 0:
            active_objs = list(model.component_objects(Objective, active=True))
            tmp_added = False
            if len(active_objs) == 0:
                tmp_obj = Objective(expr=model.obj_expr, sense=pyo.minimize)
                model.add_component("_obbt_tmp_obj", tmp_obj)
                tmp_added = True
            elif len(active_objs) > 1:
                # Multiple targets will cause OBBT to throw error, so stop here
                raise RuntimeError("If more than one objective function is activated in the model, please retain one or disable the extra objective.")

            before = _snapshot_bounds(first_stage_vars)
            # When restricting the variable set, warmstart should be turned off
            obbt_analysis(model,
                          variables=list(first_stage_vars),
                          solver=obbt_solver_name,
                          solver_options=dict(obbt_solver_opts),
                          warmstart=False)
            if tmp_added:
                # delete temporary target
                model.del_component("_obbt_tmp_obj")

            after_change = _max_change(before, first_stage_vars)
            changed = changed or (after_change > tol)

        rounds += 1
        if (not changed) or (rounds >= max_rounds):
            if verbose:
                print(f"[Tighten] rounds={rounds}, changed={changed}")
            break



# ========================
# SimplexTracker
# ========================

from dataclasses import dataclass

@dataclass
class IterationStats:
    created: int = 0
    active: int = 0
    active_with_ub: int = 0
    ms_recomputed: int = 0

class SimplexTracker:
    def __init__(self, print_fn=print):
        self.print = print_fn
        self.cum_created = 0
        self.iter_idx = None
        self.current = None
        self._created_ids = set()
        self._active_ids = set()
        self._active_with_ub_ids = set()
        self._ms_recomp_ids = set()

    def start_iter(self, k: int):
        self.iter_idx = k
        self.current = IterationStats()
        self._created_ids.clear()
        self._active_ids.clear()
        self._active_with_ub_ids.clear()
        self._ms_recomp_ids.clear()

    def end_iter(self):
        c = self.current
        self.print(
            f"[Iter {self.iter_idx}] "
            f"created={c.created} (cum={self.cum_created}), "
            f"active={c.active}, "
            f"active+UB={c.active_with_ub}, "
            f"ms_recomputed={c.ms_recomputed}"
        )

    def note_created(self, simplex_id):
        if simplex_id not in self._created_ids:
            self._created_ids.add(simplex_id)
            self.current.created += 1
            self.cum_created += 1

    def note_active(self, simplex_id, has_ub: bool = False):
        if simplex_id not in self._active_ids:
            self._active_ids.add(simplex_id)
            self.current.active += 1
        if has_ub and simplex_id not in self._active_with_ub_ids:
            self._active_with_ub_ids.add(simplex_id)
            self.current.active_with_ub += 1

    def note_ms_recomputed(self, simplex_id):
        if simplex_id not in self._ms_recomp_ids:
            self._ms_recomp_ids.add(simplex_id)
            self.current.ms_recomputed += 1
