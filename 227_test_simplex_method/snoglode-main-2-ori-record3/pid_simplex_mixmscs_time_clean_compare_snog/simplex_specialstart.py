# simplex_specialstart.py
import csv, json, math, os
import numpy as np
import pyomo.environ as pyo   
from scipy.spatial import Delaunay
from time import perf_counter
from pyomo.opt import SolverStatus, TerminationCondition
from exact_opt import compute_exact_optima  # NEW: for validation points

from utils import SimplexTracker
from utils import (
    MIN_DIST, ACTIVE_TOL, MS_AGG, MS_CACHE_ENABLE, GAP_STOP_TOL,
    SMALL_VOL_TOL_ABS, ASPECT_BAD_TOL, EDGE_EPS,
    corners_from_var_bounds, evaluate_Q_at, tet_quality, min_dist_to_nodes,
    compute_edge_aspect,
    _print_candidates_table, plot_iteration_plotly, LAST_DEBUG,
    evaluate_true_expected_obj, append_ef_info, collect_ms_cs_issues,
)
from bundles import SurrogateLBBundle
from iter_logger import IterationLogger
from ef_upper_bounder import SimplexEFUb



# Tolerances for candidate-point feature snapping
TOL_MS_C = 1e-3      # Distance threshold between ms point and c_s point
TOL_C_VERTS = 1e2   # Distance threshold between c_s point and simplex vertices

TOL_MS_C = 1e3      # Distance threshold between ms point and c_s point
TOL_C_VERTS = 1e-6   # Distance threshold between c_s point and simplex vertices


# EXPERIMENTAL: set to True to also solve EF with Gurobi alongside ipopt (remove later)
ENABLE_EF_GUROBI = False

class SimplexMesh:
    """
    Maintain a tetrahedral mesh incrementally:
    - Initialize once from Delaunay(nodes).
    - Afterwards, update by (optionally) edge/face subdivision or full
      star-subdivision when new nodes are added.
    """

    def __init__(self, nodes):
        self.update_from_delaunay(nodes)
        self.last_split_kind = None   # record subdivision type

    def update_from_delaunay(self, nodes):
        pts = np.asarray(nodes, float)
        if len(pts) < 4:
            self.tets = []
            return
        tri = Delaunay(pts)
        # each simplex is stored as a tuple of GLOBAL vertex indices
        self.tets = [tuple(map(int, simp)) for simp in tri.simplices]

    def iter_simplices(self):
        """
        Yield (local_index, vert_idx_list) for all tetrahedra.
        local_index is just the position in self.tets for this iteration.
        vert_idx_list contains GLOBAL node indices.
        """
        for k, idxs in enumerate(self.tets):
            yield k, list(idxs)

    # ---------- Original star subdivision: 4 sub-simplices ----------
    def subdivide(self, simplex_index: int, new_node_index: int):
        """
        Replace the tetrahedron at simplex_index by 4 new tets that
        all contain new_node_index.

        Assumes:
            - new_node_index is a valid index into the global nodes list.
            - The new point lies (approximately) inside the old tetrahedron.
        """
        old = self.tets[simplex_index]
        if len(old) != 4:
            raise ValueError(f"Expected a 4-vertex simplex, got {old}")

        v0, v1, v2, v3 = old
        # Star subdivision: new point + each face of the original tetrahedron
        new_tets = [
            (new_node_index, v1, v2, v3),
            (v0, new_node_index, v2, v3),
            (v0, v1, new_node_index, v3),
            (v0, v1, v2, new_node_index),
        ]

        # Remove old tet and append new ones
        self.tets.pop(simplex_index)
        self.tets.extend(new_tets)
        self.last_split_kind = "interior"
        return 4  # LB/UB per user definition: return n_children for new_ids detection

    # ---------- New: point on edge => 2 sub-simplices ----------
    def subdivide_edge(self, simplex_index: int, new_node_index: int, edge_verts):
        """
        Subdivide a tetrahedron when the new point lies on an edge.

        Parameters
        ----------
        simplex_index : int
            Index into self.tets.
        new_node_index : int
            GLOBAL index of the new node (already appended to nodes list).
        edge_verts : tuple[int, int]
            GLOBAL indices of the two vertices (a, b) that define the edge
            on which the new point lies.
        """
        old = self.tets.pop(simplex_index)   # e.g. (i0, i1, i2, i3)
        old = list(old)

        a, b = edge_verts
        if a not in old or b not in old:
            raise ValueError(f"edge_verts {edge_verts} not subset of tet {old}")

        others = [v for v in old if v not in (a, b)]
        if len(others) != 2:
            raise ValueError(f"Expected 2 opposite vertices, got {others}")
        c, d = others

        t1 = (new_node_index, b, c, d)
        t2 = (a, new_node_index, c, d)

        self.tets.extend([t1, t2])
        self.last_split_kind = "edge"
        return 2  # LB/UB per user definition: return n_children for new_ids detection

    # ---------- New: point on face => 3 sub-simplices ----------
    def subdivide_face(self, simplex_index: int, new_node_index: int, face_verts):
        """
        Subdivide a tetrahedron when the new point lies on a face.

        Parameters
        ----------
        simplex_index : int
            Index into self.tets.
        new_node_index : int
            GLOBAL index of the new node.
        face_verts : tuple[int, int, int]
            GLOBAL indices of the three vertices (a, b, c) that define the face
            on which the new point lies.
        """
        old = self.tets.pop(simplex_index)
        old = list(old)

        a, b, c = face_verts
        for v in (a, b, c):
            if v not in old:
                raise ValueError(f"face_verts {face_verts} not subset of tet {old}")

        opp = [v for v in old if v not in (a, b, c)]
        if len(opp) != 1:
            raise ValueError(f"Expected 1 opposite vertex, got {opp}")
        d = opp[0]

        t1 = (new_node_index, b, c, d)
        t2 = (a, new_node_index, c, d)
        t3 = (a, b, new_node_index, d)

        self.tets.extend([t1, t2, t3])
        self.last_split_kind = "face"
        return 3  # LB/UB per user definition: return n_children for new_ids detection

    def as_delaunay_like(self):
        """
        Return a light-weight object with a 'simplices' attribute so that
        plotting code expecting scipy.spatial.Delaunay still works.
        """
        class _Dummy:
            pass

        obj = _Dummy()
        if self.tets:
            obj.simplices = np.asarray(self.tets, dtype=int)
        else:
            obj.simplices = np.zeros((0, 4), dtype=int)
        return obj




# small LP solver
# small LP solver (now backed by a persistent Gurobi solver)
_LB_BUNDLE = None   # global singleton, built once for given S

def solve_surrogate_lb_for_tet(fverts_per_scene, ms_scene, c_scene):
    """
    fverts_per_scene: list of length S, each is length-4 list of As_s at 4 vertices
    ms_scene: length-S iterable of ms_s
    c_scene: length-S iterable of c_s (may contain -inf)

    Convention:
      - If some ms_s = +inf, it means ms subproblem failed for that scenario;
        we don't use As+ms condition, only use sum_s c_s as LB for this simplex.
    """
    global _LB_BUNDLE

    S = len(ms_scene)

    # ---------- cheap fallback: LB_linear & LB_const ----------
    fverts_sum = [sum(fverts_per_scene[s][j] for s in range(S)) for j in range(4)]
    ms_scene_arr = np.asarray(ms_scene, float)
    c_scene_arr  = np.asarray(c_scene,  float)

    finite_c = c_scene_arr[np.isfinite(c_scene_arr)]

    # === Case 1: Exists ms_s = +inf -> treat ms as "unsolvable", use sum c_s as LB ===
    if not np.all(np.isfinite(ms_scene_arr)):
        if finite_c.size > 0:
            # Use only solved c_s for a conservative LB (still a global lower bound)
            return float(np.sum(finite_c))
        else:
            # Even c_s failed, treat simplex as "unusable": give +inf to be ignored in min(LB)
            return float('inf')

    # === Case 2: All ms_s finite -> keep original surrogate-LP logic ===
    ms_total   = float(np.sum(ms_scene_arr))
    LB_linear  = float(np.min(fverts_sum) + ms_total)

    if finite_c.size > 0:
        c_total     = float(np.sum(finite_c))
        LB_const    = c_total
        fallback_LB = max(LB_linear, LB_const)
    else:
        # all c_s = -inf, equivalent to only having As+ms part
        fallback_LB = LB_linear

    # If all c_s are -inf, no need to solve LP, use fallback_LB directly
    if finite_c.size == 0:
        return fallback_LB

    # ---------- use dedicated persistent solver for this LP ----------
    if (_LB_BUNDLE is None) or (_LB_BUNDLE.S != S):
        _LB_BUNDLE = SurrogateLBBundle(S)

    LB_sur = _LB_BUNDLE.compute_lb(
        fverts_per_scene=fverts_per_scene,
        ms_scene=ms_scene_arr,
        c_scene=c_scene_arr,
        fallback_LB=fallback_LB,
    )

    return float(LB_sur)


# ------------------------- Single tetra & scene: ms solve (persistent) -------------------------
def ms_on_tetra_for_scene(ms_bundle, tet_vertices, fverts_scene):
    """
    Solve ms and constant-cut for a single simplex(tetrahedron) in one scenario.

    Args:
        ms_bundle: Persistent model/bundle for the given scenario.
        tet_vertices (list[tuple[float]]): The 4 vertex coordinates of the tetrahedron.
        fverts_scene (list[float]): Objective values at those vertices for this scenario.

    Returns:
        tuple:
            ms_val (float): ms value; +inf if ms solve failed.
            new_pt_ms (tuple | None): Interpolated point (Kp,Ki,Kd) from ms subproblem, None if failed.
            c_val (float): c_s = min_T Q_s(K), -inf if failed.
            c_pt (tuple | None): (Kp,Ki,Kd) corresponding to c_s, None if failed.
    """
    # Update current simplex geometry + vertex function values
    ms_bundle.update_tetra(tet_vertices, fverts_scene)

    # 1) Solve ms subproblem first
    ok_ms = ms_bundle.solve()
    if ok_ms:
        ms_val, lam_star, new_pt_ms = ms_bundle.get_ms_and_point()
    else:
        ms_val = float('inf')
        lam_star = None
        new_pt_ms = None

    # 2) Try solving constant cut regardless of ms success
    ok_c, c_val, c_pt = ms_bundle.solve_const_cut()
    if not ok_c:
        print("c_s solve wrong")
        c_val = float('-inf')
        c_pt = None

    return ms_val, new_pt_ms, c_val, c_pt


# ------------------------- Evaluate all tetrahedra (per-scene) -------------------------
def evaluate_all_tetra(nodes, scen_values, ms_bundles, first_vars_list,
                       ms_cache=None, cache_on=True, tracker=None,
                       tet_mesh: SimplexMesh | None = None,
                       lb_sur_cache=None,  # LB surrogate cache
                       dbg_timelimit_path=None,  # path for timeout log
                       dbg_cs_timing_path=None,  # path for CS (Q-value) timing log
                       dbg_ms_timing_path=None,  # path for MS timing log
                       iter_num=None):  # iteration number for logging
    """
    Evaluate all Delaunay simplex formed by the node set
    across all scenarios, computing their ms values,
    lower/upper bounds, and candidate points.

    For each simplex(tetrahedron):
        - It gathers objective values at the four vertices for each scenario.
        - Solves the ms subproblem per scenario (with caching to skip repeats).
        - Aggregates per-scene ms values into a single ms (via MS_AGG).
        - Computes LB/UB and identifies the best scene and candidate point.

    Parameters
    ----------
    nodes : list[tuple[float]]
        Current first-stage points (Kp, Ki, Kd, ...).
    scen_values : list[list[float]]
        Cached Q evaluations for each scenario s at each node i.
        Shape: [S][N].
    ms_bundles : list[MSBundle]
        Scenario-specific persistent ms solvers.


    first_vars_list : list[list[pyo.Var]]
        Corresponding first-stage Pyomo variables for each scenario.
    ms_cache : dict, optional
        Cache {(scene_idx, sorted(vert_idx)) -> (ms_val, new_point)}.
    cache_on : bool, default=True
        Whether to use and update ms_cache.
    tracker : SimplexTracker, optional
        Records bookkeeping events (created simplex, ms recomputed, etc.).

    Returns
    -------
    tri : scipy.spatial.Delaunay
        The Delaunay triangulation of current nodes.
    per_tet : list[dict]
        List of simplex records containing vertices, ms results,
        LB/UB values, best scene, candidate point, and volume.
    """
    pts = np.asarray(nodes, dtype=float)
    if len(pts) < 4:
        return None, []

    if tet_mesh is not None:
        simplices = [list(t) for t in tet_mesh.tets]
        tri = tet_mesh.as_delaunay_like()
    else:
        tri = Delaunay(pts)  # divide into several non-overlapping simplex from pts
        simplices = tri.simplices

    S = len(ms_bundles)

    mins = pts.min(axis=0)

    maxs = pts.max(axis=0)
    diam = float(np.linalg.norm(maxs - mins))
    vol_tol = 1e-12 * max(diam**3, 1.0)

    # per_tet stores information for every simplex in the current iteration
    per_tet = []
    for k, simp in enumerate(simplices):
        idxs = list(map(int, simp))
        verts = [tuple(pts[i]) for i in idxs]

        v0, v1, v2, v3 = np.array(verts)
        vol = abs(np.linalg.det(np.stack([v1 - v0, v2 - v0, v3 - v0], axis=1)) / 6.0)
        if vol < vol_tol:
            continue

        # Use the ordered tuple of vertex index as the unique ID of the simplex
        simplex_id = tuple(sorted(idxs))
        if tracker is not None:
            tracker.note_created(simplex_id)

        fverts_per_scene = [[scen_values[s][i] for i in idxs] for s in range(S)]
        fverts_sum = [sum(fverts_per_scene[s][j] for s in range(S)) for j in range(4)]

        # ==========  per-scene ms + constant-cut solve with cache ==========
        key_base = tuple(sorted(idxs))
        ms_scene = []
        xms_scene = []
        c_scene = []          # c_{T,s}
        cpts_scene = []       # Point (Kp,Ki,Kd) corresponding to c_s
        # For logging: store solve metadata per scene
        ms_meta_per_scene = []
        cs_meta_per_scene = []

        for ω in range(S):
            cache_key = (int(ω), key_base)
            hit = (cache_on and (ms_cache is not None) and (cache_key in ms_cache))

            if hit:
                # cache is now (ms_val, new_pt_ms, c_val, c_pt)
                ms_val, new_pt_ms, c_val, c_pt = ms_cache[cache_key]
                # Cached: no fresh metadata, use None
                ms_meta_per_scene.append(None)
                cs_meta_per_scene.append(None)
            else:
                ms_val, new_pt_ms, c_val, c_pt = ms_on_tetra_for_scene(
                    ms_bundles[ω], verts, fverts_per_scene[ω]
                )
                # Capture solve metadata from the bundle
                _ms_meta = getattr(ms_bundles[ω], 'last_solve_meta', None)
                _cs_meta = getattr(ms_bundles[ω], 'last_cs_meta', None)
                ms_meta_per_scene.append(_ms_meta)
                cs_meta_per_scene.append(_cs_meta)

                # --- TimeLimit logging for MS solve ---
                if _ms_meta and _ms_meta.get("status") == "time_limit":
                    _tl_dual = _ms_meta.get("dual_bound")
                    _tl_prim = _ms_meta.get("primal_obj")
                    if _tl_dual is not None and _tl_prim is not None:
                        _dual_lt_prim = _tl_dual < _tl_prim
                    else:
                        _dual_lt_prim = None
                    _tl_line = (
                        f"[Iter {iter_num}] [MS TimeLimit] simplex_idx={k}, verts={simplex_id}, "
                        f"scenario={ω}, dual_bound={_tl_dual}, primal_obj={_tl_prim}, "
                        f"dual<primal={_dual_lt_prim}\n"
                    )
                    if dbg_timelimit_path is not None:
                        try:
                            with open(dbg_timelimit_path, "a", encoding="utf-8") as _ftl:
                                _ftl.write(_tl_line)
                        except Exception as e:
                            print(f"[timelimit-log] failed to write: {e}")

                # --- TimeLimit logging for CS solve ---
                if _cs_meta and _cs_meta.get("status") == "time_limit":
                    _tl_dual = _cs_meta.get("dual_bound")
                    _tl_prim = _cs_meta.get("primal_obj")
                    if _tl_dual is not None and _tl_prim is not None:
                        _dual_lt_prim = _tl_dual < _tl_prim
                    else:
                        _dual_lt_prim = None
                    _tl_line = (
                        f"[Iter {iter_num}] [CS TimeLimit] simplex_idx={k}, verts={simplex_id}, "
                        f"scenario={ω}, dual_bound={_tl_dual}, primal_obj={_tl_prim}, "
                        f"dual<primal={_dual_lt_prim}\n"
                    )
                    if dbg_timelimit_path is not None:
                        try:
                            with open(dbg_timelimit_path, "a", encoding="utf-8") as _ftl:
                                _ftl.write(_tl_line)
                        except Exception as e:
                            print(f"[timelimit-log] failed to write: {e}")

                # --- MS timing log: record only NON-OPTIMAL MS solves ---
                if dbg_ms_timing_path is not None and _ms_meta is not None and _ms_meta.get('status') != 'optimal':
                    _ms_line = (
                        f"[Iter {iter_num}] scenario={ω}, "
                        f"simplex_idx={k}, verts={simplex_id}, "
                        f"time={_ms_meta.get('time_sec', 0.0):.4f}s, "
                        f"status={_ms_meta.get('status', '?')}, "
                        f"dual_bound={_ms_meta.get('dual_bound')}, "
                        f"primal_obj={_ms_meta.get('primal_obj')}\n"
                    )
                    try:
                        with open(dbg_ms_timing_path, "a", encoding="utf-8") as _fms:
                            _fms.write(_ms_line)
                    except Exception:
                        pass

                if cache_on and (ms_cache is not None):
                    ms_cache[cache_key] = (ms_val, new_pt_ms, c_val, c_pt)

                # --- CS (Q-value) timing log: record only NON-OPTIMAL c_s solves ---
                if dbg_cs_timing_path is not None and _cs_meta is not None and _cs_meta.get('status') != 'optimal':
                    _cs_line = (
                        f"[Iter {iter_num}] scenario={ω}, "
                        f"simplex_idx={k}, verts={simplex_id}, "
                        f"time={_cs_meta.get('time_sec', 0.0):.4f}s, "
                        f"status={_cs_meta.get('status', '?')}, "
                        f"dual_bound={_cs_meta.get('dual_bound')}, "
                        f"primal_obj={_cs_meta.get('primal_obj')}\n"
                    )
                    try:
                        with open(dbg_cs_timing_path, "a", encoding="utf-8") as _fcs:
                            _fcs.write(_cs_line)
                    except Exception:
                        pass
                if tracker is not None:
                    tracker.note_ms_recomputed(simplex_id)

            ms_scene.append(ms_val)
            xms_scene.append(new_pt_ms)
            c_scene.append(c_val)
            cpts_scene.append(c_pt)
        # ============================================


        if MS_AGG == "sum":
            ms_total = float(np.sum(ms_scene))
            c_total  = float(np.sum(c_scene))
        elif MS_AGG == "mean":
            ms_total = float(np.mean(ms_scene))
            c_total  = float(np.mean(c_scene))
        else:
            raise ValueError("MS_AGG must be 'sum' or 'mean'")

        UB = float(np.max(fverts_sum) + ms_total)

        # === NEW: solve true surrogate LB (with cache) ===
        # LB surrogate cache: check cache first to avoid redundant Gurobi solves
        if lb_sur_cache is not None and key_base in lb_sur_cache:
            LB_sur = lb_sur_cache[key_base]
        else:
            LB_sur = solve_surrogate_lb_for_tet(fverts_per_scene, ms_scene, c_scene)
            # LB surrogate cache: store result for future iterations
            if lb_sur_cache is not None:
                lb_sur_cache[key_base] = LB_sur

        best_scene = int(np.argmin(ms_scene))
        x_ms_best = xms_scene[best_scene]

        # === NEW: count infeasible vertices (Q >= 1e5 in any scenario) ===
        # fverts_per_scene[s][j] is Q_s(vertex j)
        # vertex j is infeasible if exists s such that Q_s(vertex j) >= 1e5
        n_infeas_verts = 0
        for j in range(4):
            is_infeas = False
            for s in range(S):
                if fverts_per_scene[s][j] >= 1e5 - 1e-9: # tolerance
                    is_infeas = True
                    break
            if is_infeas:
                n_infeas_verts += 1

        # === NEW: record LB construction components (for diagnosis of LB>UB) ===
        min_f = float(np.min(fverts_sum))
        lb_linear = min_f + ms_total

        # c_total split: raw sum vs finite-only (matching solve_surrogate_lb_for_tet)
        c_arr = np.asarray(c_scene, float)
        finite_c_mask = np.isfinite(c_arr)
        c_total_all = float(np.sum(c_arr))        # can be -inf
        c_total_finite = float(np.sum(c_arr[finite_c_mask])) if np.any(finite_c_mask) else float('nan')

        ms_arr = np.asarray(ms_scene, float)
        all_ms_finite = bool(np.all(np.isfinite(ms_arr)))
        any_finite_c  = bool(np.any(finite_c_mask))

        if all_ms_finite and any_finite_c:
            lb_case = "all_ms_finite"
        elif all_ms_finite and not any_finite_c:
            lb_case = "all_ms_finite_no_c"
        elif not all_ms_finite and any_finite_c:
            lb_case = "some_ms_inf_use_c_only"
        else:
            lb_case = "all_fail"

        lb_terms = {
            "min_fverts_sum": min_f,
            "LB_linear": lb_linear,
            "ms_total": ms_total,
            "c_total_finite": c_total_finite,
            "c_total_all": c_total_all,
            "lb_case": lb_case,
        }

        per_tet.append({
            "simplex_index": k,
            "vert_idx": idxs,
            "verts": verts,
            "fverts_sum": fverts_sum,
            "ms_per_scene": ms_scene,
            "xms_per_scene": xms_scene,
            "c_per_scene":  c_scene,
            "c_point_per_scene": cpts_scene,   
            "ms": ms_total,
            "c_agg": c_total,
            "LB": LB_sur,
            "UB": UB,
            "x_ms_best_scene": x_ms_best,
            "best_scene": best_scene,
            "volume": vol,
            "n_infeas_verts": n_infeas_verts,
            # For logging: per-scene solve metadata
            "ms_meta_per_scene": ms_meta_per_scene,
            "cs_meta_per_scene": cs_meta_per_scene,
            "LB_terms": lb_terms,
        })


    return tri, per_tet

# ------------------------- MAIN LOOP -------------------------
def run_pid_simplex_3d(base_bundles, ms_bundles, model_list, first_vars_list,
                       target_nodes=30, min_dist=MIN_DIST, active_tol=ACTIVE_TOL, verbose=True,
                       agg_bundle=None, gap_stop_tol=GAP_STOP_TOL, tracker: SimplexTracker | None = None,
                       enable_3d_plot: bool = True,
                       plot_every: int | None = None,
                       use_exact_opt: bool = False,
                       exact_solver_name: str = "gurobi",
                       exact_solver_opts: dict | None = None,
                       time_limit: float | None = None,
                       enable_ef_ub: bool = True,
                       ef_time_ub: float = 60.0):
    """
    Starting from the 8 corner nodes, in each iteration:
        - Compute global UB from current nodes (sum over scenarios)
        - Evaluate all simplex(tetrahedra) by evaluate_all_tetra
        - Identify active simplices near the current UB.
        - Determine global LB = UB + ms_b (from best active simplex).
        - Select a new candidate node minimizing ms, subject to min_dist.
        - Update scenario evaluations, nodes, and gap.
        - Stop when UB - LB ≤ gap_stop_tol or candidate collision occurs.

    * if you see verbose, ignore it, it just prints more things...

    Parameters
    ----------
    base_bundles : list[BaseBundle]
        Scenario-specific models for true Q evaluation.
    ms_bundles : list[MSBundle]
        Scenario-specific persistent solvers for ms subproblems.
    model_list : list[pyo.ConcreteModel]
        Original Pyomo models (one per scenario).
    first_vars_list : list[list[pyo.Var]]
        First-stage variable lists for each scenario.
    target_nodes : int, 
        Maximum number of nodes to generate.
    min_dist : float, 
        Minimum allowed distance between nodes.
    active_tol : float, 
        Relaxation tolerance for active simplex filtering.
    verbose : bool, default=True
        Whether to print iteration details.
    agg_bundle : 
        Reserved for aggregated ms solving.
    gap_stop_tol : float, 
        Convergence threshold for optimal rel-gap.
    tracker : SimplexTracker, 
        Tracks created/active simplices and ms recomputations.
    enable_3d_plot : bool, default=True
        Master switch for all 3D plotting. When False, no plots are generated.
    plot_every : int | None, optional
        Draw the 3D plot every n iterations (only used if enable_3d_plot=True).

    Returns
    -------
    dict
        History and results including nodes, LB/UB/ms traces,
        added nodes, and active simplex ratios.
    """

    iter_q_times_detail = []
    per_iter_q_counts = []
    iter_ms_times_detail = []   # [iter][scene] -> list of dt
    per_iter_ms_counts   = []   # [iter] -> int, total ms call count this round


    if tracker is None:
        tracker = SimplexTracker()
    global LAST_DEBUG

    LB_hist, UB_hist, ms_hist, node_count = [], [], [], []
    UB_node_hist, add_node_hist = [], []
    ms_a_hist, ms_b_hist = [], []
    active_ratio_hist = []
    split_kind_hist = []
    selection_reason_hist = [] # NEW: history of selection reasons

    # NEW: per-iteration summary info
    iter_time_hist = []          # Cumulative time (seconds)
    simplex_hist = []            # Total simplices per round
    active_simplex_hist = []     # Active simplices per round
    t_start = perf_counter()  # NEW: Total start time

    ms_ub_active_per_iter = []
    c_hist_per_iter = []

    # NEW: give c for the lb-simplex
    lb_c_agg_hist = []        # Scalars (e.g., sum_s c_{T,s}, only finite ones are considered)
    lb_c_per_scene_hist = []  # Each round generates a list containing the c_per_scene of that singularity.

    # timing info
    timing = {
        "init_Q_time": 0.0,
        "iter_total_time": [],
        "iter_ms_time": [],
        "iter_Q_new_time": [],
        "iter_ms_time_per_scene": [],
        "iter_ms_calls_per_scene": [],
    }

    # === NEW: Monotonic bound tracking ===
    # best_lb_ever: Track the best (highest) LB ever seen - ensures LB is non-decreasing
    best_lb_ever = float('-inf')
    # ub_candidate_library: Store (point, ub_value) tuples from c_s candidates that improved UB
    # This ensures we never "forget" a good UB candidate from previous rounds
    ub_candidate_library = []  # list of (point_tuple, ub_value)
    best_ub_ever = float('inf')  # Track best UB ever seen
    
    S = len(model_list)

    # === TIMING INSTRUMENTATION: pre-loop phases ===
    _phase_times_preloop = {}

    # === Generate initial simplex vertices from variable bounds (8 corners of the bounding box) ===
    _t_phase = perf_counter()
    nodes = corners_from_var_bounds(first_vars_list[0])


    # === Preload the exact optimal solution for plotting===
    true_opt_points = None
    if enable_3d_plot:
        try:
            csv_path = "exact_opt_precomputed.csv"
            if os.path.exists(csv_path):
                rows = []
                with open(csv_path, "r", newline="", encoding="utf-8") as f:
                    reader = csv.DictReader(f)
                    for row in reader:
                        try:
                            s_idx = int(float(row.get("scenario_index", "0")))
                        except Exception:
                            continue
                        # Only plot the scenarios currently in use 
                        
                        if s_idx < S:
                            try:
                                kp = float(row["Kp"])
                                ki = float(row["Ki"])
                                kd = float(row["Kd"])
                            except Exception:
                                continue
                            rows.append((kp, ki, kd))
                if rows:
                    true_opt_points = np.asarray(rows, float)
                    if verbose:
                        print(f"[Precompute] Loaded {len(rows)} exact optima from {csv_path} for plotting.")
            else:
                if verbose:
                    print(f"[Precompute] File {csv_path} not found; true optima will not be plotted.")
        except Exception as e:
            if verbose:
                print(f"[Precompute] Failed to load exact optima for plotting: {e}")
            true_opt_points = None


    # === NEW: build initial tetrahedral mesh once using Delaunay ===
    tet_mesh = SimplexMesh(nodes)
    _phase_times_preloop["corners_and_mesh"] = perf_counter() - _t_phase

    bounds_arr = np.array([[float(v.lb), float(v.ub)] for v in first_vars_list[0]], float)
    diam = float(np.linalg.norm(bounds_arr[:,1] - bounds_arr[:,0]))  # estimate first stage variable dimension size for simplex shape quality check
    min_dist = float(min_dist)

    # cache f_ω(node_i)
    _t_phase = perf_counter()
    scen_values = [[None]*len(nodes) for _ in range(S)]
    t_init_q0 = perf_counter()
    for i, node in enumerate(nodes):
        for ω in range(S):
            scen_values[ω][i] = evaluate_Q_at(base_bundles[ω], first_vars_list[ω], node)
    timing["init_Q_time"] = perf_counter() - t_init_q0
    _phase_times_preloop["vertex_Q_evals"] = perf_counter() - _t_phase
    if verbose:
        print(f"[TIMING] Pre-loop vertex Q evals: {_phase_times_preloop['vertex_Q_evals']:.2f}s  ({len(nodes)} nodes x {S} scenarios = {len(nodes)*S} solves)")

    # ==== NEW: exact optima for validation / plotting ====
    _t_phase = perf_counter()
    exact_points_per_scen = None
    exact_point_agg = None
    if use_exact_opt:
        print("[ExactOpt] Solving per-scenario exact optima (no aggregate model) ...")
        exact_info = compute_exact_optima(
            model_list,
            first_vars_list,
            solver_name=exact_solver_name,
            solver_opts=exact_solver_opts,
        )
        exact_points_per_scen = [rec["K"] for rec in exact_info["per_scenario"]]
        exact_point_agg = exact_info["aggregate"]["K"]  # will be None
        print("[ExactOpt] Done.")
    _phase_times_preloop["exact_optima"] = perf_counter() - _t_phase



    it = 0
    stop_due_to_collision = False
    ms_cache = {}   # (scene_idx, sorted(vert_idx)) -> (ms_val, new_pt_ms, c_val, c_pt)
    lb_sur_cache = {}  # LB surrogate cache: sorted(vert_idx) -> LB_sur (float)

    # === NEW: helper to nudge candidate point slightly into simplex interior (if necessary) ===
    def _snap_feature(cand_pt, rec,
                    tol_vertex=1e-12,
                    tol_edge=1e-12,
                    tol_face=1e-12):
        if rec is None:
            # Theoretically shouldn't happen, but defensive check
            return tuple(map(float, cand_pt)), "interior", None

        verts = np.asarray(rec["verts"], float)       # (4,3)
        vert_idx = list(rec["vert_idx"])             # e.g. [11, 6, 8, 12] global index
        p = np.asarray(cand_pt, float)

        v0, v1, v2, v3 = verts
        M = np.stack([v1 - v0, v2 - v0, v3 - v0], axis=1)
        try:
            rhs = p - v0
            alpha = np.linalg.solve(M, rhs)
        except np.linalg.LinAlgError:
            return tuple(map(float, p)), "interior", None

        lam1, lam2, lam3 = alpha
        lam0 = 1.0 - (lam1 + lam2 + lam3)
        lambdas = np.array([lam0, lam1, lam2, lam3], float)

        small = lambdas < tol_vertex
        n_small = int(small.sum())

        # Vertex: 3 small 1 big, push slightly inside
        if n_small >= 3:
            centroid = verts.mean(axis=0)
            new_p = p + 1e-3 * (centroid - p)
            return tuple(map(float, new_p)), "interior", {"lambdas": lambdas}

        # Edge: 2 small 2 big
        if n_small == 2:
            big_idx = np.where(~small)[0]     # 0..3
            lam_big = lambdas[big_idx]
            lam_big /= lam_big.sum()
            snapped = lam_big[0]*verts[big_idx[0]] + lam_big[1]*verts[big_idx[1]]

            # Map local 0..3 to global node index
            edge_verts_global = (vert_idx[big_idx[0]], vert_idx[big_idx[1]])
            return (tuple(map(float, snapped)),
                    "edge",
                    {"edge_verts": edge_verts_global, "lambdas": lambdas})

        # Face: 1 small 3 big
        if n_small == 1:
            face_idx = np.where(~small)[0]  # Local indices of the three on face
            lam_face = lambdas[face_idx]
            lam_face /= lam_face.sum()
            snapped = (lam_face[0]*verts[face_idx[0]] +
                    lam_face[1]*verts[face_idx[1]] +
                    lam_face[2]*verts[face_idx[2]])

            face_verts_global = tuple(vert_idx[j] for j in face_idx)
            return (tuple(map(float, snapped)),
                    "face",
                    {"face_verts": face_verts_global, "lambdas": lambdas})

        # Other cases: interior point
        return tuple(map(float, p)), "interior", {"lambdas": lambdas}


    t_start = perf_counter()   # NEW: Total start time
    cum_time = 0.0             # NEW: Cumulative time

    # === Create CSV file for incremental logging ===
    # LB/UB per user definition: main columns use per-iteration values, extra columns for monotonic envelopes
    csv_path = "simplex_result.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Time (s)", "# Nodes", "LB", "UB", "Rel. Gap", "Abs. Gap", "best_LB_ever", "best_UB_ever", "LB_in_split", "UB_in_split"])

    # === Create debug log folder and initialize 4 log files (all overwritten on each run) ===
    debug_log_dir = "simplex_method_debug_log"
    os.makedirs(debug_log_dir, exist_ok=True)

    # File 1: Gurobi TimeLimit subproblems
    dbg_timelimit_path = os.path.join(debug_log_dir, "debug_timelimit.txt")
    with open(dbg_timelimit_path, "w", encoding="utf-8") as f:
        f.write("# debug_timelimit.txt — subproblems that hit Gurobi TimeLimit\n")
        f.write("# Format: [Iter N] [MS/CS TimeLimit] simplex_idx=..., verts=..., scenario=..., dual_bound=..., primal_obj=..., dual<primal=...\n\n")

    # File 2: Dual > Primal violations
    dbg_dual_gt_obj_path = os.path.join(debug_log_dir, "debug_dual_gt_obj.txt")
    with open(dbg_dual_gt_obj_path, "w", encoding="utf-8") as f:
        f.write("# debug_dual_gt_obj.txt — subproblems where dual_bound > primal_obj\n")
        f.write("# Format: [Iter N] [MS/CS] simplex_idx=..., verts=..., scenario=..., dual_bound=..., primal_obj=..., gap=...\n\n")

    # File 3: EF (ipopt) solver details
    dbg_ef_solver_path = os.path.join(debug_log_dir, "debug_ef_solver.txt")
    with open(dbg_ef_solver_path, "w", encoding="utf-8") as f:
        f.write("# debug_ef_solver.txt — EF solver details per iteration (dual: ipopt + gurobi)\n")
        f.write("# Each iteration has per-solver lines + winner line\n\n")

    # File 4: LB after split (parent LB, child LBs, INSIDE/OUTSIDE)
    dbg_lb_split_path = os.path.join(debug_log_dir, "debug_lb_after_split.txt")
    with open(dbg_lb_split_path, "w", encoding="utf-8") as f:
        f.write("# debug_lb_after_split.txt — LB analysis after simplex split\n")
        f.write("# Shows: selected simplex LB, all child LBs, new global LB, INSIDE/OUTSIDE\n\n")

    # File 5: CS (Q-value) solve timing
    dbg_cs_timing_path = os.path.join(debug_log_dir, "debug_cs_timing.txt")
    with open(dbg_cs_timing_path, "w", encoding="utf-8") as f:
        f.write("# debug_cs_timing.txt — NON-OPTIMAL c_s (Q-value) solves only\n")
        f.write("# Format: [Iter N] scenario=..., simplex_idx=..., verts=..., time=...s, status=..., dual_bound=..., primal_obj=...\n\n")

    # File 6: MS (ms_on_tetra) solve timing — non-optimal only
    dbg_ms_timing_path = os.path.join(debug_log_dir, "debug_ms_timing.txt")
    with open(dbg_ms_timing_path, "w", encoding="utf-8") as f:
        f.write("# debug_ms_timing.txt — NON-OPTIMAL MS (ms_on_tetra) solves only\n")
        f.write("# Format: [Iter N] scenario=..., simplex_idx=..., verts=..., time=...s, status=..., dual_bound=..., primal_obj=...\n\n")

    # File 7: Q-value (evaluate_Q_at / BaseBundle.eval_at) solve timing
    dbg_q_timing_path = os.path.join(debug_log_dir, "debug_q_timing.txt")
    with open(dbg_q_timing_path, "w", encoding="utf-8") as f:
        f.write("# debug_q_timing.txt — NON-OPTIMAL Q-value (evaluate_Q_at) solves only\n")
        f.write("# Format: [Iter N] scenario=..., point=..., time=...s, status=..., term=..., obj=...\n\n")

    # === Initialize per-iteration diagnostic logger ===
    iter_logger = IterationLogger(path="simplex_result.txt")
    # UB provenance tracking state (note: logger also stores this, but we track here for clarity)
    ub_updated_this_iter = False
    ub_source_current = "unknown"
    ub_simplex_id_current = None
    ub_origin_iter_current = None
    
    # LB/UB per user definition: incumbent UB (not reset per iteration)
    UB_incumbent = float("inf")
    
    # LB/UB per user definition: initialize selected_simplex_id for logging
    selected_simplex_id = None

    # === EF Upper Bounder initialization ===
    _t_phase = perf_counter()
    # EXPERIMENTAL: dual-solver EF — ipopt (local) + gurobi (global)
    ef_ub_ipopt = None
    ef_ub_gurobi = None
    if enable_ef_ub:
        # --- ipopt instance (original) ---
        try:
            ef_ub_ipopt = SimplexEFUb(
                model_list, first_vars_list,
                time_ub=ef_time_ub,
                solver_name="ipopt",
            )
            if verbose:
                print(f"[EF-UB] Initialized ipopt EF with {S} scenarios, time_ub={ef_time_ub}s")
        except Exception as e:
            print(f"[EF-UB] WARNING: Failed to build ipopt EF model: {e}")
            ef_ub_ipopt = None
        # --- gurobi instance (experimental, controlled by ENABLE_EF_GUROBI) ---
        if ENABLE_EF_GUROBI:
            try:
                ef_ub_gurobi = SimplexEFUb(
                    model_list, first_vars_list,
                    time_ub=ef_time_ub,
                    solver_name="gurobi",
                )
                if verbose:
                    print(f"[EF-UB] Initialized gurobi EF with {S} scenarios, time_ub={ef_time_ub}s")
            except Exception as e:
                print(f"[EF-UB] WARNING: Failed to build gurobi EF model: {e}")
                ef_ub_gurobi = None
        if ef_ub_ipopt is None and ef_ub_gurobi is None:
            enable_ef_ub = False
    _phase_times_preloop["ef_init"] = perf_counter() - _t_phase

    if verbose:
        print("\n" + "="*70)
        print("[TIMING] === Pre-loop phase summary ===")
        for _pname, _ptime in _phase_times_preloop.items():
            print(f"  {_pname:30s} : {_ptime:10.2f}s")
        print(f"  {'TOTAL':30s} : {sum(_phase_times_preloop.values()):10.2f}s")
        print("="*70 + "\n")

    # LB/UB per user definition: compute initial UB from mesh vertices BEFORE the loop
    # This initializes UB_incumbent from the initial vertex evaluations
    if len(nodes) >= 4:
        f_sum_initial = [
            sum(scen_values[s][i] for s in range(S))
            for i in range(len(nodes))
        ]
        UB_incumbent = float(min(f_sum_initial))
        # Provenance logging: initial UB from vertices before loop
        iter_logger.update_ub_provenance(updated=True, source="init", simplex_id=None, origin_iter=-1)


    # === TIMING INSTRUMENTATION: per-iteration phase dict ===
    _iter_phase_times = []  # list of dicts, one per iteration


    while len(nodes) < target_nodes:
        t_iter0 = perf_counter()
        _phases = {}  # timing for this iteration
        tracker.start_iter(it)

        # initialize timing slots for this iteration
        timing["iter_total_time"].append(0.0)
        timing["iter_ms_time"].append(0.0)
        timing["iter_Q_new_time"].append(0.0)
        timing_idx = len(timing["iter_total_time"]) - 1

        ms_prev_calls = [len(b.solve_time_hist) for b in ms_bundles]

        # === Reset iteration-specific tracking ===
        ub_updated_this_iter = False
        ub_source_this_iter = None  # will track source if UB updates
        ub_simplex_id_this_iter = None
        
        # LB/UB per user definition: track simplex count before split
        n_simplices_before = len(tet_mesh.tets)

        # 1) Global UB
        _t_phase = perf_counter()
        f_sum_per_node = [
            sum(scen_values[s][i] for s in range(S))
            for i in range(len(nodes))
        ]
        ub_idx = int(np.argmin(f_sum_per_node))
        UB_vertex = float(f_sum_per_node[ub_idx])
        # LB/UB per user definition: Initialize UB_incumbent from vertex UB only on first iter
        if it == 0:
            UB_incumbent = min(UB_incumbent, UB_vertex)
        # LB/UB per user definition: UB_global tracks incumbent, NOT reset to UB_vertex each iter
        UB_global = UB_incumbent
        UB_node = tuple(nodes[ub_idx])
        
        # Track vertex as initial UB source for this iteration (debug only)
        vertex_ub_simplex_id = None  # vertex is not tied to a specific simplex for now

        _phases["1_vertex_UB"] = perf_counter() - _t_phase

        # 2) Evaluate all tetrahedrons (single scene)
        _t_phase = perf_counter()

        # === Iter-0 speedup: use loose MIPGap for coarse initial mesh ===
        _iter0_mipgap_overridden = False
        _iter0_orig_mipgaps = []
        ITER0_MIPGAP = 1e-1  # Loose gap for the first iteration
        if it == 0:
            for b in ms_bundles:
                orig_gap = getattr(b, 'mip_gap', 1e-1)
                _iter0_orig_mipgaps.append(orig_gap)
                b.gp.set_gurobi_param('MIPGap', ITER0_MIPGAP)
            _iter0_mipgap_overridden = True
            if verbose:
                print(f"[Iter 0] MIPGap overridden to {ITER0_MIPGAP} "
                      f"(original: {_iter0_orig_mipgaps[0] if _iter0_orig_mipgaps else 'N/A'})")

        t_ms0 = perf_counter()
        tri, per_tet = evaluate_all_tetra(
            nodes, scen_values, ms_bundles, first_vars_list,
            ms_cache=ms_cache, cache_on=True, tracker=tracker,
            tet_mesh=tet_mesh,          # === NEW: use incremental mesh
            lb_sur_cache=lb_sur_cache,  # LB surrogate cache
            dbg_timelimit_path=dbg_timelimit_path,
            dbg_cs_timing_path=dbg_cs_timing_path,
            dbg_ms_timing_path=dbg_ms_timing_path,
            iter_num=it,
        )
        t_ms = perf_counter() - t_ms0
        timing["iter_ms_time"][timing_idx] = t_ms


        # === Restore original MIPGap after iter-0 ===
        if _iter0_mipgap_overridden:
            for b, orig_gap in zip(ms_bundles, _iter0_orig_mipgaps):
                b.gp.set_gurobi_param('MIPGap', orig_gap)
            if verbose:
                print(f"[Iter 0] MIPGap restored to {_iter0_orig_mipgaps[0] if _iter0_orig_mipgaps else 'N/A'}")

        # NEW: record c_per_scene
        iter_c_records = []
        for r in per_tet:
            c_list = [float(x) for x in r.get("c_per_scene", [])]
            iter_c_records.append({
                "simplex_index": int(r["simplex_index"]),
                "vert_idx": list(map(int, r["vert_idx"])),
                "c_per_scene": c_list,
            })
        c_hist_per_iter.append(iter_c_records)

        # —— new: record ms.solve() time and call times ——
        per_scene_times = []
        per_scene_calls = []
        for s, b in enumerate(ms_bundles):
            new_times = b.solve_time_hist[ms_prev_calls[s]:]
            per_scene_times.append(sum(new_times))
            per_scene_calls.append(len(new_times))
        timing["iter_ms_time_per_scene"].append(per_scene_times)
        timing["iter_ms_calls_per_scene"].append(per_scene_calls)

        # Extra record: list of "time per ms call" for each scenario this round
        iter_ms_times_detail.append([
            list(ms_bundles[s].solve_time_hist[ms_prev_calls[s]:])
            for s in range(len(ms_bundles))
        ])
        per_iter_ms_counts.append(sum(per_scene_calls))


        _phases["2_evaluate_all_tetra"] = perf_counter() - _t_phase

        # 3) active mask (Filter by UB + Shape Quality)
        _t_phase = perf_counter()
        active_mask = {
            r["simplex_index"]: (r["LB"] <= UB_global + active_tol)
            for r in per_tet
        }
        q_cut = 0
        bad_quality_count = 0      # Number of simplices removed due to quality this round

        for r in per_tet:
            sid = r["simplex_index"]
            if not active_mask.get(sid, False):
                continue
            q = tet_quality(r["verts"])
            r["quality"] = q       # 顺便把质量存进去，plot里也可以用
            if q < q_cut:
                active_mask[sid] = False
                bad_quality_count += 1

        # --- Bad-shape filter: small volume + high aspect ratio => inactive ---
        bad_shape_count = 0
        _shape_checked = 0   # how many simplices had vol < threshold (shape check triggered)
        for r in per_tet:
            sid = r["simplex_index"]
            if not active_mask.get(sid, False):
                continue
            vol = r.get("volume")
            if vol is None:
                continue  # skip if volume not available
            if vol < SMALL_VOL_TOL_ABS:
                _shape_checked += 1
                verts = r.get("verts")
                if verts is None or len(verts) < 2:
                    if verbose:
                        print(f"[Iter {it}] WARNING: simplex {sid} missing verts, "
                              f"cannot check aspect ratio")
                    continue
                try:
                    max_e, min_e, aspect = compute_edge_aspect(verts)
                except Exception:
                    continue
                if aspect >= ASPECT_BAD_TOL:
                    active_mask[sid] = False
                    bad_shape_count += 1
                    # Store diagnostics on the record
                    r["inactive_reason"] = "bad_shape_aspect"
                    r["max_edge"] = max_e
                    r["min_edge"] = min_e
                    r["aspect"] = aspect
                    # Log per-simplex detail to debug_lb_after_split.txt
                    _inact_line = (
                        f"  Inactivate simplex {sid}: vol={vol:.3e} < 1e-4, "
                        f"max_edge={max_e:.3e}, min_edge={min_e:.3e}, "
                        f"aspect={aspect:.3e}\n"
                    )
                    try:
                        with open(dbg_lb_split_path, "a", encoding="utf-8") as _fbs:
                            _fbs.write(_inact_line)
                    except Exception:
                        pass
                    if verbose:
                        print(f"[Iter {it}] {_inact_line.strip()}")

        # --- Write per-round shape-check summary to debug_lb_after_split.txt ---
        try:
            with open(dbg_lb_split_path, "a", encoding="utf-8") as _fbs:
                _fbs.write(
                    f"[Iter {it}] Shape check: triggered={_shape_checked}, "
                    f"inactivated={bad_shape_count}, "
                    f"passed={_shape_checked - bad_shape_count}\n"
                )
        except Exception:
            pass

        # === Count simplices and active ones this round ===
        n_tets = len(per_tet)
        n_active = sum(1 for r in per_tet if active_mask.get(r["simplex_index"], False))
        if verbose:
            print(f"[Iter {it}] #simplices = {n_tets}, #active = {n_active}, "
                  f"bad-quality cut = {bad_quality_count}, bad-shape cut = {bad_shape_count}")
            
        # === Per-iter tables: active vs inactive simplices (LB, UB, sum ms, sum c_s) ===
        if verbose:
            def _print_lb_ub_ms_c_table(records, title):
                print(f"== [Iter {it}] {title} simplices: LB / UB / sum(ms_s) / sum(c_s) (per-scenario) ==")
                if not records:
                    print("  (none)")
                    return
                # Sort by simplex_index for easy comparison
                records = sorted(records, key=lambda rr: rr["simplex_index"])
                header = ["simp", "LB", "UB", "sum_ms", "sum_c_s", "Range(As)", "Range(As+ms)", "#InfV"]
                colw = [8, 18, 18, 18, 18, 25, 25, 6]

                def fmt_row(cols):
                    return (
                        f"{str(cols[0]).ljust(colw[0])}"
                        f"{str(cols[1]).rjust(colw[1])}"
                        f"{str(cols[2]).rjust(colw[2])}"
                        f"{str(cols[3]).rjust(colw[3])}"
                        f"{str(cols[4]).rjust(colw[4])}"
                        f"{str(cols[5]).rjust(colw[5])}"
                        f"{str(cols[6]).rjust(colw[6])}"
                        f"{str(cols[7]).rjust(colw[7])}"
                    )

                print(fmt_row(header))
                print("-" * sum(colw))

                for rr in records:
                    simp_id = f"T{int(rr['simplex_index'])}"
                    # Divide by number of scenarios
                    LB_val  = float(rr["LB"]) / S
                    UB_val  = float(rr["UB"]) / S
                    ms_val  = float(rr.get("ms", float("nan")))
                    c_agg   = float(rr.get("c_agg", float("nan")))

                    fverts_sum = rr.get("fverts_sum", [])
                    if fverts_sum:
                        min_As = min(fverts_sum)
                        max_As = max(fverts_sum)
                        rng_As = f"[{min_As:.2e}, {max_As:.2e}]"

                        min_As_ms = min_As + ms_val
                        max_As_ms = max_As + ms_val
                        rng_As_ms = f"[{min_As_ms:.2e}, {max_As_ms:.2e}]"
                    else:
                        rng_As = "N/A"
                        rng_As_ms = "N/A"
                    
                    n_inf = rr.get("n_infeas_verts", 0)

                    row = [
                        simp_id,
                        f"{LB_val:.6e}",
                        f"{UB_val:.6e}",
                        f"{ms_val:.6e}",
                        f"{c_agg:.6e}",
                        rng_As,
                        rng_As_ms,
                        n_inf,
                    ]
                    print(fmt_row(row))
                print()

            active_recs = [r for r in per_tet if active_mask.get(r["simplex_index"], False)]
            inactive_recs = [r for r in per_tet if not active_mask.get(r["simplex_index"], False)]

            _print_lb_ub_ms_c_table(active_recs,   "ACTIVE")
            _print_lb_ub_ms_c_table(inactive_recs, "INACTIVE")




        # 4) active ratio
        total_vol = sum(r["volume"] for r in per_tet)
        active_vol = sum(r["volume"] for r in per_tet if active_mask[r["simplex_index"]])
        active_ratio = active_vol / total_vol if total_vol > 0 else 0.0

        # NEW: Number of simplices and active simplices this round
        num_simplices = len(per_tet)
        num_active_simplices = sum(
            1 for r in per_tet if active_mask.get(r["simplex_index"], False)
        )

        # collect statistics of active simplices (active / active+UB)
        for r in per_tet:
            is_active = active_mask.get(r["simplex_index"], False)
            if not is_active:
                continue
            simplex_id = tuple(sorted(r["vert_idx"]))
            has_ub = (ub_idx in r["vert_idx"])
            tracker.note_active(simplex_id, has_ub=has_ub)

        # print iteration statistics immediately
        tracker.end_iter()

        _phases["3_active_filter_print"] = perf_counter() - _t_phase

        # 5) LB_global
        _t_phase = perf_counter()
        active_LBs = [r["LB"] for r in per_tet if active_mask.get(r["simplex_index"], False)]
        if active_LBs:
            LB_global = float(min(active_LBs))
            lb_simp_rec = min(
                (r for r in per_tet if active_mask[r["simplex_index"]]),
                key=lambda r: r["LB"]
            )
        else:
            LB_global = float(min(r["LB"] for r in per_tet))
            lb_simp_rec = min(per_tet, key=lambda r: r["LB"])

        # LB/UB per user definition: best_lb_ever update disabled here, done AFTER end-of-iter
        # best_lb_ever = max(best_lb_ever, LB_global)  # DISABLED

        # LB/UB per user definition: DISABLED - UB candidates only from NEW simplices (end-of-iter)
        # The following avg_cs and library reuse blocks are disabled per user spec.
        # UB is now computed ONLY from new simplices after split.
        
        # # NEW: avg c_s UB candidate (DISABLED per user spec)
        # # Collect finite c_pts from lb_simp_rec, compute average, evaluate true objective
        # c_pts_list = lb_simp_rec.get("c_point_per_scene", [])
        # valid_c_pts = [
        #     pt for pt in c_pts_list
        #     if pt is not None and all(math.isfinite(v) for v in pt)
        # ]
        # if valid_c_pts:
        #     x_avg = tuple(np.mean(valid_c_pts, axis=0))
        #     # Evaluate TRUE objective at x_avg for each scenario
        #     ub_avg_vals = [
        #         evaluate_Q_at(base_bundles[s], first_vars_list[s], x_avg)
        #         for s in range(S)
        #     ]
        #     ub_avg_val = float(sum(ub_avg_vals))  # same aggregation as UB_global
        #     # Update UB if this improves it AND doesn't violate best_lb_ever
        #     if ub_avg_val < UB_global and ub_avg_val >= best_lb_ever - 1e-8:
        #         UB_global = ub_avg_val
        #         UB_node = x_avg
        #         ub_idx = None  # Mark that UB is not from a mesh node
        #         # === NEW: Add this candidate to the library so it's never forgotten ===
        #         ub_candidate_library.append((x_avg, ub_avg_val))
        #         # Track UB provenance
        #         ub_updated_this_iter = True
        #         ub_source_this_iter = "avg_cs"
        #         ub_simplex_id_this_iter = tuple(sorted(lb_simp_rec["vert_idx"]))
        #         if verbose:
        #             print(f"[Iter {it}] NEW: avg c_s UB candidate improved UB: {ub_avg_val:.6e} (added to library)")

        # # === NEW: Check all candidates in the library to ensure UB never increases === (DISABLED per user spec)
        # # IMPORTANT: Only use library entries that are still valid (>= best_lb_ever)
        # for lib_pt, lib_ub_val in ub_candidate_library:
        #     if lib_ub_val < UB_global and lib_ub_val >= best_lb_ever - 1e-8:
        #         UB_global = lib_ub_val
        #         UB_node = lib_pt
        #         ub_idx = None
        #         # Track UB provenance (library reuse)
        #         ub_updated_this_iter = True
        #         ub_source_this_iter = "library_reuse"
        #         ub_simplex_id_this_iter = None

        # LB/UB per user definition: best_ub_ever update disabled here, done AFTER end-of-iter
        # best_ub_ever = min(best_ub_ever, UB_global)  # DISABLED

        # LB/UB per user definition: UB provenance updated ONLY at end-of-iter, not here
        # (Removed pre-split provenance updates per user spec)

        # ======= Consistency check: selected simplex LB(=LB_global) should not exceed UB_global =======
        # Theoretically surrogate is underestimator, so LB_global <= UB_global should always hold
        # Add small tolerance to prevent pure numerical error
        if LB_global > UB_global + 1e-8:
            print(
                f"[WARNING iter {it}] LB exceeds UB! "
                f"simplex={lb_simp_rec['simplex_index']}, "
                f"LB={LB_global:.6e}, UB={UB_global:.6e}, "
                f"gap={LB_global - UB_global:.6e}  (continuing for diagnostics)"
            )
        # =======================================================================





        ms_b      = float(lb_simp_rec["ms"])
        ms_b_simp = int(lb_simp_rec["simplex_index"])

        lb_simp_idx = int(lb_simp_rec["simplex_index"]) 
        _phases["4_LB_selection"] = perf_counter() - _t_phase

        # ==================================================
        # === EF Upper Bounder solve on selected simplex ===
        # ==================================================
        _t_phase = perf_counter()
        # EXPERIMENTAL: dual-solver EF — run ipopt then gurobi, pick best
        ef_iter_info = {
            "ef_attempted": False,
            "ef_ok": False,
            "solver_status": None,
            "termination_condition": None,
            "ef_time_sec": 0.0,
            "K_ef": None,
            "ef_obj": None,
            "true_obj": None,
            "ub_updated_by_ef": False,
        }
        # Store per-solver results for logging
        _ef_dual_results = {}  # solver_name -> {ok, K, ef_obj, true_obj, status, term, time}

        if enable_ef_ub:
            ef_iter_info["ef_attempted"] = True
            verts4 = [tuple(map(float, v)) for v in lb_simp_rec["verts"]]
            c_pts = lb_simp_rec.get("c_point_per_scene", [])

            for _solver_tag, _ef_inst in [("ipopt", ef_ub_ipopt), ("gurobi", ef_ub_gurobi)]:
                if _ef_inst is None:
                    _ef_dual_results[_solver_tag] = {"ok": False, "status": "not_initialized"}
                    continue
                try:
                    _ef_inst.update_simplex_vertices(verts4)
                    if c_pts:
                        _ef_inst.set_warm_start(c_pts)
                    _ef_ok, _K_ef, _ef_obj, _ef_info = _ef_inst.solve()
                    _res = {
                        "ok": _ef_ok,
                        "K": _K_ef,
                        "ef_obj": _ef_obj,
                        "true_obj": None,
                        "status": _ef_info.get("solver_status"),
                        "term": _ef_info.get("termination_condition"),
                        "time": _ef_info.get("time_sec", 0.0),
                        "lower_bound": _ef_info.get("lower_bound"),  # Gurobi LB
                    }
                    if _ef_ok and _K_ef is not None:
                        _true = evaluate_true_expected_obj(base_bundles, first_vars_list, _K_ef)
                        _res["true_obj"] = _true
                    _ef_dual_results[_solver_tag] = _res
                    if verbose:
                        _tstr = f"{_res['true_obj']/S:.6e}" if _res['true_obj'] is not None else "N/A"
                        print(f"[Iter {it}] EF-{_solver_tag}: ok={_ef_ok}, "
                              f"ef_obj={_ef_obj}, true_obj={_tstr}, time={_res['time']:.3f}s")
                except Exception as e:
                    _ef_dual_results[_solver_tag] = {"ok": False, "status": "exception", "term": str(e)}
                    if verbose:
                        print(f"[Iter {it}] EF-{_solver_tag} exception: {e}")

            # Pick the best (lowest true_obj) among successful solvers
            _best_solver = None
            _best_true = float('inf')
            for _stag, _sres in _ef_dual_results.items():
                if _sres.get("ok") and _sres.get("true_obj") is not None:
                    if _sres["true_obj"] < _best_true:
                        _best_true = _sres["true_obj"]
                        _best_solver = _stag

            if _best_solver is not None:
                _bres = _ef_dual_results[_best_solver]
                ef_iter_info["ef_ok"] = True
                ef_iter_info["solver_status"] = f"{_best_solver}:{_bres.get('status')}"
                ef_iter_info["termination_condition"] = f"{_best_solver}:{_bres.get('term')}"
                ef_iter_info["ef_time_sec"] = sum(
                    r.get("time", 0.0) for r in _ef_dual_results.values() if isinstance(r.get("time"), (int, float))
                )
                ef_iter_info["K_ef"] = _bres["K"]
                ef_iter_info["ef_obj"] = _bres["ef_obj"]
                ef_iter_info["true_obj"] = _bres["true_obj"]

                if _best_true < UB_incumbent:
                    UB_incumbent = _best_true
                    UB_global = UB_incumbent
                    UB_node = _bres["K"]
                    ub_updated_this_iter = True
                    ub_source_this_iter = f"EF_{_best_solver}"
                    ub_simplex_id_this_iter = tuple(sorted(lb_simp_rec["vert_idx"]))
                    ef_iter_info["ub_updated_by_ef"] = True
                    if verbose:
                        print(f"[Iter {it}] EF-UB improved by {_best_solver}: "
                              f"{_best_true/S:.6e} (per-scen), K={_bres['K']}")
            else:
                # Both solvers failed
                _any = next(iter(_ef_dual_results.values()), {})
                ef_iter_info["solver_status"] = _any.get("status", "all_failed")
                ef_iter_info["termination_condition"] = _any.get("term", "all_failed")
                if verbose:
                    print(f"[Iter {it}] EF-UB: both solvers failed")

            # Build EF/MS log row (deferred: will be written at end-of-iteration
            # so LB_global_end_sum and UB_global_end_sum reflect post-split values)
            try:
                ms_issues = collect_ms_cs_issues(lb_simp_rec)
                _ef_log_row = {
                    "iter": it,
                    "simplex_id": tuple(sorted(lb_simp_rec["vert_idx"])),
                    "ef_attempted": ef_iter_info["ef_attempted"],
                    "ef_ok": ef_iter_info["ef_ok"],
                    "solver_status": ef_iter_info["solver_status"],
                    "termination_condition": ef_iter_info["termination_condition"],
                    "ef_time_sec": f"{ef_iter_info['ef_time_sec']:.3f}",
                    "K_ef": ef_iter_info["K_ef"],
                    "ef_obj": ef_iter_info["ef_obj"],
                    "true_obj": ef_iter_info["true_obj"],
                    "ub_updated": ef_iter_info["ub_updated_by_ef"],
                    "UB_incumbent": f"{UB_incumbent/S:.9f}",
                    # LB construction terms (from pre-split selected simplex)
                    "lb_simp_LB_sur": f"{lb_simp_rec['LB']:.6f}",
                    "lb_simp_LB_linear": f"{lb_simp_rec.get('LB_terms',{}).get('LB_linear', 0.0):.6f}",
                    "lb_simp_c_total_finite": f"{lb_simp_rec.get('LB_terms',{}).get('c_total_finite', float('nan')):.6f}",
                    "lb_simp_c_total_all": f"{lb_simp_rec.get('LB_terms',{}).get('c_total_all', 0.0):.6f}",
                    "lb_simp_ms_total": f"{lb_simp_rec.get('LB_terms',{}).get('ms_total', 0.0):.6f}",
                    "lb_simp_min_fverts_sum": f"{lb_simp_rec.get('LB_terms',{}).get('min_fverts_sum', 0.0):.6f}",
                    "lb_simp_lb_case": lb_simp_rec.get('LB_terms',{}).get('lb_case', ''),
                    # Placeholders — filled at end-of-iteration
                    "LB_global_end_sum": "",
                    "UB_global_end_sum": "",
                }
                _ef_log_row.update(ms_issues)
            except Exception:
                _ef_log_row = None

        else:
            # EF disabled/unavailable — still build deferred log row
            try:
                ms_issues = collect_ms_cs_issues(lb_simp_rec)
                _ef_log_row = {
                    "iter": it,
                    "simplex_id": tuple(sorted(lb_simp_rec["vert_idx"])),
                    "ef_attempted": ef_iter_info["ef_attempted"],
                    "ef_ok": ef_iter_info["ef_ok"],
                    "solver_status": ef_iter_info["solver_status"],
                    "termination_condition": ef_iter_info["termination_condition"],
                    "ef_time_sec": "",
                    "K_ef": "",
                    "ef_obj": "",
                    "true_obj": "",
                    "ub_updated": "",
                    "UB_incumbent": f"{UB_incumbent/S:.9f}",
                    # LB construction terms (from pre-split selected simplex)
                    "lb_simp_LB_sur": f"{lb_simp_rec['LB']:.6f}",
                    "lb_simp_LB_linear": f"{lb_simp_rec.get('LB_terms',{}).get('LB_linear', 0.0):.6f}",
                    "lb_simp_c_total_finite": f"{lb_simp_rec.get('LB_terms',{}).get('c_total_finite', float('nan')):.6f}",
                    "lb_simp_c_total_all": f"{lb_simp_rec.get('LB_terms',{}).get('c_total_all', 0.0):.6f}",
                    "lb_simp_ms_total": f"{lb_simp_rec.get('LB_terms',{}).get('ms_total', 0.0):.6f}",
                    "lb_simp_min_fverts_sum": f"{lb_simp_rec.get('LB_terms',{}).get('min_fverts_sum', 0.0):.6f}",
                    "lb_simp_lb_case": lb_simp_rec.get('LB_terms',{}).get('lb_case', ''),
                    # Placeholders — filled at end-of-iteration
                    "LB_global_end_sum": "",
                    "UB_global_end_sum": "",
                }
                _ef_log_row.update(ms_issues)
            except Exception:
                _ef_log_row = None

        # === Per-scenario bound debug: log dual > primal violations ===
        try:
            ms_metas = lb_simp_rec.get("ms_meta_per_scene", [])
            cs_metas = lb_simp_rec.get("cs_meta_per_scene", [])
            _simplex_id_str = str(tuple(sorted(lb_simp_rec["vert_idx"])))
            for s in range(S):
                # MS dual > primal check
                mm = ms_metas[s] if s < len(ms_metas) and ms_metas[s] is not None else {}
                ms_lb = mm.get("dual_bound")
                ms_pri = mm.get("primal_obj")
                if (ms_lb is not None and ms_pri is not None
                        and math.isfinite(ms_lb) and math.isfinite(ms_pri)
                        and ms_lb > ms_pri + 1e-8):
                    _gap = ms_lb - ms_pri
                    _line = (f"[Iter {it}] [MS] simplex_idx={lb_simp_rec['simplex_index']}, "
                             f"verts={_simplex_id_str}, scenario={s}, "
                             f"dual_bound={ms_lb}, primal_obj={ms_pri}, gap={_gap}\n")
                    with open(dbg_dual_gt_obj_path, "a", encoding="utf-8") as _fdgo:
                        _fdgo.write(_line)
                # CS dual > primal check
                cm = cs_metas[s] if s < len(cs_metas) and cs_metas[s] is not None else {}
                cs_lb = cm.get("dual_bound")
                cs_pri = cm.get("primal_obj")
                if (cs_lb is not None and cs_pri is not None
                        and math.isfinite(cs_lb) and math.isfinite(cs_pri)
                        and cs_lb > cs_pri + 1e-8):
                    _gap = cs_lb - cs_pri
                    _line = (f"[Iter {it}] [CS] simplex_idx={lb_simp_rec['simplex_index']}, "
                             f"verts={_simplex_id_str}, scenario={s}, "
                             f"dual_bound={cs_lb}, primal_obj={cs_pri}, gap={_gap}\n")
                    with open(dbg_dual_gt_obj_path, "a", encoding="utf-8") as _fdgo:
                        _fdgo.write(_line)
        except Exception:
            pass  # bound debug must never crash the algorithm


        # === NEW: record c info for the simplex ===
        c_scene = [float(x) for x in lb_simp_rec.get("c_per_scene", [])]
        finite_c = [x for x in c_scene if math.isfinite(x)]
        if finite_c:
            c_agg = float(sum(finite_c))
        else:
            c_agg = float('-inf')

        lb_c_agg_hist.append(c_agg)
        lb_c_per_scene_hist.append(c_scene)

        '''
        ms_ub_active_this_iter = []
        for r in ub_active:
            ms_scene = r.get("ms_per_scene", None)
            if ms_scene is None:
                continue
            ms_ub_active_this_iter.append([float(x) for x in ms_scene])
        ms_ub_active_per_iter.append(ms_ub_active_this_iter)
        '''


        # 6) ms_a
        # ms_a: smallest ms among all active simplices (best local improvement)
        if any(active_mask.values()):
            ms_a = float(min(r["ms"] for r in per_tet if active_mask[r["simplex_index"]]))
        else:
            ms_a = float(min(r["ms"] for r in per_tet))
        ms_iter = ms_a


        # === Print the current round's optimality gap (PRE-SPLIT, not final) ===
        # Note: This is before end-of-iter update; final gap printed later
        gap_abs = float(UB_global - LB_global)
        gap_pct = (gap_abs / (abs(UB_global) + 1e-16)) * 100.0
        if verbose:
            print(f"[Iter {it}] Pre-split gap: {gap_abs:.6e} ({gap_pct:.3f}%)")

        # 7) record
        LB_hist.append(LB_global)
        UB_hist.append(UB_global)
        ms_hist.append(ms_iter)
        node_count.append(len(nodes))
        UB_node_hist.append(UB_node)
        ms_a_hist.append(ms_a)
        ms_b_hist.append(ms_b)
        active_ratio_hist.append(active_ratio)

        # NEW: time & simplex stats for this iteration
        now = perf_counter()
        iter_time_hist.append(now - t_start)

        num_simplices = len(per_tet)
        num_active_simplices = sum(
            1 for r in per_tet if active_mask.get(r["simplex_index"], False)
        )
        simplex_hist.append(num_simplices)
        active_simplex_hist.append(num_active_simplices)
        # NEW: Record split type this round
        # Assuming SimplexMesh instance is named mesh, change if named otherwise
        kind = getattr(tet_mesh, "last_split_kind", None)  # None means no split this round
        if kind is None:
            kind = "none"
        split_kind_hist.append(kind)

        # LB/UB per user definition: CSV writing moved to AFTER end-of-iter block



        # LB/UB per user definition: Stop conditions moved to AFTER end-of-iter block

        # 8) print
        # Geometric containment: find which simplex(es) contain UB_node
        _ub_pt = np.asarray(UB_node, dtype=float)
        _geom_containing = []
        _geom_tol = 1e-8
        for _r in per_tet:
            _V = np.array(_r["verts"], dtype=float)  # (4, 3)
            # Barycentric: p = V[0] + T @ lam_123, where T = [V1-V0, V2-V0, V3-V0]^T
            _T = (_V[1:] - _V[0]).T  # (3, 3)
            try:
                _lam123 = np.linalg.solve(_T, _ub_pt - _V[0])
                _lam0 = 1.0 - _lam123.sum()
                _lam_all = np.array([_lam0, *_lam123])
                if np.all(_lam_all >= -_geom_tol):
                    _geom_containing.append((_r["simplex_index"], _lam_all))
            except np.linalg.LinAlgError:
                pass  # degenerate simplex, skip

        if verbose:
            print(f"[Iter {it}] Active simplex ratio = {active_ratio:.6f}")
            if _geom_containing:
                _simp_ids = [s for s, _ in _geom_containing]
                print(f"[Iter {it}] UB point {UB_node} is geometrically in simplices {_simp_ids}")
                for _si, _lam in _geom_containing:
                    _r_match = [r for r in per_tet if r["simplex_index"] == _si][0]
                    print(f"  T{_si} verts={_r_match['vert_idx']}: "
                          f"λ=[{', '.join(f'{l:.6f}' for l in _lam)}], "
                          f"LB={_r_match['LB']:.9f}, c_s={_r_match['c_per_scene']}")

                # === CS dual vs primal diagnostic for simplices containing UB point ===
                print(f"\n== [Iter {it}] CS DUAL vs PRIMAL diagnostic (simplices containing UB point) ==")
                for _si, _lam in _geom_containing:
                    _r_match = [r for r in per_tet if r["simplex_index"] == _si][0]
                    _cs_metas = _r_match.get("cs_meta_per_scene", [])
                    _cs_pts   = _r_match.get("c_point_per_scene", [])
                    print(f"  --- T{_si} ---")
                    for _s_idx in range(S):
                        _meta = _cs_metas[_s_idx] if _s_idx < len(_cs_metas) else {}
                        if _meta is None:
                            _meta = {}
                        _dual = _meta.get("dual_bound", None)
                        _prim = _meta.get("primal_obj", None)
                        _status = _meta.get("status", "?")
                        _cs_pt = _cs_pts[_s_idx] if _s_idx < len(_cs_pts) else None

                        _dual_str = f"{_dual:.9f}" if _dual is not None else "N/A"
                        _prim_str = f"{_prim:.9f}" if _prim is not None else "N/A"
                        _check = ""
                        if _dual is not None and _prim is not None:
                            if _dual > _prim + 1e-8:
                                _check = "  ⚠ DUAL > PRIMAL (Gurobi inconsistency!)"
                            else:
                                _check = f"  ✓ dual ≤ primal (gap={_prim - _dual:.6e})"

                        print(f"    scen {_s_idx}: status={_status}, dual={_dual_str}, primal={_prim_str}{_check}")
                        print(f"             cs_primal_K={_cs_pt}")

                        # Independent Q evaluation at the CS primal point
                        if _cs_pt is not None:
                            try:
                                _indep_Q = evaluate_Q_at(base_bundles[_s_idx], first_vars_list[_s_idx], _cs_pt)
                                _indep_str = f"{_indep_Q:.9f}"
                                _verdict = ""
                                if _dual is not None and math.isfinite(_indep_Q):
                                    if _indep_Q < _dual - 1e-8:
                                        _verdict = "  → DUAL BOUND IS UNRELIABLE (indep_Q < dual)"
                                    else:
                                        _verdict = f"  → dual ≤ indep_Q (consistent, diff={_indep_Q - _dual:.6e})"
                                print(f"             indep_Q(cs_K)={_indep_str}{_verdict}")
                            except Exception as _e:
                                print(f"             indep_Q(cs_K): FAILED ({_e})")

                    # Per-scenario UB comparison
                    _ub_per_scen = UB_global * S  # total UB
                    print(f"  sum(c_s_dual)={sum((_m or {}).get('dual_bound',0) or 0 for _m in _cs_metas):.9f}, "
                          f"UB_total={_ub_per_scen:.9f}, UB_per_scen={UB_global:.9f}")

                    # === Per-scenario Q_s(K_EF) evaluation ===
                    print(f"\n  == Per-scenario Q_s at K_EF = {UB_node} ==")
                    _sum_q_ef = 0.0
                    for _s_idx in range(S):
                        try:
                            _q_ef_s = evaluate_Q_at(base_bundles[_s_idx], first_vars_list[_s_idx], UB_node)
                            _sum_q_ef += _q_ef_s
                        except Exception as _e:
                            _q_ef_s = float('nan')
                            print(f"    scen {_s_idx}: Q_s(K_EF) evaluation FAILED: {_e}")
                            continue

                        _cs_m = _cs_metas[_s_idx] if _s_idx < len(_cs_metas) else None
                        _cs_dual_s = (_cs_m or {}).get("dual_bound", None)
                        _cs_prim_s = (_cs_m or {}).get("primal_obj", None)

                        _flag = ""
                        if _cs_dual_s is not None:
                            if _q_ef_s < _cs_dual_s - 1e-8:
                                _flag = "  ⚠ Q_s(K_EF) < c_s_dual → GUROBI CS DUAL IS WRONG"
                            elif _q_ef_s < _cs_dual_s + 1e-6:
                                _flag = "  ~ Q_s(K_EF) ≈ c_s_dual (within tol)"
                            else:
                                _flag = f"  ✓ Q_s(K_EF) > c_s_dual (diff={_q_ef_s - _cs_dual_s:.6e})"

                        _cs_d_str = f"{_cs_dual_s:.9f}" if _cs_dual_s is not None else "N/A"
                        print(f"    scen {_s_idx}: Q_s(K_EF)={_q_ef_s:.9f}, c_s_dual={_cs_d_str}{_flag}")

                    print(f"    SUM Q_s(K_EF)={_sum_q_ef:.9f}, sum(c_s)={sum((_m or {}).get('dual_bound',0) or 0 for _m in _cs_metas):.9f}, "
                          f"diff={_sum_q_ef - sum((_m or {}).get('dual_bound',0) or 0 for _m in _cs_metas):.6e}")
                print()
            else:
                print(f"[Iter {it}] UB point {UB_node} is NOT inside any simplex (outside mesh)")
            msb_src = f"T{ms_b_simp}" if ms_b_simp is not None else "N/A"
            print(f"[Iter {it}] LB = {LB_global:.6f} = UB({UB_global:.6f}) + ms_b({ms_b:.3e}) from {msb_src}")

        _phases["5_EF_UB_solve"] = perf_counter() - _t_phase

        # 9) next node candidate ranking & selection
        # ------------------------------------------------
        _t_phase = perf_counter()
        # For LB-simplex, check if "all ms are +inf" -> use c_s-based fallback point selection
        use_c_fallback = False
        lb_ms_list = lb_simp_rec.get("ms_per_scene", [])
        if lb_ms_list:
            if all(math.isinf(float(ms_val)) for ms_val in lb_ms_list):
                c_list_lb = lb_simp_rec.get("c_per_scene", [])
                if c_list_lb and any(math.isfinite(float(c)) for c in c_list_lb):
                    use_c_fallback = True
                    if verbose:
                        print(f"[Iter {it}] Using c_s-based candidate in LB simplex T{lb_simp_idx} "
                              f"because all ms_s are +inf.")

        # New strategy: select next points only from the "currently tightest simplex block".
        active = [lb_simp_rec]              
        pool_records = [lb_simp_rec]
    
        # Collect candidate points for each scene from this simplex.
        # - Normal case: use ms_per_scene + xms_per_scene
        # - Fallback case (use_c_fallback=True): use c_per_scene + c_point_per_scene
        cand_items = []

        # Tolerance parameters for point selection


        for rec in pool_records:
            sid = rec["simplex_index"]
            verts = np.array(rec["verts"])

            if use_c_fallback:
                val_list = rec.get("c_per_scene", [])
                pts_list = rec.get("c_point_per_scene", [None] * len(val_list))
                # In fallback, we only have c_s points, so source is always c_s
                sources = ["c_s_fallback"] * len(val_list)
            else:
                ms_vals = rec.get("ms_per_scene", [])
                ms_pts = rec.get("xms_per_scene", [None] * len(ms_vals))
                c_vals = rec.get("c_per_scene", [])
                c_pts = rec.get("c_point_per_scene", [None] * len(c_vals))
                
                val_list = []
                pts_list = []
                sources = []

                for s in range(len(ms_vals)):
                    ms_pt = ms_pts[s]
                    c_pt = c_pts[s] if s < len(c_pts) else None
                    
                    # Default selection: ms point
                    selected_val = ms_vals[s]
                    selected_pt = ms_pt
                    source = "ms(base)"

                    if ms_pt is not None and c_pt is not None:
                        # Check distances
                        dist_ms_c = np.linalg.norm(np.array(ms_pt) - np.array(c_pt))
                        
                        # Check distance from c_pt to vertices
                        dist_c_verts = min(np.linalg.norm(np.array(c_pt) - v) for v in verts)

                        if dist_ms_c < TOL_MS_C:
                             if dist_c_verts > TOL_C_VERTS:
                                selected_pt = c_pt
                                source = "c_s"
                                # Debug print
                                if verbose:
                                    print(f"[Iter {it}] Scene {s}: Switched to c_s point. "
                                          f"dist_ms_c={dist_ms_c:.2e} < {TOL_MS_C}, "
                                          f"dist_c_verts={dist_c_verts:.2e} > {TOL_C_VERTS}")
                             else:
                                 source = "ms(vert)"
                                 if verbose:
                                     print(f"[Iter {it}] Scene {s}: Kept ms point (c_s too close to verts). "
                                           f"dist_ms_c={dist_ms_c:.2e} < {TOL_MS_C} BUT "
                                           f"dist_c_verts={dist_c_verts:.2e} <= {TOL_C_VERTS}")
                        else:
                            source = "ms(dist)"
                            # if verbose:
                            #    print(f"[Iter {it}] Scene {s}: Kept ms point (ms too far from c_s). "
                            #          f"dist_ms_c={dist_ms_c:.2e} >= {TOL_MS_C}")

                    val_list.append(selected_val)
                    pts_list.append(selected_pt)
                    sources.append(source)

            for s in range(len(val_list)):
                cand_items.append({
                    "simplex_index": sid,
                    "scene": s,
                    "cand_ms": val_list[s],   # Note: this might be ms_val even if we picked c_pt, depending on logic above. 
                                              # Actually, if source is c_s_fallback, val_list has c_vals.
                                              # If source is c_s_cond, val_list has ms_vals (based on my logic above).
                                              # Ideally we want to sort by the "potential improvement". 
                                              # ms is the gap. c is the cut value. 
                                              # Let's stick to using ms_val for sorting unless fallback.
                    "cand_pt": pts_list[s],
                    "pt_source": sources[s],
                    "_rec": rec
                })

        MODE2 = False   # Default False: Mode 1 (selects the point with the smallest ms)

        new_node = None
        chosen_ms = None
        chosen_cand = None
        stop_due_to_collision = False

        def handle_collision(cand_pt, ci, stage_note="active"):
            nonlocal stop_due_to_collision
            X = np.asarray(nodes, float)
            P = np.asarray(cand_pt, float)
            dists = np.linalg.norm(X - P, axis=1)
            j_star = int(np.argmin(dists))
            d_star = float(dists[j_star])
            orange_ids = [r["simplex_index"] for r in per_tet if j_star in r["vert_idx"]]
            debug_pack = {
                "reason": "candidate_too_close",
                "iter": it,
                "stage": stage_note,
                "min_dist": float(min_dist),
                "closest_node_index": j_star,
                "closest_node_point": tuple(map(float, nodes[j_star])),
                "closest_distance": d_star,
                "cand_simplex": int(ci["simplex_index"]),
                "cand_scene": int(ci["scene"]),
                "cand_point": tuple(map(float, cand_pt)),
                "cand_ms": float(ci["cand_ms"]),
                "UB_global": float(UB_global),
                "LB_global": float(LB_global),
                "active_ratio": float(active_ratio),
                "UB_node": tuple(map(float, UB_node)),
                "active_mask": {int(k): bool(v) for k, v in active_mask.items()},
                "nodes_snapshot": [tuple(map(float, nd)) for nd in nodes],
                "per_tet_snapshot": [
                    {
                        "simplex_index": int(r["simplex_index"]),
                        "vert_idx": list(map(int, r["vert_idx"])),
                        "verts": [tuple(map(float, x)) for x in r['verts']],
                        "ms": float(r["ms"]),
                        "ms_per_scene": [float(x) for x in r.get("ms_per_scene", [])],
                        "LB": float(r["LB"]),
                        "UB": float(r["UB"]),
                        "best_scene": int(r["best_scene"]),
                        "x_ms_best_scene": tuple(map(float, r["x_ms_best_scene"])) if r.get("x_ms_best_scene") is not None else None,
                        "volume": float(r["volume"]),
                    } for r in per_tet
                ],
                "highlight_simplices": list(map(int, orange_ids)),
            }
            global LAST_DEBUG
            LAST_DEBUG = debug_pack
            if verbose:
                print(
                    f"[STOP] Candidate {tuple(map(float, cand_pt))} "
                    f"(scene {ci['scene']}) is too close to existing node #{j_star} at distance {d_star:.3e} "
                    f"(< {min_dist:g}). Highlighted simplices: {sorted(orange_ids)}"
                )
            stop_due_to_collision = True

        # ---------- Mode 2 (ms weighted composite point) ----------
        if MODE2:
            rec = lb_simp_rec
            ms_list  = rec.get("ms_per_scene", [])
            pts_list = rec.get("xms_per_scene", [])

            weights = []
            points  = []
            for ms_val, pt in zip(ms_list, pts_list):
                if pt is None:
                    continue
                w = max(0.0, -float(ms_val))
                weights.append(w)
                points.append(np.asarray(pt, float))

            if weights:
                w_arr = np.asarray(weights, float)
                if w_arr.sum() <= 0:
                    w_arr[:] = 1.0
                w_arr /= w_arr.sum()

                candidate_pt = sum(w * p for w, p in zip(w_arr, points))

                if min_dist_to_nodes(candidate_pt, nodes) >= min_dist:
                    cand_pt_pert, loc_type, loc_info = _snap_feature(candidate_pt, rec)
                    new_node   = cand_pt_pert
                    chosen_ms  = float(np.dot(w_arr, np.array(ms_list, float)))
                    chosen_cand = {
                        "simplex_index": rec["simplex_index"],
                        "scene": -1,
                        "cand_ms": chosen_ms,
                        "cand_pt": cand_pt_pert,
                        "_rec": rec,
                        "loc_type": loc_type,
                        "loc_info": loc_info,
                    }
                    if verbose:
                        print(
                            f"Chosen node (MODE2) {tuple(map(float, cand_pt_pert))} "
                            f"with weighted ms={chosen_ms:.3e} "
                            f"(simp T{rec['simplex_index']})"
                        )
                else:
                    dummy_ci = {
                        "simplex_index": rec["simplex_index"],
                        "scene": -1,
                        "cand_ms": 0.0,
                        "cand_pt": tuple(candidate_pt),
                    }
                    handle_collision(candidate_pt, dummy_ci, stage_note="mode2")


        # ---------- Mode 1: Minimum point in milliseconds (default) ----------
        if (not MODE2) and (not stop_due_to_collision):
            def score_item(ci):
                ms = ci["cand_ms"]
                pt = ci["cand_pt"]
                d  = (float('inf') if pt is None else min_dist_to_nodes(pt, nodes))
                return (ms, -d)   

            candidates_sorted = sorted(cand_items, key=score_item)

            for rank, ci in enumerate(candidates_sorted, start=1):
                cand_pt = ci["cand_pt"]
                if cand_pt is None:
                    continue
                if min_dist_to_nodes(cand_pt, nodes) >= min_dist:
                    cand_pt_pert, loc_type, loc_info = _snap_feature(cand_pt, ci.get("_rec", None))
                    new_node = cand_pt_pert
                    ci["loc_type"] = loc_type
                    ci["loc_info"] = loc_info
                    chosen_ms  = ci["cand_ms"]
                    chosen_cand= ci
                    if verbose:
                        metric_name = "ms" if not use_c_fallback else "c_s"
                        pt_source = ci.get("pt_source", "unknown")
                        print(
                            f"Chosen node {tuple(map(float, cand_pt_pert))} "
                            f"with {metric_name}={chosen_ms:.3e} "
                            f"(simp T{ci['simplex_index']}, scene {ci['scene']}, rank #{rank}, source={pt_source})"
                        )
                        print(f"[Iter {it}] LB simplex = T{lb_simp_idx}, "
                              f"next node simplex = T{int(ci['simplex_index'])}, scene {int(ci['scene'])}")

                        # === NEW: Print simplex vertices and new point details ===
                        simp_idx_sel = int(ci['simplex_index'])
                        verts_sel = ci["_rec"]["verts"]
                        print(f"[Selected Simp Info] Iter {it} | Simplex T{simp_idx_sel} Vertices:")
                        for v_i, v in enumerate(verts_sel):
                            print(f"  v{v_i}: {tuple(map(float, v))}")
                        print(f"  -> New Point: {tuple(map(float, new_node))}")
                    break
                else:
                    if verbose:
                        print(
                            f"Skip candidate {tuple(map(float, cand_pt))} "
                            f"(simp T{ci['simplex_index']}, scene {ci['scene']}, rank #{rank}) "
                            f"because too close to existing nodes (< {min_dist:g})."
                        )
                    handle_collision(cand_pt, ci, stage_note="active")
                    break

            if verbose:
                top_msg = "N/A"
                if len(candidates_sorted) > 0:
                    t0 = candidates_sorted[0]
                    metric_name = "ms" if not use_c_fallback else "c_s"
                    top_msg = (f"T{int(t0['simplex_index'])}, scene={t0['scene']}, "
                               f"{metric_name}={float(t0['cand_ms']):.3e}")
                print(f"[Iter {it}] candidate rank #1: {top_msg}")
                _print_candidates_table(candidates_sorted, nodes, topN=10)
                print()

        # ------------------------------------------------

        if stop_due_to_collision:
            if verbose:
                print(f"[Iter {it}] Stop due to collision.")
            break

        if new_node is None:
            if verbose:
                print("New node too close for all candidates (or infeasible ms); stop.")
            break

        #==debug, can delete
        n_tets = len(per_tet)
        n_active = sum(1 for r in per_tet if active_mask.get(r["simplex_index"], False))
        print(f"[Iter {it}] per_tet={n_tets}, active after q_cut={n_active}, tri_is_None={tri is None}")


        # Print how many simplices failed quality check before Visualization
        if verbose:
            print(f"[Iter {it}] bad-quality active simplices (q < {q_cut:g}): {bad_quality_count}")


        # === debug: Check which simplex the light green true solution points fall into, and active status ===
        if enable_3d_plot and true_opt_points is not None:
            # tri might still be None, or some _Dummy placeholder, do not use find_simplex then
            if (tri is None) or (not hasattr(tri, "find_simplex")):
                if verbose:
                    print(f"[Iter {it}] tri has no find_simplex (type={type(tri)}), skip true_opt debug.")
            else:
                for i, p in enumerate(true_opt_points):
                    p = np.asarray(p, float)
                    simp_idx = int(tri.find_simplex(p))
                    is_act = bool(active_mask.get(simp_idx, False))
                    print(
                        f"[Iter {it}] true_opt[{i}] in simplex {simp_idx}, active={is_act}"
                    )

        # Visualization (plot 3D figures)
        if enable_3d_plot and (plot_every is not None) and (it % plot_every == 0):
            # Use the actual chosen simplex for highlighting if available
            if chosen_cand is not None:
                hl_simplices = [chosen_cand["simplex_index"]]
            elif ms_b_simp is not None:
                hl_simplices = [ms_b_simp]
            else:
                hl_simplices = None
            
            # Use the actual chosen point (new_node) if available, otherwise fallback to cand_pt
            pt_to_plot = new_node if new_node is not None else cand_pt

            plot_iteration_plotly(
                it,
                nodes,
                tri,
                active_mask,
                UB_node,
                pt_to_plot,
                per_tet,
                highlight_simplices=hl_simplices,
                true_opt_points=true_opt_points,
                UB_global=UB_global,
                LB_global=LB_global,
            )

        _phases["6_candidate_selection"] = perf_counter() - _t_phase

        # add node and evaluate
        _t_phase = perf_counter()
        t_q0 = perf_counter()
        new_vals = []
        scene_times = [[] for _ in range(S)]
        q_call_cnt = 0  
        for s in range(S):
            t0_q = perf_counter()
            print(f"[Iter {it}] evaluating Q for scenario {s}")
            val, _q_meta = evaluate_Q_at(base_bundles[s], first_vars_list[s], new_node, return_meta=True)
            dt_q = perf_counter() - t0_q
            new_vals.append(val)
            scene_times[s].append(dt_q)
            q_call_cnt += 1
            # --- Q-value timing log: record only NON-OPTIMAL Q solves ---
            _q_term = _q_meta.get('termination_condition', '') if _q_meta else ''
            if _q_term not in ('optimal', 'locallyOptimal'):
                try:
                    _q_line = (
                        f"[Iter {it}] scenario={s}, "
                        f"point=({new_node[0]:.4f}, {new_node[1]:.4f}, {new_node[2]:.4f}), "
                        f"time={_q_meta.get('time_sec', 0.0):.4f}s, "
                        f"status={_q_meta.get('status', '?')}, "
                        f"term={_q_meta.get('termination_condition', '?')}, "
                        f"obj={_q_meta.get('obj')}\n"
                    )
                    with open(dbg_q_timing_path, "a", encoding="utf-8") as _fq:
                        _fq.write(_q_line)
                except Exception:
                    pass
        t_q = perf_counter() - t_q0
        timing["iter_Q_new_time"][timing_idx] = t_q

        iter_q_times_detail.append(scene_times)
        per_iter_q_counts.append(q_call_cnt)

        # === NEW: Print next point details ===
        if verbose and chosen_cand is not None and chosen_cand.get("scene", -1) >= 0:
            print(f"\n[Iter {it}] Next Point Details:")
            sid = chosen_cand["simplex_index"]
            scene = chosen_cand["scene"]
            rec = chosen_cand["_rec"]
            pt_type = chosen_cand.get("pt_source", "c_s" if use_c_fallback else "ms")
            
            lambdas = chosen_cand.get("loc_info", {}).get("lambdas", None)
            if lambdas is not None:
                vert_idx = rec["vert_idx"]
                q_verts = [scen_values[scene][v] for v in vert_idx]
                as_val = float(np.dot(lambdas, q_verts))
                ms_val = float(rec["ms_per_scene"][scene])
                as_plus_ms = as_val + ms_val
                q_val = float(new_vals[scene])
                
                header = ["Simplex", "Scene", "Type", "As", "ms", "As+ms", "Q", "(Kp, Ki, Kd)"]
                colw = [10, 8, 8, 15, 15, 15, 15, 30]
                
                def fmt_row(cols):
                    return "".join(str(c).ljust(w) for c, w in zip(cols, colw))
                
                print(fmt_row(header))
                print("-" * sum(colw))
                
                # Format coordinates
                coords_str = f"({new_node[0]:.4f}, {new_node[1]:.4f}, {new_node[2]:.4f})"
                
                row = [
                    f"T{sid}", scene, pt_type,
                    f"{as_val:.4e}", f"{ms_val:.4e}", f"{as_plus_ms:.4e}", f"{q_val:.4e}",
                    coords_str
                ]
                print(fmt_row(row))
                print()

        # === append node ===
        nodes.append(tuple(map(float, new_node)))
        new_node_index = len(nodes) - 1
        for ω in range(S):
            scen_values[ω].append(new_vals[ω])

        _phases["7_Q_eval_new_node"] = perf_counter() - _t_phase

        # === NEW: update mesh by star-subdividing the simplex that generated new_node ===
        _t_phase = perf_counter()
        if chosen_cand is not None:
            selection_reason_hist.append(chosen_cand.get("pt_source", "unknown")) # Record reason
            sid = int(chosen_cand["simplex_index"])
            loc_type = chosen_cand.get("loc_type", "interior")
            loc_info = chosen_cand.get("loc_info", None)
            
            # LB/UB per user definition: save selected simplex record BEFORE split
            selected_rec_before_split = chosen_cand.get("_rec", None)
            if selected_rec_before_split is None:
                # Try to find it from per_tet
                for r in per_tet:
                    if r["simplex_index"] == sid:
                        selected_rec_before_split = r
                        break
            selected_simplex_id = tuple(sorted(selected_rec_before_split["vert_idx"])) if selected_rec_before_split else None

            # Assign code to split type: 1=interior, 2=edge, 3=face
            split_code = {"interior": 1, "edge": 2, "face": 3}.get(loc_type, 0)
            if verbose:
                print(f"[Iter {it}] subdivision type = {loc_type} "
                      f"(code={split_code}) on simplex T{sid}")

            # LB/UB per user definition: capture n_children from subdivide
            if loc_type == "edge" and loc_info is not None:
                edge_verts = loc_info["edge_verts"]   # Here follows the meaning defined in _snap_feature
                if verbose:
                    print(f"           edge local verts = {edge_verts}")
                n_children = tet_mesh.subdivide_edge(sid, new_node_index, edge_verts)
            elif loc_type == "face" and loc_info is not None:
                face_verts = loc_info["face_verts"]
                if verbose:
                    print(f"           face local verts = {face_verts}")
                n_children = tet_mesh.subdivide_face(sid, new_node_index, face_verts)
            else:
                # Default treated as interior point, star subdivision
                n_children = tet_mesh.subdivide(sid, new_node_index)
        else:
            n_children = 0  # No split happened

        add_node_hist.append(new_node)
        if verbose:
            print(f"[Iter {it}] Elapsed: {perf_counter() - t_iter0:.3f}s")

        # ====================================================================
        # LB/UB per user definition: end-of-iter min-LB and incumbent UB from NEW simplices only
        # ====================================================================
        
        # Track which simplex indices are "new" this iteration
        # LB/UB per user definition: ALWAYS use n_children-based calculation (no it==0 special case)
        # subdivide does pop()+extend(), so new children are at the END of tets list
        n_simplices_after = len(tet_mesh.tets)
        # new_ids = last n_children indices after subdivide
        new_simplex_indices = list(range(n_simplices_after - n_children, n_simplices_after))
        
        _phases["8_mesh_subdivide"] = perf_counter() - _t_phase

        # Re-evaluate per_tet for ALL simplices after split (to get updated LB_local values)
        _t_phase = perf_counter()
        _, per_tet_end = evaluate_all_tetra(
            nodes, scen_values, ms_bundles, first_vars_list,
            ms_cache=ms_cache, cache_on=True, tracker=tracker,
            tet_mesh=tet_mesh,
            lb_sur_cache=lb_sur_cache,  # LB surrogate cache
            dbg_timelimit_path=dbg_timelimit_path,
            dbg_cs_timing_path=dbg_cs_timing_path,
            dbg_ms_timing_path=dbg_ms_timing_path,
            iter_num=it,
        )
        
        # Build per_tet dict for quick lookup
        per_tet_dict = {r["simplex_index"]: r for r in per_tet_end}
        
        # LB_global_end = min over ALL simplices of LB_local
        LB_global_end = float(min(r["LB"] for r in per_tet_end))
        lb_simp_rec_end = min(per_tet_end, key=lambda r: r["LB"])
        
        # === LB Split Diagnostic: compare parent vs children LBs ===
        if selected_rec_before_split is not None:
            parent_LB = float(selected_rec_before_split["LB"])
            parent_id = tuple(sorted(selected_rec_before_split["vert_idx"]))
            # Previous global min LB (from pre-split evaluation)
            prev_global_min_LB = float(min(r["LB"] for r in per_tet))
            
            # Determine if new global LB is INSIDE (child) or OUTSIDE (old simplex)
            new_lb_simplex_idx = lb_simp_rec_end["simplex_index"]
            if n_children > 0:
                new_lb_is_inside = new_lb_simplex_idx in new_simplex_indices
                location_str = "INSIDE" if new_lb_is_inside else "OUTSIDE"
            else:
                new_lb_is_inside = None
                location_str = "N/A"
            
            _diag_lines = []
            _diag_lines.append("=" * 70)
            _diag_lines.append(f"[Iter {it}] LB SPLIT DIAGNOSTIC")
            _diag_lines.append(f"  Selected (parent) simplex: {parent_id}")
            _diag_lines.append(f"  Parent LB: {parent_LB/S:.9f} (per-scen)")

            # EF solution on this simplex (if available)
            _ef_raw = ef_iter_info.get("ef_obj")
            _ef_recalc = ef_iter_info.get("true_obj")
            if _ef_raw is not None and _ef_recalc is not None:
                _diag_lines.append(
                    f"  EF ef_obj(IPOPT): {_ef_raw/S:.9f}, "
                    f"true_obj(recalc): {_ef_recalc/S:.9f}, "
                    f"used=min={min(_ef_raw, _ef_recalc)/S:.9f} (per-scen)"
                )
            elif _ef_recalc is not None:
                _diag_lines.append(f"  EF true_obj: {_ef_recalc/S:.9f} (per-scen)")
            elif _ef_raw is not None:
                _diag_lines.append(f"  EF ef_obj(IPOPT): {_ef_raw/S:.9f} (per-scen), true_obj: N/A")
            else:
                _diag_lines.append(f"  EF: N/A")

            # Simplex volume: |det([v1-v0, v2-v0, v3-v0])| / 6
            try:
                _sv = np.array(selected_rec_before_split["verts"])
                _vol = abs(np.linalg.det((_sv[1:] - _sv[0]))) / 6.0
                _diag_lines.append(f"  Simplex volume: {_vol:.6e}")
            except Exception:
                _diag_lines.append(f"  Simplex volume: N/A")

            _diag_lines.append(f"  Children ({n_children}):")
            
            child_lbs = []
            for new_idx in new_simplex_indices:
                if new_idx in per_tet_dict:
                    child_rec = per_tet_dict[new_idx]
                    child_lb = float(child_rec["LB"])
                    child_id = tuple(sorted(child_rec["vert_idx"]))
                    child_lbs.append(child_lb)
                    _diag_lines.append(f"    T{new_idx} {child_id}: LB = {child_lb/S:.9f}")
            
            new_global_min_id = tuple(sorted(lb_simp_rec_end["vert_idx"]))
            _diag_lines.append(f"  New global LB: {LB_global_end/S:.9f} (from T{lb_simp_rec_end['simplex_index']} {new_global_min_id})")
            _diag_lines.append(f"  Location: {location_str}")

            # --- Avg CS point Q-value (true objective at mean of c_s solution points) ---
            _cs_pts = lb_simp_rec.get("c_point_per_scene", [])
            _valid_cs_pts = [pt for pt in _cs_pts if pt is not None and all(math.isfinite(v) for v in pt)]
            if _valid_cs_pts:
                _avg_cs_pt = tuple(np.mean(_valid_cs_pts, axis=0))
                _cs_true_val = 0.0
                for s in range(S):
                    _cs_true_val += evaluate_Q_at(base_bundles[s], first_vars_list[s], _avg_cs_pt)
                _diag_lines.append(f"  Avg CS point Q-value: {_cs_true_val/S:.9f} (per-scen), point={_avg_cs_pt}")
            else:
                _diag_lines.append(f"  Avg CS point Q-value: N/A (no valid c_s points)")

            # --- Current UB info + simplex provenance ---
            try:
                _ub_ps = f"{UB_incumbent/S:.9f}" if math.isfinite(UB_incumbent) else str(UB_incumbent)
            except Exception:
                _ub_ps = "N/A"
            _diag_lines.append(f"  Current UB: {_ub_ps} (per-scen)")
            _diag_lines.append(f"  UB point: {UB_node}")
            _diag_lines.append(f"  UB source: {ub_source_this_iter if ub_updated_this_iter else ub_source_current}")
            _diag_lines.append(f"  UB simplex: {ub_simplex_id_this_iter if ub_updated_this_iter else ub_simplex_id_current}")

            # --- IPOPT EF solver status ---
            _ipopt_res = _ef_dual_results.get("ipopt", {})
            if _ipopt_res:
                _ip_ok = _ipopt_res.get("ok", False)
                _ip_status = _ipopt_res.get("status", "N/A")
                _ip_term = _ipopt_res.get("term", "N/A")
                _ip_time = _ipopt_res.get("time", 0.0)
                _ip_ef_obj = _ipopt_res.get("ef_obj")
                _ip_true = _ipopt_res.get("true_obj")
                _ip_true_ps = f"{_ip_true/S:.9f}" if _ip_true is not None else "N/A"
                _diag_lines.append(f"  EF-IPOPT: ok={_ip_ok}, status={_ip_status}, term={_ip_term}, "
                                   f"time={_ip_time:.3f}s, ef_obj={_ip_ef_obj}, true_obj(per-scen)={_ip_true_ps}")
            else:
                _diag_lines.append(f"  EF-IPOPT: not available")

            _diag_lines.append("=" * 70)
            
            # Print to console
            if verbose:
                for line in _diag_lines:
                    print(line)
            
            # Write to debug_lb_after_split.txt (append within this run)
            with open(dbg_lb_split_path, "a", encoding="utf-8") as _fdiag:
                _fdiag.write("\n".join(_diag_lines) + "\n\n")
        
        # UB_global_end: generate candidates ONLY from NEW simplices
        # LB/UB per user definition: use incumbent logic (min with previous UB)
        UB_global_end = UB_incumbent  # Start from incumbent (not reset per iteration)
        ub_source_end = ub_source_this_iter if ub_updated_this_iter else ub_source_current
        ub_updated_end = False
        
        for new_idx in new_simplex_indices:
            if new_idx not in per_tet_dict:
                continue
            rec = per_tet_dict[new_idx]
            
            # (a) avgMS point: average of per-scenario xms points for this simplex
            xms_list = rec.get("xms_per_scene", [])
            valid_xms = [pt for pt in xms_list if pt is not None and all(math.isfinite(v) for v in pt)]
            if valid_xms:
                avg_ms_pt = tuple(np.mean(valid_xms, axis=0))
                # Evaluate TRUE objective at avg_ms_pt (expected value)
                true_val = 0.0
                for s in range(S):
                    true_val += evaluate_Q_at(base_bundles[s], first_vars_list[s], avg_ms_pt)
                
                # LB/UB per user definition: allow incumbent update when true_val < UB
                # If true_val < LB_global_end, emit warning (potential invalid LB) but still update
                if true_val < LB_global_end - 1e-8:
                    print(f"[Iter {it}] WARNING: UB candidate {true_val/S:.6e} < LB {LB_global_end/S:.6e} (potential invalid LB)")
                if true_val < UB_global_end:
                    UB_global_end = true_val
                    UB_node = avg_ms_pt
                    ub_updated_end = True
                    ub_source_end = "avgMS_new_simplex"
                    ub_simplex_id_this_iter = tuple(sorted(rec["vert_idx"]))
                    # Add to library for monotonicity
                    ub_candidate_library.append((avg_ms_pt, true_val))
                    if verbose:
                        print(f"[Iter {it}] UB candidate from new simplex {new_idx} (avgMS): {true_val/S:.6e} (per scenario)")
            
            # (a2) avgCS point: average of per-scenario c_s solution points for this simplex
            cpts_list = rec.get("c_point_per_scene", [])
            valid_cpts = [pt for pt in cpts_list if pt is not None and all(math.isfinite(v) for v in pt)]
            if valid_cpts:
                avg_cs_pt = tuple(np.mean(valid_cpts, axis=0))
                # Evaluate TRUE objective at avg_cs_pt (expected value)
                cs_true_val = 0.0
                for s in range(S):
                    cs_true_val += evaluate_Q_at(base_bundles[s], first_vars_list[s], avg_cs_pt)
                
                if cs_true_val < LB_global_end - 1e-8:
                    print(f"[Iter {it}] WARNING: UB candidate (avgCS) {cs_true_val/S:.6e} < LB {LB_global_end/S:.6e} (potential invalid LB)")
                if cs_true_val < UB_global_end:
                    UB_global_end = cs_true_val
                    UB_node = avg_cs_pt
                    ub_updated_end = True
                    ub_source_end = "avgCS_new_simplex"
                    ub_simplex_id_this_iter = tuple(sorted(rec["vert_idx"]))
                    ub_candidate_library.append((avg_cs_pt, cs_true_val))
                    if verbose:
                        print(f"[Iter {it}] UB candidate from new simplex {new_idx} (avgCS): {cs_true_val/S:.6e} (per scenario)")
            
            # (b) EF candidate: use true_obj (recalculated Q-value) for UB.
            #     ef_obj is the raw IPOPT objective which can be a relaxed/local
            #     value lower than the actual Q-value — NOT valid as a UB candidate.
        if ef_iter_info.get("ef_ok"):
            _ef_recalc = ef_iter_info.get("true_obj")
            if _ef_recalc is not None and math.isfinite(_ef_recalc):
                if _ef_recalc < UB_global_end:
                    UB_global_end = _ef_recalc
                    UB_node = ef_iter_info["K_ef"]
                    ub_updated_end = True
                    ub_source_end = "EF"
                    ub_simplex_id_this_iter = selected_simplex_id
                    if verbose:
                        print(f"[Iter {it}] UB_global_end improved by EF: {_ef_recalc/S:.6e} (per-scen), "
                              f"ef_obj={ef_iter_info.get('ef_obj')}, true_obj={_ef_recalc}")
        
        # Provenance logging uses end-of-iter incumbent UB only
        # Update provenance if UB changed this iteration
        if ub_updated_end:
            ub_updated_this_iter = True
            ub_source_current = ub_source_end
            ub_origin_iter_current = it
            ub_simplex_id_current = ub_simplex_id_this_iter
            iter_logger.update_ub_provenance(
                updated=True,
                source=ub_source_end,
                simplex_id=ub_simplex_id_this_iter,
                origin_iter=it
            )
        else:
            iter_logger.update_ub_provenance(updated=False)
        
        # === OVERRIDE LB_hist/UB_hist with end-of-iteration values ===
        # These will be used for both CSV and summary table
        if LB_hist:
            LB_hist[-1] = LB_global_end
        if UB_hist:
            UB_hist[-1] = UB_global_end
        
        # Also update the current-iteration LB_global/UB_global for consistency
        LB_global = LB_global_end
        UB_global = UB_global_end
        lb_simp_rec = lb_simp_rec_end
        
        # LB/UB per user definition: update incumbent UB
        UB_incumbent = UB_global_end
        
        # ====================================================================
        # End of LB/UB per user definition
        # ====================================================================

        _phases["9_end_of_iter_reeval"] = perf_counter() - _t_phase

        t_iter = perf_counter() - t_iter0
        _phases["TOTAL"] = t_iter
        _iter_phase_times.append(_phases)

        # === Print detailed timing summary for this iteration ===
        if verbose:
            print(f"\n{'='*70}")
            print(f"[TIMING] === Iter {it} phase breakdown (total {t_iter:.2f}s) ===")
            for _pname, _ptime in _phases.items():
                if _pname == "TOTAL":
                    continue
                pct = _ptime / t_iter * 100 if t_iter > 0 else 0
                bar = '#' * int(pct / 2)
                print(f"  {_pname:30s} : {_ptime:10.2f}s  ({pct:5.1f}%)  {bar}")
            print(f"  {'TOTAL':30s} : {t_iter:10.2f}s")
            # Count cache hits vs fresh solves
            n_tets_iter = len(per_tet)
            n_fresh_ms = sum(per_scene_calls)
            n_total_possible = n_tets_iter * S
            n_cache_hits = n_total_possible - n_fresh_ms
            print(f"  #simplices={n_tets_iter}, #scenarios={S}, total_ms_pairs={n_total_possible}")
            print(f"  fresh ms solves={n_fresh_ms}, cache hits={n_cache_hits}")
            print(f"{'='*70}\n")
        timing["iter_total_time"][timing_idx] = t_iter 
        
        # === Update best_lb_ever / best_ub_ever AFTER end-of-iter ===
        # LB/UB per user definition: update monotonic envelopes using end-of-iter values
        best_lb_ever = max(best_lb_ever, LB_global_end)
        best_ub_ever = min(best_ub_ever, UB_global_end)
        
        # === Append current iteration to CSV (AFTER end-of-iter update) ===
        # LB/UB per user definition: use end-of-iter LB/UB (same as summary table)
        iter_time = iter_time_hist[-1]
        n_nodes = node_count[-1]
        lb_val = LB_global_end / S  # end-of-iteration LB
        ub_val = UB_global_end / S  # end-of-iteration UB
        abs_gap = (UB_global_end - LB_global_end) / S
        rel_gap = abs_gap / (abs(ub_val) + 1e-16)
        # Additional monotonic envelope columns
        lb_ever = best_lb_ever / S
        ub_ever = best_ub_ever / S
        # === Determine UB_in_split: is UB_node inside the split simplex? ===
        _ub_in_split_str = "N/A"
        try:
            if selected_rec_before_split is not None and UB_node is not None:
                _split_V = np.array(selected_rec_before_split["verts"], dtype=float)  # (4,3)
                _ub_pt = np.asarray(UB_node, dtype=float)
                _T = (_split_V[1:] - _split_V[0]).T  # (3,3)
                _lam123 = np.linalg.solve(_T, _ub_pt - _split_V[0])
                _lam0 = 1.0 - _lam123.sum()
                _lam_all = np.array([_lam0, *_lam123])
                _ub_in_split_str = "INSIDE" if np.all(_lam_all >= -1e-8) else "OUTSIDE"
        except (NameError, Exception):
            _ub_in_split_str = "N/A"

        # LB_in_split: reuse location_str from split diagnostic (INSIDE/OUTSIDE)
        # location_str is only defined when selected_rec_before_split is not None AND
        # the LB split diagnostic block runs; use a safe lookup.
        _lb_in_split_str = "N/A"
        try:
            _lb_in_split_str = location_str  # type: ignore[possibly-undefined]
        except NameError:
            pass

        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([f"{iter_time:.3f}", n_nodes, f"{lb_val:.9f}", f"{ub_val:.9f}", f"{rel_gap*100:.7f}%", f"{abs_gap:.5f}", f"{lb_ever:.9f}", f"{ub_ever:.9f}", _lb_in_split_str, _ub_in_split_str])

        # === Write EF solver details to debug_ef_solver.txt ===
        try:
            with open(dbg_ef_solver_path, "a", encoding="utf-8") as _fef:
                _fef.write(f"--- [Iter {it}] ---\n")
                for _stag in ["ipopt", "gurobi"]:
                    _sres = _ef_dual_results.get(_stag, {})
                    _s_ok = _sres.get("ok", False)
                    _s_true = _sres.get("true_obj")
                    _s_true_ps = f"{_s_true/S:.9f}" if _s_true is not None else "N/A"
                    _lb_str = ""
                    if _sres.get("lower_bound") is not None:
                        _lb_str = f", gurobi_LB={_sres['lower_bound']:.9f}"
                    _fef.write(
                        f"  [{_stag:6s}] ok={_s_ok}, "
                        f"status={_sres.get('status')}, term={_sres.get('term')}, "
                        f"time={_sres.get('time', 0.0):.3f}s, "
                        f"ef_obj={_sres.get('ef_obj')}, "
                        f"true_obj(per-scen)={_s_true_ps}{_lb_str}\n"
                    )
                _winner = ef_iter_info.get('solver_status', 'none').split(':')[0] if ef_iter_info.get('ef_ok') else 'none'
                _fef.write(
                    f"  [winner] {_winner}, ub_updated={ef_iter_info.get('ub_updated_by_ef', False)}, "
                    f"UB_incumbent={UB_incumbent/S:.9f}, "
                    f"LB_global_end={LB_global_end/S:.9f}, "
                    f"UB_global_end={UB_global_end/S:.9f}\n\n"
                )
        except Exception:
            pass  # logging must never crash the algorithm

        # === Convergence stopping condition (AFTER end-of-iter) ===
        if gap_stop_tol is not None and float(gap_stop_tol) > 0.0:
            gap_rel_end = float(UB_global_end - LB_global_end) / (abs(UB_global_end) + 1e-16)
            if gap_rel_end <= float(gap_stop_tol):
                if verbose:
                    print(f"[Iter {it}] Stop: UB-LB gap {gap_rel_end:.6e} <= tol {float(gap_stop_tol):.6e}.")
                break

        # === Time limit stopping condition ===
        if time_limit is not None and time_limit > 0:
            elapsed = perf_counter() - t_start
            if elapsed >= time_limit:
                if verbose:
                    print(f"[Iter {it}] Stop: Time limit reached ({elapsed:.2f}s >= {time_limit:.2f}s).")
                break

        # === Per-iteration diagnostic logging ===
        try:
            # EF info: no EF solve in this code path
            ef_info = {
                "EF_attempted": ef_iter_info.get("ef_attempted", False),
                "EF_enabled": enable_ef_ub,
                "time_sec": ef_iter_info.get("ef_time_sec"),
                "status": ef_iter_info.get("solver_status"),
                "termination_condition": ef_iter_info.get("termination_condition"),
                "solver_status": ef_iter_info.get("solver_status"),
                "used_for_UB": ef_iter_info.get("ub_updated_by_ef", False),
            }

            # UB info
            ub_info = {"updated_this_iter": ub_updated_this_iter}

            # LB simplex tracking
            # LB/UB per user definition: use pre-saved selected_simplex_id (saved BEFORE split)
            # selected_simplex_id: the simplex chosen for branching/splitting this iteration (before split)
            # Note: selected_simplex_id was saved when chosen_cand was processed, before the split
            lb_selected_simplex_id = selected_simplex_id if 'selected_simplex_id' in dir() else None
            
            # best_simplex_id_before_split: the branching simplex (same as selected)
            lb_best_before_split = lb_selected_simplex_id
            
            # best_simplex_id_after_split: computed from lb_simp_rec_end (argmin LB after split/rebuild)
            lb_best_after_split = tuple(sorted(lb_simp_rec_end["vert_idx"])) if lb_simp_rec_end else None
            
            # Check if after-split best LB simplex is a child of the selected (parent) simplex
            if n_children > 0 and lb_simp_rec_end is not None:
                stays_in_selected = (lb_simp_rec_end["simplex_index"] in new_simplex_indices)
            else:
                stays_in_selected = None
            
            lb_info = {
                "selected_simplex_id": lb_selected_simplex_id,
                "best_simplex_id_before_split": lb_best_before_split,
                "best_simplex_id_after_split": lb_best_after_split,
                "stays_in_selected": stays_in_selected,
            }

            # MS/CS fallback aggregation for the END-OF-ITER global-LB simplex
            # Use metadata stored in lb_simp_rec_end (post-split argmin LB)
            ms_status_summary = {}
            ms_fallback_scenarios = []
            ms_fallback_reason_counts = {}
            cs_status_summary = {}
            cs_fallback_scenarios = []
            cs_fallback_reason_counts = {}

            ms_meta_list = lb_simp_rec_end.get("ms_meta_per_scene", []) if lb_simp_rec_end else []
            cs_meta_list = lb_simp_rec_end.get("cs_meta_per_scene", []) if lb_simp_rec_end else []

            for s in range(len(ms_meta_list)):
                # MS metadata
                ms_meta = ms_meta_list[s]
                if ms_meta:
                    st = ms_meta.get("status", "unknown")
                    ms_status_summary[st] = ms_status_summary.get(st, 0) + 1
                    if ms_meta.get("used_fallback", False):
                        ms_fallback_scenarios.append(s)
                        reason = ms_meta.get("fallback_reason", "other")
                        ms_fallback_reason_counts[reason] = ms_fallback_reason_counts.get(reason, 0) + 1
                else:
                    ms_status_summary["cached"] = ms_status_summary.get("cached", 0) + 1

            for s in range(len(cs_meta_list)):
                # CS metadata
                cs_meta = cs_meta_list[s]
                if cs_meta:
                    st = cs_meta.get("status", "unknown")
                    cs_status_summary[st] = cs_status_summary.get(st, 0) + 1
                    if cs_meta.get("used_fallback", False):
                        cs_fallback_scenarios.append(s)
                        reason = cs_meta.get("fallback_reason", "other")
                        cs_fallback_reason_counts[reason] = cs_fallback_reason_counts.get(reason, 0) + 1
                else:
                    cs_status_summary["cached"] = cs_status_summary.get("cached", 0) + 1

            ms_agg = {
                "status_summary": ms_status_summary,
                "fallback_any": len(ms_fallback_scenarios) > 0,
                "fallback_count": len(ms_fallback_scenarios),
                "fallback_scenarios": ms_fallback_scenarios,
                "fallback_reason_counts": ms_fallback_reason_counts,
            }

            cs_agg = {
                "status_summary": cs_status_summary,
                "fallback_any": len(cs_fallback_scenarios) > 0,
                "fallback_count": len(cs_fallback_scenarios),
                "fallback_scenarios": cs_fallback_scenarios,
                "fallback_reason_counts": cs_fallback_reason_counts,
            }

            # Log this iteration
            iter_logger.log_iteration(it, ef_info, ub_info, lb_info, ms_agg, cs_agg)

        except Exception as e:
            # Logging must never crash the algorithm
            if verbose:
                print(f"[Iter {it}] Warning: diagnostic logging failed: {e}")

        it += 1

    # === Final summary table (like Gurobi log) ===
    if verbose and LB_hist:
        print("\n===== Simplex search summary (per-scenario) =====")
        header = (
            f"{'Time (s)':>10} "
            f"{'# Nodes':>8} "
            f"{'LB':>14} "
            f"{'UB':>14} "
            f"{'Rel. Gap':>10} "
            f"{'Abs. Gap':>10} "
            f"{'#simplex':>10} "
            f"{'#active':>10}"
            f"{'Split':>8} "
            f"{'Selection':>15}"
        )
        print(header)
        print("-" * len(header))

        for k in range(len(LB_hist)):
            t_k   = iter_time_hist[k]
            n_k   = node_count[k]
            # Divide LB, UB, and Abs. Gap by number of scenarios
            lb_k  = LB_hist[k] / S
            ub_k  = UB_hist[k] / S
            gap_abs = (UB_hist[k] - LB_hist[k]) / S  # Per-scenario gap
            gap_rel = gap_abs / (abs(ub_k) + 1e-16)
            nsimp = simplex_hist[k]
            nact  = active_simplex_hist[k]
            split_kind = split_kind_hist[k]
            reason = selection_reason_hist[k] if k < len(selection_reason_hist) else "N/A"

            print(
                f"{t_k:>10.3f} "
                f"{n_k:>8d} "
                f"{lb_k:>14.9f} "
                f"{ub_k:>14.9f} "
                f"{gap_rel*100:>9.4f}% "
                f"{gap_abs:>10.5f} "
                f"{nsimp:>10d} "
                f"{nact:>10d}"
                f"{split_kind:>8} "
                f"{reason:>15}"
            )



    # === Close iteration logger ===
    iter_logger.close()

    return {
        "nodes": np.array(nodes, float),
        "LB_hist": LB_hist,
        "UB_hist": UB_hist,
        "ms_hist": ms_hist,
        "ms_b_hist": ms_b_hist,
        "node_count": node_count,
        "UB_node_hist": UB_node_hist,
        "added_nodes": add_node_hist,
        "active_ratio_hist": active_ratio_hist,
        "timing": timing,
        "ms_ub_active_per_iter": ms_ub_active_per_iter,
        "iter_q_times_detail": iter_q_times_detail,
        "per_iter_q_counts": per_iter_q_counts,
        "c_hist": c_hist_per_iter,
        "lb_c_agg_hist": lb_c_agg_hist,           
        "lb_c_per_scene_hist": lb_c_per_scene_hist, 
        "iter_ms_times_detail": iter_ms_times_detail,
        "per_iter_ms_counts": per_iter_ms_counts,

    }

