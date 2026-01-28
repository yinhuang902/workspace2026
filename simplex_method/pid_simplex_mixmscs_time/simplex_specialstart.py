# simplex.py
import numpy as np
import math
import pyomo.environ as pyo   
from scipy.spatial import Delaunay
from time import perf_counter
from pyomo.opt import SolverStatus, TerminationCondition
from exact_opt import compute_exact_optima  # NEW: for validation points
import csv
import os

from utils import SimplexTracker
from utils import (
    MIN_DIST, ACTIVE_TOL, MS_AGG, MS_CACHE_ENABLE, GAP_STOP_TOL,
    corners_from_var_bounds, evaluate_Q_at, tet_quality, min_dist_to_nodes,
    _print_candidates_table, plot_iteration_plotly, LAST_DEBUG
)
from bundles import SurrogateLBBundle

# === Debug switches (temporary, can be removed later) ===
DEBUG_LB_SIMPLEX_SCATTER = True        
DEBUG_LB_SIMPLEX_SCATTER_GRID_N = 8
DEBUG_LB_SIMPLEX_SCATTER_NPTS = 300     
DEBUG_LB_SIMPLEX_SCATTER_OUTDIR = "lb_debug_plots1"

# choose c_s setting
TOL_MS_C = 1e2      # Distance threshold between ms point and c_s point
TOL_C_VERTS = 1e2   # Distance threshold between c_s point and simplex vertices
# choose ms setting
TOL_MS_C = 1e-3      # Distance threshold between ms point and c_s point
TOL_C_VERTS = 1e2   # Distance threshold between c_s point and simplex vertices

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
                       tet_mesh: SimplexMesh | None = None):
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

        for ω in range(S):
            cache_key = (int(ω), key_base)
            hit = (cache_on and (ms_cache is not None) and (cache_key in ms_cache))

            if hit:
                # cache is now (ms_val, new_pt_ms, c_val, c_pt)
                ms_val, new_pt_ms, c_val, c_pt = ms_cache[cache_key]
            else:
                ms_val, new_pt_ms, c_val, c_pt = ms_on_tetra_for_scene(
                    ms_bundles[ω], verts, fverts_per_scene[ω]
                )
                if cache_on and (ms_cache is not None):
                    ms_cache[cache_key] = (ms_val, new_pt_ms, c_val, c_pt)
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

        # === NEW: solve true surrogate LB ===
        LB_sur = solve_surrogate_lb_for_tet(fverts_per_scene, ms_scene, c_scene)

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
        })


    return tri, per_tet

# ------------------------- MAIN LOOP -------------------------
def run_pid_simplex_3d(base_bundles, ms_bundles, model_list, first_vars_list,
                       target_nodes=30, min_dist=MIN_DIST, active_tol=ACTIVE_TOL, verbose=True,
                       agg_bundle=None, gap_stop_tol=GAP_STOP_TOL, tracker: SimplexTracker | None = None,
                       plot_every: int | None = None,
                       use_exact_opt: bool = False,
                       exact_solver_name: str = "gurobi",
                       exact_solver_opts: dict | None = None,
                       time_limit: float | None = None):
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
    plot_every : int | None, optional
        Draw the 3D plot every n iterations.

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
    
    S = len(model_list)

    # ==== Custom initial four corners (Kp, Ki, Kd) ====
    nodes = [
        (-10.0, -100.0, 400.0),
        (-10.0,  100.0, 400.0),
        (-10.0, -100.0, -100.0),
        ( 10.0, -100.0, -100.0),
    ]
    nodes = [
        (-1, -99, 1),
        (0, -99, 1),
        (0, -101, 1),
        (0, -101, 0),
    ]

    nodes = [
        (-10.0, -100.0, 1000.0),
        (-10.0,  100.0, 1000.0),
        (-10.0, -100.0, -100.0),
        ( 10.0, -100.0, -100.0),
    ]
    nodes = corners_from_var_bounds(first_vars_list[0])



    # === Preload the exact optimal solution for plotting===
    true_opt_points = None
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

    bounds_arr = np.array([[float(v.lb), float(v.ub)] for v in first_vars_list[0]], float)
    diam = float(np.linalg.norm(bounds_arr[:,1] - bounds_arr[:,0]))  # estimate first stage variable dimension size for simplex shape quality check
    min_dist = float(min_dist)

    # cache f_ω(node_i)
    scen_values = [[None]*len(nodes) for _ in range(S)]
    t_init_q0 = perf_counter()
    for i, node in enumerate(nodes):
        for ω in range(S):
            scen_values[ω][i] = evaluate_Q_at(base_bundles[ω], first_vars_list[ω], node)
    timing["init_Q_time"] = perf_counter() - t_init_q0


    # cache f_ω(node_i)
    scen_values = [[None]*len(nodes) for _ in range(S)]
    t_init_q0 = perf_counter()
    for i, node in enumerate(nodes):
        for ω in range(S):
            scen_values[ω][i] = evaluate_Q_at(base_bundles[ω], first_vars_list[ω], node)
    timing["init_Q_time"] = perf_counter() - t_init_q0

    # ==== NEW: exact optima for validation / plotting ====
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



    it = 0
    stop_due_to_collision = False
    ms_cache = {}   # (scene_idx, sorted(vert_idx)) -> (ms_val, new_pt_ms, c_val, c_pt)

    # === NEW: helper to nudge candidate point slightly into simplex interior (if necessary) ===
    def _snap_feature(cand_pt, rec,
                    tol_vertex=1e-1,
                    tol_edge=1e-1,
                    tol_face=1e-1):
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
    csv_path = "simplex_result.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Time (s)", "# Nodes", "LB", "UB", "Rel. Gap", "Abs. Gap"])

    while len(nodes) < target_nodes:
        t_iter0 = perf_counter()
        tracker.start_iter(it)

        # initialize timing slots for this iteration
        timing["iter_total_time"].append(0.0)
        timing["iter_ms_time"].append(0.0)
        timing["iter_Q_new_time"].append(0.0)
        timing_idx = len(timing["iter_total_time"]) - 1

        ms_prev_calls = [len(b.solve_time_hist) for b in ms_bundles]

        # 1) Global UB
        f_sum_per_node = [
            sum(scen_values[s][i] for s in range(S))
            for i in range(len(nodes))
        ]
        ub_idx = int(np.argmin(f_sum_per_node))
        UB_global = float(f_sum_per_node[ub_idx])
        UB_node = tuple(nodes[ub_idx])

        # 2) Evaluate all tetrahedrons (single scene)
        t_ms0 = perf_counter()
        tri, per_tet = evaluate_all_tetra(
            nodes, scen_values, ms_bundles, first_vars_list,
            ms_cache=ms_cache, cache_on=True, tracker=tracker,
            tet_mesh=tet_mesh,          # === NEW: use incremental mesh
        )
        t_ms = perf_counter() - t_ms0
        timing["iter_ms_time"][timing_idx] = t_ms

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


        # 3) active mask (Filter by UB + Shape Quality)
        active_mask = {
            r["simplex_index"]: (r["LB"] <= UB_global + active_tol)
            for r in per_tet
        }
        q_cut = 1e-9
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

        # === Count simplices and active ones this round ===
        n_tets = len(per_tet)
        n_active = sum(1 for r in per_tet if active_mask.get(r["simplex_index"], False))
        if verbose:
            print(f"[Iter {it}] #simplices = {n_tets}, #active = {n_active}, "
                  f"bad-quality cut = {bad_quality_count}")
            
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

        # 5) LB_global
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


        # ======= Consistency check: selected simplex LB(=LB_global) should not exceed UB_global =======
        # Theoretically surrogate is underestimator, so LB_global <= UB_global should always hold
        # Add small tolerance to prevent pure numerical error
        if LB_global > UB_global + 1e-8:
            raise RuntimeError(
                f"[run_pid_simplex_3d] LB of selected simplex exceeds UB at iter={it}, "
                f"simplex={lb_simp_rec['simplex_index']}: "
                f"LB={LB_global:.6e}, UB={UB_global:.6e}"
            )
        # =======================================================================



        # === Debug: Drawblue-red scatter cloud from LB-simplex ===
        # ==================================================
        if DEBUG_LB_SIMPLEX_SCATTER and lb_simp_rec is not None:
            try:
                print(f"[DEBUG] about to call _debug_plot_lb_simplex_scatter, iter={it}, "
                    f"simplex={lb_simp_rec.get('simplex_index','?')}")
                _debug_plot_lb_simplex_scatter(
                    lb_simp_rec,
                    nodes=nodes,
                    scen_values=scen_values,
                    it=it,
                    grid_n=DEBUG_LB_SIMPLEX_SCATTER_GRID_N,
                    outdir=DEBUG_LB_SIMPLEX_SCATTER_OUTDIR,
                )
            except Exception as e:
                print(f"[DEBUG] WRONG: {e}")


        ms_b      = float(lb_simp_rec["ms"])
        ms_b_simp = int(lb_simp_rec["simplex_index"])

        lb_simp_idx = int(lb_simp_rec["simplex_index"]) 
        # ==================================================
        # ==================================================


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


        # === Print the current round's optimality gap ===
        gap_abs = float(UB_global - LB_global)
        gap_pct = (gap_abs / (abs(UB_global) + 1e-16)) * 100.0
        if verbose:
            print(f"[Iter {it}] Optimality abs-gap: {gap_abs:.6e} ({gap_pct:.3f}%)")

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

        # === Append current iteration to CSV ===
        iter_time = iter_time_hist[-1]
        n_nodes = node_count[-1]
        lb_val = LB_hist[-1] / S
        ub_val = UB_hist[-1] / S
        abs_gap = (UB_hist[-1] - LB_hist[-1]) / S
        rel_gap = abs_gap / (abs(ub_val) + 1e-16)
        with open(csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([f"{iter_time:.3f}", n_nodes, f"{lb_val:.9f}", f"{ub_val:.9f}", f"{rel_gap*100:.4f}%", f"{abs_gap:.5f}"])



        # === Convergence stopping condition ===
        if gap_stop_tol is not None and float(gap_stop_tol) > 0.0:
            gap_rel = float(UB_global - LB_global) / (abs(UB_global) + 1e-16)
            if gap_rel <= float(gap_stop_tol):
                if verbose:
                    print(f"[Iter {it}] Stop: UB-LB gap {gap_rel:.6e} <= tol {float(gap_stop_tol):.6e}.")
                    t_iter = perf_counter() - t_iter0
                    timing["iter_total_time"][timing_idx] = t_iter
                break

        # === Time limit stopping condition ===
        if time_limit is not None and time_limit > 0:
            elapsed = perf_counter() - t_start
            if elapsed >= time_limit:
                if verbose:
                    print(f"[Iter {it}] Stop: Time limit reached ({elapsed:.2f}s >= {time_limit:.2f}s).")
                    t_iter = perf_counter() - t_iter0
                    timing["iter_total_time"][timing_idx] = t_iter
                break

        # 8) print
        simp_with_min = [r["simplex_index"] for r in per_tet if ub_idx in r["vert_idx"]]
        if verbose:
            print(f"[Iter {it}] Active simplex ratio = {active_ratio:.6f}")
            print(f"[Iter {it}] UB node {UB_node} is in simplices {sorted(simp_with_min)}")
            msb_src = f"T{ms_b_simp}" if ms_b_simp is not None else "N/A"
            print(f"[Iter {it}] LB = {LB_global:.6f} = UB({UB_global:.6f}) + ms_b({ms_b:.3e}) from {msb_src}")

        # 9) next node candidate ranking & selection
        # ------------------------------------------------
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
                c_pts = rec.get("c_point_per_scene", [None] * len(ms_vals))
                
                val_list = []
                pts_list = []
                sources = []

                for s in range(len(ms_vals)):
                    ms_pt = ms_pts[s]
                    c_pt = c_pts[s]
                    
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
        if true_opt_points is not None:
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
        if (plot_every is not None) and (it % plot_every == 0):
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

        # add node and evaluate
        t_q0 = perf_counter()
        new_vals = []
        scene_times = [[] for _ in range(S)]
        q_call_cnt = 0  
        for s in range(S):
            t0_q = perf_counter()
            print(f"[Iter {it}] evaluating Q for scenario {s}")
            val = evaluate_Q_at(base_bundles[s], first_vars_list[s], new_node)
            dt_q = perf_counter() - t0_q
            new_vals.append(val)
            scene_times[s].append(dt_q)
            q_call_cnt += 1
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

        # === NEW: update mesh by star-subdividing the simplex that generated new_node ===
        if chosen_cand is not None:
            selection_reason_hist.append(chosen_cand.get("pt_source", "unknown")) # Record reason
            sid = int(chosen_cand["simplex_index"])
            loc_type = chosen_cand.get("loc_type", "interior")
            loc_info = chosen_cand.get("loc_info", None)

            # Assign code to split type: 1=interior, 2=edge, 3=face
            split_code = {"interior": 1, "edge": 2, "face": 3}.get(loc_type, 0)
            if verbose:
                print(f"[Iter {it}] subdivision type = {loc_type} "
                      f"(code={split_code}) on simplex T{sid}")

            if loc_type == "edge" and loc_info is not None:
                edge_verts = loc_info["edge_verts"]   # Here follows the meaning defined in _snap_feature
                if verbose:
                    print(f"           edge local verts = {edge_verts}")
                tet_mesh.subdivide_edge(sid, new_node_index, edge_verts)
            elif loc_type == "face" and loc_info is not None:
                face_verts = loc_info["face_verts"]
                if verbose:
                    print(f"           face local verts = {face_verts}")
                tet_mesh.subdivide_face(sid, new_node_index, face_verts)
            else:
                # Default treated as interior point, star subdivision
                tet_mesh.subdivide(sid, new_node_index)

        add_node_hist.append(new_node)
        if verbose:
            print(f"[Iter {it}] Elapsed: {perf_counter() - t_iter0:.3f}s")


        t_iter = perf_counter() - t_iter0
        timing["iter_total_time"][timing_idx] = t_iter 
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



# =====================================================================
# Debug helper: plot LB-determining simplex with blue/red scatter points
# =====================================================================

def _debug_plot_lb_simplex_scatter(lb_rec, nodes, scen_values, it,
                                   grid_n=8,
                                   outdir=None,
                                   n_samples=None,
                                   random_seed=None):
    import os
    import math
    import numpy as np
    import plotly.graph_objects as go

    print(f"[DEBUG] _debug_plot_lb_simplex_scatter: enter, iter={it}, "
          f"simplex={lb_rec.get('simplex_index', '?')}")

    # ---- 0) Vertex coordinates ----
    vert_idx = lb_rec["vert_idx"]
    verts = np.asarray([nodes[i] for i in vert_idx], dtype=float)  # (4,3)

    # ---- 1) fverts_sum[j] = sum_s Q_s(vertex j) ----
    S = len(scen_values)
    fverts_sum = []
    for v_idx in vert_idx:
        ssum = 0.0
        for s in range(S):
            ssum += float(scen_values[s][v_idx])
        fverts_sum.append(ssum)
    fverts_sum = np.asarray(fverts_sum, dtype=float)

    # ---- 2) ms_total & c_total ----
    ms_scene = lb_rec.get("ms_per_scene", [])
    if ms_scene:
        ms_total = float(sum(ms_scene))
    else:
        ms_total = float(lb_rec.get("ms", 0.0))

    c_scene = list(lb_rec.get("c_per_scene", []))
    finite_c = [float(c) for c in c_scene if math.isfinite(c)]
    if finite_c:
        c_total = float(sum(finite_c))
    else:
        print("[DEBUG] LB simplex no finite c_s, skip")
        return

    # === Print 4 vertices using "same source data as plot" here ===
    print(f"[DEBUG] (from scatter helper) LB simplex {lb_rec.get('simplex_index','?')}:")
    for j, node_idx in enumerate(vert_idx):
        sum_As = float(fverts_sum[j])         # sum_s Q_s(vertex j)
        sum_As_ms = sum_As + ms_total         # sum_s Q_s(vertex j) + sum_s ms_s
        print(
            f"    vert {j} (node {node_idx}): "
            f"sum(As+ms) = {sum_As_ms:.6e}, sum(c_s) = {c_total:.6e}"
        )

    # ---- 3) Sample count ----
    if n_samples is None:
        n_samples = globals().get("DEBUG_LB_SIMPLEX_SCATTER_NPTS", None)
    if n_samples is None:
        n_samples = int(grid_n ** 3)
    n_samples = int(n_samples)

    if outdir is None:
        outdir = globals().get("DEBUG_LB_SIMPLEX_SCATTER_OUTDIR", None)

    print(f"[DEBUG] _debug_plot_lb_simplex_scatter: n_samples={n_samples}, "
          f"outdir={outdir}")

    # ---- 4) Random sampling + coloring ----
    xs, ys, zs, colors = [], [], [], []
    tol = 1e-9

    if random_seed is None:
        rng = np.random.default_rng()
    else:
        rng = np.random.default_rng(seed=int(random_seed))

    n_total = 0
    n_blue = 0

    for _ in range(n_samples):
        lamb = rng.dirichlet([1.0, 1.0, 1.0, 1.0])
        p = lamb[0] * verts[0] + lamb[1] * verts[1] + lamb[2] * verts[2] + lamb[3] * verts[3]

        A_sum = float(np.dot(fverts_sum, lamb))
        val = A_sum + ms_total

        n_total += 1
        if val > c_total + tol:
            color = "blue"
            n_blue += 1
        else:
            color = "red"

        xs.append(float(p[0]))
        ys.append(float(p[1]))
        zs.append(float(p[2]))
        colors.append(color)

    blue_frac = n_blue / n_total if n_total > 0 else float("nan")
    print(
        f"[DEBUG] Iter {it}, simplex {lb_rec.get('simplex_index', '?')}: "
        f"Points with A_sum+ms_total > c_total = {n_blue}/{n_total} "
        f"({blue_frac:.3f})"
    )

    # ---- 5) Draw edges + scatter points (same as yours) ----
    fig = go.Figure()
    edges = [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)]
    for idx, (a, b) in enumerate(edges):
        fig.add_trace(go.Scatter3d(
            x=[verts[a, 0], verts[b, 0]],
            y=[verts[a, 1], verts[b, 1]],
            z=[verts[a, 2], verts[b, 2]],
            mode="lines",
            line=dict(width=4),
            name="tet_edge" if idx == 0 else None,
            showlegend=(idx == 0),
        ))

    fig.add_trace(go.Scatter3d(
        x=xs,
        y=ys,
        z=zs,
        mode="markers",
        marker=dict(size=3, color=colors, opacity=0.6),
        name="A_sum+ms_total vs c_total",
    ))

    fig.update_layout(
        title=f"LB simplex scatter (iter={it}, simplex={lb_rec.get('simplex_index', '?')})",
        scene=dict(
            xaxis_title="Kp",
            yaxis_title="Ki",
            zaxis_title="Kd",
            aspectmode="data",
        ),
        legend=dict(x=0.02, y=0.98),
    )

    if outdir is not None:
        os.makedirs(outdir, exist_ok=True)
        path = os.path.join(outdir, f"lb_simplex_iter_{it}.html")
        fig.write_html(path)
        print(f"[DEBUG] LB simplex scatter saved to: {path}")
    else:
        print("[DEBUG] _debug_plot_lb_simplex_scatter: calling fig.show()")
        fig.show()

    print(f"[DEBUG] _debug_plot_lb_simplex_scatter: exit, iter={it}")
