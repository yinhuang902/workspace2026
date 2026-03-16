"""
quadratic_surrogate.py — Full symmetric quadratic interpolation for the
simplex-method surrogate.

Provides:
    - Basis size computation
    - Vandermonde-like matrix construction for symmetric quadratic basis
    - Exact interpolation solver (M @ coeffs = fvals)
    - Quadratic evaluation (scalar and Pyomo expression)
    - Deterministic interior candidate generator (barycentric, center-biased)
    - Random interior simplex point generator
    - Full interpolation-set builder with numerical acceptance checks
    - Per-scenario logging class (QuadInterpLogger)
"""

from __future__ import annotations

import math
import os
from itertools import combinations
from time import perf_counter

import numpy as np


# ───────────────────────── Thresholds / defaults ─────────────────────────

COND_THRESHOLD = 1e12          # max acceptable condition number
MIN_INTERP_DIST = 1e-8         # min pairwise distance between interp points
MAX_RANDOM_ATTEMPTS = 50       # random interior sampling budget
Q_INVALID_THRESHOLD = 1e10 - 1.0  # Q_s >= this is treated as infeasible
BARY_EPS = 1e-4                # clip barycentric coords above this for "interior"


# ───────────────────────── Basis helpers ─────────────────────────────────

def quad_basis_size(d: int) -> int:
    """Number of free parameters in a full symmetric quadratic in d dims.

    m = 1 (constant) + d (linear) + d*(d+1)//2 (quadratic)
    """
    return 1 + d + d * (d + 1) // 2


def _quad_basis_row(x, d: int) -> np.ndarray:
    """Evaluate the symmetric quadratic basis at a single point x ∈ R^d.

    Basis ordering:
        [1, x_0, x_1, ..., x_{d-1},
         x_0^2, x_0*x_1, ..., x_0*x_{d-1},
         x_1^2, x_1*x_2, ...,
         ..., x_{d-1}^2]

    Returns a 1-D array of length m = quad_basis_size(d).
    """
    m = quad_basis_size(d)
    row = np.empty(m, dtype=float)
    row[0] = 1.0
    # linear terms
    for i in range(d):
        row[1 + i] = x[i]
    # quadratic terms
    idx = 1 + d
    for i in range(d):
        for j in range(i, d):
            row[idx] = x[i] * x[j]
            idx += 1
    return row


def build_quad_vandermonde(points, d: int) -> np.ndarray:
    """Build the m × m Vandermonde-like matrix for symmetric quadratic basis.

    Parameters
    ----------
    points : array-like, shape (m, d)
    d : int — first-stage dimension

    Returns
    -------
    M : np.ndarray, shape (m, m)
    """
    m = quad_basis_size(d)
    points = np.asarray(points, dtype=float)
    if points.shape != (m, d):
        raise ValueError(f"Expected {m} points of dim {d}, got shape {points.shape}")
    M = np.empty((m, m), dtype=float)
    for k in range(m):
        M[k, :] = _quad_basis_row(points[k], d)
    return M


# ───────────────────────── Interpolation solver ──────────────────────────

def solve_interpolation(M: np.ndarray, fvals) -> tuple:
    """Solve the exact interpolation system M @ coeffs = fvals.

    Returns
    -------
    (c_qs, g_vec, H_mat) where
        c_qs   : float           — constant term
        g_vec  : np.ndarray (d,) — linear coefficients
        H_mat  : np.ndarray (d,d) — symmetric Hessian
    """
    fvals = np.asarray(fvals, dtype=float)
    m = M.shape[0]
    coeffs = np.linalg.solve(M, fvals)

    # Reverse-map from coefficient vector to (c, g, H)
    # Infer d from m = 1 + d + d*(d+1)/2
    # Solve: d^2/2 + 3d/2 + 1 = m  =>  d = (-3 + sqrt(9 + 8*(m-1))) / 2
    disc = 9 + 8 * (m - 1)
    d = int((-3 + math.sqrt(disc)) / 2 + 0.5)
    assert quad_basis_size(d) == m, f"Cannot infer d from m={m}"

    c_qs = float(coeffs[0])
    g_vec = coeffs[1:1 + d].copy()

    H_mat = np.zeros((d, d), dtype=float)
    idx = 1 + d
    for i in range(d):
        for j in range(i, d):
            if i == j:
                # basis term is x_i^2, coefficient is 0.5*H[i,i]
                # because q_s = ... + 0.5*H[i,i]*x_i^2
                # but our basis uses x_i^2 directly, so
                # coefficient_of_x_i^2 = 0.5 * H[i,i]
                # => H[i,i] = 2 * coefficient
                H_mat[i, i] = 2.0 * coeffs[idx]
            else:
                # basis term is x_i*x_j, coefficient is 0.5*(H[i,j]+H[j,i]) = H[i,j]
                # because H is symmetric and our basis uses x_i*x_j once
                # q_s = ... + 0.5*H[i,j]*x_i*x_j + 0.5*H[j,i]*x_j*x_i
                #      = ... + H[i,j]*x_i*x_j (since H[i,j]=H[j,i])
                # So coefficient_of_x_i*x_j = H[i,j]
                H_mat[i, j] = coeffs[idx]
                H_mat[j, i] = coeffs[idx]
            idx += 1

    return c_qs, g_vec, H_mat


# ───────────────────────── Evaluation ────────────────────────────────────

def eval_quad_at(c_qs: float, g_vec, H_mat, x) -> float:
    """Evaluate q_s(x) = c + g^T x + 0.5 x^T H x."""
    x = np.asarray(x, dtype=float)
    g = np.asarray(g_vec, dtype=float)
    H = np.asarray(H_mat, dtype=float)
    return float(c_qs + g @ x + 0.5 * x @ H @ x)


# ───────────────────────── Candidate generators ──────────────────────────

def generate_interior_candidates(V, d: int) -> list[np.ndarray]:
    """Generate deterministic interior simplex candidates, center-biased.

    Priority:
        1. Centroid
        2. 3/4 centroid + 1/4 vertex j  (d+1 points)
        3. 2/3 centroid + 1/3 vertex j  (d+1 points)
        4. 1/2 centroid + 1/2 vertex j  (d+1 points)
        5. Midpoints of priority-2 pairs (O(d^2) points)
        6. 1/2 centroid + 1/2 edge midpoint  (O(d^2) points)

    All points have strictly positive barycentric coordinates (true interior).

    Parameters
    ----------
    V : array-like, shape (d+1, d)  — simplex vertices
    d : int — dimension

    Returns
    -------
    list of np.ndarray, each shape (d,)
    """
    V = np.asarray(V, dtype=float)
    n_verts = d + 1
    centroid = V.mean(axis=0)  # (d,)

    candidates = []

    # Priority 1: centroid
    candidates.append(centroid.copy())

    # Priority 2: 3/4 centroid + 1/4 vertex
    p2_pts = []
    for j in range(n_verts):
        pt = 0.75 * centroid + 0.25 * V[j]
        p2_pts.append(pt)
        candidates.append(pt)

    # Priority 3: 2/3 centroid + 1/3 vertex
    for j in range(n_verts):
        pt = (2.0 / 3.0) * centroid + (1.0 / 3.0) * V[j]
        candidates.append(pt)

    # Priority 4: 1/2 centroid + 1/2 vertex
    for j in range(n_verts):
        pt = 0.5 * centroid + 0.5 * V[j]
        candidates.append(pt)

    # Priority 5: midpoints of priority-2 pairs
    for i in range(len(p2_pts)):
        for j in range(i + 1, len(p2_pts)):
            pt = 0.5 * (p2_pts[i] + p2_pts[j])
            candidates.append(pt)

    # Priority 6: 1/2 centroid + 1/2 edge midpoint
    for i, j in combinations(range(n_verts), 2):
        edge_mid = 0.5 * (V[i] + V[j])
        pt = 0.5 * centroid + 0.5 * edge_mid
        candidates.append(pt)

    return candidates


def random_simplex_point(V) -> np.ndarray:
    """Generate a random interior point in the simplex via Dirichlet(1,...,1).

    Parameters
    ----------
    V : array-like, shape (d+1, d)

    Returns
    -------
    np.ndarray, shape (d,)
    """
    V = np.asarray(V, dtype=float)
    n_verts = V.shape[0]
    # Sample barycentric coordinates from Dirichlet(1,...,1)
    bary = np.random.dirichlet(np.ones(n_verts))
    # Clip to keep strictly interior
    bary = np.clip(bary, BARY_EPS, None)
    bary /= bary.sum()
    return bary @ V


# ───────────────────────── Numerical checks ──────────────────────────────

def _min_pairwise_dist(points: np.ndarray) -> float:
    """Minimum pairwise Euclidean distance among rows of points."""
    n = len(points)
    if n < 2:
        return float('inf')
    min_d = float('inf')
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(points[i] - points[j]))
            if d < min_d:
                min_d = d
    return min_d


def _is_too_close(pt: np.ndarray, existing: list[np.ndarray],
                  tol: float = MIN_INTERP_DIST) -> bool:
    """Check if pt is within tol of any point in existing."""
    for ex in existing:
        if np.linalg.norm(pt - ex) < tol:
            return True
    return False


def check_interpolation_matrix(M: np.ndarray, m: int) -> tuple[bool, int, float, str | None]:
    """Check rank and condition of the interpolation matrix.

    Returns
    -------
    (ok, rank, cond, reason)
    """
    rank = int(np.linalg.matrix_rank(M))
    try:
        cond = float(np.linalg.cond(M))
    except Exception:
        cond = float('inf')

    if rank < m:
        return False, rank, cond, "rank_deficient_matrix"
    if cond > COND_THRESHOLD:
        return False, rank, cond, "ill_conditioned_matrix"
    return True, rank, cond, None


# ───────────────────────── Full interpolation pipeline ───────────────────

def build_interpolation_set(
    tet_vertices,
    cs_point,
    cs_qval,
    vertex_qvals,
    eval_q_fn,
    d: int,
    min_dist: float = MIN_INTERP_DIST,
    max_random: int = MAX_RANDOM_ATTEMPTS,
) -> dict:
    """Build the interpolation set for one scenario/simplex.

    Parameters
    ----------
    tet_vertices : array-like, shape (d+1, d)
        Simplex vertices.
    cs_point : tuple | None
        The c_s minimizer point for this scenario (highest priority).
    cs_qval : float | None
        Q_s value at cs_point.
    vertex_qvals : list[float | None]
        Q_s values at each simplex vertex for this scenario.
        None or non-finite means the vertex is invalid for this scenario.
    eval_q_fn : callable(point) -> float
        Function to evaluate Q_s at an arbitrary interior point.
        Should return a finite float, or Q_max if infeasible.
    d : int
        First-stage dimension.
    min_dist : float
        Minimum distance between interpolation points.
    max_random : int
        Maximum random sampling attempts.

    Returns
    -------
    dict with keys:
        "success"       : bool
        "points"        : np.ndarray (m, d) if success, else None
        "fvals"         : np.ndarray (m,) if success, else None
        "quad_coeffs"   : (c_qs, g_vec, H_mat) if success, else None
        "rank"          : int
        "cond"          : float
        "reason"        : str | None  (failure reason)
        "diag"          : dict  (detailed diagnostics for logging)
    """
    V = np.asarray(tet_vertices, dtype=float)
    m = quad_basis_size(d)
    n_verts = d + 1

    diag = {
        "m_required": m,
        "cs_available": False,
        "cs_used": False,
        "vertices_used": [],
        "vertices_discarded": [],
        "det_candidates_tried": 0,
        "det_candidates_used": 0,
        "random_attempts": 0,
        "random_used": 0,
        "final_point_count": 0,
        "rank": 0,
        "cond": float('inf'),
        "accepted": False,
        "reason": None,
    }

    selected_points: list[np.ndarray] = []
    selected_fvals: list[float] = []

    def _add_point(pt: np.ndarray, fval: float, label: str) -> bool:
        """Try to add a point. Returns True if added."""
        if not math.isfinite(fval) or fval >= Q_INVALID_THRESHOLD:
            return False
        if _is_too_close(pt, selected_points, tol=min_dist):
            return False
        selected_points.append(pt.copy())
        selected_fvals.append(fval)
        return True

    # ---- Priority 1: cs point ----
    if cs_point is not None and cs_qval is not None:
        diag["cs_available"] = True
        cs_pt = np.asarray(cs_point, dtype=float)
        if math.isfinite(cs_qval) and cs_qval < Q_INVALID_THRESHOLD:
            if _add_point(cs_pt, cs_qval, "cs"):
                diag["cs_used"] = True

    if len(selected_points) >= m:
        return _finalize(selected_points, selected_fvals, d, m, diag)

    # ---- Priority 2: simplex vertices ----
    for j in range(n_verts):
        if len(selected_points) >= m:
            break
        fv = vertex_qvals[j]
        if fv is None or not math.isfinite(fv) or fv >= Q_INVALID_THRESHOLD:
            diag["vertices_discarded"].append(j)
            continue
        vpt = V[j].copy()
        if _add_point(vpt, fv, f"vertex_{j}"):
            diag["vertices_used"].append(j)
        else:
            diag["vertices_discarded"].append(j)

    if len(selected_points) >= m:
        return _finalize(selected_points, selected_fvals, d, m, diag)

    # ---- Priority 3: deterministic interior candidates ----
    det_cands = generate_interior_candidates(V, d)
    for cand in det_cands:
        if len(selected_points) >= m:
            break
        diag["det_candidates_tried"] += 1
        if _is_too_close(cand, selected_points, tol=min_dist):
            continue
        # Evaluate Q_s at this candidate
        fval = eval_q_fn(tuple(map(float, cand)))
        if not math.isfinite(fval) or fval >= Q_INVALID_THRESHOLD:
            continue
        if _add_point(cand, fval, "det_interior"):
            diag["det_candidates_used"] += 1

    if len(selected_points) >= m:
        return _finalize(selected_points, selected_fvals, d, m, diag)

    # ---- Priority 4: random interior points ----
    for attempt in range(max_random):
        if len(selected_points) >= m:
            break
        diag["random_attempts"] += 1
        rpt = random_simplex_point(V)
        if _is_too_close(rpt, selected_points, tol=min_dist):
            continue
        fval = eval_q_fn(tuple(map(float, rpt)))
        if not math.isfinite(fval) or fval >= Q_INVALID_THRESHOLD:
            continue
        if _add_point(rpt, fval, "random_interior"):
            diag["random_used"] += 1

    # ---- Check if we have enough points ----
    if len(selected_points) < m:
        diag["final_point_count"] = len(selected_points)
        diag["reason"] = "insufficient_valid_points"
        return {
            "success": False,
            "points": None,
            "fvals": None,
            "quad_coeffs": None,
            "rank": 0,
            "cond": float('inf'),
            "reason": "insufficient_valid_points",
            "diag": diag,
        }

    return _finalize(selected_points, selected_fvals, d, m, diag)


def _finalize(selected_points, selected_fvals, d, m, diag) -> dict:
    """Finalize the interpolation: build matrix, check, solve."""
    points_arr = np.array(selected_points[:m], dtype=float)
    fvals_arr = np.array(selected_fvals[:m], dtype=float)
    diag["final_point_count"] = m

    # Check minimum pairwise distance
    mpd = _min_pairwise_dist(points_arr)
    if mpd < MIN_INTERP_DIST:
        diag["reason"] = "duplicate_or_close_points"
        diag["min_pairwise_dist"] = mpd
        return {
            "success": False, "points": None, "fvals": None,
            "quad_coeffs": None, "rank": 0, "cond": float('inf'),
            "reason": "duplicate_or_close_points", "diag": diag,
        }

    # Build Vandermonde matrix
    M = build_quad_vandermonde(points_arr, d)

    # Check rank and condition
    ok, rank, cond, reason = check_interpolation_matrix(M, m)
    diag["rank"] = rank
    diag["cond"] = cond

    if not ok:
        diag["accepted"] = False
        diag["reason"] = reason
        return {
            "success": False, "points": points_arr, "fvals": fvals_arr,
            "quad_coeffs": None, "rank": rank, "cond": cond,
            "reason": reason, "diag": diag,
        }

    # Solve interpolation
    try:
        c_qs, g_vec, H_mat = solve_interpolation(M, fvals_arr)
    except Exception as e:
        diag["accepted"] = False
        diag["reason"] = f"solve_error: {e}"
        return {
            "success": False, "points": points_arr, "fvals": fvals_arr,
            "quad_coeffs": None, "rank": rank, "cond": cond,
            "reason": f"solve_error: {e}", "diag": diag,
        }

    diag["accepted"] = True
    diag["reason"] = None
    return {
        "success": True,
        "points": points_arr,
        "fvals": fvals_arr,
        "quad_coeffs": (c_qs, g_vec, H_mat),
        "rank": rank,
        "cond": cond,
        "reason": None,
        "diag": diag,
    }


# ───────────────────────── Logging ───────────────────────────────────────

class QuadInterpLogger:
    """Manages the quadratic interpolation diagnostic log file.

    Writes detailed per-iteration / per-simplex / per-scenario diagnostics
    to a text file.
    """

    def __init__(self, path: str):
        self.path = path
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(self.path, "w", encoding="utf-8") as f:
            f.write("# debug_quad_interp.txt — quadratic interpolation diagnostics\n")
            f.write("# Per iteration / per simplex / per scenario\n\n")

    def log_scenario(
        self,
        iter_num: int,
        simplex_id,
        scenario_id: int,
        cs_available: bool,
        cs_used: bool,
        vertices_used: list,
        vertices_discarded: list,
        det_tried: int,
        det_used: int,
        random_attempts: int,
        random_used: int,
        final_points,
        rank: int,
        cond: float,
        quad_available: bool,
        ms_solved: bool,
        skip_reason: str | None,
        contributed_via: str,
        surrogate_lb_included: str,
    ):
        """Append one scenario log entry."""
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(f"[Iter {iter_num}] simplex={simplex_id} scenario={scenario_id}\n")
                f.write(f"  cs_available={cs_available} cs_used={cs_used}\n")
                f.write(f"  vertices_used={vertices_used} vertices_discarded={vertices_discarded}\n")
                f.write(f"  det_candidates: tried={det_tried} used={det_used}\n")
                f.write(f"  random: attempts={random_attempts} used={random_used}\n")
                if final_points is not None:
                    pts_str = "; ".join(
                        f"({', '.join(f'{v:.6f}' for v in pt)})"
                        for pt in (final_points if hasattr(final_points, '__len__') else [])
                    )
                    f.write(f"  final_points=[{pts_str}]\n")
                else:
                    f.write(f"  final_points=None\n")
                f.write(f"  matrix_rank={rank} cond={cond:.3e}\n")
                f.write(f"  quad_available={quad_available}\n")
                f.write(f"  ms_solved={ms_solved}\n")
                if skip_reason:
                    f.write(f"  skip_reason={skip_reason}\n")
                f.write(f"  contributed_via={contributed_via}\n")
                f.write(f"  surrogate_lb_included={surrogate_lb_included}\n")
                f.write("\n")
        except Exception:
            pass  # never crash the algorithm over logging

    def log_simplex_summary(
        self,
        iter_num: int,
        simplex_id,
        n_quad_available: int,
        n_quad_unavailable: int,
        n_ms_solved: int,
        n_cs_only: int,
        simplex_status: str,
        lb_value: float,
    ):
        """Append a per-simplex summary line."""
        try:
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(
                    f"[Iter {iter_num}] simplex={simplex_id} SUMMARY: "
                    f"quad_avail={n_quad_available} quad_unavail={n_quad_unavailable} "
                    f"ms_solved={n_ms_solved} cs_only={n_cs_only} "
                    f"status={simplex_status} LB={lb_value:.6e}\n\n"
                )
        except Exception:
            pass
