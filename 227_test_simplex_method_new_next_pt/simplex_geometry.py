"""
simplex_geometry.py — Dimension-general simplex geometry utilities.

Supports d-dimensional simplices (d+1 vertices in R^d) for the generalized
simplex method.  All functions accept arrays of shape (d+1, d).

Functions
---------
simplex_volume(V)
    Unsigned volume of a d-simplex.
barycentric_coordinates(p, V)
    Barycentric coordinates of point p w.r.t. simplex V.
simplex_quality(V)
    Quality metric (generalizes tet_quality from 3D).
is_degenerate(V, tol)
    Check whether simplex is degenerate (near-zero volume).
snap_to_feature(p, V, vert_idx, tol_vertex, tol_edge, tol_face)
    Classify point as interior/vertex/edge/face and return snapped coords.
"""

from __future__ import annotations

import logging
import math
from itertools import combinations

import numpy as np
import _safe_linalg

logger = logging.getLogger(__name__)

# Tolerance for degenerate simplex detection
_DEGEN_TOL = 1e-15


def simplex_volume(V) -> float:
    """Unsigned volume of a d-simplex defined by d+1 vertices.

    Parameters
    ----------
    V : array-like, shape (d+1, d)

    Returns
    -------
    float
        The unsigned d-dimensional volume (area for d=2, volume for d=3, etc.).

    Notes
    -----
    Uses |det(edge_matrix)| / d!  where edge_matrix is built from edges
    emanating from v0.
    """
    V = np.asarray(V, dtype=float)
    n_verts, d = V.shape
    if n_verts != d + 1:
        raise ValueError(f"Expected {d+1} vertices for dim={d}, got {n_verts}")

    # Edge matrix: columns are v1-v0, v2-v0, ..., vd-v0
    edge_mat = (V[1:] - V[0]).T   # shape (d, d)
    return float(abs(_safe_linalg.det(edge_mat)) / math.factorial(d))


def barycentric_coordinates(p, V) -> np.ndarray:
    """Compute barycentric coordinates of point p w.r.t. simplex V.

    Parameters
    ----------
    p : array-like, shape (d,)
    V : array-like, shape (d+1, d)

    Returns
    -------
    np.ndarray, shape (d+1,)
        Barycentric coordinates [lam0, lam1, ..., lamd] where sum = 1.
        lam0 = 1 - sum(lam1..lamd).

    Notes
    -----
    Solves  M @ alpha = p - v0  where M = [v1-v0, ..., vd-v0].T
    Then lam = [1-sum(alpha), alpha[0], ..., alpha[d-1]].

    For degenerate simplices, uses lstsq fallback with clipping.
    """
    V = np.asarray(V, dtype=float)
    p = np.asarray(p, dtype=float)
    n_verts, d = V.shape

    # Edge matrix
    M = (V[1:] - V[0]).T   # shape (d, d)
    rhs = p - V[0]          # shape (d,)

    try:
        alpha = _safe_linalg.solve(M, rhs)
    except np.linalg.LinAlgError:
        # Degenerate: use uniform barycentric (centroid approx)
        alpha = np.full(M.shape[1], 1.0 / (M.shape[1] + 1))

    lam0 = 1.0 - alpha.sum()
    lambdas = np.empty(n_verts, dtype=float)
    lambdas[0] = lam0
    lambdas[1:] = alpha
    return lambdas


def simplex_quality(V) -> float:
    """Quality metric for a d-simplex (edge-ratio based).

    A generalization of tet_quality: returns
        d! * vol / sum(edge_lengths^d)
    which is 0 for degenerate simplices and approaches a maximum for
    regular simplices.

    Parameters
    ----------
    V : array-like, shape (d+1, d)

    Returns
    -------
    float
    """
    V = np.asarray(V, dtype=float)
    n_verts, d = V.shape
    vol = simplex_volume(V)
    edges = [float(np.linalg.norm(V[i] - V[j]))
             for i, j in combinations(range(n_verts), 2)]
    denom = float(np.sum(np.power(edges, d))) + 1e-16
    return float(math.factorial(d) * vol / denom)


def is_degenerate(V, tol: float = _DEGEN_TOL) -> bool:
    """Check if a simplex is degenerate (near-zero volume).

    Parameters
    ----------
    V : array-like, shape (d+1, d)
    tol : float

    Returns
    -------
    bool
    """
    return simplex_volume(V) < tol


def vol_tolerance(pts, dim: int) -> float:
    """Compute a reasonable volume tolerance for degeneracy checks.

    Based on the diameter of the point cloud and the dimension.

    Parameters
    ----------
    pts : array-like, shape (N, d)
    dim : int

    Returns
    -------
    float
    """
    pts = np.asarray(pts, dtype=float)
    mins = pts.min(axis=0)
    maxs = pts.max(axis=0)
    diam = float(np.linalg.norm(maxs - mins))
    return 1e-12 * max(diam ** dim, 1.0)


def snap_to_feature(cand_pt, verts, vert_idx,
                    tol_vertex=1e-12, tol_edge=1e-12, tol_face=1e-12):
    """Classify and snap a candidate point to the appropriate simplex feature.

    Generalizes _snap_feature from 3D to arbitrary dimension.

    Parameters
    ----------
    cand_pt : array-like, shape (d,)
    verts : array-like, shape (d+1, d)
        Simplex vertex coordinates.
    vert_idx : list of int
        Global node indices of the vertices.
    tol_vertex, tol_edge, tol_face : float
        Thresholds for classifying near-zero barycentric coordinates.

    Returns
    -------
    snapped_pt : tuple of float
        The (possibly adjusted) point coordinates.
    feature : str
        One of "interior", "vertex", "edge", "face", or "sub_face".
    info : dict or None
        Additional info (lambdas, edge_verts, face_verts, etc.).
    """
    verts = np.asarray(verts, dtype=float)
    p = np.asarray(cand_pt, dtype=float)
    n_verts, d = verts.shape  # n_verts = d + 1

    lambdas = barycentric_coordinates(p, verts)

    small = lambdas < tol_vertex
    n_small = int(small.sum())
    n_big = n_verts - n_small

    # All or almost all coordinates small => near vertex, push inside
    if n_big <= 1:
        centroid = verts.mean(axis=0)
        new_p = p + 1e-3 * (centroid - p)
        return tuple(map(float, new_p)), "interior", {"lambdas": lambdas}

    # Exactly 2 big coordinates => edge
    if n_big == 2:
        big_idx = np.where(~small)[0]
        lam_big = lambdas[big_idx].copy()
        lam_big /= lam_big.sum()
        snapped = sum(lam_big[k] * verts[big_idx[k]] for k in range(2))
        edge_verts_global = tuple(vert_idx[j] for j in big_idx)
        return (tuple(map(float, snapped)),
                "edge",
                {"edge_verts": edge_verts_global, "lambdas": lambdas})

    # Exactly d big coordinates (n_small == 1) => face (codim-1 facet)
    if n_small == 1:
        face_idx = np.where(~small)[0]
        lam_face = lambdas[face_idx].copy()
        lam_face /= lam_face.sum()
        snapped = sum(lam_face[k] * verts[face_idx[k]] for k in range(len(face_idx)))
        face_verts_global = tuple(vert_idx[j] for j in face_idx)
        return (tuple(map(float, snapped)),
                "face",
                {"face_verts": face_verts_global, "lambdas": lambdas})

    # For d > 3: between 3 and d-1 big coords => sub-face
    # (e.g., a 2-face of a 4-simplex)
    if 2 < n_big < n_verts - 1:
        big_idx = np.where(~small)[0]
        lam_big = lambdas[big_idx].copy()
        lam_big /= lam_big.sum()
        snapped = sum(lam_big[k] * verts[big_idx[k]] for k in range(len(big_idx)))
        sub_face_verts = tuple(vert_idx[j] for j in big_idx)
        return (tuple(map(float, snapped)),
                "sub_face",
                {"sub_face_verts": sub_face_verts, "lambdas": lambdas, "n_big": n_big})

    # Interior point (all lambdas reasonably large)
    return tuple(map(float, p)), "interior", {"lambdas": lambdas}
