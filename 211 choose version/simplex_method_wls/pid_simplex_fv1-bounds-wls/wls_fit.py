# wls_fit.py
"""
Weighted Least Squares (WLS) affine fitting for simplex lower-bound computation.

Given sample points with per-scenario Q-values, fit an affine function
    A_s(K) = b0 + b1*Kp + b2*Ki + b3*Kd
per scenario using WLS.  Then evaluate A_s at the 4 simplex vertices
to produce predicted vertex values that, when passed through the existing
barycentric update_tetra / LP pipeline, reproduce the same affine A_s
(because an affine function is uniquely determined by its values at 4
non-degenerate tetrahedron vertices).
"""

import numpy as np


def fit_wls_affine(points_xyz, values, weights=None):
    """
    Fit an affine model  y = b0 + b1*x1 + b2*x2 + b3*x3  via WLS.

    Parameters
    ----------
    points_xyz : array-like, shape (N, 3)
        Coordinates of the sample points.
    values : array-like, shape (N,)
        Observed y-values at each point.
    weights : array-like, shape (N,) or None
        Non-negative weights.  If None, uniform weights (OLS).

    Returns
    -------
    beta : ndarray, shape (4,)
        Coefficients [b0, b1, b2, b3].
        Returns None if the system is rank-deficient or N < 4.
    """
    X = np.asarray(points_xyz, dtype=float)
    y = np.asarray(values, dtype=float)
    N = X.shape[0]
    if N < 4:
        return None

    # Design matrix: [1, Kp, Ki, Kd]
    A = np.column_stack([np.ones(N), X])  # (N, 4)

    if weights is not None:
        w = np.asarray(weights, dtype=float)
        # sqrt-weighted system: W^{1/2} A beta = W^{1/2} y
        sw = np.sqrt(np.maximum(w, 0.0))
        A = A * sw[:, None]
        y = y * sw

    # Solve via least squares
    result = np.linalg.lstsq(A, y, rcond=None)
    beta = result[0]
    rank = result[2]

    # Check rank: need rank == 4 for a unique affine fit in 3D
    if rank < 4:
        return None

    return beta


def eval_affine(beta, x):
    """
    Evaluate the affine function at a single point.

    Parameters
    ----------
    beta : ndarray, shape (4,)
        Coefficients [b0, b1, b2, b3].
    x : array-like, shape (3,)
        Point (Kp, Ki, Kd).

    Returns
    -------
    float
        b0 + b1*Kp + b2*Ki + b3*Kd
    """
    x = np.asarray(x, dtype=float)
    return float(beta[0] + beta[1]*x[0] + beta[2]*x[1] + beta[3]*x[2])


def compute_wls_fverts_for_scene(
    tet_vertices,
    q_at_verts,
    extra_points,
    extra_q_values,
    q_threshold=2.0,
):
    """
    For ONE scenario, build sample set, filter by Q < threshold,
    fit WLS affine, and return predicted vertex values.

    Parameters
    ----------
    tet_vertices : list of 4 tuples (Kp, Ki, Kd)
        The 4 simplex vertices.
    q_at_verts : list of 4 floats
        True Q_s values at the 4 vertices.
    extra_points : list of tuples (Kp, Ki, Kd)
        Additional sample points (e.g., centroid, avg cs point).
    extra_q_values : list of floats
        True Q_s values at each extra point (same length as extra_points).
    q_threshold : float
        Only use points with Q_s(p) < threshold.

    Returns
    -------
    fverts_pred : list of 4 floats or None
        Predicted A_s(vertex_j) for j=0..3.
        None if fallback to raw vertex values is needed
        (insufficient points or rank-deficient).
    """
    # Build full sample set
    all_points = list(tet_vertices) + list(extra_points)
    all_q = list(q_at_verts) + list(extra_q_values)

    # Filter by Q < threshold (per scenario)
    filtered_pts = []
    filtered_q = []
    for pt, q in zip(all_points, all_q):
        if q < q_threshold:
            filtered_pts.append(pt)
            filtered_q.append(q)

    if len(filtered_pts) < 4:
        return None  # fallback

    # Fit WLS affine (uniform weights)
    beta = fit_wls_affine(filtered_pts, filtered_q, weights=None)
    if beta is None:
        return None  # fallback (rank-deficient)

    # Evaluate at the 4 vertices
    fverts_pred = [eval_affine(beta, v) for v in tet_vertices]
    return fverts_pred
