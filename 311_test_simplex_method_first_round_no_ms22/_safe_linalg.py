"""
Safe linear algebra routines that avoid LAPACK calls.

On some Windows environments (e.g., certain conda builds of numpy),
the LAPACK backend (OpenBLAS) is broken and calling np.linalg.det(),
np.linalg.solve(), or np.linalg.lstsq() will crash the entire Python
process with exit code 1 and no traceback.

This module provides pure-numpy (BLAS-only) alternatives for small matrices
(up to 4x4) that are sufficient for the simplex algorithm.
"""
import numpy as np


def det(M):
    """Determinant of a square matrix (up to 4x4), avoiding LAPACK.

    For d <= 4, uses closed-form cofactor expansion.
    For d > 4, falls back to np.linalg.det (which may crash on broken envs).
    """
    M = np.asarray(M, dtype=float)
    n = M.shape[0]
    assert M.shape == (n, n), f"Expected square matrix, got {M.shape}"

    if n == 1:
        return float(M[0, 0])
    if n == 2:
        return float(M[0, 0] * M[1, 1] - M[0, 1] * M[1, 0])
    if n == 3:
        return float(
            M[0, 0] * (M[1, 1] * M[2, 2] - M[1, 2] * M[2, 1])
            - M[0, 1] * (M[1, 0] * M[2, 2] - M[1, 2] * M[2, 0])
            + M[0, 2] * (M[1, 0] * M[2, 1] - M[1, 1] * M[2, 0])
        )
    if n == 4:
        # Expand along first row
        d = 0.0
        for j in range(4):
            minor = np.delete(np.delete(M, 0, axis=0), j, axis=1)
            cofactor = det(minor)  # 3x3 recursion
            d += ((-1) ** j) * M[0, j] * cofactor
        return float(d)
    if n == 5:
        # Expand along first row (reuses 4x4 cofactor expansion)
        d = 0.0
        for j in range(5):
            minor = np.delete(np.delete(M, 0, axis=0), j, axis=1)
            cofactor = det(minor)  # 4x4 recursion
            d += ((-1) ** j) * M[0, j] * cofactor
        return float(d)

    # Fallback for n >= 6 (may crash on broken LAPACK)
    return float(np.linalg.det(M))


def solve(A, b):
    """Solve A @ x = b for x, using Cramer's rule for small systems.

    A : (d, d) array
    b : (d,) array
    Returns : (d,) array

    Raises LinAlgError if A is singular.
    """
    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float)
    n = A.shape[0]
    assert A.shape == (n, n), f"Expected square A, got {A.shape}"
    assert b.shape == (n,), f"Expected b of length {n}, got {b.shape}"

    det_A = det(A)
    if abs(det_A) < 1e-30:
        raise np.linalg.LinAlgError("Singular matrix")

    x = np.empty(n, dtype=float)
    for j in range(n):
        Aj = A.copy()
        Aj[:, j] = b
        x[j] = det(Aj) / det_A
    return x
