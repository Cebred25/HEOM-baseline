"""Constraint projection utilities.

Provides small, well-tested projection helpers used by the
kernel-regularization routines.  Implementations are intentionally
elementary and operate on NumPy arrays.
"""

from __future__ import annotations

import numpy as np
from typing import Tuple


def project_to_simplex(x: np.ndarray) -> np.ndarray:
    r"""Project a real vector onto the probability simplex.

    Finds the closest vector `y` (in Euclidean norm) such that
    ``y >= 0`` and ``sum(y) == 1``.  Uses the algorithm from
    "Efficient Projections onto the l1-Ball for Learning in High
    Dimensions" (Duchi et al.).  Works for 1D arrays only.
    """

    x = np.asarray(x, dtype=float)
    if x.ndim != 1:
        raise ValueError("project_to_simplex expects a 1D array")
    n = x.size
    if n == 0:
        return x.copy()

    # sort in descending order
    u = np.sort(x)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0]
    if rho.size == 0:
        theta = 0.0
    else:
        rho = rho[-1]
        theta = (cssv[rho] - 1) / (rho + 1)

    w = np.maximum(x - theta, 0.0)
    return w


def project_trace_preserving(M: np.ndarray, d: int, vec_convention: str = "col") -> np.ndarray:
    """Project a superoperator matrix onto the trace-preserving subspace.

    The projection removes the component along the (vectorized) identity
    so that ``bra_I @ M == 0``.  `M` is modified and returned.
    """

    M = np.asarray(M, dtype=complex).copy()
    d2 = d * d
    if M.shape != (d2, d2):
        raise ValueError("M must be (d2,d2) shaped")

    order = "F" if vec_convention == "col" else "C"
    u = np.eye(d, dtype=complex).reshape(d2, order=order)
    u = u / np.linalg.norm(u)
    violation = u.conj().T @ M
    M = M - np.outer(u, violation)
    return M


def project_hermiticity_preserving(M: np.ndarray, d: int, vec_convention: str = "col") -> np.ndarray:
    """Project a superoperator matrix to satisfy Hermiticity-preserving symmetry.

    Enforces the condition
        M[i,j] = conj( M[i',j'] )
    where the primed indices correspond to swapping subsystem indices
    consistent with the chosen vectorization convention.
    """

    M = np.asarray(M, dtype=complex).copy()
    d2 = d * d
    if M.shape != (d2, d2):
        raise ValueError("M must be (d2,d2) shaped")

    def unravel(idx: int) -> Tuple[int, int]:
        if vec_convention == "col":
            return idx % d, idx // d
        else:
            return idx // d, idx % d

    Mnew = M.copy()
    for i in range(d2):
        a1, a2 = unravel(i)
        ip = a2 + a1 * d if vec_convention == "col" else a2 * d + a1
        for j in range(d2):
            b1, b2 = unravel(j)
            jp = b2 + b1 * d if vec_convention == "col" else b2 * d + b1
            Mnew[i, j] = 0.5 * (M[i, j] + np.conj(M[ip, jp]))
    return Mnew

