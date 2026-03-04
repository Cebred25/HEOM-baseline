r"""Validation utilities for dynamics comparisons and error metrics.

This module provides a collection of initial states useful for validating
reconstructed maps and a handful of metrics to quantify differences between
trajectories.
"""

from __future__ import annotations

import numpy as np
from typing import List


def validation_states(d: int, n_random: int = 10, seed: int = 0) -> List[np.ndarray]:
    """Return a list of physically valid density matrices of size ``d``.

    The returned ensemble includes:

    * ``d`` site-localized pure states ``|i><i|``
    * ``d*(d-1)/2`` equal-weight coherent superpositions ``(|i>+|j>)(...)``
    * a maximally mixed (thermal-like) state ``I/d``
    * ``n_random`` random density matrices drawn by Ginibre construction.

    The random states are seeded for reproducibility.
    """

    states: List[np.ndarray] = []

    # site-localized
    for i in range(d):
        rho = np.zeros((d, d), dtype=complex)
        rho[i, i] = 1.0
        states.append(rho)

    # coherent superpositions
    for i in range(d):
        for j in range(i + 1, d):
            vec = np.zeros(d, dtype=complex)
            vec[i] = 1.0
            vec[j] = 1.0
            vec = vec / np.linalg.norm(vec)
            rho = np.outer(vec, vec.conj())
            states.append(rho)

    # thermal-like mixed state
    states.append(np.eye(d, dtype=complex) / d)

    # random density matrices via Ginibre
    rng = np.random.default_rng(seed)
    for k in range(n_random):
        A = rng.standard_normal((d, d)) + 1j * rng.standard_normal((d, d))
        rho = A @ A.conj().T
        rho = rho / np.trace(rho)
        states.append(rho)

    return states


def trace_distance(rho: np.ndarray, sigma: np.ndarray) -> float:
    r"""Compute trace distance between two density matrices.

    .. math::
        D(\rho,\sigma)=\frac12 \|\rho-\sigma\|_1

    The ``1``-norm is computed from the eigenvalues of the Hermitian
    difference.
    """

    delta = rho - sigma
    # eigenvalues maybe complex due to numerical noise
    vals = np.linalg.eigvals(delta)
    return 0.5 * np.sum(np.abs(vals))


def population_curves(rhos: np.ndarray) -> np.ndarray:
    """Return site populations for each density in a trajectory.

    Parameters
    ----------
    rhos : ndarray, shape (n_t, d, d)
        Sequence of density matrices.

    Returns
    -------
    pops : ndarray, shape (n_t, d)
        Diagonal elements \rho_{ii}(t).
    """

    rhos = np.asarray(rhos)
    if rhos.ndim != 3:
        raise ValueError("rhos must be 3-dimensional")
    n_t, d, d2 = rhos.shape
    if d != d2:
        raise ValueError("rho slices must be square")
    pops = np.real(np.diagonal(rhos, axis1=1, axis2=2))
    return pops


def coherence_magnitude(rhos: np.ndarray) -> np.ndarray:
    r"""Compute a measure of coherence at each time step.

    We use the Frobenius norm of the off-diagonal part:
    ``\|rho - diag(rho)\|_F`` which equals
    ``sqrt(sum_{i\neq j} |rho_{ij}|^2)``.
    """

    rhos = np.asarray(rhos)
    if rhos.ndim != 3:
        raise ValueError("rhos must be 3-dimensional")
    n_t, d, d2 = rhos.shape
    if d != d2:
        raise ValueError("rho slices must be square")
    mags = np.zeros(n_t, dtype=float)
    for n in range(n_t):
        rho = rhos[n]
        off = rho - np.diag(np.diag(rho))
        mags[n] = np.linalg.norm(off)
    return mags


# ---------------------------------------------------------------------------
# Memory-time analysis helpers
# ---------------------------------------------------------------------------

def kernel_norm_curve(K: np.ndarray, norm: str = "fro") -> np.ndarray:
    r"""Return a norm curve for a sequence of kernels.

    Parameters
    ----------
    K : ndarray, shape ``(n_t, N, N)``
        Time-dependent kernel superoperators.
    norm : {'fro', 'op'}
        Which matrix norm to compute at each time step: Frobenius or operator
        (spectral) norm.

    Returns
    -------
    knorm : ndarray, shape ``(n_t,)``
        Norm of ``K[t]`` for each time index.
    """

    K = np.asarray(K)
    if K.ndim != 3:
        raise ValueError("K must be a 3‑dimensional array")
    n_t, d1, d2 = K.shape
    if d1 != d2:
        raise ValueError("kernel slices must be square")

    knorm = np.empty(n_t, dtype=float)
    for i in range(n_t):
        mat = K[i]
        if norm == "fro":
            knorm[i] = np.linalg.norm(mat)
        elif norm == "op":
            knorm[i] = np.linalg.norm(mat, ord=2)
        else:
            raise ValueError(f"unsupported norm '{norm}'")
    return knorm


def memory_time_threshold(
    times: np.ndarray, knorm: np.ndarray, eps: float = 1e-2
) -> float:
    r"""Estimate memory time via threshold decay.

    ``tau_mem`` is defined as the smallest ``t`` for which
    ``knorm(t) <= eps * knorm(0)`` *and* the norm remains below that
    threshold thereafter.
    """

    times = np.asarray(times)
    knorm = np.asarray(knorm)
    if times.ndim != 1 or knorm.ndim != 1 or times.size != knorm.size:
        raise ValueError("times and knorm must be one-dimensional arrays of equal length")

    thresh = eps * knorm[0]
    for idx, val in enumerate(knorm):
        if val <= thresh and np.all(knorm[idx:] <= thresh):
            return float(times[idx])
    return float(times[-1])


def memory_time_tailweight(
    times: np.ndarray, knorm: np.ndarray, delta: float = 1e-2
) -> float:
    r"""Estimate memory time via tail weight of the norm curve.

    The tail weight is

    .. math::
        W(t) = \frac{\int_t^{\infty} knorm(s) \,ds}{\int_0^{\infty} knorm(s) \,ds}

    ``tau_mem`` is the smallest time for which ``W(t) < delta``.
    """

    times = np.asarray(times)
    knorm = np.asarray(knorm)
    if times.ndim != 1 or knorm.ndim != 1 or times.size != knorm.size:
        raise ValueError("times and knorm must be one-dimensional arrays of equal length")

    # compute trapezoidal integral manually to avoid deprecated np.trapz
    def _trapz(y: np.ndarray, x: np.ndarray) -> float:
        # assumes x is strictly increasing
        return float(np.sum((y[1:] + y[:-1]) * (x[1:] - x[:-1]) / 2.0))

    total = _trapz(knorm, times)
    if total <= 0:
        return float(times[-1])

    for idx, t in enumerate(times):
        tail = _trapz(knorm[idx:], times[idx:])
        if tail / total < delta:
            return float(t)
    return float(times[-1])
