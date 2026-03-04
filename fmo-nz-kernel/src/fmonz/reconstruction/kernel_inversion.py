r"""Tools for time differentiation and Volterra inversion.

This module provides utilities required for stages D and E.  Time
differentiation is handled by :func:`time_derivative_superop`, which
supports simple central/forward/backward finite differencing with an
optional smoothing step (currently only Savitzky–Golay).  The
Volterra inversion routine
:func:`invert_volterra_kernel` implements the sequential algorithm
outlined in the project specification.
"""

from __future__ import annotations

import numpy as np
from typing import Any, Dict, Optional


def time_derivative_superop(
    Lambda: np.ndarray,
    dt: float,
    method: str = "central",
    smooth: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Compute time derivative of the superoperator sequence.

    Parameters
    ----------
    Lambda : ndarray, shape (n_t, d2, d2)
        Dynamical map at each time point.
    dt : float
        Time step spacing (assumed uniform).
    method : {"central", "forward", "backward"}
        Differencing scheme to use.  "central" uses central differences
        for interior points with forward/backward at the boundaries.
    smooth : dict or None
        If provided, should contain a ``"method"`` key (currently only
        ``"savitzky_golay"`` is recognized) and parameters for the
        smoothing algorithm (e.g. ``window_length`` and ``polyorder``).
        Smoothing is applied along the time axis to each matrix element
        before differentiation.

    Returns
    -------
    dLambda : ndarray, shape (n_t, d2, d2)
        Time derivative at each point.
    """

    Lambda = np.asarray(Lambda, dtype=complex)
    if Lambda.ndim != 3:
        raise ValueError("Lambda must be a 3D array")
    n_t, d2, _ = Lambda.shape
    if n_t < 2:
        return np.zeros_like(Lambda)

    data = Lambda.copy()
    # optional smoothing
    if smooth is not None:
        if smooth.get("method") == "savitzky_golay":
            try:
                from scipy.signal import savgol_filter
            except ImportError as exc:
                raise RuntimeError("scipy required for smoothing") from exc
            wl = smooth.get("window_length", 5)
            po = smooth.get("polyorder", 2)
            # apply along axis=0 for each matrix element
            shape = data.shape
            data = data.reshape(n_t, -1)
            data = savgol_filter(data, wl, po, axis=0)
            data = data.reshape(shape)
        else:
            raise ValueError(f"unknown smoothing method {smooth.get('method')}")

    dLambda = np.zeros_like(data)

    if method not in ("central", "forward", "backward"):
        raise ValueError(f"unknown method {method}")

    # compute interior points
    if method == "central":
        for n in range(1, n_t - 1):
            dLambda[n] = (data[n + 1] - data[n - 1]) / (2 * dt)
        # boundaries
        dLambda[0] = (data[1] - data[0]) / dt
        dLambda[-1] = (data[-1] - data[-2]) / dt
    elif method == "forward":
        for n in range(n_t - 1):
            dLambda[n] = (data[n + 1] - data[n]) / dt
        dLambda[-1] = (data[-1] - data[-2]) / dt
    else:  # backward
        dLambda[0] = (data[1] - data[0]) / dt
        for n in range(1, n_t):
            dLambda[n] = (data[n] - data[n - 1]) / dt

    return dLambda


def invert_volterra_kernel(
    Lambda: np.ndarray,
    dLambda: np.ndarray,
    dt: float,
    quadrature: str = "trapezoid",
    K0_rule: str = "from_dLambda0",
) -> np.ndarray:
    """Sequential Volterra inversion to obtain memory kernel K(t).

    Parameters
    ----------
    Lambda : ndarray, shape (n_t, d2, d2)
        Dynamical map.
    dLambda : ndarray, same shape as ``Lambda``
        Time derivative of the dynamical map.
    dt : float
        Time step spacing.
    quadrature : str
        Integration rule; only ``"trapezoid"`` is implemented.
    K0_rule : str
        How to set the kernel at t=0.  ``"from_dLambda0"`` uses the simple
        approximation ``K_0 = (2/dt) dLambda[0]``.  ``"zero"`` forces
        ``K_0=0``.

    Returns
    -------
    K : ndarray, shape (n_t, d2, d2)
        Memory kernel at each time.
    """

    Lambda = np.asarray(Lambda, dtype=complex)
    dLambda = np.asarray(dLambda, dtype=complex)
    if Lambda.shape != dLambda.shape:
        raise ValueError("Lambda and dLambda must have same shape")
    n_t, d2, _ = Lambda.shape
    if n_t == 0:
        return np.zeros((0, d2, d2), dtype=complex)

    K = np.zeros_like(Lambda)

    if quadrature != "trapezoid":
        raise ValueError("only trapezoid quadrature supported")

    # initialize K0
    if K0_rule == "from_dLambda0":
        K[0] = (2 / dt) * dLambda[0]
    elif K0_rule == "zero":
        K[0] = np.zeros((d2, d2), dtype=complex)
    else:
        raise ValueError(f"unknown K0_rule {K0_rule}")

    # precompute weights
    def weight(m, n):
        if m == 0 or m == n:
            return dt / 2
        else:
            return dt

    for n in range(1, n_t):
        R = dLambda[n].copy()
        # subtract contributions from earlier kernels
        for m in range(1, n + 1):
            w = weight(n - m, n)
            R -= w * (K[n - m] @ Lambda[m])
        K[n] = (2 / dt) * R
    return K


def enforce_constraints_on_kernel(
    K: np.ndarray,
    d: int,
    vec_convention: str = "col",
    enforce_tp: bool = True,
    enforce_hp: bool = True,
    smooth: Optional[Dict[str, Any]] = None,
) -> np.ndarray:
    """Project a kernel sequence onto a physically consistent subspace.

    Parameters
    ----------
    K : ndarray, shape (n_t, d2, d2)
        Raw memory kernel where d2 = d*d.
    d : int
        Hilbert space dimension (used to build vectorized identity).
    vec_convention : {"col","row"}
        Flattening convention used when computing ``bra_I``.
    enforce_tp : bool
        If True, impose trace preservation by ensuring
        ``bra_I @ K[t] == 0`` for all t.
    enforce_hp : bool
        If True, impose Hermiticity preservation by symmetrizing
        matrix elements according to index transpose pairs.
    smooth : dict or None
        Optional temporal smoothing to regularize the kernel before
        projection.  Format identical to ``time_derivative_superop``.

    Returns
    -------
    Knew : ndarray, same shape as ``K``
        Kernel after enforcing requested constraints.
    """

    K = np.asarray(K, dtype=complex).copy()
    if K.ndim != 3:
        raise ValueError("K must be a 3D array")
    n_t, d2, d2b = K.shape
    if d2 != d2b:
        raise ValueError("kernel slices must be square")
    if d2 != d * d:
        raise ValueError("inconsistent dimension d")

    # optional smoothing
    if smooth is not None:
        if smooth.get("method") == "savitzky_golay":
            try:
                from scipy.signal import savgol_filter
            except ImportError as exc:
                raise RuntimeError("scipy required for smoothing") from exc
            wl = smooth.get("window_length", 5)
            po = smooth.get("polyorder", 2)
            shape = K.shape
            flat = K.reshape(n_t, -1)
            flat = savgol_filter(flat, wl, po, axis=0)
            K = flat.reshape(shape)
        else:
            raise ValueError(f"unknown smoothing method {smooth.get('method')}")

    # prepare trace-preservation vector
    if enforce_tp:
        I = np.eye(d, dtype=complex)
        order = "F" if vec_convention == "col" else "C"
        u = I.reshape(d2, order=order)
        # normalize to unit HS norm
        u = u / np.linalg.norm(u)
        # project each time slice
        for n in range(n_t):
            # compute bra_I @ K = u^† K (row vector)
            violation = u.conj().T @ K[n]
            K[n] = K[n] - np.outer(u, violation)

    # hermiticity preservation: enforce K_{ij} = conj(K_{i'j'})
    if enforce_hp:
        # precompute index pairs mapping
        # for column stacking, index = a + b*d
        def unravel(idx):
            if vec_convention == "col":
                return idx % d, idx // d
            else:
                return idx // d, idx % d

        for n in range(n_t):
            M = K[n]
            Mnew = M.copy()
            for i in range(d2):
                a1, a2 = unravel(i)
                ip = a2 + a1 * d if vec_convention == "col" else a2 * d + a1
                for j in range(d2):
                    b1, b2 = unravel(j)
                    jp = b2 + b1 * d if vec_convention == "col" else b2 * d + b1
                    Mnew[i, j] = 0.5 * (M[i, j] + np.conj(M[ip, jp]))
            K[n] = Mnew

    return K

def propagate_nz(
    K: np.ndarray,
    rho0: np.ndarray,
    dt: float,
    method: str = "euler",
) -> np.ndarray:
    r"""Forward Nakajima–Zwanzig propagation using a memory kernel.

    The discrete equation in Liouville space is

        v_dot[n] = \sum_{m=0}^n w(m,n) K[n-m] v[m]

    where ``v[n]`` is the vectorized density at time step ``n``.  The
    implementation uses an explicit Euler step, with trapezoidal quadrature
    weights ``w`` following the same convention used during kernel
    extraction.  Only ``method="euler"`` is currently supported; the
    argument exists to allow future extension.
    """

    K = np.asarray(K, dtype=complex)
    if K.ndim != 3:
        raise ValueError("K must be 3-dimensional")
    n_t, d2, d2b = K.shape
    if d2 != d2b:
        raise ValueError("kernel slices must be square")

    from fmonz.utils.vec import vec, unvec

    rho0 = np.asarray(rho0, dtype=complex)
    d = rho0.shape[0]
    if rho0.shape != (d, d):
        raise ValueError("rho0 must be square")
    if d * d != d2:
        raise ValueError("dimension mismatch between rho0 and K")

    def weight(m, n):
        if m == 0 or m == n:
            return dt / 2
        else:
            return dt

    v = np.zeros((n_t, d2), dtype=complex)
    v[0] = vec(rho0)

    for n in range(n_t - 1):
        rhs = np.zeros(d2, dtype=complex)
        for m in range(n + 1):
            w = weight(n - m, n)
            rhs += w * (K[n - m] @ v[m])
        v[n + 1] = v[n] + dt * rhs

    rhos = np.zeros((n_t, d, d), dtype=complex)
    for n in range(n_t):
        rhos[n] = unvec(v[n], d)
    return rhos


# --- markov approximation helpers ---

def build_markov_generator(K: np.ndarray, dt: float, mode: str = "integral") -> np.ndarray:
    """Construct a Markovian generator from a memory kernel sequence.

    Two simple prescriptions are offered:

    * ``mode="integral"`` – perform the quadrature
      ``L = \sum_n w_n K[n]`` with the same trapezoidal weights used
      elsewhere.  This approximates the formal time integral of the
      kernel.  It is the most faithful Markov approximation.
    * ``mode="instantaneous"`` – simply take the first time slice
      ``L = K[0]`` and ignore all memory.  This corresponds to
      dropping the convolution entirely and is often used in the
      literature as the extreme Markov limit.

    Parameters
    ----------
    K : ndarray, shape (n_t, d2, d2)
        Memory kernel sequence.
    dt : float
        Time step used when computing the kernel.
    mode : {"integral","instantaneous"}
        Choice of approximation.

    Returns
    -------
    L : ndarray, shape (d2,d2)
        Markovian generator superoperator.
    """

    K = np.asarray(K, dtype=complex)
    if K.ndim != 3:
        raise ValueError("K must be a 3D array")
    n_t, d2, d2b = K.shape
    if d2 != d2b:
        raise ValueError("kernel slices must be square")

    if mode == "instantaneous":
        return K[0].copy()
    elif mode == "integral":
        def weight(n):
            if n == 0 or n == n_t - 1:
                return dt / 2
            else:
                return dt

        L = np.zeros((d2, d2), dtype=complex)
        for n in range(n_t):
            L += weight(n) * K[n]
        return L
    else:
        raise ValueError(f"unknown mode {mode}")


def propagate_markov(L: np.ndarray, rho0: np.ndarray, times: np.ndarray) -> np.ndarray:
    """Propagate density matrices using a Markovian generator.

    The evolution is simply

        v_dot = L @ v

    which is integrated with an explicit Euler step over the provided
    ``times`` grid.  ``rho0`` must be a valid density matrix and the
    generator ``L`` should have compatible dimensions.
    """

    times = np.asarray(times)
    if times.ndim != 1:
        raise ValueError("times must be a 1D array")
    n_t = times.size
    if n_t == 0:
        return np.zeros((0, *rho0.shape), dtype=complex)

    rho0 = np.asarray(rho0, dtype=complex)
    d = rho0.shape[0]
    if rho0.shape != (d, d):
        raise ValueError("rho0 must be square")

    d2 = d * d
    if L.shape != (d2, d2):
        raise ValueError("generator shape incompatible with rho0 dimension")

    from fmonz.utils.vec import vec, unvec

    dt = times[1] - times[0] if n_t > 1 else 0.0
    v = np.zeros((n_t, d2), dtype=complex)
    v[0] = vec(rho0)
    for n in range(n_t - 1):
        v[n + 1] = v[n] + dt * (L @ v[n])

    rhos = np.zeros((n_t, d, d), dtype=complex)
    for n in range(n_t):
        rhos[n] = unvec(v[n], d)
    return rhos

