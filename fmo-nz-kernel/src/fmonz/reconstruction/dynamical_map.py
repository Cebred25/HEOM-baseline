r"""Dynamical map representation and utilities.

Stage C focuses on reconstructing the reduced dynamical map \(\Lambda(t)\) from
HEOM trajectories computed for a complete operator basis.  The map is defined
in Liouville space and has shape ``(n_t, d^2, d^2)`` where ``d`` is the system
dimension and ``n_t`` the number of time points.  A helper is also provided to
write the map to disk with minimal metadata.
"""

from __future__ import annotations

import numpy as np
from typing import Sequence, List

from fmonz.solvers.heom_interface import HEOMSolver


def reconstruct_dynamical_map(
    heom: HEOMSolver,
    basis_ops: List[np.ndarray],
    times: Sequence[float],
    vec_convention: str = "col",
) -> np.ndarray:
    r"""Reconstruct \Lambda(t) from HEOM propagation of basis operators.

    Parameters
    ----------
    heom : HEOMSolver
        Instance providing a :meth:`propagate` method returning the reduced
        density matrix trajectory for a given initial operator.
    basis_ops : list of (d,d) arrays
        Complete operator basis spanning the system Hilbert space.
    times : sequence of float
        Time grid for which to reconstruct the map.
    vec_convention : {"col","row"}
        How to flatten the (d,d) states into vectors.  "col" uses
        column‑stacking (Fortran order); "row" uses row‑stacking (C order).

    Returns
    -------
    Lambda : ndarray, shape (n_t, d2, d2)
        Dynamical map where columns correspond to basis operators and rows to
        the resulting vectorized states.
    """

    times = np.asarray(times)
    n_t = times.size
    if n_t == 0:
        raise ValueError("empty time grid")

    if len(basis_ops) == 0:
        raise ValueError("empty operator basis")

    d = basis_ops[0].shape[0]
    d2 = d * d
    Lambda = np.zeros((n_t, d2, d2), dtype=complex)

    for k, B in enumerate(basis_ops):
        rhos = heom.propagate(B, times)
        rhos = np.asarray(rhos)
        if rhos.ndim != 3 or rhos.shape[1:] != (d, d):
            raise ValueError("HEOM solver returned array with incorrect shape")

        for n in range(n_t):
            mat = rhos[n]
            if vec_convention == "col":
                vec = mat.reshape(d2, order="F")
            elif vec_convention == "row":
                vec = mat.reshape(d2, order="C")
            else:
                raise ValueError("unknown vec_convention %r" % vec_convention)
            Lambda[n, :, k] = vec

    return Lambda


def save_map(filename: str, Lambda: np.ndarray, d: int, dt: float, convention: str = "col"):
    """Write dynamical map and metadata to a compressed ``.npz`` file.

    The saved object contains arrays ``Lambda`` (complex), ``d`` (int),
    ``dt`` (float), and ``convention`` (str).  This simple container facilitates
    later analysis without recomputing the map.
    """
    np.savez_compressed(
        filename,
        Lambda=Lambda,
        d=d,
        dt=dt,
        convention=convention,
    )


# The heavy lifting for time differentiation lives in ``kernel_inversion``.
# We re-export the function here so that earlier tests and other modules
# can still import it from ``reconstruction.dynamical_map`` without
# creating a circular dependency.
from .kernel_inversion import time_derivative_superop
