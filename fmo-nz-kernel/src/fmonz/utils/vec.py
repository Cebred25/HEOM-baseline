"""Vector conventions and helpers.

Throughout the repository we work in the Liouville (vectorized) picture.
The preferred convention is column‑stacking (Fortran order) so that the
vectorization of a density matrix ``rho`` satisfies::

    vec(rho) = rho.reshape(d*d, order="F")

This matches the usual mathematics in the open‑systems literature.

The companion ``unvec`` undoes the operation.
"""

import numpy as np


def vec(mat: np.ndarray) -> np.ndarray:
    """Return the column‑stacked vector form of a (d,d) matrix."""
    return mat.reshape(mat.size, order="F")


def unvec(vec: np.ndarray, d: int) -> np.ndarray:
    """Inverse of :func:`vec`.  ``d`` is the system dimension."""
    return vec.reshape((d, d), order="F")


def flatten(matrix):
    # legacy alias preserved for tests
    return vec(matrix)
