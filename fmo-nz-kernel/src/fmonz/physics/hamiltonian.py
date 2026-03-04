"""Hamiltonian construction and manipulation.

The simple Frenkel exciton model on the single‑exciton manifold is
defined by site energies on the diagonal and pairwise couplings off
–diagonal.  The configuration is provided via
:class:`fmonz.config.SystemConfig`.
"""

from __future__ import annotations

import numpy as np
from fmonz.config import SystemConfig


def build_hamiltonian(sys: SystemConfig) -> np.ndarray:
    """Return the system Hamiltonian for the given configuration.

    Parameters
    ----------
    sys : SystemConfig
        Contains ``d``, ``site_energies`` and ``couplings``.

    Returns
    -------
    H : ndarray, shape (d,d)
        Complex-valued Hamiltonian matrix.
    """

    d = sys.d
    H = np.zeros((d, d), dtype=complex)
    energies = np.asarray(sys.site_energies, dtype=complex)
    if energies.shape != (d,):
        raise ValueError("site_energies must be length d")

    H[np.diag_indices(d)] = energies

    couplings = np.asarray(sys.couplings, dtype=complex)
    if couplings.shape != (d, d):
        raise ValueError("couplings must be a (d,d) array")

    # enforce zero diagonal if the user supplied nonzero
    np.fill_diagonal(couplings, 0)
    H += couplings
    return H
