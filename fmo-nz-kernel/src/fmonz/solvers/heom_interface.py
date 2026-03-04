"""Unified interface for HEOM solvers and helpers.

This module exposes a dataclass for bath parameters and a standard
function that any concrete HEOM implementation should provide.  The
idea is that the remainder of the package can be written in terms of the
abstract signature below; a specific backend (e.g. the in‑house solver
or a QuTiP adapter) simply needs to implement the same call.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from typing import Sequence

from fmonz.config import BathParams


@dataclass
class HEOMSolver:
    """Minimal placeholder for a solver instance.

    Concrete subclasses may hold pre‑computed matrices, memory, etc.
    """

    bath: BathParams

    def propagate(self, rho0: np.ndarray, times: Sequence[float]) -> np.ndarray:
        """Propagate a density matrix according to the HEOM equations.

        Parameters
        ----------
        rho0 : (d,d) ndarray
            Initial reduced density matrix.
        times : sequence of float
            Times at which to return the state.

        Returns
        -------
        rhos : ndarray, shape (n_t, d, d)
            Reduced density matrix at each requested time.
        """
        raise NotImplementedError


class DummyHEOM(HEOMSolver):
    """Trivial solver that returns the initial operator at all times.

    This is useful for exercising the pipeline without a real HEOM
    backend.  The behaviour mirrors the dummy class used in the unit
    tests.
    """

    def __init__(self, bath: BathParams, d: int) -> None:
        super().__init__(bath)
        self.d = d

    def propagate(self, rho0: np.ndarray, times: Sequence[float]) -> np.ndarray:
        times = np.asarray(times)
        return np.stack([rho0 for _ in times])


def heom_propagate_basis_operator(
    H: np.ndarray,
    bath_params: BathParams,
    rho0: np.ndarray,
    times: np.ndarray,
) -> np.ndarray:
    """Return reduced dynamics for a single initial operator.

    This free function mirrors the signature described in the project
    goal.  ``H`` and ``bath_params`` together determine the HEOM
    generator; ``rho0`` is the initial density matrix (``(d,d)``), and
    ``times`` is a one‑dimensional array of time points.  The output has
    shape ``(len(times), d, d)``.

    A real solver would assemble and integrate the hierarchy here.  The
    base implementation simply raises ``NotImplementedError`` so that
    test code can confirm the interface exists without depending on a
    particular HEOM backend.
    """

    raise NotImplementedError
