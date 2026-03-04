"""Unified interface for HEOM solvers and helpers.

This module defines the abstract solver surface used by the rest of the
pipeline and provides a simple dummy implementation for testing.  Concrete
backends (for example a QuTiP adapter) should subclass ``HEOMSolver`` and
implement the ``propagate`` method.
"""

from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from typing import Sequence

from fmonz.config import BathParams


@dataclass
class HEOMSolver:
    """Abstract base class for HEOM propagation backends.

    Subclasses should implement :meth:`propagate` and may cache solver
    state (for example, a QuTiP ``HEOMSolver`` object) to make repeated
    calls efficient.
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

    Useful for exercising the pipeline without a real HEOM backend.
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

    The default implementation attempts to instantiate the optional QuTiP
    adapter when available; otherwise it raises ``NotImplementedError`` so
    callers can fall back to other backends.
    """

    try:
        # adapter may be named QuTiPHEOMSolver in this codebase
        from fmonz.solvers.heom_quutip import QuTiPHEOMSolver
    except Exception:  # pragma: no cover - optional import
        raise NotImplementedError("no HEOM backend available")

    solver = QuTiPHEOMSolver(H, bath_params, tlist=times)
    return solver.propagate(rho0, times)
