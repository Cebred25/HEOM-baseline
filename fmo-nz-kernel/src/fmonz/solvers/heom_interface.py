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
<<<<<<< HEAD
    """Minimal placeholder for a solver instance.

    Concrete subclasses may hold pre‑computed matrices, memory, etc.
=======
    """Abstract base class for HEOM propagation backends.

    Subclasses should implement :meth:`propagate` and may cache solver
    state (for example, a QuTiP ``HEOMSolver`` object) to make repeated
    calls efficient.  The pipeline code only assumes the surface defined
    here, so different implementations can be swapped transparently.
>>>>>>> b1437fe (Add FMO NZ kernel scripts and update packaging metadata)
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
<<<<<<< HEAD
=======
    generator; ``rho0`` is the initial density matrix ``(d,d)``, and
    ``times`` is a one‑dimensional array of time points.  The output has
    shape ``(len(times), d, d)``.

    The default implementation attempts to instantiate the QuTiP adapter
    if the library is available.  Failing that, ``NotImplementedError`` is
    raised; callers may catch this to fall back to other backends.
    """

    # lazy import so that the interface module itself doesn't depend on
    # qutip; only the adapter will trigger the actual import error.
    try:
        from fmonz.solvers.heom_quutip import QuTiPHEOM
    except ImportError:  # pragma: no cover - imported optionally
        raise NotImplementedError("no HEOM backend available")

    solver = QuTiPHEOM(H, bath_params)
    return solver.propagate(rho0, times)


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
>>>>>>> b1437fe (Add FMO NZ kernel scripts and update packaging metadata)
    generator; ``rho0`` is the initial density matrix (``(d,d)``), and
    ``times`` is a one‑dimensional array of time points.  The output has
    shape ``(len(times), d, d)``.

    A real solver would assemble and integrate the hierarchy here.  The
    base implementation simply raises ``NotImplementedError`` so that
    test code can confirm the interface exists without depending on a
    particular HEOM backend.
    """

    raise NotImplementedError
