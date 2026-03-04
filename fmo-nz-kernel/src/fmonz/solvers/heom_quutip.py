<<<<<<< HEAD
"""Optional QuTiP adapter for HEOM propagation."""


class QuTiPHEOM:
    def __init__(self, params):
        pass

    def run(self):
        raise NotImplementedError
=======
"""QuTiP-based HEOM solver adapter.

The class conforms to the :class:`~fmonz.solvers.heom_interface.HEOMSolver`
interface so that the remainder of the pipeline can remain agnostic to the
underlying backend.  In particular the expensive ``HEOMSolver`` object from
QuTiP is constructed once and reused for multiple propagations.

The implementation follows the recipe outlined in the project instructions:
for a system Hamiltonian ``H`` it builds a Liouvillian and one
``DrudeLorentzBath`` per site with coupling operator
``Q_i=|i\rangle\langle i|``.  An optional terminator correction is added when
requested in the bath parameters.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from fmonz.config import BathParams
from fmonz.solvers.heom_interface import HEOMSolver


# defer the heavy import until the module is actually used so that the
# entire package can be installed without qutip.
try:
    from qutip import Qobj, basis, liouvillian
    from qutip.solver.heom import HEOMSolver as _QT_HEOMSolver, DrudeLorentzBath

    _have_qutip = True
except ImportError:  # pragma: no cover - optional dependency
    _have_qutip = False


class QuTiPHEOM(HEOMSolver):
    """HEOM backend implemented using QuTiP's HEOMSolver.

    Parameters
    ----------
    H : ndarray
        System Hamiltonian (``(d,d)`` array).
    bath : BathParams
        Parameters describing the bosonic environment.
    """

    def __init__(self, H: np.ndarray, bath: BathParams):
        if not _have_qutip:
            raise ImportError("QuTiP is required for the QuTiPHEOM backend")
        super().__init__(bath)
        self.d = H.shape[0]
        self._H = Qobj(H)

        # assemble baths, one projector per site
        baths = []
        for i in range(self.d):
            Qi = basis(self.d, i) * basis(self.d, i).dag()
            bath_obj = DrudeLorentzBath(
                Qi,
                bath.reorg_energy,
                bath.cutoff,
                bath.temperature,
                Nk=bath.matsubara_terms,
                use_pade=bath.use_pade,
            )
            baths.append(bath_obj)

        # system Liouvillian
        HL = liouvillian(self._H)
        if bath.add_terminator:
            # terminator() returns a tuple (name, superop)
            for b in baths:
                _, term = b.terminator()
                HL = HL + term

        # create solver instance (expensive) and keep for reuse
        self._solver = _QT_HEOMSolver(HL, baths, max_depth=bath.hierarchy_depth)

    def propagate(self, rho0: np.ndarray, times: Sequence[float]) -> np.ndarray:
        """Return reduced density matrix trajectory for a given initial state.

        Parameters are identical to :meth:`HEOMSolver.propagate`.
        """
        rho0_q = Qobj(rho0)
        result = self._solver.run(rho0_q, times)
        # convert list of Qobj states to a numpy array
        states = result.states
        arr = np.stack([st.full() for st in states])
        return arr

    @property
    def backend_name(self) -> str:  # pragma: no cover - simple property
        return "qutip"
>>>>>>> b1437fe (Add FMO NZ kernel scripts and update packaging metadata)
