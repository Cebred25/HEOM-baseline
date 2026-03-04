"""QuTiP-based HEOM solver adapter.

The class conforms to the :class:`~fmonz.solvers.heom_interface.HEOMSolver`
interface so that the remainder of the pipeline can remain agnostic to the
underlying backend.  In particular the expensive ``HEOMSolver`` object from
QuTiP is constructed once and reused for multiple propagations.

This module provides `QuTiPHEOMSolver` which exposes a lightweight
``run`` API and a stable hash useful for caching run outputs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import hashlib
import json
import numpy as np

from fmonz.config import BathParams
from fmonz.solvers.heom_interface import HEOMSolver


# defer the heavy import until the module is actually used so that the
# entire package can be installed without qutip.
try:
    from qutip import Qobj, basis, liouvillian
    from qutip.solver.heom import HEOMSolver as _QT_HEOMSolver, DrudeLorentzBath

    _have_qutip = True
except Exception:  # pragma: no cover - optional dependency
    _have_qutip = False
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


class QuTiPHEOMSolver(HEOMSolver):
    """HEOM backend implemented using QuTiP's HEOMSolver.

    Args:
        H: system Hamiltonian as a (d,d) numpy array
        bath: BathParams instance
        tlist: optional default time grid used for caching/hash
        options: dict of misc options (validate_output, renorm_tol, ...)
    """

    def __init__(self, H: np.ndarray, bath: BathParams, tlist: np.ndarray | None = None, *, options: dict | None = None):
        if not _have_qutip:
            raise ImportError("QuTiP is required for the QuTiPHEOM backend")
        super().__init__(bath)
        self.d = int(H.shape[0])
        self._H = Qobj(H)
        self._H_mat = np.array(H, copy=True)
        self.tlist = None if tlist is None else np.asarray(tlist)
        self.options = dict(validate_output=True, renorm_tol=1e-8)
        if options:
            self.options.update(options)

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
            for b in baths:
                try:
                    _, term = b.terminator()
                except Exception:
                    term = None
                if term is not None:
                    HL = HL + term

        # create solver instance (expensive) and keep for reuse
        self._solver = _QT_HEOMSolver(HL, baths, max_depth=bath.hierarchy_depth)

    def stable_hash(self, dt: float | None = None, n_steps: int | None = None) -> str:
        """Return a stable short hash representing solver configuration.

        The hash covers H entries, bath parameters, hierarchy depth, Nk,
        and time grid information when provided.
        """
        m = hashlib.sha1()
        # H content
        m.update(self._H_mat.tobytes())
        m.update(str(self._H_mat.shape).encode())
        # bath params
        bp = dict(
            reorg_energy=self.bath.reorg_energy,
            cutoff=self.bath.cutoff,
            temperature=self.bath.temperature,
            hierarchy_depth=self.bath.hierarchy_depth,
            matsubara_terms=self.bath.matsubara_terms,
            use_pade=self.bath.use_pade,
            add_terminator=self.bath.add_terminator,
        )
        m.update(json.dumps(bp, sort_keys=True).encode())
        if dt is not None:
            m.update(str(float(dt)).encode())
        if n_steps is not None:
            m.update(str(int(n_steps)).encode())
        return m.hexdigest()[:12]

    def run(self, rho0: np.ndarray, tlist: np.ndarray | None = None, *, validate_output: bool | None = None) -> np.ndarray:
        """Propagate a single initial operator and return (n_t, d, d) array.

        If ``tlist`` is omitted the instance default ``self.tlist`` is used.
        """
        if tlist is None:
            if self.tlist is None:
                raise ValueError("no time grid supplied")
            times = self.tlist
        else:
            times = np.asarray(tlist)

        validate = self.options.get("validate_output", True) if validate_output is None else bool(validate_output)

        rho0_q = Qobj(rho0)
        result = self._solver.run(rho0_q, times)
        states = result.states
        arr = np.stack([st.full() for st in states])

        if validate:
            arr = self._postprocess_output(arr)

        return arr

    # backward-compatible alias expected by some callers/tests
    def propagate(self, rho0: np.ndarray, times: Sequence[float]) -> np.ndarray:
        return self.run(rho0, tlist=np.asarray(times))

    def _postprocess_output(self, arr: np.ndarray) -> np.ndarray:
        """Enforce Hermiticity and approximate trace conservation.

        Returns a new array (complex) with small numerical corrections.
        """
        # Ensure complex dtype
        arr = np.asarray(arr, dtype=complex)
        n_t = arr.shape[0]
        for i in range(n_t):
            A = arr[i]
            # symmetrize
            A = 0.5 * (A + A.conj().T)
            # trace renormalize if tiny drift
            tr = np.trace(A)
            if abs(tr - 1.0) > self.options.get("renorm_tol", 1e-8):
                # if drift is tiny, rescale; otherwise leave for downstream checks
                if abs(tr - 1.0) < 1e-6:
                    A = A / tr
            arr[i] = A
        return arr

    @property
    def backend_name(self) -> str:  # pragma: no cover - simple property
        return "qutip"
>>>>>>> b1437fe (Add FMO NZ kernel scripts and update packaging metadata)
