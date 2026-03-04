import numpy as np

from fmonz.solvers.heom_interface import heom_propagate_basis_operator
from fmonz.config import BathParams


def test_heom_interface_signature():
    # just ensure the function exists and either raises or returns a valid
    # array.  When QuTiP is installed the helper will actually propagate.
    H = np.eye(2)
    bath = BathParams(temperature=300.0, reorg_energy=1.0, cutoff=50.0, hierarchy_depth=1)
    rho0 = np.eye(2) / 2
    times = np.linspace(0, 1, 5)
    try:
        out = heom_propagate_basis_operator(H, bath, rho0, times)
    except NotImplementedError:
        # expected when no backend is available
        return
    assert out.shape == (len(times), 2, 2)


def test_quitp_backend_basic():
    """Sanity check for the QuTiP adapter when the library is present."""
    import pytest

    pytest.importorskip("qutip")
    from fmonz.solvers.heom_quutip import QuTiPHEOMSolver

    H = np.zeros((2, 2))
    bath = BathParams(temperature=1.0, reorg_energy=0.0, cutoff=1.0, hierarchy_depth=1)
    solver = QuTiPHEOMSolver(H, bath, tlist=np.linspace(0, 0.5, 3))
    times = np.linspace(0, 0.5, 3)
    rho0 = np.eye(2) / 2
    rhos = solver.propagate(rho0, times)
    assert rhos.shape == (3, 2, 2)
    assert np.allclose(rhos[0], rho0, atol=1e-8)
    traces = np.trace(rhos, axis1=1, axis2=2)
    assert np.allclose(traces, 1.0)
