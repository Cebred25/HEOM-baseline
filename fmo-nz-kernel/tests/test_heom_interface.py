import numpy as np

from fmonz.solvers.heom_interface import heom_propagate_basis_operator
from fmonz.config import BathParams


def test_heom_interface_signature():
    # just ensure the function exists and raises NotImplementedError
    H = np.eye(2)
    bath = BathParams(temperature=300.0, reorg_energy=1.0, cutoff=50.0, hierarchy_depth=1)
    rho0 = np.eye(2) / 2
    times = np.linspace(0, 1, 5)
    try:
        heom_propagate_basis_operator(H, bath, rho0, times)
    except NotImplementedError:
        return
    # if it didn't raise then ensure output shape
    out = heom_propagate_basis_operator(H, bath, rho0, times)
    assert out.shape == (len(times), 2, 2)
