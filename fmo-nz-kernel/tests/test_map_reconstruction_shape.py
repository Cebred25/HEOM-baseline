import numpy as np

from fmonz.reconstruction.dynamical_map import (
    reconstruct_dynamical_map,
    save_map,
)
from fmonz.utils.basis import operator_basis


class DummyHEOM:
    def propagate(self, rho0, times):
        # return the same operator at each time step
        times = np.asarray(times)
        return np.stack([rho0 for _ in times])


def test_reconstruction_basic():
    d = 2
    basis = operator_basis(d, kind="matrix")
    heom = DummyHEOM()
    times = np.linspace(0, 1, 5)
    Lambda = reconstruct_dynamical_map(heom, basis, times)
    assert Lambda.shape == (len(times), d * d, d * d)
    # each column should equal vec(basis[k]) at all times
    for n in range(len(times)):
        for k in range(d * d):
            expected = basis[k].reshape(d * d, order="F")
            assert np.allclose(Lambda[n, :, k], expected)


def test_save_map_roundtrip(tmp_path):
    d = 3
    times = np.linspace(0, 0.5, 2)
    Lambda = np.zeros((len(times), d * d, d * d), dtype=complex)
    fname = tmp_path / "map.npz"
    save_map(str(fname), Lambda, d=d, dt=0.1, convention="col")
    loaded = np.load(fname)
    assert loaded["Lambda"].shape == Lambda.shape
    assert int(loaded["d"]) == d
    assert float(loaded["dt"]) == 0.1
    assert loaded["convention"] == "col"
