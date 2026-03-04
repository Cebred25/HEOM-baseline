import numpy as np
from fmonz.physics.hamiltonian import build_hamiltonian
from fmonz.config import SystemConfig


def test_build_hamiltonian_simple():
    d = 2
    energies = [1.0, 2.0]
    couplings = [[0.0, 0.5], [0.5, 0.0]]
    sys = SystemConfig(d=d, site_energies=np.array(energies), couplings=np.array(couplings))
    H = build_hamiltonian(sys)
    assert H.shape == (2, 2)
    assert H[0, 0] == 1.0
    assert H[1, 1] == 2.0
    assert H[0, 1] == 0.5
    assert H[1, 0] == 0.5
