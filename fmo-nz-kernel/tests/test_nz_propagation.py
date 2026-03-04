import numpy as np

from fmonz.reconstruction.kernel_inversion import propagate_nz
from fmonz.utils.vec import vec, unvec


def test_propagation_zero_kernel():
    d = 2
    d2 = d * d
    n_t = 5
    K = np.zeros((n_t, d2, d2), dtype=complex)
    rho0 = np.eye(d) / d
    rhos = propagate_nz(K, rho0, dt=0.1)
    # kernel zero -> state should remain constant
    for n in range(n_t):
        assert np.allclose(rhos[n], rho0)


def test_propagation_scalar():
    # d=1 scalar case; K constant k -> v[n+1] = v[n] + dt*k*sum_weights*v[n]
    d = 1
    d2 = 1
    n_t = 4
    k_val = 2.0
    K = np.zeros((n_t, d2, d2), dtype=complex)
    K[:, 0, 0] = k_val
    rho0 = np.array([[1.0]])
    dt = 0.1
    rhos = propagate_nz(K, rho0, dt=dt)
    vs = np.array([r[0, 0] for r in rhos])
    # compute expected via manual recurrence
    def weight(m, n):
        if m == 0 or m == n:
            return dt / 2
        else:
            return dt
    expected = [1.0]
    for n in range(n_t - 1):
        rhs = 0.0
        for m in range(n + 1):
            rhs += weight(n - m, n) * k_val * expected[m]
        expected.append(expected[n] + dt * rhs)
    assert np.allclose(vs, expected)


def test_dimension_mismatch():
    # K has incompatible flattened dimension
    K = np.zeros((3, 5, 5))
    rho0 = np.eye(2)
    with np.testing.assert_raises(ValueError):
        propagate_nz(K, rho0, dt=0.1)
