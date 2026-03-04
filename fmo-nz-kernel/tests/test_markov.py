import numpy as np

from fmonz.reconstruction.kernel_inversion import (
    build_markov_generator,
    propagate_markov,
)


def test_build_markov_instantaneous():
    # simple kernel where K varies; expect first slice
    K = np.zeros((4, 2, 2), dtype=complex)
    K[0] = np.eye(2)
    K[1] = np.eye(2) * 2
    L = build_markov_generator(K, dt=0.1, mode="instantaneous")
    assert np.allclose(L, np.eye(2))


def test_build_markov_integral():
    # weight trapezoid
    K = np.zeros((3, 1, 1), dtype=complex)
    K[0, 0, 0] = 1.0
    K[1, 0, 0] = 2.0
    K[2, 0, 0] = 3.0
    dt = 0.5
    L = build_markov_generator(K, dt, mode="integral")
    # weights: dt/2, dt, dt/2 -> 0.25,0.5,0.25
    expected = 1.0 * 0.25 + 2.0 * 0.5 + 3.0 * 0.25
    assert np.allclose(L, expected)


def test_build_markov_invalid_mode():
    K = np.zeros((1, 1, 1))
    try:
        build_markov_generator(K, dt=0.1, mode="unknown")
    except ValueError:
        return
    assert False


def test_propagate_markov_constant():
    # constant generator L -> simple exponential growth via Euler
    d = 2
    d2 = d * d
    L = np.eye(d2) * 0.1
    rho0 = np.eye(d) / d
    times = np.linspace(0, 0.3, 4)
    rhos = propagate_markov(L, rho0, times)
    # compare to iterative formula v[n+1]=v[n]+dt*L*v[n]
    from fmonz.utils.vec import vec, unvec
    v = vec(rho0)
    dt = times[1] - times[0]
    for n in range(len(times) - 1):
        v = v + dt * (L @ v)
        assert np.allclose(unvec(v, d), rhos[n + 1])


def test_propagate_markov_dimension_mismatch():
    # generator does not have correct flattened dimension
    L = np.zeros((3, 3))
    rho0 = np.eye(2)
    times = np.linspace(0, 1, 3)
    with np.testing.assert_raises(ValueError):
        propagate_markov(L, rho0, times)
