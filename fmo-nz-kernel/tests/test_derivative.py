import numpy as np

from fmonz.reconstruction.kernel_inversion import time_derivative_superop


def make_linear_map(n_t, d2, slope=1.0, dt=1.0):
    # Lambda[n] = I * (slope * t_n) with t_n = n*dt
    L = np.zeros((n_t, d2, d2), dtype=complex)
    for n in range(n_t):
        L[n] = np.eye(d2) * (slope * n * dt)
    return L


def test_central_derivative():
    n_t = 5
    d2 = 3
    dt = 0.1
    L = make_linear_map(n_t, d2, slope=2.0, dt=dt)
    dL = time_derivative_superop(L, dt)
    # derivative should equal identity*2.0 at interior and boundaries
    for n in range(n_t):
        assert np.allclose(dL[n], np.eye(d2) * 2.0)


def test_one_sided_method():
    n_t = 4
    d2 = 2
    dt = 0.05
    L = make_linear_map(n_t, d2, slope=3.0, dt=dt)
    # use forward differences everywhere (equivalent to previous one-sided)
    dL = time_derivative_superop(L, dt, method="forward")
    assert np.allclose(dL, np.eye(d2) * 3.0)


def test_smoothing_option():
    # create a noisy linear map and smooth it
    n_t = 7
    d2 = 2
    dt = 0.2
    L = make_linear_map(n_t, d2, slope=1.0)
    rng = np.random.default_rng(0)
    L += (rng.standard_normal(L.shape) + 1j * rng.standard_normal(L.shape)) * 1e-3
    dL_raw = time_derivative_superop(L, dt, method="central")
    dL_smooth = time_derivative_superop(
        L,
        dt,
        method="central",
        smooth={"method": "savitzky_golay", "window_length": 5, "polyorder": 2},
    )
    # smoothing should reduce variance in off-diagonal elements
    var_raw = np.var(dL_raw)
    var_smooth = np.var(dL_smooth)
    # compute mean absolute error relative to ideal slope
    ideal = np.eye(d2) * 1.0
    err_raw = np.mean(np.abs(dL_raw - ideal))
    err_smooth = np.mean(np.abs(dL_smooth - ideal))
    assert err_smooth <= err_raw + 1e-8
