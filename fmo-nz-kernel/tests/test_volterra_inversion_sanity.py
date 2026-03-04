import numpy as np

from fmonz.reconstruction.kernel_inversion import (
    time_derivative_superop,
    invert_volterra_kernel,
)


def test_time_derivative_linear():
    # Lambda[n] = t_n * A for constant matrix A -> derivative should equal A
    d = 2
    A = np.array([[1.0, 2.0], [3.0, 4.0]])
    dt = 0.1
    times = np.linspace(0, 1, 11)
    Lambda = np.stack([t * A for t in times])
    dLambda = time_derivative_superop(Lambda, dt)
    for n in range(len(times)):
        assert np.allclose(dLambda[n], A)


def test_time_derivative_smoothing():
    # smoothing should reduce error compared to unsmoothed finite difference
    A = np.array([[2.0]])
    dt = 0.1
    times = np.linspace(0, 1, 21)
    clean = np.stack([t * A for t in times])
    rng = np.random.default_rng(0)
    noisy = clean + 0.01 * rng.standard_normal(clean.shape)
    d_unsm = time_derivative_superop(noisy, dt)
    d_smooth = time_derivative_superop(
        noisy, dt, smooth={"method": "savitzky_golay", "window_length": 7, "polyorder": 2}
    )
    err_unsm = np.linalg.norm(d_unsm - A)
    err_s = np.linalg.norm(d_smooth - A)
    assert err_s < err_unsm


def test_volterra_inversion_recover():
    # random kernel and Lambda; compute dLambda via discretized convolution,
    # then try to invert back
    dt = 0.05
    n_t = 6
    d2 = 3
    rng = np.random.default_rng(1)
    K_true = rng.standard_normal((n_t, d2, d2)) + 1j * rng.standard_normal((n_t, d2, d2))
    # choose Lambda randomly but ensure Lambda[0]=I
    Lambda = np.zeros((n_t, d2, d2), dtype=complex)
    Lambda[0] = np.eye(d2, dtype=complex)
    for n in range(1, n_t):
        Lambda[n] = rng.standard_normal((d2, d2))
    # compute dLambda using forward formula
    def w(m, n):
        return dt / 2 if m == 0 or m == n else dt

    dLambda = np.zeros_like(Lambda)
    for n in range(n_t):
        R = np.zeros((d2, d2), dtype=complex)
        for m in range(n + 1):
            R += w(m, n) * (K_true[n - m] @ Lambda[m])
        dLambda[n] = R

    K_rec = invert_volterra_kernel(Lambda, dLambda, dt)
    assert np.allclose(K_rec, K_true)

    # check that zero rule gives zero at t0 regardless
    K_zero = invert_volterra_kernel(Lambda, dLambda, dt, K0_rule="zero")
    assert np.allclose(K_zero[0], 0)
