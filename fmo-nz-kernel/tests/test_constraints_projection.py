import numpy as np

from fmonz.reconstruction.kernel_inversion import enforce_constraints_on_kernel


def random_kernel(n_t, d2, seed=0):
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n_t, d2, d2)) + 1j * rng.standard_normal((n_t, d2, d2))


def test_trace_preservation_projection():
    d = 3
    d2 = d * d
    Kraw = random_kernel(4, d2)
    Kproj = enforce_constraints_on_kernel(Kraw, d, enforce_hp=False, enforce_tp=True)
    # bra_I action should be zero
    order = "F"
    u = np.eye(d).reshape(d2, order=order)
    u = u / np.linalg.norm(u)
    for n in range(4):
        v = u.conj().T @ Kproj[n]
        assert np.allclose(v, 0, atol=1e-12)


def test_hermiticity_preservation_projection():
    d = 2
    d2 = d * d
    Kraw = random_kernel(3, d2, seed=1)
    Kproj = enforce_constraints_on_kernel(Kraw, d, enforce_tp=False, enforce_hp=True)
    # check the symmetry property
    for n in range(3):
        M = Kproj[n]
        for i in range(d2):
            for j in range(d2):
                # compute swapped indices manually
                a1 = i % d
                a2 = i // d
                b1 = j % d
                b2 = j // d
                ip = a2 + a1 * d
                jp = b2 + b1 * d
                assert np.allclose(M[i, j], np.conj(M[ip, jp]))


def test_smoothing_in_constraint():
    d = 2
    d2 = d * d
    Kraw = random_kernel(5, d2)
    # apply strong smoothing to see change
    Kproj = enforce_constraints_on_kernel(
        Kraw,
        d,
        enforce_tp=False,
        enforce_hp=False,
        smooth={"method": "savitzky_golay", "window_length": 5, "polyorder": 2},
    )
    assert not np.allclose(Kproj, Kraw)
