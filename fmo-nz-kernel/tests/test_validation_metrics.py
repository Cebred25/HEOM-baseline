import numpy as np
import pytest

from fmonz.validation.metrics import (
    validation_states,
    trace_distance,
    population_curves,
    coherence_magnitude,
)


def test_validation_states_counts():
    d = 3
    states = validation_states(d, n_random=5, seed=42)
    # should include d localized + d(d-1)/2 superpositions + thermal + random
    expected = d + d * (d - 1) // 2 + 1 + 5
    assert len(states) == expected
    for rho in states:
        assert rho.shape == (d, d)
        assert np.allclose(np.trace(rho), 1.0)
        # positive semidefiniteness
        eigs = np.linalg.eigvals(rho)
        assert np.all(eigs >= -1e-8)


def test_trace_distance():
    rho = np.diag([0.7, 0.3])
    sigma = np.diag([0.4, 0.6])
    # distance = 0.5*(|0.3|+| -0.3|) = 0.3
    assert np.allclose(trace_distance(rho, sigma), 0.3)


def test_population_and_coherence():
    # simple trajectory: two-level system oscillating
    rhos = np.zeros((4, 2, 2), dtype=complex)
    for n in range(4):
        angle = n * np.pi / 3
        psi = np.array([np.cos(angle), np.sin(angle)], dtype=complex)
        rhos[n] = np.outer(psi, psi.conj())
    pops = population_curves(rhos)
    assert pops.shape == (4, 2)
    # coherence magnitude equals Frobenius norm of off-diagonals.
    # for 2x2 pure state this gives sqrt(2)*|cosθ sinθ| = |sin2θ|/sqrt(2)
    mags = coherence_magnitude(rhos)
    expected = np.array([
        0.0,
        np.abs(np.sin(2 * np.pi / 3)) / np.sqrt(2),
        np.abs(np.sin(4 * np.pi / 3)) / np.sqrt(2),
        0.0,
    ])
    assert np.allclose(mags, expected)

def test_kernel_norm_and_memory_times():
    # simple scalar kernel: norm curve should reproduce exponential
    times = np.linspace(0, 5, 101)
    kn_scalar = np.exp(-times)
    # shape (n,1,1) so kernel_norm_curve reduces to scalar norms
    K = kn_scalar[:, None, None]

    from fmonz.validation.metrics import (
        kernel_norm_curve,
        memory_time_threshold,
        memory_time_tailweight,
    )

    kn_fro = kernel_norm_curve(K, norm="fro")
    assert kn_fro.shape == times.shape
    assert np.allclose(kn_fro, kn_scalar)

    kn_op = kernel_norm_curve(K, norm="op")
    assert np.allclose(kn_op, kn_scalar)

    with pytest.raises(ValueError):
        kernel_norm_curve(K, norm="invalid")

    # threshold time should be near -log(eps)
    tau_th = memory_time_threshold(times, kn_fro, eps=0.1)
    assert abs(tau_th + np.log(0.1)) <= times[1] - times[0]

    # tail weight for exponential has W(t)=exp(-t) approximately
    tau_tw = memory_time_tailweight(times, kn_fro, delta=0.1)
    # allow tolerance of two time steps due to discretization error
    assert abs(tau_tw + np.log(0.1)) <= 2 * (times[1] - times[0])

    # constant kernel never decays
    const = np.ones_like(kn_fro)
    assert memory_time_threshold(times, const, eps=0.01) == times[-1]
    # result should be at or very near the final time; allow one-step error
    tau_const = memory_time_tailweight(times, const, delta=0.01)
    assert abs(tau_const - times[-1]) <= times[1] - times[0]