# HEOM-baseline

This workspace now contains a new package skeleton under `fmo-nz-kernel/` with the following layout:

```
fmo-nz-kernel/
  README.md
  pyproject.toml
  src/
    fmonz/
      __init__.py
      config.py
      utils/
        linalg.py
        basis.py
        vec.py
        io.py
        differencing.py
        regularization.py
        constraints.py
      physics/
        hamiltonian.py
        liouvillian.py
        initial_states.py
      solvers/
        heom_interface.py
        heom_quutip.py              # optional implementation adapter
        nz_propagator.py
        markov_propagator.py
      reconstruction/
        dynamical_map.py
        kernel_inversion.py
        memory_metrics.py
      validation/
        metrics.py
        benchmarks.py
      plotting/
        figures.py
  scripts/
    run_pipeline.py
    run_reconstruct_map.py
    run_extract_kernel.py
    run_validate.py
    run_markov_compare.py
  tests/
    test_vec_conventions.py
    test_basis_orthonormality.py
    test_map_reconstruction_shape.py
    test_volterra_inversion_sanity.py
    test_constraints_projection.py
    test_hamiltonian.py
    test_heom_interface.py
  data/
    (generated outputs, gitignored)
  results/
    (figures, kernel dumps, reports)
```

## Conventions and Stage A model

The code now implements the base conventions described in the project goal:

- Single‑exciton system dimension `d` (typically 7 or 8) is configurable via
  `fmonz.config.SystemConfig`.
- Density matrices are `d×d` complex arrays, and vectorization uses column-
  stacking (Fortran order) by default.  Helpers `fmonz.utils.vec` and
  `unvec` manage the conversion and are exercised in the unit tests.
- Dynamical maps act on flattened density operators as
  ``vec(rho(t)) = Lambda[t] @ vec(rho0)``.
- A TOML configuration file is parsed by :func:`fmonz.config.load_config` into
  structured dataclasses (`SystemConfig`, `BathParams`, `TimeGrid`).
- `physics/hamiltonian.build_hamiltonian` constructs the Hamiltonian from a
  system config.
- HEOM solver interface is defined in `solvers/heom_interface.py`.  A
  prototype function ``heom_propagate_basis_operator`` follows the signature
  required for Stage A and simply raises ``NotImplementedError`` for now, and a
  corresponding dataclass ``BathParams`` holds bath/HEOM parameters.

With these pieces in place we are ready to implement the HEOM propagation and
later pipeline stages.

### Stage B – Operator basis

A complete Hilbert‑Schmidt orthonormal operator basis is now provided by
``fmonz.utils.basis.operator_basis``.  Two constructions are supported:

* ``kind="matrix"`` – normalized matrix units $E_{ij}$ (useful for general
  linear algebra).  These are simply the elementary basis matrices with
  HS-norm one.
* ``kind="gellmann"`` – the generalized Gell‑Mann (Hermitian) basis plus the
  identity normalized by $1/\sqrt{d}$.  This is convenient when enforcing
  trace preservation and other physical constraints.

The basis routine returns a list of $d^2$ matrices which are numerically
orthonormal under the inner product $\langle A,B\rangle=\mathrm{Tr}(A^\dagger B)$.
Unit tests verify orthonormality and confirm the Gram–Schmidt helper works
for arbitrary input sets.

---

With Stage B complete we can now expand maps and kernels in an operator basis
and work toward the Volterra inversion algorithms.

### Stage D – Differentiation of \(\Lambda(t)\)

A robust routine ``time_derivative_superop`` computes the time
derivative of a sequence of superoperators.  The default method is
central finite differences with first‑order forward/backward edges.  An
optional Savitzky–Golay smoothing step (via ``scipy.signal.savgol_filter``)
can be applied before differentiation to suppress noise.

### Stage E – Kernel inversion

The Volterra equation is inverted sequentially by
``invert_volterra_kernel``.  Starting from the estimated
``\Lambda(t)`` and its derivative, the code solves for the memory
kernel ``K(t)`` using trapezoidal quadrature and isolates each new term
by exploiting ``\Lambda(0)=I``.  The initial kernel value can either be
estimated from ``d\Lambda(0)`` or forced to zero; both options are
available via the ``K0_rule`` argument.

The resulting kernel array has shape ``(n_t,d^2,d^2)`` and may be
saved/loaded like the map itself.

### Stage F – Regularization and physical constraints

A post‑processing step ``enforce_constraints_on_kernel`` projects the raw
kernel sequence onto a subspace consistent with physically required
properties:

* **Temporal smoothness** can be optionally imposed via a
  Savitzky–Golay filter (or future regularizers) to penalize large
  time–derivatives.
* **Trace preservation (TP)**: the superoperator must satisfy
  \(\langle I|K(t)=0\) for all times.  We enforce this by subtracting the
  component along the vectorized identity.
* **Hermiticity preservation (HP)**: if the input density is Hermitian,
  the output of the kernel acts on its vectorization must also be
  Hermitian.  In practice we symmetrize matrix elements so that
  \(K_{ij}=\overline{K_{i'j'}}\) with row/column indices swapped.

The function accepts boolean flags to enable/disable each projection and
reuses the same ``smooth`` dictionary structure used elsewhere.

With these tools the extracted kernel can be made physically reasonable
before validation and Markov‑comparison begins.

### Stage H – Markov approximation & comparison

A simple Markov generator is obtained from the kernel by either
integrating it over time or taking only the instantaneous value.  The
helper ``build_markov_generator(K, dt, mode)`` returns the superoperator
``L`` used in the Markov master equation.  Two modes are available:
``"integral"`` (default) and ``"instantaneous"``.

Once a generator is available the dynamics is trivial to simulate with
``propagate_markov(L, rho0, times)``; the routine uses explicit Euler
integration in Liouville space.  These functions make it easy to
quantify the deviation between true NZ evolution and the Markov limit.

---

At this point the codebase contains a complete pipeline from HEOM data
through kernel extraction, regularization, and forward propagation
(both non‑Markovian and Markovian).

### Stage I – Validation & metrics

The newest additions focus on verifying the behaviour of dynamical maps
and comparing NZ/Markov results against exact HEOM benchmarks.  A small
collection of utilities now lives in ``validation/metrics.py``:

* ``random_pure_states(d, n=1)`` generates trace‑1, positive semidefinite
  density matrices uniformly sampled from the unit sphere in Hilbert space.
* ``trace_distance(rho, sigma)`` computes the trace norm distance between
  two density matrices.
* ``populations(rho)`` returns the diagonal elements as a real vector.
* ``coherence_magnitude(rho)`` returns the Frobenius norm of the
  off‑diagonal block, giving a single scalar measure of coherence.

Beyond simple state-level metrics, the package now includes utilities for
analyzing the memory kernel itself.  The following helpers live in the
same module:

* ``kernel_norm_curve(K, norm='fro')`` computes a time-dependent norm of the
  kernel sequence.  Both Frobenius (default) and operator/spectral norms are
  supported.
* ``memory_time_threshold(times, knorm, eps=1e-2)`` returns the earliest
  time where ``knorm(t)`` has decayed below ``eps*knorm(0)`` and remains
  underneath thereafter.
* ``memory_time_tailweight(times, knorm, delta=1e-2)`` finds the time when
  the normalized tail integral
  ``W(t)=\int_t^\infty knorm/\int_0^\infty knorm`` falls below ``delta``.

These definitions provide a quantitative memory timescale ``\tau_{\rm mem}``
which can be compared to system timescales such as population transfer
rates (obtained from ``population_curves``) or characteristic energy
splittings from the Hamiltonian.  Armed with ``\tau_{\rm mem}``, one can
assess the validity of the Born–Markov approximation and identify regimes
where non‑Markovian effects are important.

Unit tests (`tests/test_validation_metrics.py`) confirm all of the above
behave as expected: random states are Hermitian with unit trace, trace
distance is zero for identical matrices and maximal between orthogonal
pure states, population/coherence metrics reduce to known formulas for
simple two‑level examples, and the new memory-time routines reproduce
analytic behaviour for decaying kernels.

With validation metrics in place we can now proceed to benchmark NZ
propagation against exact dynamics and quantify Born–Markov failure.

### Data products and pipeline driver

A simple command‑line driver script (`scripts/run_pipeline.py`) orchestrates
the entire extraction/validation workflow.  It accepts a TOML configuration
file and an output directory, e.g.:```
python scripts/run_pipeline.py --config config.toml --outdir results/run01 \
    --use-dummy
``` 

The script produces the following artifacts under the specified `outdir`:

* **Lambda.npz** – compressed dynamical map along with dimension and step
  size metadata (see :func:`fmonz.reconstruction.dynamical_map.save_map`).
* **dLambda.npz** – derivative of the map.
* **K_raw.npz**, **K_reg.npz** – raw and constraint‑regularized memory
  kernels.
* **L_markov.npz** – Markovian generator obtained from the regularized kernel.
* **validation_report.json** – small JSON summary of memory times and simple
  trace‑distance errors for a handful of test states.
* **kernel_norm.png/.pdf** – figure showing the norm curve used to extract
  memory times.

Helper functions in ``fmonz.utils.io`` (`save_npz`, `save_report`) tidy the
file‑format conventions and make it easy to extend or repurpose the script.

The driver currently defaults to a trivial ``DummyHEOM`` solver; replace or
wrap it with a production backend to process real HEOM data.

---

---

With the kernel extractor working we can now proceed to validation (Stage
i) and quantify Markov vs non‑Markov behaviour.
