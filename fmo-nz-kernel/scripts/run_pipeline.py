"""Entry point for running the full analysis pipeline.

This script ties together the various stages described in the README: it
loads a configuration file, constructs the Hamiltonian and operator basis,
propagates the HEOM for each basis operator, reconstructs the dynamical map,
computes the time derivative and memory kernel, applies regularization, and
then compares non‑Markovian and Markovian propagation.  Various intermediate
artifacts are written to disk so that the computation can be resumed or
analyzed offline.

The output directory will contain:

* ``Lambda.npz`` – dynamical map with metadata
* ``dLambda.npz`` – time derivative of the map
* ``K_raw.npz`` and ``K_reg.npz`` – raw and regularized kernels
* ``L_markov.npz`` – Markovian generator
* ``validation_report.json`` – simple metrics summary
* figures (``kernel_norm.png``/``.pdf``)

The code is deliberately generic; a real HEOM backend may be plugged in by
modifying the ``HEOMSolver`` instance.  For demonstration or testing the
``--use-dummy`` flag forces a trivial solver that returns the initial
operator unchanged.
"""

from __future__ import annotations

import argparse
import datetime
import os
import sys
from pathlib import Path

import numpy as np

from fmonz.config import load_config
from fmonz.physics.hamiltonian import build_hamiltonian
from fmonz.utils.basis import operator_basis
from fmonz.solvers.heom_interface import DummyHEOM, HEOMSolver
from fmonz.reconstruction.dynamical_map import reconstruct_dynamical_map, save_map
from fmonz.reconstruction.kernel_inversion import (
    time_derivative_superop,
    invert_volterra_kernel,
    enforce_constraints_on_kernel,
    build_markov_generator,
    propagate_nz,
    propagate_markov,
)
from fmonz.utils.io import save_npz, save_report
from fmonz.validation.metrics import (
    validation_states,
    trace_distance,
    kernel_norm_curve,
    memory_time_threshold,
    memory_time_tailweight,
)


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        description="Run the FMO NZ-kernel extraction and validation pipeline",
    )
    parser.add_argument(
        "--config",
        required=True,
        help="Path to TOML configuration file",
    )
    parser.add_argument(
        "--outdir",
        help="Directory in which to save outputs (created if necessary)",
    )
    parser.add_argument(
        "--use-dummy",
        action="store_true",
        help="Use the built-in dummy HEOM solver instead of a real backend",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    dt = cfg.time.dt
    times = cfg.time.times
    d = cfg.system.d

    # build objects
    H = build_hamiltonian(cfg.system)

    if args.use_dummy:
        heom = DummyHEOM(cfg.bath, d)
    else:
        try:
            from fmonz.solvers.heom_quutip import QuTiPHEOMSolver
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "QuTiP backend requested but qutip is not installed"
            ) from exc
        heom = QuTiPHEOMSolver(H, cfg.bath, tlist=times)

    basis = operator_basis(d, kind="matrix")
    Lambda = reconstruct_dynamical_map(heom, basis, times)

    if args.outdir:
        outdir = Path(args.outdir)
    else:
        outdir = Path("results") / datetime.datetime.now().strftime("run_%Y%m%d_%H%M")
    outdir.mkdir(parents=True, exist_ok=True)

    save_map(str(outdir / "Lambda.npz"), Lambda, d=d, dt=dt, convention="col")
    dLambda = time_derivative_superop(Lambda, dt)
    save_npz(str(outdir / "dLambda.npz"), dLambda=dLambda, dt=dt)

    K_raw = invert_volterra_kernel(Lambda, dLambda, dt)
    save_npz(str(outdir / "K_raw.npz"), K=K_raw, d=d, dt=dt)

    K_reg = enforce_constraints_on_kernel(K_raw, d, vec_convention="col")
    save_npz(str(outdir / "K_reg.npz"), K=K_reg, d=d, dt=dt)

    L_markov = build_markov_generator(K_reg, dt)
    save_npz(str(outdir / "L_markov.npz"), L=L_markov, dt=dt)

    # validation / memory times
    report: dict[str, float] = {}
    knorm = kernel_norm_curve(K_reg, norm="fro")
    report["tau_mem_threshold"] = memory_time_threshold(times, knorm)
    report["tau_mem_tailweight"] = memory_time_tailweight(times, knorm)

    states = validation_states(d, n_random=3, seed=0)
    for idx, rho0 in enumerate(states[:2]):
        r_nz = propagate_nz(K_reg, rho0, dt)
        r_m = propagate_markov(L_markov, rho0, times)
        report[f"trace_dist_nz_{idx}"] = float(trace_distance(r_nz[-1], rho0))
        report[f"trace_dist_markov_{idx}"] = float(trace_distance(r_m[-1], rho0))

    save_report(str(outdir / "validation_report.json"), report)

    # figures
    try:
        import matplotlib.pyplot as plt

        plt.figure()
        plt.plot(times, knorm, "-o")
        plt.xlabel("time")
        plt.ylabel("kernel norm")
        plt.title("Kernel norm curve")
        plt.savefig(str(outdir / "kernel_norm.png"))
        plt.savefig(str(outdir / "kernel_norm.pdf"))
        plt.close()
    except ImportError:
        print("matplotlib not available; skipping figure generation")

    print(f"Pipeline complete; results saved in {outdir}")


if __name__ == "__main__":
    main()
