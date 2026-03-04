<<<<<<< HEAD
"""Script to reconstruct dynamical maps from data."""

def main():
    print("Reconstruct map stub")


if __name__ == "__main__":
    main()
=======
"""Script to reconstruct dynamical maps from a configuration file.

This driver executes the first stage of the pipeline: given a Hamiltonian
and bath parameters it propagates a complete operator basis and writes the
resulting dynamical map ``Lambda(t)`` to disk.  It mirrors the behaviour
of the larger :mod:`run_pipeline` script but stops after the map is
constructed.
"""

from __future__ import annotations

import argparse
import datetime
from pathlib import Path

import numpy as np

from fmonz.config import load_config
from fmonz.physics.hamiltonian import build_hamiltonian
from fmonz.solvers.heom_interface import DummyHEOM
from fmonz.reconstruction.dynamical_map import reconstruct_dynamical_map, save_map


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Reconstruct dynamical map from HEOM runs"
    )
    parser.add_argument("--config", required=True, help="Path to TOML config file")
    parser.add_argument(
        "--outdir",
        help="Directory to store output; defaults to ./results/YYYYmmdd_HHMM",
    )
    parser.add_argument(
        "--use-dummy",
        action="store_true",
        help="Use the trivial dummy solver",
    )
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    times = cfg.time.times
    d = cfg.system.d
    H = build_hamiltonian(cfg.system)

    if args.use_dummy:
        heom = DummyHEOM(cfg.bath, d)
    else:
        try:
            from fmonz.solvers.heom_quutip import QuTiPHEOM
        except ImportError:
            raise RuntimeError("QuTiP is not available; re-run with --use-dummy")
        heom = QuTiPHEOM(H, cfg.bath)

    from fmonz.utils.basis import operator_basis

    basis = operator_basis(d, kind="matrix")
    Lambda = reconstruct_dynamical_map(heom, basis, times)

    if args.outdir:
        outdir = Path(args.outdir)
    else:
        outdir = Path("results") / datetime.datetime.now().strftime("run_%Y%m%d_%H%M")
    outdir.mkdir(parents=True, exist_ok=True)

    save_map(str(outdir / "Lambda.npz"), Lambda, d=d, dt=cfg.time.dt, convention="col")
    print("Lambda saved to", outdir / "Lambda.npz")


if __name__ == "__main__":
    main()
>>>>>>> b1437fe (Add FMO NZ kernel scripts and update packaging metadata)
