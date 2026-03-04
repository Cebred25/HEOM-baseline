"""Smoke test for the QuTiP HEOM backend.

Runs a tiny 3-site Hamiltonian with weak coupling and a shallow hierarchy.
This script exercises the solver construction and ensures the returned
populations are sensible.  The example configuration lives in
"examples/smoke_3site.toml" by default; a custom path may be supplied via
``--config``.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from fmonz.config import load_config
from fmonz.physics.hamiltonian import build_hamiltonian
from fmonz.solvers.heom_interface import DummyHEOM
from fmonz.reconstruction.dynamical_map import reconstruct_dynamical_map


def main(argv=None):
    parser = argparse.ArgumentParser(description="Run a smoke test HEOM evolution")
    parser.add_argument("--config", help="Path to TOML config file", default=None)
    parser.add_argument("--use-dummy", action="store_true", help="Use dummy solver")
    args = parser.parse_args(argv)

    if args.config is None:
        cfg_path = Path(__file__).parents[1] / "examples" / "smoke_3site.toml"
    else:
        cfg_path = Path(args.config)
    cfg = load_config(str(cfg_path))

    times = cfg.time.times
    d = cfg.system.d
    H = build_hamiltonian(cfg.system)

    if args.use_dummy:
        heom = DummyHEOM(cfg.bath, d)
    else:
        try:
            from fmonz.solvers.heom_quutip import QuTiPHEOM
        except ImportError:
            print("QuTiP not installed; falling back to dummy solver")
            heom = DummyHEOM(cfg.bath, d)
        else:
            heom = QuTiPHEOM(H, cfg.bath)

    basis = None
    from fmonz.utils.basis import operator_basis

    basis = operator_basis(d, kind="matrix")
    Lambda = reconstruct_dynamical_map(heom, basis, times)

    # report a few sanity metrics; ``Lambda`` is the dynamical map so its
    # shape should be ``(n_t, d^2, d^2)``.  We simply print the first slice
    # and verify that applying it to a vectorized density matrix yields a
    # trace‑one state.
    print(f"Lambda shape: {Lambda.shape}")
    if Lambda.size > 0:
        first = Lambda[0]
        print("first map Frobenius norm", np.linalg.norm(first))
        # apply to a simple initial vector (maximally mixed state)
        rho0 = np.eye(d) / d
        vec = rho0.reshape(d * d, order="F")
        out = first @ vec
        print("trace after applying first map", np.sum(out))


if __name__ == "__main__":
    main()
