import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from fmonz.utils import io


def test_save_npz_and_report(tmp_path):
    arr = np.arange(6).reshape(2, 3)
    fname = tmp_path / "test.npz"
    io.save_npz(str(fname), foo=arr, bar=3)
    data = np.load(fname)
    assert np.array_equal(data["foo"], arr)
    assert int(data["bar"]) == 3

    report = {"a": 1, "b": [1, 2]}
    repfile = tmp_path / "rep.json"
    io.save_report(str(repfile), report)
    with open(repfile) as f:
        loaded = json.load(f)
    assert loaded == report


def test_run_pipeline_script(tmp_path):
    # write a minimal config file
    config_text = """
[system]
d = 2
site_energies = [0.0, 0.0]
couplings = [[0.0, 0.0],[0.0, 0.0]]

[bath]
temperature = 300.0
reorganization_energy = 1.0
cutoff = 50.0
hierarchy_depth = 1
matsubara_terms = 0

[time]
dt = 0.1
n_steps = 5
"""
    cfgfile = tmp_path / "config.toml"
    cfgfile.write_text(config_text)

    outdir = tmp_path / "results"
    script = Path(__file__).parents[1] / "scripts" / "run_pipeline.py"
    # run via subprocess so that sys.path behaves normally
    proc = subprocess.run(
        [sys.executable, str(script), "--config", str(cfgfile), "--outdir", str(outdir), "--use-dummy"],
        cwd=str(script.parent.parent),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
<<<<<<< HEAD
=======

    # also try with the QuTiP backend if available; it should produce the
    # same set of output files (the dynamics are trivial given the toy
    # Hamiltonian, so we don't check their contents).
    try:
        import qutip  # noqa: F401
    except ImportError:
        # nothing to do when qutip is missing
        pass
    else:
        outdir2 = tmp_path / "results2"
        proc2 = subprocess.run(
            [sys.executable, str(script), "--config", str(cfgfile), "--outdir", str(outdir2)],
            cwd=str(script.parent.parent),
            capture_output=True,
            text=True,
        )
        assert proc2.returncode == 0, proc2.stderr
        for fname in [
            "Lambda.npz",
            "dLambda.npz",
            "K_raw.npz",
            "K_reg.npz",
            "L_markov.npz",
            "validation_report.json",
        ]:
            assert (outdir2 / fname).exists(), f"missing {fname}"
>>>>>>> b1437fe (Add FMO NZ kernel scripts and update packaging metadata)
    # expected outputs
    for fname in [
        "Lambda.npz",
        "dLambda.npz",
        "K_raw.npz",
        "K_reg.npz",
        "L_markov.npz",
        "validation_report.json",
        "kernel_norm.png",
    ]:
        assert (outdir / fname).exists(), f"missing {fname}"
<<<<<<< HEAD
=======


def run_script(script, cfgfile, tmp_path):
    outdir = tmp_path / "out"
    proc = subprocess.run(
        [sys.executable, str(script), "--config", str(cfgfile), "--outdir", str(outdir), "--use-dummy"],
        cwd=str(script.parent.parent),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
    return outdir


def test_reconstruct_script(tmp_path):
    # reuse minimal config from above
    config_text = """
[system]
d = 2
site_energies = [0.0, 0.0]
couplings = [[0.0, 0.0],[0.0, 0.0]]

[bath]
temperature = 300.0
reorganization_energy = 1.0
cutoff = 50.0
hierarchy_depth = 1
matsubara_terms = 0

[time]
dt = 0.1
n_steps = 5
"""
    cfgfile = tmp_path / "config.toml"
    cfgfile.write_text(config_text)

    script = Path(__file__).parents[1] / "scripts" / "run_reconstruct_map.py"
    outdir = run_script(script, cfgfile, tmp_path)
    assert (outdir / "Lambda.npz").exists()


def test_smoke_script(tmp_path):
    # the smoke script uses its own example config when none provided
    script = Path(__file__).parents[1] / "scripts" / "run_heom_smoke.py"
    # run it once just to ensure it executes without error
    proc = subprocess.run(
        [sys.executable, str(script), "--use-dummy"],
        cwd=str(script.parent.parent),
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
>>>>>>> b1437fe (Add FMO NZ kernel scripts and update packaging metadata)
