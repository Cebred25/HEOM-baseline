"""Command-line interface for fmonz (optional Typer-based).

This module provides lightweight commands that wrap the existing scripts
in the package. Installing the optional `cli` extras gives access to the
`fmonz` command via an entry point (not configured here to keep packaging
changes minimal). If `typer` is not installed this module still imports
safely only when invoked.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

try:
    import typer
    from rich import print as rprint
except Exception:  # pragma: no cover - optional dependency
    typer = None


def _run_script(script_name: str, argv: list[str] | None = None) -> int:
    base = Path(__file__).parents[2]
    script = base / "scripts" / script_name
    cmd = [sys.executable, str(script)] + (argv or [])
    proc = subprocess.run(cmd, cwd=str(base), capture_output=False)
    return proc.returncode


if typer is not None:
    app = typer.Typer()


    @app.command()
    def pipeline(config: str, outdir: str | None = None, use_dummy: bool = False, n_jobs: int = 1, parallel: str = "none"):
        """Run the full pipeline (wraps scripts/run_pipeline.py)."""
        args = ["--config", config]
        if outdir:
            args += ["--outdir", outdir]
        if use_dummy:
            args += ["--use-dummy"]
        if n_jobs != 1:
            args += ["--n-jobs", str(n_jobs)]
        if parallel != "none":
            args += ["--parallel", parallel]
        raise SystemExit(_run_script("run_pipeline.py", args))


    @app.command()
    def map(config: str, outdir: str | None = None, use_dummy: bool = False, resume: bool = False, n_jobs: int = 1, parallel: str = "none"):
        """Reconstruct Lambda(t) (wraps scripts/run_reconstruct_map.py)."""
        args = ["--config", config]
        if outdir:
            args += ["--outdir", outdir]
        if use_dummy:
            args += ["--use-dummy"]
        if resume:
            args += ["--resume"]
        if n_jobs != 1:
            args += ["--n-jobs", str(n_jobs)]
        if parallel != "none":
            args += ["--parallel", parallel]
        raise SystemExit(_run_script("run_reconstruct_map.py", args))


    @app.command()
    def smoke(config: str | None = None, use_dummy: bool = True):
        """Run a quick smoke test."""
        args: list[str] = []
        if config:
            args += ["--config", config]
        if use_dummy:
            args += ["--use-dummy"]
        raise SystemExit(_run_script("run_heom_smoke.py", args))


else:  # minimal fallback

    def pipeline(config: str, outdir: str | None = None, use_dummy: bool = False, **_):
        return _run_script("run_pipeline.py", ["--config", config] + (["--outdir", outdir] if outdir else []) + (["--use-dummy"] if use_dummy else []))

    def map(config: str, outdir: str | None = None, use_dummy: bool = False, resume: bool = False, **_):
        args = ["--config", config]
        if outdir:
            args += ["--outdir", outdir]
        if use_dummy:
            args += ["--use-dummy"]
        if resume:
            args += ["--resume"]
        return _run_script("run_reconstruct_map.py", args)

    def smoke(config: str | None = None, use_dummy: bool = True):
        args: list[str] = []
        if config:
            args += ["--config", config]
        if use_dummy:
            args += ["--use-dummy"]
        return _run_script("run_heom_smoke.py", args)
