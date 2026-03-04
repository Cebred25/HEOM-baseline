"""Configuration management for the FMO NZ kernel project.

This module defines structured dataclasses for the various pieces of
input needed by the pipeline.  A TOML (or YAML conversion) file can be
read via :func:`load_config` to obtain a :class:`FullConfig` instance.
The schema roughly matches the description in the project goal:

```
[system]
d = 7
site_energies = [ ... ]
couplings = [ [...], ... ]

[bath]
temperature = 300.0
reorganization_energy = 35.0
cutoff = 50.0
hierarchy_depth = 4
matsubara_terms = 2

[time]
dt = 0.1
n_steps = 1001
```
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
import numpy as np


@dataclass
class SystemConfig:
    d: int
    site_energies: np.ndarray
    couplings: np.ndarray



@dataclass
class BathParams:
<<<<<<< HEAD
=======
    """Parameters describing a bosonic bath used by HEOM solvers.

    The naming follows the convention in the QuTiP documentation when
    possible.  Older code in the repository still refers to the more
    verbose names ``temperature``, ``reorg_energy`` and ``cutoff``; those
    fields are retained for backwards compatibility.  The additional
    options are used only by the QuTiP backend.

    Attributes
    ----------
    temperature
        Bath temperature (we work in units where $k_B=1$).
    reorg_energy
        Reorganization energy \(\lambda\).
    cutoff
        Drude–Lorentz cutoff frequency \(\gamma\).
    hierarchy_depth
        Maximum hierarchy depth to include in the HEOM.
    matsubara_terms
        Number of Matsubara or P\"ade expansion terms (``Nk`` in
        QuTiP parlance).
    use_pade
        If ``True`` use a P\"ade expansion for the bath correlation
        function instead of the Matsubara series.  Default ``False``.
    add_terminator
        Whether to include the approximate terminator correction when the
        hierarchy is truncated.  Default ``True``.
    """

>>>>>>> b1437fe (Add FMO NZ kernel scripts and update packaging metadata)
    temperature: float
    reorg_energy: float
    cutoff: float
    hierarchy_depth: int
    matsubara_terms: int = 0
<<<<<<< HEAD
=======
    use_pade: bool = False
    add_terminator: bool = True
>>>>>>> b1437fe (Add FMO NZ kernel scripts and update packaging metadata)



@dataclass
class TimeGrid:
    dt: float
    n_steps: int

    @property
    def t_max(self) -> float:
        return self.dt * (self.n_steps - 1)

    @property
    def times(self) -> np.ndarray:
        return np.linspace(0, self.t_max, self.n_steps)



@dataclass
class FullConfig:
    system: SystemConfig
    bath: BathParams
    time: TimeGrid


def load_config(path: str) -> FullConfig:
    """Read a TOML configuration file and return a :class:`FullConfig`.

    The function expects the file to contain ``system``, ``bath`` and
    ``time`` sections as shown above.  Arrays are converted to numpy
    arrays automatically.
    """

    try:
        import toml
    except ImportError as exc:
        raise RuntimeError("toml library required to load configuration") from exc

    data = toml.load(path)
    sys = data["system"]
    bath = data["bath"]
    time = data["time"]

    return FullConfig(
        system=SystemConfig(
            d=int(sys["d"]),
            site_energies=np.array(sys["site_energies"]),
            couplings=np.array(sys["couplings"]),
        ),
        bath=BathParams(
            temperature=float(bath["temperature"]),
            reorg_energy=float(bath["reorganization_energy"]),
            cutoff=float(bath["cutoff"]),
            hierarchy_depth=int(bath["hierarchy_depth"]),
            matsubara_terms=int(bath.get("matsubara_terms", 0)),
<<<<<<< HEAD
=======
            use_pade=bool(bath.get("use_pade", False)),
            add_terminator=bool(bath.get("add_terminator", True)),
>>>>>>> b1437fe (Add FMO NZ kernel scripts and update packaging metadata)
        ),
        time=TimeGrid(
            dt=float(time["dt"]),
            n_steps=int(time["n_steps"]),
        ),
    )
