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
    temperature: float
    reorg_energy: float
    cutoff: float
    hierarchy_depth: int
    matsubara_terms: int = 0



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
        ),
        time=TimeGrid(
            dt=float(time["dt"]),
            n_steps=int(time["n_steps"]),
        ),
    )
