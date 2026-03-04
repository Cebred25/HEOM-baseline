"""Input/output helpers.

This small module contains convenience functions for writing common
artifacts produced by the pipeline.  Higher-level logic (scripts) can use
these to keep file formats consistent without repeating boilerplate.
"""

import json
from typing import Any


def save_array(path, arr):
    import numpy as np

    np.save(path, arr)


def save_npz(path: str, **kwargs: Any) -> None:
    """Save multiple arrays or values into a compressed ``.npz`` file.

    Parameters
    ----------
    path : str
        Destination file name (should end with ``.npz``).
    **kwargs : Any
        Keyword arguments passed to ``numpy.savez_compressed``.
    """

    import numpy as np

    np.savez_compressed(path, **kwargs)


def save_report(path: str, report: Any) -> None:
    """Write a JSON-serializable object to disk with indentation.

    Parameters
    ----------
    path : str
        Output file name (usually ending in ``.json``).
    report : Any
        Object that can be serialized by :func:`json.dump`.
    """

    with open(path, "w") as f:
        json.dump(report, f, indent=2)
