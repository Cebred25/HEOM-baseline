"""Basis-related utilities.

A convenient orthonormal operator basis is needed for map reconstruction
and kernel extraction.  The Hilbert–Schmidt inner product

    ⟨A,B⟩ = Tr(A† B)

is used throughout.  Two common choices are supported:

* ``"matrix"`` – the normalized matrix units E_{ij}; these are not Hermitian
  except when i=j but are simple and complete.
* ``"gellmann"`` – the standard generalized Gell-Mann (Hermitian) basis
  plus the properly normalized identity.  This basis is convenient when one
  wants to enforce physical constraints such as trace preservation or
  positivity.

The public function ``operator_basis`` constructs a list of ``d*d``
operators for a given dimension.

The auxiliary ``orthonormalize`` implements Gram–Schmidt and may be applied
when needing to convert an arbitrary list of operators into an HS-orthonormal
set.
"""

from __future__ import annotations

import numpy as np
from typing import List


def orthonormalize(vectors: List[np.ndarray]) -> List[np.ndarray]:
    """Gram–Schmidt orthonormalization under the HS inner product.

    The input list ``vectors`` is mutated in-place and also returned.  Each
    element is assumed to be a square matrix; all matrices are flattened
    internally for inner-product computations.
    """
    ortho = []
    for v in vectors:
        w = v.astype(complex).copy()
        for u in ortho:
            coeff = np.trace(u.conj().T @ w)
            w = w - coeff * u
        norm = np.sqrt(np.trace(w.conj().T @ w))
        if norm < 1e-15:
            continue
        ortho.append(w / norm)
    return ortho


def operator_basis(d: int, kind: str = "gellmann") -> List[np.ndarray]:
    """Return a HS-orthonormal operator basis for dimension ``d``.

    Parameters
    ----------
    d : int
        Hilbert space dimension.
    kind : {"gellmann", "matrix"}
        Choice of basis.  ``"matrix"`` returns the normalized matrix units
        ``E_{ij}`` with Hilbert–Schmidt norm one; ``"gellmann"`` returns the
        generalized Gell‑Mann basis with the identity.

    Returns
    -------
    basis : list of ndarray
        List of ``d*d`` operators each shape ``(d,d)`` satisfying
        ``Tr(B_i.conj().T @ B_j) == delta_{ij}``.
    """
    if kind not in ("gellmann", "matrix"):
        raise ValueError("unknown basis kind %r" % kind)

    basis: List[np.ndarray] = []
    if kind == "matrix":
        for i in range(d):
            for j in range(d):
                E = np.zeros((d, d), dtype=complex)
                E[i, j] = 1.0
                # normalization: HS norm sqrt(Tr(E^† E)) = 1
                basis.append(E)
        return basis

    # gellmann
    # identity normalized by 1/sqrt(d)
    ident = np.eye(d, dtype=complex) / np.sqrt(d)
    basis.append(ident)

    # off-diagonal symmetric and antisymmetric generators
    for i in range(d):
        for j in range(i + 1, d):
            # symmetric (|i><j| + |j><i|) / sqrt(2)
            S = np.zeros((d, d), dtype=complex)
            S[i, j] = 1
            S[j, i] = 1
            S = S / np.sqrt(2)
            basis.append(S)
            # antisymmetric (-i|i><j| + i|j><i|)/sqrt(2)
            A = np.zeros((d, d), dtype=complex)
            A[i, j] = -1j
            A[j, i] = 1j
            A = A / np.sqrt(2)
            basis.append(A)

    # d-1 diagonal traceless matrices
    for k in range(1, d):
        D = np.zeros((d, d), dtype=complex)
        for l in range(k):
            D[l, l] = 1
        D[k, k] = -k
        D = D / np.sqrt(k * (k + 1))
        basis.append(D)

    # At this point we have 1 + 2*(d choose 2) + (d-1) = d^2 elements
    assert len(basis) == d * d
    return basis
