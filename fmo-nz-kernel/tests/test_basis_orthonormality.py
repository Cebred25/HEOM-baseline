import numpy as np


def hs_inner(A, B):
    return np.trace(A.conj().T @ B)


def check_orthonormal(basis):
    d2 = len(basis)
    for i in range(d2):
        for j in range(d2):
            val = hs_inner(basis[i], basis[j])
            if i == j:
                assert np.allclose(val, 1.0)
            else:
                assert np.allclose(val, 0.0, atol=1e-12)


def test_orthonormalize_gram_schmidt():
    from fmonz.utils.basis import orthonormalize
    # take some random matrices and orthonormalize them
    rng = np.random.default_rng(0)
    mats = [rng.standard_normal((3, 3)) + 1j * rng.standard_normal((3, 3))
            for _ in range(5)]
    out = orthonormalize(mats)
    for v in out:
        assert np.allclose(hs_inner(v, v), 1.0)
    # mutually orthogonal
    for i in range(len(out)):
        for j in range(i+1, len(out)):
            assert np.allclose(hs_inner(out[i], out[j]), 0.0)


def test_operator_basis_matrix_units():
    from fmonz.utils.basis import operator_basis
    basis = operator_basis(3, kind="matrix")
    assert len(basis) == 9
    check_orthonormal(basis)


def test_operator_basis_gellmann():
    from fmonz.utils.basis import operator_basis
    basis = operator_basis(4, kind="gellmann")
    assert len(basis) == 16
    check_orthonormal(basis)
