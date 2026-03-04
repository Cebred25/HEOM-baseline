import numpy as np


def test_vec_unvec_roundtrip():
    from fmonz.utils.vec import vec, unvec
    arr = np.arange(9).reshape(3, 3)
    v = vec(arr)
    assert v.shape == (9,)
    arr2 = unvec(v, 3)
    assert np.allclose(arr, arr2)


def test_flatten_alias():
    from fmonz.utils.vec import flatten, vec
    arr = np.eye(2)
    assert np.all(flatten(arr) == vec(arr))
