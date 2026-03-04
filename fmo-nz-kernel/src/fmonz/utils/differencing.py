"""Finite differencing routines."""


def finite_diff(f, x, h=1e-6):
    return (f(x + h) - f(x - h)) / (2 * h)
