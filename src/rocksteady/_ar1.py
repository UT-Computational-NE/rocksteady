"""
AR(1) correlation primitives shared by the drift test (``stats``) and the
fit covariances (``fit``). numpy only, with no other rocksteady imports, so
any module can use them.

``R`` is the AR(1) correlation matrix ``R_ij = phi^|i-j|`` of ``n``
consecutive samples. It is never formed: ``R @ x`` costs O(n) time and
memory instead of O(n^2).
"""

import numpy as np


def _scan(col: list, phi: float) -> list:
    """``f + b - x`` for one column: forward ``f_i = x_i + phi f_(i-1)`` and backward ``b_i = x_i + phi b_(i+1)``."""
    n = len(col)
    f = [0.0] * n
    acc = 0.0
    for i, v in enumerate(col):
        acc = v + phi * acc
        f[i] = acc
    acc = 0.0
    for i in range(n - 1, -1, -1):
        acc = col[i] + phi * acc
        f[i] = f[i] + acc - col[i]  # (f + b) - x, in the order numpy evaluates it
    return f


def ar1_apply(x: np.ndarray, phi: float) -> np.ndarray:
    """
    ``R @ x`` for the AR(1) correlation matrix ``R_ij = phi^|i-j|``, in O(n).

    ``x`` may be a vector or an (n, m) matrix; columns are treated
    independently. Uses ``R x = f + b - x`` with the forward and backward
    recursions ``f_i = x_i + phi f_{i-1}`` and ``b_i = x_i + phi b_{i+1}``.
    The recursions run on Python floats, which is several times faster than
    indexing numpy rows for the short windows of the streaming detector.
    """
    x = np.asarray(x, dtype=float)
    if phi == 0.0 or len(x) == 0:
        return x.copy()
    phi = float(phi)
    if x.ndim == 1:
        return np.array(_scan(x.tolist(), phi))
    cols = x.reshape(len(x), -1).T.tolist()
    return np.array([_scan(c, phi) for c in cols]).T.reshape(x.shape)


def ar1_quad(x: np.ndarray, phi: float) -> np.ndarray:
    """``x' R x`` for the columns of ``x`` (an (n, m) matrix): an (m, m) matrix, in O(n m^2)."""
    x = np.asarray(x, dtype=float)
    return x.T @ ar1_apply(x, phi)


__all__ = ["ar1_apply", "ar1_quad"]
