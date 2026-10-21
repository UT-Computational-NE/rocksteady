"""
AR(1) correlation primitives shared by the drift test (``stats``) and the
fit covariances (``fit``). numpy only, with no other rocksteady imports, so
any module can use them.

``R`` is the AR(1) correlation matrix ``R_ij = phi^|i-j|`` of ``n``
consecutive samples. It is never formed: ``R @ x`` costs O(n) time and
memory instead of O(n^2).

With ``steps`` (the spacing of consecutive samples, in sample spacings), the
samples need not be consecutive: ``R_ij = phi^|s_i - s_j|`` for sample
positions ``s``, the correlation of an AR(1) (Ornstein-Uhlenbeck) process
observed with some samples missing.
"""

from typing import Optional, Sequence

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


def _scan_steps(col: list, decay: list) -> list:
    """As ``_scan``, with ``decay[i]`` the correlation between samples ``i`` and ``i + 1``."""
    n = len(col)
    f = [0.0] * n
    acc = 0.0
    for i, v in enumerate(col):
        acc = v + (decay[i - 1] * acc if i else 0.0)
        f[i] = acc
    acc = 0.0
    for i in range(n - 1, -1, -1):
        acc = col[i] + (decay[i] * acc if i < n - 1 else 0.0)
        f[i] = f[i] + acc - col[i]
    return f


def gap_steps(t: np.ndarray, dt: float) -> Optional[np.ndarray]:
    """
    Spacing of consecutive times ``t`` in whole sample spacings ``dt`` (at
    least 1), or None when every step is one period (consecutive samples).
    """
    if len(t) < 2 or not dt > 0:
        return None
    steps = np.maximum(np.rint(np.diff(np.asarray(t, dtype=float)) / dt), 1.0)
    return None if np.all(steps == 1.0) else steps


def ar1_apply(x: np.ndarray, phi: float, steps: Optional[Sequence[float]] = None) -> np.ndarray:
    """
    ``R @ x`` for the AR(1) correlation matrix ``R_ij = phi^|i-j|``, in O(n).

    ``x`` may be a vector or an (n, m) matrix; columns are treated
    independently. Uses ``R x = f + b - x`` with the forward and backward
    recursions ``f_i = x_i + phi f_{i-1}`` and ``b_i = x_i + phi b_{i+1}``.
    The recursions run on Python floats, which is several times faster than
    indexing numpy rows for the short windows of the streaming detector.

    ``steps`` (length n - 1): the spacing of consecutive samples in sample
    periods. Then ``R_ij = phi^(s_j - s_i)`` for positions ``s``, and the
    recursions decay by ``phi^step`` across each step.
    """
    x = np.asarray(x, dtype=float)
    if phi == 0.0 or len(x) == 0:
        return x.copy()
    phi = float(phi)
    if steps is None:
        def scan(c):
            return _scan(c, phi)
    else:
        steps = np.asarray(steps, dtype=float)
        if steps.shape != (len(x) - 1,):
            raise ValueError(f"steps must have length {len(x) - 1}, got shape {steps.shape}")
        decay = (phi ** steps).tolist()

        def scan(c):
            return _scan_steps(c, decay)
    if x.ndim == 1:
        return np.array(scan(x.tolist()))
    cols = x.reshape(len(x), -1).T.tolist()
    return np.array([scan(c) for c in cols]).T.reshape(x.shape)


def ar1_quad(x: np.ndarray, phi: float, steps: Optional[Sequence[float]] = None) -> np.ndarray:
    """``x' R x`` for the columns of ``x`` (an (n, m) matrix): an (m, m) matrix, in O(n m^2)."""
    x = np.asarray(x, dtype=float)
    return x.T @ ar1_apply(x, phi, steps)


__all__ = ["ar1_apply", "ar1_quad", "gap_steps"]
