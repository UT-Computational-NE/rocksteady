"""
Shared argument checks, so constructors and plain-data specs (which are built
through the same constructors) reject invalid configurations the same way,
before any data are processed. Each message names the component and option.
"""

import math
from numbers import Integral, Real
from typing import Any, Optional


def finite(value: Any, what: str, *, positive: bool = False, nonnegative: bool = False) -> float:
    """``value`` as a finite float, optionally > 0 or >= 0; otherwise ValueError naming ``what``."""
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{what} must be a number, got {value!r}")
    v = float(value)
    if not math.isfinite(v):
        raise ValueError(f"{what} must be finite, got {value!r}")
    if positive and not v > 0:
        raise ValueError(f"{what} must be > 0, got {value!r}")
    if nonnegative and not v >= 0:
        raise ValueError(f"{what} must be >= 0, got {value!r}")
    return v


def optional_finite(value: Any, what: str, **kw) -> Optional[float]:
    return None if value is None else finite(value, what, **kw)


def positive_int(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{what} must be a positive integer, got {value!r}")
    return int(value)


def name(value: Any, what: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{what} must be a non-empty string, got {value!r}")
    return value


def ordered(lo: Optional[float], hi: Optional[float], what_lo: str, what_hi: str, *, strict: bool = False) -> None:
    """``lo <= hi`` (``lo < hi`` with ``strict``) when both are given."""
    if lo is None or hi is None:
        return
    if (lo >= hi) if strict else (lo > hi):
        raise ValueError(f"{what_lo} ({lo!r}) must be {'<' if strict else '<='} {what_hi} ({hi!r})")
