"""
Builders for derived signals.

Each builder returns a callable that takes the raw sample mapping and
returns a float, for use as a ``Signal`` source:

    Signal("T_avg", mean(["T1", "T2", "T3"]))
    Signal("offset", difference(mean(inner), mean(outer)))

Builders are small callable objects rather than lambdas, so they have a
readable ``repr`` and can be pickled (for example, to multiprocessing
workers). Operands of ``difference`` may be column names or other sources.
"""

from typing import Callable, Mapping, Sequence

import numpy as np

from . import registry
from .core import Source


def _read(source: Source, sample: Mapping[str, float]) -> float:
    return float(sample[source]) if isinstance(source, str) else float(source(sample))


class _Mean:
    def __init__(self, cols: Sequence[str]):
        self.cols = tuple(cols)

    def __call__(self, sample: Mapping[str, float]) -> float:
        return float(np.mean([sample[c] for c in self.cols]))

    def __repr__(self) -> str:
        return f"mean({list(self.cols)})"


class _Spread:
    def __init__(self, cols: Sequence[str]):
        self.cols = tuple(cols)

    def __call__(self, sample: Mapping[str, float]) -> float:
        return float(max(sample[c] for c in self.cols) - min(sample[c] for c in self.cols))

    def __repr__(self) -> str:
        return f"spread({list(self.cols)})"


class _Difference:
    def __init__(self, a: Source, b: Source):
        self.a, self.b = a, b

    def __call__(self, sample: Mapping[str, float]) -> float:
        return _read(self.a, sample) - _read(self.b, sample)

    def __repr__(self) -> str:
        return f"difference({self.a!r}, {self.b!r})"


@registry.signal_builders.register("column")
def column(col: str) -> str:
    """A raw column, possibly under another signal name."""
    return col


@registry.signal_builders.register("mean")
def mean(cols: Sequence[str]) -> Callable[[Mapping[str, float]], float]:
    """Arithmetic mean of several columns."""
    if not cols:
        raise ValueError("mean needs at least one column")
    return _Mean(cols)


@registry.signal_builders.register("spread")
def spread(cols: Sequence[str]) -> Callable[[Mapping[str, float]], float]:
    """Maximum minus minimum over several columns."""
    if not cols:
        raise ValueError("spread needs at least one column")
    return _Spread(cols)


@registry.signal_builders.register("difference")
def difference(a: Source, b: Source) -> Callable[[Mapping[str, float]], float]:
    """``a - b``, where each operand is a column name or another source."""
    return _Difference(a, b)


__all__ = ["column", "mean", "spread", "difference"]
