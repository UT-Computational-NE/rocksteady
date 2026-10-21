"""
Registries of named providers.

Every concrete component (criterion, disturbance, override, signal
builder, slope-uncertainty method, fit model) is registered under a short name, so
that it can be created by name (``rocksteady.factory``) and described in a
plain-data detector spec. The built-in providers register themselves when
``rocksteady`` is imported. Register your own the same way::

    from rocksteady import registry, Criterion, CriterionResult

    @registry.criteria.register("range")
    class RangeCriterion(Criterion):
        ...

after which ``{"type": "range", ...}`` works in a spec. A name can only be
registered once; ``unregister`` removes it.

Function providers follow the protocols below (``SlopeMethod``,
``FitModel``, ``SignalBuilder``). Registration checks that a slope method
or fit model accepts ``(t, y)`` positionally, and ``check_options`` checks
option names (and values, through an optional ``validate_options``
attribute) before any data is processed.
"""

import inspect
from typing import Any, Callable, Dict, Iterator, List, Mapping, Optional, Protocol, Tuple

import numpy as np

from .core import Criterion, Disturbance, Override, Source


class SlopeMethod(Protocol):
    """
    ``(t, y, **options) -> (b, u_b, k)``: the OLS slope of block means ``y``
    at block times ``t``, its standard uncertainty, and the coverage factor
    the method supplies. Optional attribute ``validate_options(**options)``
    raises ``ValueError`` for invalid values.
    """

    def __call__(self, t: np.ndarray, y: np.ndarray, **options: Any) -> Tuple[float, float, float]:
        ...


class FitModel(Protocol):
    """
    ``(t, y, **options) -> fit.FitResult`` over one segment (NaN samples
    ignored). Optional attributes: ``validate_options(**options)``, and
    ``evaluate(evaluation, t)`` to evaluate ``FitResult.evaluation``.
    """

    def __call__(self, t: np.ndarray, y: np.ndarray, **options: Any) -> Any:
        ...


class SignalBuilder(Protocol):
    """``(**params) -> Source``: a column name or a callable ``sample -> float``."""

    def __call__(self, **params: Any) -> Source:
        ...


def check_options(provider: Callable, options: Mapping[str, Any], what: str, positional: int = 2) -> None:
    """
    Check ``options`` for a function provider before any data: their names
    must bind to its signature after ``positional`` data arguments, and a
    ``validate_options`` attribute, if present, checks their values.
    Raises ``ValueError`` starting with ``what``.
    """
    try:
        sig = inspect.signature(provider)
    except (TypeError, ValueError):  # not introspectable (a builtin): nothing to check
        sig = None
    if sig is not None:
        try:
            sig.bind(*([None] * positional), **options)
        except TypeError as exc:
            raise ValueError(f"{what}: invalid options {sorted(options)} ({exc})") from None
    check = getattr(provider, "validate_options", None)
    if check is not None:
        try:
            check(**options)
        except ValueError as exc:
            raise ValueError(f"{what}: {exc}") from None


class Registry:
    """
    A name -> provider mapping for one kind of component.

    Parameters
    ----------
    kind : str
        What the providers are, for messages ("criterion", ...).
    base : type, optional
        If given, providers must be subclasses of it. Otherwise they must
        be callable.
    """

    def __init__(self, kind: str, base: Optional[type] = None, positional: int = 0):
        self.kind, self.base, self.positional = kind, base, positional
        self._items: Dict[str, Any] = {}

    def register(self, name: str, provider: Any = None):
        """Register ``provider`` under ``name``. Without ``provider``, return a decorator."""

        def add(obj):
            if self.base is not None:
                if not (isinstance(obj, type) and issubclass(obj, self.base)):
                    raise TypeError(f"a {self.kind} provider must be a subclass of {self.base.__name__}, got {obj!r}")
            elif not callable(obj):
                raise TypeError(f"a {self.kind} provider must be callable, got {obj!r}")
            elif self.positional:
                try:
                    # Only the data positions: required options come later, from the spec, and
                    # check_options binds them in full once they are known.
                    inspect.signature(obj).bind_partial(*([None] * self.positional))
                except TypeError:
                    raise TypeError(f"a {self.kind} provider must accept (t, y) positionally, got "
                                    f"{getattr(obj, '__name__', obj)!r}") from None
                except ValueError:
                    pass  # no signature to inspect
            if name in self._items:
                raise ValueError(f"{self.kind} {name!r} is already registered (to {self._items[name]!r})")
            self._items[name] = obj
            return obj

        return add if provider is None else add(provider)

    def unregister(self, name: str) -> None:
        self.get(name)
        del self._items[name]

    def get(self, name: str) -> Any:
        try:
            return self._items[name]
        except KeyError:
            raise KeyError(f"unknown {self.kind} {name!r}; registered: {', '.join(self.names()) or 'none'}") from None

    def create(self, name: str, /, **params) -> Any:
        """Call the provider registered under ``name`` with ``params``."""
        return self.get(name)(**params)

    def names(self) -> List[str]:
        return sorted(self._items)

    def __contains__(self, name: object) -> bool:
        return name in self._items

    def __iter__(self) -> Iterator[str]:
        return iter(self.names())

    def __repr__(self) -> str:
        return f"Registry({self.kind!r}: {self.names()})"


criteria = Registry("criterion", Criterion)
disturbances = Registry("disturbance", Disturbance)
overrides = Registry("override", Override)
signal_builders = Registry("signal builder")
#: Slope-uncertainty methods (``SlopeMethod``): ``(t, y, **options) -> (b, u_b, k)`` (see ``stats``).
slope_methods = Registry("slope method", positional=2)
#: Segment fit models (``FitModel``): ``(t, y, **options) -> FitResult`` (see ``fit``).
fit_models = Registry("fit model", positional=2)

__all__ = ["Registry", "criteria", "disturbances", "overrides", "signal_builders", "slope_methods", "fit_models",
           "SlopeMethod", "FitModel", "SignalBuilder", "check_options"]
