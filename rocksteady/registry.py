"""
Registries of named providers.

Every concrete component (criterion, disturbance, override, signal
builder, slope-uncertainty method) is registered under a short name, so
that it can be created by name (``rocksteady.factory``) and described in a
plain-data detector spec. The built-in providers register themselves when
``rocksteady`` is imported. Register your own the same way:

    from rocksteady import registry, Criterion, CriterionResult

    @registry.criteria.register("range")
    class RangeCriterion(Criterion):
        ...

after which ``{"type": "range", ...}`` works in a spec. A name can only be
registered once; ``unregister`` removes it.
"""

from typing import Any, Dict, Iterator, List, Optional

from .core import Criterion, Disturbance, Override


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

    def __init__(self, kind: str, base: Optional[type] = None):
        self.kind, self.base = kind, base
        self._items: Dict[str, Any] = {}

    def register(self, name: str, provider: Any = None):
        """Register ``provider`` under ``name``. Without ``provider``, return a decorator."""

        def add(obj):
            if self.base is not None:
                if not (isinstance(obj, type) and issubclass(obj, self.base)):
                    raise TypeError(f"a {self.kind} provider must be a subclass of {self.base.__name__}, got {obj!r}")
            elif not callable(obj):
                raise TypeError(f"a {self.kind} provider must be callable, got {obj!r}")
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
#: Slope-uncertainty methods: callables ``(t, y) -> (b, u_b, k)`` (see ``stats``).
slope_methods = Registry("slope method")

__all__ = ["Registry", "criteria", "disturbances", "overrides", "signal_builders", "slope_methods"]
