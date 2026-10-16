"""
Factories: build components and whole detectors from registered names.

    rs.criterion("drift", signal="T1", limit_per_s=1.7e-4)
    rs.factory.signal("T_avg", builder="mean", cols=["T1", "T2"])
    rs.build_detector(spec)

A detector spec is a plain mapping (JSON, YAML or TOML compatible):

    {
      "timing": {"sample_period_s": 1.0, "block_s": 30.0, "window_s": 1200.0},
      "signals": [
        "T1",                                               # a raw column
        {"name": "T_avg", "builder": "mean", "cols": ["T1", "T2"]},
        {"name": "offset", "builder": "difference",
         "a": {"builder": "mean", "cols": ["T1", "T2"]}, "b": "T3"},
      ],
      "criteria": [{"type": "drift", "signal": "T_avg", "limit_per_s": 1.7e-4}],
      "disturbances": [{"type": "event_log", "events": [[600.0, 0.0]]}],
      "overrides": [],
    }

Any list entry may instead be an already-built object (a ``Signal``,
``Criterion``, ``Disturbance`` or ``Override``), and ``timing`` may be a
``Timing``, so specs can carry things plain data cannot, such as a
level-dependent noise callable. Built objects in a spec are used as they
are, not copied: give each detector its own (see ``core.Disturbance``).
Plain-data entries build new instances on every call, so
``build_detector(spec)`` called twice gives two independent detectors.
"""

from typing import Any, Mapping

from . import registry
from .core import Criterion, Detector, Disturbance, Override, Signal, Source, Timing

SPEC_KEYS = ("timing", "signals", "criteria", "disturbances", "overrides")


def criterion(kind: str, /, **params) -> Criterion:
    """Create the criterion registered as ``kind``."""
    return registry.criteria.create(kind, **params)


def disturbance(kind: str, /, **params) -> Disturbance:
    """Create the disturbance registered as ``kind``."""
    return registry.disturbances.create(kind, **params)


def override(kind: str, /, **params) -> Override:
    """Create the override registered as ``kind``."""
    return registry.overrides.create(kind, **params)


def source(spec: Any) -> Source:
    """
    Resolve a signal source: a column name, a callable, or a mapping
    ``{"builder": name, **params}`` whose params may themselves be sources.
    """
    if isinstance(spec, str) or callable(spec):
        return spec
    if isinstance(spec, Mapping):
        params = dict(spec)
        try:
            builder = params.pop("builder")
        except KeyError:
            raise ValueError(f"a source mapping needs a 'builder' key: {spec!r}") from None
        params = {k: source(v) if isinstance(v, Mapping) else v for k, v in params.items()}
        return registry.signal_builders.create(builder, **params)
    raise TypeError(f"cannot make a signal source from {spec!r}")


def signal(name: str, builder: str = "column", **params) -> Signal:
    """
    Create a ``Signal``. With the default builder, ``col`` (default: the
    signal name) is the raw column read.
    """
    if builder == "column" and not params:
        params = {"col": name}
    return Signal(name, source({"builder": builder, **params}))


def _signal_from_spec(entry: Any) -> Signal:
    if isinstance(entry, Signal):
        return entry
    if isinstance(entry, str):
        return Signal(entry, entry)
    if isinstance(entry, Mapping):
        params = dict(entry)
        try:
            name = params.pop("name")
        except KeyError:
            raise ValueError(f"a signal spec needs a 'name': {entry!r}") from None
        return signal(name, **params)
    raise TypeError(f"cannot make a signal from {entry!r}")


def _component(entry: Any, base: type, reg: registry.Registry):
    if isinstance(entry, base):
        return entry
    if isinstance(entry, Mapping):
        params = dict(entry)
        try:
            kind = params.pop("type")
        except KeyError:
            raise ValueError(f"a {reg.kind} spec needs a 'type' key: {entry!r}") from None
        try:
            return reg.create(kind, **params)
        except TypeError as exc:
            raise ValueError(f"bad parameters for {reg.kind} {kind!r}: {exc}") from None
    raise TypeError(f"cannot make a {reg.kind} from {entry!r}")


def build_detector(spec: Mapping[str, Any]) -> Detector:
    """Assemble a ``Detector`` from a spec (see the module docstring)."""
    unknown = set(spec) - set(SPEC_KEYS)
    if unknown:
        raise ValueError(f"unknown detector spec keys {sorted(unknown)}; expected some of {list(SPEC_KEYS)}")
    if not spec.get("signals"):
        raise ValueError("a detector spec needs at least one signal")
    timing = spec.get("timing", {})
    if not isinstance(timing, Timing):
        try:
            timing = Timing(**timing)
        except TypeError as exc:
            raise ValueError(f"bad timing spec: {exc}") from None
    return Detector(
        [_signal_from_spec(e) for e in spec["signals"]],
        [_component(e, Criterion, registry.criteria) for e in spec.get("criteria", ())],
        [_component(e, Disturbance, registry.disturbances) for e in spec.get("disturbances", ())],
        [_component(e, Override, registry.overrides) for e in spec.get("overrides", ())],
        timing,
    )


__all__ = ["criterion", "disturbance", "override", "source", "signal", "build_detector"]
