"""Tests of the provider registries, the factories and detector specs."""

import json
import os
import subprocess
import sys

import numpy as np
import pytest

import rocksteady as rs
from rocksteady import registry
from rocksteady.signals import difference, mean, spread

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from test_regression import drift_case  # noqa: E402


class _Range(rs.Criterion):
    def __init__(self, signal, limit, name=None):
        self.signal, self.limit, self.name = signal, limit, name or f"{signal}_range"

    def evaluate(self, t_blocks, y_blocks):
        return rs.CriterionResult(self.name, float(np.ptp(y_blocks)), 0.0, self.limit, 1.0)


@pytest.fixture
def range_registered():
    registry.criteria.register("range", _Range)
    yield
    registry.criteria.unregister("range")


def test_builtins_are_registered():
    assert {"deviation", "drift", "bound"} <= set(registry.criteria.names())
    assert {"step", "event_log"} <= set(registry.disturbances)
    assert "floor_latch" in registry.overrides
    assert {"column", "mean", "spread", "difference"} <= set(registry.signal_builders)


def test_user_provider_by_name(range_registered):
    c = rs.criterion("range", signal="x", limit=1.0)
    assert isinstance(c, _Range) and c.name == "x_range"
    det = rs.build_detector({"timing": {"window_s": 120.0, "block_s": 10.0}, "signals": ["x"],
                             "criteria": [{"type": "range", "signal": "x", "limit": 1.0}]})
    labels = [det.update(float(i), {"x": 5.0}).label for i in range(130)]
    assert labels[-1] == rs.Label.STEADY


def test_registry_errors(range_registered):
    with pytest.raises(ValueError, match="already registered"):
        registry.criteria.register("range", _Range)
    with pytest.raises(KeyError, match="registered: bound, deviation, drift, range"):
        rs.criterion("nope")
    with pytest.raises(TypeError, match="subclass of Criterion"):
        registry.criteria.register("bad", object)
    with pytest.raises(TypeError, match="callable"):
        registry.signal_builders.register("bad", 3)
    assert "bad" not in registry.criteria


def _hand_built(timing):
    signals = [rs.Signal("a", "a"), rs.Signal("b", "b"), rs.Signal("m", mean(["a", "b"])),
               rs.Signal("s", spread(["a", "b"])), rs.Signal("d", difference(mean(["a", "b"]), "b"))]
    criteria = [rs.DriftCriterion(s.name, 1e-3, name=s.name) for s in signals]
    return rs.Detector(signals, criteria, [rs.EventLog([(1500.0, 0.0)])], timing=timing)


SPEC = {
    "timing": {"sample_period_s": 1.0, "block_s": 10.0, "window_s": 300.0, "settle_s": 60.0, "hold_s": 120.0},
    "signals": ["a", "b", {"name": "m", "builder": "mean", "cols": ["a", "b"]},
                {"name": "s", "builder": "spread", "cols": ["a", "b"]},
                {"name": "d", "builder": "difference", "a": {"builder": "mean", "cols": ["a", "b"]}, "b": "b"}],
    "criteria": [{"type": "drift", "signal": n, "limit_per_s": 1e-3, "name": n} for n in ["a", "b", "m", "s", "d"]],
    "disturbances": [{"type": "event_log", "events": [[1500.0, 0.0]]}],
}


def test_spec_matches_hand_built_detector():
    _, df = drift_case()
    spec = json.loads(json.dumps(SPEC))  # specs survive a JSON round trip
    a = rs.batch.run(rs.build_detector(spec), df)
    b = rs.batch.run(_hand_built(rs.Timing(**SPEC["timing"])), df)
    assert (a["label"] == b["label"]).all()
    np.testing.assert_array_equal(a["eta"].to_numpy(), b["eta"].to_numpy())


def test_spec_may_mix_objects_and_data():
    spec = dict(SPEC, timing=rs.Timing(**SPEC["timing"]),
                signals=[rs.Signal("a", "a"), *SPEC["signals"][1:]],
                criteria=[rs.DriftCriterion("a", 1e-3, name="a"), *SPEC["criteria"][1:]],
                overrides=[rs.FloorLatch("a", 1.0), {"type": "floor_latch", "signal": "b", "floor": 1.0,
                                                      "label": "indeterminate", "name": "b_floor"}])
    det = rs.build_detector(spec)
    assert [type(o).__name__ for o in det.overrides] == ["FloorLatch", "FloorLatch"]
    assert det.overrides[1].label == rs.Label.INDETERMINATE


@pytest.mark.parametrize("spec, match", [
    ({"signals": ["x"], "criterion": []}, "unknown detector spec keys"),
    ({"criteria": []}, "at least one signal"),
    ({"signals": ["x"], "criteria": [{"signal": "x"}]}, "needs a 'type'"),
    ({"signals": ["x"], "criteria": [{"type": "drift", "signal": "x", "limit": 1.0}]}, "bad parameters"),
    ({"signals": ["x"], "criteria": [{"type": "drift", "signal": "y", "limit_per_s": 1.0}]}, "unknown signal"),
    ({"signals": ["x"], "timing": {"window": 10}}, "bad timing"),
    ({"signals": [{"builder": "mean", "cols": ["x"]}]}, "needs a 'name'"),
])
def test_spec_errors(spec, match):
    with pytest.raises(ValueError, match=match):
        rs.build_detector(spec)


def test_streaming_detector_needs_no_pandas():
    code = (
        "import sys; sys.modules['pandas'] = None\n"
        "import rocksteady as rs\n"
        "det = rs.build_detector({'timing': {'window_s': 60.0, 'block_s': 10.0}, 'signals': ['x'],"
        " 'criteria': [{'type': 'drift', 'signal': 'x', 'limit_per_s': 1.0}]})\n"
        "print([det.update(float(i), {'x': 1.0 + 1e-3 * (i % 3)}).label.value for i in range(70)][-1])\n"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True)
    assert out.returncode == 0, out.stderr
    assert out.stdout.strip() == "steady"
