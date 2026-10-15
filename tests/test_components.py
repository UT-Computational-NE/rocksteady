"""Tests of the signal builders and the floor-latch override."""

import pickle

import pytest

import rocksteady as rs
from rocksteady import Label
from rocksteady.signals import difference, mean, spread


def test_signal_builders():
    s = {"a": 1.0, "b": 4.0, "c": 7.0}
    assert mean(["a", "b", "c"])(s) == pytest.approx(4.0)
    assert spread(["a", "b", "c"])(s) == pytest.approx(6.0)
    assert difference("c", "a")(s) == pytest.approx(6.0)
    assert difference(mean(["b", "c"]), "a")(s) == pytest.approx(4.5)
    assert repr(difference(mean(["b", "c"]), "a")) == "difference(mean(['b', 'c']), 'a')"
    with pytest.raises(ValueError):
        mean([])


def test_signal_builders_pickle():
    f = pickle.loads(pickle.dumps(difference(mean(["a", "b"]), spread(["a", "b"]))))
    assert f({"a": 1.0, "b": 3.0}) == pytest.approx(0.0)


def _ctx(v, onset=False):
    return rs.Context(t=0.0, sample={}, values={"p": v}, disturbed=onset, disturbance_onset=onset,
                      since_disturbance_s=0.0)


def test_floor_latch_latches_on_downward_crossing_and_releases():
    o = rs.FloorLatch("p", floor=1.0, release_factor=2.0)
    assert o.apply(_ctx(0.0)) is None  # starting below the floor is not a crossing
    assert o.apply(_ctx(0.0)) is None  # staying below is not a crossing either
    assert o.apply(_ctx(5.0)) is None
    assert o.apply(_ctx(0.0)) == Label.STEADY  # crossing
    assert o.apply(_ctx(1.5)) == Label.STEADY  # hysteresis band
    assert o.apply(_ctx(2.5)) is None  # released
    assert o.apply(_ctx(0.0)) == Label.STEADY
    assert o.apply(_ctx(0.0, onset=True)) is None  # a disturbance releases
    o.apply(_ctx(3.0))
    o.apply(_ctx(0.0))
    o.reset()
    assert o.apply(_ctx(0.0)) is None


def test_floor_latch_custom_label_and_name():
    o = rs.FloorLatch("p", floor=1.0, label=Label.INDETERMINATE, name="off")
    o.apply(_ctx(2.0))
    assert o.apply(_ctx(0.5)) == Label.INDETERMINATE and o.name == "off"
