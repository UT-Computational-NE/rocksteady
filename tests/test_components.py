"""Tests of the signal builders and the floor-latch override."""

import itertools
import pickle

import numpy as np

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


@pytest.mark.parametrize("bad", [np.nan, np.inf, -np.inf])
def test_spread_is_missing_when_any_channel_is_not_finite(bad):  # #16
    cols = ["a", "b", "c"]
    for pos in range(3):
        values = [1.0, 2.0, 3.0]
        values[pos] = bad
        for perm in itertools.permutations(range(3)):
            s = spread([cols[i] for i in perm])
            assert np.isnan(s(dict(zip(cols, values))))


def test_detector_treats_a_missing_channel_in_spread_as_missing():  # #16
    det = rs.Detector([rs.Signal("spread", spread(["a", "b", "c"]))], [rs.DeviationCriterion("spread", 1, 0)],
                      timing=rs.Timing(block_s=1, window_s=3))
    labels = [det.update(t, {"a": 1.0, "b": np.nan, "c": 2.0}).label for t in range(3)]
    assert labels == [Label.INDETERMINATE] * 3
