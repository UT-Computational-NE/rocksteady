"""
The package's public frozen values rebuild from their own fields (#111): ``dataclasses.replace``, the constructor
given every field, and pickle each return an equal value, and equal values hash alike. A constructor that
normalises a field to a form it then refuses fails here.
"""

import dataclasses
import pickle

import numpy as np
import pytest

import rocksteady as rs


def _column_p(sample):
    """A signal source that pickles: a module-level function."""
    return sample["p"]


#: Every public frozen value, each with every optional field set to a value other than its default.
CASES = [
    rs.Signal("power", "p"),
    rs.Signal("power", _column_p),
    rs.Timing(sample_period_s=0.5, block_s=10.0, window_s=60.0, settle_s=30.0, hold_s=20.0, hold_when_undecided=True,
              undecided_counts_toward_hold=False, max_gap_s=2.0, min_block_coverage=0.8, grid_tolerance=0.25),
    rs.Subtype("transient", level=np.array([1.0, 5.0]), peak=(4.0, None), max_rise_s=30.0, rise_fraction=0.9,
               events={"valve": False, "heater": np.True_}),
    rs.Segment(3, 9, rs.State.STEADY, forced=True, subtype="hot"),
]


def _public_frozen_types():
    return {obj for obj in (getattr(rs, name) for name in rs.__all__)
            if isinstance(obj, type) and dataclasses.is_dataclass(obj) and obj.__dataclass_params__.frozen}


def test_every_public_frozen_value_has_a_case():
    assert {type(v) for v in CASES} == _public_frozen_types()


@pytest.mark.parametrize("value", CASES, ids=lambda v: type(v).__name__)
def test_a_frozen_value_rebuilds_from_its_own_fields(value):
    rebuilt = {
        "replace": dataclasses.replace(value),
        "constructor": type(value)(**{f.name: getattr(value, f.name) for f in dataclasses.fields(value) if f.init}),
        "pickle": pickle.loads(pickle.dumps(value)),
    }
    for how, other in rebuilt.items():
        assert other == value, how
        assert hash(other) == hash(value), how
