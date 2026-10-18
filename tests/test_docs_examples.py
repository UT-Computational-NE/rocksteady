"""The documentation's example files run as published (#49)."""

import os
import subprocess
import sys

import pytest

EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "source", "examples")


@pytest.mark.parametrize("name", ["adapter_basic.py", "adapter_custom.py"])
def test_docs_example_runs(name):
    # A fresh interpreter each: the custom example registers components by name, once per process.
    proc = subprocess.run([sys.executable, os.path.join(EXAMPLES, name)], capture_output=True, text=True,
                          timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]


def test_documented_override_and_quiet_clock_behaviour():
    """Two statements of the docs (adapter contract, methods state machine), checked against the code."""
    import numpy as np

    import rocksteady as rs

    seen = []

    class Recorder(rs.Override):
        name = "recorder"

        def apply(self, ctx):
            seen.append(ctx.t)
            return None

    det = rs.Detector([rs.Signal("x", "x")], [rs.DeviationCriterion("x", 1, 0)], (), [Recorder()],
                      rs.Timing(block_s=1, window_s=3, max_gap_s=2))
    reasons = [det.update(float(t), {"x": x}).reason.value for t, x in [(0, 0.0), (1, np.nan), (2, 0.0)]]
    assert seen == [0.0, 2.0] and reasons[1] == "missing"  # overrides skip missing samples
    tm = rs.Timing(sample_period_s=1, block_s=1, window_s=4.4, settle_s=0.2)
    assert tm.window_samples == 4 and tm.quiet_samples == round((4.4 + 0.2) / 1) == 5  # unrounded window
