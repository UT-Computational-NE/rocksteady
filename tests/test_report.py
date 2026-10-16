"""
Tests of fit and comparison reports and the text formatter.

The formatter is checked against tests/data/report_snapshot.txt. After an
intended layout change, regenerate it with ``python tests/test_report.py``.
"""

import os
import sys

import numpy as np
import pandas as pd
import pytest

import rocksteady as rs
from rocksteady.report import compare_report, fit_report

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from test_regression import actuator_case  # noqa: E402

SNAPSHOT = os.path.join(HERE, "data", "report_snapshot.txt")
LABELS = (["indeterminate"] * 100 + ["steady"] * 900 + ["settling"] * 700 + ["steady"] * 900
          + ["indeterminate"] * 50 + ["steady"] * 350)
MODELS = {"y": {"steady": 0, "transient": "first_order"}, "z": 1}


def record():
    """Plateau, first-order rise (tau = 100 s), plateau, a gap, plateau; plus a ramp z."""
    rng = np.random.default_rng(0)
    n = len(LABELS)
    t = np.arange(n, dtype=float)
    y = np.where(t < 1000, 10.0, 20 + (10 - 20) * np.exp(-(t - 1000) / 100)) + rng.normal(0, 0.05, n)
    y[2600:2650] = np.nan
    z = 0.001 * t + rng.normal(0, 0.01, n)
    return pd.DataFrame({"y": y, "z": z}, index=pd.date_range("2026-01-12 06:00", periods=n, freq="1s"))


def build():
    vals = record()
    segs = rs.segments(LABELS)
    fits = fit_report(vals, segs, MODELS, units={"y": "K"})
    ref = vals["y"] - 0.5 + np.random.default_rng(1).normal(0, 0.02, len(vals))
    comp = compare_report(vals["y"], ref, segs, names=("y", "y_ref"), units="K")
    return vals, segs, fits, comp


def test_fit_report_states_intervals_and_parameters():
    vals, segs, rep, _ = build()
    assert [b.name for b in rep] == ["y", "z"]
    for block in rep:
        assert list(block.by_state.index) == ["all", "steady", "transient", "indeterminate"]
        n = block.by_state["n"]
        assert n["all"] == n["steady"] + n["transient"] + n["indeterminate"]
    assert rep["y"].by_state.loc["all", "n"] == len(vals) - 50  # the NaN gap is excluded
    rise = rep["y"].fit_results[2]
    assert rise.model == "first_order" and abs(rise.params["tau"] - 100) <= 3 * rise.u["tau"]
    assert rep["y"].by_interval.loc[2, "param_tau"] == rise.params["tau"]
    assert rep["y"].by_interval["n"].tolist() == [100, 900, 700, 900, 0, 350]
    assert set(rep.by_state.index.get_level_values("signal")) == {"y", "z"}
    assert set(rep.by_interval["signal"]) == {"y", "z"} and set(rep.fits) == {"y", "z"}
    with pytest.raises(KeyError):
        rep["nope"]


def test_order_zero_on_one_plateau_has_no_correlation():
    vals = pd.DataFrame({"y": 5.0 + np.random.default_rng(2).normal(0, 0.1, 400)})
    rep = fit_report(vals, rs.segments(["steady"] * 400), {"y": 0}, dt=1.0)
    assert np.isnan(rep["y"].by_state.loc["steady", "r"]) and "—" in str(rep)


def test_compare_report_offset():
    a = pd.Series(np.sin(np.linspace(0, 20, 1000)) * 3)
    segs = rs.segments(["steady"] * 600 + ["settling"] * 400)
    rep = compare_report(a, a - 0.25, segs, names=("model", "measured"), dt=2.0)
    row = rep["model"].by_state.loc["all"]
    assert row["bias"] == pytest.approx(0.25) and row["r"] == pytest.approx(1.0) and row["duration_s"] == 2000.0
    assert "model vs measured" in rep.to_string()
    assert rep["model"].by_interval["state"].tolist() == ["steady", "transient"]


def test_bad_inputs():
    vals = record()
    with pytest.raises(KeyError, match="no column 'w'"):
        fit_report(vals, rs.segments(LABELS), {"w": 0})
    with pytest.raises(ValueError, match="model for 'y'"):
        fit_report(vals, rs.segments(LABELS), {"y": {"steady": 0}})


def test_batch_fit_report_runs_detector():
    det, df = actuator_case()
    rep = rs.batch.fit_report(det, df, {"level": {"steady": 0, "transient": "first_order"}, "act": 1})
    segs = rs.segments_from(det, rs.batch.run(det, df))
    assert len(rep["level"].by_interval) == len(segs)
    assert rep["level"].by_interval["state"].tolist() == [s.state.value for s in segs]
    assert "level   fit vs data" in str(rep)


def test_batch_fit_report_with_repeated_index_labels():
    det = rs.Detector([rs.Signal("x", "x")], [rs.DeviationCriterion("x", limit=100, noise=0)],
                      timing=rs.Timing(block_s=1, window_s=3))
    df = pd.DataFrame({"x": np.arange(6.0), "raw": np.arange(6.0) * 2}, index=[0, 0, 1, 1, 2, 2])
    rep = rs.batch.fit_report(det, df, {"x": 1, "raw": 1})
    assert rep["raw"].by_state.loc["all", "n"] == 6
    np.testing.assert_allclose(rep.fits["raw"].fitted, np.arange(6.0) * 2, atol=1e-9)


def test_batch_fit_report_reuses_saved_times():  # #20
    det = rs.Detector([rs.Signal("x", "x")], [rs.DeviationCriterion("x", 100, 0)],
                      timing=rs.Timing(block_s=1, window_s=3))
    df = pd.DataFrame({"x": 2 * np.arange(10.0)})
    t = 1.25 * np.arange(10.0)
    out = rs.batch.run(det, df, t=t)
    reused = rs.batch.fit_report(det, df, {"x": 1}, labels=out)
    explicit = rs.batch.fit_report(det, df, {"x": 1}, labels=out, t=t)
    for rep in (reused, explicit):
        assert rep.fits["x"].results[-1].params["slope"] == pytest.approx(1.6)
        assert rep["x"].by_interval["t_end_s"].iloc[-1] == pytest.approx(11.25)
    pd.testing.assert_frame_equal(reused["x"].by_interval, explicit["x"].by_interval)
    with pytest.raises(ValueError, match="conflicts with the t_s"):
        rs.batch.fit_report(det, df, {"x": 1}, labels=out, t=np.arange(10.0))
    with pytest.raises(ValueError, match="10 rows"):
        rs.batch.fit_report(det, df, {"x": 1}, labels=out.iloc[:5])
    with pytest.raises(ValueError, match="same index"):
        rs.batch.fit_report(det, df, {"x": 1}, labels=out.set_index(out.index + 100))
    # A DatetimeIndex keeps its clock times when the saved times agree with it.
    dfi = df.set_index(pd.date_range("2026-01-12 06:00", periods=10, freq="1s"))
    rep = rs.batch.fit_report(det, dfi, {"x": 1}, labels=rs.batch.run(det, dfi))
    assert "06:00:" in str(rep)


@pytest.mark.parametrize("index", [None, "datetime"])
def test_empty_records_keep_their_schema(index):  # #21
    det = rs.Detector([rs.Signal("x", "x")], [rs.DeviationCriterion("x", 1, 0)],
                      timing=rs.Timing(block_s=1, window_s=3))
    empty = pd.DataFrame({"x": pd.Series(dtype=float)})
    if index:
        empty.index = pd.DatetimeIndex([])
    full = rs.batch.run(det, pd.DataFrame({"x": np.zeros(5)}), with_results=True)
    for with_results in (False, True):
        out = rs.batch.run(det, empty, with_results=with_results)
        expected = ["label", "eta", "limiting", "since_disturbance_s", "forced_by"]
        expected += ["ratio_x_constant"] if with_results else []
        assert list(out.columns) == expected + ["t_s"] and len(out) == 0
    assert list(full.columns) == list(rs.batch.run(det, empty, with_results=True).columns)
    assert rs.segments_from(det, rs.batch.run(det, empty)) == []
    rep = rs.batch.fit_report(det, empty, {"x": 0})
    assert rep["x"].by_state.loc["all", "n"] == 0 and len(rep["x"].by_interval) == 0
    assert "mean" in rep["x"].by_interval.columns and "x   fit vs data" in str(rep)
    comp = compare_report(empty["x"], empty["x"], [], names=("a", "b"), dt=1.0)
    assert {"bias", "t_start_s", "t_end_s"} <= set(comp["a"].by_interval.columns)


def test_formatter_snapshot():
    _, _, fits, comp = build()
    text = fits.to_string() + "\n\n" + comp.to_string() + "\n"
    with open(SNAPSHOT, encoding="utf-8") as f:
        assert text == f.read()


if __name__ == "__main__":
    _, _, fits, comp = build()
    with open(SNAPSHOT, "w", encoding="utf-8") as f:
        f.write(fits.to_string() + "\n\n" + comp.to_string() + "\n")
    print(open(SNAPSHOT, encoding="utf-8").read())
