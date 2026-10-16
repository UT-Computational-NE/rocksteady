"""Provider protocols, option checks before data, ownership docs and entry points (#27)."""

import subprocess
import sys

import numpy as np
import pytest

import rocksteady as rs
from rocksteady import registry


@pytest.mark.parametrize("method, options, match", [
    ("ar1", {"confidence": 1.2}, r"slope method 'ar1': confidence must be in \(0.5, 1\)"),
    ("ar1", {"phi_max": 1.0}, "phi_max must be in"),
    ("ar1", {"lags": 3}, r"slope method 'ar1': invalid options \['lags'\]"),
    ("nw", {"max_lag_frac": 2.0}, "max_lag_frac must be in"),
    ("nw", {"k": -1.0}, "k must be >= 0"),
])
def test_slope_method_options_are_checked_when_the_criterion_is_built(method, options, match):
    with pytest.raises(ValueError, match=match):
        rs.DriftCriterion("x", 1.0, method=method, method_options=options)
    spec = {"timing": {"block_s": 1, "window_s": 5}, "signals": ["x"],
            "criteria": [{"type": "drift", "signal": "x", "limit_per_s": 1.0, "method": method,
                          "method_options": options}]}
    with pytest.raises(ValueError, match=match):
        rs.build_detector(spec)


def test_custom_method_options_are_bound_to_its_signature():
    def slope(t, y, scale=1.0):
        return 0.0, 0.1 * scale, 2.0

    rs.DriftCriterion("x", 1.0, method=slope, method_options={"scale": 2.0})
    with pytest.raises(ValueError, match=r"slope method 'slope': invalid options \['scal'\]"):
        rs.DriftCriterion("x", 1.0, method=slope, method_options={"scal": 2.0})


def test_registration_checks_the_data_arguments():
    with pytest.raises(TypeError, match=r"must accept \(t, y\) positionally"):
        registry.fit_models.register("bad_model", lambda y: None)
    with pytest.raises(TypeError, match=r"must accept \(t, y\) positionally"):
        registry.slope_methods.register("bad_slope", lambda *, t, y: (0, 0, 0))
    assert "bad_model" not in registry.fit_models and "bad_slope" not in registry.slope_methods

    def model(t, y, **options):
        raise AssertionError("not called")

    registry.fit_models.register("kw_model", model)
    try:
        rs.fit.validate_models({"x": "kw_model"})  # **options accepts any option name
    finally:
        registry.fit_models.unregister("kw_model")


def test_check_options_names_the_provider():
    def fn(t, y, a=1):
        pass

    fn.validate_options = lambda a=1: (_ for _ in ()).throw(ValueError("a must be odd")) if a % 2 == 0 else None
    registry.check_options(fn, {"a": 3}, "thing")
    with pytest.raises(ValueError, match="^thing: a must be odd$"):
        registry.check_options(fn, {"a": 2}, "thing")
    with pytest.raises(ValueError, match=r"^thing: invalid options \['b'\]"):
        registry.check_options(fn, {"b": 2}, "thing")


def test_build_detector_gives_independent_components():
    spec = {"timing": {"block_s": 1, "window_s": 5}, "signals": ["x"],
            "criteria": [{"type": "deviation", "signal": "x", "limit": 1, "noise": 0}],
            "disturbances": [{"type": "step", "signal": "x", "threshold": 100.0, "m": 2}],
            "overrides": [{"type": "floor_latch", "signal": "x", "floor": 1.0}]}
    a, b = rs.build_detector(spec), rs.build_detector(spec)
    assert a.disturbances[0] is not b.disturbances[0] and a.overrides[0] is not b.overrides[0]
    for t, x in enumerate([5.0, 5.0, 5.0, 0.5]):
        a.update(float(t), {"x": x})
    assert a.overrides[0]._latched and not b.overrides[0]._latched


def test_documented_entry_points_exist():
    import rocksteady.factory as factory

    doc = factory.__doc__
    assert "rs.factory.signal(" in doc and "rs.signal(" not in doc.replace("rs.factory.signal(", "")
    assert rs.factory.signal("T_avg", builder="mean", cols=["T1", "T2"]).value({"T1": 1.0, "T2": 3.0}) == 2.0


def test_streaming_core_does_not_import_pandas():
    code = ("import sys, rocksteady as rs; "
            "d = rs.Detector([rs.Signal('x', 'x')], [rs.DriftCriterion('x', 1.0)], "
            "timing=rs.Timing(block_s=1, window_s=5)); "
            "[d.update(float(t), {'x': 0.0}) for t in range(10)]; "
            "print('pandas' in sys.modules)")
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout.strip()
    assert out == "False"


def test_providers_with_required_options_can_register():
    def model(t, y, *, gain):
        return rs.fit.fit_poly(t, np.asarray(y) * gain)

    def slope(t, y, *, scale):
        return 0.0, 0.1 * scale, 2.0

    registry.fit_models.register("required_gain", model)
    registry.slope_methods.register("required_scale", slope)
    try:
        r = rs.fit.fit_model({"model": "required_gain", "gain": 2}, np.arange(10.0), np.arange(10.0))
        assert r.params["level"] == pytest.approx(9.0)
        # The full bind still runs once the options are known: a missing required option fails early.
        with pytest.raises(ValueError, match=r"model for 'x' in state 'steady': required_gain: invalid options"):
            rs.fit.validate_models({"x": "required_gain"})
        rs.fit.validate_models({"x": {"model": "required_gain", "gain": 2}})
        crit = rs.DriftCriterion("x", 1.0, method="required_scale", method_options={"scale": 2.0})
        assert crit.evaluate(np.arange(5.0), np.zeros(5)).uncertainty == pytest.approx(0.2)
        with pytest.raises(ValueError, match="slope method 'required_scale': invalid options"):
            rs.DriftCriterion("x", 1.0, method="required_scale")
    finally:
        registry.fit_models.unregister("required_gain")
        registry.slope_methods.unregister("required_scale")
