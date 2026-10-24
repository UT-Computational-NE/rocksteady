"""
Reports: fits and comparisons over a record, by state and by segment.

- ``fit_report``: each signal against its piecewise fit (``e = fit - data``),
  per state, plus per-segment statistics and fitted parameters.
- ``compare_report``: one series against another (``e = a - b``), per state
  and per segment.
- ``batch.fit_report``: run a detector (or take its labels), split the
  record into states, and build a ``fit_report`` from its signal values.

A ``Report`` holds DataFrames (``by_state``, ``by_interval``) and prints as
plain-text tables::

    power [kW]   fit vs data   (models: steady poly 0, transient exp, indeterminate exp)
      state             n     time   bias    std    MAE   RMSE  max|e|  NRMSE       r  u(bias)  tau_int  N_eff
      all           8,640  24.00 h   ...

``Report.provenance`` records how the tables were made (software version,
time origin and sample spacing, the normalized model spec and parameter
order per signal and state, the uncertainty methods, NRMSE normalization,
and, from ``batch.fit_report``, the detector's effective timing and
validity policy). The DataFrames carry it in ``.attrs["rocksteady"]``, and
``Report.to_dict()`` / ``to_json()`` export it with the tables and fits.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .fit import FitResult, PiecewiseFit, normalize_model, piecewise_fit, resolve_model, validate_models
from .metrics import STATES, signal_stats, stats_by_segment, stats_by_state
from .states import Segment

FIT_INTERVAL_COLUMNS = ["segment", "state", "forced", "start", "stop", "t_start_s", "t_end_s", "model", "t_ref_s",
                        "n", "mean", "std", "tau_int_s", "n_eff", "u_mean", "rmse", "low_effective_sample", "flags"]
_HIDDEN_PARAMS = {"lambda"}  # derived from tau; shown in DataFrames, not in the text table
_TIME_PARAMS = {"tau", "T"}


# ---------------------------------------------------------------------------
# Report containers
# ---------------------------------------------------------------------------


@dataclass
class SignalReport:
    """One signal's block of a report."""

    name: str
    title: str
    by_state: Any  # DataFrame indexed by state
    by_interval: Any  # DataFrame, one row per segment
    units: Optional[str] = None
    fit: Optional[PiecewiseFit] = None
    fit_results: List[FitResult] = field(default_factory=list)
    time_origin: Any = None  # a Timestamp when the record had a DatetimeIndex

    def to_string(self, intervals: bool = True, time_origin=None) -> str:
        time_origin = self.time_origin if time_origin is None else time_origin
        head = f"{self.name}{f' [{self.units}]' if self.units else ''}   {self.title}"
        lines = [head, _indent(_state_table(self.by_state))]
        if any(self.by_state["low_effective_sample"] & (self.by_state["n"] > 0)):
            lines.append("  * N_eff < 10: too few effective samples for a normal coverage factor")
        if intervals and len(self.by_interval):
            lines += ["", f"{self.name}{f' [{self.units}]' if self.units else ''}   intervals",
                      _indent(_interval_table(self.by_interval, self.fit_results, time_origin))]
        return "\n".join(lines)


UNCERTAINTY_DESCRIPTION = {
    "mean_and_bias": "u = std / sqrt(N_eff), N_eff = n dt / tau_int; tau_int from the pair-weighted pooled ACF "
                     "(stats.acf_tau), within pieces split at segment edges, non-finite samples and time gaps "
                     "> 1.5 dt; fit residuals for signals, de-meaned errors for comparisons",
    "parameters": "sandwich covariance (fit.sandwich_cov) per model option 'uncertainty': 'ar1' (default; "
                  "bias-corrected lag-1 residual autocorrelation, consecutive valid samples one sample apart, "
                  "flag time_gaps when they are not), 'ar1_time' (AR(1) over sample positions) or 'iid'",
}


def _jsonable(x):
    """``json.dumps`` default: numpy scalars and arrays, timestamps, sets."""
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, (set, frozenset)):
        return sorted(x)
    if hasattr(x, "isoformat"):
        return x.isoformat()
    raise TypeError(f"not JSON serializable: {type(x).__name__}")


class Report:
    """A set of ``SignalReport`` blocks, printable as text and available as DataFrames."""

    def __init__(self, blocks: Sequence[SignalReport], time_origin=None, provenance: Optional[dict] = None):
        self.blocks = list(blocks)
        self.time_origin = time_origin
        self.provenance: Dict[str, Any] = dict(provenance or {})

    def __getitem__(self, name: str) -> SignalReport:
        for b in self.blocks:
            if b.name == name:
                return b
        raise KeyError(name)

    def __iter__(self):
        return iter(self.blocks)

    @property
    def by_state(self):
        """All blocks' state tables, indexed by (signal, state)."""
        import pandas as pd

        out = pd.concat({b.name: b.by_state for b in self.blocks}, names=["signal", "state"])
        out.attrs["rocksteady"] = self.provenance
        return out

    @property
    def by_interval(self):
        """All blocks' per-segment tables, with a ``signal`` column."""
        import pandas as pd

        out = pd.concat([b.by_interval.assign(signal=b.name) for b in self.blocks], ignore_index=True)
        out.attrs["rocksteady"] = self.provenance
        return out

    @property
    def fits(self) -> Dict[str, PiecewiseFit]:
        return {b.name: b.fit for b in self.blocks if b.fit is not None}

    def to_string(self, intervals: bool = True) -> str:
        return "\n\n".join(b.to_string(intervals, self.time_origin) for b in self.blocks)

    def to_dict(self) -> Dict[str, Any]:
        """Provenance, both tables (as records) and every segment's fit (``FitResult.to_dict``)."""
        by_state = self.by_state.reset_index()
        return {
            "provenance": self.provenance,
            "by_state": by_state.to_dict(orient="records"),
            "by_interval": self.by_interval.to_dict(orient="records"),
            "fits": {b.name: [r.to_dict() for r in b.fit_results] for b in self.blocks if b.fit_results},
        }

    def to_json(self, path=None, **kwargs) -> str:
        """``to_dict`` as JSON (NaN and Infinity kept as JavaScript literals); written to ``path`` if given."""
        text = json.dumps(self.to_dict(), default=_jsonable, **{"indent": 1, **kwargs})
        if path is not None:
            with open(path, "w") as fh:
                fh.write(text)
        return text

    def __str__(self) -> str:
        return self.to_string()

    def __repr__(self) -> str:
        return f"Report({[b.name for b in self.blocks]})"


# ---------------------------------------------------------------------------
# Building reports
# ---------------------------------------------------------------------------


def _times(values, t, dt):
    """Sample times in seconds, the sample spacing, and the time origin (a Timestamp or None)."""
    origin = None
    if t is None:
        index = getattr(values, "index", None)
        if index is not None and hasattr(index, "tz_localize") and len(index):  # a DatetimeIndex
            origin = index[0]
            t = (index - index[0]).total_seconds().to_numpy()
        else:
            n = len(next(iter(values.values()))) if isinstance(values, Mapping) else len(values)
            t = np.arange(n) * (1.0 if dt is None else dt)
    t = np.asarray(t, dtype=float)
    if dt is None:
        dt = float(np.median(np.diff(t))) if len(t) > 1 else 1.0
    return t, float(dt), origin


def _column(values, name) -> np.ndarray:
    try:
        return np.asarray(values[name], dtype=float)
    except KeyError:
        raise KeyError(f"no column {name!r} in values") from None


def _models_title(spec, states: Sequence[str]) -> str:
    parts = []
    for st in [s for s in states if s != "all"]:
        name, opts = resolve_model(spec, st)
        label = f"poly {opts.get('order', 0)}" if name == "poly" else name
        parts.append(f"{st} {label}")
    return f"(models: {', '.join(parts)})"


def fit_report(values, segs: Sequence[Segment], models: Mapping[str, Any], *, t=None, dt: Optional[float] = None,
               units: Optional[Mapping[str, str]] = None, states: Sequence[str] = STATES,
               nrmse: str = "mean") -> Report:
    """
    Each signal against its piecewise fit, by state and by segment.

    Parameters
    ----------
    values : DataFrame or mapping of arrays
        One column per signal named in ``models``.
    segs : list of Segment
        From ``rocksteady.segments`` or ``segments_from``.
    models : mapping
        ``{signal: model spec}``; see ``rocksteady.fit``. Example:
        ``{"power": {"steady": 0, "transient": "exp"}, "Tran": 1}``.
    t : array, optional
        Sample times in seconds. Default: from a ``DatetimeIndex``, else
        the row number times ``dt``.
    dt : float, optional
        Sample spacing in seconds. Default: the median spacing of ``t``.
    units : mapping, optional
        ``{signal: units}`` for the headers.
    states : sequence
        Rows of the state table.
    nrmse : "mean", "range", "std" or "none"
    """
    validate_models(models)
    t, dt, origin = _times(values, t, dt)
    units = dict(units or {})
    blocks = []
    for name, spec in models.items():
        y = _column(values, name)
        pw = piecewise_fit(t, y, segs, spec)
        by_state = stats_by_state(pw.fitted, y, segs, dt=dt, states=states, nrmse=nrmse, t=t)
        rows = []
        for i, (s, r) in enumerate(zip(segs, pw.results)):
            ss = signal_stats(y[s.slice], r, dt=dt, t=t[s.slice])
            e = r.residuals[np.isfinite(r.residuals)]
            rows.append({"segment": i, "state": s.state.value, "forced": s.forced, "start": s.start, "stop": s.stop,
                         "t_start_s": t[s.start], "t_end_s": t[s.stop - 1], "model": r.label, "t_ref_s": r.t_ref,
                         "n": ss.n, "mean": ss.mean, "std": ss.std, "tau_int_s": ss.tau_int_s,
                         "n_eff": ss.n_eff, "u_mean": ss.u_mean,
                         "rmse": float(np.sqrt(np.mean(e * e))) if len(e) else float("nan"),
                         "low_effective_sample": ss.low_effective_sample,
                         "flags": ",".join(sorted(r.flags)),
                         **{f"param_{k}": v for k, v in r.params.items()},
                         **{f"u_{k}": v for k, v in r.u.items()}})
        import pandas as pd

        blocks.append(SignalReport(name, f"fit vs data   {_models_title(spec, states)}", by_state,
                                   pd.DataFrame(rows, columns=None if rows else FIT_INTERVAL_COLUMNS),
                                   units.get(name), pw, pw.results, origin))
    fit_states = [s for s in states if s != "all"]
    order: Dict[str, Dict[str, List[str]]] = {}
    for b in blocks:
        for s, r in zip(segs, b.fit_results):
            order.setdefault(b.name, {}).setdefault(s.state.value, list(r.params))
    prov = _provenance("fit_report", origin, dt, nrmse, states, units)
    prov.update({
        "error": "e = fit - data",
        "models": {name: {st: normalize_model(spec, st) for st in fit_states} for name, spec in models.items()},
        "parameter_order": order,
        "parameter_reference": "params are stated at t_ref (poly: segment centre; first_order, exp: first valid "
                               "sample); FitResult.evaluation is the stable form for fit.evaluate",
    })
    return Report(blocks, origin, prov)


def compare_report(a, b, segs: Sequence[Segment], *, names: Sequence[str] = ("a", "b"), t=None,
                   dt: Optional[float] = None, units: Optional[str] = None, states: Sequence[str] = STATES,
                   nrmse: str = "mean") -> Report:
    """
    Series ``a`` against reference ``b`` (``e = a - b``), by state and by
    segment. ``names`` labels them in the header (``"a vs b"``).
    """
    index_src = a if hasattr(a, "index") else None
    av, bv = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    t, dt, origin = _times(index_src if index_src is not None else {"a": av}, t, dt)
    by_state = stats_by_state(av, bv, segs, dt=dt, states=states, nrmse=nrmse, t=t)
    per = stats_by_segment(av, bv, segs, dt=dt, nrmse=nrmse, t=t)
    starts, stops = per["start"].to_numpy(dtype=int), per["stop"].to_numpy(dtype=int)
    per["t_start_s"] = t[starts] if len(per) else np.array([], dtype=float)
    per["t_end_s"] = t[stops - 1] if len(per) else np.array([], dtype=float)
    block = SignalReport(names[0], f"{names[0]} vs {names[1]}", by_state, per, units, time_origin=origin)
    prov = _provenance("compare_report", origin, dt, nrmse, states, {names[0]: units} if units else None)
    prov.update({"error": f"e = {names[0]} - {names[1]}", "names": list(names[:2])})
    return Report([block], origin, prov)


def _provenance(kind: str, origin, dt: float, nrmse: str, states: Sequence[str], units) -> Dict[str, Any]:
    from . import __version__

    return {
        "kind": kind, "rocksteady_version": __version__,
        "time": {"unit": "s", "origin": None if origin is None else str(origin), "sample_period_s": float(dt)},
        "states": [str(s) for s in states], "nrmse": nrmse,
        "nrmse_normalization": {"mean": "|mean(reference)|", "range": "max - min of reference",
                                "std": "std of reference", "none": "not computed"}[nrmse],
        "uncertainty": dict(UNCERTAINTY_DESCRIPTION), "units": dict(units or {}),
    }


# ---------------------------------------------------------------------------
# Text formatting
# ---------------------------------------------------------------------------


def _indent(text: str, by: str = "  ") -> str:
    return "\n".join(by + line for line in text.splitlines())


def _duration(seconds: float) -> str:
    if not np.isfinite(seconds):
        return ""
    if seconds < 120:
        return f"{seconds:.0f} s"
    if seconds < 7200:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.2f} h"


def _decimals(values) -> int:
    """Decimals that give three significant figures to the smallest non-zero of ``values`` (typical spreads)."""
    v = np.asarray([abs(x) for x in values if np.isfinite(x) and x != 0], dtype=float)
    if not len(v):
        return 2
    return int(np.clip(2 - np.floor(np.log10(v.min())), 0, 8))


def _fmt(x, dec: int, signed: bool = False) -> str:
    if x is None or not np.isfinite(x):
        return ""
    if abs(x) < 0.5 * 10.0 ** -dec:
        x = 0.0  # no "-0.000"
    return f"{x:+.{dec}f}" if signed else f"{x:.{dec}f}"


def _table(header: Sequence[str], rows: Sequence[Sequence[str]], left: Sequence[int] = (0,)) -> str:
    """Align columns: numbers right, the columns in ``left`` left."""
    cols = list(zip(header, *rows)) if rows else [(h,) for h in header]
    widths = [max(len(c) for c in col) for col in cols]

    def line(cells):
        padded = (c.ljust(w) if i in left else c.rjust(w) for i, (c, w) in enumerate(zip(cells, widths)))
        return "  ".join(padded).rstrip()

    return "\n".join([line(header)] + [line(r) for r in rows])


def _state_table(df) -> str:
    dec = _decimals(np.concatenate([df[c].to_numpy(float) for c in ("std", "mae", "rmse")]))
    header = ["state", "n", "time", "bias", "std", "MAE", "RMSE", "max|e|", "NRMSE", "r", "u(bias)",
              "tau_int", "N_eff"]
    rows = []
    for state, r in df.iterrows():
        if r["n"] == 0:
            rows.append([str(state), "0"] + [""] * (len(header) - 2))
            continue
        rows.append([
            str(state), f"{int(r['n']):,}", _duration(r["duration_s"]), _fmt(r["bias"], dec, signed=True),
            _fmt(r["std"], dec), _fmt(r["mae"], dec), _fmt(r["rmse"], dec), _fmt(r["max_abs"], dec),
            f"{r['nrmse']:.2%}" if np.isfinite(r["nrmse"]) else "",
            f"{r['r']:.4f}" if np.isfinite(r["r"]) else "—",
            _fmt(r["u_bias"], dec + 1), _duration(r["tau_int_s"]),
            f"{r['n_eff']:,.0f}" + ("*" if r["low_effective_sample"] else ""),
        ])
    return _table(header, rows)


def _clock(seconds: float, origin, multi_day: bool = False) -> str:
    if origin is None:
        return f"{seconds:,.0f} s"
    import pandas as pd

    # Whole nanoseconds, truncated, as pd.Timedelta(seconds=...) gives; that form warns on pandas 2.x with numpy 2.5+.
    ts = origin + pd.Timedelta(int(float(seconds) * 1_000_000_000), "ns")
    return ts.strftime("%m-%d %H:%M:%S" if multi_day else "%H:%M:%S")


def _param_text(r: Optional[FitResult]) -> str:
    if r is None:
        return ""
    poly = r.model == "poly"
    # Polynomial: the shape beyond the level (which the mean column shows), without uncertainties.
    names = [k for k in r.params if k != "level"] if poly else [k for k in r.params if k not in _HIDDEN_PARAMS]
    u = r.u
    parts = []
    for k in names:
        v = r.params[k]
        if not np.isfinite(v):
            continue
        unit = " s" if k in _TIME_PARAMS else ""
        if poly or not np.isfinite(u[k]):
            parts.append(f"{k} {v:.4g}{unit}")
        else:
            parts.append(f"{k} {v:.4g} ± {u[k]:.2g}{unit}")
    if r.flags:
        parts.append(f"[{', '.join(sorted(r.flags))}]")
    return "; ".join(parts)


def _sig(x, digits: int = 3) -> str:
    """``x`` to ``digits`` significant figures (blank if not finite)."""
    return f"{x:.{digits}g}" if x is not None and np.isfinite(x) else ""


def _interval_table(df, fits: Sequence[FitResult], origin) -> str:
    """
    One row per segment. Segments can differ by orders of magnitude (a
    shutdown at exactly zero next to full power), so each row is formatted
    on its own: spreads to three significant figures, levels to the
    decimals that resolve that row's spread.
    """
    is_fit = "model" in df.columns
    if is_fit:
        header = ["#", "state", "start", "end", "n", "model", "mean", "std", "tau_int", "N_eff", "u(mean)", "RMSE",
                  "parameters"]
    else:
        header = ["#", "state", "start", "end", "n", "bias", "std", "RMSE", "r", "tau_int", "N_eff", "u(bias)"]
    rows = []
    multi_day = len(df) > 0 and float(df["t_end_s"].max()) >= 86400.0
    for i, r in df.reset_index(drop=True).iterrows():
        start, end = _clock(r["t_start_s"], origin, multi_day), _clock(r["t_end_s"], origin, multi_day)
        state = r["state"] + (" (forced)" if r.get("forced", False) else "")  # an override, not data, set it
        n_eff = f"{r['n_eff']:,.0f}" + ("*" if r["low_effective_sample"] else "") if r["n"] else ""
        spread = r["std"] if np.isfinite(r["std"]) and r["std"] > 0 else float("nan")
        dec = int(np.clip(2 - np.floor(np.log10(spread)), 0, 8)) if np.isfinite(spread) else 2
        if is_fit:
            rows.append([str(int(r["segment"]) + 1), state, start, end, f"{int(r['n']):,}", r["model"],
                         _fmt(r["mean"], dec), _sig(r["std"]), _duration(r["tau_int_s"]), n_eff,
                         _sig(r["u_mean"], 2), _sig(r["rmse"]), _param_text(fits[i] if i < len(fits) else None)])
        else:
            rows.append([str(int(r["segment"]) + 1), state, start, end, f"{int(r['n']):,}",
                         _fmt(r["bias"], dec, signed=True), _sig(r["std"]), _sig(r["rmse"]),
                         f"{r['r']:.4f}" if np.isfinite(r["r"]) else "—", _duration(r["tau_int_s"]), n_eff,
                         _sig(r["u_bias"], 2)])
    return _table(header, rows, left=(1, 5, 12) if is_fit else (1,))


__all__ = ["Report", "SignalReport", "fit_report", "compare_report"]
