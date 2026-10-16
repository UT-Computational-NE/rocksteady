"""
Batch evaluation of a record with a detector, signal values, and interval
extraction.

The detector is causal, so running it over a stored record gives exactly
the labels the online detector would have produced.
"""

from typing import TYPE_CHECKING, List, Optional, Sequence, Tuple

import numpy as np

from .core import Detector, Label

if TYPE_CHECKING:  # pandas is imported where it is used, so the streaming detector does not need it
    import pandas as pd


def _unique_columns(df) -> list:
    """Column names of ``df``, which must be unique: each sample is a name -> value mapping."""
    names = list(df.columns)
    seen, dupes = set(), set()
    for c in names:
        (dupes if c in seen else seen).add(c)
    if dupes:
        raise ValueError(f"duplicate column names {sorted(map(str, dupes))}; samples need unique names")
    return names


def run(detector: Detector, df: "pd.DataFrame", t: Optional[np.ndarray] = None,
        with_results: bool = False) -> "pd.DataFrame":
    """
    Evaluate every row of ``df`` in order.

    Parameters
    ----------
    detector : Detector
        It is reset before the run.
    df : DataFrame
        One row per sample; columns are what the signals' sources read.
    t : array of float, optional
        Sample times in seconds. Default: seconds since the first row if
        ``df`` has a ``DatetimeIndex``, otherwise the row number times the
        sample period. Times in an ``EventLog`` must use the same origin.
    with_results : bool
        Also return one column per criterion with its margin ratio.

    Returns
    -------
    DataFrame indexed like ``df`` with columns ``label``, ``eta``,
    ``limiting``, ``since_disturbance_s``, ``forced_by`` and, optionally,
    ``ratio_<criterion>``.
    """
    import pandas as pd

    detector.reset()
    if t is None:
        if isinstance(df.index, pd.DatetimeIndex) and len(df):
            t = (df.index - df.index[0]).total_seconds().to_numpy()
        else:
            t = np.arange(len(df)) * detector.timing.sample_period_s
    cols = _unique_columns(df)
    data = df.to_numpy()
    rows = []
    for i in range(len(df)):
        v = detector.update(float(t[i]), dict(zip(cols, data[i])))
        row = {"label": v.label.value, "eta": v.eta, "limiting": v.limiting,
               "since_disturbance_s": v.since_disturbance_s, "forced_by": v.forced_by}
        if with_results:
            for name, r in v.results.items():
                row[f"ratio_{name}"] = r.ratio
        rows.append(row)
    out = pd.DataFrame(rows, index=df.index)
    out["t_s"] = t
    return out


def signal_values(detector: Detector, df: "pd.DataFrame") -> "pd.DataFrame":
    """
    The detector's signal values for every row of ``df``: one column per
    signal, indexed like ``df``. A row whose signal cannot be read (missing
    column, non-numeric value) gives NaN for that signal.
    """
    import pandas as pd

    cols = _unique_columns(df)
    data = df.to_numpy()
    out = np.full((len(df), len(detector.signals)), np.nan)
    for i in range(len(df)):
        sample = dict(zip(cols, data[i]))
        for j, s in enumerate(detector.signals):
            try:
                out[i, j] = s.value(sample)
            except (KeyError, TypeError, ValueError):
                pass
    return pd.DataFrame(out, index=df.index, columns=[s.name for s in detector.signals])


def fit_report(detector: Detector, df: "pd.DataFrame", models, *, labels=None, backfill: bool = True,
               min_samples: int = 1, t: Optional[np.ndarray] = None, units=None, states=None, nrmse: str = "mean"):
    """
    Fit and report a record by state, in one call.

    Runs ``detector`` over ``df`` (unless ``labels`` are given: the
    ``batch.run`` output, or a label sequence), splits the record into
    segments by state (``segments_from``, with the certifying window
    backfilled except for override-forced runs), and fits the signals
    named in ``models`` (the detector's signal values, or raw columns of
    ``df``) with ``report.fit_report``. Returns a ``report.Report``.
    """
    from .metrics import STATES
    from .report import fit_report as _fit_report
    from .states import segments_from

    if labels is None:
        labels = run(detector, df, t=t)
    segs = segments_from(detector, labels, backfill=backfill, min_samples=min_samples)
    import pandas as pd

    values = signal_values(detector, df)
    for c in models:  # raw columns, added by position: values has exactly one row per row of df
        if c not in values.columns and c in df.columns:
            values[c] = pd.to_numeric(df[c], errors="coerce").to_numpy(dtype=float)
    if t is None and not hasattr(df.index, "tz_localize"):
        t = np.arange(len(df)) * detector.timing.sample_period_s
    return _fit_report(values, segs, models, t=t, dt=detector.timing.sample_period_s, units=units,
                       states=STATES if states is None else states, nrmse=nrmse)


def runs(mask: Sequence[bool]) -> List[Tuple[int, int]]:
    """Half-open index runs ``[start, stop)`` where ``mask`` is True."""
    m = np.asarray(mask, dtype=bool)
    if len(m) == 0:
        return []
    d = np.diff(m.astype(np.int8))
    starts = list(np.where(d == 1)[0] + 1)
    stops = list(np.where(d == -1)[0] + 1)
    if m[0]:
        starts = [0] + starts
    if m[-1]:
        stops = stops + [len(m)]
    return [(int(a), int(b)) for a, b in zip(starts, stops)]


def label_array(labels: Sequence) -> np.ndarray:
    """Labels (a Series, a list of ``Label`` or of strings) as an array of label strings."""
    values = labels.to_numpy() if hasattr(labels, "to_numpy") else labels
    return np.array([str(v) for v in values], dtype=object)


def intervals(labels: Sequence, label: Label) -> List[Tuple[int, int]]:
    """Index runs carrying one label."""
    return runs(label_array(labels) == label.value)


def accepted_intervals(labels: Sequence) -> List[Tuple[int, int]]:
    """
    Runs that reached STEADY, including the HOLDING span that preceded them.

    With ``hold_s = 0`` these are simply the STEADY runs.
    """
    lab = label_array(labels)
    out = []
    for a, b in runs((lab == Label.STEADY.value) | (lab == Label.HOLDING.value)):
        if (lab[a:b] == Label.STEADY.value).any():
            out.append((a, b))
    return out
