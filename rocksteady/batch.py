"""
Batch evaluation of a record with a detector, and interval extraction.

The detector is causal, so running it over a stored record gives exactly
the labels the online detector would have produced.
"""

from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .core import Detector, Label


def run(detector: Detector, df: pd.DataFrame, t: Optional[np.ndarray] = None,
        with_results: bool = False) -> pd.DataFrame:
    """
    Evaluate every row of ``df`` in order.

    Parameters
    ----------
    detector : Detector
        It is reset before the run.
    df : DataFrame
        One row per sample; columns are what the signals' sources read.
    t : array of float, optional
        Sample times in seconds. Default: row index times the sample period.
    with_results : bool
        Also return one column per criterion with its margin ratio.

    Returns
    -------
    DataFrame indexed like ``df`` with columns ``label``, ``eta``,
    ``limiting``, ``since_disturbance_s``, ``forced_by`` and, optionally,
    ``ratio_<criterion>``.
    """
    detector.reset()
    if t is None:
        t = np.arange(len(df)) * detector.timing.sample_period_s
    cols = list(df.columns)
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
    return list(zip(starts, stops))


def intervals(labels: pd.Series, label: Label) -> List[Tuple[int, int]]:
    """Index runs carrying one label."""
    return runs(labels.to_numpy() == label.value)


def accepted_intervals(labels: pd.Series) -> List[Tuple[int, int]]:
    """
    Runs that reached STEADY, including the HOLDING span that preceded them.

    With ``hold_s = 0`` these are simply the STEADY runs.
    """
    lab = labels.to_numpy()
    out = []
    for a, b in runs((lab == Label.STEADY.value) | (lab == Label.HOLDING.value)):
        if (lab[a:b] == Label.STEADY.value).any():
            out.append((a, b))
    return out
