"""
Intervals by state.

A labelled record is split into contiguous intervals (``Segment``), each
in one of three states:

- ``steady``: a run that reached STEADY, together with any HOLDING span
  immediately before it (``batch.accepted_intervals``).
- ``transient``: DISTURBED, SETTLING, and HOLDING that never confirmed.
- ``indeterminate``: INDETERMINATE outside steady intervals (the start of
  a record, after a data gap, or a window too noisy to decide).

The segments cover every sample, without overlap.

Labels are causal: a sample is labelled steady only once the window
*ending* at it passes, so the data in that certifying window were already
steady. ``backfill_samples`` (``segments_from`` sets it to the detector's
window) extends each steady segment back over that window. Runs that an
override forced to steady (for example a zero-power rule after a
shutdown) were not certified by data and are not backfilled, and backfill
never crosses a disturbance.
"""

import warnings
from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence

import numpy as np

from .batch import accepted_intervals, label_array
from .core import Detector, Label


class State(str, Enum):
    STEADY = "steady"
    TRANSIENT = "transient"
    INDETERMINATE = "indeterminate"

    def __str__(self) -> str:
        return self.value


@dataclass(frozen=True)
class Segment:
    """Samples ``[start, stop)`` in one state."""

    start: int
    stop: int
    state: State

    def __len__(self) -> int:
        return self.stop - self.start

    @property
    def slice(self) -> slice:
        return slice(self.start, self.stop)


def _runs_of(states: Sequence[State], start: int) -> List[Segment]:
    """Maximal runs of equal state, offset by ``start``."""
    out: List[Segment] = []
    for i, st in enumerate(states):
        if out and out[-1].state == st:
            out[-1] = Segment(out[-1].start, start + i + 1, st)
        else:
            out.append(Segment(start + i, start + i + 1, st))
    return out


def _merge_short(runs_: List[Segment], min_samples: int) -> List[Segment]:
    """Within one gap between steady segments, give runs shorter than ``min_samples`` their longer neighbour's state."""
    runs_ = list(runs_)
    while len(runs_) > 1:
        short = [i for i, r in enumerate(runs_) if len(r) < min_samples]
        if not short:
            break
        i = short[0]
        nbrs = [j for j in (i - 1, i + 1) if 0 <= j < len(runs_)]
        j = max(nbrs, key=lambda k: len(runs_[k]))
        lo, hi = min(i, j), max(i, j)
        runs_[lo : hi + 1] = [Segment(runs_[lo].start, runs_[hi].stop, runs_[j].state)]
        # merging may join equal neighbours; re-run the maximal-run pass
        merged: List[Segment] = []
        for r in runs_:
            if merged and merged[-1].state == r.state:
                merged[-1] = Segment(merged[-1].start, r.stop, r.state)
            else:
                merged.append(r)
        runs_ = merged
    return runs_


def segments(labels: Sequence, *, backfill_samples: int = 0, min_samples: int = 1,
             forced: Optional[Sequence[bool]] = None, t: Optional[Sequence[float]] = None,
             backfill_s: Optional[float] = None) -> List[Segment]:
    """
    Split a labelled record into segments by state.

    Parameters
    ----------
    labels : sequence of labels
        Label strings or ``Label`` values, one per sample (for example the
        ``label`` column of ``batch.run``).
    backfill_samples : int
        Extend each steady segment backwards by this many samples, but
        never into an earlier steady segment and never across a DISTURBED
        sample. For a window of W samples the certifying window of the
        first steady sample holds W - 1 earlier samples, which is what
        ``segments_from`` uses. Each accepted interval stays its
        own segment, even when backfill closes the gap to the previous one.
    min_samples : int
        Transient and indeterminate runs shorter than this take the state
        of their longer neighbour between the same two steady segments.
        Steady segments are kept whatever their length.
    forced : sequence of bool, optional
        True where an override forced the label (``batch.run``'s
        ``forced_by`` is set). A steady run that starts forced was not
        certified by a window of data, so it is not backfilled.
    t, backfill_s : sample times (s) and a backfill duration, optional
        Backfill by time instead of by count: each steady segment extends
        back to the first stored sample at or after ``t[a] - backfill_s``.
        Use this when samples can be missing from the record (timestamp
        gaps the detector tolerated), so that backfill covers exactly the
        certifying window. Overrides ``backfill_samples``.
    """
    if backfill_samples < 0 or min_samples < 1:
        raise ValueError("backfill_samples must be >= 0 and min_samples >= 1")
    if (t is None) != (backfill_s is None):
        raise ValueError("give both t and backfill_s to backfill by time")
    lab = label_array(labels)
    forced_at = np.zeros(len(lab), dtype=bool) if forced is None else np.asarray(forced, dtype=bool)
    if len(forced_at) != len(lab):
        raise ValueError("forced must have one entry per label")
    disturbed = np.flatnonzero(lab == Label.DISTURBED.value)
    times = None if t is None else np.asarray(t, dtype=float)
    if times is not None and len(times) != len(lab):
        raise ValueError("t must have one entry per label")
    by_time = times is not None and backfill_s > 0
    steady: List[Segment] = []
    prev_stop = 0
    for a, b in accepted_intervals(lab):
        start = a
        if by_time and not forced_at[a]:
            first = times[a] - backfill_s
            tol = 1e-9 * max(1.0, abs(first))
            start = max(int(np.searchsorted(times[: a + 1], first - tol, side="left")), prev_stop)
        elif backfill_samples and not forced_at[a]:
            start = max(a - int(backfill_samples), prev_stop)
        if start < a:
            j = int(np.searchsorted(disturbed, a)) - 1  # last disturbed sample before the run
            if j >= 0:
                start = max(start, int(disturbed[j]) + 1)
        steady.append(Segment(start, b, State.STEADY))
        prev_stop = b
    out: List[Segment] = []
    pos = 0
    for st in steady + [Segment(len(lab), len(lab), State.STEADY)]:
        if st.start > pos:  # a gap of non-steady samples
            gap = [State.INDETERMINATE if x == Label.INDETERMINATE.value else State.TRANSIENT
                   for x in lab[pos : st.start]]
            gap_runs = _runs_of(gap, pos)
            out.extend(_merge_short(gap_runs, int(min_samples)) if min_samples > 1 else gap_runs)
        if len(st):
            out.append(st)
        pos = st.stop
    return out


def segments_from(detector: Detector, run_output, *, backfill: bool = True,
                  min_samples: int = 1) -> List[Segment]:
    """
    ``segments`` for a detector's output, backfilled over the certifying
    window: the evaluated span that ends at each steady run's first sample.

    ``run_output`` is the DataFrame returned by ``batch.run`` (preferred).
    Its ``forced_by`` column keeps override-forced steady runs from being
    backfilled, and its ``t_s`` column makes backfill follow time, which is
    exact when the record has tolerated timestamp gaps. With labels only,
    backfill counts ``evaluated_samples - 1`` stored samples, and a warning
    is issued for a detector with overrides, because forced runs cannot
    then be told apart.
    """
    forced = None
    times = None
    if hasattr(run_output, "columns"):
        if "forced_by" in run_output.columns:
            forced = run_output["forced_by"].notna().to_numpy()
        if "t_s" in run_output.columns:
            times = run_output["t_s"].to_numpy(dtype=float)
        labels = run_output["label"]
    else:
        labels = run_output
        if backfill and detector.overrides:
            warnings.warn("segments_from got labels only for a detector with overrides; pass the batch.run "
                          "output so that override-forced steady runs are not backfilled", stacklevel=2)
    # The window certifying the first steady sample a is its evaluated span [a - W + 1, a]: W - 1 samples before it.
    n_back = max(detector.timing.evaluated_samples - 1, 0) if backfill else 0
    if times is not None:
        return segments(labels, min_samples=min_samples, forced=forced, t=times,
                        backfill_s=n_back * detector.timing.sample_period_s)
    return segments(labels, backfill_samples=n_back, min_samples=min_samples, forced=forced)


def check_segments(segs: Sequence[Segment], n: int, *, require_full: bool = False) -> List[Segment]:
    """
    Check segments against a record of ``n`` samples: each ``0 <= start <
    stop <= n``, in order, not overlapping. A subset of the record (partial
    coverage) is allowed unless ``require_full``. Returns the segments as a
    list; raises ``ValueError`` naming the first bad segment.
    """
    segs = list(segs)
    prev = 0
    for i, s in enumerate(segs):
        if not isinstance(s, Segment):
            raise ValueError(f"segment {i} is not a Segment: {s!r}")
        if not 0 <= s.start < s.stop <= n:
            raise ValueError(f"segment {i} [{s.start}, {s.stop}) is outside a record of {n} samples or empty")
        if s.start < prev:
            raise ValueError(f"segment {i} [{s.start}, {s.stop}) overlaps or precedes the previous segment, which "
                             f"ends at {prev}")
        prev = s.stop
    if require_full and sum(len(s) for s in segs) != n:
        raise ValueError(f"segments cover {sum(len(s) for s in segs)} of {n} samples; full coverage required")
    return segs


def state_runs(segs: Sequence[Segment], state: State) -> List[Segment]:
    """The segments in one state."""
    return [s for s in segs if s.state == State(state)]


__all__ = ["State", "Segment", "segments", "segments_from", "state_runs", "check_segments"]
