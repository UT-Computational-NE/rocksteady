"""
Choosing block and window lengths from data.

Block and window lengths depend on two time scales of the application's own
data, so rocksteady has no default for them:

- **block length**: several times the *noise correlation time* of the raw
  samples, so block means are nearly independent, but short compared with
  the *process time scale* that should still be resolved;
- **window**: enough blocks for the criteria's statistics (about 40 for the
  ``"ar1"`` drift test), and no longer than the reaction time that can be
  accepted, since every steady call lags by about one window.

``suggest_timing`` measures both time scales: the noise correlation time on
a stretch known to be steady, and optionally the time constant of a typical
transient. It returns a proposal with its reasoning, which the adapter
reviews and adopts; nothing is set automatically.
"""

import math
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .core import Timing
from .stats import linear_fit, tau_int


@dataclass
class TimingSuggestion:
    """
    A proposed block and window length, with the measurements behind it.

    ``noise_tau_s`` is the integrated autocorrelation time of the steady
    stretch (about a line); ``process_tau_s`` the time constant of the
    transient (NaN if none was given or it could not be resolved);
    ``block_mean_phi`` the lag-1 autocorrelation of block means at the
    suggested block length, measured on the steady stretch (NaN when it is
    too short to tell). ``notes`` explain each choice and any conflict.
    """

    sample_period_s: float
    block_s: float
    window_s: float
    n_blocks: int
    noise_tau_s: float
    process_tau_s: float = float("nan")
    block_mean_phi: float = float("nan")
    notes: List[str] = field(default_factory=list)

    def timing(self, **overrides) -> Timing:
        """A ``Timing`` with the suggested lengths; other options (``settle_s``, ``hold_s``, ...) as keywords."""
        kw = {"sample_period_s": self.sample_period_s, "block_s": self.block_s, "window_s": self.window_s}
        kw.update(overrides)
        return Timing(**kw)

    def __str__(self) -> str:
        head = (f"block_s = {self.block_s:g} s, window_s = {self.window_s:g} s ({self.n_blocks} blocks); "
                f"noise correlation time {self.noise_tau_s:.3g} s")
        if math.isfinite(self.process_tau_s):
            head += f", process time constant {self.process_tau_s:.3g} s"
        return "\n".join([head] + [f"- {n}" for n in self.notes])


def _lag1(x: np.ndarray) -> float:
    x = x - x.mean()
    xx = float(x @ x)
    return float(x[1:] @ x[:-1]) / xx if xx > 0 and len(x) > 2 else float("nan")


def suggest_timing(steady, dt: float, *, transient=None, transient_t=None, block_factor: float = 3.0,
                   min_blocks: int = 40, max_window_s: Optional[float] = None,
                   block_to_process: float = 0.2) -> TimingSuggestion:
    """
    Propose ``block_s`` and ``window_s`` from the application's data.

    Parameters
    ----------
    steady : array
        A stretch of one signal known to be steady (non-finite samples are
        dropped), sampled every ``dt`` seconds. Longer is better: a few
        hundred noise correlation times.
    dt : float
        Sample period in seconds.
    transient : array, optional
        A typical transient of the same signal (a step response), for its
        time constant; ``transient_t`` gives its times (default: every ``dt``).
    block_factor : float
        Block length in noise correlation times (default 3: block means then
        have a lag-1 autocorrelation of roughly 0.1 to 0.3).
    min_blocks : int
        Blocks per window (default 40, where the ``"ar1"`` drift test holds
        its calibration).
    max_window_s : float, optional
        The longest window the application accepts (its reaction time). If
        ``min_blocks`` blocks do not fit, fewer are used and a note says so.
    block_to_process : float
        Largest block length as a fraction of the process time constant
        (default 0.2), so blocks follow the transient.

    Returns
    -------
    TimingSuggestion
        Use ``suggestion.timing(settle_s=..., hold_s=...)`` for a ``Timing``,
        once reviewed.
    """
    dt = float(dt)
    if not dt > 0 or not math.isfinite(dt):
        raise ValueError(f"dt must be a positive number of seconds, got {dt!r}")
    if not block_factor > 0 or not 0 < block_to_process <= 1 or int(min_blocks) < 3:
        raise ValueError("block_factor must be > 0, block_to_process in (0, 1], min_blocks >= 3")
    y = np.asarray(steady, dtype=float)
    y = y[np.isfinite(y)]
    if len(y) < 20:
        raise ValueError(f"steady needs at least 20 finite samples, got {len(y)}")
    notes: List[str] = []
    noise_tau = float(tau_int(y, dt))
    if len(y) * dt < 100 * noise_tau:
        notes.append(f"the steady stretch spans only {len(y) * dt / noise_tau:.0f} noise correlation times; the "
                     "estimate is rough (a few hundred are better)")
    samples = max(1, int(round(block_factor * noise_tau / dt)))
    notes.append(f"block: {block_factor:g} x the noise correlation time {noise_tau:.3g} s, rounded to "
                 f"{samples} sample{'s' if samples != 1 else ''}, so block means are nearly independent")

    process_tau = float("nan")
    if transient is not None:
        from .fit import fit_first_order

        yt = np.asarray(transient, dtype=float)
        tt = np.arange(len(yt)) * dt if transient_t is None else np.asarray(transient_t, dtype=float)
        r = fit_first_order(tt, yt)
        if r.flags & {"tau_unidentifiable", "tau_at_bound", "too_few_samples"}:
            notes.append(f"the transient's time constant is not resolved ({', '.join(sorted(r.flags))}); "
                         "it was not used")
        else:
            process_tau = float(r.params["tau"])
            cap = max(1, int(math.floor(block_to_process * process_tau / dt + 1e-9)))  # rounding of a fitted tau
            if samples > cap:
                notes.append(f"CONFLICT: the noise wants blocks of {samples * dt:g} s, but the process time "
                             f"constant {process_tau:.3g} s allows at most {cap * dt:g} s "
                             f"({block_to_process:g} tau); using {cap * dt:g} s, so block means stay correlated "
                             "and the drift test's k grows. Consider a less noisy signal or averaging channels.")
                samples = cap
            else:
                notes.append(f"the block is at most {block_to_process:g} x the process time constant "
                             f"{process_tau:.3g} s, so blocks follow the transient")
    block_s = samples * dt

    n_blocks = int(min_blocks)
    if max_window_s is not None:
        fit_blocks = int(math.floor(float(max_window_s) / block_s + 1e-9))
        if fit_blocks < n_blocks:
            notes.append(f"max_window_s = {max_window_s:g} s holds only {fit_blocks} blocks, fewer than "
                         f"{min_blocks}; the drift test's calibration weakens below about 40 blocks and fails "
                         "below about 20 (see the calibration tables)")
            n_blocks = max(fit_blocks, 3)
    window_s = n_blocks * block_s
    notes.append(f"window: {n_blocks} blocks = {window_s:g} s; a steady call lags by about one window")
    if math.isfinite(process_tau) and window_s < process_tau:
        notes.append(f"the window is shorter than the process time constant {process_tau:.3g} s: an asymptote "
                     "test (AsymptoteCriterion) has little power on approaches that slow, and a drift test sees "
                     "a nearly linear approach")

    nb = len(y) // samples
    phi_b = float("nan")
    if nb >= 10:
        phi_b = _lag1(linear_fit(np.arange(nb, dtype=float), y[: nb * samples].reshape(nb, samples).mean(axis=1))[2])
        notes.append(f"measured lag-1 autocorrelation of block means on the steady stretch: {phi_b:.2f}")
    return TimingSuggestion(dt, block_s, window_s, n_blocks, noise_tau, process_tau, phi_b, notes)


__all__ = ["TimingSuggestion", "suggest_timing"]
