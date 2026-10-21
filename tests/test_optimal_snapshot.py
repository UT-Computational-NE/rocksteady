"""
Best-fit segments snapshot: ``optimal_segments`` results pinned to 0.5.0.

Seeded synthetic records are run through ``subset``, ``refine`` and
``exact`` and compared with stored segments, states, objective and counts. A
change that alters any of them fails here; if the change is intended,
regenerate with

    python tests/test_optimal_snapshot.py

and record the reason in CHANGELOG.md.
"""

import json
import os

import numpy as np
import pandas as pd
import pytest

from rocksteady.optimal import optimal_segments

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
SNAPSHOT = os.path.join(DATA, "optimal_snapshot.json")


def _startup(seed=6, n=1800):
    rng = np.random.default_rng(seed)
    t = np.arange(float(n))
    y = np.zeros(n)
    up, down = int(n * 2 / 9), int(n * 13 / 18)
    y[up:] = 100 * (1 - np.exp(-(t[up:] - up) / 60.0))
    y[down:] = y[down - 1] * np.exp(-(t[down:] - down) / 30.0)
    return t, y + rng.normal(0, 0.5, n), (up, down)


def _plateaus(seed=2, n=400):
    rng = np.random.default_rng(seed)
    t = np.arange(float(n))
    return t, np.where((t % 100) < 50, 0.0, 4.0) + rng.normal(0, 0.05, n)


def _days(seed=12):
    """Three local days, each with a low and a high plateau and the moves between them, and a gap."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2026-03-02 00:00", periods=3 * 24 * 6, freq="10min", tz="America/Chicago")
    hour = idx.hour + idx.minute / 60.0
    y = np.where((hour >= 9) & (hour < 16), 100.0, 0.0) + rng.normal(0, 1.0, len(idx))
    y[200:206] = np.nan
    return idx, y


def _cases():
    t, y, (up, down) = _startup()
    lin = {"steady": 0, "transient": 1}
    late = [0, up + 37, up + 300, down - 41, down + 150, len(y)]
    caps = {"steady": 3, "transient": 2}
    yield "startup-subset", dict(t=t, y=y, models=lin, method="subset", candidates=late, max_counts=caps,
                                 penalty=10.0)
    yield "startup-refine", dict(t=t, y=y, models=lin, method="refine", candidates=late, refine_s=120.0,
                                 max_counts=caps, penalty=10.0)
    yield "startup-exact", dict(t=t, y=y, models=lin, method="exact", grid_s=5, candidates=late, max_counts=caps,
                                penalty=10.0)
    yield "startup-exact-bic", dict(t=t, y=y, models=lin, method="exact", grid_s=10, max_counts=caps)
    expm = {"steady": 0, "transient": {"model": "exp", "T_min": 5.0, "T_max": 1e4}}
    grid = list(range(0, len(y), 100)) + [len(y)]
    yield "startup-exp-refine", dict(t=t, y=y, models=expm, method="refine", candidates=grid, refine_s=60.0,
                                     max_counts=caps, max_duration_s={"transient": 900.0})
    yield "startup-exp-subset-noprune", dict(t=t, y=y, models=expm, method="subset", candidates=grid,
                                             max_counts=caps, max_duration_s={"transient": 900.0}, prune=False)
    alt = {"steady": ["transient"], "transient": ["steady"]}
    yield "startup-alternating", dict(t=t, y=y, models=lin, method="exact", grid_s=10, transitions=alt,
                                      min_duration_s={"steady": 40.0, "transient": 20.0},
                                      penalty={"steady": 0.5, "transient": 5.0})
    three = {"steady": 0, "transient": 1, "indeterminate": 2}
    yield "startup-three-states", dict(t=t, y=y, models=three, method="exact", grid_s=20, per=600.0,
                                       max_counts={"steady": 2, "indeterminate": 1},
                                       transitions={"steady": ["transient", "indeterminate"],
                                                    "transient": ["steady", "indeterminate"],
                                                    "indeterminate": ["steady"]},
                                       penalty={"steady": 5.0, "transient": 20.0, "indeterminate": 50.0})

    t, y = _plateaus()
    pen = {"steady": 1.0, "transient": 20.0}
    yield "plateaus-per-period", dict(t=t, y=y, models=lin, method="exact", grid_s=10, penalty=pen, per=100.0,
                                      max_counts={"steady": 1})
    yield "plateaus-spanning", dict(t=t, y=np.zeros(len(t)), models={"steady": 0}, method="exact", grid_s=50,
                                    penalty=0, per=100.0, max_counts={"steady": 2})
    yield "plateaus-subset-periods", dict(t=t, y=y, models=lin, method="subset",
                                          candidates=list(range(0, 401, 25)), per=100.0,
                                          max_counts={"steady": 2, "transient": 1}, penalty=pen,
                                          max_duration_s={"steady": 120.0})
    yield "plateaus-refine-periods", dict(t=t, y=y, models=lin, method="refine",
                                          candidates=list(range(0, 401, 30)), refine_s=20.0, per=100.0,
                                          max_counts={"steady": 2, "transient": 2}, penalty=pen)

    idx, y = _days()
    yield "days-exact", dict(t=idx, y=y, models=lin, method="exact", grid_s=1800, per="1D",
                             max_counts={"steady": 3, "transient": 2}, penalty=50.0)
    yield "days-refine", dict(t=idx, y=y, models=lin, method="refine", candidates=list(range(0, len(y), 18)),
                              refine_s=3600.0, per="1D", max_counts={"steady": 3, "transient": 2})


CASES = dict(_cases())


def _run(kw):
    kw = dict(kw)
    t, y, models = kw.pop("t"), kw.pop("y"), kw.pop("models")
    r = optimal_segments(t, y, models, **kw)
    return {"segments": [[s.start, s.stop, s.state.value] for s in r.segments],
            "objective": r.objective, "sse": r.sse, "n_candidates": r.n_candidates,
            "counts": {"index": [str(i) for i in r.counts.index], "columns": list(r.counts.columns),
                       "values": r.counts.to_numpy().tolist()}}


def _snapshot():
    with open(SNAPSHOT) as f:
        return json.load(f)


@pytest.mark.parametrize("name", list(CASES))
def test_results_match_the_snapshot(name):
    want = _snapshot()[name]
    got = _run(CASES[name])
    assert got["segments"] == want["segments"]
    assert got["counts"] == want["counts"]
    assert got["n_candidates"] == want["n_candidates"]
    assert got["objective"] == pytest.approx(want["objective"], rel=1e-12)
    assert got["sse"] == pytest.approx(want["sse"], rel=1e-12)


def test_infeasible_constraints_still_raise():
    """Transients of 120-150 s each touch two 100 s periods, so some period holds two of them."""
    t, y = _plateaus()
    with pytest.raises(ValueError, match="no segmentation satisfies"):
        optimal_segments(t, y, {"steady": 0, "transient": 1}, method="exact", grid_s=10, per=100.0,
                         max_counts={"steady": 0, "transient": 1}, min_duration_s={"transient": 120.0},
                         max_duration_s={"transient": 150.0})


if __name__ == "__main__":
    out = {name: _run(kw) for name, kw in CASES.items()}
    with open(SNAPSHOT, "w") as f:
        json.dump(out, f, indent=1)
        f.write("\n")
    print(f"wrote {SNAPSHOT}: {len(out)} cases")
