"""
Check that the library-based STNCL adapter reproduces loop_steady_state.py
(the plan's reference implementation) on the example run.

The reference evaluates fixed 30 s blocks and labels each window at its
centre; the library evaluates a trailing window at every sample. The two
coincide at samples where the trailing window is block-aligned, so the
comparison is made there, shifted by half a window.

Each class is checked twice: at the plan's limits, and with every limit
multiplied by RELAX so that windows do pass and the pass/accept logic is
exercised. Margin ratios (eta) are compared at every aligned window.

Usage: python validate_stncl.py <run.csv> <path to loop_steady_state.py dir>
"""

import dataclasses
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "adapters"))
sys.path.insert(0, sys.argv[2])

import loop_steady_state as ref  # noqa: E402
import stncl_steady_state as adapter  # noqa: E402
from rocksteady import batch  # noqa: E402

RELAX = 10.0
LIMIT_KEYS = ("b_fluid", "b_wall", "b_diag")


def reference_eta(df, dt, cls_name, res):
    """Worst margin ratio over signals at each window centre, from the reference's own functions."""
    T, _, _ = ref.build_targets(df, ref.CFG)
    k, wb, tb = res["k"], res["wb"], res["tb"]
    cls = ref.CLASSES[cls_name]
    eta = np.full(len(tb), -np.inf)
    for _, (arr, group) in T.items():
        y = ref.block_average(arr, k)
        bcrit = ref.limit_for(group, cls) / 60.0
        for s in range(0, len(tb) - wb + 1):
            b, ub = ref.hac_slope(tb[s:s + wb], y[s:s + wb])
            eta[s + wb // 2] = max(eta[s + wb // 2], (abs(b) + 2 * ub) / bcrit)
    return eta


def compare(df, dt, cls, scale):
    ref_cls, lib_cls = dict(ref.CLASSES[cls]), adapter.CLASSES[cls]
    ref.CLASSES[cls].update({key: ref_cls[key] * scale for key in LIMIT_KEYS})
    adapter.CLASSES[cls] = dataclasses.replace(lib_cls, **{key: getattr(lib_cls, key) * scale for key in LIMIT_KEYS})
    try:
        t0 = time.time()
        res = ref.evaluate(df, dt, cls)
        eta_ref = reference_eta(df, dt, cls, res)
        t1 = time.time()
        out = batch.run(adapter.build_detector(cls), df)
        t2 = time.time()
    finally:
        ref.CLASSES[cls] = ref_cls
        adapter.CLASSES[cls] = lib_cls

    k, wb = res["k"], res["wb"]
    passes = res["passes"]
    centres = np.array([s + wb // 2 for s in range(0, len(passes) - wb + 1)])
    ends = (centres - wb // 2 + wb) * k - 1  # trailing-window end sample
    lib_pass = out["label"].isin(["steady", "holding"]).to_numpy()[ends]
    ref_pass = passes[centres]
    eta_lib = out["eta"].to_numpy()[ends]
    d_eta = float(np.max(np.abs(eta_lib - eta_ref[centres]) / np.abs(eta_ref[centres])))
    mism = int((lib_pass != ref_pass).sum())
    print(f"class {cls} x{scale:g}: windows compared {len(centres)}, reference pass {int(ref_pass.sum())}, "
          f"library pass {int(lib_pass.sum())}, mismatches {mism}, max rel. eta diff {d_eta:.1e}; "
          f"ref {t1-t0:.1f}s, lib {t2-t1:.1f}s")
    acc_ref = [(int(a * k), int(b * k)) for a, b in res["accepted"]]
    acc_lib = [(float(out["t_s"].iloc[a]), float(out["t_s"].iloc[b - 1])) for a, b in batch.accepted_intervals(out["label"])]
    print(f"   accepted (reference, block->s): {acc_ref}")
    print(f"   accepted (library, s):          {acc_lib}")
    fails = out.loc[out["label"].isin(["settling", "indeterminate"]) & (out["since_disturbance_s"] > 0), "limiting"]
    print(f"   limiting signals: {fails.value_counts().head(4).to_dict()}")
    print(f"   label counts: {out['label'].value_counts().to_dict()}")
    print(f"   eta_drift at t=6000 s: {out['eta'].iloc[6000]:.6f} (limiting {out['limiting'].iloc[6000]})")


df = pd.read_csv(sys.argv[1])
df = df[[c for c in df.columns if c not in ref.CFG["exclude"]]]
for cls in ("operating", "calibration"):
    for scale in (1.0, RELAX):
        compare(df, 1.0, cls, scale)
