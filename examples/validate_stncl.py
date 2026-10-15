"""
Check that the library-based STNCL adapter reproduces loop_steady_state.py
(the plan's reference implementation) on the example run.

The reference evaluates fixed 30 s blocks and labels each window at its
centre; the library evaluates a trailing window at every sample. The two
coincide at samples where the trailing window is block-aligned, so the
comparison is made there, shifted by half a window.

Usage: python validate_stncl.py <run.csv> <path to loop_steady_state.py dir>
"""

import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "adapters"))
sys.path.insert(0, sys.argv[2])

import loop_steady_state as ref  # noqa: E402
from rocksteady import batch  # noqa: E402
from stncl_steady_state import build_detector  # noqa: E402

df = pd.read_csv(sys.argv[1])
df = df[[c for c in df.columns if c not in ref.CFG["exclude"]]]
dt = 1.0

for cls in ("operating", "calibration"):
    t0 = time.time()
    res = ref.evaluate(df, dt, cls)
    t1 = time.time()
    det = build_detector(cls)
    out = batch.run(det, df)
    t2 = time.time()

    k, wb = res["k"], res["wb"]
    passes = res["passes"]
    centres = [s + wb // 2 for s in range(0, len(passes) - wb + 1)]
    ends = [(c - wb // 2 + wb) * k - 1 for c in centres]  # trailing-window end sample
    lib_pass = out["label"].isin(["steady", "holding"]).to_numpy()[ends]
    ref_pass = passes[centres]
    mism = int((lib_pass != ref_pass).sum())
    print(f"class {cls}: windows compared {len(centres)}, reference pass {int(ref_pass.sum())}, "
          f"library pass {int(lib_pass.sum())}, mismatches {mism}; ref {t1-t0:.1f}s, lib {t2-t1:.1f}s")
    acc_ref = [(a * k, b * k) for a, b in res["accepted"]]
    acc_lib = [(out["t_s"].iloc[a], out["t_s"].iloc[b - 1]) for a, b in batch.accepted_intervals(out["label"])]
    print(f"   accepted (reference, block->s): {acc_ref}")
    print(f"   accepted (library, s):          {acc_lib}")
    # Limiting signal distribution where the library window fails.
    fails = out.loc[out["label"].isin(["settling", "indeterminate"]) & (out["since_disturbance_s"] > 0), "limiting"]
    print(f"   limiting signals: {fails.value_counts().head(4).to_dict()}")
    print(f"   label counts: {out['label'].value_counts().to_dict()}")
    print(f"   eta_drift at t=6000 s: {out['eta'].iloc[6000]:.3f} (limiting {out['limiting'].iloc[6000]})")
