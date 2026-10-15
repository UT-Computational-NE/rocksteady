"""
Check that the library-based TRIGA adapter reproduces the existing
steady_state_classifier.OnlineSteadyStateTagger label for label.

Usage: python validate_triga.py <dir containing triga_modsim_tools> <blocks.pkl> [...]
Each pickle is a DataFrame of 1 s blocks with columns power, Tran, Shim1,
Shim2, Reg (as written by DataTagger.state_df).
"""

import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, sys.argv[1])  # triga_modsim_tools (current version)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.join(os.path.dirname(HERE), "adapters"))

from triga_modsim_tools.dataproc.steady_state_classifier import OnlineSteadyStateTagger as Old  # noqa: E402
from triga_steady_state import OnlineSteadyStateTagger as New  # noqa: E402

RODS = ["Tran", "Shim1", "Shim2", "Reg"]

for path in sys.argv[2:]:
    st = pd.read_pickle(path)
    P = st["power"].to_numpy()
    Z = st[RODS].to_numpy()
    old, new = Old(), New()
    t0 = time.time()
    a = [old.update(P[i], Z[i]) for i in range(len(st))]
    t1 = time.time()
    b = [new.update(P[i], Z[i]) for i in range(len(st))]
    t2 = time.time()
    s = pd.Series(a, index=st.index)
    mism = (s != pd.Series(b, index=st.index))
    print(f"{path}: {len(st)} samples, mismatches {int(mism.sum())}, old {t1-t0:.1f}s, new {t2-t1:.1f}s")
    if mism.any():
        print(pd.DataFrame({"old": s[mism], "new": pd.Series(b, index=st.index)[mism]}).head(20))
