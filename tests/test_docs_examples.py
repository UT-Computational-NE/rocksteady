"""The documentation's example files run as published (#49)."""

import os
import subprocess
import sys

import pytest

EXAMPLES = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "source", "examples")


@pytest.mark.parametrize("name", ["adapter_basic.py", "adapter_custom.py"])
def test_docs_example_runs(name):
    # A fresh interpreter each: the custom example registers components by name, once per process.
    proc = subprocess.run([sys.executable, os.path.join(EXAMPLES, name)], capture_output=True, text=True,
                          timeout=300)
    assert proc.returncode == 0, proc.stderr[-2000:]
