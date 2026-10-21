"""tools/check.sh names its interpreter, and stops before the checks when it lacks the dev extra (#84)."""

import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECK = os.path.join(ROOT, "tools", "check.sh")


@pytest.mark.skipif(not os.path.exists(CHECK) or sys.platform == "win32",
                    reason="tools/check.sh is not in the sdist, and needs bash")
def test_an_interpreter_without_the_dev_extra_is_named_and_stops_the_checks(tmp_path):
    bare = tmp_path / "bare-python"  # this interpreter, without site-packages: no pytest, no sphinx
    bare.write_text(f'#!/bin/sh\nexec "{sys.executable}" -S "$@"\n')
    bare.chmod(0o755)
    run = subprocess.run(["bash", CHECK], env={**os.environ, "PYTHON": str(bare)},
                         capture_output=True, text=True, timeout=60)
    assert run.returncode != 0
    assert str(bare) in run.stderr
    assert "pytest" in run.stderr and "sphinx" in run.stderr
    assert 'pip install -e ".[dev]"' in run.stderr
    assert "passed" not in run.stdout  # stopped before the tests ran
