"""tools/check.sh names its interpreter, and stops before the checks when it lacks the dev extra (#84)."""

import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECK = os.path.join(ROOT, "tools", "check.sh")

pytestmark = pytest.mark.skipif(not os.path.exists(CHECK) or sys.platform == "win32",
                                reason="tools/check.sh is not in the sdist, and needs bash")


def _check(python):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}  # a PYTHONPATH could undo -S below
    return subprocess.run(["bash", CHECK], env={**env, "PYTHON": str(python)},
                          capture_output=True, text=True, timeout=60)


def test_an_interpreter_without_the_dev_extra_is_named_and_stops_the_checks(tmp_path):
    bare = tmp_path / "bare-python"  # this interpreter, without site-packages: no pytest, no sphinx
    bare.write_text(f'#!/bin/sh\nexec "{sys.executable}" -S "$@"\n')
    bare.chmod(0o755)
    run = _check(bare)
    assert run.returncode == 1
    assert str(bare) in run.stderr and sys.executable in run.stderr  # as given, and the binary it runs
    assert "pytest" in run.stderr and "sphinx" in run.stderr
    assert f'{sys.executable} -m pip install -e ".[dev]"' in run.stderr  # that interpreter's own pip
    assert "passed" not in run.stdout  # stopped before the tests ran


def test_a_missing_interpreter_is_named_and_stops_the_checks(tmp_path):
    run = _check(tmp_path / "no-such-python")
    assert run.returncode == 1
    assert "check.sh:" in run.stderr and str(tmp_path / "no-such-python") in run.stderr
