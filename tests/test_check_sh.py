"""tools/check.sh names its interpreter and stops when it lacks the extra it needs (#84); runs a selection (#87);
the full run typechecks (#93) and lints (#106)."""

import os
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECK = os.path.join(ROOT, "tools", "check.sh")

pytestmark = pytest.mark.skipif(not os.path.exists(CHECK) or sys.platform == "win32",
                                reason="tools/check.sh is not in the sdist, and needs bash")


def _check(python, *args, pythonpath=None):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}  # a PYTHONPATH could undo -S below
    env["PYTHON"] = str(python)
    if pythonpath is not None:
        env["PYTHONPATH"] = str(pythonpath)
    return subprocess.run(["bash", CHECK, *args], env=env, capture_output=True, text=True, timeout=60)


def test_an_interpreter_without_the_dev_extra_is_named_and_stops_the_checks(tmp_path):
    bare = tmp_path / "bare-python"  # this interpreter, without site-packages: no pytest, no sphinx
    bare.write_text(f'#!/bin/sh\nexec "{sys.executable}" -S "$@"\n')
    bare.chmod(0o755)
    run = _check(bare)
    assert run.returncode == 1
    assert str(bare) in run.stderr and sys.executable in run.stderr  # as given, and the binary it runs
    assert "pytest" in run.stderr and "sphinx" in run.stderr and "mypy" in run.stderr  # the full run typechecks
    assert "pylint" in run.stderr and "pycodestyle" in run.stderr  # and lints
    # That interpreter's own pip, for the missing distributions only: `pip install -e` from a worktree would point
    # the environment's editable install at the worktree.
    # With the minimum versions the extras in pyproject.toml ask for.
    want = "'pytest>=7' 'mypy>=1.10' 'pylint>=3' 'pycodestyle>=2.11' 'sphinx>=7' 'sphinx-rtd-theme>=2' 'myst-parser>=2'"
    assert f"{sys.executable} -m pip install {want}" in run.stderr
    # A fresh environment also needs the package's own dependencies: the editable install, from the main checkout.
    assert 'pip install -e ".[dev]" from the main checkout' in run.stderr
    assert "passed" not in run.stdout  # stopped before the tests ran


def test_a_missing_interpreter_is_named_and_stops_the_checks(tmp_path):
    run = _check(tmp_path / "no-such-python")
    assert run.returncode == 1
    assert "check.sh:" in run.stderr and str(tmp_path / "no-such-python") in run.stderr


def test_arguments_select_the_tests_and_skip_mypy_lint_and_the_docs_build(tmp_path):
    # As in CI's test jobs, which install only the test extra: the docs modules cannot be found.
    (tmp_path / "sitecustomize.py").write_text(
        "import sys\nfor m in ('sphinx', 'sphinx_rtd_theme', 'myst_parser'):\n    sys.modules[m] = None\n")
    run = _check(sys.executable, "tests/test_check_sh.py::test_a_missing_interpreter_is_named_and_stops_the_checks",
                 pythonpath=tmp_path)
    assert run.returncode == 0, run.stderr
    assert "1 passed" in run.stdout
    assert "check.sh: the selected tests passed; no mypy, lint or docs build" in run.stdout


def test_a_selection_leaves_out_the_slow_tests_unless_it_asks_for_them():
    slow = ("tests/test_asymptote.py", "-k", "test_false_pass_rate_at_the_limit", "--co")  # 9 slow cases
    default = _check(sys.executable, *slow)
    assert default.returncode == 5, default.stdout + default.stderr  # pytest: nothing collected
    assert "test_false_pass_rate_at_the_limit" not in default.stdout
    asked = _check(sys.executable, *slow, "-m", "slow")  # a later -m wins over "not slow"
    assert asked.returncode == 0, asked.stdout + asked.stderr
    assert asked.stdout.count("test_false_pass_rate_at_the_limit") == 9


def test_a_selection_asks_only_for_the_test_extra(tmp_path):
    run = _check(tmp_path / "no-such-python", "tests/test_check_sh.py")
    assert run.returncode == 1
    assert "test extra" in run.stderr and "dev extra" not in run.stderr
