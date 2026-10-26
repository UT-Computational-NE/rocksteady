"""tools/check.sh names its interpreter and stops when it lacks the extra it needs (#84) or the package's dependencies,
or when the interpreter is not executable or exits non-zero (#90); runs a selection (#87); the full run typechecks
(#93) and lints (#106)."""

import os
import shutil
import subprocess
import sys
import sysconfig

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CHECK = os.path.join(ROOT, "tools", "check.sh")

pytestmark = pytest.mark.skipif(not os.path.exists(CHECK) or sys.platform == "win32",
                                reason="tools/check.sh is not in the sdist, and needs bash")


def _check(python, *args, pythonpath=None, path_first=None, script=CHECK):
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}  # a PYTHONPATH could undo -S below
    env["PYTHON"] = str(python)
    if pythonpath is not None:
        env["PYTHONPATH"] = str(pythonpath)
    if path_first is not None:
        env["PATH"] = str(path_first) + os.pathsep + env["PATH"]
    return subprocess.run(["bash", script, *args], env=env, capture_output=True, text=True, timeout=60)


def _check_beside(tmp_path, dependencies):
    """A copy of tools/check.sh in a scratch project whose pyproject.toml has ``dependencies = [<dependencies>]``,
    run on a passing test there."""
    (tmp_path / "tools").mkdir()
    shutil.copy(CHECK, tmp_path / "tools" / "check.sh")
    (tmp_path / "pyproject.toml").write_text(
        f'[project]\nname = "scratch"\ndependencies = [{dependencies}]\n\n'
        '[project.optional-dependencies]\ntest = ["pytest>=7"]\n')
    (tmp_path / "test_scratch.py").write_text("def test_scratch():\n    pass\n")
    return _check(sys.executable, "test_scratch.py", script=tmp_path / "tools" / "check.sh")


def test_a_dependency_is_checked_as_its_distribution_when_its_module_is_named_otherwise(tmp_path):
    run = _check_beside(tmp_path, '"numpy>=1.22", "python-dateutil>=2.8"')  # python-dateutil's module is dateutil
    assert run.returncode == 0, run.stderr
    assert "1 passed" in run.stdout


def test_a_missing_distribution_after_one_with_extras_is_named_with_its_requirement(tmp_path):
    # Two missing distributions, one with extras and one with a dotted name, each after a requirement with extras.
    run = _check_beside(tmp_path, '"numpy>=1.22", "pandas[excel]>=1.5", "no-such-distribution[x]>=1", '
                                  '"no.such.distribution>=2"')
    assert run.returncode == 1
    assert "lacks no-such-distribution no.such.distribution, which the package needs." in run.stderr
    assert f"{sys.executable} -m pip install 'no-such-distribution[x]>=1' 'no.such.distribution>=2')" in run.stderr
    assert "passed" not in run.stdout


def test_a_dependency_list_over_lines_with_comments_is_read_whole(tmp_path):
    # A comment holding ] and a quote, single quotes, and spaces in a requirement and before a comma: the missing
    # distributions after them are named, and their requirements are given as written, with extras, minimum versions
    # and a marker, whose ; must not split the message.
    run = _check_beside(tmp_path, '\n    "numpy>=1.22",  # arrays ] that don\'t end the list\n'
                                  "    'pandas >= 1.5' ,\n    \"no-such-distribution [x] >= 1\",\n"
                                  "    \"no-such-marked>=2; python_version >= '3'\",\n")
    assert run.returncode == 1
    assert "lacks no-such-distribution no-such-marked, which the package needs." in run.stderr
    want = "'no-such-distribution [x] >= 1' \"no-such-marked>=2; python_version >= '3'\")"
    assert f"{sys.executable} -m pip install {want}" in run.stderr
    assert "passed" not in run.stdout


def test_an_interpreter_without_the_dev_extra_is_named_and_stops_the_checks(tmp_path):
    bare = tmp_path / "bare-python"  # this interpreter, without site-packages: no pytest, no sphinx
    bare.write_text(f'#!/bin/sh\nexec "{sys.executable}" -S "$@"\n')
    bare.chmod(0o755)
    run = _check(bare)
    assert run.returncode == 1
    assert str(bare) in run.stderr and sys.executable in run.stderr  # as given, and the binary it runs
    assert "pytest" in run.stderr and "sphinx" in run.stderr and "mypy" in run.stderr  # the full run typechecks
    assert "pylint" in run.stderr and "pycodestyle" in run.stderr  # and lints
    assert "lacks numpy pandas, which the package needs, and pytest" in run.stderr  # its own dependencies first
    # That interpreter's own pip, for the missing distributions only: `pip install -e` from a worktree would point
    # the environment's editable install at the worktree.
    # With the minimum versions the extras in pyproject.toml ask for.
    want = ("'numpy>=1.22' 'pandas>=1.5' 'pytest>=7' 'mypy>=1.10' 'pylint>=3' 'pycodestyle>=2.11' 'sphinx>=7' "
            "'sphinx-rtd-theme>=2' 'myst-parser>=2'")
    assert f"{sys.executable} -m pip install {want}" in run.stderr
    # A fresh environment is set up with the editable install, from the main checkout.
    assert 'pip install -e ".[dev]" from the main checkout' in run.stderr
    assert "passed" not in run.stdout  # stopped before the tests ran


def test_a_missing_interpreter_is_named_and_stops_the_checks(tmp_path):
    run = _check(tmp_path / "no-such-python")
    assert run.returncode == 1
    assert f"check.sh: there is no interpreter {tmp_path / 'no-such-python'}." in run.stderr


def test_a_missing_interpreter_named_without_a_directory_is_named_and_stops_the_checks():
    run = _check("no-such-python-on-path")  # looked up on PATH, as the default python is
    assert run.returncode == 1
    assert "check.sh: there is no interpreter no-such-python-on-path." in run.stderr


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


def test_a_python_that_is_not_executable_is_named_as_such(tmp_path):
    plain = tmp_path / "plain-python"
    plain.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    plain.chmod(0o644)
    run = _check(plain)
    assert run.returncode == 1
    assert f"check.sh: {plain} is not executable" in run.stderr
    assert "there is no interpreter" not in run.stderr


def test_a_python_on_path_that_is_not_executable_is_named_as_such(tmp_path):
    plain = tmp_path / "py3x"  # named without a directory, as found on PATH
    plain.write_text(f'#!/bin/sh\nexec "{sys.executable}" "$@"\n')
    plain.chmod(0o644)
    run = _check("py3x", path_first=tmp_path)
    assert run.returncode == 1
    assert "check.sh: py3x is not executable" in run.stderr


def test_a_python_that_exits_non_zero_is_named_and_stops_the_checks(tmp_path):
    broken = tmp_path / "broken-python"  # it runs, but every call fails, as in a broken environment
    broken.write_text("#!/bin/sh\necho 'no module named encodings' >&2\nexit 3\n")
    broken.chmod(0o755)
    run = _check(broken)
    assert run.returncode == 1
    assert f"check.sh: {broken} exited with status 3" in run.stderr
    assert "no module named encodings" in run.stderr  # its own error, passed through


@pytest.mark.parametrize("fails_on", ["importlib.util", "re.escape"], ids=["finding the modules", "naming them"])
def test_a_python_that_exits_non_zero_after_the_version_check_is_named_and_stops_the_checks(tmp_path, fails_on):
    # It runs without site-packages, so modules are missing, and fails the one call whose code names `fails_on`.
    broken = tmp_path / "broken-python"
    broken.write_text(f'#!/bin/sh\ncase "$*" in *{fails_on}*) echo boom >&2; exit 5;; esac\n'
                      f'exec "{sys.executable}" -S "$@"\n')
    broken.chmod(0o755)
    run = _check(broken)
    assert run.returncode == 1
    assert f"check.sh: {broken} exited with status 5" in run.stderr
    assert "boom" in run.stderr and "lacks" not in run.stderr


@pytest.mark.parametrize("args", [(), ("tests/test_check_sh.py",)], ids=["full run", "selection"])
def test_an_interpreter_without_the_packages_dependencies_is_named_and_stops_the_checks(tmp_path, args):
    # This interpreter without site-packages (-S), on a path holding everything installed there but numpy, so that
    # numpy's distribution is missing as well as its module.
    site = tmp_path / "site"
    site.mkdir()
    for lib in {sysconfig.get_paths()["purelib"], sysconfig.get_paths()["platlib"]}:
        for entry in os.listdir(lib):
            if not entry.lower().startswith("numpy") and not (site / entry).exists():
                (site / entry).symlink_to(os.path.join(lib, entry))
    bare = tmp_path / "bare-python"
    bare.write_text(f'#!/bin/sh\nexec "{sys.executable}" -S "$@"\n')
    bare.chmod(0o755)
    run = _check(bare, *args, pythonpath=site)
    assert run.returncode == 1
    # A selection needs only pytest; a full run may lack the dev extra too (CI's test jobs install the test extra).
    end = "." if args else ""
    assert f"lacks numpy, which the package needs{end}" in run.stderr
    end = ")" if args else ""
    assert f"{sys.executable} -m pip install 'numpy>=1.22'{end}" in run.stderr  # its minimum version in pyproject.toml
    assert "passed" not in run.stdout and "error" not in run.stdout  # stopped before pytest collected anything
