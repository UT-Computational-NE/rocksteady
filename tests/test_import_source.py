"""
The tests import the rocksteady they should, whatever else the environment points at (#99). In a checkout,
pytest imports the checkout's own src/ through the ini's ``pythonpath``, and so do the processes the tests
start: ``python -c`` and scripts through the conftest's PYTHONPATH, and pytest runs given this tree's config
through its ``pythonpath``. An unpacked sdist's tests, run with ``-o pythonpath=``, import the installed
package.
"""

import os
import pathlib
import subprocess
import sys
import xml.etree.ElementTree as ET

import pytest

import rocksteady

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src") + os.sep
# An unpacked sdist has PKG-INFO at its root, a checkout does not. Its tests run against an installed package.
IN_SDIST = os.path.exists(os.path.join(ROOT, "PKG-INFO"))
# The one skip of the tests that check a checkout's own src/;
# test_the_checkout_only_tests_run_in_a_checkout_and_are_skipped_in_an_sdist checks where it applies.
checkout_only = pytest.mark.skipif(IN_SDIST, reason="an sdist's tests run against an installed package")


@checkout_only
def test_the_tests_import_the_checkouts_own_src():
    assert rocksteady.__file__.startswith(SRC), rocksteady.__file__


@pytest.mark.skipif(not IN_SDIST, reason="only an sdist's tests run against an installed package")
def test_an_sdists_tests_import_the_installed_package():
    """Run them with `-o pythonpath=`, as CI's build job does; otherwise they would test the sdist's own src/."""
    assert not rocksteady.__file__.startswith(SRC), rocksteady.__file__


def test_a_process_a_test_starts_imports_the_same_rocksteady():
    """Tests that run `python -c` or a script must check the code the tests check."""
    run = subprocess.run([sys.executable, "-c", "import rocksteady; print(rocksteady.__file__)"],
                         capture_output=True, text=True, check=True, timeout=120)
    assert run.stdout.strip() == rocksteady.__file__


def _run_pytest(*args, env):
    """pytest run in a child process with this tree's config (ROOT's pyproject.toml) and ``env``."""
    env = dict(env)
    env.pop("PYTEST_ADDOPTS", None)  # this run's own pytest options, not the outer run's
    return subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-c",
                           os.path.join(ROOT, "pyproject.toml"), "--rootdir", ROOT, *args],
                          env=env, capture_output=True, text=True, check=False, timeout=120)


def _pytest_with_a_decoy_first(tmp_path, target):
    """
    pytest run on ``target`` with this checkout's config, and a decoy rocksteady first on PYTHONPATH that stands
    in for an editable install of another checkout.
    """
    decoy = tmp_path / "decoy" / "rocksteady"
    decoy.mkdir(parents=True)
    (decoy / "__init__.py").write_text("")
    return _run_pytest(str(target), env=dict(os.environ, PYTHONPATH=str(tmp_path / "decoy")))


@checkout_only
def test_pytest_imports_the_checkouts_own_src_over_another_install(tmp_path):
    imported_from = tmp_path / "imported_from.txt"
    probe = tmp_path / "test_probe.py"
    probe.write_text(f"import pathlib\n\nimport rocksteady\n\n\ndef test_where():\n"
                     f"    pathlib.Path({str(imported_from)!r}).write_text(rocksteady.__file__)\n")
    run = _pytest_with_a_decoy_first(tmp_path, probe)
    assert run.returncode == 0, run.stdout + run.stderr
    assert pathlib.Path(imported_from).read_text() == os.path.join(SRC, "rocksteady", "__init__.py")


@checkout_only
def test_a_process_a_test_starts_imports_the_same_rocksteady_over_another_install(tmp_path):
    """
    test_a_process_a_test_starts_imports_the_same_rocksteady, under a decoy: the conftest must put the tests'
    rocksteady before it.
    """
    node_id = f"{__file__}::{test_a_process_a_test_starts_imports_the_same_rocksteady.__name__}"
    run = _pytest_with_a_decoy_first(tmp_path, node_id)
    assert run.returncode == 0 and "1 passed" in run.stdout, run.stdout + run.stderr


def test_the_checkout_only_tests_run_in_a_checkout_and_are_skipped_in_an_sdist(tmp_path):
    """
    The checkout-only tests, and no others, are marked ``checkout_only``; each runs in a checkout and is skipped
    in an sdist, and the sdist's own test the other way round. They check that a checkout's tests, and the
    processes they start, import its own src/, so a checkout must run each of them. ``--setup-only`` applies each
    test's skip without running it.
    """
    want = ["test_the_tests_import_the_checkouts_own_src",
            "test_pytest_imports_the_checkouts_own_src_over_another_install",
            "test_a_process_a_test_starts_imports_the_same_rocksteady_over_another_install"]
    marked = sorted(name for name, f in globals().items()
                    if name.startswith("test_") and checkout_only.mark in getattr(f, "pytestmark", []))
    assert marked == sorted(want)
    report = tmp_path / "report.xml"
    # The skips depend only on PKG-INFO, and the child runs no test, so it needs none of the outer run's options
    # (an sdist's `-o pythonpath=`).
    run = _run_pytest("--setup-only", f"--junitxml={report}", __file__, env=os.environ)
    assert run.returncode == 0, run.stdout + run.stderr
    skipped = {case.get("name"): case.find("skipped") is not None for case in ET.parse(report).iter("testcase")}
    assert {name: skipped.get(name) for name in want} == {name: IN_SDIST for name in want}
    assert skipped.get("test_an_sdists_tests_import_the_installed_package") is (not IN_SDIST)
