"""pytest imports the checkout's own src/, whatever else the environment points at (#99)."""

import os
import pathlib
import subprocess
import sys

import pytest

import rocksteady

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SRC = os.path.join(ROOT, "src") + os.sep
# An unpacked sdist has PKG-INFO at its root, a checkout does not. Its tests run against an installed package.
IN_SDIST = os.path.exists(os.path.join(ROOT, "PKG-INFO"))


@pytest.mark.skipif(IN_SDIST, reason="an sdist's tests run against an installed package")
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


@pytest.mark.skipif(IN_SDIST, reason="an sdist's tests run against an installed package")
def test_pytest_imports_the_checkouts_own_src_over_another_install(tmp_path):
    """A decoy rocksteady first on PYTHONPATH stands in for an editable install of another checkout."""
    decoy = tmp_path / "decoy" / "rocksteady"
    decoy.mkdir(parents=True)
    (decoy / "__init__.py").write_text("")
    imported_from = tmp_path / "imported_from.txt"
    probe = tmp_path / "test_probe.py"
    probe.write_text(f"import pathlib\n\nimport rocksteady\n\n\ndef test_where():\n"
                     f"    pathlib.Path({str(imported_from)!r}).write_text(rocksteady.__file__)\n")
    env = dict(os.environ, PYTHONPATH=str(tmp_path / "decoy"))
    env.pop("PYTEST_ADDOPTS", None)  # this run's own pytest options, not the outer run's
    run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-c",
                          os.path.join(ROOT, "pyproject.toml"), "--rootdir", ROOT, str(probe)],
                         env=env, capture_output=True, text=True, check=False, timeout=120)
    assert run.returncode == 0, run.stdout + run.stderr
    assert pathlib.Path(imported_from).read_text() == os.path.join(SRC, "rocksteady", "__init__.py")
