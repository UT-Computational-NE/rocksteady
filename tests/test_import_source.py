"""pytest imports the checkout's own src/, whatever else the environment points at (#99)."""

import os
import subprocess
import sys

import pytest

import rocksteady

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_pytest_imports_the_checkouts_own_src_over_another_install(tmp_path):
    """A decoy rocksteady first on PYTHONPATH stands in for an editable install of another checkout."""
    decoy = tmp_path / "decoy" / "rocksteady"
    decoy.mkdir(parents=True)
    (decoy / "__init__.py").write_text("")
    loaded = tmp_path / "loaded.txt"
    probe = tmp_path / "test_probe.py"
    probe.write_text(f"import rocksteady\n\n\ndef test_where():\n"
                     f"    open({str(loaded)!r}, 'w').write(rocksteady.__file__)\n")
    env = dict(os.environ, PYTHONPATH=str(tmp_path / "decoy"))
    run = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-c",
                          os.path.join(ROOT, "pyproject.toml"), "--rootdir", ROOT, str(probe)],
                         env=env, capture_output=True, text=True, check=False, timeout=120)
    assert run.returncode == 0, run.stdout + run.stderr
    assert loaded.read_text() == os.path.join(ROOT, "src", "rocksteady", "__init__.py")


@pytest.mark.skipif(not os.environ.get("ROCKSTEADY_TEST_INSTALLED"), reason="set by CI's build job only")
def test_the_build_job_tests_the_installed_package():
    """CI's build job runs the sdist's tests with `-o pythonpath=`, so they import the installed wheel."""
    assert not rocksteady.__file__.startswith(os.path.join(ROOT, "src") + os.sep), rocksteady.__file__
