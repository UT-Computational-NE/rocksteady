"""pytest imports the checkout's own src/, whatever else the environment points at (#99)."""

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_pytest_imports_the_checkouts_own_src_over_another_install(tmp_path):
    """A decoy rocksteady first on PYTHONPATH stands in for an editable install of another checkout."""
    decoy = tmp_path / "decoy" / "rocksteady"
    decoy.mkdir(parents=True)
    (decoy / "__init__.py").write_text("")
    probe = tmp_path / "test_probe.py"
    probe.write_text("import rocksteady\n\ndef test_where():\n    print('LOADED', rocksteady.__file__)\n")
    env = dict(os.environ, PYTHONPATH=str(tmp_path / "decoy"))
    out = subprocess.run([sys.executable, "-m", "pytest", "-q", "-s", "-p", "no:cacheprovider", "-c",
                          os.path.join(ROOT, "pyproject.toml"), "--rootdir", ROOT, str(probe)],
                         env=env, capture_output=True, text=True, check=False)
    loaded = [line.split(" ", 1)[1] for line in out.stdout.splitlines() if line.startswith("LOADED ")]
    assert loaded == [os.path.join(ROOT, "src", "rocksteady", "__init__.py")], out.stdout + out.stderr
