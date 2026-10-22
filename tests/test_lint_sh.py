"""tools/lint.sh fails on the mechanical misses it enforces: two blank lines around top-level definitions (#111)."""

import importlib.util
import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LINT = os.path.join(ROOT, "tools", "lint.sh")

pytestmark = pytest.mark.skipif(
    not os.path.exists(LINT) or sys.platform == "win32"
    or any(importlib.util.find_spec(m) is None for m in ("pylint", "pycodestyle")),
    reason="tools/lint.sh is not in the sdist, needs bash, and needs the lint extra")

CLEAN = '"""A module."""\n\n\ndef f():\n    return 1\n\n\ndef g():\n    return 2\n'


def _lint(tmp_path, source):
    """tools/lint.sh run on a tree holding one module, ``source``, beside the repo's own pyproject.toml."""
    for d in ("src", "tests", "tools", "examples"):
        (tmp_path / d).mkdir()
    shutil.copy(LINT, tmp_path / "tools" / "lint.sh")
    shutil.copy(os.path.join(ROOT, "pyproject.toml"), tmp_path / "pyproject.toml")
    (tmp_path / "src" / "module.py").write_text(source)
    env = dict(os.environ, PYTHON=sys.executable)
    return subprocess.run(["bash", str(tmp_path / "tools" / "lint.sh")], env=env, capture_output=True, text=True,
                          timeout=120)


def test_a_clean_module_passes(tmp_path):
    run = _lint(tmp_path, CLEAN)
    assert run.returncode == 0, run.stdout + run.stderr


@pytest.mark.parametrize("source, code", [
    (CLEAN.replace("\n\n\ndef g", "\n\ndef g"), "E302"),  # one blank line before a top-level def
    (CLEAN + "\nX = 1\n", "E305"),  # one blank line after the last one
    (CLEAN.replace("\n\n\ndef g", "\n\n\n\ndef g"), "E303"),  # three
], ids=["one blank line before", "one blank line after", "three blank lines"])
def test_a_top_level_definition_needs_two_blank_lines_around_it(tmp_path, source, code):
    run = _lint(tmp_path, source)
    assert run.returncode != 0
    assert code in run.stdout + run.stderr
