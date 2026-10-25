"""tools/lint.sh fails on the blank lines PEP 8 asks for around definitions (#111), in each directory it lints
(#115)."""

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
METHODS = 'class C:\n    def a(self):\n        return 1\n\n    def b(self):\n        return 2\n'
NESTED = 'def h():\n    x = 1\n\n    def inner():\n        return x\n    return inner\n'
DIRS = ("src", "tests", "tools", "examples")  # the directories tools/lint.sh lints


def _lint(tmp_path, source, where="src"):
    """tools/lint.sh run on a tree holding one module, ``source`` in directory ``where``, beside the repo's own
    pyproject.toml."""
    for d in DIRS:
        (tmp_path / d).mkdir()
    shutil.copy(LINT, tmp_path / "tools" / "lint.sh")
    shutil.copy(os.path.join(ROOT, "pyproject.toml"), tmp_path / "pyproject.toml")
    (tmp_path / where / "module.py").write_text(source)
    env = dict(os.environ, PYTHON=sys.executable)
    return subprocess.run(["bash", str(tmp_path / "tools" / "lint.sh")], env=env, capture_output=True, text=True,
                          timeout=120)


def test_a_clean_module_passes(tmp_path):
    run = _lint(tmp_path, CLEAN + "\n\n" + METHODS + "\n\n" + NESTED)
    assert run.returncode == 0, run.stdout + run.stderr


@pytest.mark.parametrize("source, code", [
    (CLEAN.replace("\n\n\ndef g", "\n\ndef g"), "E302"),  # one blank line before a top-level def
    (CLEAN + "\nX = 1\n", "E305"),  # one blank line after the last one
    (CLEAN.replace("\n\n\ndef g", "\n\n\n\ndef g"), "E303"),  # three
    (METHODS.replace("\n\n    def b", "\n    def b"), "E301"),  # no blank line before a method
    (NESTED.replace("\n\n    def inner", "\n    def inner"), "E306"),  # nor before a nested definition
], ids=["one blank line before", "one blank line after", "three blank lines", "a method", "a nested definition"])
def test_definitions_need_their_blank_lines(tmp_path, source, code):
    run = _lint(tmp_path, source)
    assert run.returncode != 0
    assert code in run.stdout + run.stderr


@pytest.mark.parametrize("where", DIRS)
def test_one_run_reports_pylints_and_pycodestyles_findings_in_each_directory(tmp_path, where):
    run = _lint(tmp_path, "import os\n" + CLEAN.replace("\n\n\ndef g", "\n\ndef g"), where)
    assert run.returncode != 0
    assert "unused-import" in run.stdout and "E302" in run.stdout
