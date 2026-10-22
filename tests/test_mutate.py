"""tools/mutate.py: each mutant applied alone, the selection run, the file restored byte for byte (#101)."""

import os
import signal
import subprocess
import sys
import time

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MUTATE = os.path.join(ROOT, "tools", "mutate.py")

CALC = 'def clamp(x):\n    if x < 0:\n        return 0\n    return x\n'
TEST_CALC = ('from calc import clamp\n\n\ndef test_negative_is_clamped():\n    assert clamp(-1) == 0\n\n\n'
             'def test_positive_is_kept():\n    assert clamp(2) == 2\n')


def _project(tmp_path, mutants, calc=CALC):
    """A module, its tests and a spec in ``tmp_path``; returns the spec's path."""
    (tmp_path / "calc.py").write_bytes(calc.encode() if isinstance(calc, str) else calc)
    (tmp_path / "test_calc.py").write_text(TEST_CALC)
    spec = tmp_path / "mutants.py"
    spec.write_text(f"FILE = 'calc.py'\nSELECT = ['test_calc.py', '-q', '-p', 'no:cacheprovider']\n"
                    f"MUTANTS = {mutants!r}\n")
    return spec


def _env(**extra):
    """This environment without the bytecode settings a caller (tools/mutate.py itself, say) may have set."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONDONTWRITEBYTECODE", "PYTHONPYCACHEPREFIX")}
    return {**env, **extra}


def _mutate(tmp_path, spec, env=None):
    return subprocess.run([sys.executable, MUTATE, str(spec)], cwd=tmp_path, capture_output=True, text=True,
                          timeout=120, check=False, env=env or _env())


def test_a_mutant_the_tests_catch_is_red(tmp_path):
    run = _mutate(tmp_path, _project(tmp_path, [("guard", "    if x < 0:", "    if False:")]))
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stdout == "guard: red (test_calc.py::test_negative_is_clamped)\n"
    assert (tmp_path / "calc.py").read_text() == CALC


@pytest.mark.parametrize("equivalent, code", [(False, 1), (True, 0)])
def test_a_survivor_fails_the_run_unless_marked_equivalent(tmp_path, equivalent, code):
    """clamp(0) is never tested, so moving the boundary survives."""
    mark = ("clamp(0) is 0 either way",) if equivalent else ()
    guard = ("guard", "    if x < 0:", "    if False:") + mark  # wrongly marked when equivalent: still red
    run = _mutate(tmp_path, _project(tmp_path, [guard, ("boundary", "x < 0", "x <= 0") + mark]))
    assert run.returncode == code, run.stdout + run.stderr
    assert run.stdout.splitlines() == (
        ["guard: red (test_calc.py::test_negative_is_clamped), though marked equivalent",
         "boundary: survived, marked equivalent (clamp(0) is 0 either way)"] if equivalent
        else ["guard: red (test_calc.py::test_negative_is_clamped)", "boundary: survived"])
    assert (tmp_path / "calc.py").read_text() == CALC


@pytest.mark.parametrize("bad, why", [
    (("bad", "x > 0", "y"), "bad: old matches 0 times; it must match exactly once"),
    (("bad", "x", "y"), "bad: old matches 3 times; it must match exactly once"),
    (("bad", "return x", "return x"), "bad: new is the same as old"),
    (("guard", "return x", "return 1"), "guard: the name is used twice"),
], ids=["no match", "three matches", "no change", "a name twice"])
def test_a_spec_that_cannot_run_is_refused_before_any_test_runs(tmp_path, bad, why):
    spec = _project(tmp_path, [("guard", "    if x < 0:", "    if False:"), bad])
    (tmp_path / "conftest.py").write_text("open('ran', 'w').close()\n")  # pytest leaves this mark when it starts
    run = _mutate(tmp_path, spec)
    assert run.returncode == 2
    assert (run.stdout, run.stderr) == ("", f"mutate.py: refused, no test run:\n  {why}\n")
    assert not (tmp_path / "ran").exists() and (tmp_path / "calc.py").read_text() == CALC


@pytest.mark.parametrize("spec, why", [
    ("FILE = 'calc.py'\nMUTANTS = []\n", "must define FILE, SELECT and MUTANTS; it lacks SELECT"),
    ("FILE = 'calc.py'\nSELECT = []\nMUTANTS = [('guard', 'x < 0')]\n",
     "MUTANTS[0] must be (name, old, new) or (name, old, new, reason), all strings; got ('guard', 'x < 0')"),
    ("FILE = 'calc.py'\nSELECT = [\n", "mutants.py cannot be run: SyntaxError"),
], ids=["no SELECT", "a short mutant", "a syntax error"])
def test_a_malformed_spec_is_refused(tmp_path, spec, why):
    (tmp_path / "mutants.py").write_text(spec)
    run = _mutate(tmp_path, tmp_path / "mutants.py")
    assert run.returncode == 2 and why in run.stderr


def test_the_file_is_restored_byte_for_byte_even_when_pytest_cannot_collect(tmp_path):
    calc = CALC.replace("\n", "\r\n").replace("def clamp", "# Grenzwert: °C\r\ndef clamp").encode()[:-2]
    spec = _project(tmp_path, [("syntax", "        return 0", "        return ("),
                               ("guard", "    if x < 0:", "    if False:")], calc=calc)
    os.utime(tmp_path / "calc.py", ns=(1_000_000_000_123, 1_000_000_000_456))
    run = _mutate(tmp_path, spec)
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stdout.splitlines()[0] == "syntax: red (test_calc.py)"
    assert (tmp_path / "calc.py").read_bytes() == calc
    assert (tmp_path / "calc.py").stat().st_mtime_ns == 1_000_000_000_456  # nothing looks changed to make or an editor


def test_a_failing_test_is_named_in_full(tmp_path):
    spec = _project(tmp_path, [("guard", "    if x < 0:", "    if False:")])
    (tmp_path / "test_calc.py").write_text("import pytest\n\nfrom calc import clamp\n\n\n"
                                           "@pytest.mark.parametrize('x', [-1], ids=['a negative x'])\n"
                                           "def test_clamp(x):\n    assert clamp(x) == 0\n")
    run = _mutate(tmp_path, spec)
    assert run.stdout == "guard: red (test_calc.py::test_clamp[a negative x])\n", run.stdout + run.stderr


@pytest.mark.parametrize("prefix", [False, True], ids=["__pycache__", "PYTHONPYCACHEPREFIX"])
def test_cached_bytecode_cannot_hide_a_mutant(tmp_path, prefix):
    """A same-size mutant written in the second its file was compiled looks unchanged to the cache; force that."""
    spec = _project(tmp_path, [("guard", "    if x < 0:", "    if False:")])
    env = _env(PYTHONPYCACHEPREFIX=str(tmp_path / "prefix")) if prefix else _env()
    subprocess.run([sys.executable, "-c", "import py_compile; py_compile.compile('calc.py', invalidation_mode="
                    "py_compile.PycInvalidationMode.UNCHECKED_HASH)"], cwd=tmp_path, env=env, check=True)
    run = _mutate(tmp_path, spec, env)
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stdout == "guard: red (test_calc.py::test_negative_is_clamped)\n"
    # Nor does a run leave a mutant's bytecode behind, for a later run to load.
    assert not list(tmp_path.rglob("calc.*.pyc"))


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM], ids=["Ctrl-C", "SIGTERM"])
def test_an_interrupted_run_restores_the_file(tmp_path, sig):
    calc = "MUTATED = False\n"
    (tmp_path / "calc.py").write_text(calc)
    (tmp_path / "test_calc.py").write_text("import time\n\nimport calc\n\n\ndef test_slow():\n"
                                           "    if calc.MUTATED:\n        time.sleep(60)\n")
    spec = tmp_path / "mutants.py"
    spec.write_text("FILE = 'calc.py'\nSELECT = ['test_calc.py', '-q', '-p', 'no:cacheprovider']\n"
                    "MUTANTS = [('slow', 'False', 'True')]\n")
    with subprocess.Popen([sys.executable, MUTATE, str(spec)], cwd=tmp_path, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, env=_env()) as tool:
        deadline = time.monotonic() + 60
        while (tmp_path / "calc.py").read_text() == calc:  # until the mutant is in place and its test is running
            assert tool.poll() is None and time.monotonic() < deadline
            time.sleep(0.05)
        time.sleep(1.0)
        tool.send_signal(sig)
        _, err = tool.communicate(timeout=60)
    assert tool.returncode == 130
    assert err == "mutate.py: interrupted during slow; calc.py is restored\n"
    assert (tmp_path / "calc.py").read_text() == calc


def test_a_selection_that_fails_without_mutants_is_refused(tmp_path):
    spec = _project(tmp_path, [("guard", "    if x < 0:", "    if False:")], calc=CALC.replace("return 0", "return -1"))
    run = _mutate(tmp_path, spec)
    assert run.returncode == 2
    assert run.stdout == ""
    assert run.stderr.startswith("mutate.py: the selection fails before any mutant (pytest exit 1); fix it first:\n")
