"""tools/mutate.py: each mutant applied alone, the selection run, the file restored byte for byte (#101)."""

import contextlib
import importlib.util
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
SELECT = ['test_calc.py', '-q', '-p', 'no:cacheprovider']
GUARD = ("guard", "    if x < 0:", "    if False:")  # the tests catch it: clamp(-1) is no longer 0
CAUGHT = "guard: red (test_calc.py::test_negative_is_clamped)"
# A test that sleeps once calc.MUTATED is True, or always (SLOW = True): a run to interrupt.
SLOW_CALC = "MUTATED = False\n"
SLOW_TEST = ("import time\n\nimport calc\n\nSLOW = {}\n\n\ndef test_slow():\n"
             "    if calc.MUTATED or SLOW:\n        time.sleep(60)\n")
SLOW = ("slow", "False", "True")


def _project(tmp_path, mutants, calc=CALC, tests=TEST_CALC, extra=""):
    """A module, its tests and a spec (``extra`` appended) in ``tmp_path``; returns the spec's path."""
    (tmp_path / "calc.py").write_bytes(calc.encode() if isinstance(calc, str) else calc)
    (tmp_path / "test_calc.py").write_text(tests)
    spec = tmp_path / "mutants.py"
    spec.write_text(f"FILE = 'calc.py'\nSELECT = {SELECT!r}\nMUTANTS = {mutants!r}\n{extra}")
    return spec


def _env(**extra):
    """This environment without the bytecode settings a caller (tools/mutate.py itself, say) may have set."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONDONTWRITEBYTECODE", "PYTHONPYCACHEPREFIX")}
    return {**env, **extra}


def _mutate(tmp_path, spec, env=None):
    return subprocess.run([sys.executable, MUTATE, str(spec)], cwd=tmp_path, capture_output=True, text=True,
                          timeout=180, check=False, env=env or _env())


@contextlib.contextmanager
def _running(tmp_path, spec, until):
    """A run of the tool in the background, yielded once ``until()`` holds (the run must not end first)."""
    with subprocess.Popen([sys.executable, MUTATE, str(spec)], cwd=tmp_path, stdout=subprocess.PIPE,
                          stderr=subprocess.PIPE, text=True, env=_env()) as tool:
        try:
            deadline = time.monotonic() + 60
            while not until():
                assert tool.poll() is None and time.monotonic() < deadline
                time.sleep(0.05)
            time.sleep(1.0)  # well inside pytest's run
            yield tool
        finally:
            if tool.poll() is None:
                tool.kill()


def test_a_mutant_the_tests_catch_is_red(tmp_path):
    run = _mutate(tmp_path, _project(tmp_path, [GUARD]))
    assert (run.returncode, run.stdout, run.stderr) == (0, CAUGHT + "\n", "")
    assert (tmp_path / "calc.py").read_text() == CALC


@pytest.mark.parametrize("equivalent, code", [(False, 1), (True, 0)])
def test_a_survivor_fails_the_run_unless_marked_equivalent(tmp_path, equivalent, code):
    """clamp(0) is never tested, so moving the boundary survives."""
    mark = ("clamp(0) is 0 either way",) if equivalent else ()
    guard = GUARD + mark  # wrongly marked when equivalent: still red
    run = _mutate(tmp_path, _project(tmp_path, [guard, ("boundary", "x < 0", "x <= 0") + mark]))
    assert run.returncode == code, run.stdout + run.stderr
    assert run.stdout.splitlines() == (
        [CAUGHT + ", though marked equivalent", "boundary: survived, marked equivalent (clamp(0) is 0 either way)"]
        if equivalent else [CAUGHT, "boundary: survived"])
    assert (tmp_path / "calc.py").read_text() == CALC


@pytest.mark.parametrize("extra, why, calc", [
    ([("bad", "x > 0", "y")], "bad: old matches 0 times; it must match exactly once", CALC),
    ([("bad", "x", "y")], "bad: old matches 3 times; it must match exactly once", CALC),
    ([("bad", " " * 6, "y")], "bad: old matches 3 times; it must match exactly once", CALC),  # overlapping
    ([("bad", "    return 0\n    return x", "y")],
     "bad: old matches 0 times; it must match exactly once (the file has CRLF line endings)",
     CALC.replace("\n", "\r\n")),
    ([("bad", "return x", "return x")], "bad: new is the same as old", CALC),
    ([("bad", "return x", "return \udc80")], "bad: old or new cannot be encoded as UTF-8", CALC),
    ([("bad", "return x", "return 1", "")], "bad: the reason is empty; give one, or leave it out", CALC),
    ([("a\nb", "return x", "return 1")], "'a\\nb': a name must be one non-empty line", CALC),
    ([GUARD], "guard: the name is used 2 times", CALC),
    ([GUARD, GUARD], "guard: the name is used 3 times", CALC),  # one line, not one per repeat
], ids=["no match", "three matches", "overlapping matches", "CRLF", "no change", "not UTF-8", "an empty reason",
        "a name of two lines", "a name twice", "a name three times"])
def test_a_mutant_that_cannot_run_is_refused_before_any_test_runs(tmp_path, extra, why, calc):
    spec = _project(tmp_path, [GUARD, *extra], calc=calc)
    (tmp_path / "conftest.py").write_text("open('ran', 'w').close()\n")  # pytest leaves this mark when it starts
    run = _mutate(tmp_path, spec)
    assert (run.returncode, run.stdout, run.stderr) == (2, "", f"mutate.py: refused, no test run:\n  {why}\n")
    assert not (tmp_path / "ran").exists() and (tmp_path / "calc.py").read_bytes() == calc.encode()


@pytest.mark.parametrize("spec, why", [
    ("FILE = 'calc.py'\nMUTANTS = []\n", "mutants.py must define FILE, SELECT and MUTANTS; it lacks SELECT"),
    (f"FILE = 'calc.py'\nSELECT = []\nMUTANTS = [{GUARD[:2]!r}]\n",
     "MUTANTS[0] must be (name, old, new) or (name, old, new, reason), all strings; got ('guard', '    if x < 0:')"),
    ("FILE = 'calc.py'\nSELECT = []\nMUTANTS = []\n", "MUTANTS is empty: give at least one mutant"),
    (f"FILE = 'calc.py'\nSELECT = 'test_calc.py'\nMUTANTS = [{GUARD!r}]\n",
     "SELECT must be a list of pytest arguments (strings), got 'test_calc.py'"),
    (f"FILE = 'nope.py'\nSELECT = []\nMUTANTS = [{GUARD!r}]\n", "FILE nope.py does not exist"),
    (f"FILE = 'calc.py'\nSELECT = []\nMUTANTS = [{GUARD!r}]\nTIMEOUT = 0\n",
     "TIMEOUT must be a positive number of seconds, got 0"),
    ("FILE = 'calc.py'\nSELECT = [\n", "mutants.py cannot be run: SyntaxError: "),
], ids=["no SELECT", "a short mutant", "no mutants", "SELECT a string", "no FILE", "a zero TIMEOUT", "a syntax error"])
def test_a_malformed_spec_is_refused(tmp_path, spec, why):
    (tmp_path / "calc.py").write_text(CALC)
    (tmp_path / "mutants.py").write_text(spec)
    run = _mutate(tmp_path, "mutants.py")
    assert (run.returncode, run.stdout) == (2, "")
    assert run.stderr.startswith(f"mutate.py: {why}") and run.stderr.endswith("\n") and run.stderr.count("\n") == 1


def test_the_file_is_restored_byte_for_byte_even_when_pytest_cannot_collect(tmp_path):
    calc = CALC.replace("\n", "\r\n").replace("def clamp", "# Grenzwert: °C\r\ndef clamp").encode()[:-2]
    spec = _project(tmp_path, [("syntax", "        return 0", "        return ("), GUARD], calc=calc)
    os.utime(tmp_path / "calc.py", ns=(1_000_000_000_123, 1_000_000_000_456))
    run = _mutate(tmp_path, spec)
    assert run.returncode == 0, run.stdout + run.stderr
    assert run.stdout.splitlines()[0] == "syntax: red (test_calc.py)"
    assert (tmp_path / "calc.py").read_bytes() == calc
    assert (tmp_path / "calc.py").stat().st_mtime_ns == 1_000_000_000_456  # nothing looks changed to make or an editor


def test_a_failing_test_is_named_in_full(tmp_path):
    spec = _project(tmp_path, [GUARD], tests=("import pytest\n\nfrom calc import clamp\n\n\n"
                                              "@pytest.mark.parametrize('x', [-1], ids=['a negative x'])\n"
                                              "def test_clamp(x):\n    assert clamp(x) == 0\n"))
    run = _mutate(tmp_path, spec)
    assert run.stdout == "guard: red (test_calc.py::test_clamp[a negative x])\n", run.stdout + run.stderr


def test_a_run_without_a_summary_names_pytests_exit_status(tmp_path):
    spec = _project(tmp_path, [GUARD])
    spec.write_text(spec.read_text().replace("'-q'", "'-q', '-rN'"))  # no short test summary to name a test
    run = _mutate(tmp_path, spec)
    assert run.stdout == "guard: red (pytest exit 1)\n", run.stdout + run.stderr


@pytest.mark.parametrize("prefix", [False, True], ids=["__pycache__", "PYTHONPYCACHEPREFIX"])
def test_cached_bytecode_cannot_hide_a_mutant(tmp_path, prefix):
    """A same-size mutant written in the second its file was compiled looks unchanged to the cache; force that."""
    spec = _project(tmp_path, [GUARD])
    env = _env(PYTHONPYCACHEPREFIX=str(tmp_path / "prefix")) if prefix else _env()
    subprocess.run([sys.executable, "-c", "import py_compile; py_compile.compile('calc.py', invalidation_mode="
                    "py_compile.PycInvalidationMode.UNCHECKED_HASH)"], cwd=tmp_path, env=env, check=True)
    run = _mutate(tmp_path, spec, env)
    assert (run.returncode, run.stdout) == (0, CAUGHT + "\n"), run.stderr
    # Nor does a run leave a mutant's bytecode behind, for a later run to load.
    assert not list(tmp_path.rglob("calc.*.pyc"))


def test_a_mutant_whose_tests_hang_times_out_red_and_the_rest_still_run(tmp_path):
    """Decided on #101: a hang fails in CI too."""
    spec = _project(tmp_path, [SLOW, ("quick", "MUTATED = False", "MUTATED = False  # unchanged")],
                    calc=SLOW_CALC, tests=SLOW_TEST.format(False), extra="TIMEOUT = 3\n")
    started = time.monotonic()
    run = _mutate(tmp_path, spec)
    assert time.monotonic() - started < 50
    assert run.stdout.splitlines() == ["slow: red (timed out after 3 s)", "quick: survived"], run.stderr
    assert run.returncode == 1 and (tmp_path / "calc.py").read_text() == SLOW_CALC


def _load_tool():
    loader = importlib.util.spec_from_file_location("mutate", MUTATE)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


@pytest.mark.parametrize("baseline_s, given, want", [(0.4, None, 60.0), (30.0, None, 300.0), (30.0, 5, 5.0)])
def test_the_default_timeout_is_ten_times_the_unmutated_run_and_at_least_a_minute(baseline_s, given, want):
    """Decided on #101; TIMEOUT in the spec replaces it."""
    assert _load_tool().timeout(baseline_s, given) == want


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
@pytest.mark.parametrize("sig", [signal.SIGINT, signal.SIGTERM], ids=["Ctrl-C", "SIGTERM"])
def test_an_interrupted_run_restores_the_file(tmp_path, sig):
    spec = _project(tmp_path, [SLOW], calc=SLOW_CALC, tests=SLOW_TEST.format(False))
    with _running(tmp_path, spec, lambda: (tmp_path / "calc.py").read_text() != SLOW_CALC) as tool:
        tool.send_signal(sig)
        _, err = tool.communicate(timeout=60)
    assert (tool.returncode, err) == (130, "mutate.py: interrupted during slow; calc.py is restored\n")
    assert (tmp_path / "calc.py").read_text() == SLOW_CALC


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signals")
def test_an_interrupt_before_any_mutant_says_the_file_is_unchanged(tmp_path):
    spec = _project(tmp_path, [SLOW], calc=SLOW_CALC, tests=SLOW_TEST.format(True))
    (tmp_path / "conftest.py").write_text("open('ran', 'w').close()\n")
    with _running(tmp_path, spec, (tmp_path / "ran").exists) as tool:
        tool.send_signal(signal.SIGINT)
        _, err = tool.communicate(timeout=60)
    assert (tool.returncode, err) == (130, "mutate.py: interrupted before any mutant; calc.py is unchanged\n")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file locks")
def test_a_second_run_on_the_same_file_is_refused(tmp_path):
    """It would read the first run's mutant as the original, and write it back at the end."""
    spec = _project(tmp_path, [SLOW], calc=SLOW_CALC, tests=SLOW_TEST.format(False))
    with _running(tmp_path, spec, lambda: (tmp_path / "calc.py").read_text() != SLOW_CALC):
        second = _mutate(tmp_path, spec)
    assert (second.returncode, second.stdout) == (2, "")
    assert second.stderr == "mutate.py: another mutate.py run is mutating calc.py; wait for it to end\n"


def test_a_selection_that_fails_without_mutants_is_refused(tmp_path):
    run = _mutate(tmp_path, _project(tmp_path, [GUARD], calc=CALC.replace("return 0", "return -1")))
    assert (run.returncode, run.stdout) == (2, "")
    assert run.stderr.startswith("mutate.py: the selection fails before any mutant (pytest exit 1); fix it first:\n")


def test_a_selection_that_times_out_without_mutants_is_refused(tmp_path):
    spec = _project(tmp_path, [SLOW], calc=SLOW_CALC, tests=SLOW_TEST.format(True), extra="TIMEOUT = 2\n")
    run = _mutate(tmp_path, spec)
    assert (run.returncode, run.stdout) == (2, "")
    assert run.stderr == ("mutate.py: the selection times out before any mutant (TIMEOUT = 2 s); raise TIMEOUT or "
                          "fix it first\n")
