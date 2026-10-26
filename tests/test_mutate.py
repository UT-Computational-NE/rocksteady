"""tools/mutate.py: each mutant applied alone, the selection run, the file restored byte for byte (#101)."""

import contextlib
import errno
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
_SLEEPING_TEST = ("import time\n\nimport calc\n\nSLOW = {{}}\n\n\ndef test_slow():\n"
                  "    if calc.MUTATED or SLOW:\n        time.sleep({})\n")
SLOW_TEST = _SLEEPING_TEST.format(60)
HANG_TEST = _SLEEPING_TEST.format(120)  # sleeps twice as long as the hang test's bound on the whole run
SLOW = ("slow", "False", "True")
# A conftest.py that makes every pytest start-up in its project 3 s slower, as on a loaded machine.
SLOW_START = "import time\n\ntime.sleep(3)\n"


def _spec(spec, file, mutants, extra=""):
    """A spec at ``spec`` that mutates ``file`` (``extra`` appended); returns its path."""
    spec.write_text(f"FILE = {file!r}\nSELECT = {SELECT!r}\nMUTANTS = {mutants!r}\n{extra}")
    return spec


def _project(tmp_path, mutants, calc=CALC, tests=TEST_CALC, extra=""):
    """A module, its tests and a spec (``extra`` appended) in ``tmp_path``; returns the spec's path."""
    (tmp_path / "calc.py").write_bytes(calc.encode() if isinstance(calc, str) else calc)
    (tmp_path / "test_calc.py").write_text(tests)
    return _spec(tmp_path / "mutants.py", "calc.py", mutants, extra)


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
    ([("bad", "x > 0", "y")], "bad: old matches 0 times; it must match exactly once",  # one line: CRLF is no cause
     CALC.replace("\n", "\r\n")),
    ([("bad", "return x", "return x")], "bad: new is the same as old", CALC),
    ([("bad", "return x", "return \udc80")], "bad: old or new cannot be encoded as UTF-8", CALC),
    ([("bad", "return x", "return 1", "")], "bad: the reason is empty; give one, or leave it out", CALC),
    ([("a\nb", "return x", "return 1")], "'a\\nb': a name must be one non-empty line", CALC),
    ([("a\rb", "return x", "return 1")], "'a\\rb': a name must be one non-empty line", CALC),
    ([GUARD], "guard: the name is used 2 times", CALC),
    ([GUARD, GUARD], "guard: the name is used 3 times", CALC),  # one line, not one per repeat
], ids=["no match", "three matches", "overlapping matches", "CRLF", "one line in CRLF", "no change", "not UTF-8",
        "an empty reason", "a name of two lines", "a name of two CR lines", "a name twice", "a name three times"])
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
    (f"FILE = 'calc.py'\nSELECT = []\nMUTANTS = [{GUARD!r}]\nTIMEOUT = True\n",
     "TIMEOUT must be a positive number of seconds, got True"),
    (f"FILE = 'calc.py'\nSELECT = []\nMUTANTS = [{list(GUARD)!r}]\n",
     "MUTANTS[0] must be (name, old, new) or (name, old, new, reason), all strings; got ['guard',"),
    ("FILE = 'calc.py'\nSELECT = [\n", "mutants.py cannot be run: SyntaxError: "),
], ids=["no SELECT", "a short mutant", "no mutants", "SELECT a string", "no FILE", "a zero TIMEOUT", "TIMEOUT True",
        "a list mutant", "a syntax error"])
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
                    calc=SLOW_CALC, tests=HANG_TEST.format(False), extra="TIMEOUT = 20\n")
    (tmp_path / "conftest.py").write_text(SLOW_START)  # the unmutated run and quick still end within TIMEOUT
    started = time.monotonic()
    run = _mutate(tmp_path, spec)
    assert time.monotonic() - started < 60  # a hang not stopped would take over 120 s
    assert run.stdout.splitlines() == ["slow: red (timed out after 20 s)", "quick: survived"], run.stderr
    assert run.returncode == 1 and (tmp_path / "calc.py").read_text() == SLOW_CALC


def _load_tool():
    loader = importlib.util.spec_from_file_location("mutate", MUTATE)
    module = importlib.util.module_from_spec(loader)
    loader.loader.exec_module(module)
    return module


@pytest.mark.parametrize("given, limits", [(None, [None, 300.0]), (5, [5, 5.0])])
def test_the_mutants_get_the_timeout_of_the_measured_unmutated_run(tmp_path, monkeypatch, given, limits):
    """The unmutated run takes 30 s on a stub clock: a mutant may take ten times that, or TIMEOUT."""
    tool = _load_tool()
    spec = _project(tmp_path, [GUARD], extra="" if given is None else f"TIMEOUT = {given}\n")
    seen = []
    monkeypatch.setattr(tool, "_run", lambda select, limit: (seen.append(limit), (0 if not seen[1:] else 1, ""))[1])
    clock = iter([100.0, 130.0])
    monkeypatch.setattr(tool.time, "monotonic", lambda: next(clock))
    monkeypatch.setattr(tool.signal, "signal", lambda *args: None)  # leave this process's handlers alone
    monkeypatch.chdir(tmp_path)
    assert tool.main([str(spec)]) == 0
    assert seen == limits


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
@pytest.mark.parametrize("name", ["calc.py", "./calc.py", "link.py", "hard.py"],
                         ids=["same path", "a ./ path", "a symlink", "a hard link"])
def test_a_second_run_on_the_same_file_is_refused(tmp_path, name):
    """
    It would read the first run's mutant as the original, and write it back at the end. The second run starts
    after the first run's first restore: the lock must hold on the file that mutants and restores write.
    """
    restored_first = ("restored first", "MUTATED = False", "MUTATED = False  # unchanged")
    spec = _project(tmp_path, [restored_first, SLOW], calc=SLOW_CALC, tests=SLOW_TEST.format(False))
    (tmp_path / "link.py").symlink_to("calc.py")
    os.link(tmp_path / "calc.py", tmp_path / "hard.py")  # the same inode: realpath leads from neither name to the other
    second_spec = _spec(tmp_path / "second.py", name, [restored_first, SLOW])
    with _running(tmp_path, spec, lambda: (tmp_path / "calc.py").read_text() == "MUTATED = True\n"):
        second = _mutate(tmp_path, second_spec)
    assert (second.returncode, second.stdout) == (2, "")
    assert second.stderr == f"mutate.py: another mutate.py run is mutating {name}; wait for it to end\n"


@pytest.mark.skipif(sys.platform == "win32" or os.geteuid() == 0, reason="POSIX file locks; root writes anyway")
def test_a_file_the_tool_cannot_write_is_refused_before_any_test_runs(tmp_path):
    """The lock opens the file for writing: over NFS an exclusive flock needs that, and every mutant writes it."""
    spec = _project(tmp_path, [GUARD])
    (tmp_path / "conftest.py").write_text("open('ran', 'w').close()\n")
    (tmp_path / "calc.py").chmod(0o444)
    run = _mutate(tmp_path, spec)
    assert (run.returncode, run.stdout) == (2, "")
    assert run.stderr == "mutate.py: cannot lock calc.py: [Errno 13] Permission denied: 'calc.py'\n"
    assert not (tmp_path / "ran").exists() and (tmp_path / "calc.py").read_text() == CALC


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX file locks")
def test_a_lock_that_fails_for_another_reason_is_not_called_another_run(tmp_path, monkeypatch, capsys):
    """Only a lock another process holds is contention; NFS, say, can fail the flock itself."""
    import fcntl  # POSIX only

    def unsupported(fd, operation):
        raise OSError(errno.EBADF, os.strerror(errno.EBADF))

    tool = _load_tool()
    spec = _project(tmp_path, [GUARD])
    monkeypatch.setattr(fcntl, "flock", unsupported)
    monkeypatch.setattr(tool.signal, "signal", lambda *args: None)  # leave this process's handlers alone
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as stop:
        tool.main([str(spec)])
    assert stop.value.code == 2
    assert capsys.readouterr().err == f"mutate.py: cannot lock calc.py: [Errno 9] {os.strerror(errno.EBADF)}\n"


def test_a_run_leaves_nothing_in_the_temp_directory(tmp_path):
    temp = tmp_path / "temp"
    temp.mkdir()
    run = _mutate(tmp_path, _project(tmp_path, [GUARD]), _env(TMPDIR=str(temp)))
    assert (run.returncode, run.stdout) == (0, CAUGHT + "\n"), run.stderr
    assert not list(temp.iterdir())


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


CHILD_TEST = ("import os\nimport subprocess\nimport sys\nimport time\n\nimport calc\n\n\ndef test_child():\n"
              "    if calc.MUTATED:\n"
              "        subprocess.Popen([sys.executable, '-c', 'import os, signal, time; "
              "signal.signal(signal.SIGTERM, signal.SIG_IGN); "  # only SIGKILL ends it
              "open(\"child.pid\", \"w\").write(str(os.getpid())); time.sleep(60)'])\n"
              "        {}\n")


def _gone(pid, wait_s=10.0):
    """Whether process ``pid`` has ended (and been reaped), within ``wait_s`` seconds."""
    deadline = time.monotonic() + wait_s
    while time.monotonic() < deadline:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return True
        time.sleep(0.1)
    return False


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
@pytest.mark.parametrize("how", ["timeout", "Ctrl-C", "left behind"])
def test_a_process_a_test_started_ends_with_the_run(tmp_path, how):
    """Decided on #101: a timed-out mutant's run is killed, all of it; an interrupted or finished one too."""
    wait = "while not os.path.exists('child.pid'): time.sleep(0.05)" if how == "left behind" else "time.sleep(60)"
    spec = _project(tmp_path, [SLOW], calc=SLOW_CALC, tests=CHILD_TEST.format(wait),
                    extra="TIMEOUT = 20\n" if how == "timeout" else "")
    if how == "timeout":
        # The unmutated run still ends, and the child still starts, within TIMEOUT.
        (tmp_path / "conftest.py").write_text(SLOW_START)
        run = _mutate(tmp_path, spec)
        assert run.stdout == "slow: red (timed out after 20 s)\n", run.stderr
    elif how == "left behind":  # the test passes and leaves its child running
        run = _mutate(tmp_path, spec)
        assert run.stdout == "slow: survived\n", run.stderr
    else:
        with _running(tmp_path, spec, (tmp_path / "child.pid").exists) as tool:
            tool.send_signal(signal.SIGINT)
            tool.communicate(timeout=60)
        assert tool.returncode == 130
    pid = int((tmp_path / "child.pid").read_text())
    assert _gone(pid), f"the child {pid} outlived the run"
    assert (tmp_path / "calc.py").read_text() == SLOW_CALC


def _dump(when):
    import glob, threading
    with open("/tmp/diag.txt", "a") as out:
        out.write(f"--- {when} t={time.monotonic():.6f} handler={signal.getsignal(signal.SIGINT)!r} "
                  f"pythreads={[(t.name, t.daemon) for t in threading.enumerate()]}\n")
        for st in sorted(glob.glob("/proc/self/task/*/status")):
            task = st.rsplit("/", 1)[0]
            fields = dict(l.split(":", 1) for l in open(st) if ":" in l)
            try:
                sysc = open(task + "/syscall").read().split()[0]
            except OSError as e:
                sysc = repr(e)
            try:
                wchan = open(task + "/wchan").read()
            except OSError as e:
                wchan = repr(e)
            out.write(task.rsplit("/", 1)[1] + " " + " ".join(f"{k}={fields[k].strip()}" for k in
                      ("Name", "State", "SigPnd", "ShdPnd", "SigBlk", "SigIgn", "SigCgt")) +
                      f" syscall={sysc} wchan={wchan}\n")


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX signal masks")
def test_a_signal_during_the_restore_waits_for_it_to_finish(tmp_path, monkeypatch):
    """A second Ctrl-C between the truncating open and the write would leave the file empty."""
    tool = _load_tool()
    path = tmp_path / "calc.py"
    path.write_text("mutated\n")
    real_open = open

    def interrupted_open(file, mode="r", *args, **kwargs):
        handle = real_open(file, mode, *args, **kwargs)
        if "w" in mode:
            os.kill(os.getpid(), signal.SIGINT)  # lands between the truncation and the write
            kill_t.append(time.monotonic())
        return handle

    monkeypatch.setattr(tool, "open", interrupted_open, raising=False)
    raised = False
    kill_t = []
    try:
        tool._restore(str(path), CALC.encode(), os.stat(path))
    except KeyboardInterrupt:
        raised = True
    end_t = time.monotonic()
    late = "n/a"
    if not raised:
        late = "none in 2 s"
        try:
            while time.monotonic() < end_t + 2:
                pass
        except KeyboardInterrupt:
            late = f"{(time.monotonic() - end_t) * 1e6:.0f} us after the restore"
    since = f"{(end_t - kill_t[0]) * 1e6:.0f} us" if kill_t else "no return from kill"
    _dump(f"after restore kill->end={since} raised={raised} content_ok={path.read_text() == CALC} late={late}")
    assert raised
    assert path.read_text() == CALC


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX process groups")
@pytest.mark.parametrize("stubborn", [False, True], ids=["plain", "a child ignoring SIGTERM"])
def test_a_nested_run_in_a_timed_out_mutant_restores_its_own_file(tmp_path, stubborn):
    """
    The group gets SIGTERM before SIGKILL, so a run of the tool that a mutant's test started restores its file.
    Sent SIGTERM, that run kills its own group at once, so a child that ignores SIGTERM cannot hold it up.
    """
    inner = tmp_path / "inner"
    inner.mkdir()
    tests = CHILD_TEST.format("time.sleep(60)") if stubborn else SLOW_TEST.format(False)
    inner_spec = _project(inner, [SLOW], calc=SLOW_CALC, tests=tests)
    nested = ("import subprocess\nimport sys\nimport time\n\nimport calc\n\n\ndef test_nested():\n"
              "    if calc.MUTATED:\n"
              f"        subprocess.Popen([sys.executable, {MUTATE!r}, {str(inner_spec)!r}], cwd={str(inner)!r})\n"
              "        time.sleep(60)\n")
    spec = _project(tmp_path, [SLOW], calc=SLOW_CALC, tests=nested, extra="TIMEOUT = 15\n")
    run = _mutate(tmp_path, spec)
    assert run.stdout == "slow: red (timed out after 15 s)\n", run.stderr
    deadline = time.monotonic() + 10
    while (inner / "calc.py").read_text() != SLOW_CALC and time.monotonic() < deadline:
        time.sleep(0.1)
    assert (inner / "calc.py").read_text() == SLOW_CALC
    if stubborn:
        assert _gone(int((inner / "child.pid").read_text()))
