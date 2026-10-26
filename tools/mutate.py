"""
Mutation probe: break a file one mutant at a time and check that the tests notice (#101).

    python tools/mutate.py SPEC.py

``SPEC.py`` is a Python file (keep it in your scratchpad, outside the repo) that defines:

- ``FILE``: the file to mutate, relative to the current directory.
- ``SELECT``: a list of the pytest arguments to run for each mutant, e.g. ``["tests/test_optimal.py", "-q", "-x"]``.
- ``MUTANTS``: a list of ``(name, old, new)`` tuples, or ``(name, old, new, reason)`` to mark a mutant
  equivalent (it changes no behaviour the tests could see, e.g. only speed). ``old`` must occur exactly once
  in ``FILE``, overlapping occurrences counted; triple-quoted strings carry multi-line code with no escaping.
- ``TIMEOUT`` (optional): seconds before a mutant's run is stopped. Default: ten times the unmutated run's
  time, but at least 60 s.

For each mutant in turn, the tool replaces ``old`` with ``new``, runs ``python -m pytest *SELECT`` with this
interpreter, and writes the file back byte for byte, with its modification time, also when pytest cannot collect
or the run is interrupted (a second Ctrl-C waits for the restore). The file's cached bytecode is deleted first
and the runs write none, so no mutant can run as a cached original. A mutant is red when pytest exits non-zero
or times out (a hanging suite fails in CI too), and survived when it passes. The report, on stdout, has one
line per mutant: red (with the first failing test), survived, or survived and marked equivalent; errors go to
stderr. A second run on the same file, by whatever path it names the file, is refused while one is running (the
lock is on the file itself, opened for writing); run no other tests in the same checkout meanwhile, or they test
the mutant.

pytest runs in its own process group, which the tool ends after each run, so no process a test started outlives
the run. After a timeout or Ctrl-C the group gets SIGTERM and 5 s to end (a run of this tool inside it restores
its own file) before SIGKILL. When the tool is itself sent SIGTERM (by such an outer run, say), it kills its
group at once, so that it restores its file well inside the outer run's 5 s.

Exit status: 0 when every mutant is red or marked equivalent; 1 when one survived unmarked; 2 when the spec is
refused (malformed, a ``FILE`` the tool cannot write or lock, or a mutant that cannot run: an ``old`` matching
zero times or more than once, a repeated name) or the selection fails before any mutant, all checked before a
mutant is written; 130 when interrupted (Ctrl-C or SIGTERM).
"""

import collections
import glob
import importlib.util
import os
import runpy
import signal
import subprocess
import sys
import time
from numbers import Real
from typing import NamedTuple, Optional


class Mutant(NamedTuple):
    name: str
    old: str
    new: str
    reason: Optional[str]  # why it is equivalent; None when it is not marked


def _refuse(message):
    """Stop with exit status 2, as for every spec that cannot run."""
    print(f"mutate.py: {message}", file=sys.stderr)
    raise SystemExit(2)


def _load(path):
    """The spec, checked: ``(FILE, SELECT, [Mutant], TIMEOUT or None)``."""
    try:
        spec = runpy.run_path(path)
    except Exception as e:  # any error in the caller's spec
        _refuse(f"{path} cannot be run: {type(e).__name__}: {e}")
    missing = [k for k in ("FILE", "SELECT", "MUTANTS") if k not in spec]
    if missing:
        _refuse(f"{path} must define FILE, SELECT and MUTANTS; it lacks {', '.join(missing)}")
    file, select, given = spec["FILE"], spec["SELECT"], spec.get("TIMEOUT")
    if not isinstance(file, str) or not os.path.isfile(file):
        _refuse(f"FILE {file} does not exist")
    if not isinstance(select, (list, tuple)) or not all(isinstance(a, str) for a in select):
        _refuse(f"SELECT must be a list of pytest arguments (strings), got {select!r}")
    if given is not None and (isinstance(given, bool) or not isinstance(given, Real) or not given > 0):
        _refuse(f"TIMEOUT must be a positive number of seconds, got {given!r}")
    if not spec["MUTANTS"]:
        _refuse("MUTANTS is empty: give at least one mutant")
    mutants = []
    for i, m in enumerate(spec["MUTANTS"]):
        if not isinstance(m, tuple) or len(m) not in (3, 4) or not all(isinstance(v, str) for v in m):
            _refuse(f"MUTANTS[{i}] must be (name, old, new) or (name, old, new, reason), all strings; got {m!r}")
        mutants.append(Mutant(*m) if len(m) == 4 else Mutant(*m, None))
    return file, list(select), mutants, given


def _occurrences(source, old):
    """How often ``old`` occurs in ``source``, overlapping occurrences counted: once means one place to mutate."""
    count, at = 0, source.find(old)
    while at >= 0:
        count, at = count + 1, source.find(old, at + 1)
    return count


def _refusals(source, mutants):
    """Why the mutants cannot run on ``source`` (bytes), one line per problem."""
    out = []
    uses = collections.Counter(m.name for m in mutants)
    for name, n in uses.items():
        if not name or "\n" in name or "\r" in name:
            out.append(f"{name!r}: a name must be one non-empty line")
        elif n > 1:
            out.append(f"{name}: the name is used {n} times")
    for name, old, new, reason in mutants:
        try:
            count = _occurrences(source, old.encode())
            new.encode()
        except UnicodeEncodeError:
            out.append(f"{name}: old or new cannot be encoded as UTF-8")
            continue
        if count != 1:
            crlf = " (the file has CRLF line endings)" if not count and "\n" in old and b"\r\n" in source else ""
            out.append(f"{name}: old matches {count} times; it must match exactly once{crlf}")
        elif old == new:
            out.append(f"{name}: new is the same as old")
        if reason == "":
            out.append(f"{name}: the reason is empty; give one, or leave it out")
    return out


def timeout(baseline_s, given):
    """Seconds a mutant's run may take: ``TIMEOUT`` when given, else ten times the unmutated run, at least 60."""
    return float(given) if given is not None else max(60.0, 10.0 * baseline_s)


def _lock(path):
    """
    Hold a lock on the file itself for this run (an open file the caller keeps), or refuse when another run holds
    it: the second run would read the first one's mutant as the original. The lock is on the inode, so it also
    holds against a run that names the file by another path (a symlink, a hard link). The file is opened for
    writing, which an exclusive flock over NFS needs; a file the tool cannot write, or lock, is refused as such,
    not as another run. Without fcntl (Windows), no lock.
    """
    try:
        import fcntl  # POSIX only
    except ImportError:
        return None
    try:
        handle = open(path, "r+b")
        fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        _refuse(f"another mutate.py run is mutating {path}; wait for it to end")
    except OSError as e:
        _refuse(f"cannot lock {path}: {e}")
    return handle


def _drop_bytecode(path):
    """
    Delete the file's cached bytecode. The cache trusts a source of the same size and modification second, so a
    same-size mutant written in the second the file was compiled would run as the original. The runs write none.
    """
    stem = os.path.splitext(os.path.basename(path))[0]
    cached = glob.glob(os.path.join(os.path.dirname(os.path.abspath(path)), "__pycache__", stem + ".*.pyc"))
    for pyc in {*cached, importlib.util.cache_from_source(path)}:
        if os.path.exists(pyc):
            os.remove(pyc)


class _Ended(KeyboardInterrupt):
    """SIGTERM: this run is being ended (by an outer run, say), and has only that run's grace to restore."""


def _ended(signum, frame):
    raise _Ended


def _end(proc, grace=5.0):
    """
    End pytest, if it is still running, and every process its tests started (its own process group): SIGTERM,
    so that a nested tools/mutate.py run restores its file, then after ``grace`` seconds (at once, for 0) SIGKILL
    for whatever is left.
    """
    if not hasattr(os, "killpg"):  # Windows: no process groups
        proc.kill()
        proc.wait()
        return
    # The group has ended: ESRCH, or on macOS EPERM when only zombies are left in it.
    ended = (ProcessLookupError, PermissionError)
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ended:
        proc.wait()
        return
    deadline = time.monotonic() + grace
    while time.monotonic() < deadline:  # the whole group, not only pytest: a nested run may still be restoring
        proc.poll()  # reap pytest, so that it does not linger in the group
        try:
            os.killpg(proc.pid, 0)
        except ended:
            break
        time.sleep(0.05)
    else:
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except ended:
            pass
    proc.wait()


def _run(select, limit):
    """
    pytest's exit status and output, or ``(None, "")`` when the run took longer than ``limit`` seconds. pytest
    runs in its own session, so a timeout or an interrupt can end every process its tests started; the terminal's
    Ctrl-C reaches this process alone, which ends the group.
    """
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    with subprocess.Popen([sys.executable, "-m", "pytest", *select], stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, text=True, env=env, start_new_session=True) as proc:
        try:
            output, _ = proc.communicate(timeout=limit)
        except subprocess.TimeoutExpired:
            _end(proc)
            return None, ""
        except _Ended:  # SIGTERM: kill the group at once, to restore the file within an outer run's grace
            _end(proc, grace=0.0)
            raise
        except BaseException:  # Ctrl-C: end the group, then let it unwind
            _end(proc)
            raise
        _end(proc)  # pytest has ended; a process one of its tests left running has not
    return proc.returncode, output


def _write_in_place(path, data):
    """
    Write ``data`` over the file, to the same inode, which ``_lock`` holds. Writing a new file and swapping it in
    (``os.replace``) would let a second run take the lock while this run still mutates the file.
    """
    with open(path, "wb") as f:
        f.write(data)


def _restore(path, original, stat):
    """
    Write the original back, with its modification time; Ctrl-C and SIGTERM wait until it is done. They are held
    by their Python handlers, not blocked: a signal mask holds a signal in one thread only, and a signal sent to
    the process goes to any thread that does not block it, after which the main thread runs the Python handler at
    its next signal check (on CPython 3.10, its next GIL re-take or explicit check), mask or not: in the middle of
    the write, or after the restore. It must be called from the main thread, as ``signal.signal`` must.
    """
    held = []
    previous = {}
    try:
        for sig in (signal.SIGINT, signal.SIGTERM):
            previous[sig] = signal.getsignal(sig)  # stored before the swap, which a signal can follow at once
            signal.signal(sig, lambda signum, frame: held.append(signum))
    finally:  # a signal that comes before its handler is swapped raises here, so the write is in a finally
        try:
            _write_in_place(path, original)
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns))
        finally:
            for sig, handler in previous.items():
                signal.signal(sig, handler)
            for sig in held:
                signal.raise_signal(sig)  # to this thread, which runs its handler before raise_signal returns


def _verdict(code, output, limit):
    """Why a mutant is red: the first failing test pytest's short summary names, a timeout, or pytest's status."""
    if code is None:
        return f"timed out after {limit:g} s"
    for line in output.splitlines():
        if line.startswith(("FAILED ", "ERROR ")):  # "FAILED <node id> - <message>"; an id can hold spaces
            return line.split(" ", 1)[1].split(" - ", 1)[0]
    return f"pytest exit {code}"


def main(argv):
    # SIGTERM, like Ctrl-C, unwinds through the finally below, so the file is restored.
    signal.signal(signal.SIGTERM, _ended)
    if len(argv) != 1:
        _refuse("usage: python tools/mutate.py SPEC.py")
    path, select, mutants, given = _load(argv[0])
    _held = _lock(path)  # released when the process ends
    with open(path, "rb") as f:
        original = f.read()
    stat = os.stat(path)
    refused = _refusals(original, mutants)
    if refused:
        _refuse("refused, no test run:\n  " + "\n  ".join(refused))
    _drop_bytecode(path)
    try:
        started = time.monotonic()
        code, output = _run(select, given)
        limit = timeout(time.monotonic() - started, given)
    except KeyboardInterrupt:
        print(f"mutate.py: interrupted before any mutant; {path} is unchanged", file=sys.stderr)
        return 130
    if code is None:
        _refuse(f"the selection times out before any mutant (TIMEOUT = {given:g} s); raise TIMEOUT or fix it first")
    if code != 0:
        _refuse(f"the selection fails before any mutant (pytest exit {code}); fix it first:\n{output}")
    survivors = 0
    for name, old, new, reason in mutants:
        try:
            try:
                _write_in_place(path, original.replace(old.encode(), new.encode(), 1))
                code, output = _run(select, limit)
            finally:
                _restore(path, original, stat)
        except KeyboardInterrupt:  # subprocess.run has killed pytest
            print(f"mutate.py: interrupted during {name}; {path} is restored", file=sys.stderr)
            return 130
        if code != 0:
            note = ", though marked equivalent" if reason else ""
            print(f"{name}: red ({_verdict(code, output, limit)}){note}")
        elif reason:
            print(f"{name}: survived, marked equivalent ({reason})")
        else:
            print(f"{name}: survived")
            survivors += 1
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
