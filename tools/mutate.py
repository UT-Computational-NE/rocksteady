"""
Mutation probe: break a file one mutant at a time and check that the tests notice (#101).

    python tools/mutate.py SPEC.py

``SPEC.py`` is a Python file (keep it in your scratchpad, outside the repo) that defines:

- ``FILE``: the file to mutate, relative to the current directory.
- ``SELECT``: the pytest arguments to run for each mutant, e.g. ``["tests/test_optimal.py", "-q", "-x"]``.
- ``MUTANTS``: a list of ``(name, old, new)`` tuples, or ``(name, old, new, reason)`` to mark a mutant
  equivalent (it changes no behaviour the tests could see, e.g. only speed). ``old`` must occur exactly once
  in ``FILE``; triple-quoted strings carry multi-line code with no escaping.

For each mutant in turn, the tool replaces ``old`` with ``new``, runs ``python -m pytest *SELECT`` with this
interpreter, and writes the file back byte for byte, with its modification time, also when pytest cannot collect
or the run is interrupted. The file's cached bytecode is deleted first and the runs write none, so no mutant can
run as a cached original. A mutant is red when pytest exits non-zero, and survived when it passes. The report,
on stdout, has one line per mutant: red (with the first failing test), survived, or survived and marked
equivalent; errors go to stderr. Run no other tests in the same checkout meanwhile: they would test the mutant.

Exit status: 0 when every mutant is red or marked equivalent; 1 when one survived unmarked; 2 when the spec is
refused (a malformed spec, an ``old`` matching zero times or more than once, a repeated name) or the selection
fails before any mutant, all checked before a mutant is written; 130 when interrupted (Ctrl-C or SIGTERM).
"""

import glob
import importlib.util
import os
import runpy
import signal
import subprocess
import sys


def _refuse(message):
    """Stop with exit status 2, as for every spec that cannot run."""
    print(f"mutate.py: {message}", file=sys.stderr)
    raise SystemExit(2)


def _load(path):
    """``FILE``, ``SELECT`` and ``MUTANTS`` of the spec, checked: (file, select, [(name, old, new, reason)])."""
    try:
        spec = runpy.run_path(path)
    except Exception as e:  # pylint: disable=broad-exception-caught  # any error in the caller's spec
        _refuse(f"{path} cannot be run: {type(e).__name__}: {e}")
    missing = [k for k in ("FILE", "SELECT", "MUTANTS") if k not in spec]
    if missing:
        _refuse(f"{path} must define FILE, SELECT and MUTANTS; it lacks {', '.join(missing)}")
    mutants = []
    for i, m in enumerate(spec["MUTANTS"]):
        if not isinstance(m, tuple) or len(m) not in (3, 4) or not all(isinstance(v, str) for v in m):
            _refuse(f"MUTANTS[{i}] must be (name, old, new) or (name, old, new, reason), all strings; got {m!r}")
        mutants.append((m[0], m[1], m[2], m[3] if len(m) == 4 else None))
    return spec["FILE"], list(spec["SELECT"]), mutants


def _refusals(source, mutants):
    """Why the mutants cannot run on ``source`` (bytes): each ``old`` must occur once, each name once."""
    out, seen = [], set()
    for name, old, new, _ in mutants:
        if name in seen:
            out.append(f"{name}: the name is used twice")
        seen.add(name)
        count = source.count(old.encode())
        if count != 1:
            out.append(f"{name}: old matches {count} times; it must match exactly once")
        elif old == new:
            out.append(f"{name}: new is the same as old")
    return out


def _first_failure(output):
    """The first failing test that pytest's short summary names, or what went wrong."""
    for line in output.splitlines():
        if line.startswith(("FAILED ", "ERROR ")):  # "FAILED <node id> - <message>"; an id can hold spaces
            return line.split(" ", 1)[1].split(" - ", 1)[0]
    return "no test passed"


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


def _run(select):
    env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
    run = subprocess.run([sys.executable, "-m", "pytest", *select], capture_output=True, text=True, check=False,
                         env=env)
    return run.returncode, run.stdout + run.stderr


def main(argv):
    if len(argv) != 1:
        _refuse("usage: python tools/mutate.py SPEC.py")
    path, select, mutants = _load(argv[0])
    with open(path, "rb") as f:
        original = f.read()
    times = os.stat(path)
    refused = _refusals(original, mutants)
    if refused:
        print("mutate.py: refused, no test run:\n  " + "\n  ".join(refused), file=sys.stderr)
        return 2
    _drop_bytecode(path)
    code, output = _run(select)
    if code != 0:
        print(f"mutate.py: the selection fails before any mutant (pytest exit {code}); fix it first:\n{output}",
              file=sys.stderr)
        return 2
    # SIGTERM, like Ctrl-C, unwinds through the finally below, so the file is restored.
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    survivors = 0
    for name, old, new, reason in mutants:
        try:
            try:
                with open(path, "wb") as f:
                    f.write(original.replace(old.encode(), new.encode(), 1))
                code, output = _run(select)
            finally:
                with open(path, "wb") as f:
                    f.write(original)
                os.utime(path, ns=(times.st_atime_ns, times.st_mtime_ns))
        except KeyboardInterrupt:  # subprocess.run has killed pytest
            print(f"mutate.py: interrupted during {name}; {path} is restored", file=sys.stderr)
            return 130
        if code != 0:
            note = ", though marked equivalent" if reason else ""
            print(f"{name}: red ({_first_failure(output)}){note}")
        elif reason:
            print(f"{name}: survived, marked equivalent ({reason})")
        else:
            print(f"{name}: survived")
            survivors += 1
    return 1 if survivors else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
