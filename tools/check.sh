#!/usr/bin/env bash
# The fast checks CI runs (.github/workflows/tests.yml): the tests without the slow Monte Carlo, mypy (configured in
# pyproject.toml), tools/lint.sh (pylint on the messages it enforces, pycodestyle on blank lines), and the strict docs
# build. Needs the package's dependencies and the modules of the dev extra (a selection: the test extra); when they are
# missing it says how to install them, and it names an interpreter that is not executable or exits non-zero. Exports
# PYTHONPATH=src for the docs build, which imports this checkout's rocksteady (pytest, and the processes tests start,
# get src/ from pyproject.toml and tests/conftest.py). Set PYTHON to pick the interpreter (default: python on PATH).
# With arguments, it runs only that pytest selection, without mypy, lint or the docs build:
# tools/check.sh tests/test_optimal.py -k infeasible
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
py="${PYTHON:-python}"
# Agent shells can inherit another environment's python first on PATH: say which one runs, and stop early
# when it lacks the package's dependencies or the extra the run needs: dev (mypy, the `lint` extra's pylint and
# pycodestyle, and the modules of the `test` and `docs` extras in pyproject.toml), or only test for a selection.
selection=$#  # the number of pytest arguments; 0 for the full run
if [ "$selection" -gt 0 ]; then
  extra=test; needs=(pytest)
else
  extra=dev; needs=(pytest mypy pylint pycodestyle sphinx sphinx_rtd_theme myst_parser)
fi
file="$py"  # bash's command -v passes over a file that is not executable, unless found by name on PATH
[[ "$py" == */* ]] || file="$(command -v "$py" || true)"
if [[ -e "$file" && ! -x "$file" ]]; then
  echo "check.sh: $py is not executable. Make it executable (chmod +x), or set PYTHON to one that has the $extra" \
       "extra." >&2
  exit 1
fi
if ! command -v "$py" >/dev/null; then
  echo "check.sh: there is no interpreter $py. Set PYTHON to one that has the $extra extra." >&2
  exit 1
fi
crashed() {  # after a call to "$py" exits non-zero with status $1, so that set -e does not end the script unnamed
  echo "check.sh: $py exited with status $1 (any error it printed is above). Set PYTHON to one that has the $extra" \
       "extra." >&2
  exit 1
}
info="$("$py" -c 'import sys; print(sys.version.split()[0], sys.executable)')" || crashed $?
exe="${info#* }"
echo "check.sh: $py is Python ${info%% *} at $exe"
# The distributions of the package's dependencies that are not installed (pyproject.toml's `dependencies`, a list of
# TOML strings, which may have extras and comments; neither markers nor an installed one's extras are checked), a ;,
# the modules of the extra that cannot be found, a ;, then the missing requirements as written there, quoted for the
# shell. A list it cannot read stops the run.
out="$("$py" -c 'import importlib.metadata as md, importlib.util as u, re, shlex, sys
text = open("pyproject.toml").read()
rest = text[re.search(r"^dependencies\s*=\s*\[", text, re.M).end():]
basic, literal = r"\"((?:[^\"\\]|\\.)*)\"", r"\x27([^\x27]*)\x27"  # TOML strings, with and without escapes
comment = r"#[^\n]*(?![^\n])"  # to the end of its line, never to a # inside it, so a failed read cannot backtrack
listed = re.match(r"(?:\s|,|" + comment + "|" + basic + "|" + literal + r")*(?=\])", rest).group(0)
found = re.findall(comment + "|" + basic + "|" + literal, listed)
wanted = [re.sub(r"\\(.)", r"\1", b) or s for b, s in found if b or s]
def name(req):
    return re.match(r"\s*([\w.-]+)", req).group(1)
def installed(req):
    try:
        md.distribution(name(req))
    except md.PackageNotFoundError:
        return False
    return True
absent = [r for r in wanted if not installed(r)]
print(" ".join(name(r) for r in absent), end=";")
print(" ".join(m for m in sys.argv[1:] if u.find_spec(m) is None), end=";")
print(" ".join(shlex.quote(r) for r in absent))
' "${needs[@]}")" || crashed $?
missing_deps="${out%%;*}" out="${out#*;}"
missing_extra="${out%%;*}" deps_install="${out#*;}"  # requirements last: a marker may hold a ;
missing="${missing_deps}${missing_deps:+${missing_extra:+ }}${missing_extra}"
if [ -n "$missing" ]; then
  what="${missing_deps:+$missing_deps, which the package needs}"
  what="${what}${missing_extra:+${what:+, and }$missing_extra, from the $extra extra}"
  # Name the distributions (the extra's module names with - for _), with the requirements in pyproject.toml, not
  # `pip install -e ".[$extra]"`: run from a worktree, that would point the environment's editable install at the
  # worktree, for every session using the environment. A fresh environment needs that install once, from the main
  # checkout.
  extra_install="$("$py" -c 'import re, shlex, sys
text = open("pyproject.toml").read()
for dist in sys.argv[1:]:
    m = re.search(r"\"(" + re.escape(dist) + r"\s*(?:[<>=!~][^\"]*)?)\"", text)
    print(shlex.quote(m.group(1) if m else dist), end=" ")' ${missing_extra//_/-})" || crashed $?
  install="${deps_install}${deps_install:+${extra_install:+ }}${extra_install% }"
  echo "check.sh: $py ($exe) lacks $what. Install them into it" \
       "($exe -m pip install $install), or set PYTHON to an interpreter that has them. A fresh environment is set" \
       "up with pip install -e \".[dev]\" from the main checkout, not a worktree." >&2
  exit 1
fi
"$py" -m pytest -m "not slow" -q "$@"  # with arguments, only that selection; a later -m overrides "not slow"
if [ "$selection" -gt 0 ]; then
  echo "check.sh: the selected tests passed; no mypy, lint or docs build"
  exit 0
fi
"$py" -m mypy
PYTHON="$py" tools/lint.sh
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
"$py" -m sphinx -W --keep-going -q -b html docs/source "$out"
echo "check.sh: tests, mypy, lint and strict docs build passed"
