#!/usr/bin/env bash
# The fast checks CI runs (.github/workflows/tests.yml): the tests without the slow Monte Carlo, mypy (configured
# in pyproject.toml), pylint on the messages tools/lint.sh enforces, and the strict docs build. Needs the modules of
# the dev extra (a selection: the test extra); when they are missing it says how to install them. Runs the code of
# the checkout it is called from, so a worktree tests its own src/ even when the environment's editable install
# points at another checkout. Set PYTHON to pick the interpreter (default: python on PATH). With arguments, it runs
# only that pytest selection, without mypy, pylint or the docs build: tools/check.sh tests/test_optimal.py -k infeasible
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
py="${PYTHON:-python}"
# Agent shells can inherit another environment's python first on PATH: say which one runs, and stop early
# when it lacks the extra the run needs: dev (mypy, pylint and the modules of the `test` and `docs` extras in
# pyproject.toml), or only test for a selection.
selection=$([ $# -gt 0 ] && echo yes || echo no)
if [ "$selection" = yes ]; then
  extra=test; needs=(pytest)
else
  extra=dev; needs=(pytest mypy pylint sphinx sphinx_rtd_theme myst_parser)
fi
if ! command -v "$py" >/dev/null; then
  echo "check.sh: there is no interpreter $py. Set PYTHON to one that has the $extra extra." >&2
  exit 1
fi
info="$("$py" -c 'import sys; print(sys.version.split()[0], sys.executable)')"
exe="${info#* }"
echo "check.sh: $py is Python ${info%% *} at $exe"
missing="$("$py" -c 'import importlib.util as u, sys
print(" ".join(m for m in sys.argv[1:] if u.find_spec(m) is None))' "${needs[@]}")"
if [ -n "$missing" ]; then
  # Name the distributions (module names with - for _), with the extras' minimum versions from pyproject.toml, not
  # `pip install -e ".[$extra]"`: run from a worktree, that would point the environment's editable install at the
  # worktree, for every session using the environment. A fresh environment needs that install once, from the main
  # checkout, for the package's own dependencies.
  reqs="$("$py" -c 'import re, sys
text = open("pyproject.toml").read()
for dist in sys.argv[1:]:
    m = re.search(r"\"(" + re.escape(dist) + r"(?:[<>=!~][^\"]*)?)\"", text)
    print(repr(m.group(1) if m else dist), end=" ")' ${missing//_/-})"
  echo "check.sh: $py ($exe) lacks $missing, from the $extra extra. Install them into it" \
       "($exe -m pip install ${reqs% }), or set PYTHON to an interpreter that has them. A fresh environment needs" \
       "the package's dependencies too: pip install -e \".[dev]\" from the main checkout, not a worktree." >&2
  exit 1
fi
"$py" -m pytest -m "not slow" -q "$@"  # with arguments, only that selection; a later -m overrides "not slow"
if [ "$selection" = yes ]; then
  echo "check.sh: the selected tests passed; no mypy, pylint or docs build"
  exit 0
fi
"$py" -m mypy
PYTHON="$py" tools/lint.sh
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
"$py" -m sphinx -W --keep-going -q -b html docs/source "$out"
echo "check.sh: tests, mypy, pylint and strict docs build passed"
