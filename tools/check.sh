#!/usr/bin/env bash
# The fast checks CI runs (.github/workflows/tests.yml): the tests without the slow Monte Carlo, mypy (configured
# in pyproject.toml), and the strict docs build. Needs `pip install -e ".[dev]"` (a selection: `.[test]`). Runs the
# code of the checkout it is called from, so a worktree tests its own src/ even when the environment's editable
# install points at another checkout. Set PYTHON to pick the interpreter (default: python on PATH). With arguments,
# it runs only that pytest selection, without mypy or the docs build: tools/check.sh tests/test_optimal.py -k infeasible
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
py="${PYTHON:-python}"
# Agent shells can inherit another environment's python first on PATH: say which one runs, and stop early
# when it lacks the extra the run needs: dev (mypy and the modules of the `test` and `docs` extras in
# pyproject.toml), or only test for a selection.
selection=$([ $# -gt 0 ] && echo yes || echo no)
if [ "$selection" = yes ]; then
  extra=test; needs=(pytest)
else
  extra=dev; needs=(pytest mypy sphinx sphinx_rtd_theme myst_parser)
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
  # Name the distributions (module names with - for _), not `pip install -e ".[$extra]"`: run from a worktree, that
  # would point the environment's editable install at the worktree, for every session using the environment.
  echo "check.sh: $py ($exe) lacks $missing, from the $extra extra. Install them into it" \
       "($exe -m pip install ${missing//_/-}), or set PYTHON to an interpreter that has them." >&2
  exit 1
fi
"$py" -m pytest -m "not slow" -q "$@"  # with arguments, only that selection; a later -m overrides "not slow"
if [ "$selection" = yes ]; then
  echo "check.sh: the selected tests passed; no mypy or docs build"
  exit 0
fi
"$py" -m mypy
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
"$py" -m sphinx -W --keep-going -q -b html docs/source "$out"
echo "check.sh: tests, mypy and strict docs build passed"
