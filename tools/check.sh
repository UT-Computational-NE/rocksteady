#!/usr/bin/env bash
# The fast checks CI runs (.github/workflows/tests.yml): the tests without the slow Monte Carlo, and the strict
# docs build. Needs `pip install -e ".[dev]"`. Runs the code of the checkout it is called from, so a worktree
# tests its own src/ even when the environment's editable install points at another checkout. Set PYTHON to pick
# the interpreter (default: python on PATH).
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
py="${PYTHON:-python}"
# Agent shells can inherit another environment's python first on PATH: say which one runs, and stop early
# when it lacks the dev extra.
missing="$("$py" -c 'import importlib.util as u
print(" ".join(m for m in ("pytest", "sphinx", "sphinx_rtd_theme", "myst_parser") if u.find_spec(m) is None))')"
if [ -n "$missing" ]; then
  echo "check.sh: $py lacks $missing. Install the dev extra into it (pip install -e \".[dev]\"), or set PYTHON" \
       "to an interpreter that has it." >&2
  exit 1
fi
echo "check.sh: $("$py" -c 'import sys; print("Python", sys.version.split()[0], "at", sys.executable)')"
"$py" -m pytest -m "not slow" -q
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
"$py" -m sphinx -W --keep-going -q -b html docs/source "$out"
echo "check.sh: tests and strict docs build passed"
