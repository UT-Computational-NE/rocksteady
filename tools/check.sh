#!/usr/bin/env bash
# The fast checks CI runs (.github/workflows/tests.yml): the tests without the slow Monte Carlo, and the strict
# docs build. Needs `pip install -e ".[dev]"`. Runs the code of the checkout it is called from, so a worktree
# tests its own src/ even when the environment's editable install points at another checkout. Set PYTHON to pick
# the interpreter (default: python on PATH).
set -euo pipefail
cd "$(git -C "$(dirname "$0")" rev-parse --show-toplevel)"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
"${PYTHON:-python}" -m pytest -m "not slow" -q
out="$(mktemp -d)"
trap 'rm -rf "$out"' EXIT
"${PYTHON:-python}" -m sphinx -W --keep-going -q -b html docs/source "$out"
echo "check.sh: tests and strict docs build passed"
