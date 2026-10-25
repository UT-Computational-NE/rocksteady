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
info="$("$py" -c 'import sys; print(sys.version.split()[0], sys.executable)')" || {
  echo "check.sh: $py exited with status $? (its own error is above). Set PYTHON to one that has the $extra extra." >&2
  exit 1
}
exe="${info#* }"
echo "check.sh: $py is Python ${info%% *} at $exe"
# The missing modules of the package's dependencies (pyproject.toml's `dependencies`, each module named as its
# distribution, as numpy's and pandas's are), a ;, then those of the extra.
missing="$("$py" -c 'import importlib.util as u, re, sys
deps = re.search(r"^dependencies = \[(.*?)\]", open("pyproject.toml").read(), re.M | re.S).group(1)
deps = [re.match(r"[\w.-]+", d).group(0) for d in re.findall(r"\"([^\"]+)\"", deps)]
print(";".join(" ".join(m for m in modules if u.find_spec(m) is None) for modules in (deps, sys.argv[1:])))
' "${needs[@]}")"
missing_deps="${missing%%;*}" missing_extra="${missing#*;}"
missing="${missing_deps}${missing_deps:+${missing_extra:+ }}${missing_extra}"
if [ -n "$missing" ]; then
  what="${missing_deps:+$missing_deps, which the package needs}"
  what="${what}${missing_extra:+${what:+, and }$missing_extra, from the $extra extra}"
  # Name the distributions (module names with - for _), with the minimum versions from pyproject.toml, not
  # `pip install -e ".[$extra]"`: run from a worktree, that would point the environment's editable install at the
  # worktree, for every session using the environment. A fresh environment needs that install once, from the main
  # checkout.
  reqs="$("$py" -c 'import re, sys
text = open("pyproject.toml").read()
for dist in sys.argv[1:]:
    m = re.search(r"\"(" + re.escape(dist) + r"(?:[<>=!~][^\"]*)?)\"", text)
    print(repr(m.group(1) if m else dist), end=" ")' ${missing//_/-})"
  echo "check.sh: $py ($exe) lacks $what. Install them into it" \
       "($exe -m pip install ${reqs% }), or set PYTHON to an interpreter that has them. A fresh environment is set" \
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
