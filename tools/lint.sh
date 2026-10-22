#!/usr/bin/env bash
# pylint on the messages this repo enforces, and pycodestyle on blank lines, over src, tests, tools and examples.
# tools/check.sh (full run) and CI's lint job run this. Both linters always run, so one run reports every finding;
# it fails when either finds one. pylint reads pyproject.toml's [tool.pylint] sections, written for editors; with
# --disable=all only the four messages below remain. Left out on purpose (#106): unused-argument (fake models and
# fixtures take arguments they do not use), cell-var-from-loop (closures in tools/benchmark.py run before the loop
# moves on), pointless-statement (an expression inside pytest.raises), and no-member / not-an-iterable (false alarms
# from the numpy and pandas stubs). pycodestyle checks the blank lines PEP 8 asks for, which pylint does not: two
# around a top-level definition, one before a method or a nested definition, none in excess. Set PYTHON to pick the
# interpreter.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
py="${PYTHON:-python}"
enforced=unused-import,unused-variable,self-assigning-variable,unnecessary-lambda-assignment
status=0
"$py" -m pylint --rcfile=pyproject.toml --score=n --disable=all --enable="$enforced" \
  src tests tools examples || status=1
"$py" -m pycodestyle --select=E301,E302,E303,E305,E306 src tests tools examples || status=1
exit "$status"
