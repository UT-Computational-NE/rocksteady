#!/usr/bin/env bash
# pylint on the messages this repo enforces, and pycodestyle on blank lines, over src, tests, tools and
# examples. tools/check.sh (full run) and
# CI's lint job run this. It reads pyproject.toml's [tool.pylint] sections, written for editors; with --disable=all
# only the four messages below remain. Left out on purpose (#106): unused-argument (fake models and fixtures take
# arguments they do not use), cell-var-from-loop (closures in tools/benchmark.py run before the loop moves on),
# pointless-statement (an expression inside pytest.raises), and no-member / not-an-iterable (false alarms from the
# numpy and pandas stubs). Set PYTHON to pick the interpreter.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
enforced=unused-import,unused-variable,self-assigning-variable,unnecessary-lambda-assignment
"${PYTHON:-python}" -m pylint --rcfile=pyproject.toml --score=n --disable=all --enable="$enforced" \
  src tests tools examples
# Blank lines around definitions (PEP 8), which pylint does not check: two around a top-level one, one around a
# method, none in excess.
"${PYTHON:-python}" -m pycodestyle --select=E301,E302,E303,E305,E306 src tests tools examples
