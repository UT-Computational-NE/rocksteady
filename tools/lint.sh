#!/usr/bin/env bash
# pylint on the messages this repo enforces, over src, tests, tools and examples. tools/check.sh (full run) and
# CI's lint job run this; the [tool.pylint] sections of pyproject.toml configure editors. Left out on purpose
# (#106): unused-argument (fake models and fixtures take arguments they do not use), cell-var-from-loop (closures
# in tools/benchmark.py run before the loop moves on), pointless-statement (an expression inside pytest.raises),
# and no-member / not-an-iterable (false alarms from the numpy and pandas stubs). Set PYTHON to pick the interpreter.
set -euo pipefail
cd "$(dirname "$0")/.."
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
enforced=unused-import,unused-variable,self-assigning-variable,unnecessary-lambda-assignment
"${PYTHON:-python}" -m pylint --rcfile=pyproject.toml --score=n --disable=all --enable="$enforced" \
  src tests tools examples
