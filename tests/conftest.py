"""
Processes the tests start (``python -c``, a script) import the rocksteady the tests import (#99).

pytest's ``pythonpath`` puts this checkout's src/ first for the tests themselves; a child process starts
from the environment, whose editable install may point at another checkout. Putting the tests' own
rocksteady first on PYTHONPATH gives children src/ in a checkout, and the installed package when an
sdist's tests run with ``-o pythonpath=``.
"""

import os

import rocksteady

_HOME = os.path.dirname(os.path.dirname(os.path.abspath(rocksteady.__file__)))
_REST = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
os.environ["PYTHONPATH"] = os.pathsep.join([_HOME] + _REST)
