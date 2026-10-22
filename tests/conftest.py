"""
Processes the tests start (``python -c``, a script) import the rocksteady the tests import (#99).

pytest's ``pythonpath`` puts this checkout's src/ first for the tests themselves; a child process starts
from the environment, whose editable install may point at another checkout. Putting the tests' own
rocksteady first on PYTHONPATH gives children src/ in a checkout, and the installed package when an
sdist's tests run with ``-o pythonpath=``.

A Pandas4Warning fails the test that raises it, where pandas defines the class (``pytest_configure``).
"""

import os

import pandas as pd

import rocksteady

_HOME = os.path.dirname(os.path.dirname(os.path.abspath(rocksteady.__file__)))
_REST = [p for p in os.environ.get("PYTHONPATH", "").split(os.pathsep) if p]
os.environ["PYTHONPATH"] = os.pathsep.join([_HOME] + _REST)


def pytest_configure(config):
    """Add ``error::pandas.errors.Pandas4Warning`` to the ``filterwarnings`` of pyproject.toml (#109).

    The class subclasses DeprecationWarning, so ``error::FutureWarning`` misses it. pandas defines it from 3.0
    only, and pytest stops on an unknown class in ``filterwarnings``, so the filter is added only where it exists.
    """
    if hasattr(pd.errors, "Pandas4Warning"):
        config.addinivalue_line("filterwarnings", "error::pandas.errors.Pandas4Warning")
