"""The test suite's warning policy: a FutureWarning or a Pandas4Warning fails the test that raises it (#75, #109)."""

import warnings

import pandas as pd
import pytest


def test_a_future_warning_fails_the_test_that_raises_it():
    """pyproject.toml turns FutureWarning into an error, so a numpy or pandas FutureWarning cannot pass silently."""
    with pytest.raises(FutureWarning, match="deprecated"):
        warnings.warn("this call is deprecated", FutureWarning)


def test_a_pandas4_warning_fails_the_test_that_raises_it():
    """tests/conftest.py turns Pandas4Warning into an error where pandas defines it (3.0 and later): it subclasses
    DeprecationWarning, not FutureWarning, so the pyproject.toml filter misses it."""
    category = getattr(pd.errors, "Pandas4Warning", None)
    if category is None:
        pytest.skip(f"pandas {pd.__version__} has no Pandas4Warning")
    with pytest.raises(category, match="will raise"):
        warnings.warn("in a future version, this call will raise", category)
