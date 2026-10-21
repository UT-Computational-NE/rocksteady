"""The test suite's warning policy (#75): a FutureWarning fails the test that raises it."""

import warnings

import pytest


def test_a_future_warning_fails_the_test_that_raises_it():
    """pyproject.toml turns FutureWarning into an error, so a numpy or pandas deprecation cannot pass silently."""
    with pytest.raises(FutureWarning, match="deprecated"):
        warnings.warn("this call is deprecated", FutureWarning)
