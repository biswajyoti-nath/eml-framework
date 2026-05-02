"""Shared pytest fixtures for the EML test suite."""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp


@pytest.fixture
def x() -> sp.Symbol:
    """Return the canonical free variable symbol ``x``."""
    return sp.symbols("x")


@pytest.fixture
def positive_sample_points() -> np.ndarray:
    """Return a sample grid on [0.1, 5.0] — safe for log evaluation."""
    return np.linspace(0.1, 5.0, 100)


@pytest.fixture
def full_sample_points() -> np.ndarray:
    """Return a sample grid on [-3.0, 3.0] — includes negative and zero."""
    return np.linspace(-3.0, 3.0, 61)
