"""Tests for mode resolution helpers."""

import numpy as np
import pytest

import asdex
from asdex.modes import _assert_hessian_mode, _assert_jacobian_mode
from asdex.pattern import SparsityPattern

# Invalid mode validation


def test_assert_jacobian_mode_invalid():
    """Invalid Jacobian mode raises ValueError."""
    with pytest.raises(ValueError, match="Unknown mode"):
        _assert_jacobian_mode("invalid")


def test_assert_hessian_mode_invalid():
    """Invalid Hessian mode raises ValueError."""
    with pytest.raises(ValueError, match="Unknown mode"):
        _assert_hessian_mode("invalid")


def test_jacobian_coloring_invalid_mode():
    """jacobian_coloring raises ValueError on invalid mode."""
    with pytest.raises(ValueError, match="Unknown mode"):
        asdex.jacobian_coloring(lambda x: x, np.zeros(3), mode="invalid")  # ty: ignore[invalid-argument-type]


def test_hessian_coloring_invalid_mode():
    """hessian_coloring raises ValueError on invalid mode."""
    with pytest.raises(ValueError, match="Unknown mode"):
        asdex.hessian_coloring(lambda x: x.sum(), np.zeros(3), mode="invalid")  # ty: ignore[invalid-argument-type]


def test_jacobian_invalid_mode():
    """Jacobian raises ValueError on invalid mode."""
    with pytest.raises(ValueError, match="Unknown mode"):
        asdex.jacobian(lambda x: x, np.zeros(3), mode="invalid")  # ty: ignore[invalid-argument-type]


def test_hessian_invalid_mode():
    """Hessian raises ValueError on invalid mode."""
    with pytest.raises(ValueError, match="Unknown mode"):
        asdex.hessian(lambda x: x.sum(), np.zeros(3), mode="invalid")  # ty: ignore[invalid-argument-type]


def test_jacobian_coloring_from_sparsity_invalid_mode():
    """jacobian_coloring_from_sparsity raises ValueError on invalid mode."""
    sparsity = SparsityPattern.from_coo([0, 1], [0, 1], (2, 2))
    with pytest.raises(ValueError, match="Unknown mode"):
        asdex.jacobian_coloring_from_sparsity(sparsity, mode="invalid")  # ty: ignore[invalid-argument-type]


def test_hessian_coloring_from_sparsity_invalid_mode():
    """hessian_coloring_from_sparsity raises ValueError on invalid mode."""
    sparsity = SparsityPattern.from_coo([0, 1], [0, 1], (2, 2))
    with pytest.raises(ValueError, match="Unknown mode"):
        asdex.hessian_coloring_from_sparsity(sparsity, mode="invalid")  # ty: ignore[invalid-argument-type]


def test_check_jacobian_correctness_invalid_method():
    """check_jacobian_correctness raises ValueError on invalid method."""
    coloring = asdex.jacobian_coloring(lambda x: x, np.zeros(3), mode="fwd")
    x = np.ones(3)
    with pytest.raises(ValueError, match="Unknown method"):
        asdex.check_jacobian_correctness(lambda x: x, x, coloring, method="invalid")  # ty: ignore[invalid-argument-type]


def test_check_hessian_correctness_invalid_method():
    """check_hessian_correctness raises ValueError on invalid method."""
    coloring = asdex.hessian_coloring(lambda x: x.sum(), np.zeros(3))
    x = np.ones(3)
    with pytest.raises(ValueError, match="Unknown method"):
        asdex.check_hessian_correctness(
            lambda x: x.sum(),
            x,
            coloring,
            method="invalid",  # ty: ignore[invalid-argument-type]
        )
