"""Tests for the cumlogsumexp handler.

Cumulative log-sum-exp has the same structural sparsity pattern as cumsum:
each output depends on the prefix (or suffix when ``reverse=True``)
along the scan axis, with independent lanes across other dimensions.
"""

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity
from tests._utils import assert_jacobian_sparsity_exact


@pytest.mark.array_ops
@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse"])
def test_cumlogsumexp_1d(reverse):
    """1D cumlogsumexp matches the numerical Jacobian in both directions."""

    def f(x):
        return lax.cumlogsumexp(x, axis=0, reverse=reverse)

    x = jnp.array([0.5, 1.0, 2.0, 4.0])
    assert_jacobian_sparsity_exact(f, x)


@pytest.mark.array_ops
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cumlogsumexp_dtypes(dtype):
    """Cumlogsumexp matches the numerical Jacobian for both supported precisions."""

    def f(x):
        return lax.cumlogsumexp(x, axis=0)

    with jax.enable_x64(dtype == np.float64):
        x = jnp.asarray([0.5, 1.0, 2.0], dtype=dtype)
        assert_jacobian_sparsity_exact(f, x)


@pytest.mark.array_ops
@pytest.mark.parametrize(
    ("shape", "axis"),
    [
        pytest.param((3, 4), 0, id="axis0"),
        pytest.param((3, 4), 1, id="axis1"),
    ],
)
@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse"])
def test_cumlogsumexp_2d(shape, axis, reverse):
    """2D cumlogsumexp along each axis matches the numerical Jacobian."""
    n = int(np.prod(shape))

    def f(x):
        return lax.cumlogsumexp(x.reshape(shape), axis=axis, reverse=reverse).flatten()

    x = jnp.arange(1.0, n + 1.0) / 4
    assert_jacobian_sparsity_exact(f, x)


@pytest.mark.array_ops
@pytest.mark.parametrize("axis", [0, 1, 2], ids=["axis0", "axis1", "axis2"])
@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse"])
def test_cumlogsumexp_3d(axis, reverse):
    """3D cumlogsumexp along each axis matches the numerical Jacobian."""
    shape = (2, 3, 4)
    n = int(np.prod(shape))

    def f(x):
        return lax.cumlogsumexp(x.reshape(shape), axis=axis, reverse=reverse).flatten()

    x = jnp.arange(1.0, n + 1.0) / 5
    assert_jacobian_sparsity_exact(f, x)


@pytest.mark.array_ops
@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse"])
def test_cumlogsumexp_size_one(reverse):
    """Size-1 scan axis matches the numerical Jacobian in both directions."""

    def f(x):
        return lax.cumlogsumexp(x.reshape(1, 3), axis=0, reverse=reverse).flatten()

    x = jnp.array([0.5, 1.0, 2.0])
    assert_jacobian_sparsity_exact(f, x)


@pytest.mark.array_ops
@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse"])
def test_cumlogsumexp_zero_size_scan_axis(reverse):
    """An empty scan axis returns an empty pattern."""

    def f(x):
        return lax.cumlogsumexp(x.reshape(0, 3), axis=0, reverse=reverse).flatten()

    result = jacobian_sparsity(f, np.zeros(0))
    assert result.shape == (0, 0)
    assert result.nnz == 0


@pytest.mark.array_ops
def test_cumlogsumexp_zero_size_other_axis():
    """A zero-length non-scan axis also returns an empty pattern."""

    def f(x):
        return lax.cumlogsumexp(x.reshape(3, 0), axis=0).flatten()

    result = jacobian_sparsity(f, np.zeros(0))
    assert result.shape == (0, 0)
    assert result.nnz == 0
