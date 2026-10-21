"""Tests for the _prop_cumsum handler.

Cumulative sum produces a lower-triangular (forward) or upper-triangular (reverse)
dependency pattern along the scan axis,
with independent lanes across other dimensions.
The same handler covers ``cumlogsumexp``, which shares this structural pattern.
"""

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity
from tests._utils import (
    assert_flat_jacobian_sparsity_exact,
    assert_jacobian_sparsity_exact,
)

_CUMULATIVE = [
    pytest.param(lax.cumsum, id="cumsum"),
    pytest.param(lax.cumlogsumexp, id="cumlogsumexp"),
]


# 1D tests


@pytest.mark.array_ops
@pytest.mark.parametrize("cumulative", _CUMULATIVE)
@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse"])
def test_cumulative_1d(cumulative, reverse):
    """1D cumulative ops match the numerical Jacobian in both directions."""

    def f(x):
        return cumulative(x, axis=0, reverse=reverse)

    x = jnp.array([0.5, 1.0, 2.0, 4.0])
    assert_jacobian_sparsity_exact(f, x)


@pytest.mark.array_ops
@pytest.mark.parametrize("cumulative", _CUMULATIVE)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cumulative_dtypes(cumulative, dtype):
    """Cumulative ops match the numerical Jacobian in both supported precisions."""

    def f(x):
        return cumulative(x, axis=0)

    with jax.enable_x64(dtype == np.float64):
        x = jnp.asarray([0.5, 1.0, 2.0], dtype=dtype)
        assert_jacobian_sparsity_exact(f, x)


# 2D tests


@pytest.mark.array_ops
@pytest.mark.parametrize("cumulative", _CUMULATIVE)
@pytest.mark.parametrize(
    ("shape", "axis"),
    [
        pytest.param((3, 4), 0, id="axis0"),
        pytest.param((3, 4), 1, id="axis1"),
    ],
)
@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse"])
def test_cumulative_2d(cumulative, shape, axis, reverse):
    """2D cumulative ops match the flattened numerical Jacobian on each axis."""
    n = int(np.prod(shape))

    def f(x):
        return cumulative(x, axis=axis, reverse=reverse)

    x = jnp.arange(1.0, n + 1.0).reshape(shape) / 4
    assert_flat_jacobian_sparsity_exact(f, x)


# 3D tests


@pytest.mark.array_ops
@pytest.mark.parametrize("cumulative", _CUMULATIVE)
@pytest.mark.parametrize("axis", [0, 1, 2], ids=["axis0", "axis1", "axis2"])
@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse"])
def test_cumulative_3d(cumulative, axis, reverse):
    """3D cumulative ops match the flattened numerical Jacobian on each axis."""
    shape = (2, 3, 4)
    n = int(np.prod(shape))

    def f(x):
        return cumulative(x, axis=axis, reverse=reverse)

    x = jnp.arange(1.0, n + 1.0).reshape(shape) / 5
    assert_flat_jacobian_sparsity_exact(f, x)


# Edge cases


@pytest.mark.array_ops
@pytest.mark.parametrize("cumulative", _CUMULATIVE)
@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse"])
def test_cumulative_size_one(cumulative, reverse):
    """Size-1 scan axis matches the flattened numerical Jacobian."""

    def f(x):
        return cumulative(x, axis=0, reverse=reverse)

    x = jnp.array([0.5, 1.0, 2.0]).reshape(1, 3)
    assert_flat_jacobian_sparsity_exact(f, x)


@pytest.mark.array_ops
@pytest.mark.parametrize("cumulative", _CUMULATIVE)
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((0, 3), id="empty_scan_axis"),
        pytest.param((3, 0), id="empty_other_axis"),
    ],
)
def test_cumulative_zero_size(cumulative, shape):
    """Empty inputs produce an empty pattern without crashing."""

    def f(x):
        return cumulative(x, axis=0)

    assert_flat_jacobian_sparsity_exact(f, np.zeros(shape))


@pytest.mark.array_ops
@pytest.mark.parametrize("cumulative", _CUMULATIVE)
def test_cumulative_zero_size_output(cumulative):
    """An empty output over a non-empty input also gives an empty pattern."""

    def f(x):
        return cumulative(jnp.zeros((0, 3)), axis=0)

    assert_flat_jacobian_sparsity_exact(f, np.zeros(3))


# Compositions


@pytest.mark.array_ops
@pytest.mark.parametrize("cumulative", _CUMULATIVE)
def test_cumulative_chained(cumulative):
    """Chaining forward then reverse matches the numerical Jacobian."""

    def f(x):
        y = cumulative(x, axis=0)
        return cumulative(y, axis=0, reverse=True)

    x = jnp.array([0.5, 1.0, 2.0, 4.0])
    assert_jacobian_sparsity_exact(f, x)


@pytest.mark.array_ops
@pytest.mark.parametrize("cumulative", _CUMULATIVE)
def test_cumulative_after_broadcast(cumulative):
    """Non-contiguous input patterns from a prior broadcast are tracked."""

    def f(x):
        # Broadcast (3,) -> (2, 3), then accumulate along axis 0
        y = jnp.broadcast_to(x, (2, 3))
        return cumulative(y, axis=0).flatten()

    x = jnp.array([0.5, 1.0, 2.0])
    assert_jacobian_sparsity_exact(f, x)


# Conservative audit


@pytest.mark.array_ops
@pytest.mark.parametrize("cumulative", _CUMULATIVE)
def test_cumulative_sparser_than_conservative(cumulative):
    """Cumulative pattern is strictly sparser than conservative (all-ones)."""
    shape = (3, 4)
    n = int(np.prod(shape))

    def f(x):
        return cumulative(x.reshape(shape), axis=0).flatten()

    result = jacobian_sparsity(f, np.zeros(n))
    # Conservative would be n*n = 144 nonzeros
    assert result.nnz < n * n
    # Exact count: 4 lanes * (1+2+3) = 24 nonzeros
    expected_nnz = 4 * (1 + 2 + 3)
    assert result.nnz == expected_nnz


# High-level API


@pytest.mark.array_ops
def test_jnp_cumsum():
    """jnp.cumsum lowers to the cumsum primitive."""

    def f(x):
        return jnp.cumsum(x)

    x = jnp.array([0.5, 1.0, 2.0, 4.0])
    assert_jacobian_sparsity_exact(f, x)


@pytest.mark.array_ops
def test_jnp_cumsum_2d_axis():
    """jnp.cumsum with explicit axis on 2D array."""
    shape = (3, 4)
    n = int(np.prod(shape))

    def f(x):
        return jnp.cumsum(x, axis=1)

    x = jnp.arange(1.0, n + 1.0).reshape(shape) / 4
    assert_flat_jacobian_sparsity_exact(f, x)
