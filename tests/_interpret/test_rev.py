"""Tests for the rev (reverse) propagation handler.

Tests reversals along single and multiple dimensions,
identity cases, size-1 dimensions, and high-level functions
that lower to rev.
"""

import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity


def _rev_jacobian(shape: tuple[int, ...], dimensions: tuple[int, ...]):
    """Build the expected permutation Jacobian for a rev operation.

    For each flat output index,
    compute which flat input index it reads from
    by flipping coordinates along the reversed dimensions.
    """
    n = int(np.prod(shape))
    expected = np.zeros((n, n), dtype=int)
    for out_flat in range(n):
        out_coord = list(np.unravel_index(out_flat, shape))
        in_coord = tuple(
            shape[d] - 1 - out_coord[d] if d in dimensions else out_coord[d]
            for d in range(len(shape))
        )
        in_flat = np.ravel_multi_index(in_coord, shape)
        expected[out_flat, in_flat] = 1
    return expected


_SHAPES_AND_DIMS = [
    pytest.param((5,), (), id="1d_empty"),
    pytest.param((5,), (0,), id="1d_full"),
    pytest.param((1,), (0,), id="1d_size_one"),
    pytest.param((3, 4), (), id="2d_empty"),
    pytest.param((3, 4), (0,), id="2d_dim0"),
    pytest.param((3, 4), (1,), id="2d_dim1"),
    pytest.param((3, 4), (0, 1), id="2d_both"),
    pytest.param((1, 5), (0,), id="2d_size_one_reversed"),
    pytest.param((1, 5), (1,), id="2d_size_one_kept"),
    pytest.param((2, 3, 4), (), id="3d_empty"),
    pytest.param((2, 3, 4), (1,), id="3d_single_dim"),
    pytest.param((2, 3, 4), (0, 2), id="3d_two_dims"),
    pytest.param((2, 3, 4), (0, 1, 2), id="3d_all_dims"),
    pytest.param((2, 3, 2, 4), (1, 3), id="4d"),
]


# Core rev tests
@pytest.mark.array_ops
@pytest.mark.parametrize(("shape", "dimensions"), _SHAPES_AND_DIMS)
def test_rev(shape, dimensions):
    """Each output reads from one input with flipped coordinates along reversed dims."""
    n = int(np.prod(shape))

    def f(x):
        return lax.rev(x.reshape(shape), dimensions=dimensions).flatten()

    result = jacobian_sparsity(f, np.zeros(n)).todense().astype(int)
    expected = _rev_jacobian(shape, dimensions)
    np.testing.assert_array_equal(result, expected)


# High-level functions
@pytest.mark.array_ops
def test_jnp_flip():
    """jnp.flip lowers to rev; verify end-to-end."""

    def f(x):
        return jnp.flip(x)

    result = jacobian_sparsity(f, np.zeros(5)).todense().astype(int)
    expected = _rev_jacobian((5,), (0,))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_jnp_flip_axis():
    """jnp.flip with explicit axis."""
    shape = (3, 4)

    def f(x):
        return jnp.flip(x.reshape(shape), axis=1).flatten()

    result = jacobian_sparsity(f, np.zeros(12)).todense().astype(int)
    expected = _rev_jacobian(shape, (1,))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_jnp_flipud():
    """jnp.flipud reverses along axis 0."""
    shape = (3, 4)

    def f(x):
        return jnp.flipud(x.reshape(shape)).flatten()

    result = jacobian_sparsity(f, np.zeros(12)).todense().astype(int)
    expected = _rev_jacobian(shape, (0,))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_jnp_fliplr():
    """jnp.fliplr reverses along axis 1."""
    shape = (3, 4)

    def f(x):
        return jnp.fliplr(x.reshape(shape)).flatten()

    result = jacobian_sparsity(f, np.zeros(12)).todense().astype(int)
    expected = _rev_jacobian(shape, (1,))
    np.testing.assert_array_equal(result, expected)


# Edge cases
@pytest.mark.array_ops
def test_rev_2d_square():
    """Reverse both dims of a square matrix; result is full reversal of flat order."""
    shape = (3, 3)

    def f(x):
        return lax.rev(x.reshape(shape), dimensions=(0, 1)).flatten()

    result = jacobian_sparsity(f, np.zeros(9)).todense().astype(int)
    # Full reversal of a 3x3: anti-identity on the full flattened array.
    expected = np.eye(9, dtype=int)[::-1]
    np.testing.assert_array_equal(result, expected)


# Double reverse (involution)
@pytest.mark.array_ops
def test_double_rev_is_identity():
    """Reversing twice along the same dimensions gives the identity."""

    def f(x):
        arr = x.reshape(2, 3)
        return lax.rev(lax.rev(arr, dimensions=(0, 1)), dimensions=(0, 1)).flatten()

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    expected = np.eye(6, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_rev_then_rev_different_dims():
    """Reversing dim 0 then dim 1 separately equals reversing both at once."""
    shape = (2, 3)

    def f(x):
        arr = x.reshape(shape)
        return lax.rev(lax.rev(arr, dimensions=(0,)), dimensions=(1,)).flatten()

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    expected = _rev_jacobian(shape, (0, 1))
    np.testing.assert_array_equal(result, expected)


# Size-0 dimension


@pytest.mark.array_ops
def test_rev_zero_size():
    """Reversing a zero-sized array produces an empty Jacobian."""

    def f(x):
        return lax.rev(x[:0], dimensions=(0,))

    result = jacobian_sparsity(f, np.zeros(3))
    assert result.shape == (0, 3)
    assert result.nnz == 0
