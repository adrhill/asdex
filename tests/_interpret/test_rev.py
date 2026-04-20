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


# 1D
@pytest.mark.array_ops
def test_rev_1d():
    """1D reversal: output[i] = input[n-1-i]."""
    shape = (5,)

    def f(x):
        return lax.rev(x, dimensions=(0,))

    result = jacobian_sparsity(f, shape).todense().astype(int)
    expected = _rev_jacobian(shape, (0,))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_rev_1d_size_one():
    """Reversing a single-element array is the identity."""
    shape = (1,)

    def f(x):
        return lax.rev(x, dimensions=(0,))

    result = jacobian_sparsity(f, shape).todense().astype(int)
    expected = np.eye(1, dtype=int)
    np.testing.assert_array_equal(result, expected)


# 2D
@pytest.mark.array_ops
def test_rev_2d_dim0():
    """Reverse along dim 0 (flip rows)."""
    shape = (3, 4)

    def f(x):
        return lax.rev(x.reshape(shape), dimensions=(0,)).flatten()

    result = jacobian_sparsity(f, 12).todense().astype(int)
    expected = _rev_jacobian(shape, (0,))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_rev_2d_dim1():
    """Reverse along dim 1 (flip columns)."""
    shape = (3, 4)

    def f(x):
        return lax.rev(x.reshape(shape), dimensions=(1,)).flatten()

    result = jacobian_sparsity(f, 12).todense().astype(int)
    expected = _rev_jacobian(shape, (1,))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_rev_2d_both_dims():
    """Reverse along both dimensions."""
    shape = (3, 4)

    def f(x):
        return lax.rev(x.reshape(shape), dimensions=(0, 1)).flatten()

    result = jacobian_sparsity(f, 12).todense().astype(int)
    expected = _rev_jacobian(shape, (0, 1))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_rev_2d_size_one_dim():
    """Reversing a size-1 dimension is a no-op."""
    shape = (1, 5)

    def f(x):
        return lax.rev(x.reshape(shape), dimensions=(0,)).flatten()

    result = jacobian_sparsity(f, 5).todense().astype(int)
    expected = np.eye(5, dtype=int)
    np.testing.assert_array_equal(result, expected)


# 3D
@pytest.mark.array_ops
def test_rev_3d_single_dim():
    """Reverse a single dimension of a 3D array."""
    shape = (2, 3, 4)

    def f(x):
        return lax.rev(x.reshape(shape), dimensions=(1,)).flatten()

    result = jacobian_sparsity(f, 24).todense().astype(int)
    expected = _rev_jacobian(shape, (1,))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_rev_3d_two_dims():
    """Reverse two of three dimensions."""
    shape = (2, 3, 4)

    def f(x):
        return lax.rev(x.reshape(shape), dimensions=(0, 2)).flatten()

    result = jacobian_sparsity(f, 24).todense().astype(int)
    expected = _rev_jacobian(shape, (0, 2))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_rev_3d_all_dims():
    """Reverse all three dimensions."""
    shape = (2, 3, 4)

    def f(x):
        return lax.rev(x.reshape(shape), dimensions=(0, 1, 2)).flatten()

    result = jacobian_sparsity(f, 24).todense().astype(int)
    expected = _rev_jacobian(shape, (0, 1, 2))
    np.testing.assert_array_equal(result, expected)


# 4D
@pytest.mark.array_ops
def test_rev_4d():
    """Reverse selected dimensions of a 4D array."""
    shape = (2, 3, 2, 4)

    def f(x):
        return lax.rev(x.reshape(shape), dimensions=(1, 3)).flatten()

    result = jacobian_sparsity(f, 48).todense().astype(int)
    expected = _rev_jacobian(shape, (1, 3))
    np.testing.assert_array_equal(result, expected)


# High-level functions
@pytest.mark.array_ops
def test_jnp_flip():
    """jnp.flip lowers to rev; verify end-to-end."""

    def f(x):
        return jnp.flip(x)

    result = jacobian_sparsity(f, 5).todense().astype(int)
    expected = _rev_jacobian((5,), (0,))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_jnp_flip_axis():
    """jnp.flip with explicit axis."""
    shape = (3, 4)

    def f(x):
        return jnp.flip(x.reshape(shape), axis=1).flatten()

    result = jacobian_sparsity(f, 12).todense().astype(int)
    expected = _rev_jacobian(shape, (1,))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_jnp_flipud():
    """jnp.flipud reverses along axis 0."""
    shape = (3, 4)

    def f(x):
        return jnp.flipud(x.reshape(shape)).flatten()

    result = jacobian_sparsity(f, 12).todense().astype(int)
    expected = _rev_jacobian(shape, (0,))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_jnp_fliplr():
    """jnp.fliplr reverses along axis 1."""
    shape = (3, 4)

    def f(x):
        return jnp.fliplr(x.reshape(shape)).flatten()

    result = jacobian_sparsity(f, 12).todense().astype(int)
    expected = _rev_jacobian(shape, (1,))
    np.testing.assert_array_equal(result, expected)


# Edge cases
@pytest.mark.array_ops
def test_rev_empty_dimensions():
    """Reversing no dimensions is the identity."""

    def f(x):
        return lax.rev(x, dimensions=())

    result = jacobian_sparsity(f, 4).todense().astype(int)
    expected = np.eye(4, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_rev_2d_square():
    """Reverse both dims of a square matrix; result is full reversal of flat order."""
    shape = (3, 3)

    def f(x):
        return lax.rev(x.reshape(shape), dimensions=(0, 1)).flatten()

    result = jacobian_sparsity(f, 9).todense().astype(int)
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

    result = jacobian_sparsity(f, 6).todense().astype(int)
    expected = np.eye(6, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_rev_then_rev_different_dims():
    """Reversing dim 0 then dim 1 separately equals reversing both at once."""
    shape = (2, 3)

    def f(x):
        arr = x.reshape(shape)
        return lax.rev(lax.rev(arr, dimensions=(0,)), dimensions=(1,)).flatten()

    result = jacobian_sparsity(f, 6).todense().astype(int)
    expected = _rev_jacobian(shape, (0, 1))
    np.testing.assert_array_equal(result, expected)


# Size-0 dimension


@pytest.mark.array_ops
def test_rev_zero_size():
    """Reversing a zero-sized array produces an empty Jacobian."""

    def f(x):
        return lax.rev(x[:0], dimensions=(0,))

    result = jacobian_sparsity(f, 3)
    assert result.shape == (0, 3)
    assert result.nnz == 0
