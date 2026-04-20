"""Tests for the prop_sort handler.

Sort along one dimension mixes elements within slices along that dimension,
producing block-diagonal patterns for multi-dimensional arrays.
"""

import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity


def _sort_jacobian(in_shape: tuple[int, ...], dimension: int) -> np.ndarray:
    """Build the expected Jacobian for sort along a given dimension.

    Each output element depends on all inputs sharing the same
    non-sort coordinates.
    """
    ndim = len(in_shape)
    if dimension < 0:
        dimension += ndim

    n = int(np.prod(in_shape))
    expected = np.zeros((n, n), dtype=int)

    kept_dims = [d for d in range(ndim) if d != dimension]

    for flat_out in range(n):
        out_coords = np.unravel_index(flat_out, in_shape)
        for flat_in in range(n):
            in_coords = np.unravel_index(flat_in, in_shape)
            # Same batch slice: all non-sort coords match
            if all(out_coords[d] == in_coords[d] for d in kept_dims):
                expected[flat_out, flat_in] = 1

    return expected


# 1D sort
@pytest.mark.array_ops
def test_sort_1d():
    """1D sort: all outputs depend on all inputs."""

    def f(x):
        return jnp.sort(x)

    result = jacobian_sparsity(f, input_shape=(3,)).todense().astype(int)
    expected = np.ones((3, 3), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_sort_1d_size_one():
    """Size-1 sort dimension is a no-op: identity pattern."""

    def f(x):
        return jnp.sort(x)

    result = jacobian_sparsity(f, input_shape=(1,)).todense().astype(int)
    expected = np.ones((1, 1), dtype=int)
    np.testing.assert_array_equal(result, expected)


# 2D sort
@pytest.mark.array_ops
def test_sort_2d_axis1():
    """Sort (2, 3) along axis=1: two 3x3 blocks."""

    def f(x):
        return lax.sort(x.reshape(2, 3), dimension=1).flatten()

    result = jacobian_sparsity(f, input_shape=(6,)).todense().astype(int)
    expected = _sort_jacobian((2, 3), dimension=1)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_sort_2d_axis0():
    """Sort (2, 3) along axis=0: three 2x2 blocks."""

    def f(x):
        return lax.sort(x.reshape(2, 3), dimension=0).flatten()

    result = jacobian_sparsity(f, input_shape=(6,)).todense().astype(int)
    expected = _sort_jacobian((2, 3), dimension=0)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_sort_2d_negative_axis():
    """Negative dimension: axis=-1 is the same as the last axis."""

    def f(x):
        return jnp.sort(x.reshape(2, 3), axis=-1).flatten()

    result = jacobian_sparsity(f, input_shape=(6,)).todense().astype(int)
    expected = _sort_jacobian((2, 3), dimension=-1)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_sort_2d_size_one_sort_dim():
    """Sort along a size-1 dimension: identity pattern."""

    def f(x):
        return lax.sort(x.reshape(3, 1), dimension=1).flatten()

    result = jacobian_sparsity(f, input_shape=(3,)).todense().astype(int)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


# 3D sort
@pytest.mark.array_ops
@pytest.mark.parametrize("dimension", [0, 1, 2])
def test_sort_3d(dimension):
    """Sort (2, 3, 4) along each dimension produces the correct block pattern."""
    in_shape = (2, 3, 4)
    n = int(np.prod(in_shape))

    def f(x):
        return lax.sort(x.reshape(in_shape), dimension=dimension).flatten()

    result = jacobian_sparsity(f, input_shape=(n,)).todense().astype(int)
    expected = _sort_jacobian(in_shape, dimension)
    np.testing.assert_array_equal(result, expected)


# Multi-operand sort
@pytest.mark.array_ops
def test_sort_multi_operand():
    """Multi-key sort: state_indices from all operands are unioned per slice.

    lax.sort((keys, values), num_keys=1) sorts values by keys.
    Both outputs depend on all inputs in the same batch slice.
    """
    in_shape = (2, 3)
    n = int(np.prod(in_shape))

    def f(x):
        keys = x[:n].reshape(in_shape)
        vals = x[n:].reshape(in_shape)
        sorted_keys, sorted_vals = lax.sort((keys, vals), dimension=1, num_keys=1)
        return jnp.concatenate([sorted_keys.flatten(), sorted_vals.flatten()])

    result = jacobian_sparsity(f, input_shape=(2 * n,)).todense().astype(int)

    # Both outputs have block-diagonal patterns,
    # but each block depends on inputs from both operands.
    # Output has 12 rows (6 sorted_keys + 6 sorted_vals), 12 columns (6 keys + 6 vals).
    expected = np.zeros((2 * n, 2 * n), dtype=int)
    for b in range(2):  # 2 batch slices
        out_slice = slice(b * 3, (b + 1) * 3)
        in_slice_keys = slice(b * 3, (b + 1) * 3)
        in_slice_vals = slice(n + b * 3, n + (b + 1) * 3)
        # sorted_keys block
        expected[out_slice, in_slice_keys] = 1
        expected[out_slice, in_slice_vals] = 1
        # sorted_vals block (offset by n in output)
        expected[n + out_slice.start : n + out_slice.stop, in_slice_keys] = 1
        expected[n + out_slice.start : n + out_slice.stop, in_slice_vals] = 1

    np.testing.assert_array_equal(result, expected)


# High-level API
@pytest.mark.array_ops
@pytest.mark.fallback
def test_argsort():
    """Argsort is piecewise constant (zero derivative) but detected as dense.

    TODO(sort): argsort returns integer indices whose values don't change
    under infinitesimal perturbations.
    Structural sparsity detection can't distinguish index-like outputs
    from value-like outputs in multi-operand sort.
    """

    def f(x):
        return jnp.argsort(x).astype(float)

    result = jacobian_sparsity(f, input_shape=(4,)).todense().astype(int)
    # Precise: zero matrix (piecewise constant function)
    # Detected: dense (structural state_indices from sort permutation)
    expected = np.ones((4, 4), dtype=int)
    np.testing.assert_array_equal(result, expected)


# Compositions
@pytest.mark.array_ops
def test_sort_then_reduce():
    """Sort followed by sum along sort axis collapses to reduction pattern."""

    def f(x):
        sorted_x = jnp.sort(x.reshape(2, 3), axis=1)
        return jnp.sum(sorted_x, axis=1)

    result = jacobian_sparsity(f, input_shape=(6,)).todense().astype(int)
    # Sum along axis=1 after sort: each output depends on its row
    expected = np.zeros((2, 6), dtype=int)
    expected[0, 0:3] = 1
    expected[1, 3:6] = 1
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_sort_non_contiguous_input():
    """Sort after broadcast: non-contiguous input dependencies."""

    def f(x):
        # x.shape = (3,), broadcast to (2, 3), then sort along axis=1
        broadcasted = jnp.broadcast_to(x, (2, 3))
        return jnp.sort(broadcasted, axis=1).flatten()

    result = jacobian_sparsity(f, input_shape=(3,)).todense().astype(int)
    # Both rows share the same 3 inputs after broadcast.
    # Sort within each row: all 3 outputs per row depend on all 3 inputs.
    expected = np.ones((6, 3), dtype=int)
    np.testing.assert_array_equal(result, expected)


# Size-0 dimension


@pytest.mark.array_ops
def test_sort_zero_size():
    """Sorting a zero-sized array produces an empty Jacobian."""

    def f(x):
        return jnp.sort(x[:0])

    result = jacobian_sparsity(f, input_shape=(3,))
    assert result.shape == (0, 3)
    assert result.nnz == 0
