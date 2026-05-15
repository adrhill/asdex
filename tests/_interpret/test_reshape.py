"""Tests for reshape propagation.

https://docs.jax.dev/en/latest/_autosummary/jax.lax.reshape.html
"""

import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity


def _reshape_with_dims_jacobian(
    in_shape: tuple[int, ...],
    new_sizes: tuple[int, ...],
    dimensions: tuple[int, ...],
) -> np.ndarray:
    """Build expected Jacobian for reshape with dimensions param.

    The dimensions param transposes the input before flattening.
    """
    n = int(np.prod(in_shape))
    perm = np.arange(n).reshape(in_shape).transpose(dimensions).ravel()
    expected = np.zeros((n, n), dtype=int)
    for out_idx, in_idx in enumerate(perm):
        expected[out_idx, in_idx] = 1
    return expected


_SHAPES_RESHAPE = [
    # 1D to other
    pytest.param((6,), (6,), id="1d_identity"),
    pytest.param((6,), (2, 3), id="1d_to_2d"),
    pytest.param((6,), (3, 2), id="1d_to_2d_alt"),
    pytest.param((12,), (2, 2, 3), id="1d_to_3d"),
    pytest.param((24,), (2, 3, 2, 2), id="1d_to_4d"),
    # 2D to other
    pytest.param((2, 3), (6,), id="2d_to_1d"),
    pytest.param((3, 4), (2, 6), id="2d_to_2d"),
    pytest.param((3, 4), (2, 2, 3), id="2d_to_3d"),
    # 3D to other
    pytest.param((2, 3, 4), (24,), id="3d_to_1d"),
    pytest.param((2, 3, 4), (6, 4), id="3d_to_2d"),
    pytest.param((2, 2, 6), (3, 4, 2), id="3d_to_3d"),
    # Size-1 dimensions
    pytest.param((1,), (1,), id="scalar_like"),
    pytest.param((3,), (1, 3, 1), id="1d_add_ones"),
    pytest.param((6,), (1, 2, 1, 3, 1), id="1d_many_ones"),
    pytest.param((1, 6, 1), (6,), id="remove_ones"),
]

_SHAPES_AND_DIMS_RESHAPE = [
    # 2D permutations
    pytest.param((2, 3), (6,), (0, 1), id="2d_identity_perm"),
    pytest.param((2, 3), (6,), (1, 0), id="2d_transpose"),
    pytest.param((3, 4), (12,), (1, 0), id="2d_transpose_larger"),
    # 3D permutations
    pytest.param((2, 3, 4), (24,), (0, 1, 2), id="3d_identity_perm"),
    pytest.param((2, 3, 4), (24,), (2, 1, 0), id="3d_full_reverse"),
    pytest.param((2, 3, 4), (24,), (0, 2, 1), id="3d_swap_last_two"),
    pytest.param((2, 3, 4), (24,), (1, 0, 2), id="3d_swap_first_two"),
    pytest.param((2, 3, 4), (24,), (1, 2, 0), id="3d_cyclic"),
    pytest.param((2, 3, 4), (24,), (2, 0, 1), id="3d_cyclic_reverse"),
    # 4D permutations
    pytest.param((2, 3, 2, 2), (24,), (3, 2, 1, 0), id="4d_full_reverse"),
    pytest.param((2, 3, 2, 2), (24,), (0, 2, 1, 3), id="4d_swap_middle"),
]


# Core reshape tests
@pytest.mark.array_ops
@pytest.mark.parametrize(("in_shape", "new_sizes"), _SHAPES_RESHAPE)
def test_reshape(in_shape, new_sizes):
    """Reshape is identity for sparsity: each output depends on exactly one input."""
    n = int(np.prod(in_shape))

    def f(x):
        return lax.reshape(x.reshape(in_shape), new_sizes).flatten()

    result = jacobian_sparsity(f, np.zeros(n)).todense().astype(int)
    expected = np.eye(n, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
@pytest.mark.parametrize(
    ("in_shape", "new_sizes", "dimensions"), _SHAPES_AND_DIMS_RESHAPE
)
def test_reshape_with_dimensions(in_shape, new_sizes, dimensions):
    """Reshape with dimensions transposes before flattening."""
    n = int(np.prod(in_shape))

    def f(x):
        return lax.reshape(
            x.reshape(in_shape), new_sizes, dimensions=dimensions
        ).flatten()

    result = jacobian_sparsity(f, np.zeros(n)).todense().astype(int)
    expected = _reshape_with_dims_jacobian(in_shape, new_sizes, dimensions)
    np.testing.assert_array_equal(result, expected)


# Constants
@pytest.mark.array_ops
def test_reshape_constant():
    """Reshaping a constant array produces zero sparsity."""

    def f(_x):
        const = jnp.array([1.0, 2.0, 3.0, 4.0])
        return const.reshape(2, 2).flatten()

    result = jacobian_sparsity(f, np.zeros(2)).todense().astype(int)
    expected = np.zeros((4, 2), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_reshape_then_slice_constant():
    """Reshaping and slicing a constant produces zero sparsity."""

    def f(_x):
        const = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        mat = const.reshape(2, 3)
        return mat[0, :]  # First row

    result = jacobian_sparsity(f, np.zeros(2)).todense().astype(int)
    expected = np.zeros((3, 2), dtype=int)
    np.testing.assert_array_equal(result, expected)


# Non-contiguous input patterns
@pytest.mark.array_ops
def test_reshape_after_broadcast():
    """Reshape following a broadcast preserves the broadcast's dep structure.

    Input (3,) broadcast to (2, 3), then reshaped to (6,).
    Each output pair shares the same input dependency.
    """

    def f(x):
        broadcasted = jnp.broadcast_to(x, (2, 3))
        return broadcasted.reshape(6)

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    # out[0],out[1] -> broadcast row 0 and row 1 of col 0 -> in[0]
    # But broadcast flattens as row-major: [row0, row1] = [(0,1,2), (0,1,2)]
    expected = np.array(
        [
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
            [0, 0, 1],
        ]
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_reshape_after_slice():
    """Reshape after slicing preserves per-element state_indices from the slice."""

    def f(x):
        sliced = x[1:5]  # 4 elements from indices 1..4
        return sliced.reshape(2, 2).flatten()

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    # out[i] depends on in[i+1] for i in 0..3
    expected = np.zeros((4, 6), dtype=int)
    expected[0, 1] = 1
    expected[1, 2] = 1
    expected[2, 3] = 1
    expected[3, 4] = 1
    np.testing.assert_array_equal(result, expected)


# High-level functions
@pytest.mark.array_ops
def test_jnp_reshape():
    """jnp.reshape lowers to lax.reshape."""

    def f(x):
        return jnp.reshape(x, (3, 2)).flatten()

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    expected = np.eye(6, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_jnp_ravel():
    """jnp.ravel on a reshaped array is identity on flat indices."""

    def f(x):
        return jnp.ravel(x.reshape(2, 3))

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    expected = np.eye(6, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_jnp_flatten():
    """ndarray.flatten() lowers through reshape."""

    def f(x):
        return x.reshape(2, 2, 3).flatten()

    result = jacobian_sparsity(f, np.zeros(12)).todense().astype(int)
    expected = np.eye(12, dtype=int)
    np.testing.assert_array_equal(result, expected)


# Edge cases
@pytest.mark.array_ops
def test_reshape_with_dimensions_size_one():
    """Dimensions param with size-1 dims in the original shape."""

    def f(x):
        return lax.reshape(x.reshape(1, 4), (4,), dimensions=(1, 0))

    result = jacobian_sparsity(f, np.zeros(4)).todense().astype(int)
    # Transpose of (1, 4) with dims=(1, 0) -> (4, 1), then flatten = identity
    expected = np.eye(4, dtype=int)
    np.testing.assert_array_equal(result, expected)


# Compositions with other ops
@pytest.mark.array_ops
def test_reshape_then_transpose():
    """Reshape to 2D then transpose: composition of two permutations."""

    def f(x):
        mat = x.reshape(2, 3)
        return mat.T.flatten()

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    # Transpose of (2,3) -> (3,2): flat mapping [0,3,1,4,2,5]
    expected = np.zeros((6, 6), dtype=int)
    for out_idx, in_idx in enumerate([0, 3, 1, 4, 2, 5]):
        expected[out_idx, in_idx] = 1
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_reshape_then_rev():
    """Reshape to 2D then reverse along axis 0."""

    def f(x):
        mat = x.reshape(2, 3)
        return jnp.flip(mat, axis=0).flatten()

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    # flip axis 0 of (2,3): row 0 and row 1 swap
    # flat: [3,4,5,0,1,2]
    expected = np.zeros((6, 6), dtype=int)
    for out_idx, in_idx in enumerate([3, 4, 5, 0, 1, 2]):
        expected[out_idx, in_idx] = 1
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_reshape_with_dimensions_const_propagation():
    """Reshape with ``dimensions`` propagates const values for downstream precision.

    A constant index array reshaped with ``dimensions=(1, 0)``
    transposes before flattening.
    The propagated const value enables the downstream gather
    to use precise indices.
    """

    def f(x):
        indices = jnp.array([[2, 0], [1, 2]])
        # Column-major flatten: transpose then reshape.
        flat_indices = lax.reshape(indices, (4,), dimensions=(1, 0))
        # flat_indices = [2, 1, 0, 2]
        return x[flat_indices]

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.array(
        [
            [0, 0, 1],  # out[0] = x[2]
            [0, 1, 0],  # out[1] = x[1]
            [1, 0, 0],  # out[2] = x[0]
            [0, 0, 1],  # out[3] = x[2]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


# Size-0 dimension


@pytest.mark.array_ops
def test_reshape_zero_size():
    """Reshaping a zero-sized array preserves the empty dependency list."""

    def f(x):
        return lax.reshape(x[:0], (0,))

    result = jacobian_sparsity(f, np.zeros(3))
    assert result.shape == (0, 3)
    assert result.nnz == 0
