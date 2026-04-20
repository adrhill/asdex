"""Tests for reshape propagation.

https://docs.jax.dev/en/latest/_autosummary/jax.lax.reshape.html
"""

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity


# Identity reshape (no dimensions param)
@pytest.mark.array_ops
def test_reshape_1d_to_2d():
    """Reshaping 1D to 2D without permutation is the identity on flat indices."""

    def f(x):
        return x.reshape(2, 3).flatten()

    result = jacobian_sparsity(f, input_shape=(6,)).todense().astype(int)
    expected = np.eye(6, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_reshape_2d_to_1d():
    """Reshaping 2D back to 1D is the identity on flat indices."""

    def f(x):
        return x.reshape(3, 2).reshape(6)

    result = jacobian_sparsity(f, input_shape=(6,)).todense().astype(int)
    expected = np.eye(6, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_reshape_2d_to_3d():
    """Reshaping 2D to 3D without permutation preserves flat element order."""

    def f(x):
        return x.reshape(3, 4).reshape(2, 2, 3).flatten()

    result = jacobian_sparsity(f, input_shape=(12,)).todense().astype(int)
    expected = np.eye(12, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_reshape_same_shape():
    """Reshaping to the same shape is a no-op."""

    def f(x):
        return lax.reshape(x, (4,))

    result = jacobian_sparsity(f, input_shape=(4,)).todense().astype(int)
    expected = np.eye(4, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_reshape_size_one_dims():
    """Reshaping with size-1 dimensions preserves element identity."""

    def f(x):
        return x.reshape(1, 3, 1).flatten()

    result = jacobian_sparsity(f, input_shape=(3,)).todense().astype(int)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_reshape_scalar_to_1d():
    """Reshaping a scalar-like (1,) to (1,) is identity."""

    def f(x):
        return x.reshape(1)

    result = jacobian_sparsity(f, input_shape=(1,)).todense().astype(int)
    expected = np.eye(1, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_reshape_4d():
    """Reshape through a 4D intermediate preserves flat ordering."""

    def f(x):
        return x.reshape(2, 3, 2, 2).flatten()

    result = jacobian_sparsity(f, input_shape=(24,)).todense().astype(int)
    expected = np.eye(24, dtype=int)
    np.testing.assert_array_equal(result, expected)


# Reshape with dimensions param
@pytest.mark.array_ops
def test_reshape_with_dimensions_2d():
    """Reshape with dimensions=(1,0) permutes a 2D array before flattening.

    ravel(order='F') on a (2, 3) matrix emits dimensions=(1, 0).
    Each output element still depends on exactly one input element.
    """

    def f(x):
        mat = x.reshape(2, 3)
        return mat.ravel(order="F")  # column-major: [a, d, b, e, c, f]

    result = jacobian_sparsity(f, input_shape=(6,)).todense().astype(int)
    # Input flat: [a=0, b=1, c=2, d=3, e=4, f=5] in (2,3)
    # F-order ravel: [a, d, b, e, c, f] = [0, 3, 1, 4, 2, 5]
    expected = np.zeros((6, 6), dtype=int)
    expected[0, 0] = 1  # out[0] <- a
    expected[1, 3] = 1  # out[1] <- d
    expected[2, 1] = 1  # out[2] <- b
    expected[3, 4] = 1  # out[3] <- e
    expected[4, 2] = 1  # out[4] <- c
    expected[5, 5] = 1  # out[5] <- f
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_reshape_with_dimensions_3d():
    """Reshape with dimensions=(2,1,0) permutes a 3D array before flattening.

    ravel(order='F') on a (2, 3, 4) tensor emits dimensions=(2, 1, 0).
    Verifies correct handling with higher-rank permutations.
    """

    def f(x):
        tensor = x.reshape(2, 3, 4)
        return tensor.ravel(order="F")

    result = jacobian_sparsity(f, input_shape=(24,)).todense().astype(int)
    x_test = jax.random.normal(jax.random.key(42), (24,))
    actual_jac = jax.jacobian(f)(x_test)
    actual_nonzero = (np.abs(actual_jac) > 1e-10).astype(int)
    np.testing.assert_array_equal(result, actual_nonzero)


@pytest.mark.array_ops
def test_reshape_with_dimensions_identity_perm():
    """dimensions=(0, 1) on a 2D array is equivalent to no permutation."""

    def f(x):
        return lax.reshape(x.reshape(2, 3), (6,), dimensions=(0, 1))

    result = jacobian_sparsity(f, input_shape=(6,)).todense().astype(int)
    expected = np.eye(6, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_reshape_with_dimensions_3d_partial_perm():
    """Non-reversal permutation on a 3D array: dimensions=(0, 2, 1).

    Swaps last two axes before flattening.
    """

    def f(x):
        return lax.reshape(x.reshape(2, 3, 4), (24,), dimensions=(0, 2, 1))

    result = jacobian_sparsity(f, input_shape=(24,)).todense().astype(int)
    # Build expected via numpy: transpose then ravel
    perm = np.arange(24).reshape(2, 3, 4).transpose(0, 2, 1).ravel()
    expected = np.zeros((24, 24), dtype=int)
    for out_idx, in_idx in enumerate(perm):
        expected[out_idx, in_idx] = 1
    np.testing.assert_array_equal(result, expected)


# Constants
@pytest.mark.array_ops
def test_reshape_constant():
    """Reshaping a constant array produces zero sparsity."""

    def f(_x):
        const = jnp.array([1.0, 2.0, 3.0, 4.0])
        return const.reshape(2, 2).flatten()

    result = jacobian_sparsity(f, input_shape=(2,)).todense().astype(int)
    expected = np.zeros((4, 2), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_reshape_then_slice_constant():
    """Reshaping and slicing a constant produces zero sparsity."""

    def f(_x):
        const = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        mat = const.reshape(2, 3)
        return mat[0, :]  # First row

    result = jacobian_sparsity(f, input_shape=(2,)).todense().astype(int)
    expected = np.zeros((3, 2), dtype=int)
    np.testing.assert_array_equal(result, expected)


# Chained reshapes
@pytest.mark.array_ops
def test_reshape_roundtrip():
    """Reshape to 2D and back to 1D is identity."""

    def f(x):
        return x.reshape(2, 3).reshape(6)

    result = jacobian_sparsity(f, input_shape=(6,)).todense().astype(int)
    expected = np.eye(6, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_reshape_chain_different_shapes():
    """Chaining multiple reshapes still tracks 1-to-1 dependencies."""

    def f(x):
        return x.reshape(2, 6).reshape(3, 4).reshape(12)

    result = jacobian_sparsity(f, input_shape=(12,)).todense().astype(int)
    expected = np.eye(12, dtype=int)
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

    result = jacobian_sparsity(f, input_shape=(3,)).todense().astype(int)
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

    result = jacobian_sparsity(f, input_shape=(6,)).todense().astype(int)
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

    result = jacobian_sparsity(f, input_shape=(6,)).todense().astype(int)
    expected = np.eye(6, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_jnp_ravel():
    """jnp.ravel on a reshaped array is identity on flat indices."""

    def f(x):
        return jnp.ravel(x.reshape(2, 3))

    result = jacobian_sparsity(f, input_shape=(6,)).todense().astype(int)
    expected = np.eye(6, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_jnp_flatten():
    """ndarray.flatten() lowers through reshape."""

    def f(x):
        return x.reshape(2, 2, 3).flatten()

    result = jacobian_sparsity(f, input_shape=(12,)).todense().astype(int)
    expected = np.eye(12, dtype=int)
    np.testing.assert_array_equal(result, expected)


# Edge cases
@pytest.mark.array_ops
def test_reshape_size_one_input():
    """Reshaping a single element to various shapes with size-1 dims."""

    def f(x):
        return x.reshape(1, 1, 1).flatten()

    result = jacobian_sparsity(f, input_shape=(1,)).todense().astype(int)
    expected = np.eye(1, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_reshape_all_size_one_dims():
    """Reshape adding multiple size-1 dimensions."""

    def f(x):
        return x.reshape(1, 2, 1, 3, 1).flatten()

    result = jacobian_sparsity(f, input_shape=(6,)).todense().astype(int)
    expected = np.eye(6, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_reshape_with_dimensions_size_one():
    """Dimensions param with size-1 dims in the original shape."""

    def f(x):
        return lax.reshape(x.reshape(1, 4), (4,), dimensions=(1, 0))

    result = jacobian_sparsity(f, input_shape=(4,)).todense().astype(int)
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

    result = jacobian_sparsity(f, input_shape=(6,)).todense().astype(int)
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

    result = jacobian_sparsity(f, input_shape=(6,)).todense().astype(int)
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

    result = jacobian_sparsity(f, input_shape=(3,)).todense().astype(int)
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

    result = jacobian_sparsity(f, input_shape=(3,))
    assert result.shape == (0, 3)
    assert result.nnz == 0
