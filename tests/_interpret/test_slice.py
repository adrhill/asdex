"""Tests for slice propagation.

https://docs.jax.dev/en/latest/_autosummary/jax.lax.slice.html
"""

import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity


@pytest.mark.array_ops
def test_multidim_slice():
    """Multi-dimensional slice tracks per-element dependencies precisely.

    Each output element depends on exactly one input element.
    """

    def f(x):
        # Reshape to 2D and slice in multiple dimensions
        mat = x.reshape(3, 4)
        sliced = mat[0:2, 1:3]  # 2D slice extracts 2x2 submatrix
        return sliced.flatten()

    result = jacobian_sparsity(f, 12).todense().astype(int)
    # Input (3x4): indices 0-11 in row-major order
    # Slice [0:2, 1:3] extracts: [0,1]=1, [0,2]=2, [1,1]=5, [1,2]=6
    expected = np.zeros((4, 12), dtype=int)
    expected[0, 1] = 1  # out[0] <- in[1]
    expected[1, 2] = 1  # out[1] <- in[2]
    expected[2, 5] = 1  # out[2] <- in[5]
    expected[3, 6] = 1  # out[3] <- in[6]
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_roll():
    """jnp.roll correctly tracks the cyclic permutation.

    output[i] depends on input[(i-shift) % n].
    """

    def f(x):
        return jnp.roll(x, shift=1)

    result = jacobian_sparsity(f, 3).todense().astype(int)
    # Precise: cyclic permutation matrix
    # output[0] <- input[2], output[1] <- input[0], output[2] <- input[1]
    expected = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_slice_constant_array():
    """Slicing a constant array produces zero sparsity."""

    def f(_x):
        const = jnp.array([1.0, 2.0, 3.0, 4.0])
        return const[1:3]  # Slice constant, no input dependency

    result = jacobian_sparsity(f, 2).todense().astype(int)
    expected = np.zeros((2, 2), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_slice_mixed_with_constant():
    """Slicing input and concatenating with sliced constant."""

    def f(x):
        const = jnp.array([10.0, 20.0])
        sliced_x = x[1:3]  # x[1], x[2]
        sliced_const = const[:1]  # const[0]
        return jnp.concatenate([sliced_const, sliced_x])

    result = jacobian_sparsity(f, 4).todense().astype(int)
    # Output: [const[0], x[1], x[2]]
    expected = np.array(
        [
            [0, 0, 0, 0],  # out[0] <- const[0]
            [0, 1, 0, 0],  # out[1] <- x[1]
            [0, 0, 1, 0],  # out[2] <- x[2]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


# Size-0 dimension


@pytest.mark.array_ops
def test_slice_to_zero_size():
    """Slicing to an empty range produces a zero-row Jacobian."""

    def f(x):
        return x[:0]

    result = jacobian_sparsity(f, 3)
    assert result.shape == (0, 3)
    assert result.nnz == 0
