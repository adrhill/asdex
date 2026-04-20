"""Tests for broadcast_in_dim propagation.

https://docs.jax.dev/en/latest/_autosummary/jax.lax.broadcast_in_dim.html
"""

import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity


@pytest.mark.array_ops
def test_array_broadcast():
    """Broadcasting a non-scalar array tracks per-element dependencies precisely.

    Each output element depends on the input element it was replicated from.
    """

    def f(x):
        # x is shape (3,), reshape to (3, 1) and broadcast to (3, 2)
        col = x.reshape(3, 1)
        broadcasted = jnp.broadcast_to(col, (3, 2))
        return broadcasted.flatten()

    result = jacobian_sparsity(f, 3).todense().astype(int)
    # Output (3x2) flattened: [0,0], [0,1], [1,0], [1,1], [2,0], [2,1]
    # Each row comes from one input: out[0,1] <- in[0], out[2,3] <- in[1], etc.
    expected = np.array(
        [
            [1, 0, 0],  # out[0] <- in[0]
            [1, 0, 0],  # out[1] <- in[0]
            [0, 1, 0],  # out[2] <- in[1]
            [0, 1, 0],  # out[3] <- in[1]
            [0, 0, 1],  # out[4] <- in[2]
            [0, 0, 1],  # out[5] <- in[2]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_scalar_broadcast():
    """Broadcasting a scalar preserves per-element structure."""

    def f(x):
        # Each element broadcast independently
        return jnp.array([jnp.broadcast_to(x[0], (2,)).sum(), x[1] * 2])

    result = jacobian_sparsity(f, 2).todense().astype(int)
    expected = np.array([[1, 0], [0, 1]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_broadcast_constant():
    """Broadcasting a constant array produces zero sparsity."""

    def f(_x):
        const = jnp.array([1.0, 2.0])  # Shape (2,)
        return jnp.broadcast_to(const, (3, 2)).flatten()  # Shape (6,)

    result = jacobian_sparsity(f, 2).todense().astype(int)
    expected = np.zeros((6, 2), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_broadcast_input_add_constant():
    """Broadcasting input and adding a constant preserves input structure."""

    def f(x):
        const = jnp.array([[1.0], [2.0]])  # Shape (2, 1)
        x_col = x.reshape(2, 1)  # Shape (2, 1)
        broadcasted = jnp.broadcast_to(x_col, (2, 3))  # Shape (2, 3)
        return (broadcasted + const).flatten()

    result = jacobian_sparsity(f, 2).todense().astype(int)
    # Each row of output depends on corresponding input element
    # Output shape (2, 3) flattened: rows 0-2 from x[0], rows 3-5 from x[1]
    expected = np.array(
        [
            [1, 0],  # out[0] <- x[0]
            [1, 0],  # out[1] <- x[0]
            [1, 0],  # out[2] <- x[0]
            [0, 1],  # out[3] <- x[1]
            [0, 1],  # out[4] <- x[1]
            [0, 1],  # out[5] <- x[1]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


# Size-0 dimension


@pytest.mark.array_ops
def test_broadcast_size_zero_dim():
    """Broadcasting a zero-sized array produces an empty Jacobian.

    Zero-sized inputs have no elements, so output index sets should be empty.
    """

    def f(x):
        return jnp.broadcast_to(x[:0].reshape(0, 1), (0, 3)).flatten()

    result = jacobian_sparsity(f, 3)
    assert result.shape == (0, 3)
    assert result.nnz == 0


@pytest.mark.array_ops
def test_broadcast_expand_dims_zero():
    """expand_dims on a zero-sized array produces an empty Jacobian.

    Reproducer from GitHub issue #86.
    """

    def f(x):
        return jnp.expand_dims(x[:0], axis=1).flatten()

    result = jacobian_sparsity(f, 3)
    assert result.shape == (0, 3)
    assert result.nnz == 0
