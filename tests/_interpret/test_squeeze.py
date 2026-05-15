"""Tests for squeeze propagation.

https://docs.jax.dev/en/latest/_autosummary/jax.lax.squeeze.html
"""

import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity

_SHAPES_AND_DIMS = [
    pytest.param((5,), (), id="1d_empty"),
    pytest.param((1,), (0,), id="1d_squeeze"),
    pytest.param((3, 1), (), id="2d_empty"),
    pytest.param((3, 1), (1,), id="2d_squeeze_dim1"),
    pytest.param((1, 4), (0,), id="2d_squeeze_dim0"),
    pytest.param((1, 1), (0,), id="2d_squeeze_first"),
    pytest.param((1, 1), (1,), id="2d_squeeze_second"),
    pytest.param((1, 1), (0, 1), id="2d_squeeze_both"),
    pytest.param((2, 1, 3), (), id="3d_empty"),
    pytest.param((2, 1, 3), (1,), id="3d_squeeze_middle"),
    pytest.param((1, 2, 1), (0, 2), id="3d_squeeze_outer"),
    pytest.param((1, 1, 1), (0, 1, 2), id="3d_squeeze_all"),
]


# Core squeeze tests
@pytest.mark.array_ops
@pytest.mark.parametrize(("shape", "dimensions"), _SHAPES_AND_DIMS)
def test_squeeze(shape, dimensions):
    """Squeeze is identity for sparsity: each output depends on exactly one input."""
    n = int(np.prod(shape))

    def f(x):
        return lax.squeeze(x.reshape(shape), dimensions=dimensions).flatten()

    result = jacobian_sparsity(f, np.zeros(n)).todense().astype(int)
    expected = np.eye(n, dtype=int)
    np.testing.assert_array_equal(result, expected)


# Const propagation tests
@pytest.mark.array_ops
def test_squeeze_constant():
    """Squeezing a constant array produces zero sparsity."""

    def f(_x):
        const = jnp.array([[1.0, 2.0, 3.0]])  # Shape (1, 3)
        return jnp.squeeze(const, axis=0)  # Shape (3,)

    result = jacobian_sparsity(f, np.zeros(2)).todense().astype(int)
    expected = np.zeros((3, 2), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_squeeze_propagates_consts_to_gather():
    """Squeeze in a const chain preserves const values for downstream gather.

    When a closure array goes through ``slice → squeeze``,
    the squeezed result must remain in ``state_consts``
    so that downstream gather/scatter can resolve indices precisely.
    """
    # Closure array: each row selects a different pair of input indices.
    index_table = jnp.array([[0], [2], [4]])  # (3, 1)

    def f(x):
        # slice → squeeze extracts the column, producing a 1D index array.
        col = index_table[:, 0]  # lowers to slice + squeeze
        return x[col]

    result = jacobian_sparsity(f, np.zeros(5)).todense().astype(int)
    expected = np.array(
        [
            [1, 0, 0, 0, 0],  # out[0] <- x[0]
            [0, 0, 1, 0, 0],  # out[1] <- x[2]
            [0, 0, 0, 0, 1],  # out[2] <- x[4]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_squeeze_const_chain_with_select_n():
    """Squeeze preserves consts through a ``slice → squeeze → lt → select_n`` chain.

    This is the pattern JAX emits for negative-index wrapping:
    ``lt(idx, 0)`` checks for negatives,
    ``select_n`` picks ``idx`` or ``idx + size``.
    If squeeze breaks const propagation,
    the gather sees dynamic indices and falls back to conservative.
    """
    # Simulate the pattern from sif2jax network flow problems.
    index_array = jnp.array([[1, 3], [0, 2]])  # (2, 2) closure const

    def f(x):
        src = index_array[:, 0]  # slice + squeeze → [1, 0]
        dst = index_array[:, 1]  # slice + squeeze → [3, 2]
        return x[src] * x[dst]

    result = jacobian_sparsity(f, np.zeros(5)).todense().astype(int)
    expected = np.array(
        [
            [0, 1, 0, 1, 0],  # out[0] <- x[1] * x[3]
            [1, 0, 1, 0, 0],  # out[1] <- x[0] * x[2]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


# Size-0 dimension


@pytest.mark.array_ops
def test_squeeze_zero_size():
    """Squeezing a zero-sized array with a size-1 dimension."""

    def f(x):
        return jnp.squeeze(x[:0].reshape(0, 1), axis=1)

    result = jacobian_sparsity(f, np.zeros(3))
    assert result.shape == (0, 3)
    assert result.nnz == 0
