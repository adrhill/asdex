"""Tests for vmap sparsity (block-diagonal patterns)."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity


@pytest.mark.vmap
def test_vmap_elementwise():
    """Vmapped element-wise operations produce diagonal sparsity."""

    def f(x):
        return jax.vmap(lambda xi: xi**2)(x.reshape(3, 2)).flatten()

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    # Element-wise squaring: each output depends on exactly one input
    expected = np.eye(6, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.vmap
def test_vmap_block_diagonal():
    """Vmapped functions with internal dependencies produce block-diagonal sparsity."""

    def g(x):
        return jnp.array([x[0] + x[1], x[0] * x[1]])

    def f(x):
        return jax.vmap(g)(x.reshape(2, 2)).flatten()

    result = jacobian_sparsity(f, np.zeros(4)).todense().astype(int)
    # Two 2x2 dense blocks on the diagonal
    expected = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.vmap
def test_vmap_larger_batch():
    """Vmapped reduction produces block-diagonal with dense rows."""

    def g(x):
        return jnp.array([jnp.sum(x)])

    def f(x):
        return jax.vmap(g)(x.reshape(4, 3)).flatten()

    result = jacobian_sparsity(f, np.zeros(12)).todense().astype(int)
    # Each output depends on 3 consecutive inputs (one batch)
    expected = np.array(
        [
            [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)
