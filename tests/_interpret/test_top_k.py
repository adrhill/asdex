"""Tests for the prop_top_k handler.

The values output has reduction-along-last-axis sparsity.
The indices output has zero derivative.
"""

import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity


def _top_k_values_jacobian(in_shape: tuple[int, ...], k: int) -> np.ndarray:
    """Build the expected Jacobian for the values output of top_k.

    Each value output depends on all inputs along the last axis
    in its batch slice.
    """
    n_in = int(np.prod(in_shape))
    last_dim = in_shape[-1]
    batch_shape = in_shape[:-1]
    n_batches = int(np.prod(batch_shape)) if batch_shape else 1
    n_out = n_batches * k

    expected = np.zeros((n_out, n_in), dtype=int)
    for b in range(n_batches):
        in_start = b * last_dim
        out_start = b * k
        for j in range(k):
            for i in range(last_dim):
                expected[out_start + j, in_start + i] = 1
    return expected


# Core shape tests
_SHAPES_AND_K = [
    pytest.param((5,), 1, id="1d_k1"),
    pytest.param((5,), 3, id="1d_k3"),
    pytest.param((5,), 5, id="1d_k_equals_n"),
    pytest.param((1,), 1, id="1d_size_one"),
    pytest.param((3, 4), 2, id="2d"),
    pytest.param((3, 1), 1, id="2d_last_dim_one"),
    pytest.param((1, 5), 3, id="2d_batch_one"),
    pytest.param((2, 3, 4), 2, id="3d"),
    pytest.param((2, 1, 4), 3, id="3d_middle_one"),
    pytest.param((2, 3, 1, 5), 2, id="4d"),
]


@pytest.mark.array_ops
@pytest.mark.parametrize(("in_shape", "k"), _SHAPES_AND_K)
def test_top_k_values(in_shape, k):
    """Values output: each element depends on all inputs in its batch slice."""
    n_in = int(np.prod(in_shape))

    def f(x):
        vals, _ = lax.top_k(x.reshape(in_shape), k)
        return vals.flatten()

    result = jacobian_sparsity(f, input_shape=(n_in,)).todense().astype(int)
    expected = _top_k_values_jacobian(in_shape, k)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
@pytest.mark.parametrize(("in_shape", "k"), _SHAPES_AND_K)
def test_top_k_indices_zero_derivative(in_shape, k):
    """Indices output has zero derivative — no input dependencies."""
    n_in = int(np.prod(in_shape))
    batch_size = int(np.prod(in_shape[:-1])) if len(in_shape) > 1 else 1
    n_out = batch_size * k

    def f(x):
        _, idx = lax.top_k(x.reshape(in_shape), k)
        return idx.flatten().astype(float)

    result = jacobian_sparsity(f, input_shape=(n_in,)).todense().astype(int)
    expected = np.zeros((n_out, n_in), dtype=int)
    np.testing.assert_array_equal(result, expected)


# Compositions
@pytest.mark.array_ops
def test_top_k_after_broadcast():
    """top_k after broadcast: non-contiguous input dependencies."""

    def f(x):
        # x.shape = (3,), broadcast to (2, 3), then top_k along last axis
        vals, _ = lax.top_k(jnp.broadcast_to(x, (2, 3)), 2)
        return vals.flatten()

    result = jacobian_sparsity(f, input_shape=(3,)).todense().astype(int)
    # Each row of the broadcast shares the same 3 inputs.
    # top_k(k=2) per row: 2 outputs per row, each depending on all 3 inputs.
    expected = np.array(
        [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1],
        ]
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_top_k_chained():
    """Chained top_k: top_k of top_k narrows selection further."""

    def f(x):
        vals, _ = lax.top_k(x.reshape(2, 4), 3)  # (2, 3)
        vals2, _ = lax.top_k(vals, 2)  # (2, 2)
        return vals2.flatten()

    result = jacobian_sparsity(f, input_shape=(8,)).todense().astype(int)
    # Row 0 outputs depend on inputs 0..3, row 1 on inputs 4..7
    expected = np.zeros((4, 8), dtype=int)
    expected[0:2, 0:4] = 1
    expected[2:4, 4:8] = 1
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_top_k_then_reduce():
    """top_k followed by sum along batch axis."""

    def f(x):
        vals, _ = lax.top_k(x.reshape(2, 3), 2)  # (2, 2)
        return jnp.sum(vals, axis=0)  # (2,)

    result = jacobian_sparsity(f, input_shape=(6,)).todense().astype(int)
    # Sum over batch: each output depends on all 6 inputs
    expected = np.ones((2, 6), dtype=int)
    np.testing.assert_array_equal(result, expected)


# High-level API
@pytest.mark.array_ops
def test_jnp_top_k_1d():
    """jnp.lax.top_k on a 1D input."""

    def f(x):
        vals, _ = lax.top_k(x, 2)
        return vals

    result = jacobian_sparsity(f, input_shape=(4,)).todense().astype(int)
    expected = np.ones((2, 4), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_top_k_values_used_in_arithmetic():
    """Using top_k values in downstream arithmetic preserves dependencies."""

    def f(x):
        vals, _ = lax.top_k(x.reshape(2, 3), 2)
        return (vals * 2 + 1).flatten()

    result = jacobian_sparsity(f, input_shape=(6,)).todense().astype(int)
    expected = _top_k_values_jacobian((2, 3), 2)
    np.testing.assert_array_equal(result, expected)


# Size-0 dimension


@pytest.mark.array_ops
def test_top_k_zero_size():
    """top_k with k=0 on a zero-sized array produces an empty Jacobian."""

    def f(x):
        vals, _ = lax.top_k(x[:0], 0)
        return vals

    result = jacobian_sparsity(f, input_shape=(3,))
    assert result.shape == (0, 3)
    assert result.nnz == 0
