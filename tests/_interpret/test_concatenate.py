"""Tests for concatenate propagation.

https://docs.jax.dev/en/latest/_autosummary/jax.lax.concatenate.html
"""

import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity


@pytest.mark.array_ops
def test_stack():
    """jnp.stack tracks per-element dependencies precisely."""

    def f(x):
        a, b = x[:2], x[2:]
        return jnp.stack([a, b]).flatten()

    result = jacobian_sparsity(f, input_shape=(4,)).todense().astype(int)
    # Each output depends on exactly one input (identity)
    expected = np.eye(4, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_nested_slice_concat():
    """Multiple 1D slices followed by concatenate should preserve structure."""

    def f(x):
        a = x[:2]
        b = x[2:]
        return jnp.concatenate([b, a])  # [x2, x3, x0, x1]

    result = jacobian_sparsity(f, input_shape=(4,)).todense().astype(int)
    # Permutation: swap first 2 and last 2
    expected = np.array(
        [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=int
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_empty_concatenate():
    """Concatenating with empty arrays preserves correct sparsity."""

    def f(x):
        empty = jnp.array([])
        return jnp.concatenate([empty, x, empty])

    result = jacobian_sparsity(f, input_shape=(2,))
    expected = np.eye(2, dtype=int)
    np.testing.assert_array_equal(result.todense().astype(int), expected)


@pytest.mark.array_ops
def test_concatenate_with_constants():
    """Concatenating non-empty constants with input tracks dependencies correctly.

    Constants have no input dependency,
    so only input elements contribute non-zeros.
    """

    def f(x):
        a = jnp.array([1.0])
        b = jnp.array([2.0, 3.0])
        return jnp.concatenate([a, x, b])

    result = jacobian_sparsity(f, input_shape=(2,)).todense().astype(int)
    # Output: [a[0], x[0], x[1], b[0], b[1]] (5 elements)
    # Only x[0] and x[1] depend on input
    expected = np.array(
        [
            [0, 0],  # out[0] <- a[0] (constant)
            [1, 0],  # out[1] <- x[0]
            [0, 1],  # out[2] <- x[1]
            [0, 0],  # out[3] <- b[0] (constant)
            [0, 0],  # out[4] <- b[1] (constant)
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_concatenate_mixed_empty_and_nonempty_constants():
    """Concatenating empty and non-empty constants with input works correctly."""

    def f(x):
        const = jnp.array([1.0])
        empty = jnp.array([])
        return jnp.concatenate([const, empty, x])

    result = jacobian_sparsity(f, input_shape=(2,)).todense().astype(int)
    # Output: [const[0], x[0], x[1]] (3 elements)
    expected = np.array(
        [
            [0, 0],  # out[0] <- const[0] (constant)
            [1, 0],  # out[1] <- x[0]
            [0, 1],  # out[2] <- x[1]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_all_constants_no_input_dependency():
    """Output that depends only on constants has all-zero sparsity."""

    def f(x):
        a = jnp.array([1.0, 2.0])
        b = jnp.array([3.0])
        return jnp.concatenate([a, b])

    result = jacobian_sparsity(f, input_shape=(2,)).todense().astype(int)
    # Output has no dependency on input at all
    expected = np.zeros((3, 2), dtype=int)
    np.testing.assert_array_equal(result, expected)


# Size-0 dimension


@pytest.mark.array_ops
def test_concatenate_with_zero_sized():
    """Concatenating a zero-sized slice with a normal array.

    The zero-sized part contributes no elements;
    the result matches the non-empty input.
    """

    def f(x):
        return jnp.concatenate([x, x[:0]])

    result = jacobian_sparsity(f, input_shape=(3,)).todense().astype(int)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)
