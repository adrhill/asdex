"""Tests for associative_scan (jax.lax.associative_scan).

``associative_scan`` is not a JAX primitive.
It decomposes into ``slice``, ``add``, ``pad``, ``concatenate``, ``reshape``,
and ``rev`` — all of which have precise handlers.

These tests verify that the decomposition produces correct sparsity patterns
without needing a dedicated handler.

https://docs.jax.dev/en/latest/_autosummary/jax.lax.associative_scan.html
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian, jacobian_sparsity


@pytest.mark.control_flow
def test_associative_scan_cumsum_1d():
    """Cumulative sum via associative_scan: lower-triangular pattern.

    cumsum[i] = sum(x[0..i]),
    so output i depends on inputs 0 through i.
    """

    def f(x):
        return jax.lax.associative_scan(jnp.add, x)

    result = jacobian_sparsity(f, input_shape=(4,)).todense().astype(int)
    expected = np.tril(np.ones((4, 4), dtype=int))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_associative_scan_cumprod_1d():
    """Cumulative product via associative_scan.

    cumprod[i] = prod(x[0..i]),
    so output i depends on inputs 0 through i.
    """

    def f(x):
        return jax.lax.associative_scan(jnp.multiply, x)

    result = jacobian_sparsity(f, input_shape=(4,)).todense().astype(int)
    expected = np.tril(np.ones((4, 4), dtype=int))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_associative_scan_cummax_1d():
    """Cumulative max via associative_scan.

    ``jnp.maximum`` decomposes into elementwise ``select_n``,
    so each output depends on all preceding inputs.
    Lower-triangular pattern, same as cumulative sum.
    """

    def f(x):
        return jax.lax.associative_scan(jnp.maximum, x)

    result = jacobian_sparsity(f, input_shape=(4,)).todense().astype(int)
    expected = np.tril(np.ones((4, 4), dtype=int))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_associative_scan_2d_axis0():
    """Associative scan along axis=0 of a 2D array.

    Each column scans independently,
    so output (i, j) depends on inputs (0..i, j).
    """

    def f(x):
        return jax.lax.associative_scan(jnp.add, x.reshape(3, 2), axis=0).ravel()

    result = jacobian_sparsity(f, input_shape=(6,)).todense().astype(int)
    # 3x2 array scanned along axis 0: columns are independent
    expected = np.array(
        [
            # x00 x01 x10 x11 x20 x21
            [1, 0, 0, 0, 0, 0],  # y00 = x00
            [0, 1, 0, 0, 0, 0],  # y01 = x01
            [1, 0, 1, 0, 0, 0],  # y10 = x00 + x10
            [0, 1, 0, 1, 0, 0],  # y11 = x01 + x11
            [1, 0, 1, 0, 1, 0],  # y20 = x00 + x10 + x20
            [0, 1, 0, 1, 0, 1],  # y21 = x01 + x11 + x21
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_associative_scan_2d_axis1():
    """Associative scan along axis=1 of a 2D array.

    Each row scans independently,
    so output (i, j) depends on inputs (i, 0..j).
    """

    def f(x):
        return jax.lax.associative_scan(jnp.add, x.reshape(2, 3), axis=1).ravel()

    result = jacobian_sparsity(f, input_shape=(6,)).todense().astype(int)
    # 2x3 array scanned along axis 1: rows are independent
    expected = np.array(
        [
            # x00 x01 x02 x10 x11 x12
            [1, 0, 0, 0, 0, 0],  # y00 = x00
            [1, 1, 0, 0, 0, 0],  # y01 = x00 + x01
            [1, 1, 1, 0, 0, 0],  # y02 = x00 + x01 + x02
            [0, 0, 0, 1, 0, 0],  # y10 = x10
            [0, 0, 0, 1, 1, 0],  # y11 = x10 + x11
            [0, 0, 0, 1, 1, 1],  # y12 = x10 + x11 + x12
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_associative_scan_reverse():
    """Reversed associative scan: upper-triangular pattern.

    reverse=True scans from right to left,
    so output i depends on inputs i through n-1.
    """

    def f(x):
        return jax.lax.associative_scan(jnp.add, x, reverse=True)

    result = jacobian_sparsity(f, input_shape=(4,)).todense().astype(int)
    expected = np.triu(np.ones((4, 4), dtype=int))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
@pytest.mark.filterwarnings("ignore::asdex.DenseColoringWarning")
def test_associative_scan_jacobian_values():
    """Verify sparse Jacobian values match dense jax.jacobian for cumulative sum."""

    def f(x):
        return jax.lax.associative_scan(jnp.add, x)

    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    sparse_jac = jacobian(f, x.shape)(x).todense()
    dense_jac = np.array(jax.jacobian(f)(x))
    np.testing.assert_allclose(sparse_jac, dense_jac)


@pytest.mark.control_flow
def test_associative_scan_length_one():
    """Single-element input: identity, diagonal pattern."""

    def f(x):
        return jax.lax.associative_scan(jnp.add, x)

    result = jacobian_sparsity(f, input_shape=(1,)).todense().astype(int)
    expected = np.array([[1]], dtype=int)
    np.testing.assert_array_equal(result, expected)
