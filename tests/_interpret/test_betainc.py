"""Tests for regularized_incomplete_beta handler.

`jax.scipy.special.betainc(a, b, x)` computes the regularized incomplete beta
function I_x(a, b). It has three inputs and the Jacobian depends on which
inputs are traced vs constant.

When all inputs are traced, the Jacobian is elementwise (diagonal) for each input.
"""

import jax
import jax.numpy as jnp
import jax.scipy.special
import numpy as np
import pytest

from asdex import jacobian_sparsity


@pytest.mark.elementwise
def test_betainc_x_only():
    """When only x is traced, Jacobian is diagonal."""

    def f(x):
        return jax.scipy.special.betainc(1.0, 2.0, x)

    x = np.zeros(5)
    result = jacobian_sparsity(f, x).todense().astype(int)
    expected = np.eye(5, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_betainc_a_only():
    """When only a is traced, Jacobian is diagonal."""

    def f(a):
        x = jnp.array([0.2, 0.5, 0.8])
        return jax.scipy.special.betainc(a, 2.0, x)

    a = np.zeros(3)
    result = jacobian_sparsity(f, a).todense().astype(int)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_betainc_b_only():
    """When only b is traced, Jacobian is diagonal."""

    def f(b):
        x = jnp.array([0.2, 0.5, 0.8])
        return jax.scipy.special.betainc(1.0, b, x)

    b = np.zeros(3)
    result = jacobian_sparsity(f, b).todense().astype(int)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_betainc_all_traced():
    """When all inputs are traced, each output depends on corresponding a, b, x."""

    def f(inputs):
        a, b, x = inputs
        return jax.scipy.special.betainc(a, b, x)

    a = np.zeros(3)
    b = np.zeros(3)
    x = np.zeros(3)
    inputs = (a, b, x)

    # Jacobian wrt all inputs: 3 outputs, 9 inputs (3+3+3)
    result = jacobian_sparsity(f, inputs).todense().astype(int)
    # out[i] depends on a[i], b[i], x[i] -> three diagonals
    expected = np.array(
        [
            [1, 0, 0, 1, 0, 0, 1, 0, 0],  # out[0] <- a[0], b[0], x[0]
            [0, 1, 0, 0, 1, 0, 0, 1, 0],  # out[1] <- a[1], b[1], x[1]
            [0, 0, 1, 0, 0, 1, 0, 0, 1],  # out[2] <- a[2], b[2], x[2]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_betainc_jacobian_numerical():
    """Verify structural pattern matches numerical Jacobian."""

    def f(x):
        return jax.scipy.special.betainc(1.0, 2.0, x)

    x = jnp.array([0.2, 0.5, 0.8])
    structural = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    numerical = (np.abs(jax.jacobian(f)(x)) > 1e-10).astype(int)
    np.testing.assert_array_equal(structural, numerical)


@pytest.mark.elementwise
def test_betainc_broadcasting():
    """Broadcasting: scalar a, b with vector x."""

    def f(x):
        # Scalar a=1.0, b=2.0, vector x
        return jax.scipy.special.betainc(1.0, 2.0, x)

    x = np.zeros(4)
    result = jacobian_sparsity(f, x).todense().astype(int)
    expected = np.eye(4, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_betainc_2d():
    """2D input arrays."""

    def f(x):
        return jax.scipy.special.betainc(1.0, 2.0, x)

    shape = (3, 4)
    n = int(np.prod(shape))
    x = np.zeros(shape)
    result = jacobian_sparsity(f, x).todense().astype(int)
    expected = np.eye(n, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_betainc_size_zero():
    """Size-0 input array."""

    def f(x):
        return jax.scipy.special.betainc(1.0, 2.0, x)

    x = np.zeros((0,))
    result = jacobian_sparsity(f, x)
    assert result.shape == (0, 0)
    assert result.nnz == 0
