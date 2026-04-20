"""Tests for while_loop propagation.

https://docs.jax.dev/en/latest/_autosummary/jax.lax.while_loop.html
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity


@pytest.mark.control_flow
def test_while_simple_accumulation():
    """while_loop that accumulates carry + const preserves carry dependencies.

    The body adds a constant to the carry,
    so output depends only on the initial carry (identity pattern).
    """

    def f(x):
        def body(carry):
            return carry + 1.0

        def cond(carry):
            return jnp.sum(carry) < 100.0

        return jax.lax.while_loop(cond, body, x)

    result = jacobian_sparsity(f, input_shape=(3,)).todense().astype(int)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_while_dependency_spreading():
    """while_loop where the body mixes carry elements spreads dependencies.

    The body shifts elements cyclically,
    so after enough iterations all outputs depend on all inputs.
    """

    def f(x):
        def body(carry):
            return jnp.roll(carry, 1)

        def cond(carry):
            return jnp.sum(carry) < 100.0

        return jax.lax.while_loop(cond, body, x)

    result = jacobian_sparsity(f, input_shape=(3,)).todense().astype(int)
    # Rolling mixes all elements after enough iterations
    expected = np.ones((3, 3), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_while_immediate_convergence():
    """while_loop with identity body converges in one iteration."""

    def f(x):
        def body(carry):
            return carry

        def cond(carry):
            return jnp.sum(carry) < 100.0

        return jax.lax.while_loop(cond, body, x)

    result = jacobian_sparsity(f, input_shape=(4,)).todense().astype(int)
    expected = np.eye(4, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_while_tuple_carry():
    """while_loop with tuple carry: (array, scalar).

    The body adds a constant to each carry element independently,
    so array output depends only on the initial array (identity).
    """

    def f(x):
        def body(carry):
            arr, count = carry
            return (arr + 1.0, count + 1.0)

        def cond(carry):
            _, count = carry
            return count < 5.0

        arr_out, _ = jax.lax.while_loop(cond, body, (x, 0.0))
        return arr_out

    result = jacobian_sparsity(f, input_shape=(3,)).todense().astype(int)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_while_carry_interaction():
    """while_loop where one carry variable flows into another.

    The body accumulates the sum of one array into a scalar,
    so the scalar output depends on all elements of the array input.
    """

    def f(x):
        def body(carry):
            a, b = carry
            return (a, b + jnp.sum(a))

        def cond(carry):
            _, b = carry
            return b < 100.0

        _, total = jax.lax.while_loop(cond, body, (x, 0.0))
        return jnp.array([total])

    result = jacobian_sparsity(f, input_shape=(3,)).todense().astype(int)
    expected = np.ones((1, 3), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_while_with_closure_const():
    """while_loop where the body captures an external array via closure.

    The closure constant doesn't add input dependencies,
    so elementwise multiply with a constant preserves the diagonal pattern.
    """
    weights = jnp.array([1.0, 0.0, 1.0])

    def f(x):
        def body(carry):
            return carry * weights

        def cond(carry):
            return jnp.sum(carry) < 100.0

        return jax.lax.while_loop(cond, body, x)

    result = jacobian_sparsity(f, input_shape=(3,)).todense().astype(int)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_while_body_closure_captured_index():
    """While body uses a closure-captured index array for gather.

    The gather projects all carry elements onto carry[0].
    Without forwarding state_consts into the body jaxpr,
    the gather falls back to conservative and the result is dense.
    With the fix, the fixed point is sparse.
    """
    indices = jnp.array([0, 0, 0])

    def f(x):
        def body(carry):
            return carry[indices]

        def cond(carry):
            return jnp.sum(carry) < 100.0

        return jax.lax.while_loop(cond, body, x)

    result = jacobian_sparsity(f, input_shape=(3,)).todense().astype(int)
    # 0 iterations: identity -> out[i]<-{i}
    # 1+ iterations: all read carry[0] -> out[i]<-{0}
    # Union: out[0]<-{0}, out[1]<-{0,1}, out[2]<-{0,2}
    expected = np.array(
        [
            [1, 0, 0],
            [1, 1, 0],
            [1, 0, 1],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)
