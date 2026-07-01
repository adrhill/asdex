"""Tests for state_consts propagation into nested jaxprs.

Verifies that _seed_const_vals and _forward_const_vals correctly transfer
concrete index values into jit-wrapped and custom_jvp functions,
enabling precise gather/scatter tracking instead of conservative fallback.

https://docs.jax.dev/en/latest/jaxpr.html
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian, jacobian_sparsity


@pytest.mark.array_ops
def test_jit_closure_captured_index():
    """jit-wrapped function with closure-captured index resolves gather precisely.

    The index array becomes a constvar in the nested ClosedJaxpr.
    _seed_const_vals populates state_consts for it,
    enabling the gather handler to track precise element dependencies.
    Without the fix, the result is dense.
    """
    indices = jnp.array([2, 0, 1])

    @jax.jit
    def permute(x):
        return x[indices]

    def f(x):
        return permute(x)

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    # Permutation: out[0]←x[2], out[1]←x[0], out[2]←x[1]
    expected = np.array(
        [
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_custom_jvp_closure_captured_index():
    """custom_jvp function with closure-captured index resolves gather precisely.

    The index array is hoisted to the top-level jaxpr and passed as an operand.
    _forward_const_vals transfers its const_val to the call_jaxpr's invar,
    enabling the gather handler to track precise element dependencies.
    Without the fix, the result is dense.
    """
    indices = jnp.array([2, 0, 1])

    @jax.custom_jvp
    def permute(x):
        return x[indices]

    @permute.defjvp
    def permute_jvp(primals, tangents):
        (x,) = primals
        (t,) = tangents
        return permute(x), permute(t)

    def f(x):
        return permute(x)

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    # Permutation: out[0]←x[2], out[1]←x[0], out[2]←x[1]
    expected = np.array(
        [
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_remat2_checkpoint():
    """remat2 primitive traces through the wrapped jaxpr.

    jax.checkpoint (remat) wraps a computation for rematerialization during backprop.
    The sparsity pattern should be identical to the unwrapped computation.
    """

    @jax.checkpoint
    def f(x):
        y = jnp.sin(x)
        return jnp.cos(y)

    x = jnp.array([0.0, 1.0, 2.0])
    result = jacobian_sparsity(f, x).todense().astype(int)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_remat2_closure_captured_index():
    """remat2 with closure-captured index resolves gather precisely.

    Same as test_jit_closure_captured_index but with jax.checkpoint.
    """
    indices = jnp.array([2, 0, 1])

    @jax.checkpoint
    def permute(x):
        return x[indices]

    def f(x):
        return permute(x)

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.array(
        [
            [0, 0, 1],  # out[0] ← x[2]
            [1, 0, 0],  # out[1] ← x[0]
            [0, 1, 0],  # out[2] ← x[1]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_remat2_decompression():
    """remat2 works with full Jacobian computation, not just sparsity detection."""

    @jax.checkpoint
    def f(x):
        return x**2

    x = jnp.array([1.0, 2.0, 3.0])
    J = jacobian(f, x, output_format="dense")(x)
    expected = np.diag([2.0, 4.0, 6.0])
    np.testing.assert_allclose(J, expected)


@pytest.mark.elementwise
def test_remat2_nested():
    """Nested checkpoints trace through both layers correctly.

    Each checkpoint wraps its computation in a remat2 primitive.
    Nested checkpoints produce nested remat2 primitives,
    both of which must be traced through.
    """

    @jax.checkpoint
    def inner(x):
        return jnp.sin(x)

    @jax.checkpoint
    def outer(x):
        return jnp.cos(inner(x))

    x = jnp.array([1.0, 2.0, 3.0])
    result = jacobian_sparsity(outer, x).todense().astype(int)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_remat2_differentiated():
    """remat2 with differentiated=True traces correctly.

    When taking the gradient of a checkpointed function,
    the jaxpr contains remat2 with differentiated=True.
    This rematerializes the forward computation during backprop.
    """

    @jax.checkpoint
    def f_inner(x):
        return jnp.sum(jnp.sin(x))

    def grad_f(x):
        return jax.grad(f_inner)(x)

    x = jnp.array([1.0, 2.0, 3.0])
    result = jacobian_sparsity(grad_f, x).todense().astype(int)
    # d/dx[cos(x_i)] only depends on x_i
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)
