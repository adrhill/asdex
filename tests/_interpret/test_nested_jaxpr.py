"""Tests for state_consts propagation into nested jaxprs.

Verifies that seed_const_vals and forward_const_vals correctly transfer
concrete index values into jit-wrapped and custom_jvp functions,
enabling precise gather/scatter tracking instead of conservative fallback.

https://docs.jax.dev/en/latest/jaxpr.html
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity


@pytest.mark.array_ops
def test_jit_closure_captured_index():
    """jit-wrapped function with closure-captured index resolves gather precisely.

    The index array becomes a constvar in the nested ClosedJaxpr.
    seed_const_vals populates state_consts for it,
    enabling the gather handler to track precise element dependencies.
    Without the fix, the result is dense.
    """
    indices = jnp.array([2, 0, 1])

    @jax.jit
    def permute(x):
        return x[indices]

    def f(x):
        return permute(x)

    result = jacobian_sparsity(f, 3).todense().astype(int)
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
    forward_const_vals transfers its const_val to the call_jaxpr's invar,
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

    result = jacobian_sparsity(f, 3).todense().astype(int)
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
