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
from jax import lax

from asdex import jacobian, jacobian_sparsity
from tests._utils import assert_jacobian_sparsity_exact


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


@pytest.mark.bug
def test_custom_vjp_follows_primal_not_rule():
    """custom_vjp is propagated through its primal, not its backward rule.

    TODO(custom_vjp_call): trace ``fwd`` and ``bwd`` and follow the rule.
    The primal round-trips through a bitcast and has zero derivative,
    while the rule is the identity, so the precise pattern is the identity.
    Today the detected pattern is empty and misses every nonzero.
    """

    @jax.custom_vjp
    def bits_identity(x):
        return lax.bitcast_convert_type(
            lax.bitcast_convert_type(x, jnp.int32), jnp.float32
        )

    bits_identity.defvjp(lambda x: (bits_identity(x), None), lambda _, g: (g,))

    # The int32 round trip needs float32, even if another test enabled x64
    x = jnp.array([1.0, 2.0, 3.0], dtype=jnp.float32)
    with pytest.raises(AssertionError):
        assert_jacobian_sparsity_exact(bits_identity, x)


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


@pytest.mark.array_ops
def test_jit_const_output_escapes_to_outer_consumer():
    """Const values computed inside a jit call reach outer consumers.

    The identity matrix passes through the jit-wrapped function,
    so its concrete value must be forwarded
    from the inner jaxpr's outputs to the outer equation's outvars,
    mirroring the existing bounds forwarding.
    The outer matmul can then skip the known zeros,
    keeping the diagonal pattern instead of a dense fallback.
    """

    @jax.jit
    def passthrough(w, x):
        return w * 1.0, x * 1.0

    def f(x):
        w, x2 = passthrough(jnp.eye(3), x)
        return w @ x2

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)
    assert_jacobian_sparsity_exact(f, np.array([1.0, 2.0, 3.0]))


@pytest.mark.array_ops
def test_jit_const_index_chain_resolves_outer_gather():
    """A const index array computed inside jit resolves an outer gather precisely.

    The index chain crosses the jit boundary
    and then runs through ``jnp.floor_divide``,
    whose expansion uses div, sign, rem, and select_n.
    Const values must survive the whole chain
    for the gather to resolve statically instead of going dense.
    """

    def f(x):
        @jax.jit
        def make(i, xx):
            return i * 2, xx * 1.0

        idx, x2 = make(jnp.array([1, 0, 2]), x)
        return x2[jnp.floor_divide(idx, 2)]

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.array(
        [
            [0, 1, 0],  # out[0] <- x2[2 // 2] = x[1]
            [1, 0, 0],  # out[1] <- x2[0 // 2] = x[0]
            [0, 0, 1],  # out[2] <- x2[4 // 2] = x[2]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)
    assert_jacobian_sparsity_exact(f, np.array([1.0, 2.0, 3.0]))


@pytest.mark.array_ops
def test_reused_jit_jaxpr_does_not_leak_stale_consts():
    """A cached jit jaxpr reused with a data-dependent input forgets old consts.

    ``jnp.clip`` is jit-wrapped,
    so both calls below share one inner jaxpr and therefore the same inner vars.
    The first call stores const values on those vars.
    The second call receives a data-dependent index,
    so the stale consts must not resolve its gather as if it were static.
    """

    def f(x):
        i = jnp.clip(jnp.arange(3), 0, 5)
        j = jnp.clip(jnp.floor(x[:3]).astype(int), 0, 5)
        return jnp.concatenate([x[i], x[j]])

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    expected = np.zeros((6, 6), dtype=int)
    # out[k] <- x[k] for the static index
    expected[:3, :3] = np.eye(3, dtype=int)
    # The data-dependent index can reach any element of x
    expected[3:, :] = 1
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_reused_jit_jaxpr_in_while_body_does_not_leak_stale_consts():
    """A cached jit jaxpr called on a loop carry forgets consts from an earlier call.

    ``jnp.take`` is jit-wrapped and first called with a static index,
    which stores a const on the inner index var.
    Inside the loop body the same jaxpr receives the carry,
    which has no const value,
    so the gather must not reuse the stale index 0.
    """

    def f(x):
        first = jnp.take(x, jnp.int32(0))

        def body(carry):
            i, acc = carry
            return i + 1, acc + jnp.take(x, i)

        _, acc = jax.lax.while_loop(
            lambda carry: carry[0] < 3, body, (jnp.int32(0), jnp.float32(0))
        )
        return jnp.stack([first, acc])

    x = np.zeros(4, dtype=np.float32)
    result = jacobian_sparsity(f, x).todense().astype(int)
    expected = np.array(
        [
            [1, 0, 0, 0],  # first <- x[0]
            [1, 1, 1, 1],  # acc <- x[i] for a loop-carried i
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)
