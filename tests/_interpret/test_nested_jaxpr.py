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
from jax._src import config
from jax.ad_checkpoint import checkpoint_name

from asdex import hessian, jacobian, jacobian_sparsity
from tests._utils import assert_hessian_sparsity_exact, assert_jacobian_sparsity_exact

try:
    from jax._src.hijax import HiPrim
except ImportError:  # JAX < 0.11.2 names the HiJax primitive base class differently
    from jax._src.hijax import VJPHiPrimitive as HiPrim  # ty: ignore[unresolved-import]


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


@pytest.fixture
def remat3():
    """Trace ``jax.checkpoint`` and ``checkpoint_name`` with JAX's ``jax_remat3`` flag enabled.

    ``jax_remat3`` switches to a new implementation of rematerialization built on HiJax.
    It is off by default in JAX 0.11,
    and users opt in with ``jax.config.update("jax_remat3", True)``.
    Instead of a ``remat2`` equation, ``jax.checkpoint`` then emits ``call_hi_primitive``
    wrapping a ``RematTraced``, and ``checkpoint_name`` one wrapping a ``CheckpointName``.
    ``jax.custom_remat``, which emits a ``CustomRemat``, requires the flag.
    The remat3 tests below use this fixture to cover those equations,
    so asdex keeps working for users who opt in,
    and once JAX makes the new implementation the default.
    """
    with config.remat3(True):
        yield


@pytest.mark.elementwise
@pytest.mark.usefixtures("remat3")
def test_remat3_checkpoint():
    """call_hi_primitive with RematTraced traces through the wrapped jaxpr."""

    @jax.checkpoint
    def f(x):
        return jnp.cos(jnp.sin(x)) * x[::-1]

    assert_jacobian_sparsity_exact(f, jnp.array([1.0, 2.0, 3.0, 4.0]))


@pytest.mark.array_ops
@pytest.mark.usefixtures("remat3")
def test_remat3_closure_captured_index():
    """A closure-captured index enters RematTraced as an operand and resolves the gather.

    remat3 hoists the captured index array to a const of the outer jaxpr,
    so its value must be forwarded into the wrapped jaxpr.
    """
    indices = jnp.array([2, 0, 1])

    @jax.checkpoint
    def f(x):
        return x[indices]

    assert_jacobian_sparsity_exact(f, jnp.array([1.0, 2.0, 3.0]))


@pytest.mark.array_ops
@pytest.mark.usefixtures("remat3")
def test_remat3_multiple_operands_and_outputs():
    """Pytree inputs and outputs of RematTraced stay aligned with the flat eqn variables."""

    @jax.checkpoint
    def g(d):
        return {"prod": d["a"] * d["b"][:2], "sum": jnp.sum(d["b"])}

    def f(x):
        out = g({"a": x[:2], "b": x[2:]})
        return jnp.concatenate([out["prod"], out["sum"][None]])

    assert_jacobian_sparsity_exact(f, jnp.array([1.0, 2.0, 3.0, 4.0, 5.0]))


@pytest.mark.elementwise
@pytest.mark.usefixtures("remat3")
def test_remat3_nested():
    """Nested checkpoints produce nested RematTraced primitives, both traced through."""

    @jax.checkpoint
    def inner(x):
        return jnp.sin(x)

    @jax.checkpoint
    def outer(x):
        return jnp.cos(inner(x))

    assert_jacobian_sparsity_exact(outer, jnp.array([1.0, 2.0, 3.0]))


@pytest.mark.hessian
@pytest.mark.usefixtures("remat3")
def test_remat3_differentiated():
    """The gradient of a remat3 checkpoint traces through RematTraced and optimization_barrier.

    Under remat3, the backward pass pins the rematerialized inputs
    with ``optimization_barrier`` before calling RematTraced again.
    """

    def f(x):
        return jnp.sum(jax.checkpoint(lambda z: jnp.sin(z) * jnp.roll(z, 1))(x))

    assert_hessian_sparsity_exact(f, jnp.array([1.0, 2.0, 3.0, 4.0]))


@pytest.mark.hessian
@pytest.mark.usefixtures("remat3")
def test_remat3_hessian_decompression():
    """The sparse Hessian of a remat3 checkpoint matches the dense JAX Hessian."""

    def f(x):
        return jnp.sum(jax.checkpoint(lambda z: jnp.sin(z) * jnp.roll(z, 1))(x))

    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    H = hessian(f, x, output_format="dense")(x)
    np.testing.assert_allclose(H, jax.hessian(f)(x), rtol=1e-6)


@pytest.mark.elementwise
@pytest.mark.usefixtures("remat3")
def test_custom_remat_follows_primal():
    """call_hi_primitive with CustomRemat traces the wrapped primal function.

    The closure-captured weights are hoisted to leading operands,
    so the zero weight at position 0 clears that row.
    Like ``custom_vjp_call``, the custom backward rule is not followed.
    """
    w = jnp.array([0.0, 1.0, 2.0])

    def g(x):
        return jnp.sin(x) * w

    def g_fwd(policy, x):
        return g(x), None

    def g_rem(res, x):
        return g(x), x

    def g_bwd(res, ct):
        return (ct * jnp.cos(res) * w,)

    f = jax.custom_remat(g, g_fwd, g_rem, g_bwd)

    x = jnp.array([1.0, 2.0, 3.0])
    result = jacobian_sparsity(f, x).todense().astype(int)
    # custom_remat does not support reverse mode outside of jax.checkpoint,
    # so the reference comes from forward mode, which differentiates the primal.
    expected = (np.abs(jax.jacfwd(f)(x)) > 1e-10).astype(int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
@pytest.mark.usefixtures("remat3")
@pytest.mark.parametrize(
    "policy",
    [
        pytest.param(None, id="no_policy"),
        pytest.param(
            jax.checkpoint_policies.save_only_these_names("a"), id="save_names"
        ),
        pytest.param(
            jax.checkpoint_policies.save_and_offload_only_these_names(
                names_which_can_be_saved=[],
                names_which_can_be_offloaded=["a"],
                offload_src="device",
                offload_dst="pinned_host",
            ),
            id="offload_names",
        ),
    ],
)
def test_remat3_checkpoint_name(policy):
    """checkpoint_name emits a CheckpointName HiJax primitive that passes its operand through.

    The Hessian traces the gradient,
    where name-based policies save or offload the named residual,
    the latter through ``device_put``.
    """

    def g(x):
        return jnp.sin(checkpoint_name(jnp.sin(x) * x[::-1], "a"))

    def f(x):
        return jnp.sum(jax.checkpoint(g, policy=policy)(x))

    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    assert_jacobian_sparsity_exact(g, x)
    assert_hessian_sparsity_exact(f, x)


@pytest.mark.elementwise
def test_unknown_hi_primitive_raises():
    """HiJax primitives without a handler raise instead of guessing a pattern."""

    class Double(HiPrim):
        def __init__(self, aval):
            self.in_avals = (aval,)
            self.out_aval = aval
            self.params = {}
            super().__init__()

        def expand(self, *args):
            (x,) = args
            return 2 * x

    def f(x):
        return Double(jax.typeof(x))(x)

    with pytest.raises(NotImplementedError, match="HiJax primitive 'Double'"):
        jacobian_sparsity(f, np.zeros(3))


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
