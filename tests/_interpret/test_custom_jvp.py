"""Tests for the _prop_custom_jvp_call handler.

JAX differentiates a ``custom_jvp`` function with its rule, never its primal,
so index sets must follow the rule's tangents.
The primal is still propagated for const values,
so indices computed inside it resolve downstream.

https://docs.jax.dev/en/latest/_autosummary/jax.custom_jvp.html
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax
from jax.custom_derivatives import SymbolicZero

from asdex import hessian_sparsity, jacobian, jacobian_sparsity
from tests._utils import (
    assert_jacobian_sparsity_conservative,
    assert_jacobian_sparsity_exact,
)


@jax.custom_jvp
def _straight_through_round(x):
    return jnp.round(x)


_straight_through_round.defjvp(lambda p, t: (_straight_through_round(*p), t[0]))


@jax.custom_jvp
def _approx_rsqrt(x):
    """Approximate ``1/sqrt(x)`` with the Quake III bit hack, without a Newton step.

    Only valid for float32 inputs,
    so tests pin the dtype in case another test enabled x64.
    """
    i = jnp.int32(0x5F3759DF) - (lax.bitcast_convert_type(x, jnp.int32) >> 1)
    return lax.bitcast_convert_type(i, jnp.float32)


@_approx_rsqrt.defjvp
def _approx_rsqrt_jvp(primals, tangents):
    # Calls the primal, so the rule's jaxpr contains this custom_jvp_call again
    y = _approx_rsqrt(primals[0])
    return y, -0.5 * y**3 * tangents[0]


@pytest.mark.elementwise
def test_straight_through_estimator_1d():
    """A zero-derivative primal with an identity rule gives the identity pattern."""
    x = jnp.array([0.3, 1.7, -2.4])
    assert_jacobian_sparsity_exact(_straight_through_round, x)
    result = jacobian_sparsity(_straight_through_round, x).todense().astype(int)
    np.testing.assert_array_equal(result, np.eye(3, dtype=int))


@pytest.mark.elementwise
def test_straight_through_estimator_2d():
    """The identity rule keeps the elementwise pattern on a non-square 2D input."""

    def f(x):
        return _straight_through_round(x.reshape(3, 4)).ravel()

    x = jnp.arange(12.0) / 7
    assert_jacobian_sparsity_exact(f, x)
    result = jacobian_sparsity(f, x).todense().astype(int)
    np.testing.assert_array_equal(result, np.eye(12, dtype=int))


@pytest.mark.elementwise
def test_bitcast_primal_with_nonzero_rule():
    """A primal built from bitcasts has zero derivative, but its rule does not."""
    x = jnp.array([1.0, 4.0, 9.0], dtype=jnp.float32)
    assert_jacobian_sparsity_exact(_approx_rsqrt, x)
    result = jacobian_sparsity(_approx_rsqrt, x).todense().astype(int)
    np.testing.assert_array_equal(result, np.eye(3, dtype=int))


@pytest.mark.jacobian
def test_bitcast_primal_sparse_jacobian_values():
    """The sparse Jacobian matches JAX's, instead of coming out all zero."""
    x = jnp.array([1.0, 4.0, 9.0], dtype=jnp.float32)
    result = jacobian(_approx_rsqrt, x, output_format="dense")(x)
    np.testing.assert_allclose(result, jax.jacobian(_approx_rsqrt)(x))


@pytest.mark.array_ops
def test_rule_sparser_than_primal():
    """A rule that ignores the primal's coupling gives the rule's sparser pattern."""

    @jax.custom_jvp
    def f(x):
        return x * jnp.sum(x)

    f.defjvp(lambda p, t: (f(*p), 2.0 * t[0]))

    x = jnp.array([1.0, 2.0, 3.0])
    assert_jacobian_sparsity_exact(f, x)
    result = jacobian_sparsity(f, x).todense().astype(int)
    np.testing.assert_array_equal(result, np.eye(3, dtype=int))


@pytest.mark.array_ops
def test_rule_reads_other_elements_than_primal():
    """The rule may read different elements than the primal does."""

    @jax.custom_jvp
    def f(x):
        return x[:2]

    f.defjvp(lambda p, t: (f(*p), t[0][1:]))

    x = jnp.array([1.0, 2.0, 3.0])
    assert_jacobian_sparsity_exact(f, x)
    result = jacobian_sparsity(f, x).todense().astype(int)
    expected = np.array(
        [
            [0, 1, 0],  # out[0] has tangent t[1]
            [0, 0, 1],  # out[1] has tangent t[2]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_closure_dependencies_are_dropped():
    """Dependencies through closed-over values are dropped, as in JAX.

    The primal closes over ``y``, which depends on ``x`` in reverse order.
    JAX passes no tangent for closed-over values,
    so only the rule's dependency on its argument remains.
    """

    def f(x):
        y = 2.0 * x[::-1]

        @jax.custom_jvp
        def g(z):
            return z * y

        g.defjvp(lambda p, t: (3.0 * p[0], 3.0 * t[0]))
        return g(x)

    x = jnp.array([1.0, 2.0, 3.0])
    assert_jacobian_sparsity_exact(f, x)
    result = jacobian_sparsity(f, x).todense().astype(int)
    np.testing.assert_array_equal(result, np.eye(3, dtype=int))


@pytest.mark.elementwise
def test_integer_argument():
    """An integer argument gets a float0 tangent without breaking propagation."""

    @jax.custom_jvp
    def f(x, n):
        return x * n

    f.defjvp(lambda p, t: (f(*p), t[0] * p[1]))

    x = jnp.array([1.0, 2.0, 3.0])
    assert_jacobian_sparsity_exact(lambda x: f(x, 2), x)
    result = jacobian_sparsity(lambda x: f(x, 2), x).todense().astype(int)
    np.testing.assert_array_equal(result, np.eye(3, dtype=int))


@pytest.mark.elementwise
def test_jnp_ldexp():
    """``jnp.ldexp`` has a custom_jvp whose rule broadcasts a float0 zero."""

    def f(x):
        return jnp.ldexp(x, 2)

    x = jnp.array([1.5, 3.0, 0.7])
    assert_jacobian_sparsity_exact(f, x)
    result = jacobian_sparsity(f, x).todense().astype(int)
    np.testing.assert_array_equal(result, np.eye(3, dtype=int))


@pytest.mark.elementwise
def test_symbolic_zero_output_tangent():
    """An output whose rule tangent is a symbolic zero has no dependencies."""

    @jax.custom_jvp
    def f(x):
        return x, jnp.round(x)

    def f_jvp(primals, tangents):
        out = f(*primals)
        zero = SymbolicZero(jax.typeof(out[1]).to_tangent_aval())
        return out, (tangents[0], zero)

    f.defjvp(f_jvp, symbolic_zeros=True)

    def g(x):
        return jnp.concatenate(f(x))

    x = jnp.array([1.0, 2.0, 3.0])
    assert_jacobian_sparsity_exact(g, x)
    result = jacobian_sparsity(g, x).todense().astype(int)
    expected = np.zeros((6, 3), dtype=int)
    expected[:3] = np.eye(3, dtype=int)  # out[:3] = x
    # out[3:] has a symbolic zero tangent
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_primal_const_output_resolves_outer_gather():
    """Const values computed inside the primal reach outer consumers.

    The rule gives no information about values,
    so the primal must still be propagated for them.
    """

    @jax.custom_jvp
    def f(x):
        return x * 1.0, jnp.array([2, 0, 1]) + 0

    f.defjvp(lambda p, t: (f(*p), (t[0], np.zeros(3, dtype=jax.dtypes.float0))))

    def g(x):
        y, idx = f(x)
        return y[idx]

    x = jnp.array([1.0, 2.0, 3.0])
    assert_jacobian_sparsity_exact(g, x)
    result = jacobian_sparsity(g, x).todense().astype(int)
    expected = np.array(
        [
            [0, 0, 1],  # out[0] <- x[2]
            [1, 0, 0],  # out[1] <- x[0]
            [0, 1, 0],  # out[2] <- x[1]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_scalar_input():
    """A scalar input gives a 1x1 pattern."""
    x = jnp.array(0.3)
    result = jacobian_sparsity(_straight_through_round, x).todense().astype(int)
    np.testing.assert_array_equal(result, np.ones((1, 1), dtype=int))


@pytest.mark.elementwise
def test_zero_size_input():
    """A zero-size input gives an empty pattern."""
    x = jnp.zeros((0,))
    result = jacobian_sparsity(_straight_through_round, x).todense()
    assert result.shape == (0, 0)


@pytest.mark.elementwise
def test_jax_nn_relu():
    """``jax.nn.relu`` is a custom_jvp with an elementwise rule."""

    def f(x):
        return jax.nn.relu(x.reshape(3, 4)).ravel()

    # Negative entries have a zero derivative at this point,
    # so the numerical Jacobian is only covered, not matched.
    x = jnp.array([-1.0, 2.0, 3.0, -0.5, 0.5, -2.0, 1.0, 4.0, 1.0, 1.0, -3.0, 2.0])
    assert_jacobian_sparsity_conservative(f, x)
    result = jacobian_sparsity(f, x).todense().astype(int)
    np.testing.assert_array_equal(result, np.eye(12, dtype=int))


@pytest.mark.reduction
def test_jax_nn_softmax():
    """``jax.nn.softmax`` couples elements only along the softmax axis."""

    def f(x):
        return jax.nn.softmax(x.reshape(3, 4)).ravel()

    x = jnp.arange(12.0) / 5
    assert_jacobian_sparsity_exact(f, x)
    result = jacobian_sparsity(f, x).todense().astype(int)
    expected = np.kron(np.eye(3, dtype=int), np.ones((4, 4), dtype=int))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.hessian
def test_hessian_through_rule():
    """The Hessian follows the rule, matching JAX's Hessian."""

    def f(x):
        return jnp.sum(_approx_rsqrt(x) * x[::-1])

    x = jnp.array([1.0, 4.0, 9.0], dtype=jnp.float32)
    result = hessian_sparsity(f, x).todense().astype(int)
    expected = (np.abs(jax.hessian(f)(x)) > 1e-10).astype(int)
    np.testing.assert_array_equal(result, expected)
