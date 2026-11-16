"""Tests for the `composite` primitive.

`lax.composite` embeds the jaxpr of its decomposition in the equation,
so a composite has exactly the pattern of the decomposition it stands for.

https://docs.jax.dev/en/latest/_autosummary/jax.lax.composite.html
"""

from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from asdex import jacobian_sparsity
from tests._utils import (
    assert_flat_jacobian_sparsity_exact,
    assert_jacobian_sparsity_exact,
    numerical_flat_jacobian_sparsity,
)


def _assert_matches_plain(f, plain, x):
    """Assert the detected pattern of ``f`` matches the numerical pattern of ``plain``.

    ``jax.jacobian`` cannot differentiate a ``composite`` in JAX 0.11
    (both its JVP and its transpose rule raise ``ValueError``),
    so the shared exactness helpers cannot be applied to a function containing one.
    ``plain`` is the same function with the composite inlined,
    which is what the composite is defined to compute.
    """
    detected = jacobian_sparsity(f, x).todense().astype(int)
    np.testing.assert_array_equal(detected, numerical_flat_jacobian_sparsity(plain, x))


# Decompositions


@pytest.mark.array_ops
@pytest.mark.parametrize(
    "op", [jnp.negative, partial(jnp.multiply, 2.0)], ids=["negate", "scale"]
)
def test_elementwise(op):
    """An elementwise decomposition keeps the diagonal pattern of its operand."""
    _assert_matches_plain(lax.composite(op, "elementwise"), op, np.ones(4))


@pytest.mark.array_ops
def test_elementwise_mix():
    """Mixing several elementwise ops inside one decomposition stays diagonal."""

    def decomposition(x):
        return jnp.tanh(x) * jnp.exp(x) + jnp.sin(x) / (1.0 + jnp.square(x))

    _assert_matches_plain(
        lax.composite(decomposition, "mix"), decomposition, np.ones(5)
    )


@pytest.mark.array_ops
def test_2d_input():
    """A composite over a 2-D input propagates the pattern of its decomposition."""

    def decomposition(x):
        return jnp.transpose(jnp.cumsum(x, axis=0) + jnp.flip(x, axis=1))

    x = np.arange(6.0).reshape(3, 2) + 1.0
    _assert_matches_plain(lax.composite(decomposition, "2d"), decomposition, x)


@pytest.mark.array_ops
def test_3d_input():
    """A composite over a 3-D input unions the reduced-axis dependencies."""

    def decomposition(x):
        return jnp.sum(x, axis=1)

    x = np.arange(12.0).reshape(2, 3, 2) + 1.0
    _assert_matches_plain(lax.composite(decomposition, "3d"), decomposition, x)


@pytest.mark.array_ops
def test_non_elementwise_decomposition():
    """A decomposition mixing a shift, a reduction and a concatenate keeps its pattern."""

    def decomposition(x):
        return jnp.concatenate([x[1:], jnp.sum(x)[None]])

    f = lax.composite(decomposition, "shifted")
    _assert_matches_plain(f, decomposition, np.ones(4))


@pytest.mark.array_ops
def test_closure_constant_operand():
    """A closure-captured constant is an operand of the composite but not a dependency.

    JAX hoists it into the decomposition jaxpr's leading invar
    and passes it as the equation's leading operand.
    """
    weights = jnp.array([1.0, 0.0, 2.0])

    def decomposition(x):
        return x * weights

    f = lax.composite(decomposition, "weighted")
    _assert_matches_plain(f, decomposition, np.ones(3))


@pytest.mark.array_ops
def test_closure_constant_index():
    """A closure-captured index array stays precise through the composite.

    The constant is a leading operand of the composite, so the handler must
    forward its value into the decomposition jaxpr for the gather to resolve.
    """
    indices = jnp.array([2, 0, 1])

    def decomposition(x):
        return x[indices]

    f = lax.composite(decomposition, "gather")
    _assert_matches_plain(f, decomposition, np.arange(3.0))


@pytest.mark.array_ops
def test_multiple_operands_and_outputs():
    """A composite with two operands and two outputs aligns both with the equation."""

    def decomposition(x):
        a, b = x[:2], x[2:]
        return jnp.concatenate([a * b, jnp.sin(a)])

    def f(x):
        return lax.composite(lambda a, b: (a * b, jnp.sin(a)), "multi")(x[:2], x[2:])

    _assert_matches_plain(f, decomposition, np.ones(4))


@pytest.mark.array_ops
def test_softmax_decomposition_with_attributes():
    """Keyword arguments become composite attributes, not operands.

    This is the decorator form recommended by the ``lax.composite`` docs.
    ``jax.nn.softmax`` itself does not lower to a composite in JAX 0.11.
    """
    my_softmax = partial(lax.composite, name="my.softmax")(jax.nn.softmax)

    def decomposition(x):
        return jax.nn.softmax(x, axis=1)

    def f(x):
        return my_softmax(x, axis=1)

    x = np.arange(6.0).reshape(2, 3) + 1.0
    _assert_matches_plain(f, decomposition, x)


@pytest.mark.array_ops
def test_zero_size_input():
    """A composite over a zero-element input has an empty pattern."""
    result = jacobian_sparsity(lax.composite(jnp.sin, "empty"), np.zeros(0)).todense()
    assert result.shape == (0, 0)


# Nesting


@pytest.mark.array_ops
def test_nested_composite():
    """A composite whose decomposition contains a composite traces both."""
    inner = lax.composite(jnp.sin, "inner")

    def decomposition(x):
        return jnp.sin(x) * 2.0

    def f(x):
        return lax.composite(lambda y: inner(y) * 2.0, "outer")(x)

    _assert_matches_plain(f, decomposition, np.ones(3))


@pytest.mark.array_ops
@pytest.mark.parametrize("wrap", [lambda op: op, jax.jit], ids=["plain", "jit"])
def test_composite_in_bigger_function(wrap):
    """A composite feeding later ops contributes the dependencies those ops read."""

    def assembled(op):
        def f(x):
            y = op(x)
            return y[1:] + jnp.sum(y)

        return f

    x = np.arange(3.0) + 1.0
    plain = assembled(jnp.cumsum)
    assert_flat_jacobian_sparsity_exact(plain, x)
    _assert_matches_plain(
        assembled(wrap(lax.composite(jnp.cumsum, "in_bigger"))), plain, x
    )


@pytest.mark.array_ops
def test_composite_inside_scan():
    """A composite in a scan body contributes the body's pattern."""

    def scanned(op):
        def f(x):
            def body(carry, _):
                return op(carry), None

            return lax.scan(body, x, None, length=3)[0]

        return f

    x = np.arange(3.0) + 1.0
    _assert_matches_plain(
        scanned(lax.composite(jnp.sin, "inscan")), scanned(jnp.sin), x
    )


@pytest.mark.array_ops
def test_composite_inside_cond():
    """A composite in one cond branch contributes that branch's pattern."""

    def branched(op):
        def f(x):
            return lax.cond(x[0] > 0.0, op, jnp.ones_like, x)

        return f

    x = np.arange(3.0) + 1.0
    _assert_matches_plain(
        branched(lax.composite(jnp.sin, "incond")), branched(jnp.sin), x
    )


@pytest.mark.array_ops
def test_composite_inside_grad():
    """A composite reaches detection through the primal of a custom_jvp.

    ``jax.grad`` cannot differentiate a composite directly in JAX 0.11,
    so the composite is wrapped in ``jax.custom_jvp``,
    whose traced primal keeps the composite equation.
    """
    composite = lax.composite(lambda y: jnp.sin(y) * 2.0, "in_grad")

    @jax.custom_jvp
    def f(x):
        return composite(x)

    @f.defjvp
    def f_jvp(primals, tangents):
        (x,), (t,) = primals, tangents
        y = f(x)
        return y, 2.0 * y * t

    def loss(x):
        return jnp.sum(f(x))

    assert_jacobian_sparsity_exact(jax.grad(loss), np.ones(3))
