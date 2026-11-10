"""Tests for the _prop_polynomial handler.

JAX 0.12 added ``jax.lax.polynomial``, which evaluates a polynomial with Horner's rule.
JAX also uses it internally, e.g. in ``jnp.sinc`` and ``jax.scipy.special.ndtri``,
and the derivative of a ``polynomial`` is again a ``polynomial``.

https://docs.jax.dev/en/latest/_autosummary/jax.lax.polynomial.html
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from asdex import jacobian_sparsity
from tests._utils import (
    assert_flat_jacobian_sparsity_exact,
    assert_hessian_sparsity_exact,
    assert_jacobian_sparsity_exact,
)

pytestmark = [
    pytest.mark.elementwise,
    pytest.mark.skipif(
        not hasattr(lax, "polynomial"),
        reason="jax.lax.polynomial was added in JAX 0.12",
    ),
]


def _primitive_names(f, *args) -> list[str]:
    """Return the names of the top-level primitives ``f`` traces to."""
    return [eqn.primitive.name for eqn in jax.make_jaxpr(f)(*args).jaxpr.eqns]


def test_polynomial_literal_coefficients_1d():
    """Literal coefficients contribute nothing, so the Jacobian is diagonal."""

    def f(x):
        return lax.polynomial(x, [1.0, 5.0, 2.0])

    x = jnp.array([0.5, 1.5, 2.5])
    assert "polynomial" in _primitive_names(f, x)
    assert_jacobian_sparsity_exact(f, x)
    result = jacobian_sparsity(f, x).todense().astype(int)
    np.testing.assert_array_equal(result, np.eye(3, dtype=int))


@pytest.mark.parametrize("shape", [(3, 4), (2, 3, 4), (2, 1, 3, 2)])
def test_polynomial_literal_coefficients_nd(shape):
    """The Jacobian stays diagonal for multi-dimensional inputs."""

    def f(x):
        return lax.polynomial(x, [1.0, 5.0, 2.0])

    x = jnp.linspace(0.5, 2.5, int(np.prod(shape))).reshape(shape)
    assert_flat_jacobian_sparsity_exact(f, x)


def test_polynomial_coefficient_array():
    """An array of coefficients is unstacked into scalars that every output reads."""

    def f(xc):
        return lax.polynomial(xc[:3], xc[3:])

    xc = jnp.array([0.5, 1.5, 2.5, 1.0, 2.0])
    assert_jacobian_sparsity_exact(f, xc)
    result = jacobian_sparsity(f, xc).todense().astype(int)
    expected = np.array(
        [
            [1, 0, 0, 1, 1],  # out[0] = c[0] + c[1] * x[0]
            [0, 1, 0, 1, 1],
            [0, 0, 1, 1, 1],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


def test_polynomial_elementwise_coefficients():
    """Coefficients of the shape of x are read elementwise."""

    def f(v):
        x, a0, a2 = v[:2], v[2:4], v[4:]
        return lax.polynomial(x, [a0, 3.0, a2])

    v = jnp.array([0.5, 1.5, 1.0, 2.0, 3.0, 4.0])
    assert_jacobian_sparsity_exact(f, v)
    result = jacobian_sparsity(f, v).todense().astype(int)
    expected = np.array(
        [
            [1, 0, 1, 0, 1, 0],  # out[0] = a0[0] + 3 * x[0] + a2[0] * x[0]^2
            [0, 1, 0, 1, 0, 1],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


def test_polynomial_broadcast_coefficients():
    """Coefficients with size-1 dimensions broadcast against x.

    out[i, j] reads a0[i, 0] and a1[0, j], not the elements at the same flat position.
    """

    def f(v):
        x = v[:6].reshape(2, 3)
        a0 = v[6:8].reshape(2, 1)
        a1 = v[8:].reshape(1, 3)
        return lax.polynomial(x, [a0, a1, 2.0]).ravel()

    v = jnp.linspace(0.5, 3.0, 11)
    assert_jacobian_sparsity_exact(f, v)
    result = jacobian_sparsity(f, v).todense().astype(int)
    expected = np.array(
        [
            # x[i, j]          a0[i]  a1[j]
            [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0],  # out[0, 0]
            [0, 1, 0, 0, 0, 0, 1, 0, 0, 1, 0],  # out[0, 1]
            [0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 1],  # out[0, 2]
            [0, 0, 0, 1, 0, 0, 0, 1, 1, 0, 0],  # out[1, 0]
            [0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0],  # out[1, 1]
            [0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1],  # out[1, 2]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


def test_polynomial_size_one_x():
    """A size-1 x broadcasts against a larger coefficient.

    Verified against forward mode only:
    in JAX 0.12.0, reverse mode fails for a ``polynomial`` whose x is broadcast.
    """

    def f(v):
        return lax.polynomial(v[:1], [v[1:], 2.0])

    v = jnp.array([0.5, 1.0, 2.0, 3.0])
    result = jacobian_sparsity(f, v).todense().astype(int)
    numerical = (np.abs(jax.jacfwd(f)(v)) > 1e-10).astype(int)
    np.testing.assert_array_equal(result, numerical)
    expected = np.array(
        [
            [1, 1, 0, 0],  # out[0] = a0[0] + 2 * x
            [1, 0, 1, 0],
            [1, 0, 0, 1],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


def test_polynomial_scalar_input():
    """A scalar polynomial has a 1x1 Jacobian."""

    def f(x):
        return lax.polynomial(x, [1.0, 5.0, 2.0])

    result = jacobian_sparsity(f, jnp.array(0.5)).todense().astype(int)
    np.testing.assert_array_equal(result, np.ones((1, 1), dtype=int))


def test_polynomial_constant_x():
    """With a constant x, the output only depends on the coefficients."""
    points = jnp.array([0.5, 1.5, 2.5])

    def f(c):
        return lax.polynomial(points, [c[0], c[1]])

    c = jnp.array([1.0, 2.0])
    assert_jacobian_sparsity_exact(f, c)
    result = jacobian_sparsity(f, c).todense().astype(int)
    np.testing.assert_array_equal(result, np.ones((3, 2), dtype=int))


def test_polynomial_many_coefficients():
    """More coefficients than the unroll threshold still emit a single primitive."""

    def f(x):
        return lax.polynomial(x, [1.0] * 12)

    x = jnp.array([0.5, 1.5, 2.5])
    assert "polynomial" in _primitive_names(f, x)
    assert_jacobian_sparsity_exact(f, x)


def test_polynomial_size_zero():
    """A size-0 polynomial yields an empty pattern."""

    def f(x):
        return lax.polynomial(x[:0], [x[0], 2.0])

    result = jacobian_sparsity(f, jnp.ones(3))
    assert result.shape == (0, 3)
    assert result.nnz == 0


@pytest.mark.hessian
def test_polynomial_hessian():
    """The derivative of a polynomial is a polynomial, so the Hessian is diagonal."""

    def f(x):
        return jnp.sum(lax.polynomial(x, [1.0, 5.0, 2.0, 3.0]))

    assert_hessian_sparsity_exact(f, jnp.array([0.5, 1.5, 2.5]))


@pytest.mark.hessian
def test_polynomial_hessian_coefficients():
    """The Hessian couples x with the coefficients it multiplies."""

    def f(v):
        return jnp.sum(lax.polynomial(v[:2], [v[2], v[3], v[4]]))

    assert_hessian_sparsity_exact(f, jnp.array([0.5, 1.5, 1.0, 2.0, 3.0]))


@pytest.mark.parametrize(
    "fn",
    [
        pytest.param(jnp.sinc, id="sinc"),
        pytest.param(jax.scipy.special.ndtri, id="ndtri"),
        pytest.param(jax.scipy.special.erfcx, id="erfcx"),
        pytest.param(jax.scipy.special.expi, id="expi"),
        pytest.param(jax.scipy.stats.norm.ppf, id="norm_ppf"),
    ],
)
def test_polynomial_internal_users(fn):
    """Functions that JAX implements with ``polynomial`` keep their diagonal Jacobian."""
    x = jnp.array([0.2, 0.4, 0.7])
    result = jacobian_sparsity(fn, x).todense().astype(int)
    np.testing.assert_array_equal(result, np.eye(3, dtype=int))


@pytest.mark.hessian
def test_sinc_hessian():
    """The Hessian of sinc differentiates through ``polynomial`` and stays diagonal."""

    def f(x):
        return jnp.sum(jnp.sinc(x))

    assert_hessian_sparsity_exact(f, jnp.array([0.2, 0.4, 0.7]))
