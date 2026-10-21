"""Tests for incomplete gamma and Hurwitz zeta handlers.

The generic binary-elementwise behavior (both argnums, 1D/2D/3D shapes,
broadcasting, scalar mixing, and zero-sized operands)
is covered by the shared parametrization in ``test_elementwise.py``.
This file keeps the domain-specific fixed-operand cases
and the ``igamma_grad_a`` reachability check.

Covers the primitives ``igamma`` (``gammainc``),
``igammac`` (``gammaincc``), ``igamma_grad_a``, and ``zeta``.

Zeta inputs with ``q <= 0`` are poles where ``jax.jacobian`` is NaN,
so numerical references only use ``q > 0``.
"""

import jax
import jax.numpy as jnp
import jax.scipy.special
import numpy as np
import pytest

from asdex import jacobian_sparsity
from tests._utils import (
    assert_hessian_sparsity_exact,
    assert_jacobian_sparsity_exact,
)

_INCOMPLETE_GAMMA_OPS = [
    jax.scipy.special.gammainc,
    jax.scipy.special.gammaincc,
]


@pytest.mark.elementwise
@pytest.mark.parametrize("op", _INCOMPLETE_GAMMA_OPS)
def test_incomplete_gamma_x_only(op):
    """With a fixed, only x is traced, so the Jacobian is diagonal in x."""
    x = jnp.array([0.3, 1.2, 2.7])

    def f(x):
        return op(2.0, x)

    assert_jacobian_sparsity_exact(f, x)


@pytest.mark.elementwise
@pytest.mark.parametrize("op", _INCOMPLETE_GAMMA_OPS)
def test_incomplete_gamma_a_only(op):
    """With x fixed, only a is traced, so the Jacobian is diagonal in a."""
    a = jnp.array([0.5, 1.2, 2.7])

    def f(a):
        return op(a, 3.0)

    assert_jacobian_sparsity_exact(f, a)


@pytest.mark.elementwise
def test_zeta_x_only():
    """With q fixed, only x is traced, so the Jacobian is diagonal in x."""
    x = jnp.array([1.4, 2.3, 3.1])

    def f(x):
        return jax.scipy.special.zeta(x, 2.0)

    assert_jacobian_sparsity_exact(f, x)


@pytest.mark.elementwise
def test_zeta_q_only():
    """With x fixed, only q is traced, so the Jacobian is diagonal in q."""
    q = jnp.array([0.4, 1.1, 2.2])

    def f(q):
        return jax.scipy.special.zeta(2.5, q)

    assert_jacobian_sparsity_exact(f, q)


@pytest.mark.elementwise
def test_zeta_s_below_one():
    """Zeta has finite custom-JVP derivatives below the s > 1 convergence region.

    The primal is NaN for s <= 1 on some CPUs,
    but the custom JVP series still gives exact numerical references.
    """
    s = jnp.array([0.2, 0.5, 0.8])

    def f(s):
        return jax.scipy.special.zeta(s, 1.3)

    assert_jacobian_sparsity_exact(f, s)


@pytest.mark.hessian
def test_incomplete_gamma_hessian_x():
    """Second-order derivatives of gammainc wrt x are detected exactly."""
    x = jnp.array([0.5, 1.1, 2.3])

    def f(x):
        return jnp.sum(jax.scipy.special.gammainc(2.0, x))

    assert_hessian_sparsity_exact(f, x)


@pytest.mark.hessian
def test_zeta_hessian_both_args():
    """Second-order derivatives of zeta wrt both arguments are detected exactly."""
    n = 3
    v = jnp.array([1.3, 1.8, 2.4, 0.5, 1.1, 2.0])

    def f(v):
        return jnp.sum(jax.scipy.special.zeta(v[:n], v[n:]))

    assert_hessian_sparsity_exact(f, v)


@pytest.mark.elementwise
@pytest.mark.parametrize("op", _INCOMPLETE_GAMMA_OPS)
def test_igamma_grad_a_through_grad(op):
    """Differentiating gammainc/gammaincc wrt a emits ``igamma_grad_a``.

    Without a handler, detection raises NotImplementedError.
    """
    a = jnp.array([0.5, 1.2, 2.7])

    def f(a):
        return jax.grad(lambda z: jnp.sum(op(z, 3.0)))(a)

    primitives = {eqn.primitive.name for eqn in jax.make_jaxpr(f)(a).jaxpr.eqns}
    # Deliberate reachability pin: igamma_grad_a is internal to JAX,
    # so assert the lowering still emits it.
    assert "igamma_grad_a" in primitives

    result = jacobian_sparsity(f, a).todense().astype(int)
    # Manual expected pattern: JAX has no derivative rule for igamma_grad_a,
    # so jax.jacobian cannot provide a numerical reference here.
    np.testing.assert_array_equal(result, np.eye(3, dtype=int))


@pytest.mark.elementwise
@pytest.mark.parametrize("op", _INCOMPLETE_GAMMA_OPS)
def test_igamma_grad_a_depends_on_x_too(op):
    """``igamma_grad_a(a, x)`` depends on both operands.

    With x traced, a handler that drops the second operand would miss nonzeros.
    """
    n = 2
    v = jnp.array([0.7, 1.8, 2.3, 0.6])

    def f(v):
        a, x = v[:n], v[n:]
        return jax.grad(lambda z: jnp.sum(op(z, x)))(a)

    result = jacobian_sparsity(f, v).todense().astype(int)
    # Manual expected pattern: no JAX derivative rule for igamma_grad_a.
    expected = np.array(
        [
            [1, 0, 1, 0],  # out[0] <- a[0], x[0]
            [0, 1, 0, 1],  # out[1] <- a[1], x[1]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)
