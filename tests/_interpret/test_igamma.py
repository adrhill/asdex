"""Tests for incomplete gamma and Hurwitz zeta handlers.

Covers the binary elementwise primitives ``igamma`` (``gammainc``),
``igammac`` (``gammaincc``), ``igamma_grad_a``, and ``zeta``.
Each output element depends only on the corresponding elements
of both operands.
"""

import jax
import jax.numpy as jnp
import jax.scipy.special
import numpy as np
import pytest

from asdex import jacobian_sparsity
from tests._utils import assert_jacobian_sparsity_exact

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

    result = jacobian_sparsity(f, x).todense().astype(int)
    np.testing.assert_array_equal(result, np.eye(3, dtype=int))
    assert_jacobian_sparsity_exact(f, x)


@pytest.mark.elementwise
@pytest.mark.parametrize("op", _INCOMPLETE_GAMMA_OPS)
def test_incomplete_gamma_a_only(op):
    """With x fixed, only a is traced, so the Jacobian is diagonal in a."""
    a = jnp.array([0.5, 1.2, 2.7])

    def f(a):
        return op(a, 3.0)

    result = jacobian_sparsity(f, a).todense().astype(int)
    np.testing.assert_array_equal(result, np.eye(3, dtype=int))
    assert_jacobian_sparsity_exact(f, a)


@pytest.mark.elementwise
@pytest.mark.parametrize("op", _INCOMPLETE_GAMMA_OPS)
def test_incomplete_gamma_both_inputs(op):
    """A scalar first operand and vector second operand share the input vector."""
    v = jnp.array([1.4, 0.3, 1.1, 2.5])

    def f(v):
        return op(v[0], v[1:])

    result = jacobian_sparsity(f, v).todense().astype(int)
    expected = np.array(
        [
            [1, 1, 0, 0],  # out[0] <- a = v[0], x = v[1]
            [1, 0, 1, 0],  # out[1] <- a = v[0], x = v[2]
            [1, 0, 0, 1],  # out[2] <- a = v[0], x = v[3]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)
    assert_jacobian_sparsity_exact(f, v)


@pytest.mark.elementwise
@pytest.mark.parametrize("op", _INCOMPLETE_GAMMA_OPS)
@pytest.mark.parametrize("shape", [(4,), (3, 4), (2, 3, 4)])
def test_incomplete_gamma_shapes(op, shape):
    """1D, 2D, and 3D non-square inputs keep the elementwise diagonal pattern."""
    n = int(np.prod(shape))
    x = jnp.linspace(0.2, 3.0, n)

    def f(x):
        return op(2.0, x.reshape(shape)).ravel()

    result = jacobian_sparsity(f, x).todense().astype(int)
    np.testing.assert_array_equal(result, np.eye(n, dtype=int))
    assert_jacobian_sparsity_exact(f, x)


@pytest.mark.elementwise
@pytest.mark.parametrize("op", _INCOMPLETE_GAMMA_OPS)
@pytest.mark.parametrize("shape", [(3, 4), (2, 3, 4)])
def test_incomplete_gamma_multidim_input(op, shape):
    """Multi-dimensional inputs are flattened elementwise by the public API."""
    n = int(np.prod(shape))
    x = jnp.linspace(0.2, 3.0, n).reshape(shape)

    def f(x):
        return op(2.0, x)

    result = jacobian_sparsity(f, x).todense().astype(int)
    np.testing.assert_array_equal(result, np.eye(n, dtype=int))


@pytest.mark.elementwise
@pytest.mark.parametrize("op", _INCOMPLETE_GAMMA_OPS)
def test_incomplete_gamma_scalar_a_2d_x(op):
    """A scalar first operand broadcasts across a 2D second operand."""
    v = jnp.r_[1.7, jnp.linspace(0.2, 2.5, 12)]

    def f(v):
        return op(v[0], v[1:].reshape(3, 4)).ravel()

    result = jacobian_sparsity(f, v).todense().astype(int)
    expected = np.zeros((12, 13), dtype=int)
    expected[:, 0] = 1  # every output depends on the scalar a
    expected[:, 1:] = np.eye(12, dtype=int)  # out[i] <- x[i]
    np.testing.assert_array_equal(result, expected)
    assert_jacobian_sparsity_exact(f, v)


@pytest.mark.elementwise
@pytest.mark.parametrize("op", _INCOMPLETE_GAMMA_OPS)
def test_incomplete_gamma_scalar_x(op):
    """An array first operand broadcasts against a scalar second operand."""
    v = jnp.array([0.5, 1.2, 2.7, 3.0])

    def f(v):
        return op(v[:-1], v[-1])

    result = jacobian_sparsity(f, v).todense().astype(int)
    expected = np.zeros((3, 4), dtype=int)
    expected[:, :3] = np.eye(3, dtype=int)  # out[i] <- a[i]
    expected[:, 3] = 1  # every output depends on the scalar x
    np.testing.assert_array_equal(result, expected)
    assert_jacobian_sparsity_exact(f, v)


@pytest.mark.elementwise
@pytest.mark.parametrize("op", _INCOMPLETE_GAMMA_OPS)
def test_incomplete_gamma_broadcasting(op):
    """A (3,1) operand broadcasts across a (3,4) operand per row."""
    v = jnp.r_[jnp.array([1.5, 2.0, 0.7]), jnp.linspace(0.2, 2.5, 12)]

    def f(v):
        a = v[:3].reshape(3, 1)
        x = v[3:].reshape(3, 4)
        return op(a, x).ravel()

    result = jacobian_sparsity(f, v).todense().astype(int)
    expected = np.zeros((12, 15), dtype=int)
    for i in range(3):
        rows = slice(i * 4, (i + 1) * 4)
        expected[rows, i] = 1  # row i reads a[i, 0]
        expected[rows, 3 + i * 4 : 3 + (i + 1) * 4] = np.eye(4, dtype=int)
    np.testing.assert_array_equal(result, expected)
    assert_jacobian_sparsity_exact(f, v)


@pytest.mark.elementwise
@pytest.mark.parametrize("op", _INCOMPLETE_GAMMA_OPS)
def test_incomplete_gamma_zero_size(op):
    """Zero-sized operands produce an empty pattern instead of crashing."""

    def f(x):
        return op(2.0, x)

    result = jacobian_sparsity(f, np.zeros((0,)))
    assert result.shape == (0, 0)
    assert result.nnz == 0

    def g(v):
        return op(v[:0], v[0])

    result = jacobian_sparsity(g, np.zeros(1))
    assert result.shape == (0, 1)
    assert result.nnz == 0


@pytest.mark.elementwise
def test_zeta_x_only():
    """With q fixed, only x is traced, so the Jacobian is diagonal in x."""
    x = jnp.array([1.4, 2.3, 3.1])

    def f(x):
        return jax.scipy.special.zeta(x, 2.0)

    result = jacobian_sparsity(f, x).todense().astype(int)
    np.testing.assert_array_equal(result, np.eye(3, dtype=int))
    assert_jacobian_sparsity_exact(f, x)


@pytest.mark.elementwise
def test_zeta_q_only():
    """With x fixed, only q is traced, so the Jacobian is diagonal in q."""
    q = jnp.array([0.4, 1.1, 2.2])

    def f(q):
        return jax.scipy.special.zeta(2.5, q)

    result = jacobian_sparsity(f, q).todense().astype(int)
    np.testing.assert_array_equal(result, np.eye(3, dtype=int))
    assert_jacobian_sparsity_exact(f, q)


@pytest.mark.elementwise
def test_zeta_both_inputs():
    """A scalar first operand and vector second operand share the input vector."""
    v = jnp.array([2.5, 0.4, 1.1, 2.2])

    def f(v):
        return jax.scipy.special.zeta(v[0], v[1:])

    result = jacobian_sparsity(f, v).todense().astype(int)
    expected = np.array(
        [
            [1, 1, 0, 0],  # out[0] <- x = v[0], q = v[1]
            [1, 0, 1, 0],  # out[1] <- x = v[0], q = v[2]
            [1, 0, 0, 1],  # out[2] <- x = v[0], q = v[3]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)
    assert_jacobian_sparsity_exact(f, v)


@pytest.mark.elementwise
@pytest.mark.parametrize("shape", [(4,), (3, 4), (2, 3, 2)])
def test_zeta_shapes(shape):
    """1D, 2D, and 3D non-square inputs keep the elementwise diagonal pattern."""
    n = int(np.prod(shape))
    inputs = jnp.r_[
        jnp.linspace(1.3, 3.2, n),
        jnp.linspace(0.4, 2.0, n),
    ]

    def f(v):
        x = v[:n].reshape(shape)
        q = v[n:].reshape(shape)
        return jax.scipy.special.zeta(x, q).ravel()

    result = jacobian_sparsity(f, inputs).todense().astype(int)
    expected = np.zeros((n, 2 * n), dtype=int)
    expected[np.arange(n), np.arange(n)] = 1  # out[i] <- x[i]
    expected[np.arange(n), n + np.arange(n)] = 1  # out[i] <- q[i]
    np.testing.assert_array_equal(result, expected)
    assert_jacobian_sparsity_exact(f, inputs)


@pytest.mark.elementwise
def test_zeta_broadcasting():
    """A (3,1) operand broadcasts across a (3,4) operand per row."""
    v = jnp.r_[jnp.array([2.0, 2.5, 3.0]), jnp.linspace(0.4, 2.0, 12)]

    def f(v):
        x = v[:3].reshape(3, 1)
        q = v[3:].reshape(3, 4)
        return jax.scipy.special.zeta(x, q).ravel()

    result = jacobian_sparsity(f, v).todense().astype(int)
    expected = np.zeros((12, 15), dtype=int)
    for i in range(3):
        rows = slice(i * 4, (i + 1) * 4)
        expected[rows, i] = 1  # row i reads x[i, 0]
        expected[rows, 3 + i * 4 : 3 + (i + 1) * 4] = np.eye(4, dtype=int)
    np.testing.assert_array_equal(result, expected)
    assert_jacobian_sparsity_exact(f, v)


@pytest.mark.elementwise
def test_zeta_zero_size():
    """Zero-sized operands produce an empty pattern instead of crashing."""

    def f(x):
        return jax.scipy.special.zeta(x, 2.0)

    result = jacobian_sparsity(f, np.zeros((0,)))
    assert result.shape == (0, 0)
    assert result.nnz == 0

    def g(v):
        return jax.scipy.special.zeta(v[:0], v[0])

    result = jacobian_sparsity(g, np.zeros(1))
    assert result.shape == (0, 1)
    assert result.nnz == 0


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
    assert "igamma_grad_a" in primitives

    result = jacobian_sparsity(f, a).todense().astype(int)
    np.testing.assert_array_equal(result, np.eye(3, dtype=int))
