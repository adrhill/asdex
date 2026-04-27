"""Tests for ``has_aux``, ``holomorphic``, and ``allow_int`` kwargs.

Mirrors ``jax.jacrev`` / ``jax.jacfwd`` / ``jax.grad`` / ``jax.hessian``
semantics on the asdex public API.
"""

import warnings

import jax.numpy as jnp
import numpy as np
import pytest

import asdex

warnings.filterwarnings("ignore", category=asdex.DenseColoringWarning)


# has_aux — Jacobian


@pytest.mark.jacobian
def test_jacobian_has_aux_returns_tuple():
    """``jacobian(f, has_aux=True)(x)`` returns ``(jac, aux)``."""

    def f(x):
        y = jnp.array([x[0] * x[1], x[0] + x[1]])
        aux = {"norm": jnp.sum(x**2)}
        return y, aux

    x = jnp.array([2.0, 3.0])
    jac, aux = asdex.jacobian(f, np.zeros(2), has_aux=True, output_format="dense")(x)

    expected = np.array([[x[1], x[0]], [1.0, 1.0]])
    np.testing.assert_allclose(jac, expected)
    np.testing.assert_allclose(aux["norm"], 13.0)


@pytest.mark.jacobian
def test_value_and_jacobian_has_aux_returns_nested_tuple():
    """``value_and_jacobian(f, has_aux=True)(x)`` returns ``((value, aux), jac)``."""

    def f(x):
        y = jnp.array([x[0] ** 2, x[1] ** 2])
        aux = "metadata"
        return y, aux

    x = jnp.array([2.0, 3.0])
    (value, aux), jac = asdex.value_and_jacobian(
        f, np.zeros(2), has_aux=True, output_format="dense"
    )(x)

    np.testing.assert_allclose(value, jnp.array([4.0, 9.0]))
    assert aux == "metadata"
    expected = np.array([[2 * x[0], 0.0], [0.0, 2 * x[1]]])
    np.testing.assert_allclose(jac, expected)


@pytest.mark.jacobian
def test_jacobian_from_coloring_has_aux():
    """``jacobian_from_coloring`` also supports ``has_aux``."""

    def f_no_aux(x):
        return jnp.array([x[0] * x[1], x[0] + x[1]])

    def f(x):
        return f_no_aux(x), jnp.sum(x)

    x = jnp.array([2.0, 3.0])
    coloring = asdex.jacobian_coloring(f, np.zeros(2), has_aux=True)
    jac, aux = asdex.jacobian_from_coloring(
        f, coloring, output_format="dense", has_aux=True
    )(x)

    expected = np.array([[x[1], x[0]], [1.0, 1.0]])
    np.testing.assert_allclose(jac, expected)
    np.testing.assert_allclose(aux, 5.0)


@pytest.mark.jacobian
def test_jacobian_has_aux_multi_input():
    """``has_aux`` works with multi-input functions."""

    def f(x, y):
        out = jnp.array([x[0] * y[0], x[1] + y[1]])
        aux = jnp.sum(x) + jnp.sum(y)
        return out, aux

    x, y = jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])
    (Jx, Jy), aux = asdex.jacobian(
        f, np.zeros(2), np.zeros(2), argnums=(0, 1), has_aux=True, output_format="dense"
    )(x, y)

    np.testing.assert_allclose(Jx, np.array([[y[0], 0.0], [0.0, 1.0]]))
    np.testing.assert_allclose(Jy, np.array([[x[0], 0.0], [0.0, 1.0]]))
    np.testing.assert_allclose(aux, 10.0)


# has_aux — Hessian


@pytest.mark.hessian
def test_hessian_has_aux_returns_tuple():
    """``hessian(f, has_aux=True)(x)`` returns ``(hess, aux)``."""

    def f(x):
        y = x[0] ** 2 + x[0] * x[1] + x[1] ** 2
        aux = {"input_sum": jnp.sum(x)}
        return y, aux

    x = jnp.array([2.0, 3.0])
    hess, aux = asdex.hessian(f, np.zeros(2), has_aux=True, output_format="dense")(x)

    expected = np.array([[2.0, 1.0], [1.0, 2.0]])
    np.testing.assert_allclose(hess, expected)
    np.testing.assert_allclose(aux["input_sum"], 5.0)


@pytest.mark.hessian
def test_value_and_hessian_has_aux_returns_nested_tuple():
    """``value_and_hessian(f, has_aux=True)(x)`` returns ``((value, aux), hess)``."""

    def f(x):
        y = x[0] ** 2 + x[1] ** 2
        return y, "tag"

    x = jnp.array([2.0, 3.0])
    (value, aux), hess = asdex.value_and_hessian(
        f, np.zeros(2), has_aux=True, output_format="dense"
    )(x)

    np.testing.assert_allclose(value, 13.0)
    assert aux == "tag"
    np.testing.assert_allclose(hess, np.array([[2.0, 0.0], [0.0, 2.0]]))


@pytest.mark.hessian
def test_hessian_has_aux_multi_input():
    """``has_aux`` works for multi-input Hessians."""

    def f(x, y):
        out = x[0] * y[0] + x[1] ** 2 + y[1] ** 2
        aux = x[0] + y[0]
        return out, aux

    x, y = jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])
    hess, aux = asdex.hessian(
        f, np.zeros(2), np.zeros(2), argnums=(0, 1), has_aux=True, output_format="dense"
    )(x, y)

    # Block-structured Hessian: H[i][j] = d^2 f / d arg_i d arg_j.
    np.testing.assert_allclose(hess[0][0], np.array([[0.0, 0.0], [0.0, 2.0]]))
    np.testing.assert_allclose(hess[1][1], np.array([[0.0, 0.0], [0.0, 2.0]]))
    np.testing.assert_allclose(hess[0][1], np.array([[1.0, 0.0], [0.0, 0.0]]))
    np.testing.assert_allclose(hess[1][0], np.array([[1.0, 0.0], [0.0, 0.0]]))
    np.testing.assert_allclose(aux, 4.0)


# holomorphic


@pytest.mark.jacobian
def test_holomorphic_allows_complex_input():
    """``holomorphic=True`` permits complex-valued inputs and outputs."""

    def f(z):
        return jnp.array([z[0] ** 2, z[0] * z[1]])

    z = jnp.array([1.0 + 2.0j, 3.0 + 0.5j])
    jac = asdex.jacobian(f, np.zeros(2), holomorphic=True, output_format="dense")(z)
    expected = np.array([[2 * z[0], 0.0], [z[1], z[0]]])
    np.testing.assert_allclose(jac, expected)


@pytest.mark.jacobian
def test_holomorphic_false_rejects_complex_input():
    """Without ``holomorphic=True``, complex inputs raise ``TypeError``."""

    def f(z):
        return jnp.array([z[0] ** 2])

    z = jnp.array([1.0 + 2.0j, 3.0 + 0.5j])
    with pytest.raises(TypeError):
        asdex.jacobian(f, np.zeros(2), output_format="dense")(z)


# allow_int


@pytest.mark.jacobian
def test_allow_int_permits_int_input():
    """``allow_int=True`` bypasses the reverse-mode integer-input dtype check."""

    def f(x):
        return jnp.array([x[0] + x[1], x[0] * 2], dtype=jnp.float32)

    x = jnp.array([1, 2], dtype=jnp.int32)
    # Only testing that the validator does not reject; downstream jax.vjp will
    # yield a ``float0`` cotangent for integer inputs, which is expected.
    jac = asdex.jacobian(f, np.zeros(2), mode="rev", allow_int=True)(x)
    assert jac.shape == (2, 2)


@pytest.mark.jacobian
def test_allow_int_false_rejects_int_input():
    """Without ``allow_int=True``, integer inputs raise ``TypeError`` in reverse mode."""

    def f(x):
        return jnp.array([x[0] + x[1]])

    x = jnp.array([1, 2], dtype=jnp.int32)
    with pytest.raises(TypeError):
        asdex.jacobian(f, np.zeros(2), mode="rev", output_format="dense")(x)
