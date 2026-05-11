"""Tests for ``has_aux``, ``holomorphic``, and ``allow_int`` kwargs.

Mirrors ``jax.jacrev`` / ``jax.jacfwd`` / ``jax.grad`` / ``jax.hessian``
semantics on the asdex public API.
"""

import warnings
from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import ShapeDtypeStruct

import asdex
from asdex._api_utils import (
    _check_input_dtype_fwd,
    _check_input_dtype_rev,
    _check_output_dtype_rev,
    avals_from_args,
)

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
def test_allow_int_dense_output():
    """``allow_int=True`` with dense output handles float0 cotangents."""

    def f(x):
        return jnp.array([x[0] + x[1], x[0] * 2], dtype=jnp.float32)

    x = jnp.array([1, 2], dtype=jnp.int32)
    jac = asdex.jacobian(
        f, np.zeros(2), mode="rev", allow_int=True, output_format="dense"
    )(x)
    assert jac.shape == (2, 2)
    np.testing.assert_array_equal(jac, np.zeros((2, 2)))


@pytest.mark.jacobian
def test_allow_int_false_rejects_int_input():
    """Without ``allow_int=True``, integer inputs raise ``TypeError`` in reverse mode."""

    def f(x):
        return jnp.array([x[0] + x[1]])

    x = jnp.array([1, 2], dtype=jnp.int32)
    with pytest.raises(TypeError):
        asdex.jacobian(f, np.zeros(2), mode="rev", output_format="dense")(x)


# holomorphic dtype validation


@pytest.mark.jacobian
def test_holomorphic_rev_rejects_real_input():
    """``holomorphic=True`` with real input in reverse mode raises TypeError."""

    def f(x):
        return jnp.array([x[0] ** 2])

    x = jnp.array([1.0, 2.0])
    with pytest.raises(TypeError, match=r"holomorphic.*complex"):
        asdex.jacobian(
            f, np.zeros(2), holomorphic=True, mode="rev", output_format="dense"
        )(x)


@pytest.mark.jacobian
def test_holomorphic_fwd_rejects_real_input():
    """``holomorphic=True`` with real input in forward mode raises TypeError."""

    def f(x):
        return jnp.array([x[0] ** 2])

    x = jnp.array([1.0, 2.0])
    with pytest.raises(TypeError, match=r"holomorphic.*complex"):
        asdex.jacobian(
            f, np.zeros(2), holomorphic=True, mode="fwd", output_format="dense"
        )(x)


@pytest.mark.jacobian
def test_holomorphic_rev_rejects_real_output():
    """``holomorphic=True`` with complex input but real output raises TypeError."""

    def f(z):
        return jnp.array([jnp.abs(z[0])])  # abs returns real

    z = jnp.array([1.0 + 2.0j, 3.0 + 0.5j])
    with pytest.raises(TypeError, match=r"holomorphic.*complex"):
        asdex.jacobian(
            f, np.zeros(2), holomorphic=True, mode="rev", output_format="dense"
        )(z)


@pytest.mark.jacobian
def test_holomorphic_fwd_rejects_real_output():
    """``holomorphic=True`` with complex input but real output raises TypeError."""

    def f(z):
        return jnp.array([jnp.abs(z[0])])  # abs returns real

    z = jnp.array([1.0 + 2.0j, 3.0 + 0.5j])
    with pytest.raises(TypeError, match=r"holomorphic.*complex"):
        asdex.jacobian(
            f, np.zeros(2), holomorphic=True, mode="fwd", output_format="dense"
        )(z)


@pytest.mark.jacobian
def test_rev_rejects_complex_output_without_holomorphic():
    """Complex output without ``holomorphic=True`` raises TypeError in rev mode."""

    def f(x):
        return jnp.array([x[0] + 1j])  # Returns complex

    x = jnp.array([1.0, 2.0])
    with pytest.raises(TypeError, match="holomorphic=True"):
        asdex.jacobian(f, np.zeros(2), mode="rev", output_format="dense")(x)


@pytest.mark.jacobian
def test_rev_rejects_non_floating_output():
    """Non-floating output (e.g. int) raises TypeError in rev mode."""

    def f(x):
        return (x * 10).astype(jnp.int32)  # Returns int with input dependency

    x = jnp.array([1.0, 2.0])
    with pytest.raises(TypeError, match="floating"):
        asdex.jacobian(f, np.zeros(2), mode="rev", output_format="dense")(x)


# kwargs binding


@pytest.mark.jacobian
def test_jacobian_with_kwargs():
    """Functions with kwargs work correctly via bind_kwargs."""

    def f(x, scale=1.0, offset=0.0):
        return x * scale + offset

    x = jnp.array([1.0, 2.0, 3.0])
    jac = asdex.jacobian(f, x, output_format="dense")(x, scale=2.0, offset=1.0)
    expected = jnp.diag(jnp.full(3, 2.0))
    np.testing.assert_allclose(jac, expected)


@pytest.mark.hessian
def test_hessian_with_kwargs():
    """Hessian with kwargs works correctly."""

    def f(x, scale=1.0):
        return scale * jnp.sum(x**2)

    x = jnp.array([1.0, 2.0])
    hess = asdex.hessian(f, x, output_format="dense")(x, scale=3.0)
    expected = 6.0 * jnp.eye(2)
    np.testing.assert_allclose(hess, expected)


# Extended dtype validation


@pytest.mark.jacobian
def test_extended_dtype_input_fwd_raises():
    """Extended dtype (e.g. PRNG key) input raises TypeError in fwd mode."""
    key = jax.random.key(0)
    with pytest.raises(TypeError, match="Unsupported input"):
        _check_input_dtype_fwd(holomorphic=False, x=key)


@pytest.mark.jacobian
def test_extended_dtype_input_rev_raises():
    """Extended dtype input raises TypeError in rev mode (without allow_int)."""
    key = jax.random.key(0)
    with pytest.raises(TypeError, match="inexact"):
        _check_input_dtype_rev(holomorphic=False, allow_int=False, x=key)


@pytest.mark.jacobian
def test_extended_dtype_output_rev_raises():
    """Extended dtype output raises TypeError in rev mode."""
    key = jax.random.key(0)
    with pytest.raises(TypeError, match="Unsupported output"):
        _check_output_dtype_rev(holomorphic=False, y=key)


@pytest.mark.jacobian
def test_non_numeric_dtype_input_rev_raises():
    """Non-numeric dtype (e.g. string) input raises TypeError in rev mode."""
    struct = ShapeDtypeStruct((3,), np.str_)
    with pytest.raises(TypeError, match="numerical-valued"):
        _check_input_dtype_rev(holomorphic=False, allow_int=False, x=struct)


# Empty args validation


@pytest.mark.jacobian
def test_jacobian_no_sample_inputs_raises():
    """Calling jacobian with no sample inputs raises TypeError."""
    with pytest.raises(TypeError, match="at least one"):
        avals_from_args(())


# Known limitations


@pytest.mark.jacobian
def test_has_aux_unsupported_primitive_in_aux_succeeds():
    """Unsupported primitives in aux should not break detection.

    The aux branch doesn't affect the main output, so unsupported primitives
    there should be ignored during sparsity detection.
    """

    def f_with_aux(x):
        main = jnp.sum(x**2)
        aux = jnp.fft.fft(x.astype(jnp.complex64))
        return main, aux

    x = jnp.ones(4)
    pattern = asdex.jacobian_sparsity(f_with_aux, x, has_aux=True)
    assert pattern.shape == (1, 4)
    np.testing.assert_array_equal(pattern.todense(), [[1, 1, 1, 1]])


@pytest.mark.jacobian
def test_scalar_sample_input():
    """Plain Python scalars should work as sample inputs, matching JAX."""

    def f(x):
        return x**2

    pattern = asdex.jacobian_sparsity(f, 3.0)
    assert pattern.shape == (1, 1)
    np.testing.assert_array_equal(pattern.todense(), [[1]])


# Complex PyTree inputs and outputs


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_nested_dict_input(mode, output_format, assert_trees_allclose):
    """Nested dict inputs (dict of dicts) match jax.jacobian."""

    def f(params):
        return params["layer1"]["w"] @ params["layer2"]["w"].T

    params = {
        "layer1": {"w": jnp.array([[1.0, 2.0], [3.0, 4.0]])},
        "layer2": {"w": jnp.array([[5.0, 6.0], [7.0, 8.0]])},
    }
    J = asdex.jacobian(f, params, mode=mode, output_format=output_format)(params)
    J_jax = jax.jacobian(f)(params)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_nested_dict_output(mode, output_format, assert_trees_allclose):
    """Nested dict outputs match jax.jacobian."""

    def f(x):
        return {
            "stats": {"mean": jnp.mean(x), "sum": jnp.sum(x)},
            "scaled": x * 2,
        }

    x = jnp.array([1.0, 2.0, 3.0])
    J = asdex.jacobian(f, x, mode=mode, output_format=output_format)(x)
    J_jax = jax.jacobian(f)(x)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_list_of_arrays_input(mode, output_format, assert_trees_allclose):
    """List of arrays as input matches jax.jacobian."""

    def f(params):
        return params[0] + params[1] * 2 + params[2] * 3

    params = [jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]), jnp.array([5.0, 6.0])]
    J = asdex.jacobian(f, params, mode=mode, output_format=output_format)(params)
    J_jax = jax.jacobian(f)(params)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_tuple_of_dicts_output(mode, output_format, assert_trees_allclose):
    """Tuple of dicts as output matches jax.jacobian."""

    def f(x):
        return ({"a": x[:2]}, {"b": x[1:], "c": x * 2})

    x = jnp.array([1.0, 2.0, 3.0])
    J = asdex.jacobian(f, x, mode=mode, output_format=output_format)(x)
    J_jax = jax.jacobian(f)(x)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_mixed_pytree_input_output(mode, output_format, assert_trees_allclose):
    """Both input and output as complex PyTrees match jax.jacobian."""

    def f(params):
        w, b = params["w"], params["b"]
        return {"y": w @ jnp.ones(w.shape[1]) + b, "norm": jnp.sum(w**2)}

    params = {"w": jnp.eye(3, 2), "b": jnp.zeros(3)}
    J = asdex.jacobian(f, params, mode=mode, output_format=output_format)(params)
    J_jax = jax.jacobian(f)(params)
    assert_trees_allclose(J, J_jax)


# Complex argnums scenarios


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_argnums_all_negative(mode, output_format, assert_trees_allclose):
    """All negative argnums match jax.jacobian."""

    def f(x, y, z):
        return x * y + z

    x, y, z = jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]), jnp.array([5.0, 6.0])
    J = asdex.jacobian(
        f, x, y, z, argnums=(-3, -2, -1), mode=mode, output_format=output_format
    )(x, y, z)
    J_jax = jax.jacobian(f, argnums=(-3, -2, -1))(x, y, z)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_argnums_mixed_sign(mode, output_format, assert_trees_allclose):
    """Mixed positive and negative argnums match jax.jacobian."""

    def f(x, y, z):
        return x + y * z

    x, y, z = jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]), jnp.array([5.0, 6.0])
    J = asdex.jacobian(
        f, x, y, z, argnums=(0, -1), mode=mode, output_format=output_format
    )(x, y, z)
    J_jax = jax.jacobian(f, argnums=(0, -1))(x, y, z)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_argnums_skip_middle(mode, output_format, assert_trees_allclose):
    """Argnums skipping middle position matches jax.jacobian."""

    def f(a, b, c, d):
        return a + c + d

    a, b, c, d = jnp.ones(2), jnp.ones(2), jnp.ones(2), jnp.ones(2)
    J = asdex.jacobian(
        f, a, b, c, d, argnums=(0, 2, 3), mode=mode, output_format=output_format
    )(a, b, c, d)
    J_jax = jax.jacobian(f, argnums=(0, 2, 3))(a, b, c, d)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_argnums_reversed_four_args(
    mode, output_format, assert_trees_allclose
):
    """Reversed argnums with four args matches jax.jacobian."""

    def f(a, b, c, d):
        return a * b + c * d

    a = jnp.array([1.0, 2.0])
    b = jnp.array([3.0, 4.0])
    c = jnp.array([5.0, 6.0])
    d = jnp.array([7.0, 8.0])
    J = asdex.jacobian(
        f, a, b, c, d, argnums=(3, 2, 1, 0), mode=mode, output_format=output_format
    )(a, b, c, d)
    J_jax = jax.jacobian(f, argnums=(3, 2, 1, 0))(a, b, c, d)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_argnums_pytree_args_reversed(
    mode, output_format, assert_trees_allclose
):
    """Reversed argnums with PyTree args matches jax.jacobian."""

    def f(p, q):
        return p["a"] * q["b"]

    p = {"a": jnp.array([1.0, 2.0, 3.0])}
    q = {"b": jnp.array([4.0, 5.0, 6.0])}
    J = asdex.jacobian(f, p, q, argnums=(1, 0), mode=mode, output_format=output_format)(
        p, q
    )
    J_jax = jax.jacobian(f, argnums=(1, 0))(p, q)
    assert_trees_allclose(J, J_jax)


# Hessian complex scenarios


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_nested_dict_input(mode, output_format, assert_trees_allclose):
    """Hessian with nested dict input matches jax.hessian."""

    def f(params):
        w = params["layer"]["w"]
        return jnp.sum(w**2)

    params = {"layer": {"w": jnp.array([1.0, 2.0, 3.0])}}
    H = asdex.hessian(f, params, mode=mode, output_format=output_format)(params)
    H_jax = jax.hessian(f)(params)
    assert_trees_allclose(H, H_jax, atol=1e-6)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_argnums_reversed(mode, output_format, assert_trees_allclose):
    """Hessian with reversed argnums matches jax.hessian."""

    def f(x, y):
        return jnp.dot(x, y) + jnp.sum(x**2)

    x, y = jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])
    H = asdex.hessian(f, x, y, argnums=(1, 0), mode=mode, output_format=output_format)(
        x, y
    )
    H_jax = jax.hessian(f, argnums=(1, 0))(x, y)
    assert_trees_allclose(H, H_jax, atol=1e-6)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_argnums_negative(mode, output_format, assert_trees_allclose):
    """Hessian with negative argnums matches jax.hessian."""

    def f(x, y, z):
        return jnp.dot(x, y) + jnp.sum(z**2)

    x = jnp.array([1.0, 2.0])
    y = jnp.array([3.0, 4.0])
    z = jnp.array([5.0, 6.0])
    H = asdex.hessian(
        f, x, y, z, argnums=(-3, -1), mode=mode, output_format=output_format
    )(x, y, z)
    H_jax = jax.hessian(f, argnums=(-3, -1))(x, y, z)
    assert_trees_allclose(H, H_jax, atol=1e-6)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_argnums_skip_middle(mode, output_format, assert_trees_allclose):
    """Hessian skipping middle arg matches jax.hessian."""

    def f(a, b, c):
        return jnp.sum(a**2) + jnp.dot(a, c)

    a, b, c = jnp.ones(3), jnp.ones(3), jnp.ones(3) * 2
    H = asdex.hessian(
        f, a, b, c, argnums=(0, 2), mode=mode, output_format=output_format
    )(a, b, c)
    H_jax = jax.hessian(f, argnums=(0, 2))(a, b, c)
    assert_trees_allclose(H, H_jax, atol=1e-6)


# has_aux with complex PyTrees


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_has_aux_pytree_output(mode, output_format, assert_trees_allclose):
    """has_aux with PyTree main output matches jax.jacobian structure."""

    def f(x):
        main = {"y": x**2, "z": x * 2}
        aux = {"metadata": "info", "count": 42}
        return main, aux

    x = jnp.array([1.0, 2.0, 3.0])
    J, aux = asdex.jacobian(f, x, has_aux=True, mode=mode, output_format=output_format)(
        x
    )
    J_jax = jax.jacobian(lambda x: f(x)[0])(x)
    assert_trees_allclose(J, J_jax)
    assert aux["metadata"] == "info"
    assert aux["count"] == 42


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_has_aux_pytree_input(mode, output_format, assert_trees_allclose):
    """has_aux with PyTree input matches jax.jacobian."""

    def f(params):
        main = params["a"] * params["b"]
        aux = {"sum": jnp.sum(params["a"]) + jnp.sum(params["b"])}
        return main, aux

    params = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0, 4.0])}
    J, aux = asdex.jacobian(
        f, params, has_aux=True, mode=mode, output_format=output_format
    )(params)
    J_jax = jax.jacobian(lambda p: f(p)[0])(params)
    assert_trees_allclose(J, J_jax)
    np.testing.assert_allclose(aux["sum"], 10.0)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_has_aux_nested_dict(mode, output_format, assert_trees_allclose):
    """Hessian has_aux with nested dict input matches jax.hessian."""

    def f(params):
        w = params["layer"]["w"]
        main = jnp.sum(w**2)
        aux = {"grad_norm": jnp.linalg.norm(w)}
        return main, aux

    params = {"layer": {"w": jnp.array([1.0, 2.0, 3.0])}}
    H, _aux = asdex.hessian(
        f, params, has_aux=True, mode=mode, output_format=output_format
    )(params)
    H_jax = jax.hessian(lambda p: f(p)[0])(params)
    assert_trees_allclose(H, H_jax, atol=1e-6)


# Multi-input with complex PyTrees


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_two_nested_dicts(mode, output_format, assert_trees_allclose):
    """Two nested dict args match jax.jacobian."""

    def f(p, q):
        return p["layer"]["w"] * q["layer"]["w"]

    p = {"layer": {"w": jnp.array([1.0, 2.0])}}
    q = {"layer": {"w": jnp.array([3.0, 4.0])}}
    J = asdex.jacobian(f, p, q, argnums=(0, 1), mode=mode, output_format=output_format)(
        p, q
    )
    J_jax = jax.jacobian(f, argnums=(0, 1))(p, q)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_asymmetric_nested_pytrees(mode, output_format, assert_trees_allclose):
    """Two PyTree args with different structures match jax.jacobian."""

    def f(model, data):
        return model["w"] @ data["x"] + model["b"]

    model = {"w": jnp.eye(2, 3), "b": jnp.zeros(2)}
    data = {"x": jnp.array([1.0, 2.0, 3.0])}
    J = asdex.jacobian(
        f, model, data, argnums=(0, 1), mode=mode, output_format=output_format
    )(model, data)
    J_jax = jax.jacobian(f, argnums=(0, 1))(model, data)
    assert_trees_allclose(J, J_jax)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_two_nested_dicts(mode, output_format, assert_trees_allclose):
    """Hessian with two nested dict args matches jax.hessian."""

    def f(p, q):
        return jnp.dot(p["layer"]["w"], q["layer"]["w"])

    p = {"layer": {"w": jnp.array([1.0, 2.0])}}
    q = {"layer": {"w": jnp.array([3.0, 4.0])}}
    H = asdex.hessian(f, p, q, argnums=(0, 1), mode=mode, output_format=output_format)(
        p, q
    )
    H_jax = jax.hessian(f, argnums=(0, 1))(p, q)
    assert_trees_allclose(H, H_jax, atol=1e-6)


# Edge cases with single-element and scalar leaves


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_scalar_leaves_in_pytree(mode, output_format, assert_trees_allclose):
    """PyTree with scalar leaves matches jax.jacobian."""

    def f(params):
        return params["a"] * params["scale"] + params["offset"]

    params = {
        "a": jnp.array([1.0, 2.0]),
        "scale": jnp.array(2.0),
        "offset": jnp.array(1.0),
    }
    J = asdex.jacobian(f, params, mode=mode, output_format=output_format)(params)
    J_jax = jax.jacobian(f)(params)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_single_element_arrays(mode, output_format, assert_trees_allclose):
    """PyTree with single-element arrays matches jax.jacobian."""

    def f(params):
        return params["vec"] * params["scalar"][0]

    params = {"vec": jnp.array([1.0, 2.0, 3.0]), "scalar": jnp.array([2.0])}
    J = asdex.jacobian(f, params, mode=mode, output_format=output_format)(params)
    J_jax = jax.jacobian(f)(params)
    assert_trees_allclose(J, J_jax)


# Multi-dimensional array leaves in PyTrees


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_2d_arrays_in_pytree(mode, output_format, assert_trees_allclose):
    """PyTree with 2D array leaves matches jax.jacobian."""

    def f(params):
        return params["W"] @ params["x"]

    params = {"W": jnp.eye(3, 2), "x": jnp.array([1.0, 2.0])}
    J = asdex.jacobian(f, params, mode=mode, output_format=output_format)(params)
    J_jax = jax.jacobian(f)(params)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_3d_arrays_in_pytree(mode, output_format, assert_trees_allclose):
    """PyTree with 3D array leaves matches jax.jacobian."""

    def f(params):
        return jnp.sum(params["tensor"], axis=(0, 1)) + params["bias"]

    params = {"tensor": jnp.ones((2, 3, 4)), "bias": jnp.zeros(4)}
    J = asdex.jacobian(f, params, mode=mode, output_format=output_format)(params)
    J_jax = jax.jacobian(f)(params)
    assert_trees_allclose(J, J_jax)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_2d_array_input(mode, output_format, assert_trees_allclose):
    """Hessian with 2D array in PyTree matches jax.hessian."""

    def f(params):
        W = params["W"]
        return jnp.sum(W**2) + jnp.sum(W)

    params = {"W": jnp.array([[1.0, 2.0], [3.0, 4.0]])}
    H = asdex.hessian(f, params, mode=mode, output_format=output_format)(params)
    H_jax = jax.hessian(f)(params)
    assert_trees_allclose(H, H_jax, atol=1e-6)


# Combined complex scenarios


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_complex_multi_input_multi_output(
    mode, output_format, assert_trees_allclose
):
    """Complex multi-input with complex multi-output matches jax.jacobian."""

    def f(model, data):
        y = model["W"] @ data["x"] + model["b"]
        return {"predictions": y, "loss": jnp.sum(y**2)}

    model = {"W": jnp.eye(2, 3), "b": jnp.zeros(2)}
    data = {"x": jnp.array([1.0, 2.0, 3.0])}
    J = asdex.jacobian(
        f, model, data, argnums=(0, 1), mode=mode, output_format=output_format
    )(model, data)
    J_jax = jax.jacobian(f, argnums=(0, 1))(model, data)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_reversed_argnums_complex_pytrees(
    mode, output_format, assert_trees_allclose
):
    """Reversed argnums with complex PyTrees matches jax.jacobian."""

    def f(model, config, data):
        scale = config["scale"]
        return model["W"] @ data["x"] * scale

    model = {"W": jnp.eye(2, 3)}
    config = {"scale": jnp.array([1.0, 2.0])}
    data = {"x": jnp.array([1.0, 2.0, 3.0])}
    J = asdex.jacobian(
        f, model, config, data, argnums=(2, 0), mode=mode, output_format=output_format
    )(model, config, data)
    J_jax = jax.jacobian(f, argnums=(2, 0))(model, config, data)
    assert_trees_allclose(J, J_jax)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_complex_multi_input(mode, output_format, assert_trees_allclose):
    """Hessian with complex multi-input PyTrees matches jax.hessian."""

    def f(model, data):
        W, b = model["W"], model["b"]
        x = data["x"]
        pred = jnp.dot(W, x) + b
        return jnp.sum(pred**2)

    model = {"W": jnp.array([1.0, 2.0, 3.0]), "b": jnp.array(1.0)}
    data = {"x": jnp.array([1.0, 2.0, 3.0])}
    H = asdex.hessian(
        f, model, data, argnums=(0, 1), mode=mode, output_format=output_format
    )(model, data)
    H_jax = jax.hessian(f, argnums=(0, 1))(model, data)
    assert_trees_allclose(H, H_jax, atol=1e-6)


# argnums=(0,) vs argnums=0 semantic distinction


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_argnums_single_element_tuple(
    mode, output_format, assert_trees_allclose
):
    """argnums=(0,) returns tuple of one Jacobian, not single Jacobian.

    This is a critical semantic distinction in JAX:
    argnums=0 returns J, argnums=(0,) returns (J,).
    """

    def f(x, y):
        return x * y

    x, y = jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])
    J = asdex.jacobian(f, x, y, argnums=(0,), mode=mode, output_format=output_format)(
        x, y
    )
    J_jax = jax.jacobian(f, argnums=(0,))(x, y)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_argnums_int_vs_tuple_structure(
    mode, output_format, assert_trees_allclose
):
    """argnums=0 and argnums=(0,) produce different structures."""

    def f(x):
        return x**2

    x = jnp.array([1.0, 2.0, 3.0])
    J_int = asdex.jacobian(f, x, argnums=0, mode=mode, output_format=output_format)(x)
    J_tuple = asdex.jacobian(
        f, x, argnums=(0,), mode=mode, output_format=output_format
    )(x)
    J_jax_int = jax.jacobian(f, argnums=0)(x)
    J_jax_tuple = jax.jacobian(f, argnums=(0,))(x)

    assert_trees_allclose(J_int, J_jax_int)
    assert_trees_allclose(J_tuple, J_jax_tuple)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_argnums_single_element_tuple(
    mode, output_format, assert_trees_allclose
):
    """Hessian with argnums=(0,) returns tuple structure."""

    def f(x, y):
        return jnp.sum(x**2) + jnp.dot(x, y)

    x, y = jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])
    H = asdex.hessian(f, x, y, argnums=(0,), mode=mode, output_format=output_format)(
        x, y
    )
    H_jax = jax.hessian(f, argnums=(0,))(x, y)
    assert_trees_allclose(H, H_jax, atol=1e-6)


# value_and_* APIs with PyTrees


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_value_and_jacobian_dict_input(mode, output_format, assert_trees_allclose):
    """value_and_jacobian with dict input matches JAX."""

    def f(params):
        return params["w"] @ params["x"] + params["b"]

    params = {"w": jnp.eye(2, 3), "x": jnp.array([1.0, 2.0, 3.0]), "b": jnp.zeros(2)}
    val, J = asdex.value_and_jacobian(
        f, params, mode=mode, output_format=output_format
    )(params)
    J_jax = jax.jacobian(f)(params)
    np.testing.assert_allclose(val, f(params))
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_value_and_jacobian_nested_dict(mode, output_format, assert_trees_allclose):
    """value_and_jacobian with nested dict matches JAX."""

    def f(params):
        return params["layer"]["w"] @ jnp.ones(2)

    params = {"layer": {"w": jnp.eye(3, 2)}}
    val, J = asdex.value_and_jacobian(
        f, params, mode=mode, output_format=output_format
    )(params)
    J_jax = jax.jacobian(f)(params)
    np.testing.assert_allclose(val, f(params))
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_value_and_jacobian_multi_input_pytree(
    mode, output_format, assert_trees_allclose
):
    """value_and_jacobian with multiple PyTree args matches JAX."""

    def f(model, data):
        return model["w"] @ data["x"]

    model = {"w": jnp.eye(2, 3)}
    data = {"x": jnp.array([1.0, 2.0, 3.0])}
    val, J = asdex.value_and_jacobian(
        f, model, data, argnums=(0, 1), mode=mode, output_format=output_format
    )(model, data)
    J_jax = jax.jacobian(f, argnums=(0, 1))(model, data)
    np.testing.assert_allclose(val, f(model, data))
    assert_trees_allclose(J, J_jax)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_value_and_hessian_dict_input(mode, output_format, assert_trees_allclose):
    """value_and_hessian with dict input matches JAX."""

    def f(params):
        return jnp.sum(params["w"] ** 2) + jnp.sum(params["b"] ** 2)

    params = {"w": jnp.array([1.0, 2.0]), "b": jnp.array([3.0, 4.0])}
    val, H = asdex.value_and_hessian(f, params, mode=mode, output_format=output_format)(
        params
    )
    H_jax = jax.hessian(f)(params)
    np.testing.assert_allclose(val, f(params))
    assert_trees_allclose(H, H_jax, atol=1e-6)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_value_and_hessian_multi_input(mode, output_format, assert_trees_allclose):
    """value_and_hessian with multiple inputs matches JAX."""

    def f(x, y):
        return jnp.dot(x, y) + jnp.sum(x**2)

    x, y = jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])
    val, H = asdex.value_and_hessian(
        f, x, y, argnums=(0, 1), mode=mode, output_format=output_format
    )(x, y)
    H_jax = jax.hessian(f, argnums=(0, 1))(x, y)
    np.testing.assert_allclose(val, f(x, y))
    assert_trees_allclose(H, H_jax, atol=1e-6)


# *_from_coloring APIs with PyTrees


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_from_coloring_dict_input(mode, output_format, assert_trees_allclose):
    """jacobian_from_coloring with dict input matches JAX."""

    def f(params):
        return params["w"] @ params["x"]

    params = {"w": jnp.eye(2, 3), "x": jnp.array([1.0, 2.0, 3.0])}
    coloring = asdex.jacobian_coloring(f, params, mode=mode)
    J = asdex.jacobian_from_coloring(f, coloring, output_format=output_format)(params)
    J_jax = jax.jacobian(f)(params)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_from_coloring_multi_input(mode, output_format, assert_trees_allclose):
    """jacobian_from_coloring with multiple args matches JAX."""

    def f(x, y):
        return x * y + x

    x, y = jnp.array([1.0, 2.0, 3.0]), jnp.array([4.0, 5.0, 6.0])
    coloring = asdex.jacobian_coloring(f, x, y, argnums=(0, 1), mode=mode)
    J = asdex.jacobian_from_coloring(f, coloring, output_format=output_format)(x, y)
    J_jax = jax.jacobian(f, argnums=(0, 1))(x, y)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_value_and_jacobian_from_coloring_dict(
    mode, output_format, assert_trees_allclose
):
    """value_and_jacobian_from_coloring with dict input matches JAX."""

    def f(params):
        return params["a"] * params["b"]

    params = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0, 4.0])}
    coloring = asdex.jacobian_coloring(f, params, mode=mode)
    val, J = asdex.value_and_jacobian_from_coloring(
        f, coloring, output_format=output_format
    )(params)
    J_jax = jax.jacobian(f)(params)
    np.testing.assert_allclose(val, f(params))
    assert_trees_allclose(J, J_jax)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_from_coloring_dict_input(mode, output_format, assert_trees_allclose):
    """hessian_from_coloring with dict input matches JAX."""

    def f(params):
        return jnp.sum(params["a"] ** 2) + jnp.dot(params["a"], params["b"])

    params = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0, 4.0])}
    coloring = asdex.hessian_coloring(f, params, mode=mode)
    H = asdex.hessian_from_coloring(f, coloring, output_format=output_format)(params)
    H_jax = jax.hessian(f)(params)
    assert_trees_allclose(H, H_jax, atol=1e-6)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_value_and_hessian_from_coloring_dict(
    mode, output_format, assert_trees_allclose
):
    """value_and_hessian_from_coloring with dict input matches JAX."""

    def f(params):
        return jnp.sum(params["w"] ** 2)

    params = {"w": jnp.array([1.0, 2.0, 3.0])}
    coloring = asdex.hessian_coloring(f, params, mode=mode)
    val, H = asdex.value_and_hessian_from_coloring(
        f, coloring, output_format=output_format
    )(params)
    H_jax = jax.hessian(f)(params)
    np.testing.assert_allclose(val, f(params))
    assert_trees_allclose(H, H_jax, atol=1e-6)


# Deep nesting (3+ levels)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_triple_nested_dict(mode, output_format, assert_trees_allclose):
    """Triple-nested dict input matches jax.jacobian."""

    def f(params):
        return params["net"]["layer"]["w"] @ jnp.ones(2)

    params = {"net": {"layer": {"w": jnp.eye(3, 2)}}}
    J = asdex.jacobian(f, params, mode=mode, output_format=output_format)(params)
    J_jax = jax.jacobian(f)(params)
    assert_trees_allclose(J, J_jax)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_triple_nested_dict(mode, output_format, assert_trees_allclose):
    """Hessian with triple-nested dict input matches jax.hessian."""

    def f(params):
        w = params["net"]["layer"]["w"]
        return jnp.sum(w**2)

    params = {"net": {"layer": {"w": jnp.array([1.0, 2.0, 3.0])}}}
    H = asdex.hessian(f, params, mode=mode, output_format=output_format)(params)
    H_jax = jax.hessian(f)(params)
    assert_trees_allclose(H, H_jax, atol=1e-6)


# Mixed container types


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_dict_containing_list(mode, output_format, assert_trees_allclose):
    """Dict containing list as input matches jax.jacobian."""

    def f(params):
        return params["weights"][0] + params["weights"][1] * 2

    params = {"weights": [jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])]}
    J = asdex.jacobian(f, params, mode=mode, output_format=output_format)(params)
    J_jax = jax.jacobian(f)(params)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_dict_containing_tuple(mode, output_format, assert_trees_allclose):
    """Dict containing tuple as input matches jax.jacobian."""

    def f(params):
        w, b = params["layer"]
        return w @ jnp.ones(2) + b

    params = {"layer": (jnp.eye(3, 2), jnp.zeros(3))}
    J = asdex.jacobian(f, params, mode=mode, output_format=output_format)(params)
    J_jax = jax.jacobian(f)(params)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_mixed_depths_in_dict(mode, output_format, assert_trees_allclose):
    """Dict with mixed nesting depths matches jax.jacobian."""

    def f(params):
        return params["shallow"] + params["deep"]["inner"]["w"]

    params = {
        "shallow": jnp.array([1.0, 2.0]),
        "deep": {"inner": {"w": jnp.array([3.0, 4.0])}},
    }
    J = asdex.jacobian(f, params, mode=mode, output_format=output_format)(params)
    J_jax = jax.jacobian(f)(params)
    assert_trees_allclose(J, J_jax)


# List as input/output


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_list_output(mode, output_format, assert_trees_allclose):
    """List as output matches jax.jacobian."""

    def f(x):
        return [x[:2], x[1:] * 2]

    x = jnp.array([1.0, 2.0, 3.0])
    J = asdex.jacobian(f, x, mode=mode, output_format=output_format)(x)
    J_jax = jax.jacobian(f)(x)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_list_input_list_output(mode, output_format, assert_trees_allclose):
    """List input and list output matches jax.jacobian."""

    def f(params):
        return [params[0] + params[1], params[0] * params[1]]

    params = [jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])]
    J = asdex.jacobian(f, params, mode=mode, output_format=output_format)(params)
    J_jax = jax.jacobian(f)(params)
    assert_trees_allclose(J, J_jax)


# Tuple output


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_tuple_output(mode, output_format, assert_trees_allclose):
    """Tuple as output matches jax.jacobian."""

    def f(x):
        return (x[:2], x[1:] * 2, x**2)

    x = jnp.array([1.0, 2.0, 3.0])
    J = asdex.jacobian(f, x, mode=mode, output_format=output_format)(x)
    J_jax = jax.jacobian(f)(x)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_nested_tuple_output(mode, output_format, assert_trees_allclose):
    """Nested tuple as output matches jax.jacobian."""

    def f(x):
        return ((x[:2], x[1:]), (x * 2,))

    x = jnp.array([1.0, 2.0, 3.0])
    J = asdex.jacobian(f, x, mode=mode, output_format=output_format)(x)
    J_jax = jax.jacobian(f)(x)
    assert_trees_allclose(J, J_jax)


# has_aux with PyTree aux


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_has_aux_pytree_aux(mode, output_format, assert_trees_allclose):
    """has_aux where aux itself is a PyTree."""

    def f(x):
        main = x**2
        aux = {"stats": {"mean": jnp.mean(x), "std": jnp.std(x)}, "raw": x}
        return main, aux

    x = jnp.array([1.0, 2.0, 3.0])
    J, aux = asdex.jacobian(f, x, has_aux=True, mode=mode, output_format=output_format)(
        x
    )
    J_jax = jax.jacobian(lambda x: f(x)[0])(x)
    assert_trees_allclose(J, J_jax)
    np.testing.assert_allclose(aux["stats"]["mean"], jnp.mean(x))


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_has_aux_pytree_aux(mode, output_format, assert_trees_allclose):
    """Hessian has_aux where aux itself is a PyTree."""

    def f(x):
        main = jnp.sum(x**2)
        aux = {"norm": jnp.linalg.norm(x), "info": {"size": x.shape[0]}}
        return main, aux

    x = jnp.array([1.0, 2.0, 3.0])
    H, aux = asdex.hessian(f, x, has_aux=True, mode=mode, output_format=output_format)(
        x
    )
    H_jax = jax.hessian(lambda x: f(x)[0])(x)
    assert_trees_allclose(H, H_jax, atol=1e-6)
    np.testing.assert_allclose(aux["norm"], jnp.linalg.norm(x))


# Multi-arg with different structure depths


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_nested_and_flat_args(mode, output_format, assert_trees_allclose):
    """One nested dict arg and one flat dict arg match jax.jacobian."""

    def f(nested, flat):
        return nested["layer"]["w"] * flat["scale"]

    nested = {"layer": {"w": jnp.array([1.0, 2.0])}}
    flat = {"scale": jnp.array([3.0, 4.0])}
    J = asdex.jacobian(
        f, nested, flat, argnums=(0, 1), mode=mode, output_format=output_format
    )(nested, flat)
    J_jax = jax.jacobian(f, argnums=(0, 1))(nested, flat)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_array_and_nested_dict(mode, output_format, assert_trees_allclose):
    """One array arg and one nested dict arg match jax.jacobian."""

    def f(scale, params):
        return scale * params["layer"]["w"]

    scale = jnp.array([1.0, 2.0])
    params = {"layer": {"w": jnp.array([3.0, 4.0])}}
    J = asdex.jacobian(
        f, scale, params, argnums=(0, 1), mode=mode, output_format=output_format
    )(scale, params)
    J_jax = jax.jacobian(f, argnums=(0, 1))(scale, params)
    assert_trees_allclose(J, J_jax)


# kwargs with PyTree inputs


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_kwargs_with_pytree_input(mode, output_format, assert_trees_allclose):
    """Kwargs work correctly with PyTree inputs."""

    def f(params, scale=1.0, offset=0.0):
        return params["w"] * scale + offset

    params = {"w": jnp.array([1.0, 2.0, 3.0])}
    J = asdex.jacobian(f, params, mode=mode, output_format=output_format)(
        params, scale=2.0, offset=1.0
    )
    J_jax = jax.jacobian(lambda p: f(p, scale=2.0, offset=1.0))(params)
    assert_trees_allclose(J, J_jax)


# Consistency: same logical structure for dense and bcoo


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
def test_jacobian_dense_bcoo_same_structure(mode, assert_trees_allclose):
    """Dense and BCOO outputs have identical logical structure."""

    def f(params):
        return {"y": params["w"] @ params["x"], "z": params["b"]}

    params = {"w": jnp.eye(2, 3), "x": jnp.array([1.0, 2.0, 3.0]), "b": jnp.zeros(2)}
    J_dense = asdex.jacobian(f, params, mode=mode, output_format="dense")(params)
    J_bcoo = asdex.jacobian(f, params, mode=mode, output_format="bcoo")(params)
    assert_trees_allclose(J_bcoo, J_dense)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
def test_hessian_dense_bcoo_same_structure(mode, assert_trees_allclose):
    """Hessian dense and BCOO outputs have identical logical structure."""

    def f(params):
        return jnp.sum(params["a"] ** 2) + jnp.dot(params["a"], params["b"])

    params = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0, 4.0])}
    H_dense = asdex.hessian(f, params, mode=mode, output_format="dense")(params)
    H_bcoo = asdex.hessian(f, params, mode=mode, output_format="bcoo")(params)
    assert_trees_allclose(H_bcoo, H_dense, atol=1e-6)


# Sparsity pattern shape consistency


@pytest.mark.jacobian
def test_jacobian_sparsity_shape_matches_jacobian():
    """jacobian_sparsity shape matches flattened jacobian dimensions."""

    def f(params):
        return params["w"] @ params["x"]

    params = {"w": jnp.eye(2, 3), "x": jnp.array([1.0, 2.0, 3.0])}
    pattern = asdex.jacobian_sparsity(f, params)

    # Output has 2 elements, input has 6+3=9 elements
    total_in = sum(leaf.size for leaf in jax.tree.leaves(params))
    expected_shape = (2, total_in)

    assert pattern.shape == expected_shape


@pytest.mark.hessian
def test_hessian_sparsity_shape_matches_hessian():
    """hessian_sparsity shape matches flattened hessian dimensions."""

    def f(params):
        return jnp.sum(params["a"] ** 2) + jnp.sum(params["b"] ** 2)

    params = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0, 4.0])}
    pattern = asdex.hessian_sparsity(f, params)

    total_in = sum(leaf.size for leaf in jax.tree.leaves(params))
    expected_shape = (total_in, total_in)

    assert pattern.shape == expected_shape


# Empty and edge cases


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_single_leaf_pytree(mode, output_format, assert_trees_allclose):
    """Single-leaf PyTree behaves like the leaf itself."""

    def f(params):
        return params["w"] ** 2

    params = {"w": jnp.array([1.0, 2.0, 3.0])}
    J = asdex.jacobian(f, params, mode=mode, output_format=output_format)(params)
    J_jax = jax.jacobian(f)(params)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_all_scalar_leaves(mode, output_format, assert_trees_allclose):
    """PyTree with all scalar leaves matches jax.jacobian."""

    def f(params):
        return jnp.array([params["a"] + params["b"], params["a"] * params["b"]])

    params = {"a": jnp.array(2.0), "b": jnp.array(3.0)}
    J = asdex.jacobian(f, params, mode=mode, output_format=output_format)(params)
    J_jax = jax.jacobian(f)(params)
    assert_trees_allclose(J, J_jax)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_all_scalar_leaves(mode, output_format, assert_trees_allclose):
    """Hessian with all scalar leaves in PyTree matches jax.hessian."""

    def f(params):
        return params["a"] ** 2 + params["a"] * params["b"] + params["b"] ** 2

    params = {"a": jnp.array(2.0), "b": jnp.array(3.0)}
    H = asdex.hessian(f, params, mode=mode, output_format=output_format)(params)
    H_jax = jax.hessian(f)(params)
    assert_trees_allclose(H, H_jax, atol=1e-6)


# Empty/size-0 leaves in PyTrees


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_empty_array_leaf(mode, output_format, assert_trees_allclose):
    """PyTree with empty array leaf matches jax.jacobian."""

    def f(params):
        # Only use the non-empty leaf
        return params["w"] ** 2

    params = {"w": jnp.array([1.0, 2.0]), "empty": jnp.array([])}
    J = asdex.jacobian(f, params, mode=mode, output_format=output_format)(params)
    J_jax = jax.jacobian(f)(params)
    assert_trees_allclose(J, J_jax)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_empty_array_leaf(mode, output_format, assert_trees_allclose):
    """Hessian with empty array leaf in PyTree matches jax.hessian."""

    def f(params):
        return jnp.sum(params["w"] ** 2)

    params = {"w": jnp.array([1.0, 2.0]), "empty": jnp.array([])}
    H = asdex.hessian(f, params, mode=mode, output_format=output_format)(params)
    H_jax = jax.hessian(f)(params)
    assert_trees_allclose(H, H_jax, atol=1e-6)


# namedtuple inputs


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_namedtuple_input(mode, output_format, assert_trees_allclose):
    """Namedtuple input matches jax.jacobian."""
    Params = namedtuple("Params", ["w", "b"])

    def f(params):
        return params.w @ jnp.ones(2) + params.b

    params = Params(w=jnp.eye(3, 2), b=jnp.zeros(3))
    J = asdex.jacobian(f, params, mode=mode, output_format=output_format)(params)
    J_jax = jax.jacobian(f)(params)
    assert_trees_allclose(J, J_jax)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_namedtuple_input(mode, output_format, assert_trees_allclose):
    """Hessian with namedtuple input matches jax.hessian."""
    Params = namedtuple("Params", ["a", "b"])

    def f(params):
        return jnp.sum(params.a**2) + jnp.dot(params.a, params.b)

    params = Params(a=jnp.array([1.0, 2.0]), b=jnp.array([3.0, 4.0]))
    H = asdex.hessian(f, params, mode=mode, output_format=output_format)(params)
    H_jax = jax.hessian(f)(params)
    assert_trees_allclose(H, H_jax, atol=1e-6)


# holomorphic with PyTree inputs


@pytest.mark.jacobian
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_holomorphic_pytree_input(output_format, assert_trees_allclose):
    """holomorphic=True with PyTree complex input matches jax.jacobian."""

    def f(params):
        return params["z"] ** 2

    params = {"z": jnp.array([1.0 + 2.0j, 3.0 + 0.5j])}
    J = asdex.jacobian(f, params, holomorphic=True, output_format=output_format)(params)
    J_jax = jax.jacobian(f, holomorphic=True)(params)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_holomorphic_multi_leaf_pytree(output_format, assert_trees_allclose):
    """holomorphic=True with multi-leaf complex PyTree matches jax.jacobian."""

    def f(params):
        return params["z1"] * params["z2"]

    params = {
        "z1": jnp.array([1.0 + 1.0j, 2.0 + 0.5j]),
        "z2": jnp.array([0.5 + 0.5j, 1.0 + 1.0j]),
    }
    J = asdex.jacobian(f, params, holomorphic=True, output_format=output_format)(params)
    J_jax = jax.jacobian(f, holomorphic=True)(params)
    assert_trees_allclose(J, J_jax)


# allow_int with PyTree inputs


@pytest.mark.jacobian
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_allow_int_pytree_input(output_format, assert_trees_allclose):
    """allow_int=True with PyTree containing integer leaf matches jax.jacobian."""

    def f(params):
        # Cast to float for output
        return (params["x"] + params["idx"]).astype(jnp.float32)

    params = {
        "x": jnp.array([1.0, 2.0, 3.0]),
        "idx": jnp.array([1, 0, 2], dtype=jnp.int32),
    }
    J = asdex.jacobian(
        f, params, mode="rev", allow_int=True, output_format=output_format
    )(params)
    J_jax = jax.jacobian(f, allow_int=True)(params)
    assert_trees_allclose(J, J_jax)


# Non-contiguous descending argnums


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_argnums_noncontiguous_descending(
    mode, output_format, assert_trees_allclose
):
    """Non-contiguous descending argnums (4, 2, 0) matches jax.jacobian."""

    def f(a, b, c, d, e):
        return a + c + e

    a, b, c, d, e = [jnp.ones(2) * i for i in range(5)]
    J = asdex.jacobian(
        f, a, b, c, d, e, argnums=(4, 2, 0), mode=mode, output_format=output_format
    )(a, b, c, d, e)
    J_jax = jax.jacobian(f, argnums=(4, 2, 0))(a, b, c, d, e)
    assert_trees_allclose(J, J_jax)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_argnums_noncontiguous_descending(
    mode, output_format, assert_trees_allclose
):
    """Hessian with non-contiguous descending argnums matches jax.hessian."""

    def f(a, b, c, d):
        return jnp.dot(a, c) + jnp.sum(a**2)

    a, b, c, d = [jnp.ones(2) * i for i in range(4)]
    H = asdex.hessian(
        f, a, b, c, d, argnums=(3, 2, 0), mode=mode, output_format=output_format
    )(a, b, c, d)
    H_jax = jax.hessian(f, argnums=(3, 2, 0))(a, b, c, d)
    assert_trees_allclose(H, H_jax, atol=1e-6)


# Coloring reuse with changing non-diff PyTree args


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_coloring_reuse_pytree_nondiff_arg(
    mode, output_format, assert_trees_allclose
):
    """Coloring built once reuses correctly when non-diff PyTree arg changes."""

    def f(params, config):
        return params["w"] * config["scale"] + config["offset"]

    params = {"w": jnp.array([1.0, 2.0, 3.0])}
    config1 = {
        "scale": jnp.array([1.0, 1.0, 1.0]),
        "offset": jnp.array([0.0, 0.0, 0.0]),
    }
    config2 = {
        "scale": jnp.array([2.0, 3.0, 4.0]),
        "offset": jnp.array([1.0, 2.0, 3.0]),
    }

    coloring = asdex.jacobian_coloring(f, params, config1, argnums=0, mode=mode)
    jac_fn = asdex.jacobian_from_coloring(f, coloring, output_format=output_format)

    # Test with different config values
    for config in [config1, config2]:
        J = jac_fn(params, config)
        J_jax = jax.jacobian(f, argnums=0)(params, config)
        assert_trees_allclose(J, J_jax)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_coloring_reuse_pytree_nondiff_arg(
    mode, output_format, assert_trees_allclose
):
    """Hessian coloring reuses correctly when non-diff PyTree arg changes."""

    def f(params, config):
        return jnp.sum(params["w"] ** 2) * config["scale"]

    params = {"w": jnp.array([1.0, 2.0])}
    config1 = {"scale": jnp.array(1.0)}
    config2 = {"scale": jnp.array(3.0)}

    coloring = asdex.hessian_coloring(f, params, config1, argnums=0, mode=mode)
    hess_fn = asdex.hessian_from_coloring(f, coloring, output_format=output_format)

    for config in [config1, config2]:
        H = hess_fn(params, config)
        H_jax = jax.hessian(f, argnums=0)(params, config)
        assert_trees_allclose(H, H_jax, atol=1e-6)


# Deeply nested output structure


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_triple_nested_dict_output(mode, output_format, assert_trees_allclose):
    """Triple-nested dict output matches jax.jacobian."""

    def f(x):
        return {"level1": {"level2": {"y": x**2}}}

    x = jnp.array([1.0, 2.0, 3.0])
    J = asdex.jacobian(f, x, mode=mode, output_format=output_format)(x)
    J_jax = jax.jacobian(f)(x)
    assert_trees_allclose(J, J_jax)


# Mixed nested input and output


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_nested_input_nested_output(
    mode, output_format, assert_trees_allclose
):
    """Both nested input and nested output match jax.jacobian."""

    def f(params):
        w = params["net"]["layer"]["w"]
        return {"out": {"pred": w @ jnp.ones(2)}}

    params = {"net": {"layer": {"w": jnp.eye(3, 2)}}}
    J = asdex.jacobian(f, params, mode=mode, output_format=output_format)(params)
    J_jax = jax.jacobian(f)(params)
    assert_trees_allclose(J, J_jax)


# value_and_jacobian with has_aux


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_value_and_jacobian_has_aux_pytree(mode, output_format, assert_trees_allclose):
    """value_and_jacobian with has_aux and PyTree input matches JAX."""

    def f(params):
        main = params["w"] ** 2
        aux = {"sum": jnp.sum(params["w"])}
        return main, aux

    params = {"w": jnp.array([1.0, 2.0, 3.0])}
    (val, aux), J = asdex.value_and_jacobian(
        f, params, has_aux=True, mode=mode, output_format=output_format
    )(params)
    J_jax = jax.jacobian(lambda p: f(p)[0])(params)
    np.testing.assert_allclose(val, f(params)[0])
    np.testing.assert_allclose(aux["sum"], jnp.sum(params["w"]))
    assert_trees_allclose(J, J_jax)


# value_and_hessian with has_aux


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_value_and_hessian_has_aux_pytree(mode, output_format, assert_trees_allclose):
    """value_and_hessian with has_aux and PyTree input matches JAX."""

    def f(params):
        main = jnp.sum(params["w"] ** 2)
        aux = {"norm": jnp.linalg.norm(params["w"])}
        return main, aux

    params = {"w": jnp.array([1.0, 2.0, 3.0])}
    (val, aux), H = asdex.value_and_hessian(
        f, params, has_aux=True, mode=mode, output_format=output_format
    )(params)
    H_jax = jax.hessian(lambda p: f(p)[0])(params)
    np.testing.assert_allclose(val, f(params)[0])
    np.testing.assert_allclose(aux["norm"], jnp.linalg.norm(params["w"]))
    assert_trees_allclose(H, H_jax, atol=1e-6)


# Multiple PyTree outputs with different structures


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_heterogeneous_output_pytree(
    mode, output_format, assert_trees_allclose
):
    """Output with mixed dict, tuple, list matches jax.jacobian."""

    def f(x):
        return {"a": x[:2]}, (x[1:],), [x * 2]

    x = jnp.array([1.0, 2.0, 3.0])
    J = asdex.jacobian(f, x, mode=mode, output_format=output_format)(x)
    J_jax = jax.jacobian(f)(x)
    assert_trees_allclose(J, J_jax)
