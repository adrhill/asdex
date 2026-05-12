"""Tests for ``has_aux``, ``holomorphic``, and ``allow_int`` kwargs.

Mirrors ``jax.jacrev`` / ``jax.jacfwd`` / ``jax.grad`` / ``jax.hessian``
semantics on the asdex public API.
"""

import warnings

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

    np.testing.assert_allclose(hess[0][0], np.array([[0.0, 0.0], [0.0, 2.0]]))
    np.testing.assert_allclose(hess[1][1], np.array([[0.0, 0.0], [0.0, 2.0]]))
    np.testing.assert_allclose(hess[0][1], np.array([[1.0, 0.0], [0.0, 0.0]]))
    np.testing.assert_allclose(hess[1][0], np.array([[1.0, 0.0], [0.0, 0.0]]))
    np.testing.assert_allclose(aux, 4.0)


# has_aux with PyTree inputs/outputs


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


# allow_int


@pytest.mark.jacobian
def test_allow_int_permits_int_input():
    """``allow_int=True`` bypasses the reverse-mode integer-input dtype check."""

    def f(x):
        return jnp.array([x[0] + x[1], x[0] * 2], dtype=jnp.float32)

    x = jnp.array([1, 2], dtype=jnp.int32)
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


@pytest.mark.jacobian
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_allow_int_pytree_input(output_format, assert_trees_allclose):
    """allow_int=True with PyTree containing integer leaf matches jax.jacobian."""

    def f(params):
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
        return jnp.array([jnp.abs(z[0])])

    z = jnp.array([1.0 + 2.0j, 3.0 + 0.5j])
    with pytest.raises(TypeError, match=r"holomorphic.*complex"):
        asdex.jacobian(
            f, np.zeros(2), holomorphic=True, mode="rev", output_format="dense"
        )(z)


@pytest.mark.jacobian
def test_holomorphic_fwd_rejects_real_output():
    """``holomorphic=True`` with complex input but real output raises TypeError."""

    def f(z):
        return jnp.array([jnp.abs(z[0])])

    z = jnp.array([1.0 + 2.0j, 3.0 + 0.5j])
    with pytest.raises(TypeError, match=r"holomorphic.*complex"):
        asdex.jacobian(
            f, np.zeros(2), holomorphic=True, mode="fwd", output_format="dense"
        )(z)


@pytest.mark.jacobian
def test_rev_rejects_complex_output_without_holomorphic():
    """Complex output without ``holomorphic=True`` raises TypeError in rev mode."""

    def f(x):
        return jnp.array([x[0] + 1j])

    x = jnp.array([1.0, 2.0])
    with pytest.raises(TypeError, match="holomorphic=True"):
        asdex.jacobian(f, np.zeros(2), mode="rev", output_format="dense")(x)


@pytest.mark.jacobian
def test_rev_rejects_non_floating_output():
    """Non-floating output (e.g. int) raises TypeError in rev mode."""

    def f(x):
        return (x * 10).astype(jnp.int32)

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
