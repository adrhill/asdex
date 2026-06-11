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


@pytest.mark.hessian
def test_hessian_has_aux_all_modes(hessian_mode):
    """``hessian(f, has_aux=True)`` returns correct Hessian and aux in every mode."""

    def f(x):
        y = jnp.sum(x**2) + x[0] * x[1]
        return y, {"input_sum": jnp.sum(x)}

    x = jnp.array([2.0, 3.0, 4.0])
    hess, aux = asdex.hessian(
        f, np.zeros(3), has_aux=True, mode=hessian_mode, output_format="dense"
    )(x)

    expected = jax.hessian(lambda x: f(x)[0])(x)
    np.testing.assert_allclose(hess, expected)
    np.testing.assert_allclose(aux["input_sum"], 9.0)


@pytest.mark.hessian
def test_value_and_hessian_has_aux_all_modes(hessian_mode):
    """``value_and_hessian(f, has_aux=True)`` is correct in every mode."""

    def f(x):
        y = jnp.sum(x**2) + x[0] * x[1]
        return y, {"input_sum": jnp.sum(x)}

    x = jnp.array([2.0, 3.0, 4.0])
    (value, aux), hess = asdex.value_and_hessian(
        f, np.zeros(3), has_aux=True, mode=hessian_mode, output_format="dense"
    )(x)

    np.testing.assert_allclose(value, f(x)[0])
    assert value.shape == ()
    np.testing.assert_allclose(aux["input_sum"], 9.0)
    np.testing.assert_allclose(hess, jax.hessian(lambda x: f(x)[0])(x))


@pytest.mark.hessian
def test_hessian_has_aux_forward_pass_count(hessian_mode):
    """Aux rides along with the HVP forward pass instead of an extra ``f`` call.

    Steady-state executions of ``f``'s Python body per call:
    one trace for the HVPs with aux threaded through
    ``linearize``/``vjp`` (``fwd_over_rev`` / ``rev_over_rev``),
    plus one dedicated aux call only for ``rev_over_fwd``,
    whose forward passes happen inside the vmapped HVPs.
    """
    calls = 0

    def f(x):
        nonlocal calls
        calls += 1
        y = jnp.sum(x**2) + x[0] * x[1]
        return y, {"input_sum": jnp.sum(x)}

    x = jnp.array([2.0, 3.0, 4.0])
    hess_fn = asdex.hessian(f, np.zeros(3), has_aux=True, mode=hessian_mode)
    hess_fn(x)  # warm up detection and per-closure wrapper caches

    calls = 0
    hess_fn(x)
    expected_calls = 2 if hessian_mode == "rev_over_fwd" else 1
    assert calls == expected_calls


@pytest.mark.hessian
def test_value_and_hessian_forward_pass_count(hessian_mode):
    """The value comes from the HVP forward pass instead of an extra ``f`` call.

    Same per-call execution counts as the aux test above:
    ``fwd_over_rev`` / ``rev_over_rev`` thread the value through
    the aux output of ``linearize``/``vjp``;
    only ``rev_over_fwd`` needs one dedicated call for value and aux.
    """
    calls = 0

    def f(x):
        nonlocal calls
        calls += 1
        y = jnp.sum(x**2) + x[0] * x[1]
        return y, {"input_sum": jnp.sum(x)}

    x = jnp.array([2.0, 3.0, 4.0])
    fn = asdex.value_and_hessian(f, np.zeros(3), has_aux=True, mode=hessian_mode)
    fn(x)  # warm up detection and per-closure wrapper caches

    calls = 0
    (value, aux), _hess = fn(x)
    expected_calls = 2 if hessian_mode == "rev_over_fwd" else 1
    assert calls == expected_calls
    np.testing.assert_allclose(value, 29.0 + 6.0)
    np.testing.assert_allclose(aux["input_sum"], 9.0)


# has_aux with PyTree inputs/outputs


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_has_aux_pytree_output(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """has_aux with PyTree main output matches jax.jacobian structure."""

    def f(x):
        main = {"y": x**2, "z": x * 2}
        aux = {"metadata": "info", "count": 42}
        return main, aux

    x = jnp.array([1.0, 2.0, 3.0])
    J, aux = asdex.jacobian(
        f,
        x,
        has_aux=True,
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(x)
    J_jax = jax.jacobian(lambda x: f(x)[0])(x)
    assert_trees_allclose(J, J_jax)
    assert aux["metadata"] == "info"
    assert aux["count"] == 42


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_has_aux_pytree_input(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """has_aux with PyTree input matches jax.jacobian."""

    def f(params):
        main = params["a"] * params["b"]
        aux = {"sum": jnp.sum(params["a"]) + jnp.sum(params["b"])}
        return main, aux

    params = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0, 4.0])}
    J, aux = asdex.jacobian(
        f,
        params,
        has_aux=True,
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(params)
    J_jax = jax.jacobian(lambda p: f(p)[0])(params)
    assert_trees_allclose(J, J_jax)
    np.testing.assert_allclose(aux["sum"], 10.0)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_has_aux_nested_dict(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Hessian has_aux with nested dict input matches jax.hessian."""

    def f(params):
        w = params["layer"]["w"]
        main = jnp.sum(w**2)
        aux = {"grad_norm": jnp.linalg.norm(w)}
        return main, aux

    params = {"layer": {"w": jnp.array([1.0, 2.0, 3.0])}}
    H, _aux = asdex.hessian(
        f,
        params,
        has_aux=True,
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(params)
    H_jax = jax.hessian(lambda p: f(p)[0])(params)
    assert_trees_allclose(H, H_jax, atol=1e-6)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_has_aux_pytree_aux(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """has_aux where aux itself is a PyTree."""

    def f(x):
        main = x**2
        aux = {"stats": {"mean": jnp.mean(x), "std": jnp.std(x)}, "raw": x}
        return main, aux

    x = jnp.array([1.0, 2.0, 3.0])
    J, aux = asdex.jacobian(
        f,
        x,
        has_aux=True,
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(x)
    J_jax = jax.jacobian(lambda x: f(x)[0])(x)
    assert_trees_allclose(J, J_jax)
    np.testing.assert_allclose(aux["stats"]["mean"], jnp.mean(x))


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_has_aux_pytree_aux(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Hessian has_aux where aux itself is a PyTree."""

    def f(x):
        main = jnp.sum(x**2)
        aux = {"norm": jnp.linalg.norm(x), "info": {"size": x.shape[0]}}
        return main, aux

    x = jnp.array([1.0, 2.0, 3.0])
    H, aux = asdex.hessian(
        f,
        x,
        has_aux=True,
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(x)
    H_jax = jax.hessian(lambda x: f(x)[0])(x)
    assert_trees_allclose(H, H_jax, atol=1e-6)
    np.testing.assert_allclose(aux["norm"], jnp.linalg.norm(x))


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_value_and_jacobian_has_aux_pytree(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """value_and_jacobian with has_aux and PyTree input matches JAX."""

    def f(params):
        main = params["w"] ** 2
        aux = {"sum": jnp.sum(params["w"])}
        return main, aux

    params = {"w": jnp.array([1.0, 2.0, 3.0])}
    (val, aux), J = asdex.value_and_jacobian(
        f,
        params,
        has_aux=True,
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(params)
    J_jax = jax.jacobian(lambda p: f(p)[0])(params)
    np.testing.assert_allclose(val, f(params)[0])
    np.testing.assert_allclose(aux["sum"], jnp.sum(params["w"]))
    assert_trees_allclose(J, J_jax)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_value_and_hessian_has_aux_pytree(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """value_and_hessian with has_aux and PyTree input matches JAX."""

    def f(params):
        main = jnp.sum(params["w"] ** 2)
        aux = {"norm": jnp.linalg.norm(params["w"])}
        return main, aux

    params = {"w": jnp.array([1.0, 2.0, 3.0])}
    (val, aux), H = asdex.value_and_hessian(
        f,
        params,
        has_aux=True,
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
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
def test_jacobian_holomorphic_pytree_input(
    output_format, chunk_size, assert_trees_allclose
):
    """holomorphic=True with PyTree complex input matches jax.jacobian."""

    def f(params):
        return params["z"] ** 2

    params = {"z": jnp.array([1.0 + 2.0j, 3.0 + 0.5j])}
    J = asdex.jacobian(
        f, params, holomorphic=True, output_format=output_format, chunk_size=chunk_size
    )(params)
    J_jax = jax.jacobian(f, holomorphic=True)(params)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_holomorphic_multi_leaf_pytree(
    output_format, chunk_size, assert_trees_allclose
):
    """holomorphic=True with multi-leaf complex PyTree matches jax.jacobian."""

    def f(params):
        return params["z1"] * params["z2"]

    params = {
        "z1": jnp.array([1.0 + 1.0j, 2.0 + 0.5j]),
        "z2": jnp.array([0.5 + 0.5j, 1.0 + 1.0j]),
    }
    J = asdex.jacobian(
        f, params, holomorphic=True, output_format=output_format, chunk_size=chunk_size
    )(params)
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
def test_allow_int_fwd_mode_raises():
    """``allow_int=True`` with forward mode raises TypeError (matches JAX's jacfwd)."""

    def f(x):
        return x.astype(jnp.float32) * 2

    x = jnp.array([1, 2], dtype=jnp.int32)
    with pytest.raises(TypeError, match="not supported in forward mode"):
        asdex.jacobian(
            f, np.zeros(2), mode="fwd", allow_int=True, output_format="dense"
        )(x)


@pytest.mark.jacobian
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_allow_int_pytree_input(
    output_format, chunk_size, assert_trees_allclose
):
    """allow_int=True with PyTree containing integer leaf matches jax.jacobian."""

    def f(params):
        return (params["x"] + params["idx"]).astype(jnp.float32)

    params = {
        "x": jnp.array([1.0, 2.0, 3.0]),
        "idx": jnp.array([1, 0, 2], dtype=jnp.int32),
    }
    J = asdex.jacobian(
        f,
        params,
        mode="rev",
        allow_int=True,
        output_format=output_format,
        chunk_size=chunk_size,
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
    """Functions with default kwargs work correctly."""

    def f(x, scale=1.0, offset=0.0):
        return x * scale + offset

    x = jnp.array([1.0, 2.0, 3.0])
    jac = asdex.jacobian(f, x, output_format="dense")(x, scale=2.0, offset=1.0)
    expected = jnp.diag(jnp.full(3, 2.0))
    np.testing.assert_allclose(jac, expected)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_positional_args_as_kwargs(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Positional args can be passed as kwargs at both detection and call time.

    Mirrors ``jax/_src/api.py`` which uses ``inspect.signature(fn).bind(...)``
    to resolve argument positions.
    Regression test for https://github.com/adrhill/asdex/issues/123.
    """

    def f(x, y):
        return x * y

    x = jnp.array([1.0, 2.0])
    y = jnp.array([3.0, 4.0])

    # Kwargs at both detection time and call time
    J = asdex.jacobian(
        f,
        x,
        y=y,
        argnums=0,
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(x, y=y)
    J_jax = jax.jacobian(f)(x, y=y)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_argnums_refers_to_signature_order(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Argnums indexes into signature order, not call-site kwarg order.

    Even when kwargs are passed in a different order than the signature,
    argnums=1 still refers to the second parameter in the signature (b).
    """

    def f(a, b, c):
        return a + b * c

    a = jnp.array([1.0, 2.0])
    b = jnp.array([3.0, 4.0])
    c = jnp.array([5.0, 6.0])

    # Pass kwargs in reverse order: c, a, b - but argnums=1 still means "b"
    J = asdex.jacobian(
        f,
        c=c,
        a=a,
        b=b,
        argnums=1,
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(c=c, a=a, b=b)
    J_jax = jax.jacobian(f, argnums=1)(a, b, c)
    assert_trees_allclose(J, J_jax)

    # Verify the Jacobian is w.r.t. b (diagonal of c values), not a or c
    expected = jnp.diag(c)
    np.testing.assert_allclose(J if output_format == "dense" else J.todense(), expected)


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
def test_jacobian_kwargs_with_pytree_input(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Kwargs work correctly with PyTree inputs."""

    def f(params, scale=1.0, offset=0.0):
        return params["w"] * scale + offset

    params = {"w": jnp.array([1.0, 2.0, 3.0])}
    J = asdex.jacobian(
        f, params, mode=mode, output_format=output_format, chunk_size=chunk_size
    )(params, scale=2.0, offset=1.0)
    J_jax = jax.jacobian(lambda p: f(p, scale=2.0, offset=1.0))(params)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_multiple_pytree_args_as_kwargs(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Multiple pytree positional args can be passed as kwargs."""

    def f(params, data):
        return params["w"] * data["x"] + params["b"]

    params = {"w": jnp.array([1.0, 2.0]), "b": jnp.array([0.1, 0.2])}
    data = {"x": jnp.array([3.0, 4.0])}

    J = asdex.jacobian(
        f,
        params,
        data=data,
        argnums=0,
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(params=params, data=data)
    J_jax = jax.jacobian(f, argnums=0)(params, data)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_mixed_positional_and_kwarg_pytrees(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Mix of positional pytree args and pytree kwargs."""

    def f(params, data, config):
        scale = config["scale"]
        return params["w"] * data["x"] * scale

    params = {"w": jnp.array([1.0, 2.0, 3.0])}
    data = {"x": jnp.array([2.0, 3.0, 4.0])}
    config = {"scale": jnp.array([0.5, 0.5, 0.5])}

    J = asdex.jacobian(
        f,
        params,
        data=data,
        config=config,
        argnums=0,
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(params, data=data, config=config)
    J_jax = jax.jacobian(f, argnums=0)(params, data, config)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_pytree_args_with_pytree_default_kwargs(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Pytree positional args combined with pytree default kwargs."""

    def f(params, data, config=None):
        if config is None:
            config = {"scale": jnp.ones(3), "bias": jnp.zeros(3)}
        return params["w"] * data["x"] * config["scale"] + config["bias"]

    params = {"w": jnp.array([1.0, 2.0, 3.0])}
    data = {"x": jnp.array([2.0, 3.0, 4.0])}
    config = {"scale": jnp.array([0.5, 0.5, 0.5]), "bias": jnp.array([1.0, 1.0, 1.0])}

    J = asdex.jacobian(
        f,
        params,
        data=data,
        config=config,
        argnums=0,
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(params, data=data, config=config)
    J_jax = jax.jacobian(f, argnums=0)(params, data, config)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_multi_argnums_with_kwargs(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Multiple argnums with some args passed as kwargs."""

    def f(x, y, z):
        return x["a"] * y["b"] + z["c"]

    x = {"a": jnp.array([1.0, 2.0])}
    y = {"b": jnp.array([3.0, 4.0])}
    z = {"c": jnp.array([0.1, 0.2])}

    J = asdex.jacobian(
        f,
        x,
        y=y,
        z=z,
        argnums=(0, 2),
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(x, y=y, z=z)
    J_jax = jax.jacobian(f, argnums=(0, 2))(x, y, z)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_keyword_only_pytree_args(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Keyword-only pytree args with defaults in function signature."""

    def f(params, *, data=None, config=None):
        if data is None:
            data = {"x": jnp.ones(3)}
        if config is None:
            config = {"scale": jnp.ones(3)}
        return params["w"] * data["x"] * config["scale"]

    params = {"w": jnp.array([1.0, 2.0, 3.0])}
    data = {"x": jnp.array([2.0, 3.0, 4.0])}
    config = {"scale": jnp.array([0.5, 0.5, 0.5])}

    J = asdex.jacobian(
        f,
        params,
        data=data,
        config=config,
        argnums=0,
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(params, data=data, config=config)
    J_jax = jax.jacobian(f, argnums=0)(params, data=data, config=config)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_mixed_positional_and_keyword_only_pytrees(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Mix of positional pytree args and keyword-only pytree args with defaults."""

    def f(params, data, *, scale=None, bias=None):
        if scale is None:
            scale = {"s": jnp.ones(2)}
        if bias is None:
            bias = {"b": jnp.zeros(2)}
        return params["w"] * data["x"] * scale["s"] + bias["b"]

    params = {"w": jnp.array([1.0, 2.0])}
    data = {"x": jnp.array([3.0, 4.0])}
    scale = {"s": jnp.array([0.5, 0.5])}
    bias = {"b": jnp.array([0.1, 0.2])}

    J = asdex.jacobian(
        f,
        params,
        data,
        scale=scale,
        bias=bias,
        argnums=0,
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(params, data, scale=scale, bias=bias)
    J_jax = jax.jacobian(f, argnums=0)(params, data, scale=scale, bias=bias)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_keyword_only_with_defaults(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Keyword-only args with default values."""

    def f(params, *, scale=None, bias=None):
        if scale is None:
            scale = {"s": jnp.ones(3)}
        if bias is None:
            bias = {"b": jnp.zeros(3)}
        return params["w"] * scale["s"] + bias["b"]

    params = {"w": jnp.array([1.0, 2.0, 3.0])}
    scale = {"s": jnp.array([2.0, 2.0, 2.0])}
    bias = {"b": jnp.array([0.5, 0.5, 0.5])}

    J = asdex.jacobian(
        f,
        params,
        scale=scale,
        bias=bias,
        argnums=0,
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(params, scale=scale, bias=bias)
    J_jax = jax.jacobian(f, argnums=0)(params, scale=scale, bias=bias)
    assert_trees_allclose(J, J_jax)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_pytree_args_as_kwargs(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Hessian with pytree positional args passed as kwargs."""

    def f(params, data):
        return jnp.sum(params["w"] ** 2 * data["x"])

    params = {"w": jnp.array([1.0, 2.0, 3.0])}
    data = {"x": jnp.array([1.0, 1.0, 1.0])}

    H = asdex.hessian(
        f,
        params,
        data=data,
        argnums=0,
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(params=params, data=data)
    H_jax = jax.hessian(f, argnums=0)(params, data)
    assert_trees_allclose(H, H_jax, atol=1e-6)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_mixed_args_kwargs_with_defaults(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Hessian with mixed positional/kwarg pytrees and default kwargs."""

    def f(params, data, scale=1.0):
        return scale * jnp.sum(params["w"] ** 2 * data["x"])

    params = {"w": jnp.array([1.0, 2.0])}
    data = {"x": jnp.array([2.0, 3.0])}

    H = asdex.hessian(
        f,
        params,
        data=data,
        scale=2.0,
        argnums=0,
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(params, data=data, scale=2.0)
    H_jax = jax.hessian(lambda p, d: f(p, d, scale=2.0), argnums=0)(params, data)
    assert_trees_allclose(H, H_jax, atol=1e-6)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_argnums_refers_to_signature_order(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Argnums indexes into signature order for Hessians, not call-site kwarg order."""

    def f(a, b, c):
        return jnp.sum(a**2) + jnp.sum(b**2) + jnp.sum(c**2)

    a = jnp.array([1.0, 2.0])
    b = jnp.array([3.0, 4.0])
    c = jnp.array([5.0, 6.0])

    # Pass kwargs in reverse order: c, a, b - but argnums=1 still means "b"
    H = asdex.hessian(
        f,
        c=c,
        a=a,
        b=b,
        argnums=1,
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(c=c, a=a, b=b)
    H_jax = jax.hessian(f, argnums=1)(a, b, c)
    assert_trees_allclose(H, H_jax, atol=1e-6)

    # Verify Hessian is 2*I (d^2/db^2 of sum(b^2) = 2)
    expected = 2.0 * jnp.eye(2)
    np.testing.assert_allclose(H if output_format == "dense" else H.todense(), expected)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_keyword_only_pytree_args(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Hessian with keyword-only pytree args with defaults in function signature."""

    def f(params, *, data=None):
        if data is None:
            data = {"x": jnp.ones(2)}
        return jnp.sum(params["w"] ** 2 * data["x"])

    params = {"w": jnp.array([1.0, 2.0])}
    data = {"x": jnp.array([2.0, 3.0])}

    H = asdex.hessian(
        f,
        params,
        data=data,
        argnums=0,
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(params, data=data)
    H_jax = jax.hessian(f, argnums=0)(params, data=data)
    assert_trees_allclose(H, H_jax, atol=1e-6)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_keyword_only_with_pytree_defaults(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Hessian with keyword-only args with pytree default values."""

    def f(params, *, scale=None):
        if scale is None:
            scale = {"s": jnp.ones(2)}
        return jnp.sum(params["w"] ** 2 * scale["s"])

    params = {"w": jnp.array([1.0, 2.0])}
    scale = {"s": jnp.array([2.0, 3.0])}

    H = asdex.hessian(
        f,
        params,
        scale=scale,
        argnums=0,
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(params, scale=scale)
    H_jax = jax.hessian(f, argnums=0)(params, scale=scale)
    assert_trees_allclose(H, H_jax, atol=1e-6)


@pytest.mark.jacobian
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_value_and_jacobian_pytree_kwargs(
    output_format, chunk_size, assert_trees_allclose
):
    """value_and_jacobian with pytree args passed as kwargs."""

    def f(params, data):
        return params["w"] * data["x"]

    params = {"w": jnp.array([1.0, 2.0, 3.0])}
    data = {"x": jnp.array([2.0, 3.0, 4.0])}

    val, J = asdex.value_and_jacobian(
        f,
        params,
        data=data,
        argnums=0,
        output_format=output_format,
        chunk_size=chunk_size,
    )(params=params, data=data)

    val_expected = f(params, data)
    J_jax = jax.jacobian(f, argnums=0)(params, data)

    np.testing.assert_allclose(val, val_expected)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_value_and_jacobian_keyword_only_pytrees(
    output_format, chunk_size, assert_trees_allclose
):
    """value_and_jacobian with keyword-only pytree args with defaults."""

    def f(params, *, data=None):
        if data is None:
            data = {"x": jnp.ones(3)}
        return params["w"] * data["x"]

    params = {"w": jnp.array([1.0, 2.0, 3.0])}
    data = {"x": jnp.array([2.0, 3.0, 4.0])}

    val, J = asdex.value_and_jacobian(
        f,
        params,
        data=data,
        argnums=0,
        output_format=output_format,
        chunk_size=chunk_size,
    )(params, data=data)

    val_expected = f(params, data=data)
    J_jax = jax.jacobian(f, argnums=0)(params, data=data)

    np.testing.assert_allclose(val, val_expected)
    assert_trees_allclose(J, J_jax)


@pytest.mark.hessian
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_value_and_hessian_pytree_kwargs(
    output_format, chunk_size, assert_trees_allclose
):
    """value_and_hessian with pytree args passed as kwargs."""

    def f(params, data):
        return jnp.sum(params["w"] ** 2 * data["x"])

    params = {"w": jnp.array([1.0, 2.0])}
    data = {"x": jnp.array([1.0, 2.0])}

    val, H = asdex.value_and_hessian(
        f,
        params,
        data=data,
        argnums=0,
        output_format=output_format,
        chunk_size=chunk_size,
    )(params=params, data=data)

    val_expected = f(params, data)
    H_jax = jax.hessian(f, argnums=0)(params, data)

    np.testing.assert_allclose(val, val_expected)
    assert_trees_allclose(H, H_jax, atol=1e-6)


@pytest.mark.hessian
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_value_and_hessian_keyword_only_pytrees(
    output_format, chunk_size, assert_trees_allclose
):
    """value_and_hessian with keyword-only pytree args with defaults."""

    def f(params, *, data=None):
        if data is None:
            data = {"x": jnp.ones(2)}
        return jnp.sum(params["w"] ** 2 * data["x"])

    params = {"w": jnp.array([1.0, 2.0])}
    data = {"x": jnp.array([1.0, 2.0])}

    val, H = asdex.value_and_hessian(
        f,
        params,
        data=data,
        argnums=0,
        output_format=output_format,
        chunk_size=chunk_size,
    )(params, data=data)

    val_expected = f(params, data=data)
    H_jax = jax.hessian(f, argnums=0)(params, data=data)

    np.testing.assert_allclose(val, val_expected)
    assert_trees_allclose(H, H_jax, atol=1e-6)


# Multi-pytree output with kwargs


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_pytree_output_with_positional_as_kwargs(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Jacobian of pytree-output function with positional args passed as kwargs."""

    def f(params, data):
        return {
            "y": params["w"] * data["x"],
            "z": params["w"] + data["x"],
        }

    params = {"w": jnp.array([1.0, 2.0, 3.0])}
    data = {"x": jnp.array([2.0, 3.0, 4.0])}

    J = asdex.jacobian(
        f,
        params,
        data=data,
        argnums=0,
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(params=params, data=data)
    J_jax = jax.jacobian(f, argnums=0)(params, data)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_pytree_output_with_keyword_only_args(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Jacobian of pytree-output function with keyword-only args."""

    def f(params, *, scale=None):
        if scale is None:
            scale = {"s": jnp.ones(3)}
        return {
            "y": params["w"] * scale["s"],
            "z": params["w"] ** 2,
        }

    params = {"w": jnp.array([1.0, 2.0, 3.0])}
    scale = {"s": jnp.array([0.5, 0.5, 0.5])}

    J = asdex.jacobian(
        f,
        params,
        scale=scale,
        argnums=0,
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(params, scale=scale)
    J_jax = jax.jacobian(f, argnums=0)(params, scale=scale)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_nested_pytree_output_with_kwargs(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Jacobian of nested pytree-output function with mixed kwargs."""

    def f(params, data, *, config=None):
        if config is None:
            config = {"scale": jnp.ones(2)}
        return {
            "outputs": {
                "y": params["w"] * data["x"] * config["scale"],
                "z": params["w"] + data["x"],
            },
            "norm": jnp.sum(params["w"] ** 2),
        }

    params = {"w": jnp.array([1.0, 2.0])}
    data = {"x": jnp.array([3.0, 4.0])}
    config = {"scale": jnp.array([0.5, 0.5])}

    J = asdex.jacobian(
        f,
        params,
        data,
        config=config,
        argnums=0,
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(params, data=data, config=config)
    J_jax = jax.jacobian(f, argnums=0)(params, data, config=config)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_value_and_jacobian_pytree_output_with_kwargs(
    output_format, chunk_size, assert_trees_allclose
):
    """value_and_jacobian with pytree output and kwargs."""

    def f(params, data):
        return {
            "product": params["w"] * data["x"],
            "sum": params["w"] + data["x"],
        }

    params = {"w": jnp.array([1.0, 2.0])}
    data = {"x": jnp.array([3.0, 4.0])}

    val, J = asdex.value_and_jacobian(
        f,
        params,
        data=data,
        argnums=0,
        output_format=output_format,
        chunk_size=chunk_size,
    )(params=params, data=data)

    val_expected = f(params, data)
    J_jax = jax.jacobian(f, argnums=0)(params, data)

    assert_trees_allclose(val, val_expected)
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


# VAR_KEYWORD and VAR_POSITIONAL signatures


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_function_with_var_keyword(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Functions with **kwargs in signature should work correctly.

    Regression test for Copilot review: VAR_KEYWORD handling in merge_args_kwargs.
    """

    def f(x, **kw):
        scale = kw.get("scale", 1.0)
        offset = kw.get("offset", 0.0)
        return x * scale + offset

    x = jnp.array([1.0, 2.0, 3.0])
    J = asdex.jacobian(
        f, x, mode=mode, output_format=output_format, chunk_size=chunk_size
    )(x, scale=2.0, offset=1.0)
    J_jax = jax.jacobian(f)(x, scale=2.0, offset=1.0)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_function_with_var_keyword_at_detection(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Functions with **kwargs should work when kwargs passed at detection time.

    Regression test for Copilot review: VAR_KEYWORD handling in _merge_sample_inputs.
    """

    def f(x, **kw):
        scale = kw.get("scale", 1.0)
        return x * scale

    x = jnp.array([1.0, 2.0, 3.0])
    # Pass scale at detection time to ensure correct sparsity
    J = asdex.jacobian(
        f, x, scale=2.0, mode=mode, output_format=output_format, chunk_size=chunk_size
    )(x, scale=2.0)
    J_jax = jax.jacobian(f)(x, scale=2.0)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_function_with_var_positional(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Functions with *args in signature should work correctly.

    Regression test for Copilot review: VAR_POSITIONAL handling.
    """

    def f(x, *extra):
        if extra:
            return x * extra[0]
        return x

    x = jnp.array([1.0, 2.0, 3.0])
    scale = jnp.array([2.0, 2.0, 2.0])
    J = asdex.jacobian(
        f,
        x,
        scale,
        argnums=0,
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(x, scale)
    J_jax = jax.jacobian(f, argnums=0)(x, scale)
    assert_trees_allclose(J, J_jax)


# Non-traceable kwargs (bools, strings, ints)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_bool_kwarg_at_detection(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Bool kwargs at detection time should not be traced.

    Regression test: bool kwargs were incorrectly passed to make_jaxpr,
    causing TracerBoolConversionError when used in Python if-statements.
    """

    def f(x, flag=True):
        if flag:
            return x * 2
        return x * 3

    x = jnp.array([1.0, 2.0, 3.0])

    # Bool kwarg at detection time - should work without TracerBoolConversionError
    J = asdex.jacobian(
        f, x, flag=True, mode=mode, output_format=output_format, chunk_size=chunk_size
    )(x, flag=True)
    J_jax = jax.jacobian(f)(x, flag=True)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
def test_jacobian_kwarg_changes_output_structure_between_calls(assert_trees_allclose):
    """Call-time kwargs that change the output pytree bypass the out-struct cache.

    Regression test for the per-closure eval_shape cache:
    a call without kwargs fills the cache,
    and a later call with a structure-changing kwarg must not reuse it.
    """

    def f(x, split=False):
        if split:
            return (x[:1] * 2.0, x[1:] * 2.0)
        return x * 2.0

    x = jnp.array([1.0, 2.0, 3.0])
    jac_fn = asdex.jacobian(f, x, output_format="dense")

    J_flat = jac_fn(x)
    assert_trees_allclose(J_flat, jax.jacobian(f)(x))

    J_split = jac_fn(x, split=True)
    assert_trees_allclose(J_split, jax.jacobian(f)(x, split=True))


@pytest.mark.jacobian
def test_jacobian_sparsity_bool_kwarg():
    """jacobian_sparsity should handle bool kwargs that control Python branches.

    Regression test: bool kwargs were incorrectly traced by make_jaxpr.
    """

    def f(x, use_first_half=True):
        if use_first_half:
            return x[:2]
        return x[2:]

    x = jnp.array([1.0, 2.0, 3.0, 4.0])

    # Sparsity should differ based on flag value
    pattern_true = asdex.jacobian_sparsity(f, x, use_first_half=True)
    pattern_false = asdex.jacobian_sparsity(f, x, use_first_half=False)

    # flag=True: output depends on x[0], x[1]
    expected_true = np.array([[1, 0, 0, 0], [0, 1, 0, 0]])
    np.testing.assert_array_equal(pattern_true.todense(), expected_true)

    # flag=False: output depends on x[2], x[3]
    expected_false = np.array([[0, 0, 1, 0], [0, 0, 0, 1]])
    np.testing.assert_array_equal(pattern_false.todense(), expected_false)


@pytest.mark.hessian
def test_hessian_sparsity_bool_kwarg():
    """hessian_sparsity should handle bool kwargs that control Python branches."""

    def f(x, use_quadratic=True):
        if use_quadratic:
            return jnp.sum(x**2)
        return jnp.sum(x)

    x = jnp.array([1.0, 2.0, 3.0])

    # Sparsity should differ based on flag value
    pattern_quad = asdex.hessian_sparsity(f, x, use_quadratic=True)
    pattern_linear = asdex.hessian_sparsity(f, x, use_quadratic=False)

    # Quadratic: Hessian is 2*I (diagonal)
    expected_quad = np.eye(3, dtype=int)
    np.testing.assert_array_equal(pattern_quad.todense(), expected_quad)

    # Linear: Hessian is zero (no second derivatives)
    expected_linear = np.zeros((3, 3), dtype=int)
    np.testing.assert_array_equal(pattern_linear.todense(), expected_linear)


@pytest.mark.jacobian
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_function_param_named_mode(
    output_format, chunk_size, assert_trees_allclose
):
    """Function with param named 'mode' should work (name collides with API option).

    Regression test for Copilot review: name collisions with API options.
    """

    def f(x, mode="multiply"):
        if mode == "multiply":
            return x * 2
        return x + 2

    x = jnp.array([1.0, 2.0, 3.0])
    # The API's mode="rev" should not collide with the function's mode param
    J = asdex.jacobian(
        f, x, mode="rev", output_format=output_format, chunk_size=chunk_size
    )(x, mode="multiply")
    J_jax = jax.jacobian(f)(x, mode="multiply")
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_function_param_named_argnums(
    output_format, chunk_size, assert_trees_allclose
):
    """Function with param named 'argnums' should work (name collides with API option)."""

    def f(x, argnums=0):
        return x * (argnums + 1)

    x = jnp.array([1.0, 2.0, 3.0])
    # The function's argnums param (value 2) should not collide with API's argnums
    J = asdex.jacobian(
        f, x, argnums=0, output_format=output_format, chunk_size=chunk_size
    )(x, argnums=2)
    J_jax = jax.jacobian(f)(x, argnums=2)
    assert_trees_allclose(J, J_jax)


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


# Adversarial edge cases


@pytest.mark.jacobian
@pytest.mark.bug
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_different_bool_kwarg_at_call_time(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Bug: Different bool kwarg at call time uses wrong sparsity pattern.

    The sparsity pattern is computed at detection time with flag=True,
    but we call with flag=False. The decompressed Jacobian is WRONG because
    it uses the detection-time sparsity pattern (x[0:2]) but the actual
    computation at call time uses x[2:4].

    This is a known limitation of sparse autodiff: the sparsity pattern is
    fixed at detection time. Users must ensure bool kwargs don't change
    the structural computation path between detection and call time.
    """

    def f(x, flag=True):
        if flag:
            return x[:2]
        return x[2:]

    x = jnp.array([1.0, 2.0, 3.0, 4.0])

    # Detect with flag=True (sparsity says output depends on x[0:2])
    # but call with flag=False (actual output depends on x[2:4])
    jac_fn = asdex.jacobian(
        f,
        x,
        flag=True,
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )
    J = jac_fn(x, flag=False)

    # The actual Jacobian for flag=False - WRONG RESULT expected
    J_jax = jax.jacobian(f)(x, flag=False)

    # BUG: asdex returns wrong Jacobian because sparsity pattern was fixed at detection
    # The actual Jacobian should be [[0,0,1,0], [0,0,0,1]] (depends on x[2:4])
    # but asdex uses the pattern for flag=True which has nonzeros at x[0:2]
    with pytest.raises(AssertionError):
        assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_nested_pytree_kwarg_with_non_traceable_leaves(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Nested pytree kwargs with mixed array/non-traceable leaves are bound statically.

    When a pytree contains both array leaves and non-traceable leaves (bools, ints),
    the entire pytree is bound statically to avoid TracerBoolConversionError.
    """

    def f(x, config=None):
        if config is None:
            config = {"scale": 1.0, "options": {"use_bias": True, "n_repeats": 1}}
        result = x * config["scale"]
        if config["options"]["use_bias"]:  # ty: ignore[not-subscriptable]
            result = result + 0.5
        for _ in range(config["options"]["n_repeats"]):  # ty: ignore[not-subscriptable]
            result = result * 1.1
        return result

    x = jnp.array([1.0, 2.0])
    config = {"scale": jnp.array(2.0), "options": {"use_bias": True, "n_repeats": 2}}

    J = asdex.jacobian(
        f,
        x,
        config=config,
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(x, config=config)
    J_jax = jax.jacobian(lambda x: f(x, config=config))(x)
    assert_trees_allclose(J, J_jax)


@pytest.mark.hessian
@pytest.mark.bug
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_different_bool_kwarg_at_call_time(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Bug: Hessian with different bool kwarg at call vs detection time."""

    def f(x, coupled=False):
        if coupled:
            # Coupled: Hessian has off-diagonal terms
            return (x[0] * x[1]) ** 2
        # Uncoupled: Hessian is diagonal
        return jnp.sum(x**2)

    x = jnp.array([1.0, 2.0])

    # Detect with coupled=False (diagonal Hessian)
    # but call with coupled=True (dense Hessian with off-diagonals)
    hess_fn = asdex.hessian(
        f,
        x,
        coupled=False,
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )
    H = hess_fn(x, coupled=True)

    H_jax = jax.hessian(f)(x, coupled=True)

    # BUG: asdex returns wrong Hessian because sparsity was fixed at detection
    # The diagonal pattern from detection misses the off-diagonal entries
    with pytest.raises(AssertionError):
        assert_trees_allclose(H, H_jax, atol=1e-5)


# Copilot review: Silent truncation of extra positional args


@pytest.mark.jacobian
@pytest.mark.bug
def test_var_positional_extra_args_at_call_time():
    """Bug: Extra *args passed at call time but not detection time raises.

    Copilot concern: merge_args_kwargs() silently truncates extra positional
    arguments via positional_args[:expected_nargs], potentially dropping
    user-supplied call-time positional arguments without raising.

    Finding: Does NOT silently truncate - raises ValueError. But still doesn't
    work for the use case where you want to pass extra args only at call time.
    The sparsity pattern is fixed at detection time, so you must provide all
    args at detection time.
    """

    def f(x, *extra):
        if extra:
            return x * extra[0]
        return x * 2

    x = jnp.array([1.0, 2.0, 3.0])
    scale = jnp.array([3.0, 3.0, 3.0])

    # Detection with just x (no extra args), call with extra scale
    # Raises because the number of args doesn't match
    with pytest.raises(ValueError, match="Expected 1 positional argument"):
        asdex.jacobian(f, x, argnums=0)(x, scale)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_var_positional_multiple_extra_args(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Multiple extra *args should all be passed through.

    Copilot concern: truncation could drop some but not all extra args.
    """

    def f(x, *extra):
        result = x.copy()
        for e in extra:
            result = result + e
        return result

    x = jnp.array([1.0, 2.0])
    y = jnp.array([0.5, 0.5])
    z = jnp.array([0.1, 0.1])

    # Call with extra y and z
    J = asdex.jacobian(
        f,
        x,
        y,
        z,
        argnums=0,
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(x, y, z)
    J_jax = jax.jacobian(f, argnums=0)(x, y, z)
    assert_trees_allclose(J, J_jax)


# Copilot review: VAR_POSITIONAL non-traceable elements dropped


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_var_positional_with_non_traceable_args_at_detection(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """*args containing non-traceable values (bools) at detection time.

    Copilot concern: non-traceable elements in VAR_POSITIONAL are dropped,
    changing the function being analyzed and yielding incorrect sparsity.

    Finding: Fixed - bools in positional args are now bound statically,
    preserving their original positions.
    """

    def f(x, *extra):
        # extra[0] is a flag, extra[1] is a scale
        if extra and extra[0]:
            return x * extra[1]
        return x

    x = jnp.array([1.0, 2.0])
    flag = True
    scale = jnp.array([2.0, 2.0])

    J = asdex.jacobian(
        f,
        x,
        flag,
        scale,
        argnums=0,
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(x, flag, scale)
    J_jax = jax.jacobian(f, argnums=0)(x, flag, scale)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_var_positional_mixed_traceable_nontraceable(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """*args with interleaved traceable and non-traceable values.

    Copilot concern: non-traceable elements dropped changes function.

    Finding: Fixed - bools in positional args are now bound statically.
    """

    def f(x, *extra):
        # extra = (int, array, bool, array)
        n = extra[0] if len(extra) > 0 else 1
        y = extra[1] if len(extra) > 1 else jnp.ones_like(x)
        flag = extra[2] if len(extra) > 2 else False
        z = extra[3] if len(extra) > 3 else jnp.ones_like(x)
        result = x * y
        if flag:
            result = result + z
        return result[:n]

    x = jnp.array([1.0, 2.0, 3.0])
    n = 2
    y = jnp.array([2.0, 2.0, 2.0])
    flag = True
    z = jnp.array([0.5, 0.5, 0.5])

    J = asdex.jacobian(
        f,
        x,
        n,
        y,
        flag,
        z,
        argnums=0,
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(x, n, y, flag, z)
    J_jax = jax.jacobian(f, argnums=0)(x, n, y, flag, z)
    assert_trees_allclose(J, J_jax)


# Copilot review: Python scalar numerics treated as non-traceable


@pytest.mark.jacobian
def test_python_float_scalar_kwarg_at_detection():
    """Python float scalar as kwarg should be traced, not bound statically.

    Copilot concern: _is_jax_traceable() treats Python/NumPy scalars as
    non-traceable because they lack shape/dtype, so they get bound statically.
    This contradicts the docstring saying only bool/int/str/None are non-traceable.
    """

    def f(x, scale=1.0):
        return x * scale

    x = jnp.array([1.0, 2.0, 3.0])

    # Detection with scale=2.0 (Python float)
    pattern = asdex.jacobian_sparsity(f, x, scale=2.0)
    # Expected: diagonal (x affects output element-wise)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(pattern.todense(), expected)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_python_float_scalar_changes_jacobian_value(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Python float kwarg value should affect the computed Jacobian.

    If scalars are incorrectly bound statically at detection time, then
    changing the scale at call time would have no effect. This test verifies
    that call-time scale values are actually used.
    """

    def f(x, scale=1.0):
        return x * scale

    x = jnp.array([1.0, 2.0, 3.0])

    # Detection with scale=1.0, call with scale=3.0
    J = asdex.jacobian(
        f, x, scale=1.0, mode=mode, output_format=output_format, chunk_size=chunk_size
    )(x, scale=3.0)
    J_jax = jax.jacobian(f)(x, scale=3.0)
    assert_trees_allclose(J, J_jax)

    # Should be 3 * I, not 1 * I
    expected = 3.0 * jnp.eye(3)
    np.testing.assert_allclose(J if output_format == "dense" else J.todense(), expected)


@pytest.mark.jacobian
def test_numpy_scalar_treated_as_traceable():
    """NumPy scalar (np.float64) should be traceable.

    Copilot concern: np.float64 lacks .shape/.dtype attributes in the way
    _is_jax_traceable checks, potentially making it non-traceable.
    """

    def f(x, scale):
        return x * scale

    x = jnp.array([1.0, 2.0])
    scale = np.float64(2.0)

    # If scale is traceable, it would be in argnums and affect sparsity
    # We pass scale as positional, but only differentiate w.r.t. x
    J = asdex.jacobian(f, x, scale, argnums=0, output_format="dense")(x, scale)
    J_jax = jax.jacobian(f, argnums=0)(x, scale)
    np.testing.assert_allclose(J, J_jax)


@pytest.mark.jacobian
def test_zero_dim_array_is_traceable():
    """0-D JAX arrays should be treated as traceable.

    Copilot concern: 0-D arrays might be incorrectly treated as scalars.
    """

    def f(x, scale):
        return x * scale

    x = jnp.array([1.0, 2.0])
    scale = jnp.array(2.0)  # 0-D array

    # Differentiate w.r.t. both x and scale
    Jx, Jscale = asdex.jacobian(f, x, scale, argnums=(0, 1), output_format="dense")(
        x, scale
    )
    Jx_jax, Jscale_jax = jax.jacobian(f, argnums=(0, 1))(x, scale)
    np.testing.assert_allclose(Jx, Jx_jax)
    np.testing.assert_allclose(Jscale, Jscale_jax)


# Copilot review (low confidence): non-prefix traced args


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_non_traceable_positional_before_traceable(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Non-traceable arg before a traceable arg in signature.

    Copilot suppressed concern: merge_args_kwargs() assumes traced positional
    arguments form a prefix. But merge_sample_inputs() can drop non-traceable
    POSITIONAL_OR_KEYWORD params that appear before later traceable params.

    Finding: Fixed - bools in positional args are now bound statically,
    preserving their original positions like JAX's argnums_partial.
    """

    def f(flag, x, scale):
        if flag:
            return x * scale
        return x + scale

    flag = True
    x = jnp.array([1.0, 2.0])
    scale = jnp.array([2.0, 2.0])

    # flag is non-traceable (bool), x and scale are traceable
    # argnums=1 means we differentiate w.r.t. x (the second positional param)
    J = asdex.jacobian(
        f,
        flag,
        x,
        scale,
        argnums=1,
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(flag, x, scale)
    J_jax = jax.jacobian(f, argnums=1)(flag, x, scale)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_interleaved_traceable_nontraceable_positional(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Interleaved traceable and non-traceable positional args.

    The traced args are at positions 1 and 3 (non-prefix subset of signature).

    Finding: Fixed - bools passed positionally are now bound statically,
    preserving their original positions.
    """

    def f(flag1, x, flag2, y):
        result = x * 2 if flag1 else x
        return result + y if flag2 else result - y

    flag1 = True
    x = jnp.array([1.0, 2.0])
    flag2 = False
    y = jnp.array([0.5, 0.5])

    J = asdex.jacobian(
        f,
        flag1,
        x,
        flag2,
        y,
        argnums=1,
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(flag1, x, flag2, y)
    J_jax = jax.jacobian(f, argnums=1)(flag1, x, flag2, y)
    assert_trees_allclose(J, J_jax)
