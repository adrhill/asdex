"""Tests for API variants: value_and_*, *_from_coloring.

Verifies all public API entry points produce consistent results matching JAX.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import asdex

warnings.filterwarnings("ignore", category=asdex.DenseColoringWarning)


# value_and_jacobian


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_value_and_jacobian_basic(mode, output_format, assert_trees_allclose):
    """value_and_jacobian returns matching primal and Jacobian."""

    def f(x):
        return jnp.array([x[0] ** 2, x[1] ** 2])

    x = jnp.array([2.0, 3.0])
    val, J = asdex.value_and_jacobian(f, x, mode=mode, output_format=output_format)(x)
    J_jax = jax.jacobian(f)(x)
    np.testing.assert_allclose(val, f(x))
    assert_trees_allclose(J, J_jax)


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
    if output_format == "bcoo":
        pytest.xfail("BCOO Jacobians lose PyTree structure for nested dict inputs")

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
def test_value_and_jacobian_multi_input(mode, output_format, assert_trees_allclose):
    """value_and_jacobian with multiple args matches JAX."""

    def f(x, y):
        return jnp.array([x[0] * y[0], x[1] + y[1], x[0] * x[1]])

    x, y = jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])
    val, J = asdex.value_and_jacobian(
        f, x, y, argnums=(0, 1), mode=mode, output_format=output_format
    )(x, y)
    J_jax = jax.jacobian(f, argnums=(0, 1))(x, y)
    np.testing.assert_allclose(val, f(x, y))
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


# value_and_hessian


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_value_and_hessian_basic(mode, output_format, assert_trees_allclose):
    """value_and_hessian returns matching primal and Hessian."""

    def f(x):
        return x[0] ** 2 + x[1] ** 2

    x = jnp.array([2.0, 3.0])
    val, H = asdex.value_and_hessian(f, x, mode=mode, output_format=output_format)(x)
    H_jax = jax.hessian(f)(x)
    np.testing.assert_allclose(val, f(x))
    assert_trees_allclose(H, H_jax, atol=1e-6)


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


# jacobian_from_coloring


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_from_coloring_basic(mode, output_format, assert_trees_allclose):
    """jacobian_from_coloring matches JAX."""

    def f(x):
        return jnp.array([x[0] * x[1], x[0] + x[1]])

    x = jnp.array([2.0, 3.0])
    coloring = asdex.jacobian_coloring(f, x, mode=mode)
    J = asdex.jacobian_from_coloring(f, coloring, output_format=output_format)(x)
    J_jax = jax.jacobian(f)(x)
    assert_trees_allclose(J, J_jax)


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


# value_and_jacobian_from_coloring


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_value_and_jacobian_from_coloring_basic(
    mode, output_format, assert_trees_allclose
):
    """value_and_jacobian_from_coloring matches JAX."""

    def f(x):
        return jnp.array([x[0] ** 2, x[1] ** 2])

    x = jnp.array([2.0, 3.0])
    coloring = asdex.jacobian_coloring(f, x, mode=mode)
    val, J = asdex.value_and_jacobian_from_coloring(
        f, coloring, output_format=output_format
    )(x)
    J_jax = jax.jacobian(f)(x)
    np.testing.assert_allclose(val, f(x))
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


# hessian_from_coloring


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_from_coloring_basic(mode, output_format, assert_trees_allclose):
    """hessian_from_coloring matches JAX."""

    def f(x):
        return jnp.sum(x**2)

    x = jnp.array([1.0, 2.0, 3.0])
    coloring = asdex.hessian_coloring(f, x, mode=mode)
    H = asdex.hessian_from_coloring(f, coloring, output_format=output_format)(x)
    H_jax = jax.hessian(f)(x)
    assert_trees_allclose(H, H_jax, atol=1e-6)


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


# value_and_hessian_from_coloring


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_value_and_hessian_from_coloring_basic(
    mode, output_format, assert_trees_allclose
):
    """value_and_hessian_from_coloring matches JAX."""

    def f(x):
        return jnp.sum(x**2)

    x = jnp.array([1.0, 2.0, 3.0])
    coloring = asdex.hessian_coloring(f, x, mode=mode)
    val, H = asdex.value_and_hessian_from_coloring(
        f, coloring, output_format=output_format
    )(x)
    H_jax = jax.hessian(f)(x)
    np.testing.assert_allclose(val, f(x))
    assert_trees_allclose(H, H_jax, atol=1e-6)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_value_and_hessian_from_coloring_dict(
    mode, output_format, assert_trees_allclose
):
    """value_and_hessian_from_coloring with dict input matches JAX."""
    if output_format == "bcoo":
        pytest.xfail("BCOO Hessians lose PyTree structure for single-leaf dict")

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


# Coloring reuse: precomputed coloring applied to changing non-diff args


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_coloring_reuse_changing_nondiff_arg(
    mode, output_format, assert_trees_allclose
):
    """Coloring built once reuses correctly when non-diff arg changes."""

    def loss(params, x_batch, y_batch):
        return (params * x_batch - y_batch) ** 2

    params = jnp.array([1.0, 2.0, 3.0])
    x_batch = jnp.array([0.5, 1.0, 1.5])
    y_batch = jnp.array([1.0, 2.0, 3.0])

    coloring = asdex.jacobian_coloring(
        loss, params, x_batch, y_batch, argnums=0, mode=mode
    )
    jac = asdex.jacobian_from_coloring(loss, coloring, output_format=output_format)

    for xb, yb in [
        (jnp.array([0.5, 1.0, 1.5]), jnp.array([1.0, 2.0, 3.0])),
        (jnp.array([2.0, 0.1, 0.7]), jnp.array([0.0, 1.0, 4.0])),
        (jnp.array([1.0, 1.0, 1.0]), jnp.array([2.0, 2.0, 2.0])),
    ]:
        J = jac(params, xb, yb)
        J_jax = jax.jacobian(loss, argnums=0)(params, xb, yb)
        assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_coloring_reuse_pytree_nondiff_arg(
    mode, output_format, assert_trees_allclose
):
    """Coloring built once reuses correctly when non-diff PyTree arg changes."""
    if output_format == "bcoo":
        pytest.xfail("BCOO Jacobians lose PyTree structure for single-leaf dict")

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
    if output_format == "bcoo":
        pytest.xfail("BCOO Hessians lose PyTree structure for single-leaf dict")

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


# Combined coloring tests


@pytest.mark.coloring
def test_combined_coloring_reverse_mode_disjoint_rows():
    """f(x, y) = x * y on n=3: reverse-mode coloring of [J_x | J_y] needs 1 color.

    Each output row ``i`` has nonzeros only at columns ``i`` (from x) and ``n + i``
    (from y). No two distinct rows share any column, so one color suffices.
    """

    def f(x, y):
        return x * y

    x, y = jnp.ones(3), jnp.ones(3)
    c = asdex.jacobian_coloring(f, x, y, argnums=(0, 1), mode="rev")
    assert c.num_colors == 1


@pytest.mark.coloring
def test_combined_coloring_forward_mode_couples_inputs():
    """f(x, y) = x * y on n=3: forward-mode coloring of [J_x | J_y] needs 2 colors.

    Columns ``x[i]`` and ``y[i]`` both write to output row ``i``, so they conflict.
    """

    def f(x, y):
        return x * y

    x, y = jnp.ones(3), jnp.ones(3)
    c = asdex.jacobian_coloring(f, x, y, argnums=(0, 1), mode="fwd")
    assert c.num_colors == 2


# Coloring reuse tests


@pytest.mark.jacobian
def test_jacobian_argnums_amortizes_across_changing_non_diff_args():
    """Coloring built once is reused as non-differentiated args change every call."""

    def loss(params, x_batch, y_batch):
        return (params * x_batch - y_batch) ** 2

    params = jnp.array([1.0, 2.0, 3.0])
    x_batch = jnp.array([0.5, 1.0, 1.5])
    y_batch = jnp.array([1.0, 2.0, 3.0])

    coloring = asdex.jacobian_coloring(loss, params, x_batch, y_batch, argnums=0)
    jac = asdex.jacobian_from_coloring(loss, coloring, output_format="dense")

    for xb, yb in [
        (jnp.array([0.5, 1.0, 1.5]), jnp.array([1.0, 2.0, 3.0])),
        (jnp.array([2.0, 0.1, 0.7]), jnp.array([0.0, 1.0, 4.0])),
        (jnp.array([1.0, 1.0, 1.0]), jnp.array([2.0, 2.0, 2.0])),
    ]:
        J = jac(params, xb, yb)
        J_jax = jax.jacobian(loss, argnums=0)(params, xb, yb)
        assert jax.tree.structure(J) == jax.tree.structure(J_jax)
        np.testing.assert_allclose(J, J_jax)
