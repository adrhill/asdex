"""Tests for multi-input Jacobian and Hessian detection + decompression.

Covers functions with multiple positional arguments and argnums selection.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import asdex

warnings.filterwarnings("ignore", category=asdex.DenseColoringWarning)


# Multi-input Jacobians


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_wrt_x_y_and_both_agree(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """J_x and J_y from the multi-input call match jax.jacobian."""

    def f(x, y):
        return jnp.array([x[0] * y[0], x[1] + y[1], x[0] * x[1]])

    x, y = jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])
    J = asdex.jacobian(
        f,
        x,
        y,
        argnums=(0, 1),
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(x, y)
    J_jax = jax.jacobian(f, argnums=(0, 1))(x, y)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_asymmetric_block_shapes(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Differently-sized inputs produce non-transposed blocks of correct shape."""

    def f(x, y):
        return jnp.array([x[0] + y[2], x[2] * y[1]])

    x, y = jnp.ones(3), jnp.ones(4)
    J = asdex.jacobian(
        f,
        x,
        y,
        argnums=(0, 1),
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(x, y)
    J_jax = jax.jacobian(f, argnums=(0, 1))(x, y)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_three_inputs_ordering(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """With three inputs, each block goes to the right place."""

    def f(x, y, z):
        return x * y + z

    x, y, z = jnp.full(3, 2.0), jnp.full(3, 3.0), jnp.ones(3)
    J = asdex.jacobian(
        f,
        x,
        y,
        z,
        argnums=(0, 1, 2),
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(x, y, z)
    J_jax = jax.jacobian(f, argnums=(0, 1, 2))(x, y, z)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_dict_input_preserves_pytree(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Dict input returns dict-of-Jacobians matching jax.jacobian structure."""

    def f(params):
        return params["w"] @ params["x"] + params["b"]

    inputs = {
        "w": jnp.eye(3, 2),
        "x": jnp.array([1.0, 2.0]),
        "b": jnp.zeros(3),
    }

    J = asdex.jacobian(
        f, inputs, mode=mode, output_format=output_format, chunk_size=chunk_size
    )(inputs)
    J_jax = jax.jacobian(f)(inputs)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_two_dict_args_preserves_per_arg_pytree(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Top-level tuple of dicts matches jax.jacobian structure."""

    def f(p, q):
        return p["a"] * q["b"][:3] + q["c"]

    p = {"a": jnp.array([1.0, 2.0, 3.0])}
    q = {"b": jnp.array([4.0, 5.0, 6.0, 7.0]), "c": jnp.array([8.0, 9.0, 10.0])}

    J = asdex.jacobian(
        f,
        p,
        q,
        argnums=(0, 1),
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(p, q)
    J_jax = jax.jacobian(f, argnums=(0, 1))(p, q)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_multi_input_pytree_output(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Multi-input Jacobian with PyTree output matches jax.jacobian structure."""

    def f(x, y):
        return {"a": x * y, "b": x + y}

    x, y = jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])
    J = asdex.jacobian(
        f,
        x,
        y,
        argnums=(0, 1),
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(x, y)
    J_jax = jax.jacobian(f, argnums=(0, 1))(x, y)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_pytree_input_pytree_output(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Single pytree input with pytree output matches jax.jacobian structure."""

    def f(params):
        return {"sum": params["a"] + params["b"], "prod": params["a"] * params["b"]}

    params = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0, 4.0])}
    J = asdex.jacobian(
        f, params, mode=mode, output_format=output_format, chunk_size=chunk_size
    )(params)
    J_jax = jax.jacobian(f)(params)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_mixed_pytree_array_inputs_pytree_output(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Mixed PyTree and array inputs with PyTree output matches jax.jacobian."""

    def f(params, scale):
        return {"scaled": params["a"] * scale, "sum": params["a"] + params["b"]}

    params = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0, 4.0])}
    scale = jnp.array([2.0, 3.0])
    J = asdex.jacobian(
        f,
        params,
        scale,
        argnums=(0, 1),
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(params, scale)
    J_jax = jax.jacobian(f, argnums=(0, 1))(params, scale)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_three_pytree_inputs_pytree_output(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Three PyTree inputs with PyTree output matches jax.jacobian."""

    def f(p, q, r):
        return {"pq": p["x"] * q["y"], "qr": q["y"] + r["z"]}

    p = {"x": jnp.array([1.0, 2.0])}
    q = {"y": jnp.array([3.0, 4.0])}
    r = {"z": jnp.array([5.0, 6.0])}
    J = asdex.jacobian(
        f,
        p,
        q,
        r,
        argnums=(0, 1, 2),
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(p, q, r)
    J_jax = jax.jacobian(f, argnums=(0, 1, 2))(p, q, r)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_two_nested_dicts(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Two nested dict args match jax.jacobian."""

    def f(p, q):
        return p["layer"]["w"] * q["layer"]["w"]

    p = {"layer": {"w": jnp.array([1.0, 2.0])}}
    q = {"layer": {"w": jnp.array([3.0, 4.0])}}
    J = asdex.jacobian(
        f,
        p,
        q,
        argnums=(0, 1),
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(p, q)
    J_jax = jax.jacobian(f, argnums=(0, 1))(p, q)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_asymmetric_nested_pytrees(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Two PyTree args with different structures match jax.jacobian."""

    def f(model, data):
        return model["w"] @ data["x"] + model["b"]

    model = {"w": jnp.eye(2, 3), "b": jnp.zeros(2)}
    data = {"x": jnp.array([1.0, 2.0, 3.0])}
    J = asdex.jacobian(
        f,
        model,
        data,
        argnums=(0, 1),
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(model, data)
    J_jax = jax.jacobian(f, argnums=(0, 1))(model, data)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_complex_multi_input_multi_output(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Complex multi-input with complex multi-output matches jax.jacobian."""

    def f(model, data):
        y = model["W"] @ data["x"] + model["b"]
        return {"predictions": y, "loss": jnp.sum(y**2)}

    model = {"W": jnp.eye(2, 3), "b": jnp.zeros(2)}
    data = {"x": jnp.array([1.0, 2.0, 3.0])}
    J = asdex.jacobian(
        f,
        model,
        data,
        argnums=(0, 1),
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(model, data)
    J_jax = jax.jacobian(f, argnums=(0, 1))(model, data)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_nested_and_flat_args(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """One nested dict arg and one flat dict arg match jax.jacobian."""

    def f(nested, flat):
        return nested["layer"]["w"] * flat["scale"]

    nested = {"layer": {"w": jnp.array([1.0, 2.0])}}
    flat = {"scale": jnp.array([3.0, 4.0])}
    J = asdex.jacobian(
        f,
        nested,
        flat,
        argnums=(0, 1),
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(nested, flat)
    J_jax = jax.jacobian(f, argnums=(0, 1))(nested, flat)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_array_and_nested_dict(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """One array arg and one nested dict arg match jax.jacobian."""

    def f(scale, params):
        return scale * params["layer"]["w"]

    scale = jnp.array([1.0, 2.0])
    params = {"layer": {"w": jnp.array([3.0, 4.0])}}
    J = asdex.jacobian(
        f,
        scale,
        params,
        argnums=(0, 1),
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(scale, params)
    J_jax = jax.jacobian(f, argnums=(0, 1))(scale, params)
    assert_trees_allclose(J, J_jax)


# Multi-input Hessians


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_diagonal_blocks_match_jax(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """H_xx and H_yy from multi-input call match jax.hessian."""

    def f(x, y):
        return jnp.sum(x**3) + jnp.dot(x[:2], y) + jnp.sum(y**2)

    x, y = jnp.array([1.0, 2.0, 3.0]), jnp.array([4.0, 5.0])
    H = asdex.hessian(
        f,
        x,
        y,
        argnums=(0, 1),
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(x, y)
    H_jax = jax.hessian(f, argnums=(0, 1))(x, y)
    assert_trees_allclose(H, H_jax, atol=1e-6)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_separable_has_zero_cross_blocks(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """f(x, y) = sum(x^2) + sum(y^2) has structurally empty H_xy / H_yx."""

    def f(x, y):
        return jnp.sum(x**2) + jnp.sum(y**2)

    x, y = jnp.ones(3), jnp.ones(2)
    H = asdex.hessian(
        f,
        x,
        y,
        argnums=(0, 1),
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(x, y)
    H_jax = jax.hessian(f, argnums=(0, 1))(x, y)
    assert_trees_allclose(H, H_jax, atol=1e-6)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_bilinear_has_dense_cross_blocks(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """f(x, y) = sum(x) * sum(y) has empty diagonals, dense cross blocks."""

    def f(x, y):
        return jnp.sum(x) * jnp.sum(y)

    x, y = jnp.ones(3), jnp.ones(2)
    H = asdex.hessian(
        f,
        x,
        y,
        argnums=(0, 1),
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(x, y)
    H_jax = jax.hessian(f, argnums=(0, 1))(x, y)
    assert_trees_allclose(H, H_jax, atol=1e-6)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_asymmetric_block_shapes(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Differently-sized inputs produce four blocks matching jax.hessian."""

    def f(x, y):
        return jnp.sum(x**2) + jnp.dot(x, y[:3]) + jnp.sum(y**3)

    x, y = jnp.ones(3), jnp.ones(4)
    H = asdex.hessian(
        f,
        x,
        y,
        argnums=(0, 1),
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(x, y)
    H_jax = jax.hessian(f, argnums=(0, 1))(x, y)
    assert_trees_allclose(H, H_jax, atol=1e-6)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_three_inputs_block_grid(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Three inputs: 3x3 block grid matches jax.hessian."""

    def f(x, y, z):
        return jnp.dot(x, y) + jnp.dot(y, z)

    x, y, z = jnp.ones(3), jnp.ones(3), jnp.ones(3)
    H = asdex.hessian(
        f,
        x,
        y,
        z,
        argnums=(0, 1, 2),
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(x, y, z)
    H_jax = jax.hessian(f, argnums=(0, 1, 2))(x, y, z)
    assert_trees_allclose(H, H_jax, atol=1e-6)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_dict_input_preserves_pytree_on_both_axes(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Dict input returns dict-of-dicts matching jax.hessian structure."""

    def f(p):
        return jnp.sum(p["a"] ** 2) + jnp.dot(p["a"], p["b"][:2])

    inputs = {"a": jnp.ones(2), "b": jnp.ones(3)}
    H = asdex.hessian(
        f, inputs, mode=mode, output_format=output_format, chunk_size=chunk_size
    )(inputs)
    H_jax = jax.hessian(f)(inputs)
    assert_trees_allclose(H, H_jax, atol=1e-6)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_mixed_matches_jax(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """All blocks of a mixed Hessian match jax.hessian."""

    def f(x, y):
        return jnp.dot(x, y) + jnp.sum(x**2)

    x, y = jnp.array([1.0, 2.0, 3.0]), jnp.array([4.0, 5.0, 6.0])
    H = asdex.hessian(
        f,
        x,
        y,
        argnums=(0, 1),
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(x, y)
    H_jax = jax.hessian(f, argnums=(0, 1))(x, y)
    assert_trees_allclose(H, H_jax, atol=1e-6)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_two_dict_args_matches_jax(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Hessian over tuple of dicts matches jax.hessian structure."""

    def f(p, q):
        return jnp.sum(p["a"] ** 2) + jnp.dot(p["a"], q["b"]) + jnp.sum(q["c"] ** 3)

    p = {"a": jnp.array([1.0, 2.0])}
    q = {"b": jnp.array([3.0, 4.0]), "c": jnp.array([5.0, 6.0, 7.0])}

    H = asdex.hessian(
        f,
        p,
        q,
        argnums=(0, 1),
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(p, q)
    H_jax = jax.hessian(f, argnums=(0, 1))(p, q)
    assert_trees_allclose(H, H_jax, atol=1e-6)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_two_nested_dicts(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Hessian with two nested dict args matches jax.hessian."""

    def f(p, q):
        return jnp.dot(p["layer"]["w"], q["layer"]["w"])

    p = {"layer": {"w": jnp.array([1.0, 2.0])}}
    q = {"layer": {"w": jnp.array([3.0, 4.0])}}
    H = asdex.hessian(
        f,
        p,
        q,
        argnums=(0, 1),
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(p, q)
    H_jax = jax.hessian(f, argnums=(0, 1))(p, q)
    assert_trees_allclose(H, H_jax, atol=1e-6)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_complex_multi_input(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Hessian with complex multi-input PyTrees matches jax.hessian."""

    def f(model, data):
        W, b = model["W"], model["b"]
        x = data["x"]
        pred = jnp.dot(W, x) + b
        return jnp.sum(pred**2)

    model = {"W": jnp.array([1.0, 2.0, 3.0]), "b": jnp.array(1.0)}
    data = {"x": jnp.array([1.0, 2.0, 3.0])}
    H = asdex.hessian(
        f,
        model,
        data,
        argnums=(0, 1),
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(model, data)
    H_jax = jax.hessian(f, argnums=(0, 1))(model, data)
    assert_trees_allclose(H, H_jax, atol=1e-6)


# Single-input regression tests


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_single_input_path_unchanged(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Single-array Jacobian matches jax.jacobian."""

    def f(x):
        return x[1:] - x[:-1]

    x = jnp.arange(5.0)
    J = asdex.jacobian(
        f, x, mode=mode, output_format=output_format, chunk_size=chunk_size
    )(x)
    J_jax = jax.jacobian(f)(x)
    assert_trees_allclose(J, J_jax)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_single_input_path_unchanged(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Single-array Hessian matches jax.hessian."""

    def f(x):
        return jnp.sum(x**2)

    x = jnp.ones(4)
    H = asdex.hessian(
        f, x, mode=mode, output_format=output_format, chunk_size=chunk_size
    )(x)
    H_jax = jax.hessian(f)(x)
    assert_trees_allclose(H, H_jax, atol=1e-6)


# Mode agreement tests


@pytest.mark.jacobian
def test_jacobian_fwd_and_rev_modes_agree():
    """Both fwd and rev modes match jax.jacobian."""

    def f(x, y):
        return x * y

    x, y = jnp.array([1.0, 2.0, 3.0]), jnp.array([4.0, 5.0, 6.0])
    J_jax = jax.jacobian(f, argnums=(0, 1))(x, y)
    for mode in ("fwd", "rev"):
        J = asdex.jacobian(f, x, y, argnums=(0, 1), mode=mode, output_format="dense")(
            x, y
        )
        assert jax.tree.structure(J) == jax.tree.structure(J_jax)
        jax.tree.map(np.testing.assert_allclose, J, J_jax)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
def test_hessian_all_modes_match_jax(mode):
    """All Hessian AD composition modes match jax.hessian."""

    def f(x, y):
        return jnp.dot(x, y) + jnp.sum(x**2) + jnp.sum(y**3)

    x, y = jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])
    H = asdex.hessian(f, x, y, argnums=(0, 1), mode=mode, output_format="dense")(x, y)
    H_jax = jax.hessian(f, argnums=(0, 1))(x, y)
    assert jax.tree.structure(H) == jax.tree.structure(H_jax)
    jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b, atol=1e-6), H, H_jax)


@pytest.mark.hessian
@pytest.mark.parametrize("symmetric", [True, False])
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
def test_hessian_modes_symmetric_pytree_match_jax(mode, symmetric):
    """HVP modes x symmetric flag match jax.hessian for pytree multi-arg input.

    Cross-checks every Hessian block leaf-by-leaf against jax.hessian,
    including pytree structure equality,
    for a dict-pytree first argument with argnums=(0, 1).
    """

    def f(params, y):
        return (
            jnp.dot(params["a"], params["b"])
            + jnp.sum(params["a"] ** 3)
            + jnp.dot(params["b"], y**2)
        )

    params = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0, 4.0])}
    y = jnp.array([0.5, -1.5])
    H = asdex.hessian(
        f,
        params,
        y,
        argnums=(0, 1),
        mode=mode,
        symmetric=symmetric,
        output_format="dense",
    )(params, y)
    H_jax = jax.hessian(f, argnums=(0, 1))(params, y)
    assert jax.tree.structure(H) == jax.tree.structure(H_jax)
    jax.tree.map(lambda a, b: np.testing.assert_allclose(a, b, atol=1e-6), H, H_jax)


# Mixed input dtypes


@pytest.mark.jacobian
def test_jacobian_fwd_mixed_dtypes_raises():
    """Mixed input dtypes raise a clear ``TypeError`` in forward mode.

    Forward-mode tangents are sliced from one flat seed vector,
    so all differentiated leaves must share a dtype;
    without the upfront check the failure surfaces deep inside ``jax.linearize``.
    """

    def f(x, y):
        return jnp.concatenate([x**2, y.astype(jnp.float32) ** 2])

    x = jnp.zeros(2, dtype=jnp.float32)
    y = jnp.zeros(2, dtype=jnp.float16)
    jac_fn = asdex.jacobian(f, x, y, argnums=(0, 1), mode="fwd")
    with pytest.raises(TypeError, match="mixed dtypes"):
        jac_fn(x, y)


@pytest.mark.jacobian
def test_jacobian_rev_mixed_dtypes_works(assert_trees_allclose):
    """Mixed input dtypes work in reverse mode, matching ``jax.jacrev``."""

    def f(x, y):
        return jnp.concatenate([x**2, y.astype(jnp.float32) ** 2])

    x = jnp.array([1.0, 2.0], dtype=jnp.float32)
    y = jnp.array([3.0, 4.0], dtype=jnp.float16)
    J = asdex.jacobian(f, x, y, argnums=(0, 1), mode="rev", output_format="dense")(x, y)
    J_jax = jax.jacrev(f, argnums=(0, 1))(x, y)
    assert_trees_allclose(J, J_jax)


@pytest.mark.hessian
def test_hessian_mixed_dtypes_raises(hessian_mode):
    """Mixed input dtypes raise a clear ``TypeError`` in every Hessian mode.

    HVP seeds are sliced from one flat seed vector,
    so all differentiated leaves must share a dtype;
    without the upfront check the failure surfaces deep inside ``jvp``/``vjp``.
    """

    def f(x, y):
        return jnp.sum(x**2) + jnp.sum(y.astype(jnp.float32) ** 2) * x[0]

    x = jnp.zeros(2, dtype=jnp.float32)
    y = jnp.zeros(2, dtype=jnp.float16)
    hess_fn = asdex.hessian(f, x, y, argnums=(0, 1), mode=hessian_mode)
    with pytest.raises(TypeError, match="mixed dtypes"):
        hess_fn(x, y)
