"""Tests for argnums variations.

Covers negative indices, reversed order, non-contiguous selection,
single-element tuple vs int semantics, and PyTree arguments.
"""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import asdex

warnings.filterwarnings("ignore", category=asdex.DenseColoringWarning)


# Negative argnums


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_argnums_all_negative(
    mode, output_format, chunk_size, assert_trees_allclose
):
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
def test_jacobian_argnums_mixed_sign(
    mode, output_format, chunk_size, assert_trees_allclose
):
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
def test_jacobian_single_negative_argnum(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Single negative argnum matches jax.jacobian."""

    def f(x, y):
        return y**2

    x, y = jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])
    J = asdex.jacobian(
        f,
        x,
        y,
        argnums=-1,
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(x, y)
    J_jax = jax.jacobian(f, argnums=-1)(x, y)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_negative_argnums_pytree_output(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Negative argnums with PyTree output matches jax.jacobian."""

    def f(x, y, z):
        return {"sum": x + z, "diff": y - z}

    x, y, z = jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]), jnp.array([5.0, 6.0])
    J = asdex.jacobian(
        f, x, y, z, argnums=(0, -1), mode=mode, output_format=output_format
    )(x, y, z)
    J_jax = jax.jacobian(f, argnums=(0, -1))(x, y, z)
    assert_trees_allclose(J, J_jax)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_argnums_negative(
    mode, output_format, chunk_size, assert_trees_allclose
):
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


# Reversed argnums order


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_argnums_reversed(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Reversed argnums order matches jax.jacobian."""

    def f(x, y):
        return x * y

    x, y = jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])
    J = asdex.jacobian(
        f,
        x,
        y,
        argnums=(1, 0),
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(x, y)
    J_jax = jax.jacobian(f, argnums=(1, 0))(x, y)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_argnums_reversed_four_args(
    mode, output_format, chunk_size, assert_trees_allclose
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
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Reversed argnums with PyTree args matches jax.jacobian."""

    def f(p, q):
        return p["a"] * q["b"]

    p = {"a": jnp.array([1.0, 2.0, 3.0])}
    q = {"b": jnp.array([4.0, 5.0, 6.0])}
    J = asdex.jacobian(
        f,
        p,
        q,
        argnums=(1, 0),
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(p, q)
    J_jax = jax.jacobian(f, argnums=(1, 0))(p, q)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_reversed_argnums_pytree_output(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Reversed argnums order with PyTree output matches jax.jacobian."""

    def f(x, y):
        return {"a": x * y, "b": x + y}

    x, y = jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])
    J = asdex.jacobian(
        f,
        x,
        y,
        argnums=(1, 0),
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(x, y)
    J_jax = jax.jacobian(f, argnums=(1, 0))(x, y)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_reversed_argnums_complex_pytrees(
    mode, output_format, chunk_size, assert_trees_allclose
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
def test_hessian_argnums_reversed(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Hessian with reversed argnums matches jax.hessian."""

    def f(x, y):
        return jnp.dot(x, y) + jnp.sum(x**2)

    x, y = jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])
    H = asdex.hessian(
        f,
        x,
        y,
        argnums=(1, 0),
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(x, y)
    H_jax = jax.hessian(f, argnums=(1, 0))(x, y)
    assert_trees_allclose(H, H_jax, atol=1e-6)


# Non-contiguous argnums


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_argnums_skip_middle(
    mode, output_format, chunk_size, assert_trees_allclose
):
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
def test_jacobian_subset_argnums_pytree_output(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Subset argnums (skipping middle arg) with PyTree output matches jax.jacobian."""

    def f(x, y, z):
        return {"xz": x * z, "sum": x + y + z}

    x, y, z = jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]), jnp.array([5.0, 6.0])
    J = asdex.jacobian(
        f, x, y, z, argnums=(0, 2), mode=mode, output_format=output_format
    )(x, y, z)
    J_jax = jax.jacobian(f, argnums=(0, 2))(x, y, z)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_argnums_noncontiguous_descending(
    mode, output_format, chunk_size, assert_trees_allclose
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
def test_hessian_argnums_skip_middle(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Hessian skipping middle arg matches jax.hessian."""

    def f(a, b, c):
        return jnp.sum(a**2) + jnp.dot(a, c)

    a, b, c = jnp.ones(3), jnp.ones(3), jnp.ones(3) * 2
    H = asdex.hessian(
        f, a, b, c, argnums=(0, 2), mode=mode, output_format=output_format
    )(a, b, c)
    H_jax = jax.hessian(f, argnums=(0, 2))(a, b, c)
    assert_trees_allclose(H, H_jax, atol=1e-6)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_argnums_noncontiguous_descending(
    mode, output_format, chunk_size, assert_trees_allclose
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


# Single-element tuple vs int semantic distinction


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_argnums_single_element_tuple(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """argnums=(0,) returns tuple of one Jacobian, not single Jacobian.

    This is a critical semantic distinction in JAX:
    argnums=0 returns J, argnums=(0,) returns (J,).
    """

    def f(x, y):
        return x * y

    x, y = jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])
    J = asdex.jacobian(
        f,
        x,
        y,
        argnums=(0,),
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(x, y)
    J_jax = jax.jacobian(f, argnums=(0,))(x, y)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_argnums_int_vs_tuple_structure(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """argnums=0 and argnums=(0,) produce different structures."""

    def f(x):
        return x**2

    x = jnp.array([1.0, 2.0, 3.0])
    J_int = asdex.jacobian(
        f, x, argnums=0, mode=mode, output_format=output_format, chunk_size=chunk_size
    )(x)
    J_tuple = asdex.jacobian(
        f, x, argnums=(0,), mode=mode, output_format=output_format
    )(x)
    J_jax_int = jax.jacobian(f, argnums=0)(x)
    J_jax_tuple = jax.jacobian(f, argnums=(0,))(x)

    assert_trees_allclose(J_int, J_jax_int)
    assert_trees_allclose(J_tuple, J_jax_tuple)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_argnums_int_returns_single_block(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """argnums=int returns a single Jacobian matching jax.jacobian."""

    def f(x, y):
        return jnp.array([x[0] * y[0], x[1] + y[1]])

    x, y = jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])
    J = asdex.jacobian(
        f,
        x,
        y,
        argnums=0,
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(x, y)
    J_jax = jax.jacobian(f, argnums=0)(x, y)
    assert_trees_allclose(J, J_jax)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_argnums_single_element_tuple(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Hessian with argnums=(0,) returns tuple structure."""

    def f(x, y):
        return jnp.sum(x**2) + jnp.dot(x, y)

    x, y = jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])
    H = asdex.hessian(
        f,
        x,
        y,
        argnums=(0,),
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(x, y)
    H_jax = jax.hessian(f, argnums=(0,))(x, y)
    assert_trees_allclose(H, H_jax, atol=1e-6)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_argnums_int_returns_single_block(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Hessian with argnums=int matches jax.hessian."""

    def f(x, y, z):
        return jnp.dot(x, y) + jnp.sum(z**2)

    x, y, z = jnp.ones(3), jnp.ones(3), jnp.ones(3)
    H = asdex.hessian(
        f,
        x,
        y,
        z,
        argnums=1,
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(x, y, z)
    H_jax = jax.hessian(f, argnums=1)(x, y, z)
    assert_trees_allclose(H, H_jax, atol=1e-6)


# argnums with PyTree positions


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_argnums_selects_whole_pytree_position(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """argnums=0 with pytree positions matches jax.jacobian."""

    def f(p, q):
        return p["a"] + q["b"][:2]

    p = {"a": jnp.array([1.0, 2.0])}
    q = {"b": jnp.array([3.0, 4.0, 5.0])}

    J = asdex.jacobian(
        f,
        p,
        q,
        argnums=0,
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(p, q)
    J_jax = jax.jacobian(f, argnums=0)(p, q)
    assert_trees_allclose(J, J_jax)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_argnums_int_with_pytree_position(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """argnums=int on a pytree position matches jax.hessian."""

    def f(p, q):
        return jnp.sum(p["a"] ** 2) + jnp.dot(p["a"], q["b"])

    p = {"a": jnp.array([1.0, 2.0])}
    q = {"b": jnp.array([3.0, 4.0])}

    H = asdex.hessian(
        f,
        p,
        q,
        argnums=0,
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(p, q)
    H_jax = jax.hessian(f, argnums=0)(p, q)
    assert_trees_allclose(H, H_jax, atol=1e-6)


# argnums out of bounds


@pytest.mark.jacobian
def test_jacobian_argnums_out_of_bounds_refers_to_positions():
    """Argnums indexes top-level positions, not leaves — position 2 is invalid here."""

    def f(p, q):
        return p["a"] + q["b"] + q["c"]

    p = {"a": jnp.ones(2)}
    q = {"b": jnp.ones(2), "c": jnp.ones(2)}
    with pytest.raises(ValueError, match=r"len\(args\) == 2"):
        asdex.jacobian_sparsity(f, p, q, argnums=2)


# argnums with sparsity patterns


@pytest.mark.jacobian
def test_jacobian_argnums_excludes_non_selected_from_pattern():
    """Non-selected inputs do not contribute columns to the detected pattern."""

    def f(x, y):
        return x * y

    x, y = jnp.ones(3), jnp.ones(3)
    pat = asdex.jacobian_sparsity(f, x, y, argnums=0)
    assert pat.shape == (3, 3)


@pytest.mark.jacobian
def test_jacobian_argnums_subset_returns_only_selected_blocks():
    """argnums=(0, 2) on a 3-arg function matches jax.jacobian."""

    def f(x, y, z):
        return x * y + z

    x, y, z = jnp.full(3, 2.0), jnp.full(3, 3.0), jnp.ones(3)
    J = asdex.jacobian(f, x, y, z, argnums=(0, 2), output_format="dense")(x, y, z)
    J_jax = jax.jacobian(f, argnums=(0, 2))(x, y, z)
    assert jax.tree.structure(J) == jax.tree.structure(J_jax)
    jax.tree.map(np.testing.assert_allclose, J, J_jax)


@pytest.mark.hessian
def test_hessian_argnums_subset_returns_smaller_block_grid():
    """hessian(..., argnums=(0, 2)) matches jax.hessian."""

    def f(x, y, z):
        return jnp.dot(x, y) + jnp.dot(x, z) + jnp.sum(y**2)

    x, y, z = jnp.ones(3), jnp.ones(3), jnp.ones(3)
    H = asdex.hessian(f, x, y, z, argnums=(0, 2), output_format="dense")(x, y, z)
    H_jax = jax.hessian(f, argnums=(0, 2))(x, y, z)
    assert jax.tree.structure(H) == jax.tree.structure(H_jax)
    jax.tree.map(np.testing.assert_allclose, H, H_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_single_negative_argnum_pytree_output(
    mode, output_format, chunk_size, assert_trees_allclose
):
    """Single negative argnum with PyTree output matches jax.jacobian."""

    def f(x, y):
        return {"sq": y**2, "double": 2 * y}

    x, y = jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])
    J = asdex.jacobian(
        f,
        x,
        y,
        argnums=-1,
        mode=mode,
        output_format=output_format,
        chunk_size=chunk_size,
    )(x, y)
    J_jax = jax.jacobian(f, argnums=-1)(x, y)
    assert_trees_allclose(J, J_jax)


@pytest.mark.hessian
def test_hessian_argnums_int_with_pytree_position_returns_pytree_of_pytrees():
    """argnums=int on a pytree position matches jax.hessian."""

    def f(p, q):
        return jnp.sum(p["a"] ** 2) + jnp.dot(p["a"], q["b"])

    p = {"a": jnp.array([1.0, 2.0])}
    q = {"b": jnp.array([3.0, 4.0])}

    H = asdex.hessian(f, p, q, argnums=0, output_format="dense")(p, q)
    H_jax = jax.hessian(f, argnums=0)(p, q)
    assert jax.tree.structure(H) == jax.tree.structure(H_jax)
    jax.tree.map(np.testing.assert_allclose, H, H_jax)
