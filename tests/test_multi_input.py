"""End-to-end tests for multi-input Jacobian and Hessian detection + decompression."""

import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import asdex

warnings.filterwarnings("ignore", category=asdex.DenseColoringWarning)


# Jacobians


@pytest.mark.jacobian
def test_jacobian_wrt_x_y_and_both_agree():
    """J_x and J_y from the multi-input call match jax.jacobian."""

    def f(x, y):
        return jnp.array([x[0] * y[0], x[1] + y[1], x[0] * x[1]])

    x, y = jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])
    J = asdex.jacobian(f, x, y, argnums=(0, 1), output_format="dense")(x, y)
    J_jax = jax.jacobian(f, argnums=(0, 1))(x, y)
    assert jax.tree.structure(J) == jax.tree.structure(J_jax)
    jax.tree.map(np.testing.assert_allclose, J, J_jax)


@pytest.mark.jacobian
def test_jacobian_asymmetric_block_shapes():
    """Differently-sized inputs produce non-transposed blocks of correct shape."""

    def f(x, y):
        return jnp.array([x[0] + y[2], x[2] * y[1]])

    x, y = jnp.ones(3), jnp.ones(4)
    J = asdex.jacobian(f, x, y, argnums=(0, 1), output_format="dense")(x, y)
    J_jax = jax.jacobian(f, argnums=(0, 1))(x, y)
    assert jax.tree.structure(J) == jax.tree.structure(J_jax)
    jax.tree.map(np.testing.assert_allclose, J, J_jax)


@pytest.mark.jacobian
def test_jacobian_three_inputs_ordering():
    """With three inputs, each block goes to the right place."""

    def f(x, y, z):
        return x * y + z

    x, y, z = jnp.full(3, 2.0), jnp.full(3, 3.0), jnp.ones(3)
    J = asdex.jacobian(f, x, y, z, argnums=(0, 1, 2), output_format="dense")(x, y, z)
    J_jax = jax.jacobian(f, argnums=(0, 1, 2))(x, y, z)
    assert jax.tree.structure(J) == jax.tree.structure(J_jax)
    jax.tree.map(np.testing.assert_allclose, J, J_jax)


@pytest.mark.jacobian
def test_jacobian_dict_input_preserves_pytree():
    """Dict input returns dict-of-Jacobians matching jax.jacobian structure."""

    def f(params):
        return params["w"] @ params["x"] + params["b"]

    inputs = {
        "w": jnp.eye(3, 2),
        "x": jnp.array([1.0, 2.0]),
        "b": jnp.zeros(3),
    }

    J = asdex.jacobian(f, inputs, output_format="dense")(inputs)
    J_jax = jax.jacobian(f)(inputs)
    assert jax.tree.structure(J) == jax.tree.structure(J_jax)
    jax.tree.map(np.testing.assert_allclose, J, J_jax)


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


@pytest.mark.jacobian
def test_value_and_jacobian_multi_input():
    """value_and_jacobian returns matching primal and Jacobian structure."""

    def f(x, y):
        return jnp.array([x[0] * y[0], x[1] + y[1], x[0] * x[1]])

    x, y = jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])
    val, J = asdex.value_and_jacobian(f, x, y, argnums=(0, 1), output_format="dense")(
        x, y
    )
    np.testing.assert_allclose(val, f(x, y))
    J_jax = jax.jacobian(f, argnums=(0, 1))(x, y)
    assert jax.tree.structure(J) == jax.tree.structure(J_jax)
    jax.tree.map(np.testing.assert_allclose, J, J_jax)


# Hessians


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_diagonal_blocks_match_jax(mode, output_format, assert_trees_allclose):
    """H_xx and H_yy from multi-input call match jax.hessian."""

    def f(x, y):
        return jnp.sum(x**3) + jnp.dot(x[:2], y) + jnp.sum(y**2)

    x, y = jnp.array([1.0, 2.0, 3.0]), jnp.array([4.0, 5.0])
    H = asdex.hessian(f, x, y, argnums=(0, 1), mode=mode, output_format=output_format)(
        x, y
    )
    H_jax = jax.hessian(f, argnums=(0, 1))(x, y)
    assert_trees_allclose(H, H_jax, atol=1e-6)


@pytest.mark.hessian
def test_hessian_separable_has_zero_cross_blocks():
    """f(x, y) = sum(x^2) + sum(y^2) has structurally empty H_xy / H_yx."""

    def f(x, y):
        return jnp.sum(x**2) + jnp.sum(y**2)

    x, y = jnp.ones(3), jnp.ones(2)
    H = asdex.hessian(f, x, y, argnums=(0, 1), output_format="dense")(x, y)
    H_jax = jax.hessian(f, argnums=(0, 1))(x, y)
    assert jax.tree.structure(H) == jax.tree.structure(H_jax)
    jax.tree.map(np.testing.assert_allclose, H, H_jax)


@pytest.mark.hessian
def test_hessian_bilinear_has_dense_cross_blocks():
    """f(x, y) = sum(x) * sum(y) has empty diagonals, dense cross blocks."""

    def f(x, y):
        return jnp.sum(x) * jnp.sum(y)

    x, y = jnp.ones(3), jnp.ones(2)
    H = asdex.hessian(f, x, y, argnums=(0, 1), output_format="dense")(x, y)
    H_jax = jax.hessian(f, argnums=(0, 1))(x, y)
    assert jax.tree.structure(H) == jax.tree.structure(H_jax)
    jax.tree.map(np.testing.assert_allclose, H, H_jax)


@pytest.mark.hessian
def test_hessian_asymmetric_block_shapes():
    """Differently-sized inputs produce four blocks matching jax.hessian."""

    def f(x, y):
        return jnp.sum(x**2) + jnp.dot(x, y[:3]) + jnp.sum(y**3)

    x, y = jnp.ones(3), jnp.ones(4)
    H = asdex.hessian(f, x, y, argnums=(0, 1), output_format="dense")(x, y)
    H_jax = jax.hessian(f, argnums=(0, 1))(x, y)
    assert jax.tree.structure(H) == jax.tree.structure(H_jax)
    jax.tree.map(np.testing.assert_allclose, H, H_jax)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_three_inputs_block_grid(mode, output_format, assert_trees_allclose):
    """Three inputs: 3x3 block grid matches jax.hessian."""

    def f(x, y, z):
        return jnp.dot(x, y) + jnp.dot(y, z)

    x, y, z = jnp.ones(3), jnp.ones(3), jnp.ones(3)
    H = asdex.hessian(
        f, x, y, z, argnums=(0, 1, 2), mode=mode, output_format=output_format
    )(x, y, z)
    H_jax = jax.hessian(f, argnums=(0, 1, 2))(x, y, z)
    assert_trees_allclose(H, H_jax, atol=1e-6)


@pytest.mark.hessian
def test_hessian_dict_input_preserves_pytree_on_both_axes():
    """Dict input returns dict-of-dicts matching jax.hessian structure."""

    def f(p):
        return jnp.sum(p["a"] ** 2) + jnp.dot(p["a"], p["b"][:2])

    inputs = {"a": jnp.ones(2), "b": jnp.ones(3)}
    H = asdex.hessian(f, inputs, output_format="dense")(inputs)
    H_jax = jax.hessian(f)(inputs)
    assert jax.tree.structure(H) == jax.tree.structure(H_jax)
    jax.tree.map(np.testing.assert_allclose, H, H_jax)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_hessian_mixed_matches_jax(mode, output_format, assert_trees_allclose):
    """All blocks of a mixed Hessian match jax.hessian."""

    def f(x, y):
        return jnp.dot(x, y) + jnp.sum(x**2)

    x, y = jnp.array([1.0, 2.0, 3.0]), jnp.array([4.0, 5.0, 6.0])
    H = asdex.hessian(f, x, y, argnums=(0, 1), mode=mode, output_format=output_format)(
        x, y
    )
    H_jax = jax.hessian(f, argnums=(0, 1))(x, y)
    assert_trees_allclose(H, H_jax, atol=1e-6)


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
def test_value_and_hessian_multi_input():
    """value_and_hessian returns matching primal and Hessian structure."""

    def f(x, y):
        return jnp.dot(x, y) + jnp.sum(x**2)

    x, y = jnp.array([1.0, 2.0, 3.0]), jnp.array([4.0, 5.0, 6.0])
    val, H = asdex.value_and_hessian(f, x, y, argnums=(0, 1), output_format="dense")(
        x, y
    )
    np.testing.assert_allclose(val, f(x, y))
    H_jax = jax.hessian(f, argnums=(0, 1))(x, y)
    assert jax.tree.structure(H) == jax.tree.structure(H_jax)
    jax.tree.map(np.testing.assert_allclose, H, H_jax)


# Single-input regression


@pytest.mark.jacobian
def test_jacobian_single_input_path_unchanged():
    """Single-array Jacobian matches jax.jacobian."""

    def f(x):
        return x[1:] - x[:-1]

    x = jnp.arange(5.0)
    J = asdex.jacobian(f, x, output_format="dense")(x)
    J_jax = jax.jacobian(f)(x)
    assert jax.tree.structure(J) == jax.tree.structure(J_jax)
    np.testing.assert_allclose(J, J_jax)


@pytest.mark.hessian
def test_hessian_single_input_path_unchanged():
    """Single-array Hessian matches jax.hessian."""

    def f(x):
        return jnp.sum(x**2)

    x = jnp.ones(4)
    H = asdex.hessian(f, x, output_format="dense")(x)
    H_jax = jax.hessian(f)(x)
    assert jax.tree.structure(H) == jax.tree.structure(H_jax)
    np.testing.assert_allclose(H, H_jax)


# Combined coloring


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


# argnums


@pytest.mark.jacobian
def test_jacobian_argnums_int_returns_single_block():
    """argnums=int returns a single Jacobian matching jax.jacobian."""

    def f(x, y):
        return jnp.array([x[0] * y[0], x[1] + y[1]])

    x, y = jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])
    J = asdex.jacobian(f, x, y, argnums=0, output_format="dense")(x, y)
    J_jax = jax.jacobian(f, argnums=0)(x, y)
    assert jax.tree.structure(J) == jax.tree.structure(J_jax)
    np.testing.assert_allclose(J, J_jax)


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


@pytest.mark.jacobian
def test_jacobian_argnums_excludes_non_selected_from_pattern():
    """Non-selected inputs do not contribute columns to the detected pattern."""

    def f(x, y):
        return x * y

    x, y = jnp.ones(3), jnp.ones(3)
    pat = asdex.jacobian_sparsity(f, x, y, argnums=0)
    assert pat.shape == (3, 3)


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


@pytest.mark.hessian
def test_hessian_argnums_int_returns_single_block():
    """Hessian with argnums=int matches jax.hessian."""

    def f(x, y, z):
        return jnp.dot(x, y) + jnp.sum(z**2)

    x, y, z = jnp.ones(3), jnp.ones(3), jnp.ones(3)
    H = asdex.hessian(f, x, y, z, argnums=1, output_format="dense")(x, y, z)
    H_jax = jax.hessian(f, argnums=1)(x, y, z)
    assert jax.tree.structure(H) == jax.tree.structure(H_jax)
    np.testing.assert_allclose(H, H_jax)


# Multiple pytree arguments


@pytest.mark.jacobian
def test_jacobian_two_dict_args_preserves_per_arg_pytree():
    """Top-level tuple of dicts matches jax.jacobian structure."""

    def f(p, q):
        return p["a"] * q["b"][:3] + q["c"]

    p = {"a": jnp.array([1.0, 2.0, 3.0])}
    q = {"b": jnp.array([4.0, 5.0, 6.0, 7.0]), "c": jnp.array([8.0, 9.0, 10.0])}

    J = asdex.jacobian(f, p, q, argnums=(0, 1), output_format="dense")(p, q)
    J_jax = jax.jacobian(f, argnums=(0, 1))(p, q)
    assert jax.tree.structure(J) == jax.tree.structure(J_jax)
    jax.tree.map(np.testing.assert_allclose, J, J_jax)


@pytest.mark.jacobian
def test_jacobian_argnums_selects_whole_pytree_position():
    """argnums=0 with pytree positions matches jax.jacobian."""

    def f(p, q):
        return p["a"] + q["b"][:2]

    p = {"a": jnp.array([1.0, 2.0])}
    q = {"b": jnp.array([3.0, 4.0, 5.0])}

    J = asdex.jacobian(f, p, q, argnums=0, output_format="dense")(p, q)
    J_jax = jax.jacobian(f, argnums=0)(p, q)
    assert jax.tree.structure(J) == jax.tree.structure(J_jax)
    jax.tree.map(np.testing.assert_allclose, J, J_jax)


@pytest.mark.jacobian
def test_jacobian_argnums_out_of_bounds_refers_to_positions():
    """Argnums indexes top-level positions, not leaves — position 2 is invalid here."""

    def f(p, q):
        return p["a"] + q["b"] + q["c"]

    p = {"a": jnp.ones(2)}
    q = {"b": jnp.ones(2), "c": jnp.ones(2)}
    with pytest.raises(ValueError, match=r"len\(args\) == 2"):
        asdex.jacobian_sparsity(f, p, q, argnums=2)


@pytest.mark.hessian
def test_hessian_two_dict_args_matches_jax():
    """Hessian over tuple of dicts matches jax.hessian structure."""

    def f(p, q):
        return jnp.sum(p["a"] ** 2) + jnp.dot(p["a"], q["b"]) + jnp.sum(q["c"] ** 3)

    p = {"a": jnp.array([1.0, 2.0])}
    q = {"b": jnp.array([3.0, 4.0]), "c": jnp.array([5.0, 6.0, 7.0])}

    H = asdex.hessian(f, p, q, argnums=(0, 1), output_format="dense")(p, q)
    H_jax = jax.hessian(f, argnums=(0, 1))(p, q)
    assert jax.tree.structure(H) == jax.tree.structure(H_jax)
    jax.tree.map(np.testing.assert_allclose, H, H_jax)


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


# Multi-input with PyTree output


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_multi_input_pytree_output(mode, output_format, assert_trees_allclose):
    """Multi-input Jacobian with PyTree output matches jax.jacobian structure."""

    def f(x, y):
        return {"a": x * y, "b": x + y}

    x, y = jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])
    J = asdex.jacobian(f, x, y, argnums=(0, 1), mode=mode, output_format=output_format)(
        x, y
    )
    J_jax = jax.jacobian(f, argnums=(0, 1))(x, y)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_pytree_input_pytree_output(
    mode, output_format, assert_trees_allclose
):
    """Single pytree input with pytree output matches jax.jacobian structure."""

    def f(params):
        return {"sum": params["a"] + params["b"], "prod": params["a"] * params["b"]}

    params = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0, 4.0])}
    J = asdex.jacobian(f, params, mode=mode, output_format=output_format)(params)
    J_jax = jax.jacobian(f)(params)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_negative_argnums_pytree_output(
    mode, output_format, assert_trees_allclose
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


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_reversed_argnums_pytree_output(
    mode, output_format, assert_trees_allclose
):
    """Reversed argnums order with PyTree output matches jax.jacobian."""

    def f(x, y):
        return {"a": x * y, "b": x + y}

    x, y = jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])
    J = asdex.jacobian(f, x, y, argnums=(1, 0), mode=mode, output_format=output_format)(
        x, y
    )
    J_jax = jax.jacobian(f, argnums=(1, 0))(x, y)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_subset_argnums_pytree_output(
    mode, output_format, assert_trees_allclose
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
def test_jacobian_mixed_pytree_array_inputs_pytree_output(
    mode, output_format, assert_trees_allclose
):
    """Mixed PyTree and array inputs with PyTree output matches jax.jacobian."""

    def f(params, scale):
        return {"scaled": params["a"] * scale, "sum": params["a"] + params["b"]}

    params = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0, 4.0])}
    scale = jnp.array([2.0, 3.0])
    J = asdex.jacobian(
        f, params, scale, argnums=(0, 1), mode=mode, output_format=output_format
    )(params, scale)
    J_jax = jax.jacobian(f, argnums=(0, 1))(params, scale)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_three_pytree_inputs_pytree_output(
    mode, output_format, assert_trees_allclose
):
    """Three PyTree inputs with PyTree output matches jax.jacobian."""

    def f(p, q, r):
        return {"pq": p["x"] * q["y"], "qr": q["y"] + r["z"]}

    p = {"x": jnp.array([1.0, 2.0])}
    q = {"y": jnp.array([3.0, 4.0])}
    r = {"z": jnp.array([5.0, 6.0])}
    J = asdex.jacobian(
        f, p, q, r, argnums=(0, 1, 2), mode=mode, output_format=output_format
    )(p, q, r)
    J_jax = jax.jacobian(f, argnums=(0, 1, 2))(p, q, r)
    assert_trees_allclose(J, J_jax)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_jacobian_single_negative_argnum_pytree_output(
    mode, output_format, assert_trees_allclose
):
    """Single negative argnum with PyTree output matches jax.jacobian."""

    def f(x, y):
        return {"sq": y**2, "double": 2 * y}

    x, y = jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])
    J = asdex.jacobian(f, x, y, argnums=-1, mode=mode, output_format=output_format)(
        x, y
    )
    J_jax = jax.jacobian(f, argnums=-1)(x, y)
    assert_trees_allclose(J, J_jax)
