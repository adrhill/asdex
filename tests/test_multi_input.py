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
    """J_x and J_y from the multi-input call match the single-input closures."""

    def f(x, y):
        return jnp.array([x[0] * y[0], x[1] + y[1], x[0] * x[1]])

    x, y = jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])
    J_x_ref = asdex.jacobian(lambda x: f(x, y), (2,), output_format="dense")(x)
    J_y_ref = asdex.jacobian(lambda y: f(x, y), (2,), output_format="dense")(y)

    J_x, J_y = asdex.jacobian(f, (2,), (2,), argnums=(0, 1), output_format="dense")(
        x, y
    )
    np.testing.assert_allclose(J_x, J_x_ref)
    np.testing.assert_allclose(J_y, J_y_ref)


@pytest.mark.jacobian
def test_jacobian_asymmetric_block_shapes():
    """Differently-sized inputs produce non-transposed blocks of correct shape."""

    def f(x, y):
        return jnp.array([x[0] + y[2], x[2] * y[1]])

    J_x, J_y = asdex.jacobian(f, (3,), (4,), argnums=(0, 1), output_format="dense")(
        jnp.ones(3), jnp.ones(4)
    )
    assert J_x.shape == (2, 3)
    assert J_y.shape == (2, 4)


@pytest.mark.jacobian
def test_jacobian_three_inputs_ordering():
    """With three inputs, each block goes to the right place."""

    def f(x, y, z):
        return x * y + z

    Jx, Jy, Jz = asdex.jacobian(
        f,
        (3,),
        (3,),
        (3,),
        argnums=(0, 1, 2),
        output_format="dense",
    )(jnp.full(3, 2.0), jnp.full(3, 3.0), jnp.ones(3))
    np.testing.assert_allclose(Jx, np.diag([3.0, 3.0, 3.0]))
    np.testing.assert_allclose(Jy, np.diag([2.0, 2.0, 2.0]))
    np.testing.assert_allclose(Jz, np.eye(3))


@pytest.mark.jacobian
def test_jacobian_dict_input_preserves_pytree():
    """Dict input returns dict-of-Jacobians with same keys and leaf shapes."""

    def f(params):
        return params["w"] @ params["x"] + params["b"]

    shapes = {"w": (3, 2), "x": (2,), "b": (3,)}
    inputs = {
        "w": jnp.eye(3, 2),
        "x": jnp.array([1.0, 2.0]),
        "b": jnp.zeros(3),
    }

    J = asdex.jacobian(f, shapes, output_format="dense")(inputs)
    assert set(J.keys()) == {"w", "x", "b"}
    assert J["w"].shape == (3, 3, 2)
    assert J["x"].shape == (3, 2)
    assert J["b"].shape == (3, 3)


@pytest.mark.jacobian
def test_jacobian_fwd_and_rev_modes_agree():
    """Both fwd and rev modes produce the same result for multi-input."""

    def f(x, y):
        return x * y

    xs = (jnp.array([1.0, 2.0, 3.0]), jnp.array([4.0, 5.0, 6.0]))
    Jx_fwd, Jy_fwd = asdex.jacobian(
        f, (3,), (3,), argnums=(0, 1), mode="fwd", output_format="dense"
    )(*xs)
    Jx_rev, Jy_rev = asdex.jacobian(
        f, (3,), (3,), argnums=(0, 1), mode="rev", output_format="dense"
    )(*xs)
    np.testing.assert_allclose(Jx_fwd, Jx_rev)
    np.testing.assert_allclose(Jy_fwd, Jy_rev)


@pytest.mark.jacobian
def test_value_and_jacobian_multi_input():
    """value_and_jacobian returns matching primal and blocks."""

    def f(x, y):
        return jnp.array([x[0] * y[0], x[1] + y[1], x[0] * x[1]])

    x, y = jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])
    val, (J_x, J_y) = asdex.value_and_jacobian(
        f, (2,), (2,), argnums=(0, 1), output_format="dense"
    )(x, y)
    np.testing.assert_allclose(val, f(x, y))
    J_x_jax, J_y_jax = jax.jacobian(f, argnums=(0, 1))(x, y)
    np.testing.assert_allclose(J_x, J_x_jax)
    np.testing.assert_allclose(J_y, J_y_jax)


# Hessians


@pytest.mark.hessian
def test_hessian_diagonal_blocks_match_single_input_closures():
    """H_xx and H_yy from multi-input call match closed-over single-input fns."""

    def f(x, y):
        return jnp.sum(x**3) + jnp.dot(x[:2], y) + jnp.sum(y**2)

    x, y = jnp.array([1.0, 2.0, 3.0]), jnp.array([4.0, 5.0])
    H_xx_ref = asdex.hessian(lambda x: f(x, y), (3,), output_format="dense")(x)
    H_yy_ref = asdex.hessian(lambda y: f(x, y), (2,), output_format="dense")(y)

    H = asdex.hessian(f, (3,), (2,), argnums=(0, 1), output_format="dense")(x, y)
    np.testing.assert_allclose(H[0][0], H_xx_ref)
    np.testing.assert_allclose(H[1][1], H_yy_ref)


@pytest.mark.hessian
def test_hessian_separable_has_zero_cross_blocks():
    """f(x, y) = sum(x^2) + sum(y^2) has structurally empty H_xy / H_yx."""

    def f(x, y):
        return jnp.sum(x**2) + jnp.sum(y**2)

    H = asdex.hessian(f, (3,), (2,), argnums=(0, 1), output_format="dense")(
        jnp.ones(3), jnp.ones(2)
    )
    np.testing.assert_allclose(H[0][0], 2 * np.eye(3))
    np.testing.assert_allclose(H[1][1], 2 * np.eye(2))
    np.testing.assert_allclose(H[0][1], np.zeros((3, 2)))
    np.testing.assert_allclose(H[1][0], np.zeros((2, 3)))


@pytest.mark.hessian
def test_hessian_bilinear_has_dense_cross_blocks():
    """f(x, y) = sum(x) * sum(y) has empty diagonals, dense cross blocks."""

    def f(x, y):
        return jnp.sum(x) * jnp.sum(y)

    H = asdex.hessian(f, (3,), (2,), argnums=(0, 1), output_format="dense")(
        jnp.ones(3), jnp.ones(2)
    )
    np.testing.assert_allclose(H[0][1], np.ones((3, 2)))
    np.testing.assert_allclose(H[1][0], np.ones((2, 3)))


@pytest.mark.hessian
def test_hessian_asymmetric_block_shapes():
    """Differently-sized inputs produce four blocks with correct shapes."""

    def f(x, y):
        return jnp.sum(x**2) + jnp.dot(x, y[:3]) + jnp.sum(y**3)

    H = asdex.hessian(f, (3,), (4,), argnums=(0, 1), output_format="dense")(
        jnp.ones(3), jnp.ones(4)
    )
    (H_xx, H_xy), (H_yx, H_yy) = H
    assert H_xx.shape == (3, 3)
    assert H_xy.shape == (3, 4)
    assert H_yx.shape == (4, 3)
    assert H_yy.shape == (4, 4)


@pytest.mark.hessian
def test_hessian_three_inputs_block_grid():
    """Three inputs: 3x3 block grid matches jax.hessian element-wise."""

    def f(x, y, z):
        return jnp.dot(x, y) + jnp.dot(y, z)

    x, y, z = jnp.ones(3), jnp.ones(3), jnp.ones(3)
    H_jax = jax.hessian(f, argnums=(0, 1, 2))(x, y, z)
    H = asdex.hessian(
        f,
        (3,),
        (3,),
        (3,),
        argnums=(0, 1, 2),
        output_format="dense",
    )(x, y, z)
    for i in range(3):
        for j in range(3):
            np.testing.assert_allclose(H[i][j], H_jax[i][j])


@pytest.mark.hessian
def test_hessian_dict_input_preserves_pytree_on_both_axes():
    """Dict input returns dict-of-dicts on both Hessian axes."""

    def f(p):
        return jnp.sum(p["a"] ** 2) + jnp.dot(p["a"], p["b"][:2])

    shapes = {"a": (2,), "b": (3,)}
    H = asdex.hessian(f, shapes, output_format="dense")(
        {"a": jnp.ones(2), "b": jnp.ones(3)}
    )
    assert set(H.keys()) == {"a", "b"}
    assert set(H["a"].keys()) == {"a", "b"}
    assert H["a"]["a"].shape == (2, 2)
    assert H["a"]["b"].shape == (2, 3)
    assert H["b"]["a"].shape == (3, 2)
    assert H["b"]["b"].shape == (3, 3)


@pytest.mark.hessian
def test_hessian_mixed_matches_jax_block_for_block():
    """All blocks of a mixed Hessian agree with jax.hessian numerically."""

    def f(x, y):
        return jnp.dot(x, y) + jnp.sum(x**2)

    x, y = jnp.array([1.0, 2.0, 3.0]), jnp.array([4.0, 5.0, 6.0])
    H_jax = jax.hessian(f, argnums=(0, 1))(x, y)
    H = asdex.hessian(f, (3,), (3,), argnums=(0, 1), output_format="dense")(x, y)
    for i in range(2):
        for j in range(2):
            np.testing.assert_allclose(H[i][j], H_jax[i][j])


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
def test_hessian_all_modes_match_jax(mode):
    """All Hessian AD composition modes produce the right blocks."""

    def f(x, y):
        return jnp.dot(x, y) + jnp.sum(x**2) + jnp.sum(y**3)

    x, y = jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])
    H_jax = jax.hessian(f, argnums=(0, 1))(x, y)
    H = asdex.hessian(f, (2,), (2,), argnums=(0, 1), mode=mode, output_format="dense")(
        x, y
    )
    for i in range(2):
        for j in range(2):
            np.testing.assert_allclose(H[i][j], H_jax[i][j], atol=1e-6)


@pytest.mark.hessian
def test_value_and_hessian_multi_input():
    """value_and_hessian returns matching primal and block grid."""

    def f(x, y):
        return jnp.dot(x, y) + jnp.sum(x**2)

    x, y = jnp.array([1.0, 2.0, 3.0]), jnp.array([4.0, 5.0, 6.0])
    val, H = asdex.value_and_hessian(
        f, (3,), (3,), argnums=(0, 1), output_format="dense"
    )(x, y)
    np.testing.assert_allclose(val, f(x, y))
    H_jax = jax.hessian(f, argnums=(0, 1))(x, y)
    for i in range(2):
        for j in range(2):
            np.testing.assert_allclose(H[i][j], H_jax[i][j])


# Single-input regression


@pytest.mark.jacobian
def test_jacobian_single_input_path_unchanged():
    """Single-array Jacobian API still works exactly as before the refactor."""

    def f(x):
        return x[1:] - x[:-1]

    J = asdex.jacobian(f, (5,), output_format="dense")(jnp.arange(5.0))
    np.testing.assert_allclose(J, np.eye(5)[1:] - np.eye(5)[:-1])


@pytest.mark.hessian
def test_hessian_single_input_path_unchanged():
    """Single-array Hessian API still works exactly as before the refactor."""

    def f(x):
        return jnp.sum(x**2)

    H = asdex.hessian(f, (4,), output_format="dense")(jnp.ones(4))
    np.testing.assert_allclose(H, 2 * np.eye(4))


# Combined coloring


@pytest.mark.coloring
def test_combined_coloring_reverse_mode_disjoint_rows():
    """f(x, y) = x * y on n=3: reverse-mode coloring of [J_x | J_y] needs 1 color.

    Each output row ``i`` has nonzeros only at columns ``i`` (from x) and ``n + i``
    (from y). No two distinct rows share any column, so one color suffices.
    """

    def f(x, y):
        return x * y

    c = asdex.jacobian_coloring(f, (3,), (3,), argnums=(0, 1), mode="rev")
    assert c.num_colors == 1


@pytest.mark.coloring
def test_combined_coloring_forward_mode_couples_inputs():
    """f(x, y) = x * y on n=3: forward-mode coloring of [J_x | J_y] needs 2 colors.

    Columns ``x[i]`` and ``y[i]`` both write to output row ``i``, so they conflict.
    """

    def f(x, y):
        return x * y

    c = asdex.jacobian_coloring(f, (3,), (3,), argnums=(0, 1), mode="fwd")
    assert c.num_colors == 2


# argnums


@pytest.mark.jacobian
def test_jacobian_argnums_int_returns_single_block():
    """argnums=int returns a single Jacobian, not a 1-tuple."""

    def f(x, y):
        return jnp.array([x[0] * y[0], x[1] + y[1]])

    x, y = jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])
    J = asdex.jacobian(
        f,
        (2,),
        (2,),
        argnums=0,
        output_format="dense",
    )(x, y)

    assert not isinstance(J, tuple)
    assert J.shape == (2, 2)
    np.testing.assert_allclose(J, np.array([[3.0, 0.0], [0.0, 1.0]]))


@pytest.mark.jacobian
def test_jacobian_argnums_subset_returns_only_selected_blocks():
    """argnums=(0, 2) on a 3-arg function returns exactly two blocks, in order."""

    def f(x, y, z):
        return x * y + z

    Jx, Jz = asdex.jacobian(
        f,
        (3,),
        (3,),
        (3,),
        argnums=(0, 2),
        output_format="dense",
    )(jnp.full(3, 2.0), jnp.full(3, 3.0), jnp.ones(3))

    np.testing.assert_allclose(Jx, np.diag([3.0, 3.0, 3.0]))
    np.testing.assert_allclose(Jz, np.eye(3))


@pytest.mark.jacobian
def test_jacobian_argnums_excludes_non_selected_from_pattern():
    """Non-selected inputs do not contribute columns to the detected pattern."""

    def f(x, y):
        return x * y

    pat = asdex.jacobian_sparsity(f, (3,), (3,), argnums=0)
    assert pat.shape == (3, 3)


@pytest.mark.jacobian
def test_jacobian_argnums_amortizes_across_changing_non_diff_args():
    """Coloring built once is reused as non-differentiated args change every call."""

    def loss(params, x_batch, y_batch):
        return (params * x_batch - y_batch) ** 2

    coloring = asdex.jacobian_coloring(
        loss,
        (3,),
        (3,),
        (3,),
        argnums=0,
    )
    jac = asdex.jacobian_from_coloring(loss, coloring, output_format="dense")

    params = jnp.array([1.0, 2.0, 3.0])
    for x_batch, y_batch in [
        (jnp.array([0.5, 1.0, 1.5]), jnp.array([1.0, 2.0, 3.0])),
        (jnp.array([2.0, 0.1, 0.7]), jnp.array([0.0, 1.0, 4.0])),
        (jnp.array([1.0, 1.0, 1.0]), jnp.array([2.0, 2.0, 2.0])),
    ]:
        J = jac(params, x_batch, y_batch)
        J_ref = jax.jacobian(loss, argnums=0)(params, x_batch, y_batch)
        np.testing.assert_allclose(J, J_ref)


@pytest.mark.hessian
def test_hessian_argnums_subset_returns_smaller_block_grid():
    """hessian(..., argnums=(0, 2)) returns a 2x2 grid (not 3x3) of blocks."""

    def f(x, y, z):
        return jnp.dot(x, y) + jnp.dot(x, z) + jnp.sum(y**2)

    x, y, z = jnp.ones(3), jnp.ones(3), jnp.ones(3)
    H = asdex.hessian(
        f,
        (3,),
        (3,),
        (3,),
        argnums=(0, 2),
        output_format="dense",
    )(x, y, z)
    H_jax = jax.hessian(f, argnums=(0, 2))(x, y, z)

    assert len(H) == 2
    assert len(H[0]) == 2
    for i in range(2):
        for j in range(2):
            np.testing.assert_allclose(H[i][j], H_jax[i][j])


@pytest.mark.hessian
def test_hessian_argnums_int_returns_single_block():
    """Hessian with argnums=int returns a single block, not a nested tuple."""

    def f(x, y, z):
        return jnp.dot(x, y) + jnp.sum(z**2)

    x, y, z = jnp.ones(3), jnp.ones(3), jnp.ones(3)
    H = asdex.hessian(
        f,
        (3,),
        (3,),
        (3,),
        argnums=1,
        output_format="dense",
    )(x, y, z)

    assert not isinstance(H, tuple)
    H_jax = jax.hessian(f, argnums=1)(x, y, z)
    np.testing.assert_allclose(H, H_jax)


# Multiple pytree arguments


@pytest.mark.jacobian
def test_jacobian_two_dict_args_preserves_per_arg_pytree():
    """Top-level tuple of dicts yields a tuple of dicts of Jacobian blocks."""

    def f(p, q):
        return p["a"] * q["b"][:3] + q["c"]

    shapes = ({"a": (3,)}, {"b": (4,), "c": (3,)})
    p = {"a": jnp.array([1.0, 2.0, 3.0])}
    q = {"b": jnp.array([4.0, 5.0, 6.0, 7.0]), "c": jnp.array([8.0, 9.0, 10.0])}

    J = asdex.jacobian(f, *shapes, argnums=(0, 1), output_format="dense")(p, q)

    assert isinstance(J, tuple)
    assert len(J) == 2
    assert set(J[0].keys()) == {"a"}
    assert set(J[1].keys()) == {"b", "c"}
    assert J[0]["a"].shape == (3, 3)
    assert J[1]["b"].shape == (3, 4)
    assert J[1]["c"].shape == (3, 3)

    J_jax = jax.jacobian(f, argnums=(0, 1))(p, q)
    np.testing.assert_allclose(J[0]["a"], J_jax[0]["a"])
    np.testing.assert_allclose(J[1]["b"], J_jax[1]["b"])
    np.testing.assert_allclose(J[1]["c"], J_jax[1]["c"])


@pytest.mark.jacobian
def test_jacobian_argnums_selects_whole_pytree_position():
    """argnums=0 with pytree positions returns the full first-arg pytree."""

    def f(p, q):
        return p["a"] + q["b"][:2]

    shapes = ({"a": (2,)}, {"b": (3,)})
    p = {"a": jnp.array([1.0, 2.0])}
    q = {"b": jnp.array([3.0, 4.0, 5.0])}

    J0 = asdex.jacobian(f, *shapes, argnums=0, output_format="dense")(p, q)

    assert not isinstance(J0, tuple)
    assert set(J0.keys()) == {"a"}
    J_jax = jax.jacobian(f, argnums=0)(p, q)
    np.testing.assert_allclose(J0["a"], J_jax["a"])


@pytest.mark.jacobian
def test_jacobian_argnums_out_of_bounds_refers_to_positions():
    """Argnums indexes top-level positions, not leaves — position 2 is invalid here."""

    def f(p, q):
        return p["a"] + q["b"] + q["c"]

    shapes = ({"a": (2,)}, {"b": (2,), "c": (2,)})
    with pytest.raises(ValueError, match=r"len\(args\) == 2"):
        asdex.jacobian_sparsity(f, *shapes, argnums=2)


@pytest.mark.hessian
def test_hessian_two_dict_args_matches_jax_block_for_block():
    """Hessian over tuple of dicts indexes as H[i][key_i][j][key_j]."""

    def f(p, q):
        return jnp.sum(p["a"] ** 2) + jnp.dot(p["a"], q["b"]) + jnp.sum(q["c"] ** 3)

    shapes = ({"a": (2,)}, {"b": (2,), "c": (3,)})
    p = {"a": jnp.array([1.0, 2.0])}
    q = {"b": jnp.array([3.0, 4.0]), "c": jnp.array([5.0, 6.0, 7.0])}

    H = asdex.hessian(f, *shapes, argnums=(0, 1), output_format="dense")(p, q)
    H_jax = jax.hessian(f, argnums=(0, 1))(p, q)

    assert isinstance(H, tuple)
    assert len(H) == 2
    for outer_key in ("a",):
        for inner_pos_idx, inner_keys in ((0, ("a",)), (1, ("b", "c"))):
            for inner_key in inner_keys:
                np.testing.assert_allclose(
                    H[0][outer_key][inner_pos_idx][inner_key],
                    H_jax[0][outer_key][inner_pos_idx][inner_key],
                )
    for outer_key in ("b", "c"):
        for inner_pos_idx, inner_keys in ((0, ("a",)), (1, ("b", "c"))):
            for inner_key in inner_keys:
                np.testing.assert_allclose(
                    H[1][outer_key][inner_pos_idx][inner_key],
                    H_jax[1][outer_key][inner_pos_idx][inner_key],
                )


@pytest.mark.hessian
def test_hessian_argnums_int_with_pytree_position_returns_pytree_of_pytrees():
    """argnums=int on a pytree position returns a single dict-of-dicts block."""

    def f(p, q):
        return jnp.sum(p["a"] ** 2) + jnp.dot(p["a"], q["b"])

    shapes = ({"a": (2,)}, {"b": (2,)})
    p = {"a": jnp.array([1.0, 2.0])}
    q = {"b": jnp.array([3.0, 4.0])}

    H = asdex.hessian(f, *shapes, argnums=0, output_format="dense")(p, q)

    assert not isinstance(H, tuple)
    H_jax = jax.hessian(f, argnums=0)(p, q)
    np.testing.assert_allclose(H["a"]["a"], H_jax["a"]["a"])
