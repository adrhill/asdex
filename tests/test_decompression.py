"""Tests for sparse Jacobian and Hessian computation against JAX references."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.sparse import BCOO
from numpy.testing import assert_allclose

from asdex import (
    hessian,
    hessian_coloring,
    hessian_coloring_from_sparsity,
    hessian_from_coloring,
    hessian_sparsity,
    jacobian,
    jacobian_coloring,
    jacobian_coloring_from_sparsity,
    jacobian_from_coloring,
    jacobian_sparsity,
    value_and_hessian,
    value_and_hessian_from_coloring,
    value_and_jacobian,
    value_and_jacobian_from_coloring,
)

# Reference tests against jax.jacobian (row coloring, default)


@pytest.mark.jacobian
def test_diagonal():
    """Diagonal Jacobian: f(x) = x^2."""

    def f(x):
        return x**2

    x = np.array([1.0, 2.0, 3.0, 4.0])
    result = jacobian(f, x.shape)(x).todense()
    expected = jax.jacobian(f)(x)

    assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.jacobian
@pytest.mark.filterwarnings("ignore::asdex.DenseColoringWarning")
def test_lower_triangular():
    """Lower triangular Jacobian."""

    def f(x):
        return jnp.array([x[0], x[0] + x[1], x[0] + x[1] + x[2]])

    x = np.array([1.0, 2.0, 3.0])
    result = jacobian(f, x.shape)(x).todense()
    expected = jax.jacobian(f)(x)

    assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.jacobian
@pytest.mark.filterwarnings("ignore::asdex.DenseColoringWarning")
def test_upper_triangular():
    """Upper triangular Jacobian."""

    def f(x):
        return jnp.array([x[0] + x[1] + x[2], x[1] + x[2], x[2]])

    x = np.array([1.0, 2.0, 3.0])
    result = jacobian(f, x.shape)(x).todense()
    expected = jax.jacobian(f)(x)

    assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.jacobian
def test_mixed_sparsity():
    """Mixed sparsity pattern: f(x) = [x0^2, 2*x0*x1^2, sin(x2)]."""

    def f(x):
        return jnp.array([x[0] ** 2, 2 * x[0] * x[1] ** 2, jnp.sin(x[2])])

    x = np.array([1.0, 2.0, 0.5])
    result = jacobian(f, x.shape)(x).todense()
    expected = jax.jacobian(f)(x)

    assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.jacobian
@pytest.mark.filterwarnings("ignore::asdex.DenseColoringWarning")
def test_dense():
    """Dense Jacobian: all outputs depend on all inputs."""

    def f(x):
        total = jnp.sum(x)
        return jnp.array([total, total * 2, total**2])

    x = np.array([1.0, 2.0, 3.0])
    result = jacobian(f, x.shape)(x).todense()
    expected = jax.jacobian(f)(x)

    assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.jacobian
def test_zero_jacobian():
    """Zero Jacobian: constant function."""

    def f(x):
        return jnp.array([1.0, 2.0, 3.0])

    x = np.array([1.0, 2.0])
    result = jacobian(f, x.shape)(x).todense()
    expected = jax.jacobian(f)(x)

    assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.jacobian
def test_precomputed_sparsity():
    """Using pre-computed sparsity pattern."""

    def f(x):
        return x**2

    x = np.array([1.0, 2.0, 3.0])
    sparsity = jacobian_sparsity(f, 3)

    result1 = jacobian_from_coloring(f, jacobian_coloring_from_sparsity(sparsity))(
        x
    ).todense()
    result2 = jacobian(f, x.shape)(x).todense()

    assert_allclose(result1, result2, rtol=1e-10)


@pytest.mark.jacobian
def test_precomputed_colors():
    """Using pre-computed sparsity and colored pattern."""

    def f(x):
        return (x[1:] - x[:-1]) ** 2

    x = np.array([1.0, 2.0, 4.0, 3.0, 5.0])
    sparsity = jacobian_sparsity(f, 5)
    coloring = jacobian_coloring_from_sparsity(sparsity, mode="rev")

    result1 = jacobian_from_coloring(f, coloring)(x).todense()
    result2 = jacobian(f, x.shape)(x).todense()
    expected = jax.jacobian(f)(x)

    assert_allclose(result1, result2, rtol=1e-10)
    assert_allclose(result1, expected, rtol=1e-5)


@pytest.mark.jacobian
def test_different_input_points():
    """Same sparsity pattern, different input points."""

    def f(x):
        return jnp.array([x[0] * x[1], x[1] ** 2, jnp.exp(x[2])])

    sparsity = jacobian_sparsity(f, 3)
    jac_fn = jacobian_from_coloring(f, jacobian_coloring_from_sparsity(sparsity))

    for x in [
        np.array([1.0, 2.0, 0.5]),
        np.array([0.0, 0.0, 0.0]),
        np.array([-1.0, 3.0, -0.5]),
    ]:
        result = jac_fn(x).todense()
        expected = jax.jacobian(f)(x)
        assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.jacobian
@pytest.mark.filterwarnings("ignore::asdex.DenseColoringWarning")
def test_single_output():
    """Single output (scalar-valued function)."""

    def f(x):
        return jnp.array([jnp.sum(x**2)])

    x = np.array([1.0, 2.0, 3.0])
    result = jacobian(f, x.shape)(x).todense()
    expected = jax.jacobian(f)(x)

    assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.jacobian
@pytest.mark.filterwarnings("ignore::asdex.DenseColoringWarning")
def test_single_input():
    """Single input dimension."""

    def f(x):
        return jnp.array([x[0], x[0] ** 2, jnp.sin(x[0])])

    x = np.array([2.0])
    result = jacobian(f, x.shape)(x).todense()
    expected = jax.jacobian(f)(x)

    assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.jacobian
def test_tridiagonal_pattern():
    """Tridiagonal-like pattern: each output depends on neighbors."""

    def f(x):
        n = x.shape[0]
        out = []
        for i in range(n):
            val = x[i]
            if i > 0:
                val = val + x[i - 1]
            if i < n - 1:
                val = val + x[i + 1]
            out.append(val)
        return jnp.array(out)

    x = np.array([1.0, 2.0, 3.0, 4.0])
    result = jacobian(f, x.shape)(x).todense()
    expected = jax.jacobian(f)(x)

    assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.jacobian
def test_block_diagonal():
    """Block diagonal structure."""

    def f(x):
        # First two outputs depend on first two inputs
        # Last two outputs depend on last two inputs
        return jnp.array([x[0] + x[1], x[0] * x[1], x[2] + x[3], x[2] * x[3]])

    x = np.array([1.0, 2.0, 3.0, 4.0])
    result = jacobian(f, x.shape)(x).todense()
    expected = jax.jacobian(f)(x)

    assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.jacobian
@pytest.mark.filterwarnings("ignore::asdex.DenseColoringWarning")
def test_nonlinear_functions():
    """Various nonlinear functions."""

    def f(x):
        return jnp.array(
            [
                jnp.sin(x[0]) * jnp.cos(x[1]),
                jnp.exp(x[1]) + jnp.log(x[2] + 1),
                jnp.tanh(x[2]) * x[0],
            ]
        )

    x = np.array([0.5, 1.0, 0.3])
    result = jacobian(f, x.shape)(x).todense()
    expected = jax.jacobian(f)(x)

    assert_allclose(result, expected, rtol=1e-5)


# Edge cases


@pytest.mark.jacobian
def test_wide_jacobian():
    """More inputs than outputs."""

    def f(x):
        return jnp.array([jnp.sum(x[:2]), jnp.sum(x[2:])])

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = jacobian(f, x.shape)(x).todense()
    expected = jax.jacobian(f)(x)

    assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.jacobian
@pytest.mark.filterwarnings("ignore::asdex.DenseColoringWarning")
def test_tall_jacobian():
    """More outputs than inputs."""

    def f(x):
        return jnp.array([x[0], x[1], x[0] + x[1], x[0] * x[1], x[0] - x[1]])

    x = np.array([2.0, 3.0])
    result = jacobian(f, x.shape)(x).todense()
    expected = jax.jacobian(f)(x)

    assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.jacobian
def test_empty_output():
    """Function with no outputs."""

    def f(x):
        return jnp.array([])

    x = np.array([1.0, 2.0, 3.0])
    result = jacobian(f, x.shape)(x)

    assert result.shape == (0, 3)


@pytest.mark.jacobian
def test_bcoo_format():
    """Verify output is BCOO format."""

    def f(x):
        return x**2

    x = np.array([1.0, 2.0, 3.0])
    result = jacobian(f, x.shape)(x)

    assert isinstance(result, BCOO)


# Column coloring (JVP) Jacobian tests


@pytest.mark.jacobian
def test_column_partition_diagonal():
    """Column coloring on diagonal Jacobian."""

    def f(x):
        return x**2

    x = np.array([1.0, 2.0, 3.0, 4.0])
    sparsity = jacobian_sparsity(f, x.shape)
    result = jacobian_from_coloring(
        f, jacobian_coloring_from_sparsity(sparsity, mode="fwd")
    )(x).todense()
    expected = jax.jacobian(f)(x)

    assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.jacobian
def test_column_partition_mixed():
    """Column coloring on mixed sparsity."""

    def f(x):
        return jnp.array([x[0] ** 2, 2 * x[0] * x[1] ** 2, jnp.sin(x[2])])

    x = np.array([1.0, 2.0, 0.5])
    sparsity = jacobian_sparsity(f, x.shape)
    result = jacobian_from_coloring(
        f, jacobian_coloring_from_sparsity(sparsity, mode="fwd")
    )(x).todense()
    expected = jax.jacobian(f)(x)

    assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.jacobian
def test_column_partition_tridiagonal():
    """Column coloring on tridiagonal pattern."""

    def f(x):
        n = x.shape[0]
        out = []
        for i in range(n):
            val = x[i]
            if i > 0:
                val = val + x[i - 1]
            if i < n - 1:
                val = val + x[i + 1]
            out.append(val)
        return jnp.array(out)

    x = np.array([1.0, 2.0, 3.0, 4.0])
    sparsity = jacobian_sparsity(f, x.shape)
    result = jacobian_from_coloring(
        f, jacobian_coloring_from_sparsity(sparsity, mode="fwd")
    )(x).todense()
    expected = jax.jacobian(f)(x)

    assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.jacobian
def test_precomputed_col_colors():
    """Using pre-computed column colored pattern."""

    def f(x):
        return (x[1:] - x[:-1]) ** 2

    x = np.array([1.0, 2.0, 4.0, 3.0, 5.0])
    coloring = jacobian_coloring_from_sparsity(jacobian_sparsity(f, 5), mode="fwd")

    result = jacobian_from_coloring(f, coloring)(x).todense()
    expected = jax.jacobian(f)(x)

    assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.jacobian
@pytest.mark.filterwarnings("ignore::asdex.DenseColoringWarning")
def test_auto_picks_column_for_tall():
    """Auto mode picks column coloring for tall-skinny Jacobians.

    When m >> n, column coloring needs at most n colors while
    row coloring may need up to m.
    """

    def f(x):
        # 5 outputs, 2 inputs → tall Jacobian
        return jnp.array([x[0], x[1], x[0] + x[1], x[0] * x[1], x[0] - x[1]])

    x = np.array([2.0, 3.0])
    sparsity = jacobian_sparsity(f, x.shape)

    # Auto should give same result as explicit column
    result_auto = jacobian(f, x.shape)(x).todense()
    result_col = jacobian_from_coloring(
        f, jacobian_coloring_from_sparsity(sparsity, mode="fwd")
    )(x).todense()
    expected = jax.jacobian(f)(x)

    assert_allclose(result_auto, expected, rtol=1e-5)
    assert_allclose(result_col, expected, rtol=1e-5)


@pytest.mark.jacobian
@pytest.mark.filterwarnings("ignore::asdex.DenseColoringWarning")
def test_auto_picks_row_for_wide():
    """Auto mode picks row coloring for wide Jacobians.

    When n >> m, row coloring needs at most m colors while
    column coloring may need up to n.
    """

    def f(x):
        # 2 outputs, 5 inputs → wide Jacobian
        return jnp.array([jnp.sum(x[:3]), jnp.sum(x[2:])])

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    sparsity = jacobian_sparsity(f, x.shape)

    # Auto and row should give same result
    result_auto = jacobian(f, x.shape)(x).todense()
    result_row = jacobian_from_coloring(
        f, jacobian_coloring_from_sparsity(sparsity, mode="rev")
    )(x).todense()
    expected = jax.jacobian(f)(x)

    assert_allclose(result_auto, expected, rtol=1e-5)
    assert_allclose(result_row, expected, rtol=1e-5)


@pytest.mark.jacobian
def test_precomputed_auto_coloring():
    """Passing jacobian_coloring_from_sparsity(sparsity) with auto partition."""

    def f(x):
        return x**2

    x = np.array([1.0, 2.0, 3.0])
    coloring = jacobian_coloring_from_sparsity(jacobian_sparsity(f, 3))

    result = jacobian_from_coloring(f, coloring)(x).todense()
    expected = jax.jacobian(f)(x)

    assert_allclose(result, expected, rtol=1e-5)


# Input shape mismatch guard


@pytest.mark.jacobian
def test_jacobian_shape_mismatch_raises():
    """Passing an input with wrong shape raises ValueError."""

    def f(x):
        return x**2

    coloring = jacobian_sparsity(f, (2, 3))
    colored = jacobian_coloring_from_sparsity(coloring)

    with pytest.raises(ValueError, match=r"shape .* does not match expected"):
        jacobian_from_coloring(f, colored)(np.ones(6))


@pytest.mark.jacobian
def test_value_and_jacobian_shape_mismatch_raises():
    """Passing an input with wrong shape raises ValueError."""

    def f(x):
        return x**2

    coloring = jacobian_sparsity(f, (2, 3))
    colored = jacobian_coloring_from_sparsity(coloring)

    with pytest.raises(ValueError, match=r"shape .* does not match expected"):
        value_and_jacobian_from_coloring(f, colored)(np.ones(6))


@pytest.mark.hessian
def test_hessian_shape_mismatch_raises():
    """Passing an input with wrong shape raises ValueError."""

    def f(x):
        return jnp.sum(x**2)

    coloring = hessian_sparsity(f, (2, 3))
    colored = hessian_coloring_from_sparsity(coloring)

    with pytest.raises(ValueError, match=r"shape .* does not match expected"):
        hessian_from_coloring(f, colored)(np.ones(6))


@pytest.mark.hessian
def test_value_and_hessian_shape_mismatch_raises():
    """Passing an input with wrong shape raises ValueError."""

    def f(x):
        return jnp.sum(x**2)

    coloring = hessian_sparsity(f, (2, 3))
    colored = hessian_coloring_from_sparsity(coloring)

    with pytest.raises(ValueError, match=r"shape .* does not match expected"):
        value_and_hessian_from_coloring(f, colored)(np.ones(6))


# Hessian tests


@pytest.mark.hessian
def test_hessian_quadratic():
    """Hessian of quadratic function: f(x) = x^T A x."""

    def f(x):
        # Simple quadratic: sum of squares
        return jnp.sum(x**2)

    x = np.array([1.0, 2.0, 3.0])
    result = hessian(f, x.shape)(x).todense()
    expected = jax.hessian(f)(x)

    assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.hessian
def test_hessian_rosenbrock():
    """Hessian of Rosenbrock function (sparse tridiagonal-like pattern)."""

    def f(x):
        return jnp.sum((1 - x[:-1]) ** 2 + 100 * (x[1:] - x[:-1] ** 2) ** 2)

    x = np.array([1.0, 1.0, 1.0, 1.0])
    result = hessian(f, x.shape)(x).todense()
    expected = jax.hessian(f)(x)

    assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.hessian
def test_hessian_precomputed_sparsity():
    """Using pre-computed Hessian sparsity pattern."""

    def f(x):
        return jnp.sum(x**2)

    x = np.array([1.0, 2.0, 3.0])
    sparsity = hessian_sparsity(f, 3)

    result1 = hessian_from_coloring(f, hessian_coloring_from_sparsity(sparsity))(
        x
    ).todense()
    result2 = hessian(f, x.shape)(x).todense()

    assert_allclose(result1, result2, rtol=1e-10)


@pytest.mark.hessian
def test_hessian_zero():
    """Zero Hessian: linear function."""

    def f(x):
        return jnp.sum(x)  # Linear, Hessian is zero

    x = np.array([1.0, 2.0, 3.0])
    result = hessian(f, x.shape)(x)

    assert result.shape == (3, 3)
    assert result.nse == 0  # All-zero Hessian


@pytest.mark.hessian
@pytest.mark.filterwarnings("ignore::asdex.DenseColoringWarning")
def test_hessian_single_input():
    """Hessian with single input dimension."""

    def f(x):
        return x[0] ** 3

    x = np.array([2.0])
    result = hessian(f, x.shape)(x).todense()
    expected = jax.hessian(f)(x)

    assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.hessian
def test_hessian_star_coloring_default():
    """Default Hessian uses star coloring (no explicit colors passed).

    Verify that the result matches jax.hessian for a non-trivial pattern.
    """

    def f(x):
        return x[0] ** 2 * x[1] + jnp.sin(x[1]) * x[2] + x[2] ** 3

    x = np.array([1.0, 2.0, 0.5])
    result = hessian(f, x.shape)(x).todense()
    expected = jax.hessian(f)(x)

    assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.hessian
def test_hessian_squeeze_1d_output():
    """Hessian auto-squeezes functions returning shape (1,) to scalar."""

    def f(x):
        return jnp.sum(x**2, keepdims=True)

    x = np.array([1.0, 2.0, 3.0])
    result = hessian(f, x.shape)(x).todense()
    expected = jax.hessian(lambda x: jnp.sum(x**2))(x)

    assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.hessian
def test_hessian_sparsity_squeeze_1d_output():
    """hessian_sparsity auto-squeezes functions returning shape (1,)."""

    def f(x):
        return jnp.sum(x**2, keepdims=True)

    pattern = hessian_sparsity(f, 3)
    expected = hessian_sparsity(lambda x: jnp.sum(x**2), 3)

    assert pattern.shape == expected.shape
    assert pattern.nnz == expected.nnz


@pytest.mark.hessian
def test_hessian_squeeze_non_scalar_raises():
    """Hessian coloring raises ValueError for non-scalar output like (3,)."""

    def f(x):
        return x**2

    with pytest.raises(ValueError, match="output shape"):
        hessian_coloring(f, 3)


@pytest.mark.hessian
def test_hessian_arrow_pattern():
    """Arrow-shaped Hessian: star coloring should use fewer colors.

    f(x) = x[0] * sum(x) + sum(x**2)
    This creates an arrow-like Hessian where row/col 0 is dense.
    """

    def f(x):
        return x[0] * jnp.sum(x) + jnp.sum(x**2)

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = hessian(f, x.shape)(x).todense()
    expected = jax.hessian(f)(x)

    assert_allclose(result, expected, rtol=1e-5)


# Hessian AD mode tests


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
def test_hessian_ad_modes(mode):
    """All three AD modes produce the same sparse Hessian on Rosenbrock."""

    def f(x):
        return jnp.sum((1 - x[:-1]) ** 2 + 100 * (x[1:] - x[:-1] ** 2) ** 2)

    x = np.array([1.0, 2.0, 0.5, -1.0])
    result = hessian(f, x.shape, mode=mode)(x).todense()
    expected = jax.hessian(f)(x)

    assert_allclose(result, expected, rtol=1e-5)


# Jacobian mode tests


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
def test_jacobian_ad_mode(mode):
    """jacobian(f, ..., mode=...) forces the specified AD mode."""

    def f(x):
        return (x[1:] - x[:-1]) ** 2

    x = np.array([1.0, 2.0, 4.0, 3.0, 5.0])
    result = jacobian(f, x.shape, mode=mode)(x).todense()
    expected = jax.jacobian(f)(x)

    assert_allclose(result, expected, rtol=1e-5)


# Symmetric coloring for Jacobian tests


@pytest.mark.jacobian
def test_jacobian_symmetric_coloring():
    """Jacobian with symmetric=True works on a symmetric Jacobian."""

    def f(x):
        return jax.grad(lambda y: jnp.sum(y**3))(x)

    x = np.array([1.0, 2.0, 3.0, 4.0])
    result = jacobian(f, x.shape, symmetric=True)(x).todense()
    expected = jax.jacobian(f)(x)

    assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.jacobian
def test_jacobian_symmetric_coloring_rev():
    """Jacobian with symmetric=True and mode="rev" works."""

    def f(x):
        return jax.grad(lambda y: jnp.sum(y**3))(x)

    x = np.array([1.0, 2.0, 3.0, 4.0])
    result = jacobian(f, x.shape, symmetric=True, mode="rev")(x).todense()
    expected = jax.jacobian(f)(x)

    assert_allclose(result, expected, rtol=1e-5)


# Hessian non-symmetric coloring tests


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
def test_hessian_non_symmetric_coloring(mode):
    """Hessian with symmetric=False works."""

    def f(x):
        return jnp.sum((1 - x[:-1]) ** 2 + 100 * (x[1:] - x[:-1] ** 2) ** 2)

    x = np.array([1.0, 2.0, 0.5, -1.0])
    result = hessian(f, x.shape, symmetric=False, mode=mode)(x).todense()
    expected = jax.hessian(f)(x)

    assert_allclose(result, expected, rtol=1e-5)


# Wrong-mode coloring guards


@pytest.mark.jacobian
def test_jacobian_from_coloring_rejects_hessian_coloring():
    """jacobian_from_coloring raises ValueError for Hessian-mode colorings."""

    def f(x):
        return jnp.sum(x**2)

    x = np.array([1.0, 2.0, 3.0])
    coloring = hessian_coloring(f, x.shape)
    with pytest.raises(ValueError, match="Expected 'fwd' or 'rev'"):
        jacobian_from_coloring(jax.grad(f), coloring)(x)


@pytest.mark.hessian
def test_hessian_from_coloring_rejects_jacobian_coloring():
    """hessian_from_coloring raises ValueError for Jacobian-mode colorings."""

    def f(x):
        return jnp.sum(x**2)

    x = np.array([1.0, 2.0, 3.0])
    coloring = jacobian_coloring(jax.grad(f), x.shape)
    with pytest.raises(ValueError, match="Expected 'fwd_over_rev'"):
        hessian_from_coloring(f, coloring)(x)


# value_and_jacobian tests


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
def test_value_and_jacobian_diagonal(mode):
    """value_and_jacobian returns correct value and Jacobian."""

    def f(x):
        return x**2

    x = np.array([1.0, 2.0, 3.0, 4.0])
    value, jac = value_and_jacobian(f, x.shape, mode=mode)(x)

    assert_allclose(value, f(x), rtol=1e-5)
    assert_allclose(jac.todense(), jax.jacobian(f)(x), rtol=1e-5)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
def test_value_and_jacobian_mixed(mode):
    """value_and_jacobian works on mixed sparsity pattern."""

    def f(x):
        return jnp.array([x[0] ** 2, 2 * x[0] * x[1] ** 2, jnp.sin(x[2])])

    x = np.array([1.0, 2.0, 0.5])
    value, jac = value_and_jacobian(f, x.shape, mode=mode)(x)

    assert_allclose(value, f(x), rtol=1e-5)
    assert_allclose(jac.todense(), jax.jacobian(f)(x), rtol=1e-5)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
def test_value_and_jacobian_from_coloring(mode):
    """value_and_jacobian_from_coloring works with pre-computed coloring."""

    def f(x):
        return (x[1:] - x[:-1]) ** 2

    x = np.array([1.0, 2.0, 4.0, 3.0, 5.0])
    coloring = jacobian_coloring_from_sparsity(jacobian_sparsity(f, 5), mode=mode)
    value, jac = value_and_jacobian_from_coloring(f, coloring)(x)

    assert_allclose(value, f(x), rtol=1e-5)
    assert_allclose(jac.todense(), jax.jacobian(f)(x), rtol=1e-5)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
def test_value_and_jacobian_zero(mode):
    """value_and_jacobian handles constant functions (zero Jacobian)."""

    def f(x):
        return jnp.array([1.0, 2.0, 3.0])

    x = np.array([1.0, 2.0])
    value, jac = value_and_jacobian(f, x.shape, mode=mode)(x)

    assert_allclose(value, f(x), rtol=1e-5)
    assert jac.shape == (3, 2)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
def test_value_and_jacobian_empty_output(mode):
    """value_and_jacobian handles functions with no outputs."""

    def f(x):
        return jnp.array([])

    x = np.array([1.0, 2.0, 3.0])
    value, jac = value_and_jacobian(f, x.shape, mode=mode)(x)

    assert_allclose(value, f(x), rtol=1e-5)
    assert jac.shape == (0, 3)


# value_and_hessian tests


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
def test_value_and_hessian_quadratic(mode):
    """value_and_hessian returns correct value and Hessian."""

    def f(x):
        return jnp.sum(x**2)

    x = np.array([1.0, 2.0, 3.0])
    value, hess = value_and_hessian(f, x.shape, mode=mode)(x)

    assert_allclose(value, f(x), rtol=1e-5)
    assert_allclose(hess.todense(), jax.hessian(f)(x), rtol=1e-5)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
def test_value_and_hessian_rosenbrock(mode):
    """value_and_hessian works on Rosenbrock function."""

    def f(x):
        return jnp.sum((1 - x[:-1]) ** 2 + 100 * (x[1:] - x[:-1] ** 2) ** 2)

    x = np.array([1.0, 2.0, 0.5, -1.0])
    value, hess = value_and_hessian(f, x.shape, mode=mode)(x)

    assert_allclose(value, f(x), rtol=1e-5)
    assert_allclose(hess.todense(), jax.hessian(f)(x), rtol=1e-5)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
def test_value_and_hessian_from_coloring(mode):
    """value_and_hessian_from_coloring works with pre-computed coloring."""

    def f(x):
        return x[0] ** 2 * x[1] + jnp.sin(x[1]) * x[2] + x[2] ** 3

    x = np.array([1.0, 2.0, 0.5])
    coloring = hessian_coloring_from_sparsity(hessian_sparsity(f, 3), mode=mode)
    value, hess = value_and_hessian_from_coloring(f, coloring)(x)

    assert_allclose(value, f(x), rtol=1e-5)
    assert_allclose(hess.todense(), jax.hessian(f)(x), rtol=1e-5)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
def test_value_and_hessian_zero(mode):
    """value_and_hessian handles linear functions (zero Hessian)."""

    def f(x):
        return jnp.sum(x)

    x = np.array([1.0, 2.0, 3.0])
    value, hess = value_and_hessian(f, x.shape, mode=mode)(x)

    assert_allclose(value, f(x), rtol=1e-5)
    assert hess.shape == (3, 3)
    assert hess.nse == 0


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
def test_value_and_hessian_squeeze(mode):
    """value_and_hessian auto-squeezes functions returning shape (1,)."""

    def f(x):
        return jnp.sum(x**2, keepdims=True)

    x = np.array([1.0, 2.0, 3.0])
    value, hess = value_and_hessian(f, x.shape, mode=mode)(x)

    assert_allclose(value, jnp.sum(x**2), rtol=1e-5)
    assert_allclose(hess.todense(), jax.hessian(lambda x: jnp.sum(x**2))(x), rtol=1e-5)


# Dense output format


@pytest.mark.jacobian
def test_jacobian_dense_output():
    """Jacobian with output_format="dense" returns a dense jax.Array."""

    def f(x):
        return x**2

    x = np.array([1.0, 2.0, 3.0, 4.0])
    result = jacobian(f, x.shape, output_format="dense")(x)

    assert isinstance(result, jax.Array)
    assert not isinstance(result, BCOO)
    assert_allclose(result, jax.jacobian(f)(x), rtol=1e-5)


@pytest.mark.hessian
def test_hessian_dense_output():
    """Hessian with output_format="dense" returns a dense jax.Array."""

    def f(x):
        return jnp.sum(x**3)

    x = np.array([1.0, 2.0, 3.0])
    result = hessian(f, x.shape, output_format="dense")(x)

    assert isinstance(result, jax.Array)
    assert not isinstance(result, BCOO)
    assert_allclose(result, jax.hessian(f)(x), rtol=1e-5)


@pytest.mark.jacobian
def test_value_and_jacobian_dense_output():
    """value_and_jacobian with output_format="dense" returns dense jax.Arrays."""

    def f(x):
        return x**2

    x = np.array([1.0, 2.0, 3.0])
    value, jac = value_and_jacobian(f, x.shape, output_format="dense")(x)

    assert isinstance(jac, jax.Array)
    assert not isinstance(jac, BCOO)
    assert_allclose(value, f(x), rtol=1e-5)
    assert_allclose(jac, jax.jacobian(f)(x), rtol=1e-5)


@pytest.mark.hessian
def test_value_and_hessian_dense_output():
    """value_and_hessian with output_format="dense" returns dense jax.Arrays."""

    def f(x):
        return jnp.sum(x**3)

    x = np.array([1.0, 2.0, 3.0])
    value, hess = value_and_hessian(f, x.shape, output_format="dense")(x)

    assert isinstance(hess, jax.Array)
    assert not isinstance(hess, BCOO)
    assert_allclose(value, f(x), rtol=1e-5)
    assert_allclose(hess, jax.hessian(f)(x), rtol=1e-5)
