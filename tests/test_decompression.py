"""Tests for sparse Jacobian and Hessian computation against JAX references."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.sparse import BCOO
from numpy.testing import assert_allclose
from scipy.sparse import coo_array, csc_array, csr_array

from asdex import (
    ColoredPattern,
    SparsityPattern,
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
from asdex.coloring._color_symmetric import StarSet
from asdex.decompression import (
    _flatten_grad_output,
    _flatten_selected_cotangents,
    _selected_dtype,
)
from asdex.verify import _allclose_pytree

# Reference tests against jax.jacobian (row coloring, default)


@pytest.mark.jacobian
def test_diagonal():
    """Diagonal Jacobian: f(x) = x^2."""

    def f(x):
        return x**2

    x = np.array([1.0, 2.0, 3.0, 4.0])
    result = jacobian(f, x)(x).todense()
    expected = jax.jacobian(f)(x)

    assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.jacobian
@pytest.mark.filterwarnings("ignore::asdex.DenseColoringWarning")
def test_lower_triangular():
    """Lower triangular Jacobian."""

    def f(x):
        return jnp.array([x[0], x[0] + x[1], x[0] + x[1] + x[2]])

    x = np.array([1.0, 2.0, 3.0])
    result = jacobian(f, x)(x).todense()
    expected = jax.jacobian(f)(x)

    assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.jacobian
@pytest.mark.filterwarnings("ignore::asdex.DenseColoringWarning")
def test_upper_triangular():
    """Upper triangular Jacobian."""

    def f(x):
        return jnp.array([x[0] + x[1] + x[2], x[1] + x[2], x[2]])

    x = np.array([1.0, 2.0, 3.0])
    result = jacobian(f, x)(x).todense()
    expected = jax.jacobian(f)(x)

    assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.jacobian
def test_mixed_sparsity():
    """Mixed sparsity pattern: f(x) = [x0^2, 2*x0*x1^2, sin(x2)]."""

    def f(x):
        return jnp.array([x[0] ** 2, 2 * x[0] * x[1] ** 2, jnp.sin(x[2])])

    x = np.array([1.0, 2.0, 0.5])
    result = jacobian(f, x)(x).todense()
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
    result = jacobian(f, x)(x).todense()
    expected = jax.jacobian(f)(x)

    assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.jacobian
def test_zero_jacobian():
    """Zero Jacobian: constant function."""

    def f(x):
        return jnp.array([1.0, 2.0, 3.0])

    x = np.array([1.0, 2.0])
    result = jacobian(f, x)(x).todense()
    expected = jax.jacobian(f)(x)

    assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.jacobian
def test_precomputed_sparsity():
    """Using pre-computed sparsity pattern."""

    def f(x):
        return x**2

    x = np.array([1.0, 2.0, 3.0])
    sparsity = jacobian_sparsity(f, x)

    result1 = jacobian_from_coloring(f, jacobian_coloring_from_sparsity(sparsity))(
        x
    ).todense()
    result2 = jacobian(f, x)(x).todense()

    assert_allclose(result1, result2, rtol=1e-10)


@pytest.mark.jacobian
def test_precomputed_colors():
    """Using pre-computed sparsity and colored pattern."""

    def f(x):
        return (x[1:] - x[:-1]) ** 2

    x = np.array([1.0, 2.0, 4.0, 3.0, 5.0])
    sparsity = jacobian_sparsity(f, x)
    coloring = jacobian_coloring_from_sparsity(sparsity, mode="rev")

    result1 = jacobian_from_coloring(f, coloring)(x).todense()
    result2 = jacobian(f, x)(x).todense()
    expected = jax.jacobian(f)(x)

    assert_allclose(result1, result2, rtol=1e-10)
    assert_allclose(result1, expected, rtol=1e-5)


@pytest.mark.jacobian
def test_different_input_points():
    """Same sparsity pattern, different input points."""

    def f(x):
        return jnp.array([x[0] * x[1], x[1] ** 2, jnp.exp(x[2])])

    x = np.zeros(3)
    sparsity = jacobian_sparsity(f, x)
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
    result = jacobian(f, x)(x).todense()
    expected = jax.jacobian(f)(x)

    assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.jacobian
@pytest.mark.filterwarnings("ignore::asdex.DenseColoringWarning")
def test_single_input():
    """Single input dimension."""

    def f(x):
        return jnp.array([x[0], x[0] ** 2, jnp.sin(x[0])])

    x = np.array([2.0])
    result = jacobian(f, x)(x).todense()
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
    result = jacobian(f, x)(x).todense()
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
    result = jacobian(f, x)(x).todense()
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
    result = jacobian(f, x)(x).todense()
    expected = jax.jacobian(f)(x)

    assert_allclose(result, expected, rtol=1e-5)


# Edge cases


@pytest.mark.jacobian
def test_wide_jacobian():
    """More inputs than outputs."""

    def f(x):
        return jnp.array([jnp.sum(x[:2]), jnp.sum(x[2:])])

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = jacobian(f, x)(x).todense()
    expected = jax.jacobian(f)(x)

    assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.jacobian
@pytest.mark.filterwarnings("ignore::asdex.DenseColoringWarning")
def test_tall_jacobian():
    """More outputs than inputs."""

    def f(x):
        return jnp.array([x[0], x[1], x[0] + x[1], x[0] * x[1], x[0] - x[1]])

    x = np.array([2.0, 3.0])
    result = jacobian(f, x)(x).todense()
    expected = jax.jacobian(f)(x)

    assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.jacobian
def test_empty_output():
    """Function with no outputs."""

    def f(x):
        return jnp.array([])

    x = np.array([1.0, 2.0, 3.0])
    result = jacobian(f, x)(x)

    assert result.shape == (0, 3)


@pytest.mark.jacobian
def test_bcoo_format():
    """Verify output is BCOO format."""

    def f(x):
        return x**2

    x = np.array([1.0, 2.0, 3.0])
    result = jacobian(f, x)(x)

    assert isinstance(result, BCOO)


@pytest.mark.jacobian
def test_jacobian_jit_bcoo():
    """BCOO Jacobian works under jax.jit."""

    def f(x):
        return (x[1:] - x[:-1]) ** 2

    x = jnp.ones(10)
    jac_fn = jax.jit(jacobian(f, x, output_format="bcoo"))
    result = jac_fn(x)

    assert isinstance(result, BCOO)
    expected = jax.jacobian(f)(x)
    assert_allclose(result.todense(), expected, rtol=1e-5)


@pytest.mark.jacobian
def test_jacobian_jit_dense():
    """Dense Jacobian works under jax.jit."""

    def f(x):
        return (x[1:] - x[:-1]) ** 2

    x = jnp.ones(10)
    jac_fn = jax.jit(jacobian(f, x, output_format="dense"))
    result = jac_fn(x)

    assert isinstance(result, jax.Array)
    expected = jax.jacobian(f)(x)
    assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.hessian
def test_hessian_jit_bcoo():
    """BCOO Hessian works under jax.jit."""

    def f(x):
        return jnp.sum((x[1:] - x[:-1]) ** 2)

    x = jnp.ones(10)
    hess_fn = jax.jit(hessian(f, x, output_format="bcoo"))
    result = hess_fn(x)

    assert isinstance(result, BCOO)
    expected = jax.hessian(f)(x)
    assert_allclose(result.todense(), expected, rtol=1e-5)


@pytest.mark.hessian
def test_hessian_jit_dense():
    """Dense Hessian works under jax.jit."""

    def f(x):
        return jnp.sum((x[1:] - x[:-1]) ** 2)

    x = jnp.ones(10)
    hess_fn = jax.jit(hessian(f, x, output_format="dense"))
    result = hess_fn(x)

    assert isinstance(result, jax.Array)
    expected = jax.hessian(f)(x)
    assert_allclose(result, expected, rtol=1e-5)


# Column coloring (JVP) Jacobian tests


@pytest.mark.jacobian
def test_column_partition_diagonal():
    """Column coloring on diagonal Jacobian."""

    def f(x):
        return x**2

    x = np.array([1.0, 2.0, 3.0, 4.0])
    sparsity = jacobian_sparsity(f, x)
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
    sparsity = jacobian_sparsity(f, x)
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
    sparsity = jacobian_sparsity(f, x)
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
    coloring = jacobian_coloring_from_sparsity(jacobian_sparsity(f, x), mode="fwd")

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
    sparsity = jacobian_sparsity(f, x)

    # Auto should give same result as explicit column
    result_auto = jacobian(f, x)(x).todense()
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
    sparsity = jacobian_sparsity(f, x)

    # Auto and row should give same result
    result_auto = jacobian(f, x)(x).todense()
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
    coloring = jacobian_coloring_from_sparsity(jacobian_sparsity(f, x))

    result = jacobian_from_coloring(f, coloring)(x).todense()
    expected = jax.jacobian(f)(x)

    assert_allclose(result, expected, rtol=1e-5)


# Input shape mismatch guard


@pytest.mark.jacobian
def test_jacobian_shape_mismatch_raises():
    """Passing an input with wrong shape raises ValueError."""

    def f(x):
        return x**2

    x = np.ones((2, 3))
    coloring = jacobian_sparsity(f, x)
    colored = jacobian_coloring_from_sparsity(coloring)

    with pytest.raises(ValueError, match=r"shape .* does not match expected"):
        jacobian_from_coloring(f, colored)(np.ones(6))


@pytest.mark.jacobian
def test_value_and_jacobian_shape_mismatch_raises():
    """Passing an input with wrong shape raises ValueError."""

    def f(x):
        return x**2

    x = np.ones((2, 3))
    coloring = jacobian_sparsity(f, x)
    colored = jacobian_coloring_from_sparsity(coloring)

    with pytest.raises(ValueError, match=r"shape .* does not match expected"):
        value_and_jacobian_from_coloring(f, colored)(np.ones(6))


@pytest.mark.hessian
def test_hessian_shape_mismatch_raises():
    """Passing an input with wrong shape raises ValueError."""

    def f(x):
        return jnp.sum(x**2)

    x = np.ones((2, 3))
    coloring = hessian_sparsity(f, x)
    colored = hessian_coloring_from_sparsity(coloring)

    with pytest.raises(ValueError, match=r"shape .* does not match expected"):
        hessian_from_coloring(f, colored)(np.ones(6))


@pytest.mark.hessian
def test_value_and_hessian_shape_mismatch_raises():
    """Passing an input with wrong shape raises ValueError."""

    def f(x):
        return jnp.sum(x**2)

    x = np.ones((2, 3))
    coloring = hessian_sparsity(f, x)
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
    result = hessian(f, x)(x).todense()
    expected = jax.hessian(f)(x)

    assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.hessian
def test_hessian_rosenbrock():
    """Hessian of Rosenbrock function (sparse tridiagonal-like pattern)."""

    def f(x):
        return jnp.sum((1 - x[:-1]) ** 2 + 100 * (x[1:] - x[:-1] ** 2) ** 2)

    x = np.array([1.0, 1.0, 1.0, 1.0])
    result = hessian(f, x)(x).todense()
    expected = jax.hessian(f)(x)

    assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.hessian
def test_hessian_precomputed_sparsity():
    """Using pre-computed Hessian sparsity pattern."""

    def f(x):
        return jnp.sum(x**2)

    x = np.array([1.0, 2.0, 3.0])
    sparsity = hessian_sparsity(f, x)

    result1 = hessian_from_coloring(f, hessian_coloring_from_sparsity(sparsity))(
        x
    ).todense()
    result2 = hessian(f, x)(x).todense()

    assert_allclose(result1, result2, rtol=1e-10)


@pytest.mark.hessian
def test_hessian_zero():
    """Zero Hessian: linear function."""

    def f(x):
        return jnp.sum(x)  # Linear, Hessian is zero

    x = np.array([1.0, 2.0, 3.0])
    result = hessian(f, x)(x)

    assert result.shape == (3, 3)
    assert result.nse == 0  # All-zero Hessian


@pytest.mark.hessian
@pytest.mark.filterwarnings("ignore::asdex.DenseColoringWarning")
def test_hessian_single_input():
    """Hessian with single input dimension."""

    def f(x):
        return x[0] ** 3

    x = np.array([2.0])
    result = hessian(f, x)(x).todense()
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
    result = hessian(f, x)(x).todense()
    expected = jax.hessian(f)(x)

    assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.hessian
def test_hessian_squeeze_1d_output():
    """Hessian auto-squeezes functions returning shape (1,) to scalar."""

    def f(x):
        return jnp.sum(x**2, keepdims=True)

    x = np.array([1.0, 2.0, 3.0])
    result = hessian(f, x)(x).todense()
    expected = jax.hessian(lambda x: jnp.sum(x**2))(x)

    assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.hessian
def test_hessian_sparsity_squeeze_1d_output():
    """hessian_sparsity auto-squeezes functions returning shape (1,)."""

    def f(x):
        return jnp.sum(x**2, keepdims=True)

    x = np.ones(3)
    pattern = hessian_sparsity(f, x)
    expected = hessian_sparsity(lambda x: jnp.sum(x**2), x)

    assert pattern.shape == expected.shape
    assert pattern.nnz == expected.nnz


@pytest.mark.hessian
def test_hessian_squeeze_non_scalar_raises():
    """Hessian coloring raises ValueError for non-scalar output like (3,)."""

    def f(x):
        return x**2

    x = np.ones(3)
    with pytest.raises(ValueError, match="output shape"):
        hessian_coloring(f, x)


@pytest.mark.hessian
def test_hessian_arrow_pattern():
    """Arrow-shaped Hessian: star coloring should use fewer colors.

    f(x) = x[0] * sum(x) + sum(x**2)
    This creates an arrow-like Hessian where row/col 0 is dense.
    """

    def f(x):
        return x[0] * jnp.sum(x) + jnp.sum(x**2)

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = hessian(f, x)(x).todense()
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
    result = hessian(f, x, mode=mode)(x).todense()
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
    result = jacobian(f, x, mode=mode)(x).todense()
    expected = jax.jacobian(f)(x)

    assert_allclose(result, expected, rtol=1e-5)


# Symmetric coloring for Jacobian tests


@pytest.mark.jacobian
def test_jacobian_symmetric_coloring():
    """Jacobian with symmetric=True works on a symmetric Jacobian."""

    def f(x):
        return jax.grad(lambda y: jnp.sum(y**3))(x)

    x = np.array([1.0, 2.0, 3.0, 4.0])
    result = jacobian(f, x, symmetric=True)(x).todense()
    expected = jax.jacobian(f)(x)

    assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.jacobian
def test_jacobian_symmetric_coloring_rev():
    """Jacobian with symmetric=True and mode="rev" works."""

    def f(x):
        return jax.grad(lambda y: jnp.sum(y**3))(x)

    x = np.array([1.0, 2.0, 3.0, 4.0])
    result = jacobian(f, x, symmetric=True, mode="rev")(x).todense()
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
    result = hessian(f, x, symmetric=False, mode=mode)(x).todense()
    expected = jax.hessian(f)(x)

    assert_allclose(result, expected, rtol=1e-5)


# Wrong-mode coloring guards


@pytest.mark.jacobian
def test_jacobian_from_coloring_rejects_hessian_coloring():
    """jacobian_from_coloring raises ValueError for Hessian-mode colorings."""

    def f(x):
        return jnp.sum(x**2)

    x = np.array([1.0, 2.0, 3.0])
    coloring = hessian_coloring(f, x)
    with pytest.raises(ValueError, match="Expected 'fwd' or 'rev'"):
        jacobian_from_coloring(jax.grad(f), coloring)(x)


@pytest.mark.hessian
def test_hessian_from_coloring_rejects_jacobian_coloring():
    """hessian_from_coloring raises ValueError for Jacobian-mode colorings."""

    def f(x):
        return jnp.sum(x**2)

    x = np.array([1.0, 2.0, 3.0])
    coloring = jacobian_coloring(jax.grad(f), x)
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
    value, jac = value_and_jacobian(f, x, mode=mode)(x)

    assert_allclose(value, f(x), rtol=1e-5)
    assert_allclose(jac.todense(), jax.jacobian(f)(x), rtol=1e-5)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
def test_value_and_jacobian_mixed(mode):
    """value_and_jacobian works on mixed sparsity pattern."""

    def f(x):
        return jnp.array([x[0] ** 2, 2 * x[0] * x[1] ** 2, jnp.sin(x[2])])

    x = np.array([1.0, 2.0, 0.5])
    value, jac = value_and_jacobian(f, x, mode=mode)(x)

    assert_allclose(value, f(x), rtol=1e-5)
    assert_allclose(jac.todense(), jax.jacobian(f)(x), rtol=1e-5)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
def test_value_and_jacobian_from_coloring(mode):
    """value_and_jacobian_from_coloring works with pre-computed coloring."""

    def f(x):
        return (x[1:] - x[:-1]) ** 2

    x = np.array([1.0, 2.0, 4.0, 3.0, 5.0])
    coloring = jacobian_coloring_from_sparsity(jacobian_sparsity(f, x), mode=mode)
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
    value, jac = value_and_jacobian(f, x, mode=mode)(x)

    assert_allclose(value, f(x), rtol=1e-5)
    assert jac.shape == (3, 2)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
def test_value_and_jacobian_empty_output(mode):
    """value_and_jacobian handles functions with no outputs."""

    def f(x):
        return jnp.array([])

    x = np.array([1.0, 2.0, 3.0])
    value, jac = value_and_jacobian(f, x, mode=mode)(x)

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
    value, hess = value_and_hessian(f, x, mode=mode)(x)

    assert_allclose(value, f(x), rtol=1e-5)
    assert_allclose(hess.todense(), jax.hessian(f)(x), rtol=1e-5)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
def test_value_and_hessian_rosenbrock(mode):
    """value_and_hessian works on Rosenbrock function."""

    def f(x):
        return jnp.sum((1 - x[:-1]) ** 2 + 100 * (x[1:] - x[:-1] ** 2) ** 2)

    x = np.array([1.0, 2.0, 0.5, -1.0])
    value, hess = value_and_hessian(f, x, mode=mode)(x)

    assert_allclose(value, f(x), rtol=1e-5)
    assert_allclose(hess.todense(), jax.hessian(f)(x), rtol=1e-5)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
def test_value_and_hessian_from_coloring(mode):
    """value_and_hessian_from_coloring works with pre-computed coloring."""

    def f(x):
        return x[0] ** 2 * x[1] + jnp.sin(x[1]) * x[2] + x[2] ** 3

    x = np.array([1.0, 2.0, 0.5])
    coloring = hessian_coloring_from_sparsity(hessian_sparsity(f, x), mode=mode)
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
    value, hess = value_and_hessian(f, x, mode=mode)(x)

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
    value, hess = value_and_hessian(f, x, mode=mode)(x)

    assert_allclose(value, jnp.sum(x**2), rtol=1e-5)
    assert_allclose(hess.todense(), jax.hessian(lambda x: jnp.sum(x**2))(x), rtol=1e-5)


# Output format tests


def _assert_output_type(result, output_format):
    """Assert that result has the type promised by output_format."""
    match output_format:
        case "dense":
            assert isinstance(result, jax.Array)
            assert not isinstance(result, BCOO)
        case "bcoo":
            assert isinstance(result, BCOO)
        case "numpy_dense":
            assert isinstance(result, np.ndarray)
        case "scipy_coo":
            assert isinstance(result, coo_array)
        case "scipy_csr":
            assert isinstance(result, csr_array)
        case "scipy_csc":
            assert isinstance(result, csc_array)
        case _:
            pytest.fail(f"Unknown output_format {output_format!r}")


@pytest.mark.jacobian
def test_jacobian_output_format(all_output_format, to_dense):
    """Jacobian returns the promised type and correct result for all output formats."""

    def f(x):
        return x**2

    x = np.array([1.0, 2.0, 3.0, 4.0])
    result = jacobian(f, x, output_format=all_output_format)(x)

    _assert_output_type(result, all_output_format)
    assert_allclose(to_dense(result), jax.jacobian(f)(x), rtol=1e-5)


@pytest.mark.hessian
def test_hessian_output_format(all_output_format, to_dense):
    """Hessian returns the promised type and correct result for all output formats."""

    def f(x):
        return jnp.sum(x**3)

    x = np.array([1.0, 2.0, 3.0])
    result = hessian(f, x, output_format=all_output_format)(x)

    _assert_output_type(result, all_output_format)
    assert_allclose(to_dense(result), jax.hessian(f)(x), rtol=1e-5)


@pytest.mark.jacobian
def test_value_and_jacobian_output_format(all_output_format, to_dense):
    """value_and_jacobian returns the promised type and correct result."""

    def f(x):
        return x**2

    x = np.array([1.0, 2.0, 3.0])
    value, jac = value_and_jacobian(f, x, output_format=all_output_format)(x)

    _assert_output_type(jac, all_output_format)
    assert_allclose(value, f(x), rtol=1e-5)
    assert_allclose(to_dense(jac), jax.jacobian(f)(x), rtol=1e-5)


@pytest.mark.hessian
def test_value_and_hessian_output_format(all_output_format, to_dense):
    """value_and_hessian returns the promised type and correct result."""

    def f(x):
        return jnp.sum(x**3)

    x = np.array([1.0, 2.0, 3.0])
    value, hess = value_and_hessian(f, x, output_format=all_output_format)(x)

    _assert_output_type(hess, all_output_format)
    assert_allclose(value, f(x), rtol=1e-5)
    assert_allclose(to_dense(hess), jax.hessian(f)(x), rtol=1e-5)


@pytest.mark.jacobian
def test_jacobian_output_format_empty_pattern(all_output_format, to_dense):
    """Constant functions (empty pattern) still honor the requested output format."""

    def f(x):
        return jnp.zeros(2)

    x = np.array([1.0, 2.0, 3.0])
    result = jacobian(f, x, output_format=all_output_format)(x)

    _assert_output_type(result, all_output_format)
    assert_allclose(to_dense(result), np.zeros((2, 3)))


@pytest.mark.hessian
def test_hessian_output_format_empty_pattern(all_output_format, to_dense):
    """Linear functions (empty Hessian pattern) still honor the requested format."""

    def f(x):
        return jnp.sum(x) * 2.0

    x = np.array([1.0, 2.0, 3.0])
    result = hessian(f, x, output_format=all_output_format)(x)

    _assert_output_type(result, all_output_format)
    assert_allclose(to_dense(result), np.zeros((3, 3)))


# --- Empty result with has_aux tests ---


@pytest.mark.jacobian
def test_jacobian_empty_output_with_has_aux():
    """Empty Jacobian (zero output dim) with has_aux returns (empty_jac, aux)."""

    def f(x):
        aux = jnp.sum(x)
        return jnp.zeros((0,)), aux

    x = jnp.array([1.0, 2.0, 3.0])
    jac, aux = jacobian(f, x, has_aux=True, output_format="dense")(x)

    assert jac.shape == (0, 3)
    assert_allclose(aux, 6.0)


@pytest.mark.jacobian
def test_value_and_jacobian_empty_output_with_has_aux():
    """Empty value_and_jacobian with has_aux returns ((value, aux), empty_jac)."""

    def f(x):
        aux = jnp.sum(x)
        return jnp.zeros((0,)), aux

    x = jnp.array([1.0, 2.0, 3.0])
    (value, aux), jac = value_and_jacobian(f, x, has_aux=True, output_format="dense")(x)

    assert jac.shape == (0, 3)
    assert value.shape == (0,)
    assert_allclose(aux, 6.0)


@pytest.mark.hessian
def test_hessian_empty_with_has_aux():
    """Empty Hessian (zero input dim via empty sparsity) with has_aux."""

    def f(x):
        aux = "metadata"
        return jnp.array(0.0), aux

    # Create an empty sparsity pattern (no nonzeros)
    sparsity = SparsityPattern.from_coo([], [], (3, 3))
    coloring = ColoredPattern(
        sparsity=sparsity,
        colors=np.zeros(3, dtype=np.int32),
        num_colors=1,
        symmetric=True,
        mode="fwd_over_rev",
    )

    x = jnp.array([1.0, 2.0, 3.0])
    hess, aux = hessian_from_coloring(f, coloring, has_aux=True, output_format="dense")(
        x
    )

    assert hess.shape == (3, 3)
    assert_allclose(hess, jnp.zeros((3, 3)))
    assert aux == "metadata"


@pytest.mark.hessian
def test_value_and_hessian_empty_with_has_aux():
    """Empty value_and_hessian with has_aux returns ((value, aux), empty_hess)."""

    def f(x):
        aux = 42
        return jnp.sum(x), aux

    # Create an empty sparsity pattern (no nonzeros)
    sparsity = SparsityPattern.from_coo([], [], (3, 3))
    coloring = ColoredPattern(
        sparsity=sparsity,
        colors=np.zeros(3, dtype=np.int32),
        num_colors=1,
        symmetric=True,
        mode="fwd_over_rev",
    )

    x = jnp.array([1.0, 2.0, 3.0])
    (value, aux), hess = value_and_hessian_from_coloring(
        f, coloring, has_aux=True, output_format="dense"
    )(x)

    assert hess.shape == (3, 3)
    assert_allclose(hess, jnp.zeros((3, 3)))
    assert_allclose(value, 6.0)
    assert aux == 42


# --- Empty bcoo output format ---


@pytest.mark.jacobian
def test_jacobian_empty_bcoo_format():
    """Empty Jacobian with bcoo format returns proper BCOO."""

    def f(x):
        return jnp.zeros((0,))

    x = jnp.array([1.0, 2.0, 3.0])
    jac = jacobian(f, x, output_format="bcoo")(x)

    assert isinstance(jac, BCOO)
    assert jac.shape == (0, 3)


# --- Argument validation tests ---


@pytest.mark.jacobian
def test_jacobian_wrong_number_of_args():
    """Calling jacobian with wrong number of args raises ValueError."""

    def f(x, y):
        return x + y

    x, y = jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0])
    coloring = jacobian_coloring(f, x, y, argnums=(0, 1))
    jac_fn = jacobian_from_coloring(f, coloring, output_format="dense")

    # Call with only one arg instead of two
    with pytest.raises(ValueError, match="positional argument"):
        jac_fn(x)


@pytest.mark.jacobian
def test_jacobian_pytree_structure_mismatch():
    """Calling jacobian with mismatched pytree structure raises ValueError."""

    def f(params):
        return params["a"] + params["b"]

    inputs = {"a": jnp.zeros(2), "b": jnp.zeros(2)}
    coloring = jacobian_coloring(f, inputs)
    jac_fn = jacobian_from_coloring(f, coloring, output_format="dense")

    # Call with a list instead of dict
    with pytest.raises(ValueError, match="pytree structure"):
        jac_fn([jnp.zeros(2), jnp.zeros(2)])


@pytest.mark.jacobian
def test_jacobian_shape_mismatch():
    """Calling jacobian with wrong input shapes raises ValueError."""

    def f(x):
        return x**2

    x = jnp.array([1.0, 2.0, 3.0])
    coloring = jacobian_coloring(f, x)
    jac_fn = jacobian_from_coloring(f, coloring, output_format="dense")

    # Call with a different shape
    with pytest.raises(ValueError, match="shape"):
        jac_fn(jnp.array([1.0, 2.0]))


# --- Internal function edge cases ---


def test_flatten_selected_cotangents_empty():
    """_flatten_selected_cotangents with empty pytree returns zeros."""
    # Create a sparsity pattern that selects an empty tuple
    sparsity = SparsityPattern.from_coo([0], [0], (1, 1))
    # Cotangents tuple where selected position is empty dict (no leaves)
    cotangents = ({},)
    result = _flatten_selected_cotangents(cotangents, sparsity)
    assert result.shape == (0,)


def test_flatten_grad_output_empty():
    """_flatten_grad_output with empty pytree returns zeros."""
    # Empty dict has no leaves
    result = _flatten_grad_output({})
    assert result.shape == (0,)


def test_selected_dtype_no_leaves():
    """_selected_dtype with no leaves returns jnp.float_ fallback."""
    # Create sparsity that selects an empty tuple
    sparsity = SparsityPattern.from_coo([0], [0], (1, 1))
    # Args where selected position has no dtype (empty dict)
    args = ({},)
    result = _selected_dtype(args, sparsity)
    assert result == jnp.float_


def test_gather_indices_empty_symmetric():
    """_gather_indices with nnz=0 symmetric pattern returns empty array."""
    # Create empty symmetric colored pattern
    sparsity = SparsityPattern.from_coo([], [], (3, 3))
    coloring = ColoredPattern(
        sparsity=sparsity,
        colors=np.zeros(3, dtype=np.int32),
        num_colors=1,
        symmetric=True,
        mode="fwd_over_rev",
        star_set=StarSet(
            star=np.array([], dtype=np.intp),
            hub=np.array([], dtype=np.intp),
            edge_index={},
        ),
    )
    indices = coloring._gather_indices
    assert indices.shape == (0, 2)


# PyTree output tests


@pytest.mark.jacobian
def test_jacobian_pytree_output_dict():
    """asdex.jacobian matches jax.jacobian for dict output."""

    def f(x):
        return {"a": x[:2], "b": jnp.sum(x**2)}

    x = jnp.array([1.0, 2.0, 3.0])
    result = jacobian(f, x, output_format="dense")(x)
    expected = jax.jacobian(f)(x)
    assert _allclose_pytree(result, expected, rtol=1e-5)


@pytest.mark.jacobian
def test_jacobian_pytree_output_tuple():
    """asdex.jacobian matches jax.jacobian for tuple output."""

    def f(x):
        return (x**2, x[:2])

    x = jnp.array([1.0, 2.0, 3.0])
    result = jacobian(f, x, output_format="dense")(x)
    expected = jax.jacobian(f)(x)
    assert _allclose_pytree(result, expected, rtol=1e-5)


@pytest.mark.jacobian
def test_jacobian_pytree_output_nested():
    """asdex.jacobian matches jax.jacobian for nested PyTree output."""

    def f(x):
        return {"out": [x[:2], jnp.sum(x)]}

    x = jnp.array([1.0, 2.0, 3.0])
    result = jacobian(f, x, output_format="dense")(x)
    expected = jax.jacobian(f)(x)
    assert _allclose_pytree(result, expected, rtol=1e-5)


# PyTree input tests


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
def test_jacobian_pytree_input_dict(mode):
    """asdex.jacobian matches jax.jacobian for dict input."""

    def f(params):
        return params["a"] + params["b"] * 2

    params = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0, 4.0])}
    result = jacobian(f, params, mode=mode, output_format="dense")(params)
    expected = jax.jacobian(f)(params)
    assert _allclose_pytree(result, expected, rtol=1e-5)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
def test_jacobian_pytree_input_tuple(mode):
    """asdex.jacobian matches jax.jacobian for tuple input."""

    def f(params):
        return params[0] ** 2 + params[1]

    params = (jnp.array([1.0, 2.0]), jnp.array([3.0, 4.0]))
    result = jacobian(f, params, mode=mode, output_format="dense")(params)
    expected = jax.jacobian(f)(params)
    assert _allclose_pytree(result, expected, rtol=1e-5)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
def test_jacobian_from_coloring_pytree_input(mode):
    """jacobian_from_coloring works with PyTree input."""

    def f(params):
        return params["a"] + params["b"] * 2

    params = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0, 4.0])}
    coloring = jacobian_coloring(f, params, mode=mode)
    result = jacobian_from_coloring(f, coloring, output_format="dense")(params)
    expected = jax.jacobian(f)(params)
    assert _allclose_pytree(result, expected, rtol=1e-5)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
def test_value_and_jacobian_pytree_input(mode):
    """value_and_jacobian works with PyTree input."""

    def f(params):
        return params["a"] + params["b"] * 2

    params = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0, 4.0])}
    val, jac = value_and_jacobian(f, params, mode=mode, output_format="dense")(params)
    expected_val = f(params)
    expected_jac = jax.jacobian(f)(params)
    assert_allclose(val, expected_val, rtol=1e-5)
    assert _allclose_pytree(jac, expected_jac, rtol=1e-5)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
def test_value_and_jacobian_from_coloring_pytree_input(mode):
    """value_and_jacobian_from_coloring works with PyTree input."""

    def f(params):
        return params["a"] + params["b"] * 2

    params = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0, 4.0])}
    coloring = jacobian_coloring(f, params, mode=mode)
    val, jac = value_and_jacobian_from_coloring(f, coloring, output_format="dense")(
        params
    )
    expected_val = f(params)
    expected_jac = jax.jacobian(f)(params)
    assert_allclose(val, expected_val, rtol=1e-5)
    assert _allclose_pytree(jac, expected_jac, rtol=1e-5)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
def test_jacobian_coloring_from_sparsity_pytree_input(mode):
    """jacobian_coloring_from_sparsity works with PyTree-derived sparsity."""

    def f(params):
        return params["a"] + params["b"] * 2

    params = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0, 4.0])}
    sparsity = jacobian_sparsity(f, params)
    coloring = jacobian_coloring_from_sparsity(sparsity, mode=mode)
    result = jacobian_from_coloring(f, coloring, output_format="dense")(params)
    expected = jax.jacobian(f)(params)
    assert _allclose_pytree(result, expected, rtol=1e-5)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
def test_hessian_pytree_input_dict(mode):
    """asdex.hessian matches jax.hessian for dict input."""

    def f(params):
        return jnp.sum(params["a"] ** 2) + jnp.dot(params["a"], params["b"])

    params = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0, 4.0])}
    result = hessian(f, params, mode=mode, output_format="dense")(params)
    expected = jax.hessian(f)(params)
    assert _allclose_pytree(result, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
def test_hessian_from_coloring_pytree_input(mode):
    """hessian_from_coloring works with PyTree input."""

    def f(params):
        return jnp.sum(params["a"] ** 2) + jnp.dot(params["a"], params["b"])

    params = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0, 4.0])}
    coloring = hessian_coloring(f, params, mode=mode)
    result = hessian_from_coloring(f, coloring, output_format="dense")(params)
    expected = jax.hessian(f)(params)
    assert _allclose_pytree(result, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
def test_value_and_hessian_pytree_input(mode):
    """value_and_hessian works with PyTree input."""

    def f(params):
        return jnp.sum(params["a"] ** 2) + jnp.dot(params["a"], params["b"])

    params = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0, 4.0])}
    val, hess = value_and_hessian(f, params, mode=mode, output_format="dense")(params)
    expected_val = f(params)
    expected_hess = jax.hessian(f)(params)
    assert_allclose(val, expected_val, rtol=1e-5)
    assert _allclose_pytree(hess, expected_hess, rtol=1e-5, atol=1e-5)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
def test_value_and_hessian_from_coloring_pytree_input(mode):
    """value_and_hessian_from_coloring works with PyTree input."""

    def f(params):
        return jnp.sum(params["a"] ** 2) + jnp.dot(params["a"], params["b"])

    params = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0, 4.0])}
    coloring = hessian_coloring(f, params, mode=mode)
    val, hess = value_and_hessian_from_coloring(f, coloring, output_format="dense")(
        params
    )
    expected_val = f(params)
    expected_hess = jax.hessian(f)(params)
    assert_allclose(val, expected_val, rtol=1e-5)
    assert _allclose_pytree(hess, expected_hess, rtol=1e-5, atol=1e-5)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
def test_hessian_coloring_from_sparsity_pytree_input(mode):
    """hessian_coloring_from_sparsity works with PyTree-derived sparsity."""

    def f(params):
        return jnp.sum(params["a"] ** 2) + jnp.dot(params["a"], params["b"])

    params = {"a": jnp.array([1.0, 2.0]), "b": jnp.array([3.0, 4.0])}
    sparsity = hessian_sparsity(f, params)
    coloring = hessian_coloring_from_sparsity(sparsity, mode=mode)
    result = hessian_from_coloring(f, coloring, output_format="dense")(params)
    expected = jax.hessian(f)(params)
    assert _allclose_pytree(result, expected, rtol=1e-5, atol=1e-5)


# Multi-arg and argnums tests


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
def test_jacobian_multi_arg_all_argnums(mode):
    """asdex.jacobian matches jax.jacobian for multi-arg with argnums=(0, 1)."""

    def f(x, y):
        return x + y * 2

    x = jnp.array([1.0, 2.0])
    y = jnp.array([3.0, 4.0])
    result = jacobian(f, x, y, argnums=(0, 1), mode=mode, output_format="dense")(x, y)
    expected = jax.jacobian(f, argnums=(0, 1))(x, y)
    assert _allclose_pytree(result, expected, rtol=1e-5)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
def test_jacobian_multi_arg_single_argnum_int(mode):
    """asdex.jacobian with argnums=0 (int) matches jax.jacobian."""

    def f(x, y):
        return x + y * 2

    x = jnp.array([1.0, 2.0])
    y = jnp.array([3.0, 4.0])
    result = jacobian(f, x, y, argnums=0, mode=mode, output_format="dense")(x, y)
    expected = jax.jacobian(f, argnums=0)(x, y)
    assert _allclose_pytree(result, expected, rtol=1e-5)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
def test_jacobian_multi_arg_single_argnum_tuple(mode):
    """asdex.jacobian with argnums=(1,) (tuple) matches jax.jacobian."""

    def f(x, y):
        return x + y * 2

    x = jnp.array([1.0, 2.0])
    y = jnp.array([3.0, 4.0])
    result = jacobian(f, x, y, argnums=(1,), mode=mode, output_format="dense")(x, y)
    expected = jax.jacobian(f, argnums=(1,))(x, y)
    assert _allclose_pytree(result, expected, rtol=1e-5)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
def test_jacobian_from_coloring_multi_arg(mode):
    """jacobian_from_coloring works with multi-arg function."""

    def f(x, y):
        return x + y * 2

    x = jnp.array([1.0, 2.0])
    y = jnp.array([3.0, 4.0])
    coloring = jacobian_coloring(f, x, y, argnums=(0, 1), mode=mode)
    result = jacobian_from_coloring(f, coloring, output_format="dense")(x, y)
    expected = jax.jacobian(f, argnums=(0, 1))(x, y)
    assert _allclose_pytree(result, expected, rtol=1e-5)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
def test_value_and_jacobian_multi_arg(mode):
    """value_and_jacobian works with multi-arg function."""

    def f(x, y):
        return x + y * 2

    x = jnp.array([1.0, 2.0])
    y = jnp.array([3.0, 4.0])
    val, jac = value_and_jacobian(
        f, x, y, argnums=(0, 1), mode=mode, output_format="dense"
    )(x, y)
    expected_val = f(x, y)
    expected_jac = jax.jacobian(f, argnums=(0, 1))(x, y)
    assert_allclose(val, expected_val, rtol=1e-5)
    assert _allclose_pytree(jac, expected_jac, rtol=1e-5)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
def test_value_and_jacobian_from_coloring_multi_arg(mode):
    """value_and_jacobian_from_coloring works with multi-arg function."""

    def f(x, y):
        return x + y * 2

    x = jnp.array([1.0, 2.0])
    y = jnp.array([3.0, 4.0])
    coloring = jacobian_coloring(f, x, y, argnums=(0, 1), mode=mode)
    val, jac = value_and_jacobian_from_coloring(f, coloring, output_format="dense")(
        x, y
    )
    expected_val = f(x, y)
    expected_jac = jax.jacobian(f, argnums=(0, 1))(x, y)
    assert_allclose(val, expected_val, rtol=1e-5)
    assert _allclose_pytree(jac, expected_jac, rtol=1e-5)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
def test_jacobian_coloring_from_sparsity_multi_arg(mode):
    """jacobian_coloring_from_sparsity works with multi-arg sparsity."""

    def f(x, y):
        return x + y * 2

    x = jnp.array([1.0, 2.0])
    y = jnp.array([3.0, 4.0])
    sparsity = jacobian_sparsity(f, x, y, argnums=(0, 1))
    coloring = jacobian_coloring_from_sparsity(sparsity, mode=mode)
    result = jacobian_from_coloring(f, coloring, output_format="dense")(x, y)
    expected = jax.jacobian(f, argnums=(0, 1))(x, y)
    assert _allclose_pytree(result, expected, rtol=1e-5)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
def test_hessian_multi_arg_all_argnums(mode):
    """asdex.hessian matches jax.hessian for multi-arg with argnums=(0, 1)."""

    def f(x, y):
        return jnp.sum(x**2) + jnp.dot(x, y)

    x = jnp.array([1.0, 2.0])
    y = jnp.array([3.0, 4.0])
    result = hessian(f, x, y, argnums=(0, 1), mode=mode, output_format="dense")(x, y)
    expected = jax.hessian(f, argnums=(0, 1))(x, y)
    assert _allclose_pytree(result, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
def test_hessian_multi_arg_single_argnum(mode):
    """asdex.hessian with argnums=0 matches jax.hessian."""

    def f(x, y):
        return jnp.sum(x**2) + jnp.dot(x, y)

    x = jnp.array([1.0, 2.0])
    y = jnp.array([3.0, 4.0])
    result = hessian(f, x, y, argnums=0, mode=mode, output_format="dense")(x, y)
    expected = jax.hessian(f, argnums=0)(x, y)
    assert _allclose_pytree(result, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
def test_hessian_from_coloring_multi_arg(mode):
    """hessian_from_coloring works with multi-arg function."""

    def f(x, y):
        return jnp.sum(x**2) + jnp.dot(x, y)

    x = jnp.array([1.0, 2.0])
    y = jnp.array([3.0, 4.0])
    coloring = hessian_coloring(f, x, y, argnums=(0, 1), mode=mode)
    result = hessian_from_coloring(f, coloring, output_format="dense")(x, y)
    expected = jax.hessian(f, argnums=(0, 1))(x, y)
    assert _allclose_pytree(result, expected, rtol=1e-5, atol=1e-5)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
def test_value_and_hessian_multi_arg(mode):
    """value_and_hessian works with multi-arg function."""

    def f(x, y):
        return jnp.sum(x**2) + jnp.dot(x, y)

    x = jnp.array([1.0, 2.0])
    y = jnp.array([3.0, 4.0])
    val, hess = value_and_hessian(
        f, x, y, argnums=(0, 1), mode=mode, output_format="dense"
    )(x, y)
    expected_val = f(x, y)
    expected_hess = jax.hessian(f, argnums=(0, 1))(x, y)
    assert_allclose(val, expected_val, rtol=1e-5)
    assert _allclose_pytree(hess, expected_hess, rtol=1e-5, atol=1e-5)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
def test_value_and_hessian_from_coloring_multi_arg(mode):
    """value_and_hessian_from_coloring works with multi-arg function."""

    def f(x, y):
        return jnp.sum(x**2) + jnp.dot(x, y)

    x = jnp.array([1.0, 2.0])
    y = jnp.array([3.0, 4.0])
    coloring = hessian_coloring(f, x, y, argnums=(0, 1), mode=mode)
    val, hess = value_and_hessian_from_coloring(f, coloring, output_format="dense")(
        x, y
    )
    expected_val = f(x, y)
    expected_hess = jax.hessian(f, argnums=(0, 1))(x, y)
    assert_allclose(val, expected_val, rtol=1e-5)
    assert _allclose_pytree(hess, expected_hess, rtol=1e-5, atol=1e-5)


@pytest.mark.hessian
@pytest.mark.parametrize("mode", ["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
def test_hessian_coloring_from_sparsity_multi_arg(mode):
    """hessian_coloring_from_sparsity works with multi-arg sparsity."""

    def f(x, y):
        return jnp.sum(x**2) + jnp.dot(x, y)

    x = jnp.array([1.0, 2.0])
    y = jnp.array([3.0, 4.0])
    sparsity = hessian_sparsity(f, x, y, argnums=(0, 1))
    coloring = hessian_coloring_from_sparsity(sparsity, mode=mode)
    result = hessian_from_coloring(f, coloring, output_format="dense")(x, y)
    expected = jax.hessian(f, argnums=(0, 1))(x, y)
    assert _allclose_pytree(result, expected, rtol=1e-5, atol=1e-5)


# chunk_size edge cases (basic chunk_size coverage is in e2e tests via fixture)


@pytest.mark.jacobian
def test_jacobian_chunk_size_one():
    """chunk_size=1 processes one color at a time (sequential fallback)."""

    def f(x):
        return x**2 + jnp.roll(x, 1)

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = jacobian(f, x, chunk_size=1)(x).todense()
    expected = jax.jacobian(f)(x)
    assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.hessian
def test_hessian_chunk_size_one():
    """chunk_size=1 processes one color at a time (sequential fallback)."""

    def f(x):
        return jnp.sum(x**2) + jnp.sum(x[:-1] * x[1:])

    x = np.array([1.0, 2.0, 3.0, 4.0])
    result = hessian(f, x, chunk_size=1)(x).todense()
    expected = jax.hessian(f)(x)
    assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.jacobian
def test_jacobian_chunk_size_validation():
    """chunk_size=0 or negative raises ValueError."""

    def f(x):
        return x**2

    x = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        jacobian(f, x, chunk_size=0)(x)
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        jacobian(f, x, chunk_size=-1)(x)


@pytest.mark.jacobian
def test_jacobian_chunk_size_larger_than_colors():
    """chunk_size larger than n_colors behaves like None (full vmap)."""

    def f(x):
        return x**2

    x = np.array([1.0, 2.0, 3.0])
    result = jacobian(f, x, chunk_size=100)(x).todense()
    expected = jax.jacobian(f)(x)
    assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.jacobian
def test_jacobian_chunk_size_zero_empty_jacobian():
    """chunk_size=0 raises even for empty Jacobian (no colors to process).

    Validates Copilot review concern: validation should happen before early return.
    """

    def f(x):
        return jnp.array([])

    x = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="chunk_size must be positive"):
        jacobian(f, x, chunk_size=0)(x)


@pytest.mark.jacobian
def test_jacobian_chunk_size_pytree_output():
    """chunk_size works with PyTree (dict) output.

    Validates Copilot review concern: reshape should handle arbitrary output shapes.
    """

    def f(x):
        return {"a": x[:2], "b": x[1:]}

    x = np.array([1.0, 2.0, 3.0, 4.0])
    result = jacobian(f, x, chunk_size=2, output_format="dense")(x)
    expected = jax.jacobian(f)(x)
    assert _allclose_pytree(result, expected, rtol=1e-5)


@pytest.mark.jacobian
@pytest.mark.parametrize("fmt", ["scipy_coo", "scipy_csr", "scipy_csc"])
def test_scipy_rejects_high_dimensional_output(fmt):
    """SciPy formats raise ValueError for high-dimensional Jacobian blocks."""

    def f(x):
        return x.reshape(2, 2)

    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="only support 2D"):
        jacobian(f, x, output_format=fmt)(x)


@pytest.mark.hessian
@pytest.mark.parametrize("fmt", ["scipy_coo", "scipy_csr", "scipy_csc"])
def test_scipy_rejects_high_dimensional_hessian(fmt):
    """SciPy formats raise ValueError for high-dimensional Hessian blocks."""

    def f(x):
        return jnp.sum(x**3)

    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(ValueError, match="only support 2D"):
        hessian(f, x, output_format=fmt)(x)


@pytest.mark.jacobian
@pytest.mark.parametrize("fmt", ["scipy_coo", "scipy_csr", "scipy_csc"])
def test_scipy_rejects_pytree_output(fmt):
    """SciPy formats raise ValueError for PyTree outputs with high-dimensional blocks."""

    def f(x):
        return {"a": x[:2], "b": jnp.sum(x**2)}

    x = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="only support 2D"):
        jacobian(f, x, output_format=fmt)(x)


@pytest.mark.jacobian
@pytest.mark.parametrize("fmt", ["scipy_coo", "scipy_csr", "scipy_csc"])
def test_scipy_rejects_pytree_output_with_2d_blocks(fmt):
    """SciPy formats reject PyTree outputs even when every Jacobian block is 2D."""

    def f(x):
        return {"a": x[:2] * 2.0, "b": x * 3.0}

    x = np.array([1.0, 2.0, 3.0])
    with pytest.raises(ValueError, match="only support 2D"):
        jacobian(f, x, output_format=fmt)(x)


@pytest.mark.jacobian
@pytest.mark.parametrize("fmt", ["scipy_coo", "scipy_csr", "scipy_csc"])
def test_scipy_rejects_pytree_input(fmt):
    """SciPy formats reject PyTree inputs for Jacobians."""

    def f(params):
        return params["a"] * params["b"]

    params = {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])}
    with pytest.raises(ValueError, match="only support 2D"):
        jacobian(f, params, output_format=fmt)(params)


@pytest.mark.hessian
@pytest.mark.parametrize("fmt", ["scipy_coo", "scipy_csr", "scipy_csc"])
def test_scipy_rejects_pytree_input_hessian(fmt):
    """SciPy formats reject PyTree inputs for Hessians."""

    def f(params):
        return jnp.sum(params["a"] ** 2) + jnp.sum(params["b"] ** 3)

    params = {"a": np.array([1.0, 2.0]), "b": np.array([3.0])}
    with pytest.raises(ValueError, match="only support 2D"):
        hessian(f, params, output_format=fmt)(params)


@pytest.mark.jacobian
@pytest.mark.parametrize("fmt", ["scipy_coo", "scipy_csr", "scipy_csc"])
def test_scipy_preserves_structural_zeros(fmt):
    """SciPy outputs keep pattern entries that are numerically zero.

    The structure of the returned sparse array matches the detected
    sparsity pattern independent of the evaluation point,
    so downstream sparse solvers see a fixed structure.
    """

    def f(x):
        return x**2

    # The derivative 2*x is numerically zero at x[0] = 0,
    # but the diagonal pattern entry must survive.
    x = np.array([0.0, 1.0, 2.0])
    result = jacobian(f, x, output_format=fmt)(x)

    assert result.nnz == 3
    assert_allclose(result.toarray(), np.diag(2 * x))


@pytest.mark.hessian
@pytest.mark.parametrize("fmt", ["scipy_coo", "scipy_csr", "scipy_csc"])
def test_scipy_hessian_preserves_structural_zeros(fmt):
    """SciPy Hessian outputs keep pattern entries that are numerically zero."""

    def f(x):
        return jnp.sum(x**3)

    # The second derivative 6*x is numerically zero at x[0] = 0,
    # but the diagonal pattern entry must survive.
    x = np.array([0.0, 1.0, 2.0])
    result = hessian(f, x, output_format=fmt)(x)

    assert result.nnz == 3
    assert_allclose(result.toarray(), np.diag(6 * x))


@pytest.mark.jacobian
def test_jacobian_numpy_dense_pytree(assert_trees_allclose):
    """numpy_dense supports PyTree inputs and outputs, returning numpy blocks."""

    def f(params):
        return {"out": params["a"] * params["b"], "sum": jnp.sum(params["a"])}

    params = {"a": np.array([1.0, 2.0]), "b": np.array([3.0, 4.0])}
    result = jacobian(f, params, output_format="numpy_dense")(params)
    expected = jax.jacobian(f)(params)

    assert all(isinstance(leaf, np.ndarray) for leaf in jax.tree.leaves(result))
    assert_trees_allclose(result, expected, rtol=1e-5)


@pytest.mark.hessian
def test_hessian_numpy_dense_pytree(assert_trees_allclose):
    """numpy_dense supports PyTree inputs for Hessians, returning numpy blocks."""

    def f(params):
        return jnp.sum(params["a"] ** 2) * jnp.sum(params["b"])

    params = {"a": np.array([1.0, 2.0]), "b": np.array([3.0])}
    result = hessian(f, params, output_format="numpy_dense")(params)
    expected = jax.hessian(f)(params)

    assert all(isinstance(leaf, np.ndarray) for leaf in jax.tree.leaves(result))
    assert_trees_allclose(result, expected, rtol=1e-5)
