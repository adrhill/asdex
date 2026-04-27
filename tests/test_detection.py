"""Core API tests and simple element-wise operations."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import hessian_sparsity, jacobian_sparsity

# Core API tests


@pytest.mark.elementwise
def test_simple_dependencies():
    """Test f(x) = [x0+x1, x1*x2, x2]."""

    def f(x):
        return jnp.array([x[0] + x[1], x[1] * x[2], x[2]])

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.array([[1, 1, 0], [0, 1, 1], [0, 0, 1]])
    np.testing.assert_array_equal(result, expected)


# Hessian sparsity tests


@pytest.mark.hessian
def test_hessian_linear():
    """Linear functions have zero Hessian."""

    def f(x):
        return x[0] + 2 * x[1] + 3 * x[2]

    H = hessian_sparsity(f, np.zeros(3)).todense()
    assert H.sum() == 0


@pytest.mark.hessian
def test_hessian_product():
    """f(x) = x[0] * x[1] has H[0,1] = H[1,0] != 0."""

    def f(x):
        return x[0] * x[1]

    H = hessian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = jnp.array([[0, 1, 0], [1, 0, 0], [0, 0, 0]])
    assert jnp.array_equal(H, expected)


@pytest.mark.hessian
def test_hessian_quadratic():
    """f(x) = x[0]^2 + x[1]^2 has diagonal Hessian."""

    def f(x):
        return x[0] ** 2 + x[1] ** 2

    H = hessian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = jnp.array([[1, 0, 0], [0, 1, 0], [0, 0, 0]])
    assert jnp.array_equal(H, expected)


@pytest.mark.hessian
def test_hessian_finite_difference():
    """f(x) = sum((x[1:] - x[:-1])^2) has tridiagonal Hessian."""

    def f(x):
        return ((x[1:] - x[:-1]) ** 2).sum()

    H = hessian_sparsity(f, np.zeros(5)).todense().astype(int)
    expected = np.array(
        [
            [1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1],
        ]
    )
    np.testing.assert_array_equal(H, expected)


@pytest.mark.elementwise
def test_complex_dependencies():
    """Test f(x) = [x0*x1 + sin(x2), x3, x0*x1*x3]."""

    def f(x):
        a = x[0] * x[1]
        b = jnp.sin(x[2])
        c = a + b
        return jnp.array([c, x[3], a * x[3]])

    result = jacobian_sparsity(f, np.zeros(4)).todense().astype(int)
    expected = np.array([[1, 1, 1, 0], [0, 0, 0, 1], [1, 1, 0, 1]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_diagonal_jacobian():
    """Test f(x) = x^2 (element-wise) produces diagonal sparsity."""

    def f(x):
        return x**2

    result = jacobian_sparsity(f, np.zeros(4)).todense().astype(int)
    expected = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_dense_jacobian():
    """Test f(x) = [sum(x), prod(x)] produces dense sparsity."""

    def f(x):
        return jnp.array([jnp.sum(x), jnp.prod(x)])

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.array([[1, 1, 1], [1, 1, 1]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_readme_example():
    """f(x) = [x1^2, 2*x1*x2^2, sin(x3)]."""

    def f(x):
        return jnp.array([x[0] ** 2, 2 * x[0] * x[1] ** 2, jnp.sin(x[2])])

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.array([[1, 0, 0], [1, 1, 0], [0, 0, 1]])
    np.testing.assert_array_equal(result, expected)


# Identity and constant functions


@pytest.mark.elementwise
def test_identity():
    """Identity function: f(x) = x."""

    def f(x):
        return x

    result = jacobian_sparsity(f, np.zeros(1)).todense().astype(int)
    expected = np.array([[1]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_constant():
    """Constant function: output doesn't depend on input."""

    def f(x):
        return jnp.array([1.0])

    result = jacobian_sparsity(f, np.zeros(1)).todense().astype(int)
    expected = np.array([[0]])
    np.testing.assert_array_equal(result, expected)


# Zero derivative operations


@pytest.mark.elementwise
def test_zero_derivative_ceil_round():
    """ceil/round have zero derivative: f(x) = [x1*x2, ceil(x1*x2), x1*round(x2)]."""

    def f(x):
        return jnp.array([x[0] * x[1], jnp.ceil(x[0] * x[1]), x[0] * jnp.round(x[1])])

    result = jacobian_sparsity(f, np.zeros(2)).todense().astype(int)
    # round(x2) has zero derivative, so x1*round(x2) only depends on x1
    expected = np.array([[1, 1], [0, 0], [1, 0]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_zero_derivative_floor():
    """Floor has zero derivative."""

    def f(x):
        return jnp.floor(x)

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.zeros((3, 3), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_zero_derivative_sign():
    """Sign has zero derivative."""

    def f(x):
        return jnp.sign(x)

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.zeros((3, 3), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_comparison_ops():
    """Comparison operators have zero derivative."""

    def f(x):
        return jnp.array(
            [
                (x[0] < x[1]).astype(float),
                (x[0] <= x[1]).astype(float),
                (x[0] > x[1]).astype(float),
                (x[0] >= x[1]).astype(float),
                (x[0] == x[1]).astype(float),
                (x[0] != x[1]).astype(float),
            ]
        )

    result = jacobian_sparsity(f, np.zeros(2)).todense().astype(int)
    expected = np.zeros((6, 2), dtype=int)
    np.testing.assert_array_equal(result, expected)


# Type conversion


@pytest.mark.elementwise
def test_type_conversion():
    """Type conversions preserve dependencies."""

    def f(x):
        return x.astype(jnp.float32)

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


# Power operations


@pytest.mark.elementwise
def test_power_operations():
    """Various power operations: x^e, e^x, etc."""

    def f(x):
        return jnp.array([x[0] ** 2.5, jnp.exp(x[1]), x[2] ** x[2]])

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_integer_pow_zero():
    """x^0 = 1 has no dependency on x."""

    def f(x):
        return x**0

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.zeros((3, 3), dtype=int)
    np.testing.assert_array_equal(result, expected)


# Clamp operations


@pytest.mark.elementwise
def test_clamp_scalar_bounds():
    """clamp(x, lo, hi) with scalar bounds preserves element structure."""

    def f(x):
        return jnp.clip(x, 0.0, 1.0)

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_clamp_variable_bounds():
    """clamp(x1, x2, x3) with variable bounds - all contribute."""

    def f(x):
        return jnp.array([jnp.clip(x[0], x[1], x[2])])

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.array([[1, 1, 1]])
    np.testing.assert_array_equal(result, expected)


# Binary operations


@pytest.mark.elementwise
def test_dot_product():
    """Dot product: dot(x[0:2], x[3:5])."""

    def f(x):
        return jnp.array([jnp.dot(x[:2], x[3:5])])

    result = jacobian_sparsity(f, np.zeros(5)).todense().astype(int)
    expected = np.array([[1, 1, 0, 1, 1]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_multiply_by_zero():
    """Multiplying by a known-zero constant produces empty dependencies.

    Since d(0 * x)/dx = 0,
    the handler clears deps at positions where a constant zero is detected.
    """

    def f1(x):
        return jnp.array([0 * x[0]])

    def f2(x):
        return jnp.array([x[0] * 0])

    result1 = jacobian_sparsity(f1, np.zeros(1)).todense().astype(int)
    result2 = jacobian_sparsity(f2, np.zeros(1)).todense().astype(int)
    expected = np.array([[0]])
    np.testing.assert_array_equal(result1, expected)
    np.testing.assert_array_equal(result2, expected)


@pytest.mark.elementwise
def test_binary_min_max():
    """Min and max operations."""

    def f(x):
        return jnp.array([jnp.minimum(x[0], x[1]), jnp.maximum(x[1], x[2])])

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.array([[1, 1, 0], [0, 1, 1]])
    np.testing.assert_array_equal(result, expected)


# Unary functions


@pytest.mark.elementwise
def test_unary_functions():
    """Various unary math functions preserve element structure."""

    def f(x):
        return jnp.array(
            [
                jnp.sin(x[0]),
                jnp.cos(x[1]),
                jnp.tan(x[2]),
                jnp.exp(x[0]),
                jnp.log(x[1] + 1),  # +1 to avoid log(0)
                jnp.sqrt(jnp.abs(x[2]) + 1),
                jnp.sinh(x[0]),
                jnp.cosh(x[1]),
                jnp.tanh(x[2]),
            ]
        )

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.array(
        [
            [1, 0, 0],  # sin(x0)
            [0, 1, 0],  # cos(x1)
            [0, 0, 1],  # tan(x2)
            [1, 0, 0],  # exp(x0)
            [0, 1, 0],  # log(x1+1)
            [0, 0, 1],  # sqrt(|x2|+1)
            [1, 0, 0],  # sinh(x0)
            [0, 1, 0],  # cosh(x1)
            [0, 0, 1],  # tanh(x2)
        ]
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_unary_inverse_trig():
    """Inverse trigonometric functions preserve element structure."""

    def f(x):
        return jnp.array(
            [
                jnp.arcsin(x[0]),
                jnp.arccos(x[1]),
                jnp.arctan(x[2]),
                jnp.arcsinh(x[0]),
                jnp.arccosh(x[1] + 2),
                jnp.arctanh(x[2] * 0.5),
            ]
        )

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.array(
        [
            [1, 0, 0],  # arcsin(x0)
            [0, 1, 0],  # arccos(x1)
            [0, 0, 1],  # arctan(x2)
            [1, 0, 0],  # arcsinh(x0)
            [0, 1, 0],  # arccosh(x1+2)
            [0, 0, 1],  # arctanh(x2*0.5)
        ]
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_unary_misc():
    """cbrt, rsqrt, square, exp2, logistic preserve element structure."""

    def f(x):
        return jnp.array(
            [
                jnp.cbrt(x[0]),
                jax.lax.rsqrt(jnp.abs(x[1]) + 1),
                jax.lax.square(x[2]),
                jnp.exp2(x[0]),
                jax.nn.sigmoid(x[1]),
            ]
        )

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.array(
        [
            [1, 0, 0],  # cbrt(x0)
            [0, 1, 0],  # rsqrt(|x1|+1)
            [0, 0, 1],  # square(x2)
            [1, 0, 0],  # exp2(x0)
            [0, 1, 0],  # sigmoid(x1)
        ]
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_binary_atan2_rem():
    """atan2 and remainder are binary element-wise ops."""

    def f(x):
        return jnp.array([jnp.arctan2(x[0], x[1]), jnp.remainder(x[1], x[2])])

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.array([[1, 1, 0], [0, 1, 1]])
    np.testing.assert_array_equal(result, expected)


# Multi-output Jacobian patterns


@pytest.mark.elementwise
def test_sincos_same_input():
    """Both sin and cos of the same input depend on that input."""

    def f(x):
        return jnp.array([jnp.sin(x[0]), jnp.cos(x[0])])

    result = jacobian_sparsity(f, np.zeros(1)).todense().astype(int)
    expected = np.array([[1], [1]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_vector_index_arithmetic():
    """Complex index arithmetic: ret[i] = x[i+1]^2 - x[i]^2 + x[n-1-i]^2 - x[n-i-2]^2."""

    def f(x):
        n = x.shape[0]
        out = [
            x[i + 1] ** 2 - x[i] ** 2 + x[n - 1 - i] ** 2 - x[n - i - 2] ** 2
            for i in range(n - 1)
        ]
        return jnp.array(out)

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    expected = np.array(
        [
            [1, 1, 0, 0, 1, 1],
            [0, 1, 1, 1, 1, 0],
            [0, 0, 1, 1, 0, 0],
            [0, 1, 1, 1, 1, 0],
            [1, 1, 0, 0, 1, 1],
        ]
    )
    np.testing.assert_array_equal(result, expected)


# Composite functions


@pytest.mark.elementwise
def test_composite_mixed_ops_jacobian():
    """All inputs contribute: x0 + x1*x2 + 1/x3 + x4."""

    def f(x):
        return jnp.array([x[0] + x[1] * x[2] + 1.0 / x[3] + 1.0 * x[4]])

    result = jacobian_sparsity(f, np.zeros(5)).todense().astype(int)
    expected = np.array([[1, 1, 1, 1, 1]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_composite_with_power_jacobian():
    """Composite with power term: x0 + x1*x2 + 1/x3 + x4 + x1^x4."""

    def f(x):
        foo = x[0] + x[1] * x[2] + 1.0 / x[3] + 1.0 * x[4]
        return jnp.array([foo + x[1] ** x[4]])

    result = jacobian_sparsity(f, np.zeros(5)).todense().astype(int)
    expected = np.array([[1, 1, 1, 1, 1]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_ampgo07_jacobian():
    """AMPGO07 benchmark: comparison, sin, log, abs composed together.

    The (x <= 0) comparison has zero derivative,
    so the entire expression depends on x.
    """

    def f(x):
        return jnp.array(
            [
                (x[0] <= 0).astype(float) * jnp.inf
                + jnp.sin(x[0])
                + jnp.sin(10.0 / 3.0 * x[0])
                + jnp.log(jnp.abs(x[0]))
                - 0.84 * x[0]
                + 3.0
            ]
        )

    result = jacobian_sparsity(f, np.zeros(1)).todense().astype(int)
    expected = np.array([[1]])
    np.testing.assert_array_equal(result, expected)


# Multi-variable Hessian patterns


@pytest.mark.hessian
def test_hessian_diff_cubic():
    """sum(diff(x).^3) has tridiagonal Hessian."""

    def f(x):
        d = x[1:] - x[:-1]
        return jnp.sum(d**3)

    H = hessian_sparsity(f, np.zeros(4)).todense().astype(int)
    expected = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
        ]
    )
    np.testing.assert_array_equal(H, expected)


@pytest.mark.hessian
def test_hessian_division():
    """Division creates second-order interactions: x0/x1 + x2 + 1/x3."""

    def f(x):
        return x[0] / x[1] + x[2] / 1.0 + 1.0 / x[3]

    H = hessian_sparsity(f, np.zeros(4)).todense().astype(int)
    expected = np.array(
        [
            [0, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
        ]
    )
    np.testing.assert_array_equal(H, expected)


@pytest.mark.hessian
def test_hessian_product_linear():
    """Products with constants are linear, only cross-terms are nonzero."""

    def f(x):
        return x[0] * x[1] + x[2] * 1.0 + 1.0 * x[3]

    H = hessian_sparsity(f, np.zeros(4)).todense().astype(int)
    expected = np.array(
        [
            [0, 1, 0, 0],
            [1, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    np.testing.assert_array_equal(H, expected)


@pytest.mark.hessian
def test_hessian_product_of_products():
    """(x0*x1)*(x2*x3) has all cross-terms nonzero except diagonal."""

    def f(x):
        return (x[0] * x[1]) * (x[2] * x[3])

    H = hessian_sparsity(f, np.zeros(4)).todense().astype(int)
    expected = np.array(
        [
            [0, 1, 1, 1],
            [1, 0, 1, 1],
            [1, 1, 0, 1],
            [1, 1, 1, 0],
        ]
    )
    np.testing.assert_array_equal(H, expected)


@pytest.mark.hessian
def test_hessian_product_of_sums():
    """(x0+x1)*(x2+x3) only has cross-group interactions."""

    def f(x):
        return (x[0] + x[1]) * (x[2] + x[3])

    H = hessian_sparsity(f, np.zeros(4)).todense().astype(int)
    expected = np.array(
        [
            [0, 0, 1, 1],
            [0, 0, 1, 1],
            [1, 1, 0, 0],
            [1, 1, 0, 0],
        ]
    )
    np.testing.assert_array_equal(H, expected)


@pytest.mark.hessian
def test_hessian_sum_squared():
    """(x0+x1+x2+x3)^2 has fully dense Hessian."""

    def f(x):
        return (x[0] + x[1] + x[2] + x[3]) ** 2

    H = hessian_sparsity(f, np.zeros(4)).todense().astype(int)
    expected = np.ones((4, 4), dtype=int)
    np.testing.assert_array_equal(H, expected)


@pytest.mark.hessian
def test_hessian_reciprocal_sum():
    """1/(x0+x1+x2+x3) has fully dense Hessian."""

    def f(x):
        return 1.0 / (x[0] + x[1] + x[2] + x[3])

    H = hessian_sparsity(f, np.zeros(4)).todense().astype(int)
    expected = np.ones((4, 4), dtype=int)
    np.testing.assert_array_equal(H, expected)


@pytest.mark.hessian
def test_hessian_subtraction_linear():
    """Subtraction is linear, zero Hessian."""

    def f(x):
        return (x[0] - x[1]) + (x[2] - 1.0) + (1.0 - x[3])

    H = hessian_sparsity(f, np.zeros(4)).todense().astype(int)
    expected = np.zeros((4, 4), dtype=int)
    np.testing.assert_array_equal(H, expected)


@pytest.mark.hessian
def test_hessian_composite_mixed_ops():
    """Composite: x0 + x1*x2 + 1/x3 + x4."""

    def f(x):
        return x[0] + x[1] * x[2] + 1.0 / x[3] + 1.0 * x[4]

    H = hessian_sparsity(f, np.zeros(5)).todense().astype(int)
    expected = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0],
        ]
    )
    np.testing.assert_array_equal(H, expected)


@pytest.mark.hessian
def test_hessian_composite_with_power():
    """Composite with power term: x0 + x1*x2 + 1/x3 + x4 + x1^x4."""

    def f(x):
        foo = x[0] + x[1] * x[2] + 1.0 / x[3] + 1.0 * x[4]
        return foo + x[1] ** x[4]

    H = hessian_sparsity(f, np.zeros(5)).todense().astype(int)
    expected = np.array(
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 0, 1],
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 1, 0, 0, 1],
        ]
    )
    np.testing.assert_array_equal(H, expected)


@pytest.mark.hessian
def test_hessian_ampgo07():
    """AMPGO07 benchmark: Hessian is nonzero due to sin, log."""

    def f(x):
        return (
            (x[0] <= 0).astype(float) * jnp.inf
            + jnp.sin(x[0])
            + jnp.sin(10.0 / 3.0 * x[0])
            + jnp.log(jnp.abs(x[0]))
            - 0.84 * x[0]
            + 3.0
        )

    H = hessian_sparsity(f, np.zeros(1)).todense().astype(int)
    expected = np.array([[1]])
    np.testing.assert_array_equal(H, expected)


@pytest.mark.hessian
def test_hessian_clamp_variable_bounds():
    """Clamp with variable bounds has zero Hessian (piecewise linear in each arg)."""

    def f(x):
        return jnp.clip(x[0], x[1], x[2])

    H = hessian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.zeros((3, 3), dtype=int)
    np.testing.assert_array_equal(H, expected)


@pytest.mark.hessian
def test_hessian_clamp_times_x0():
    """x0 * clamp(x0, x1, x2) creates cross-term interactions."""

    def f(x):
        return x[0] * jnp.clip(x[0], x[1], x[2])

    H = hessian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.array(
        [
            [1, 1, 1],
            [1, 0, 0],
            [1, 0, 0],
        ]
    )
    np.testing.assert_array_equal(H, expected)


@pytest.mark.hessian
def test_hessian_clamp_times_x1():
    """x1 * clamp(x0, x1, x2) creates cross-term interactions."""

    def f(x):
        return x[1] * jnp.clip(x[0], x[1], x[2])

    H = hessian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.array(
        [
            [0, 1, 0],
            [1, 1, 1],
            [0, 1, 0],
        ]
    )
    np.testing.assert_array_equal(H, expected)


@pytest.mark.hessian
def test_hessian_clamp_times_x2():
    """x2 * clamp(x0, x1, x2) creates cross-term interactions."""

    def f(x):
        return x[2] * jnp.clip(x[0], x[1], x[2])

    H = hessian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.array(
        [
            [0, 0, 1],
            [0, 0, 1],
            [1, 1, 1],
        ]
    )
    np.testing.assert_array_equal(H, expected)
