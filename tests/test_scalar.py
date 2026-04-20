"""Jacobian and Hessian sparsity tests for scalar functions f: R^n -> R.

For a unary function f: R -> R:
- The Jacobian is [[1]] if f depends on its input, [[0]] otherwise.
- The Hessian is [[1]] if f has a nonzero second derivative, [[0]] otherwise.

For a binary function f: R^2 -> R:
- The Jacobian is [[1, 1]] if f depends on both inputs.
- The Hessian is 2x2 with entries indicating second-order interactions.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import hessian_sparsity, jacobian_sparsity

_1 = np.array([[1]])
_0 = np.array([[0]])


def _jac(f, n=1):
    """Compute the dense Jacobian sparsity pattern for a scalar function."""
    return jacobian_sparsity(f, n).todense().astype(int)


def _hes(f, n=1):
    """Compute the dense Hessian sparsity pattern for a scalar function."""

    def scalar_f(x):
        return jnp.squeeze(f(x))

    return hessian_sparsity(scalar_f, n).todense().astype(int)


# Nonlinear unary functions: J = [[1]], H = [[1]]

_NONLINEAR = [
    pytest.param(jnp.sin, id="sin"),
    pytest.param(jnp.cos, id="cos"),
    pytest.param(jnp.tan, id="tan"),
    pytest.param(jnp.exp, id="exp"),
    pytest.param(jnp.exp2, id="exp2"),
    pytest.param(jnp.expm1, id="expm1"),
    pytest.param(jnp.sinh, id="sinh"),
    pytest.param(jnp.cosh, id="cosh"),
    pytest.param(jnp.tanh, id="tanh"),
    pytest.param(jax.nn.sigmoid, id="sigmoid"),
    pytest.param(jax.nn.softplus, id="softplus"),
    pytest.param(lambda x: jnp.log(x + 2), id="log"),
    pytest.param(lambda x: jnp.log1p(x + 2), id="log1p"),
    pytest.param(lambda x: jnp.sqrt(jnp.abs(x) + 1), id="sqrt"),
    pytest.param(lambda x: jnp.cbrt(x + 1), id="cbrt"),
    pytest.param(lambda x: jax.lax.rsqrt(jnp.abs(x) + 1), id="rsqrt"),
    pytest.param(lambda x: jnp.arcsin(x * 0.5), id="arcsin"),
    pytest.param(lambda x: jnp.arccos(x * 0.5), id="arccos"),
    pytest.param(jnp.arctan, id="arctan"),
    pytest.param(jnp.arcsinh, id="arcsinh"),
    pytest.param(lambda x: jnp.arccosh(x + 2), id="arccosh"),
    pytest.param(lambda x: jnp.arctanh(x * 0.5), id="arctanh"),
    pytest.param(lambda x: x**2, id="square"),
    pytest.param(lambda x: x**3, id="cube"),
    pytest.param(lambda x: x ** (2.0 / 3.0), id="pow_frac"),
    pytest.param(lambda x: 1.0 / (x + 2), id="reciprocal"),
    pytest.param(lambda x: jnp.log(jnp.abs(x) + 1), id="log_abs"),
]


@pytest.mark.elementwise
@pytest.mark.parametrize("f", _NONLINEAR)
def test_nonlinear_jacobian(f):
    """Nonlinear scalar function has Jacobian [[1]]."""
    np.testing.assert_array_equal(_jac(f), _1)


@pytest.mark.hessian
@pytest.mark.parametrize("f", _NONLINEAR)
def test_nonlinear_hessian(f):
    """Nonlinear scalar function has Hessian [[1]]."""
    np.testing.assert_array_equal(_hes(f), _1)


# Linear functions: J = [[1]], H = [[0]]

_LINEAR = [
    pytest.param(lambda x: x, id="identity"),
    pytest.param(lambda x: -x, id="negate"),
    pytest.param(lambda x: 2.0 * x, id="scale"),
    pytest.param(lambda x: x + 1.0, id="shift"),
    pytest.param(lambda x: x / 3.0, id="div_const"),
    pytest.param(jnp.abs, id="abs"),
    pytest.param(lambda x: jnp.clip(x, -1.0, 1.0), id="clip"),
]


@pytest.mark.elementwise
@pytest.mark.parametrize("f", _LINEAR)
def test_linear_jacobian(f):
    """Linear (or piecewise linear) scalar function has Jacobian [[1]]."""
    np.testing.assert_array_equal(_jac(f), _1)


@pytest.mark.hessian
@pytest.mark.parametrize("f", _LINEAR)
def test_linear_hessian(f):
    """Linear (or piecewise linear) scalar function has Hessian [[0]]."""
    np.testing.assert_array_equal(_hes(f), _0)


# Zero-derivative functions: J = [[0]], H = [[0]]

_ZERO_DERIV = [
    pytest.param(jnp.sign, id="sign"),
    pytest.param(jnp.floor, id="floor"),
    pytest.param(jnp.ceil, id="ceil"),
    pytest.param(jnp.round, id="round"),
    pytest.param(lambda x: x**0, id="pow_zero"),
]


@pytest.mark.elementwise
@pytest.mark.parametrize("f", _ZERO_DERIV)
def test_zero_deriv_jacobian(f):
    """Zero-derivative function has Jacobian [[0]]."""
    np.testing.assert_array_equal(_jac(f), _0)


@pytest.mark.hessian
@pytest.mark.parametrize("f", _ZERO_DERIV)
def test_zero_deriv_hessian(f):
    """Zero-derivative function has Hessian [[0]]."""
    np.testing.assert_array_equal(_hes(f), _0)


# Constant functions: J = [[0]], H = [[0]]

_CONSTANT = [
    pytest.param(lambda _: jnp.array(1.0), id="const"),
    pytest.param(lambda _: jnp.array(0.0), id="zero"),
]


@pytest.mark.elementwise
@pytest.mark.parametrize("f", _CONSTANT)
def test_constant_jacobian(f):
    """Constant function has Jacobian [[0]]."""
    np.testing.assert_array_equal(_jac(f), _0)


@pytest.mark.hessian
@pytest.mark.parametrize("f", _CONSTANT)
def test_constant_hessian(f):
    """Constant function has Hessian [[0]]."""
    np.testing.assert_array_equal(_hes(f), _0)


# Compositions: verifying sparsity propagation through chains


@pytest.mark.hessian
def test_composition_linear_of_nonlinear():
    """Linear(nonlinear(x)) preserves nonlinearity: H = [[1]]."""

    def f(x):
        return 2.0 * jnp.sin(x) + 1.0

    np.testing.assert_array_equal(_hes(f), _1)


@pytest.mark.hessian
def test_composition_nonlinear_of_linear():
    """Nonlinear(linear(x)) is nonlinear: H = [[1]]."""

    def f(x):
        return jnp.exp(3.0 * x + 1.0)

    np.testing.assert_array_equal(_hes(f), _1)


@pytest.mark.hessian
def test_composition_linear_chain():
    """Chain of linear functions is linear: H = [[0]]."""

    def f(x):
        return 2.0 * (3.0 * x + 1.0) - 5.0

    np.testing.assert_array_equal(_hes(f), _0)


@pytest.mark.elementwise
def test_composition_zero_of_nonlinear():
    """Zero-derivative(nonlinear(x)) kills dependency: J = [[0]]."""

    def f(x):
        return jnp.sign(jnp.sin(x))

    np.testing.assert_array_equal(_jac(f), _0)


@pytest.mark.hessian
def test_multiply_by_zero_hessian():
    """Multiplying by a known-zero constant kills Hessian dependency: 0*x^2.

    Since d(0 * x^2)/dx = 0,
    the mul handler clears deps and the Hessian is empty.
    """

    def f1(x):
        return 0.0 * x**2

    def f2(x):
        return x**2 * 0.0

    np.testing.assert_array_equal(_hes(f1), _0)
    np.testing.assert_array_equal(_hes(f2), _0)


# Binary functions f: R^2 -> R
#
# These test the fundamental Hessian building blocks:
# whether cross-terms (d²f/dx_i dx_j) appear correctly.

# Nonlinear binary: J = [[1, 1]], H has nonzero entries

_BINARY_NONLINEAR = [
    # (function, expected_hessian, id)
    # x*y: d²/dxdy = 1, d²/dx² = d²/dy² = 0
    pytest.param(
        lambda x: x[0] * x[1],
        np.array([[0, 1], [1, 0]]),
        id="mul",
    ),
    # x/y: d²/dx² = 0, d²/dxdy = -1/y², d²/dy² = 2x/y³
    pytest.param(
        lambda x: x[0] / (x[1] + 2),
        np.array([[0, 1], [1, 1]]),
        id="div",
    ),
    # x^y: all second derivatives nonzero
    pytest.param(
        lambda x: (x[0] + 2) ** x[1],
        np.array([[1, 1], [1, 1]]),
        id="pow",
    ),
    # atan2(y, x): all second derivatives nonzero
    pytest.param(
        lambda x: jnp.arctan2(x[0], x[1] + 1),
        np.array([[1, 1], [1, 1]]),
        id="atan2",
    ),
    # hypot(x, y) = sqrt(x² + y²): all second derivatives nonzero
    pytest.param(
        lambda x: jnp.sqrt(x[0] ** 2 + x[1] ** 2 + 1),
        np.array([[1, 1], [1, 1]]),
        id="hypot",
    ),
]


@pytest.mark.elementwise
@pytest.mark.parametrize(("f", "expected_hessian"), _BINARY_NONLINEAR)
def test_binary_nonlinear_jacobian(f, expected_hessian):
    """Nonlinear binary function depends on both inputs: J = [[1, 1]]."""
    np.testing.assert_array_equal(_jac(f, n=2), np.array([[1, 1]]))


@pytest.mark.hessian
@pytest.mark.parametrize(("f", "expected_hessian"), _BINARY_NONLINEAR)
def test_binary_nonlinear_hessian(f, expected_hessian):
    """Nonlinear binary function has specific 2x2 Hessian pattern."""
    np.testing.assert_array_equal(_hes(f, n=2), expected_hessian)


# Linear binary: J = [[1, 1]], H = zeros(2,2)

_BINARY_LINEAR = [
    pytest.param(lambda x: x[0] + x[1], id="add"),
    pytest.param(lambda x: x[0] - x[1], id="sub"),
    pytest.param(lambda x: jnp.minimum(x[0], x[1]), id="min"),
    pytest.param(lambda x: jnp.maximum(x[0], x[1]), id="max"),
    pytest.param(lambda x: 2.0 * x[0] + 3.0 * x[1], id="linear_combo"),
]


@pytest.mark.elementwise
@pytest.mark.parametrize("f", _BINARY_LINEAR)
def test_binary_linear_jacobian(f):
    """Linear binary function depends on both inputs: J = [[1, 1]]."""
    np.testing.assert_array_equal(_jac(f, n=2), np.array([[1, 1]]))


@pytest.mark.hessian
@pytest.mark.parametrize("f", _BINARY_LINEAR)
def test_binary_linear_hessian(f):
    """Linear binary function has zero Hessian."""
    np.testing.assert_array_equal(_hes(f, n=2), np.zeros((2, 2), dtype=int))
