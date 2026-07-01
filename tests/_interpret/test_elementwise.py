"""Tests for elementwise operation propagation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from asdex import jacobian_sparsity
from tests._utils import (
    assert_jacobian_sparsity_conservative,
    assert_jacobian_sparsity_exact,
)


@pytest.mark.array_ops
def test_constant_in_elementwise_op():
    """Constant array in binary elementwise operation preserves input structure.

    Adding a constant array to input doesn't change the sparsity pattern.
    """

    def f(x):
        const = jnp.array([1.0, 2.0, 3.0])
        return x + const

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    # Each output depends only on corresponding input (identity)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_zero_size_binary_elementwise():
    """Binary elementwise on size-0 arrays produces size-0 output."""

    def f(x):
        # Slicing to empty then adding exercises the size-0 binary path.
        a = x[:0]
        return a + a

    result = jacobian_sparsity(f, np.zeros(3))
    assert result.shape == (0, 3)
    assert result.nnz == 0


@pytest.mark.elementwise
def test_binary_broadcast_size1_dim():
    """Binary ops with size-1 broadcasting map dependencies correctly.

    For mul of (3,4) * (3,1) → (3,4),
    out[i,j] depends on in1[i,j] and in2[i,0].
    The flat modular indexing ``i % len`` gives wrong results here
    because it maps ``(i*4 + j) % 3`` instead of projecting coordinates.
    """
    weights = jnp.ones((3, 1))

    def f(x):
        mat = x.reshape(3, 4)
        return (mat * weights).reshape(-1)

    result = jacobian_sparsity(f, np.zeros(12)).todense().astype(int)
    # Each output depends only on its own input (weights are constant).
    expected = np.eye(12, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_binary_broadcast_leading_dim():
    """Broadcasting along the leading dimension tracks dependencies per row.

    For mul of (4,3) * (1,3) → (4,3),
    out[i,j] depends on in1[i,j] and in2[0,j].
    """
    scale = jnp.ones((1, 3))

    def f(x):
        mat = x.reshape(4, 3)
        return (mat * scale).reshape(-1)

    result = jacobian_sparsity(f, np.zeros(12)).todense().astype(int)
    expected = np.eye(12, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_binary_broadcast_dependent_operands():
    """Broadcasting with both operands depending on input tracks row dependencies.

    For mul of (3,4) * (3,1) where both sides depend on x,
    out[i,j] depends on all inputs in row i (block-diagonal 4x4 blocks).
    This catches the flat modular indexing bug that constant-operand tests miss.
    """

    def f(x):
        mat = x.reshape(2, 3)
        row_sums = mat.sum(axis=1, keepdims=True)  # (2,1), depends on x
        return (mat * row_sums).reshape(-1)

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    # Each output in row i depends on all 3 inputs in row i.
    # fmt: off
    expected = np.array([
        [1,1,1, 0,0,0],
        [1,1,1, 0,0,0],
        [1,1,1, 0,0,0],
        [0,0,0, 1,1,1],
        [0,0,0, 1,1,1],
        [0,0,0, 1,1,1],
    ], dtype=int)
    # fmt: on
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_erf():
    """Erf is a unary elementwise op that preserves per-element dependencies."""

    def f(x):
        return jax.lax.erf(x)

    result = jacobian_sparsity(f, np.zeros(4)).todense().astype(int)
    expected = np.eye(4, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_erfc():
    """Erfc (complementary error function) is unary elementwise with diagonal Jacobian."""

    def f(x):
        return jax.scipy.special.erfc(x)

    result = jacobian_sparsity(f, np.zeros(4)).todense().astype(int)
    expected = np.eye(4, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_erf_inv():
    """Inverse error function is unary elementwise with diagonal Jacobian."""

    def f(x):
        return jax.scipy.special.erfinv(x)

    # erfinv domain is (-1, 1)
    result = jacobian_sparsity(f, np.zeros(4)).todense().astype(int)
    expected = np.eye(4, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_digamma():
    """Digamma (psi function) is unary elementwise with diagonal Jacobian."""

    def f(x):
        return jax.scipy.special.digamma(x)

    result = jacobian_sparsity(f, np.ones(4)).todense().astype(int)
    expected = np.eye(4, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_lgamma():
    """Log-gamma function is unary elementwise with diagonal Jacobian."""

    def f(x):
        return jax.lax.lgamma(x)

    result = jacobian_sparsity(f, np.ones(4)).todense().astype(int)
    expected = np.eye(4, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_bessel_i0e():
    """Scaled Bessel I0 is unary elementwise with diagonal Jacobian."""

    def f(x):
        return jax.scipy.special.i0e(x)

    result = jacobian_sparsity(f, np.zeros(4)).todense().astype(int)
    expected = np.eye(4, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_bessel_i1e():
    """Scaled Bessel I1 is unary elementwise with diagonal Jacobian."""

    def f(x):
        return jax.scipy.special.i1e(x)

    result = jacobian_sparsity(f, np.ones(4)).todense().astype(int)
    expected = np.eye(4, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_polygamma():
    """Polygamma is elementwise in x with diagonal Jacobian.

    The order n is a literal parameter, not differentiated.
    """

    def f(x):
        return jax.scipy.special.polygamma(0, x)

    result = jacobian_sparsity(f, np.ones(4)).todense().astype(int)
    expected = np.eye(4, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_polygamma_variable_order():
    """Polygamma with variable order only depends on x, not on n.

    The order n has zero derivative (∂ψₙ/∂n = 0),
    so only the second input contributes to sparsity.
    """

    def f(x):
        n = x[0].astype(jnp.int32)
        return jax.scipy.special.polygamma(n, x[1])

    x = jnp.ones(2)
    result = jacobian_sparsity(f, x).todense().astype(int)
    J = jax.jacobian(f)(x)
    expected = (np.abs(J) > 1e-10).astype(int).reshape(result.shape)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_convert_element_type_propagates_const():
    """convert_element_type propagates const values for downstream gather.

    JAX inserts convert_element_type (int64 → int32) before gather.
    Without const propagation, the gather falls back to conservative.
    """
    indices = jnp.array([2, 0, 1])

    def f(x):
        return x[indices]

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    # out[0] <- x[2], out[1] <- x[0], out[2] <- x[1]
    expected = np.array(
        [
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


# Division zero-skipping


@pytest.mark.elementwise
def test_div_zero_numerator():
    """Division with zero numerator clears dependencies.

    d(0/y)/dy = 0, so output positions with known zero numerator
    have no dependency on any input.
    """
    numerator = jnp.array([0.0, 1.0, 0.0])

    def f(x):
        return numerator / x

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    # Only out[1] depends on x[1]; out[0] and out[2] are zero.
    expected = np.array(
        [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_div_zero_numerator_broadcast():
    """Scalar zero numerator divided by a vector clears all dependencies.

    Broadcasting a scalar zero numerator to the output shape
    should clear all output index sets.
    """

    def f(x):
        return jnp.float32(0.0) / x

    result = jacobian_sparsity(f, np.zeros(4)).todense().astype(int)
    expected = np.zeros((4, 4), dtype=int)
    np.testing.assert_array_equal(result, expected)


# Integer power zero-skipping


@pytest.mark.elementwise
def test_integer_pow_zero_base():
    """Zero base with exponent > 1 clears dependencies.

    d(0^n)/dx = n * 0^(n-1) = 0 for n > 1,
    so output positions with known zero base have no dependencies.
    """
    base = jnp.array([0.0, 1.0, 0.0])

    def f(_x):
        return jax.lax.integer_pow(base, 2)

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    # All outputs are constants (no dependency on input).
    expected = np.zeros((3, 3), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_integer_pow_zero_base_exp_zero():
    """x^0 = 1 always, so no dependencies regardless of base.

    This tests the existing n=0 special case.
    """

    def f(x):
        return jax.lax.integer_pow(x, 0)

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.zeros((3, 3), dtype=int)
    np.testing.assert_array_equal(result, expected)


# Bounds propagation through mul, div, integer_pow


@pytest.mark.elementwise
def test_mul_bounds_propagate_to_dynamic_slice():
    """Bounds from argmax flow through mul to dynamic_slice.

    argmax(x[:2]) ∈ {0,1}, so idx*2 has interval bounds [0,2].
    dynamic_slice enumerates all integer start positions in [0,2].
    argmax has zero derivative, so it contributes no index set deps.
    """

    def f(x):
        idx = jnp.argmax(x[:2])  # bounds: [0, 1]
        scaled = idx * 2  # bounds: [0, 2] via mul
        return lax.dynamic_slice(x, (scaled,), (2,))

    result = jacobian_sparsity(f, np.zeros(5)).todense().astype(int)
    # Interval [0,2] means windows at start=0, 1, 2.
    # out[0] = x[0] ∪ x[1] ∪ x[2], out[1] = x[1] ∪ x[2] ∪ x[3].
    expected = np.array(
        [
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_div_bounds_propagate_to_dynamic_slice():
    """Bounds from div propagate to dynamic_slice via lax.div.

    Uses lax.div directly (not ``//``, which lowers to a nested jaxpr
    with select_n that doesn't yet merge bounds from both branches).
    argmax(x[:4]) ∈ {0,1,2,3}, lax.div(idx, 2) ∈ {0,1}.
    dynamic_slice enumerates start positions {0,1}.
    """

    def f(x):
        idx = jnp.argmax(x[:4])  # bounds: [0, 3]
        start = lax.div(idx, jnp.asarray(2, dtype=idx.dtype))  # bounds: [0, 1] via div
        return lax.dynamic_slice(x, (start,), (3,))

    result = jacobian_sparsity(f, np.zeros(5)).todense().astype(int)
    # Interval [0,1] means windows at start=0, 1.
    # out[0] = x[0] ∪ x[1], out[1] = x[1] ∪ x[2], out[2] = x[2] ∪ x[3].
    expected = np.array(
        [
            [1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_integer_pow_even_bounds_propagate_to_dynamic_slice():
    """Even power bounds from integer_pow flow to dynamic_slice.

    argmax(x[:2]) ∈ {0,1}, so idx**2 ∈ [0,1] (even power).
    dynamic_slice enumerates start positions {0,1}.
    argmax has zero derivative, so it contributes no index set deps.
    """

    def f(x):
        idx = jnp.argmax(x[:2])  # bounds: [0, 1]
        start = jax.lax.integer_pow(idx, 2)  # bounds: [0, 1]
        return lax.dynamic_slice(x, (start,), (3,))

    result = jacobian_sparsity(f, np.zeros(5)).todense().astype(int)
    # Windows at start=0 and start=1.
    # out[0] = x[0] ∪ x[1], out[1] = x[1] ∪ x[2], out[2] = x[2] ∪ x[3].
    expected = np.array(
        [
            [1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_integer_pow_odd_bounds_propagate_to_dynamic_slice():
    """Odd power preserves monotone bounds through to dynamic_slice.

    argmax(x[:2]) ∈ {0,1}, so idx**3 ∈ [0,1] (odd power, monotone).
    argmax has zero derivative, so it contributes no index set deps.
    """

    def f(x):
        idx = jnp.argmax(x[:2])  # bounds: [0, 1]
        start = jax.lax.integer_pow(idx, 3)  # bounds: [0, 1]
        return lax.dynamic_slice(x, (start,), (3,))

    result = jacobian_sparsity(f, np.zeros(5)).todense().astype(int)
    # Same as even power: windows at 0 and 1.
    expected = np.array(
        [
            [1, 1, 0, 0, 0],
            [0, 1, 1, 0, 0],
            [0, 0, 1, 1, 0],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_div_bounds_skip_zero_crossing_divisor():
    """Division bounds are not propagated when divisor spans zero.

    When the divisor range includes zero, interval division is undefined,
    so bounds should not be propagated and the consumer falls back to conservative.
    argmax(x[:3]) ∈ {0,1,2}, so idx-1 ∈ {-1,0,1} which spans zero.
    lax.div(6, idx-1) is undefined at zero, so bounds are dropped.
    Without bounds, dynamic_slice falls back to conservative (all deps).
    """

    def f(x):
        idx = jnp.argmax(x[:3])  # bounds: [0, 2]
        divisor = idx - jnp.asarray(1, dtype=idx.dtype)  # bounds: [-1, 1] — spans zero
        start = lax.div(jnp.asarray(6, dtype=divisor.dtype), divisor)  # bounds dropped
        return lax.dynamic_slice(x, (start,), (2,))

    result = jacobian_sparsity(f, np.zeros(5)).todense().astype(int)
    # All 1s: conservative fallback since div bounds span zero.
    expected = np.ones((2, 5), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_mul_zero_second_operand():
    """Mul clears deps when the second operand is a known zero.

    Exercises the in2_val == 0 branch (vs test_binary_broadcast_size1_dim
    which uses constant ones).
    """
    mask = jnp.array([1.0, 0.0, 1.0])

    def f(x):
        return x * mask

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    # out[1] has no deps because mask[1] == 0.
    expected = np.array(
        [
            [1, 0, 0],
            [0, 0, 0],
            [0, 0, 1],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_integer_pow_zero_bounds():
    """integer_pow with y=0 propagates bounds (1, 1).

    x^0 = 1 always, so bounds are exactly (1, 1).
    When this feeds into a downstream add,
    the resulting bounds should be [1+lo, 1+hi].
    This exercises the y==0 branch in _propagate_bounds_integer_pow.
    """

    def f(x):
        idx = jnp.argmax(x[:3])  # bounds: [0, 2]
        one = jax.lax.integer_pow(idx, 0)  # bounds: [1, 1]
        start = one - jnp.int32(1)  # bounds: [0, 0] — constant 0
        return lax.dynamic_slice(x, (start,), (2,))

    result = jacobian_sparsity(f, np.zeros(5)).todense().astype(int)
    # Start is always 0, so out = [x[0], x[1]].
    expected = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


# Parametrized tests verifying detected sparsity matches numerical Jacobian


@pytest.mark.elementwise
@pytest.mark.parametrize(
    "op",
    [
        jnp.negative,  # ∂(-x)/∂x = -1
        jnp.exp,  # ∂eˣ/∂x = eˣ
        jnp.sin,  # ∂sin(x)/∂x = cos(x)
        jnp.cos,  # ∂cos(x)/∂x = -sin(x)
        jnp.tan,  # ∂tan(x)/∂x = sec²(x)
        jnp.sinh,  # ∂sinh(x)/∂x = cosh(x)
        jnp.cosh,  # ∂cosh(x)/∂x = sinh(x)
        jnp.tanh,  # ∂tanh(x)/∂x = sech²(x)
        jnp.arctan,  # ∂arctan(x)/∂x = 1/(1+x²)
        jnp.arcsinh,  # ∂arcsinh(x)/∂x = 1/√(x²+1)
        jnp.log1p,  # ∂log(1+x)/∂x = 1/(1+x)
        jnp.expm1,  # ∂(eˣ-1)/∂x = eˣ
        jnp.cbrt,  # ∂x^(1/3)/∂x = 1/(3x^(2/3))
        jnp.exp2,  # ∂2ˣ/∂x = 2ˣ·ln(2)
        jax.nn.sigmoid,  # ∂σ(x)/∂x = σ(x)(1-σ(x))
        jnp.square,  # ∂x²/∂x = 2x
        lax.erf,  # ∂erf(x)/∂x = 2e^(-x²)/√π
    ],
)
def test_unary_any_input(op):
    """Unary elementwise ops on R with nonzero derivative almost everywhere."""
    x = jax.random.normal(jax.random.key(0), (4,))
    assert_jacobian_sparsity_exact(op, x)


@pytest.mark.elementwise
@pytest.mark.parametrize(
    "op",
    [
        jnp.log,  # ∂log(x)/∂x = 1/x
        jnp.sqrt,  # ∂√x/∂x = 1/(2√x)
        lax.rsqrt,  # ∂(1/√x)/∂x = -1/(2x^(3/2))
        lax.lgamma,  # ∂log(Γ(x))/∂x = ψ(x)
        jax.scipy.special.digamma,  # ∂ψ(x)/∂x = ψ₁(x)
    ],
)
def test_unary_positive_input(op):
    """Unary elementwise ops on R+ with nonzero derivative."""
    x = jnp.abs(jax.random.normal(jax.random.key(0), (4,))) + 0.1
    assert_jacobian_sparsity_exact(op, x)


@pytest.mark.elementwise
@pytest.mark.parametrize(
    "op",
    [
        jnp.arcsin,  # ∂arcsin(x)/∂x = 1/√(1-x²)
        jnp.arccos,  # ∂arccos(x)/∂x = -1/√(1-x²)
    ],
)
def test_unary_bounded_input(op):
    """Unary elementwise ops on [-1,1] with nonzero derivative in interior."""
    x = jnp.tanh(jax.random.normal(jax.random.key(0), (4,)))  # maps to (-1, 1)
    assert_jacobian_sparsity_exact(op, x)


@pytest.mark.elementwise
@pytest.mark.parametrize(
    "op",
    [
        jnp.arctanh,  # ∂arctanh(x)/∂x = 1/(1-x²)
        jax.scipy.special.erfinv,  # ∂erf⁻¹(x)/∂x = (√π/2)·exp(erf⁻¹(x)²)
    ],
)
def test_unary_open_interval_input(op):
    """Unary elementwise ops on (-1,1) with nonzero derivative."""
    x = 0.9 * jnp.tanh(
        jax.random.normal(jax.random.key(0), (4,))
    )  # maps to (-0.9, 0.9)
    assert_jacobian_sparsity_exact(op, x)


@pytest.mark.elementwise
def test_unary_arccosh():
    """∂arccosh(x)/∂x = 1/√(x²-1), defined for x > 1."""
    x = jnp.abs(jax.random.normal(jax.random.key(0), (4,))) + 1.1
    assert_jacobian_sparsity_exact(jnp.arccosh, x)


@pytest.mark.elementwise
def test_unary_abs():
    """∂|x|/∂x = sign(x), nonzero away from x=0."""
    x = jax.random.normal(jax.random.key(0), (4,))
    x = jnp.where(jnp.abs(x) < 0.1, 0.5, x)  # avoid zero where derivative undefined
    assert_jacobian_sparsity_exact(jnp.abs, x)


@pytest.mark.elementwise
def test_unary_conj():
    """∂conj(z)/∂z = 1 (Wirtinger derivative)."""
    x = jax.random.normal(jax.random.key(0), (4,)) + 1j * jax.random.normal(
        jax.random.key(1), (4,)
    )
    assert_jacobian_sparsity_exact(jnp.conj, x, holomorphic=True)


@pytest.mark.elementwise
@pytest.mark.parametrize(
    "op",
    [
        jnp.real,  # ∂Re(z)/∂z = 1/2
        jnp.imag,  # ∂Im(z)/∂z = -i/2
    ],
)
def test_unary_real_imag(op):
    """Real/imag projections (Wirtinger derivatives)."""
    x = jax.random.normal(jax.random.key(0), (4,)) + 1j * jax.random.normal(
        jax.random.key(1), (4,)
    )
    assert_jacobian_sparsity_exact(op, x)


@pytest.mark.elementwise
def test_unary_copy():
    """∂copy(x)/∂x = 1 (identity)."""
    x = jax.random.normal(jax.random.key(0), (4,))
    assert_jacobian_sparsity_exact(lax.copy_p.bind, x)


@pytest.mark.elementwise
@pytest.mark.parametrize(
    "op",
    [
        jax.scipy.special.i0e,  # ∂(I₀(x)e^(-|x|))/∂x = (I₁ - sign(x)I₀)e^(-|x|)
        jax.scipy.special.i1e,  # ∂(I₁(x)e^(-|x|))/∂x = ((I₀+I₂)/2 - sign(x)I₁)e^(-|x|)
    ],
)
def test_unary_bessel(op):
    """Scaled Bessel functions with nonzero derivative."""
    x = jax.random.normal(jax.random.key(0), (4,))
    assert_jacobian_sparsity_exact(op, x)


# Zero-derivative primitives (piecewise constant, ∂f/∂x = 0 a.e.)


@pytest.mark.elementwise
@pytest.mark.parametrize(
    "op",
    [
        jnp.floor,  # ∂⌊x⌋/∂x = 0
        jnp.ceil,  # ∂⌈x⌉/∂x = 0
        jnp.sign,  # ∂sign(x)/∂x = 0
    ],
)
def test_zero_derivative(op):
    """Piecewise constant ops with zero derivative almost everywhere."""
    x = jax.random.normal(jax.random.key(0), (4,))
    assert_jacobian_sparsity_exact(op, x)


@pytest.mark.elementwise
def test_round():
    """∂round(x)/∂x = 0 (piecewise constant)."""
    x = jax.random.normal(jax.random.key(0), (4,))
    assert_jacobian_sparsity_exact(jnp.round, x)


# Binary elementwise primitives


@pytest.mark.elementwise
@pytest.mark.parametrize(
    "op",
    [
        jnp.maximum,  # ∂max(x,y)/∂x = 1 if x>y else 0, ∂max/∂y = 1 if y>x else 0
        jnp.minimum,  # ∂min(x,y)/∂x = 1 if x<y else 0, ∂min/∂y = 1 if y<x else 0
    ],
)
def test_binary_minmax(op):
    """max/min subgradients: winner gets 1, loser gets 0.

    Sparsity detection doesn't know which will win,
    so it conservatively marks both as dependencies.
    """
    x = jax.random.normal(jax.random.key(0), (4,))
    y = jax.random.normal(jax.random.key(1), (4,))

    def f(inputs):
        a, b = inputs[:4], inputs[4:]
        return op(a, b)

    inputs = jnp.concatenate([x, y])
    assert_jacobian_sparsity_conservative(f, inputs)


@pytest.mark.elementwise
def test_binary_power():
    """∂(x^y)/∂x = y·x^(y-1), ∂(x^y)/∂y = x^y·ln(x)."""
    base = jnp.abs(jax.random.normal(jax.random.key(0), (4,))) + 0.1
    exp = jax.random.normal(jax.random.key(1), (4,))

    def f(inputs):
        a, b = inputs[:4], inputs[4:]
        return jnp.power(a, b)

    inputs = jnp.concatenate([base, exp])
    assert_jacobian_sparsity_exact(f, inputs)


@pytest.mark.elementwise
def test_binary_arctan2():
    """∂atan2(y,x)/∂y = x/(x²+y²), ∂atan2(y,x)/∂x = -y/(x²+y²)."""
    y = jax.random.normal(jax.random.key(0), (4,))
    x = jax.random.normal(jax.random.key(1), (4,))
    x = jnp.where(jnp.abs(x) < 0.1, 0.5, x)  # avoid both being zero

    def f(inputs):
        a, b = inputs[:4], inputs[4:]
        return jnp.arctan2(a, b)

    inputs = jnp.concatenate([y, x])
    assert_jacobian_sparsity_exact(f, inputs)


@pytest.mark.elementwise
def test_binary_remainder():
    """∂(x mod y)/∂x = 1, ∂(x mod y)/∂y = -⌊x/y⌋.

    When ⌊x/y⌋ = 0, the derivative wrt y is zero at that point.
    Sparsity detection doesn't know this, so it conservatively marks both.
    """
    dividend = jax.random.normal(jax.random.key(0), (4,))
    divisor = jax.random.normal(jax.random.key(1), (4,))
    divisor = jnp.where(jnp.abs(divisor) < 0.1, 0.5, divisor)  # avoid zero

    def f(inputs):
        a, b = inputs[:4], inputs[4:]
        return jnp.remainder(a, b)

    inputs = jnp.concatenate([dividend, divisor])
    assert_jacobian_sparsity_conservative(f, inputs)


@pytest.mark.elementwise
@pytest.mark.parametrize(
    "op",
    [
        jnp.power,  # ∂(x^y)/∂x = y·x^(y-1)
        jnp.arctan2,  # ∂atan2(y,x)/∂y = x/(x²+y²)
    ],
)
def test_binary_first_arg_active(op):
    """Binary op with first argument active, second constant."""
    x = jnp.abs(jax.random.normal(jax.random.key(0), (4,))) + 0.1
    const = jnp.array([1.0, 2.0, 0.5, 1.5])

    def f(x):
        return op(x, const)

    assert_jacobian_sparsity_exact(f, x)


@pytest.mark.elementwise
@pytest.mark.parametrize(
    "op",
    [
        jnp.power,  # ∂(x^y)/∂y = x^y·ln(x)
        jnp.arctan2,  # ∂atan2(y,x)/∂x = -y/(x²+y²)
    ],
)
def test_binary_second_arg_active(op):
    """Binary op with first argument constant, second active."""
    const = jnp.array([2.0, 1.5, 3.0, 0.5])
    x = jnp.abs(jax.random.normal(jax.random.key(0), (4,))) + 0.1

    def f(x):
        return op(const, x)

    assert_jacobian_sparsity_exact(f, x)


@pytest.mark.elementwise
@pytest.mark.parametrize(
    "op",
    [
        jnp.maximum,  # ∂max(x,y)/∂x = 1 if x>y else 0
        jnp.minimum,  # ∂min(x,y)/∂x = 1 if x<y else 0
    ],
)
def test_binary_minmax_first_arg_active(op):
    """max/min with first argument active, second constant.

    Detection is conservative — it doesn't know which argument wins.
    """
    x = jax.random.normal(jax.random.key(0), (4,))
    const = jnp.array([0.0, 0.0, 0.0, 0.0])

    def f(x):
        return op(x, const)

    assert_jacobian_sparsity_conservative(f, x)


@pytest.mark.elementwise
@pytest.mark.parametrize(
    "op",
    [
        jnp.maximum,  # ∂max(x,y)/∂y = 1 if y>x else 0
        jnp.minimum,  # ∂min(x,y)/∂y = 1 if y<x else 0
    ],
)
def test_binary_minmax_second_arg_active(op):
    """max/min with first argument constant, second active.

    Detection is conservative — it doesn't know which argument wins.
    """
    const = jnp.array([0.0, 0.0, 0.0, 0.0])
    x = jax.random.normal(jax.random.key(0), (4,))

    def f(x):
        return op(const, x)

    assert_jacobian_sparsity_conservative(f, x)


# Clamp


@pytest.mark.elementwise
def test_clamp_sparsity():
    """Clamp propagates dependencies from x (see _prop_clamp docstring)."""

    def f(x):
        return lax.clamp(1.5, x, 3.5)

    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    result = jacobian_sparsity(f, x).todense().astype(int)
    expected = np.eye(4, dtype=int)  # out[i] depends on x[i]
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_clamp_conservative():
    """Clamp is conservative: detected pattern covers numerical Jacobian.

    See _prop_clamp docstring for why detection is conservative here.
    """

    def f(x):
        return lax.clamp(1.5, x, 3.5)

    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    result = jacobian_sparsity(f, x).todense().astype(int)
    # Detected: diagonal. Actual: zeros at [0,0] and [3,3] (x out of bounds).
    expected = np.eye(4, dtype=int)
    np.testing.assert_array_equal(result, expected)
    assert_jacobian_sparsity_conservative(f, x)


@pytest.mark.elementwise
def test_clamp_variable_bounds():
    """Clamp with variable lo/hi bounds propagates dependencies from all operands.

    clamp(lo, x, hi) returns lo when x < lo, hi when x > hi, else x.
    All three operands can contribute to the output.
    """

    def f(x):
        # lo=x[0], value=x[1], hi=x[2]
        return lax.clamp(x[0], x[1], x[2]).reshape(1)

    x = jnp.array([0.0, 0.5, 1.0])
    result = jacobian_sparsity(f, x).todense().astype(int)
    expected = np.array([[1, 1, 1]])  # all three inputs can affect output
    np.testing.assert_array_equal(result, expected)
    assert_jacobian_sparsity_conservative(f, x)


@pytest.mark.elementwise
def test_clamp_variable_lo_bound():
    """Clamp with variable lower bound propagates from both lo and x."""

    def f(x):
        # lo=x[0], value=x[1], hi=constant
        return lax.clamp(x[0], x[1], 10.0).reshape(1)

    x = jnp.array([0.0, 0.5])
    result = jacobian_sparsity(f, x).todense().astype(int)
    expected = np.array([[1, 1]])  # both lo and x can affect output
    np.testing.assert_array_equal(result, expected)
    assert_jacobian_sparsity_conservative(f, x)


@pytest.mark.elementwise
def test_clamp_variable_hi_bound():
    """Clamp with variable upper bound propagates from both x and hi."""

    def f(x):
        # lo=constant, value=x[0], hi=x[1]
        return lax.clamp(0.0, x[0], x[1]).reshape(1)

    x = jnp.array([0.5, 1.0])
    result = jacobian_sparsity(f, x).todense().astype(int)
    expected = np.array([[1, 1]])  # both x and hi can affect output
    np.testing.assert_array_equal(result, expected)
    assert_jacobian_sparsity_conservative(f, x)
