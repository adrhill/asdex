"""Tests for elementwise operation propagation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from asdex import jacobian_sparsity
from asdex.detection._interpret._common import _PropState
from asdex.detection._interpret._elementwise import _propagate_bounds_integer_pow
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
@pytest.mark.skipif(
    jax.__version_info__ < (0, 11, 2),
    reason="one_minus_square was added in JAX 0.11.2",
)
def test_one_minus_square():
    """One-minus-square is unary elementwise with diagonal Jacobian."""

    def f(x):
        return lax.one_minus_square(x)

    result = jacobian_sparsity(f, np.zeros(4)).todense().astype(int)
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
    argmax has zero derivative, so it contributes no index sets.
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
    argmax has zero derivative, so it contributes no index sets.
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
    argmax has zero derivative, so it contributes no index sets.
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
    Without bounds, dynamic_slice falls back to conservative (all index sets).
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
def test_max_bounds_propagate_to_dynamic_slice():
    """``max`` raises the lower end of an interval and keeps it bounded.

    argmax(x[:4]) ∈ {0,1,2,3}, so maximum(idx, 2) ∈ {2,3}.
    dynamic_slice enumerates start positions {2,3}.
    """

    def f(x):
        idx = jnp.argmax(x[:4])  # bounds: [0, 3]
        start = jnp.maximum(idx, 2)  # bounds: [2, 3] via max
        return lax.dynamic_slice(x, (start,), (2,))

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    expected = np.array(
        [
            [0, 0, 1, 1, 0, 0],  # out[0] = x[2] ∪ x[3]
            [0, 0, 0, 1, 1, 0],  # out[1] = x[3] ∪ x[4]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_min_bounds_propagate_to_dynamic_slice():
    """``min`` lowers the upper end of an interval and keeps it bounded.

    argmax(x[:4]) ∈ {0,1,2,3}, so minimum(idx, 1) ∈ {0,1}.
    dynamic_slice enumerates start positions {0,1}.
    """

    def f(x):
        idx = jnp.argmax(x[:4])  # bounds: [0, 3]
        start = jnp.minimum(idx, 1)  # bounds: [0, 1] via min
        return lax.dynamic_slice(x, (start,), (2,))

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    expected = np.array(
        [
            [1, 1, 0, 0, 0, 0],  # out[0] = x[0] ∪ x[1]
            [0, 1, 1, 0, 0, 0],  # out[1] = x[1] ∪ x[2]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_clip_preserves_bounds_through_gather():
    """``jnp.clip`` on an index keeps the index bounded.

    ``jnp.clip`` lowers to ``max`` followed by ``min``, not to ``clamp``,
    so both need bounds rules for the common
    "clip an index into range before using it" idiom to stay sparse.
    argmax(x[:4]) ∈ {0,1,2,3}, so clip(idx + 1, 0, 7) ∈ {1,2,3,4}.
    """

    def f(x):
        idx = jnp.argmax(x[:4])  # bounds: [0, 3]
        return jnp.array([x[jnp.clip(idx + 1, 0, 7)]])  # bounds: [1, 4]

    result = jacobian_sparsity(f, np.zeros(8)).todense().astype(int)
    # out[0] = x[1] ∪ x[2] ∪ x[3] ∪ x[4]
    expected = np.array([[0, 1, 1, 1, 1, 0, 0, 0]], dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_clip_tightens_bounds_through_gather():
    """A clip narrower than the incoming interval tightens it.

    argmax(x[:6]) ∈ {0,...,5}, but clip(idx, 2, 3) ∈ {2,3},
    so the gather reaches only two columns.
    """

    def f(x):
        idx = jnp.argmax(x[:6])  # bounds: [0, 5]
        return jnp.array([x[jnp.clip(idx, 2, 3)]])  # bounds: [2, 3]

    result = jacobian_sparsity(f, np.zeros(8)).todense().astype(int)
    # out[0] = x[2] ∪ x[3]
    expected = np.array([[0, 0, 1, 1, 0, 0, 0, 0]], dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_clamp_bounds_propagate_to_dynamic_slice():
    """``lax.clamp`` propagates bounds through its nested min/max form.

    argmax(x[:4]) ∈ {0,1,2,3}, so clamp(1, idx, 2) ∈ {1,2}.
    """

    def f(x):
        idx = jnp.argmax(x[:4])  # bounds: [0, 3]
        start = lax.clamp(1, idx, 2)  # bounds: [1, 2] via clamp
        return lax.dynamic_slice(x, (start,), (2,))

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    expected = np.array(
        [
            [0, 1, 1, 0, 0, 0],  # out[0] = x[1] ∪ x[2]
            [0, 0, 1, 1, 0, 0],  # out[1] = x[2] ∪ x[3]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
@pytest.mark.fallback
def test_clip_of_unbounded_index_is_conservative():
    """Clipping an index that has no bounds falls back to conservative.

    ``max`` and ``min`` only propagate bounds when *both* operands are bounded,
    so clipping an otherwise unknown index does not currently bound it.

    TODO(max/min): clip fixes the result to [0, 2] whatever the index is,
    so the precise pattern enumerates start positions {0,1,2}, giving
    rows [1, 1, 1, 0] and [0, 1, 1, 1] instead of the all-ones fallback.
    Reaching it means treating a missing operand interval as the dtype's
    full range, which is exact for integers but needs care for floats (NaN).
    """

    def f(x):
        # A comparison has zero derivative and no tracked bounds,
        # so the sum below is an integer of unknown magnitude.
        idx = jnp.sum((x > 0).astype(jnp.int32))
        return lax.dynamic_slice(x, (jnp.clip(idx, 0, 2),), (2,))

    result = jacobian_sparsity(f, np.zeros(4)).todense().astype(int)
    expected = np.ones((2, 4), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_clip_bounds_broadcast_scalar_against_array_index():
    """Scalar clip operands broadcast against an array-valued index.

    argmax over axis 1 gives a 2-element index array, each in {0,...,3},
    and the scalar clip narrows both to {1,2}.
    Exercises numpy broadcasting inside the max/min bounds rule,
    where the operand bounds have different shapes.
    """

    def f(x):
        idx = jnp.argmax(x.reshape(2, 4), axis=1)  # shape (2,), bounds: [0, 3]
        return x[jnp.clip(idx, 1, 2)]  # bounds: [1, 2]

    result = jacobian_sparsity(f, np.zeros(8)).todense().astype(int)
    expected = np.array(
        [
            [0, 1, 1, 0, 0, 0, 0, 0],  # out[0] = x[1] ∪ x[2]
            [0, 1, 1, 0, 0, 0, 0, 0],  # out[1] = x[1] ∪ x[2]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_clamp_bounds_with_inverted_operands():
    """``lax.clamp`` with lo > hi pins the result to hi.

    XLA computes ``min(max(x, lo), hi)``, so an inverted range collapses
    to hi rather than lo. The bounds rule evaluates the same nested form,
    giving the single-point interval [1, 1].
    """

    def f(x):
        idx = jnp.argmax(x[:4])  # bounds: [0, 3]
        start = lax.clamp(3, idx, 1)  # lo > hi, so bounds: [1, 1]
        return lax.dynamic_slice(x, (start,), (2,))

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    expected = np.array(
        [
            [0, 1, 0, 0, 0, 0],  # out[0] = x[1]
            [0, 0, 1, 0, 0, 0],  # out[1] = x[2]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_clip_to_single_point_bounds():
    """A clip with equal bounds collapses the interval to one value.

    The enumeration then has a single candidate,
    matching a statically known index.
    """

    def f(x):
        idx = jnp.argmax(x[:4])  # bounds: [0, 3]
        return jnp.array([x[jnp.clip(idx, 2, 2)]])  # bounds: [2, 2]

    result = jacobian_sparsity(f, np.zeros(8)).todense().astype(int)
    # out[0] = x[2]
    expected = np.array([[0, 0, 1, 0, 0, 0, 0, 0]], dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_clip_brings_index_back_under_enumeration_cap():
    """Clipping a wide interval makes bounded enumeration viable again.

    argmax over 100 elements has bounds [0, 99], which exceeds
    ``_MAX_ENUM_COMBINATIONS`` and falls back to conservative.
    Clipping to [0, 3] leaves 4 combinations, so the gather is precise.
    """

    def f_unclipped(x):
        return jnp.array([x[jnp.argmax(x)]])  # bounds: [0, 99], over the cap

    def f_clipped(x):
        return jnp.array([x[jnp.clip(jnp.argmax(x), 0, 3)]])  # bounds: [0, 3]

    unclipped = jacobian_sparsity(f_unclipped, np.zeros(100)).todense().astype(int)
    np.testing.assert_array_equal(unclipped, np.ones((1, 100), dtype=int))

    clipped = jacobian_sparsity(f_clipped, np.zeros(100)).todense().astype(int)
    expected = np.zeros((1, 100), dtype=int)
    expected[0, :4] = 1  # out[0] = x[0] ∪ x[1] ∪ x[2] ∪ x[3]
    np.testing.assert_array_equal(clipped, expected)


@pytest.mark.elementwise
def test_clip_beyond_operand_extent_clamps_starts():
    """Bounds pointing past the operand still clamp to valid start positions.

    clip(idx + 10, 5, 9) pins the start to 9,
    which dynamic_slice clamps to ``6 - 2 = 4``, matching lax semantics.
    """

    def f(x):
        idx = jnp.argmax(x[:4])  # bounds: [0, 3]
        start = jnp.clip(idx + 10, 5, 9)  # bounds: [9, 9], past the array
        return lax.dynamic_slice(x, (start,), (2,))

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    expected = np.array(
        [
            [0, 0, 0, 0, 1, 0],  # out[0] = x[4] after start clamping
            [0, 0, 0, 0, 0, 1],  # out[1] = x[5]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_clip_bounds_resolve_comparison():
    """Bounds surviving a clip let a comparison fold to a constant.

    clip(argmax(x[:4]), 0, 3) is provably below 8,
    so ``select_n`` picks the elementwise branch
    instead of unioning the dense reduction branch.
    """

    def f(x):
        idx = jnp.clip(jnp.argmax(x[:4]), 0, 3)  # bounds: [0, 3]
        return jnp.where(idx < 8, x * 2.0, jnp.sum(x) * jnp.ones_like(x))

    result = jacobian_sparsity(f, np.zeros(8)).todense().astype(int)
    np.testing.assert_array_equal(result, np.eye(8, dtype=int))


@pytest.mark.elementwise
def test_clip_negative_interval_through_max():
    """Intervals spanning negative values propagate through max correctly.

    argmax(x[:4]) - 3 has bounds [-3, 0],
    maximum(., -1) raises the lower end to [-1, 0],
    and the final +1 shifts it to the valid start range [0, 1].
    """

    def f(x):
        idx = jnp.argmax(x[:4]) - 3  # bounds: [-3, 0]
        start = jnp.maximum(idx, -1) + 1  # bounds: [0, 1]
        return lax.dynamic_slice(x, (start,), (2,))

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    expected = np.array(
        [
            [1, 1, 0, 0, 0, 0],  # out[0] = x[0] ∪ x[1]
            [0, 1, 1, 0, 0, 0],  # out[1] = x[1] ∪ x[2]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_clip_zero_size_array():
    """Clipping a zero-sized array produces an empty pattern without crashing."""

    def f(x):
        return jnp.clip(x[:0], 0.0, 1.0)

    pattern = jacobian_sparsity(f, np.zeros(8))
    assert pattern.shape == (0, 8)
    assert pattern.nnz == 0


@pytest.mark.elementwise
def test_clip_on_float_values_stays_exact():
    """Bounds propagation through max/min does not disturb ordinary float clipping.

    ``jnp.clip`` on a value (not an index) must keep the elementwise pattern.
    """

    def f(x):
        return jnp.clip(x * 2.0, -1.0, 1.0) + x

    assert_jacobian_sparsity_exact(f, np.linspace(-2.0, 2.0, 5))


@pytest.mark.elementwise
def test_mul_zero_second_operand():
    """Mul clears index sets when the second operand is a known zero.

    Exercises the in2_val == 0 branch (vs test_binary_broadcast_size1_dim
    which uses constant ones).
    """
    mask = jnp.array([1.0, 0.0, 1.0])

    def f(x):
        return x * mask

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    # out[1] has no index sets because mask[1] == 0.
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
def test_rem_integer_const_negative_dividend():
    """Integer rem const propagation follows lax.rem, which takes the dividend's sign.

    lax.rem(-4, 3) = -1, so the gather index is -1 + 2 = 1.
    np.remainder(-4, 3) = 2 would shift the index to 4,
    dropping the true dependency on x[1].
    The const chain sits in a cond branch
    because top-level arithmetic on concrete arrays
    is folded away during tracing.
    """

    def f(x):
        idx = jnp.array([-4], dtype=jnp.int32)

        def true_branch(ops):
            i, values = ops
            j = lax.rem(i, jnp.int32(3)) + jnp.int32(2)  # [-1] + 2 = [1]
            return values[j] * 1.0

        def false_branch(ops):
            _, values = ops
            return values[:1] * 0.0

        return lax.cond(x[0] > 0, true_branch, false_branch, (idx, x))

    result = jacobian_sparsity(f, np.zeros(5)).todense().astype(int)
    expected = np.array([[0, 1, 0, 0, 0]], dtype=int)  # out[0] <- x[1]
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_integer_pow_bounds_negative_odd_exponent():
    """Bounds through a negative odd exponent must not be inverted.

    x^(-3) is decreasing on positive inputs,
    so mapping (lo, hi) to (lo^y, hi^y) flips the interval.
    Propagated bounds must either be skipped
    or stay ordered and contain the true output values.
    """
    jaxpr = jax.make_jaxpr(lambda a: lax.integer_pow(a, -3))(jnp.zeros(1)).jaxpr
    eqn = jaxpr.eqns[0]

    state = _PropState(bounds={eqn.invars[0]: (np.array([2.0]), np.array([3.0]))})
    _propagate_bounds_integer_pow(eqn, -3, state)

    bounds = state.bounds.get(eqn.outvars[0])
    if bounds is not None:
        lo, hi = bounds
        assert np.all(lo <= hi)
        # True output values 3^-3 and 2^-3 must lie inside the bounds.
        assert np.all(lo <= 3.0**-3)
        assert np.all(hi >= 2.0**-3)


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


def _one_hot(n, position):
    """Input whose ``argmax`` is ``position``.

    The wrap-around tests below route ``argmax`` through overflowing integer ops,
    so which elements a slice reads depends on which position wins.
    Checking against JAX at every one-hot input covers every wrapped branch.
    """
    x = np.zeros(n)
    x[position] = 1.0
    return x


@pytest.mark.elementwise
@pytest.mark.parametrize(
    "narrow",
    [
        pytest.param(lambda i: jnp.maximum(i, jnp.int8(121)), id="max"),
        pytest.param(lambda i: jnp.clip(i, 121, 126), id="clip"),
        pytest.param(lambda i: lax.clamp(jnp.int8(121), i, jnp.int8(126)), id="clamp"),
    ],
)
def test_wrapped_add_bounds_are_dropped_before_narrowing(narrow):
    """An int8 sum that can overflow carries no bounds into max, clip, or clamp.

    argmax(x[:16]) + 120 wraps past 127 to -128,
    so its real values are {120..127} or {-128..-121}
    while unchecked interval arithmetic would claim [120, 135].
    Narrowing that interval would claim the single start 21,
    but JAX starts the slice anywhere in 21..26 or more.
    The overflowing sum must lose its bounds,
    so the slice falls back to a conservative pattern.
    """

    def f(x):
        i = jnp.argmax(x[:16]).astype(jnp.int8) + jnp.int8(120)
        start = narrow(i).astype(jnp.int32) - 100
        return lax.dynamic_slice(x, (start,), (1,))

    for position in range(16):
        assert_jacobian_sparsity_conservative(f, _one_hot(40, position))
    result = jacobian_sparsity(f, np.zeros(40)).todense().astype(int)
    # TODO(add): track wrapped values as a union of intervals.
    # The precise pattern reads x[21..27] only.
    np.testing.assert_array_equal(result, np.ones((1, 40), dtype=int))


@pytest.mark.elementwise
def test_wrapped_unsigned_add_bounds_are_dropped():
    """A uint8 sum that can overflow carries no bounds into ``min``.

    argmax(x[:16]) + 250 is {250..255} or {0..9} after wraparound,
    so minimum(., 5) can be any start in 0..5.
    """

    def f(x):
        i = jnp.argmax(x[:16]).astype(jnp.uint8) + jnp.uint8(250)
        start = jnp.minimum(i, jnp.uint8(5)).astype(jnp.int32)
        return lax.dynamic_slice(x, (start,), (1,))

    for position in range(16):
        assert_jacobian_sparsity_conservative(f, _one_hot(20, position))
    result = jacobian_sparsity(f, np.zeros(20)).todense().astype(int)
    # TODO(add): track wrapped values as a union of intervals.
    # The precise pattern reads x[0..5] only.
    np.testing.assert_array_equal(result, np.ones((1, 20), dtype=int))


@pytest.mark.elementwise
def test_wrapped_mul_bounds_are_dropped():
    """An int8 product that can overflow carries no bounds.

    argmax(x[:4]) * 64 is {0, 64, -128, -64} after int8 wraparound.
    Unchecked corner products claim [-64, 0],
    so maximum(., 0) would claim the single start 0
    while argmax = 1 really starts the slice at 64.
    """

    def f(x):
        i = jnp.argmax(x[:4]).astype(jnp.int8) * jnp.int8(64)
        start = jnp.maximum(i, jnp.int8(0)).astype(jnp.int32)
        return lax.dynamic_slice(x, (start,), (1,))

    for position in range(4):
        assert_jacobian_sparsity_conservative(f, _one_hot(100, position))
    result = jacobian_sparsity(f, np.zeros(100)).todense().astype(int)
    # TODO(mul): track wrapped values as a union of intervals.
    # The precise pattern reads x[0] and x[64] only.
    np.testing.assert_array_equal(result, np.ones((1, 100), dtype=int))


@pytest.mark.elementwise
def test_wrapped_add_bounds_do_not_fold_comparison():
    """A comparison on an overflowing sum does not fold to a constant.

    argmax(x[:8]) + 124 in int8 is {124..127} or {-128..-125},
    so ``i < 0`` can be either True or False
    and both start positions 30 and 5 stay reachable.
    """

    def f(x):
        i = jnp.argmax(x[:8]).astype(jnp.int8) + jnp.int8(124)
        start = jnp.where(i < 0, 30, 5)
        return lax.dynamic_slice(x, (start,), (1,))

    for position in range(8):
        assert_jacobian_sparsity_conservative(f, _one_hot(40, position))
    result = jacobian_sparsity(f, np.zeros(40)).todense().astype(int)
    # TODO(select_n): keep the branch values as a set instead of an interval.
    # The precise pattern reads x[5] and x[30] only.
    # Merged branch bounds enumerate every start in [5, 30]
    expected = np.zeros((1, 40), dtype=int)
    expected[0, 5:31] = 1
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_narrowing_convert_drops_bounds_that_wrap():
    """Converting bounds to a narrower int dtype drops them if values can wrap.

    argmax(x[:4]) + 126 is in [126, 129],
    which wraps to {126, 127, -128, -127} in int8.
    Casting the endpoints would give the inverted interval (126, -127),
    which enumerates no start at all.
    """

    def f(x):
        j = (jnp.argmax(x[:4]) + 126).astype(jnp.int8)
        start = jnp.maximum(j, jnp.int8(0)).astype(jnp.int32)
        return lax.dynamic_slice(x, (start,), (1,))

    for position in range(4):
        assert_jacobian_sparsity_conservative(f, _one_hot(130, position))
    result = jacobian_sparsity(f, np.zeros(130)).todense().astype(int)
    # TODO(convert_element_type): track wrapped values as a union of intervals.
    # The precise pattern reads x[0], x[126], and x[127] only.
    np.testing.assert_array_equal(result, np.ones((1, 130), dtype=int))


@pytest.mark.elementwise
@pytest.mark.parametrize(
    ("offset", "expected_cols"),
    [
        pytest.param(-2, [10, 11], id="spans_zero"),
        pytest.param(1, [11], id="positive"),
    ],
)
def test_convert_to_bool_bounds(offset, expected_cols):
    """Converting bounds to bool maps an interval around zero to [False, True].

    argmax(x[:5]) + offset is in [offset, offset + 4].
    Casting the endpoints would turn [-2, 2] into (True, True),
    although 0 is in range and converts to False.
    """

    def f(x):
        i = jnp.argmax(x[:5]) + offset
        start = i.astype(bool).astype(jnp.int32)
        return lax.dynamic_slice(x[10:20], (start,), (1,))

    for position in range(5):
        assert_jacobian_sparsity_conservative(f, _one_hot(20, position))
    result = jacobian_sparsity(f, np.zeros(20)).todense().astype(int)
    expected = np.zeros((1, 20), dtype=int)
    expected[0, expected_cols] = 1
    np.testing.assert_array_equal(result, expected)
