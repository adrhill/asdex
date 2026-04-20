"""Tests for elementwise operation propagation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from asdex import jacobian_sparsity


@pytest.mark.array_ops
def test_constant_in_elementwise_op():
    """Constant array in binary elementwise operation preserves input structure.

    Adding a constant array to input doesn't change the sparsity pattern.
    """

    def f(x):
        const = jnp.array([1.0, 2.0, 3.0])
        return x + const

    result = jacobian_sparsity(f, input_shape=(3,)).todense().astype(int)
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

    result = jacobian_sparsity(f, input_shape=(3,))
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

    result = jacobian_sparsity(f, input_shape=(12,)).todense().astype(int)
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

    result = jacobian_sparsity(f, input_shape=(12,)).todense().astype(int)
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

    result = jacobian_sparsity(f, input_shape=(6,)).todense().astype(int)
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

    result = jacobian_sparsity(f, input_shape=(4,)).todense().astype(int)
    expected = np.eye(4, dtype=int)
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

    result = jacobian_sparsity(f, input_shape=(3,)).todense().astype(int)
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

    result = jacobian_sparsity(f, input_shape=(3,)).todense().astype(int)
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

    result = jacobian_sparsity(f, input_shape=(4,)).todense().astype(int)
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

    result = jacobian_sparsity(f, input_shape=(3,)).todense().astype(int)
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

    result = jacobian_sparsity(f, input_shape=(3,)).todense().astype(int)
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

    result = jacobian_sparsity(f, input_shape=(5,)).todense().astype(int)
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

    result = jacobian_sparsity(f, input_shape=(5,)).todense().astype(int)
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

    result = jacobian_sparsity(f, input_shape=(5,)).todense().astype(int)
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

    result = jacobian_sparsity(f, input_shape=(5,)).todense().astype(int)
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

    result = jacobian_sparsity(f, input_shape=(5,)).todense().astype(int)
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

    result = jacobian_sparsity(f, input_shape=(3,)).todense().astype(int)
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

    result = jacobian_sparsity(f, input_shape=(5,)).todense().astype(int)
    # Start is always 0, so out = [x[0], x[1]].
    expected = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 1, 0, 0, 0],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)
