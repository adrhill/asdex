"""Tests for the _prop_bitcast_convert_type handler.

JAX defines the derivative of a bitcast as zero,
so outputs have no dependencies.
Const values are reinterpreted bit for bit,
which matters when a bitcast feeds an index.
"""

import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from asdex import jacobian_sparsity
from tests._utils import assert_jacobian_sparsity_exact


@pytest.mark.elementwise
def test_bitcast_same_width_has_zero_derivative():
    """A same-width bitcast has no dependencies, like its JAX derivative."""

    def f(x):
        return lax.bitcast_convert_type(x, jnp.int32).astype(jnp.float32)

    x = jnp.arange(1.0, 4.0)
    assert_jacobian_sparsity_exact(f, x)
    result = jacobian_sparsity(f, x).todense().astype(int)
    np.testing.assert_array_equal(result, np.zeros((3, 3), dtype=int))


@pytest.mark.elementwise
def test_bitcast_narrowing_adds_trailing_axis():
    """Narrowing f32[3] to i8[3, 4] gives one row per output byte."""

    def f(x):
        return lax.bitcast_convert_type(x, jnp.int8).astype(jnp.float32).ravel()

    x = jnp.arange(1.0, 4.0)
    assert_jacobian_sparsity_exact(f, x)
    result = jacobian_sparsity(f, x).todense().astype(int)
    np.testing.assert_array_equal(result, np.zeros((12, 3), dtype=int))


@pytest.mark.elementwise
def test_bitcast_widening_consumes_trailing_axis():
    """Widening f16[3, 2] to f32[3] gives one row per output element."""

    def f(x):
        halves = x.reshape(3, 2).astype(jnp.float16)
        return lax.bitcast_convert_type(halves, jnp.float32)

    x = jnp.arange(1.0, 7.0)
    assert_jacobian_sparsity_exact(f, x)
    result = jacobian_sparsity(f, x).todense().astype(int)
    np.testing.assert_array_equal(result, np.zeros((3, 6), dtype=int))


@pytest.mark.elementwise
def test_bitcast_reinterprets_const_bits():
    """A bitcast const index keeps its bits instead of converting its value.

    The smallest positive float32 subnormal has the bit pattern of int32 1.
    Converting the value instead would give index 0.
    """

    def f(x):
        i = lax.bitcast_convert_type(jnp.float32(1.4e-45), jnp.int32)
        return x[i][None]

    x = jnp.arange(5.0)
    assert_jacobian_sparsity_exact(f, x)
    result = jacobian_sparsity(f, x).todense().astype(int)
    expected = np.zeros((1, 5), dtype=int)
    expected[0, 1] = 1
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_bitcast_narrowing_const_keeps_byte_order():
    """Narrowing a const index splits it into bytes, least significant first."""
    word = np.int32(0x01000302)

    def f(x):
        return x[lax.bitcast_convert_type(word, jnp.int8)]

    x = jnp.arange(5.0)
    assert_jacobian_sparsity_exact(f, x)
    result = jacobian_sparsity(f, x).todense().astype(int)
    expected = np.zeros((4, 5), dtype=int)
    expected[[0, 1, 2, 3], [2, 3, 0, 1]] = 1
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_bitcast_widening_const_joins_bytes():
    """Widening a const index joins each trailing row of bytes into one word."""
    data = np.array([[3, 0, 0, 0], [1, 0, 0, 0]], dtype=np.int8)

    def f(x):
        return x[lax.bitcast_convert_type(data, jnp.int32)]

    x = jnp.arange(5.0)
    assert_jacobian_sparsity_exact(f, x)
    result = jacobian_sparsity(f, x).todense().astype(int)
    expected = np.zeros((2, 5), dtype=int)
    expected[[0, 1], [3, 1]] = 1
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_bitcast_zero_size():
    """A bitcast of an empty array gives an empty output."""

    def f(x):
        empty = lax.bitcast_convert_type(x[:0], jnp.int8)
        return empty.astype(jnp.float32).ravel()

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    assert result.shape == (0, 3)
