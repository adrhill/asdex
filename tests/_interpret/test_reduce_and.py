"""Tests for reduce_and, reduce_or, and reduce_xor handlers.

These are bitwise reductions with zero derivative,
so the Jacobian is always the zero matrix.
"""

import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity


# Reduce function wrappers (unified interface)
def _reduce_and(x, axes):
    return lax.reduce_and(x > 0, axes=axes)


def _reduce_or(x, axes):
    return lax.reduce_or(x > 0, axes=axes)


def _reduce_xor(x, axes):
    return lax.reduce_xor(x > 0, axes=axes)


_REDUCES = [
    pytest.param(_reduce_and, id="and"),
    pytest.param(_reduce_or, id="or"),
    pytest.param(_reduce_xor, id="xor"),
]

_SHAPES_AND_AXES = [
    pytest.param((5,), (0,), id="1d_full"),
    pytest.param((1,), (0,), id="1d_size_one"),
    pytest.param((3, 4), (0,), id="2d_axis0"),
    pytest.param((3, 4), (1,), id="2d_axis1"),
    pytest.param((3, 4), (0, 1), id="2d_both"),
    pytest.param((2, 3, 4), (1,), id="3d_single_axis"),
]


# Core tests
@pytest.mark.reduction
@pytest.mark.parametrize(("in_shape", "axes"), _SHAPES_AND_AXES)
@pytest.mark.parametrize("reduce_fn", _REDUCES)
def test_reduce_bitwise(reduce_fn, in_shape, axes):
    """Bitwise reductions have zero Jacobian regardless of shape or axes."""
    n_in = int(np.prod(in_shape))
    kept = [d for d in range(len(in_shape)) if d not in axes]
    n_out = int(np.prod([in_shape[d] for d in kept])) if kept else 1

    def f(x):
        return reduce_fn(x.reshape(in_shape), axes).flatten() * jnp.ones(n_out)

    result = jacobian_sparsity(f, input_shape=(n_in,)).todense().astype(int)
    expected = np.zeros((n_out, n_in), dtype=int)
    np.testing.assert_array_equal(result, expected)


# High-level API
@pytest.mark.reduction
def test_jnp_all_no_axis():
    """jnp.all without axis lowers to reduce_and: zero Jacobian."""

    def f(x):
        return jnp.all(x.reshape(2, 3) > 0) * jnp.ones(1)

    result = jacobian_sparsity(f, input_shape=(6,)).todense().astype(int)
    expected = np.zeros((1, 6), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_jnp_all_with_axis():
    """jnp.all with axis lowers to reduce_and: zero Jacobian."""

    def f(x):
        return jnp.all(x.reshape(2, 3) > 0, axis=1) * jnp.ones(2)

    result = jacobian_sparsity(f, input_shape=(6,)).todense().astype(int)
    expected = np.zeros((2, 6), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_jnp_any_no_axis():
    """jnp.any without axis lowers to reduce_or: zero Jacobian."""

    def f(x):
        return jnp.any(x.reshape(2, 3) > 0) * jnp.ones(1)

    result = jacobian_sparsity(f, input_shape=(6,)).todense().astype(int)
    expected = np.zeros((1, 6), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_jnp_any_with_axis():
    """jnp.any with axis lowers to reduce_or: zero Jacobian."""

    def f(x):
        return jnp.any(x.reshape(2, 3) > 0, axis=1) * jnp.ones(2)

    result = jacobian_sparsity(f, input_shape=(6,)).todense().astype(int)
    expected = np.zeros((2, 6), dtype=int)
    np.testing.assert_array_equal(result, expected)


# Compositions
@pytest.mark.reduction
def test_reduce_and_after_reduce_sum():
    """Threshold check after summation: reduce_and zeros out the sum state_indices."""

    def f(x):
        s = jnp.sum(x.reshape(2, 3), axis=1)  # (2,) with state_indices
        mask = lax.reduce_and(s > 0, axes=(0,))  # scalar, zero deriv
        return mask * jnp.ones(1)

    result = jacobian_sparsity(f, input_shape=(6,)).todense().astype(int)
    expected = np.zeros((1, 6), dtype=int)
    np.testing.assert_array_equal(result, expected)
