"""Tests for the prop_reduce handler.

All four reductions (sum, max, min, prod) share the same sparsity structure,
so the core shape/axes tests are parametrized over the reduce function.
High-level API tests verify that jnp functions lower correctly.
"""

import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity


def _reduction_jacobian(in_shape: tuple[int, ...], axes: tuple[int, ...]) -> np.ndarray:
    """Build the expected Jacobian for a reduction operation.

    Each output element has a 1 for every input element
    that reduces into it (all elements sharing the same non-reduced coordinates).
    """
    n_in = int(np.prod(in_shape))
    kept_dims = [d for d in range(len(in_shape)) if d not in axes]
    out_shape = tuple(in_shape[d] for d in kept_dims)
    n_out = int(np.prod(out_shape)) if out_shape else 1

    expected = np.zeros((n_out, n_in), dtype=int)
    for in_flat in range(n_in):
        in_coord = np.unravel_index(in_flat, in_shape)
        out_coord = tuple(in_coord[d] for d in kept_dims)
        out_flat = np.ravel_multi_index(out_coord, out_shape) if out_shape else 0
        expected[out_flat, in_flat] = 1
    return expected


# Reduce function wrappers (unified interface)
def _reduce_sum(x, axes):
    return jnp.sum(x, axis=axes)


def _reduce_max(x, axes):
    return lax.reduce_max(x, axes=axes)


def _reduce_min(x, axes):
    return lax.reduce_min(x, axes=axes)


def _reduce_prod(x, axes):
    return lax.reduce_prod(x, axes=axes)


_REDUCES = [
    pytest.param(_reduce_sum, id="sum"),
    pytest.param(_reduce_max, id="max"),
    pytest.param(_reduce_min, id="min"),
    pytest.param(_reduce_prod, id="prod"),
]

_SHAPES_AND_AXES = [
    pytest.param((5,), (0,), id="1d_full"),
    pytest.param((1,), (0,), id="1d_size_one"),
    pytest.param((3, 4), (0,), id="2d_axis0"),
    pytest.param((3, 4), (1,), id="2d_axis1"),
    pytest.param((3, 4), (0, 1), id="2d_both"),
    pytest.param((3, 1), (1,), id="2d_size_one_reduced"),
    pytest.param((1, 5), (1,), id="2d_size_one_kept"),
    pytest.param((2, 3, 4), (1,), id="3d_single_axis"),
    pytest.param((2, 3, 4), (0, 2), id="3d_two_axes"),
    pytest.param((2, 3, 4), (0, 1, 2), id="3d_full"),
    pytest.param((2, 3, 4, 5), (1, 3), id="4d"),
]


# Core reduction tests
@pytest.mark.reduction
@pytest.mark.parametrize(("in_shape", "axes"), _SHAPES_AND_AXES)
@pytest.mark.parametrize("reduce_fn", _REDUCES)
def test_reduce(reduce_fn, in_shape, axes):
    """Each output depends on all inputs along the reduced axes."""
    n_in = int(np.prod(in_shape))

    def f(x):
        return reduce_fn(x.reshape(in_shape), axes).flatten()

    result = jacobian_sparsity(f, n_in).todense().astype(int)
    expected = _reduction_jacobian(in_shape, axes)
    np.testing.assert_array_equal(result, expected)


# Compositions
@pytest.mark.reduction
@pytest.mark.parametrize("reduce_fn", _REDUCES)
def test_reduce_chained(reduce_fn):
    """Chained reductions: reduce axis 0, then axis 0 again."""
    shape = (2, 3, 4)

    def f(x):
        a = reduce_fn(x.reshape(shape), (0,))
        return reduce_fn(a, (0,)).flatten()

    result = jacobian_sparsity(f, 24).todense().astype(int)
    expected = _reduction_jacobian(shape, (0, 1))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
@pytest.mark.parametrize("reduce_fn", _REDUCES)
def test_reduce_after_broadcast(reduce_fn):
    """Reduce after broadcast: non-contiguous input dependencies."""

    def f(x):
        return reduce_fn(jnp.broadcast_to(x, (2, 3)), (0,))

    result = jacobian_sparsity(f, 3).todense().astype(int)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
@pytest.mark.parametrize("reduce_fn", _REDUCES)
def test_reduce_after_reduce_sum(reduce_fn):
    """Mix of reduce_sum then another reduction."""
    shape = (2, 3, 4)

    def f(x):
        a = jnp.sum(x.reshape(shape), axis=2)  # (2, 3)
        return reduce_fn(a, (1,)).flatten()

    result = jacobian_sparsity(f, 24).todense().astype(int)
    expected = _reduction_jacobian(shape, (1, 2))
    np.testing.assert_array_equal(result, expected)


# Edge cases
@pytest.mark.reduction
def test_reduce_sum_zero_size_input():
    """Zero-size input exercises empty union edge case."""

    def f(x):
        return jnp.sum(x)

    result = jacobian_sparsity(f, 0)
    assert result.shape == (1, 0)
    assert result.nnz == 0


# High-level API
@pytest.mark.reduction
def test_jnp_max_no_axis():
    """jnp.max without axis lowers to reduce_max over all axes."""

    def f(x):
        return jnp.max(x.reshape(2, 3)) * jnp.ones(1)

    result = jacobian_sparsity(f, 6).todense().astype(int)
    expected = np.ones((1, 6), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_jnp_max_with_axis():
    """jnp.max with axis lowers to reduce_max along that axis."""
    shape = (2, 3)

    def f(x):
        return jnp.max(x.reshape(shape), axis=1)

    result = jacobian_sparsity(f, 6).todense().astype(int)
    expected = _reduction_jacobian(shape, (1,))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_jnp_amax():
    """jnp.amax is an alias for jnp.max."""
    shape = (3, 4)

    def f(x):
        return jnp.amax(x.reshape(shape), axis=0)

    result = jacobian_sparsity(f, 12).todense().astype(int)
    expected = _reduction_jacobian(shape, (0,))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_jnp_min_no_axis():
    """jnp.min without axis lowers to reduce_min over all axes."""

    def f(x):
        return jnp.min(x.reshape(2, 3)) * jnp.ones(1)

    result = jacobian_sparsity(f, 6).todense().astype(int)
    expected = np.ones((1, 6), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_jnp_min_with_axis():
    """jnp.min with axis lowers to reduce_min along that axis."""
    shape = (2, 3)

    def f(x):
        return jnp.min(x.reshape(shape), axis=1)

    result = jacobian_sparsity(f, 6).todense().astype(int)
    expected = _reduction_jacobian(shape, (1,))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_jnp_amin():
    """jnp.amin is an alias for jnp.min."""
    shape = (3, 4)

    def f(x):
        return jnp.amin(x.reshape(shape), axis=0)

    result = jacobian_sparsity(f, 12).todense().astype(int)
    expected = _reduction_jacobian(shape, (0,))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_jnp_prod_no_axis():
    """jnp.prod without axis lowers to reduce_prod over all axes."""

    def f(x):
        return jnp.prod(x.reshape(2, 3)) * jnp.ones(1)

    result = jacobian_sparsity(f, 6).todense().astype(int)
    expected = np.ones((1, 6), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_jnp_prod_with_axis():
    """jnp.prod with axis lowers to reduce_prod along that axis."""
    shape = (2, 3)

    def f(x):
        return jnp.prod(x.reshape(shape), axis=1)

    result = jacobian_sparsity(f, 6).todense().astype(int)
    expected = _reduction_jacobian(shape, (1,))
    np.testing.assert_array_equal(result, expected)


# argmax / argmin (zero derivative)
@pytest.mark.reduction
def test_argmax():
    """Argmax has zero derivative (returns integer index, not differentiable).

    Only x[0] contributes because argmax output has empty dependency sets.
    """

    def f(x):
        idx = jnp.argmax(x)
        return x[0] * idx.astype(float)

    result = jacobian_sparsity(f, 3).todense().astype(int)
    expected = np.array([[1, 0, 0]])
    np.testing.assert_array_equal(result, expected)
