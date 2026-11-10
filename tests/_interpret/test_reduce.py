"""Tests for the _prop_reduce handler.

All four reductions (sum, max, min, prod) share the same sparsity structure,
so the core shape/axes tests are parametrized over the reduce function.
High-level API tests verify that jnp functions lower correctly.
"""

import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity
from tests._utils import (
    assert_jacobian_sparsity_conservative,
    assert_jacobian_sparsity_exact,
)

_needs_minmax = pytest.mark.skipif(
    not hasattr(jnp, "minmax"),
    reason="jnp.minmax is not available in JAX 0.11.0",
)


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
    pytest.param((5,), (), id="1d_empty"),
    pytest.param((5,), (0,), id="1d_full"),
    pytest.param((1,), (0,), id="1d_size_one"),
    pytest.param((3, 4), (), id="2d_empty"),
    pytest.param((3, 4), (0,), id="2d_axis0"),
    pytest.param((3, 4), (1,), id="2d_axis1"),
    pytest.param((3, 4), (0, 1), id="2d_both"),
    pytest.param((3, 1), (1,), id="2d_size_one_reduced"),
    pytest.param((1, 5), (1,), id="2d_size_one_kept"),
    pytest.param((2, 3, 4), (), id="3d_empty"),
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

    result = jacobian_sparsity(f, np.zeros(n_in)).todense().astype(int)
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

    result = jacobian_sparsity(f, np.zeros(24)).todense().astype(int)
    expected = _reduction_jacobian(shape, (0, 1))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
@pytest.mark.parametrize("reduce_fn", _REDUCES)
def test_reduce_after_broadcast(reduce_fn):
    """Reduce after broadcast: non-contiguous input dependencies."""

    def f(x):
        return reduce_fn(jnp.broadcast_to(x, (2, 3)), (0,))

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
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

    result = jacobian_sparsity(f, np.zeros(24)).todense().astype(int)
    expected = _reduction_jacobian(shape, (1, 2))
    np.testing.assert_array_equal(result, expected)


# Edge cases
@pytest.mark.reduction
def test_reduce_sum_scalar_empty_axis():
    """Scalar input with empty axis tuple is identity."""

    def f(x):
        return jnp.sum(x, axis=())

    assert_jacobian_sparsity_exact(f, np.array(1.0))


@pytest.mark.reduction
def test_reduce_sum_scalar_full():
    """Scalar input with axis=None is full reduction (trivially identity)."""

    def f(x):
        return jnp.sum(x, axis=None)

    assert_jacobian_sparsity_exact(f, np.array(1.0))


@pytest.mark.reduction
def test_reduce_sum_zero_size_input():
    """Zero-size input exercises empty union edge case."""

    def f(x):
        return jnp.sum(x)

    result = jacobian_sparsity(f, np.zeros(0))
    assert result.shape == (1, 0)
    assert result.nnz == 0


@pytest.mark.reduction
def test_reduce_sum_zero_size_partial():
    """Zero-size axis in partial reduction."""
    shape = (2, 0, 3)
    n_in = int(np.prod(shape))
    n_out = 2 * 3  # reduce axis 1

    def f(x):
        return jnp.sum(x.reshape(shape), axis=1).flatten()

    result = jacobian_sparsity(f, np.zeros(n_in))
    assert result.shape == (n_out, n_in)
    assert result.nnz == 0


@pytest.mark.reduction
@pytest.mark.parametrize(
    ("shape", "axes"),
    [
        pytest.param((3, 4), (-1,), id="2d_neg1"),
        pytest.param((3, 4), (-2,), id="2d_neg2"),
        pytest.param((2, 3, 4), (-1, -3), id="3d_neg_multiple"),
    ],
)
def test_reduce_negative_axis(shape, axes):
    """Negative axis indices work correctly."""

    def f(x):
        return jnp.sum(x.reshape(shape), axis=axes).flatten()

    assert_jacobian_sparsity_exact(f, np.ones(int(np.prod(shape))))


# High-level API
@pytest.mark.reduction
def test_jnp_max_no_axis():
    """jnp.max without axis lowers to reduce_max over all axes."""

    def f(x):
        return jnp.max(x.reshape(2, 3)) * jnp.ones(1)

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    expected = np.ones((1, 6), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_jnp_max_with_axis():
    """jnp.max with axis lowers to reduce_max along that axis."""
    shape = (2, 3)

    def f(x):
        return jnp.max(x.reshape(shape), axis=1)

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    expected = _reduction_jacobian(shape, (1,))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_jnp_amax():
    """jnp.amax is an alias for jnp.max."""
    shape = (3, 4)

    def f(x):
        return jnp.amax(x.reshape(shape), axis=0)

    result = jacobian_sparsity(f, np.zeros(12)).todense().astype(int)
    expected = _reduction_jacobian(shape, (0,))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_jnp_min_no_axis():
    """jnp.min without axis lowers to reduce_min over all axes."""

    def f(x):
        return jnp.min(x.reshape(2, 3)) * jnp.ones(1)

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    expected = np.ones((1, 6), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_jnp_min_with_axis():
    """jnp.min with axis lowers to reduce_min along that axis."""
    shape = (2, 3)

    def f(x):
        return jnp.min(x.reshape(shape), axis=1)

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    expected = _reduction_jacobian(shape, (1,))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_jnp_amin():
    """jnp.amin is an alias for jnp.min."""
    shape = (3, 4)

    def f(x):
        return jnp.amin(x.reshape(shape), axis=0)

    result = jacobian_sparsity(f, np.zeros(12)).todense().astype(int)
    expected = _reduction_jacobian(shape, (0,))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_jnp_prod_no_axis():
    """jnp.prod without axis lowers to reduce_prod over all axes."""

    def f(x):
        return jnp.prod(x.reshape(2, 3)) * jnp.ones(1)

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    expected = np.ones((1, 6), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_jnp_prod_with_axis():
    """jnp.prod with axis lowers to reduce_prod along that axis."""
    shape = (2, 3)

    def f(x):
        return jnp.prod(x.reshape(shape), axis=1)

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    expected = _reduction_jacobian(shape, (1,))
    np.testing.assert_array_equal(result, expected)


# argmax / argmin (zero derivative)
#
# Note: Unlike reduce_sum where axis=() means identity (no reduction),
# argmax/argmin always reduce exactly one axis. JAX's jnp.argmax with axis=None
# flattens the array first, then reduces axis 0. The jaxpr always has axes=(N,)
# with exactly one element - empty axes is not a valid case for argmax.

_SHAPES_AND_AXES_ARGMAX = [
    pytest.param((5,), 0, id="1d"),
    pytest.param((1,), 0, id="1d_size_one"),
    pytest.param((3, 4), 0, id="2d_axis0"),
    pytest.param((3, 4), 1, id="2d_axis1"),
    pytest.param((1, 5), 0, id="2d_size_one_axis0"),
    pytest.param((5, 1), 1, id="2d_size_one_axis1"),
    pytest.param((2, 3, 4), 0, id="3d_axis0"),
    pytest.param((2, 3, 4), 1, id="3d_axis1"),
    pytest.param((2, 3, 4), 2, id="3d_axis2"),
    pytest.param((2, 3, 2, 4), 2, id="4d"),
]


@pytest.mark.reduction
@pytest.mark.parametrize(("in_shape", "axis"), _SHAPES_AND_AXES_ARGMAX)
def test_argmax_zero_derivative(in_shape, axis):
    """Argmax has zero derivative - output depends on no inputs structurally."""
    n_in = int(np.prod(in_shape))
    out_shape = tuple(s for i, s in enumerate(in_shape) if i != axis)
    n_out = int(np.prod(out_shape)) if out_shape else 1

    def f(x):
        return (
            lax.argmax(x.reshape(in_shape), axis=axis, index_dtype=jnp.int32)
            .astype(float)
            .flatten()
        )

    result = jacobian_sparsity(f, np.zeros(n_in)).todense().astype(int)
    expected = np.zeros((n_out, n_in), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
@pytest.mark.parametrize(("in_shape", "axis"), _SHAPES_AND_AXES_ARGMAX)
def test_argmin_zero_derivative(in_shape, axis):
    """Argmin has zero derivative - same as argmax."""
    n_in = int(np.prod(in_shape))
    out_shape = tuple(s for i, s in enumerate(in_shape) if i != axis)
    n_out = int(np.prod(out_shape)) if out_shape else 1

    def f(x):
        return (
            lax.argmin(x.reshape(in_shape), axis=axis, index_dtype=jnp.int32)
            .astype(float)
            .flatten()
        )

    result = jacobian_sparsity(f, np.zeros(n_in)).todense().astype(int)
    expected = np.zeros((n_out, n_in), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_argmax_no_axis():
    """jnp.argmax with axis=None flattens first, then reduces axis 0.

    This is how JAX handles "full array" argmax - it never produces empty axes.
    """

    def f(x):
        return jnp.argmax(x.reshape(2, 3), axis=None).astype(float).reshape(1)

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    expected = np.zeros((1, 6), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_argmax_as_index():
    """Argmax used as index: only indexed elements contribute.

    Only x[0] contributes because argmax output has empty dependency sets.
    """

    def f(x):
        idx = jnp.argmax(x)
        return x[0] * idx.astype(float)

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.array([[1, 0, 0]])
    np.testing.assert_array_equal(result, expected)


# Generic reduce


def _fold(acc, x):
    """A monoid without a specialized primitive, so ``lax.reduce`` emits ``reduce``."""
    return acc + x + acc * x


@pytest.mark.reduction
@pytest.mark.parametrize(("in_shape", "axes"), _SHAPES_AND_AXES)
def test_reduce_generic_shapes(in_shape, axes):
    """``lax.reduce`` with a custom computation has the pattern of any reduction."""
    n_in = int(np.prod(in_shape))

    def f(x):
        return lax.reduce(x.reshape(in_shape), 0.0, _fold, axes).flatten()

    result = jacobian_sparsity(f, np.zeros(n_in)).todense().astype(int)
    np.testing.assert_array_equal(result, _reduction_jacobian(in_shape, axes))


@pytest.mark.reduction
def test_reduce_generic_matches_numerical():
    """The pattern of a generic reduction matches the numerical Jacobian."""

    def f(x):
        return lax.reduce(x.reshape(3, 4), 0.0, _fold, (1,))

    assert_jacobian_sparsity_exact(f, jnp.linspace(0.1, 1.2, 12))


@pytest.mark.reduction
def test_reduce_generic_init_value():
    """An init value that depends on the input reaches every output element.

    Not compared with the numerical Jacobian,
    since JAX cannot differentiate ``lax.reduce`` with respect to its init values.
    """

    def f(x):
        return lax.reduce(x[:6].reshape(2, 3), x[6], _fold, (1,))

    result = jacobian_sparsity(f, jnp.linspace(0.1, 0.7, 7)).todense().astype(int)
    expected = np.array(
        [
            [1, 1, 1, 0, 0, 0, 1],  # out[0] folds row 0, starting from x[6]
            [0, 0, 0, 1, 1, 1, 1],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_reduce_generic_mixing_operands():
    """A variadic computation can mix its operands, so each output reads all of them.

    The computation multiplies complex numbers stored as (real, imaginary) pairs.
    """

    def computation(accs, xs):
        return accs[0] * xs[0] - accs[1] * xs[1], accs[0] * xs[1] + accs[1] * xs[0]

    def f(x):
        re, im = x[:6].reshape(2, 3), x[6:].reshape(2, 3)
        out = lax.reduce((re, im), (1.0, 0.0), computation, (1,))
        return jnp.concatenate(out)

    x = jnp.linspace(0.1, 1.2, 12)
    assert_jacobian_sparsity_exact(f, x)
    result = jacobian_sparsity(f, x).todense().astype(int)
    expected = np.array(
        [
            # re row 0 re row 1 im row 0 im row 1
            [1, 1, 1, 0, 0, 0, 1, 1, 1, 0, 0, 0],  # real part of the product of row 0
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
            [
                1,
                1,
                1,
                0,
                0,
                0,
                1,
                1,
                1,
                0,
                0,
                0,
            ],  # imaginary part of the product of row 0
            [0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
@pytest.mark.fallback
def test_reduce_generic_max_with_payload():
    """Carrying a payload along with the maximum gets a conservative pattern.

    TODO(reduce): propagate index sets through the computation's jaxpr.
    The payload is selected by comparing the values, which has zero derivative,
    so the precise pattern of the payload output only depends on the payloads.
    """

    def computation(accs, xs):
        is_larger = xs[0] > accs[0]
        return jnp.maximum(accs[0], xs[0]), jnp.where(is_larger, xs[1], accs[1])

    def f(x):
        out = lax.reduce((x[:3], x[3:]), (-jnp.inf, 0.0), computation, (0,))
        return jnp.stack(out)

    x = jnp.array([0.3, 0.9, 0.5, 1.0, 2.0, 3.0])
    assert_jacobian_sparsity_conservative(f, x)
    result = jacobian_sparsity(f, x).todense().astype(int)
    # Precise: [[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]]
    expected = np.ones((2, 6), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
@pytest.mark.fallback
def test_reduce_generic_separable_operands():
    """Outputs that each fold a single operand get a conservative pattern.

    TODO(reduce): propagate index sets through the computation's jaxpr.
    The precise pattern keeps the operands apart:
    out0 only depends on a, and out1 only on b.
    """

    def computation(accs, xs):
        return accs[0] + xs[0] + accs[0] * xs[0], accs[1] + xs[1] + accs[1] * xs[1]

    def f(x):
        out = lax.reduce((x[:3], x[3:]), (0.0, 0.0), computation, (0,))
        return jnp.stack(out)

    x = jnp.linspace(0.1, 0.6, 6)
    assert_jacobian_sparsity_conservative(f, x)
    result = jacobian_sparsity(f, x).todense().astype(int)
    # Precise: [[1, 1, 1, 0, 0, 0], [0, 0, 0, 1, 1, 1]]
    expected = np.ones((2, 6), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_reduce_generic_zero_size_reduced_axis():
    """Reducing over a size-0 axis leaves only the init value."""

    def f(x):
        return lax.reduce(x[:3].reshape(3, 1)[:, :0], x[3], _fold, (1,))

    result = jacobian_sparsity(f, jnp.ones(4)).todense().astype(int)
    expected = np.array(
        [
            [0, 0, 0, 1],  # out[0] = init = x[3]
            [0, 0, 0, 1],
            [0, 0, 0, 1],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
def test_reduce_generic_zero_size_output():
    """A generic reduction with a size-0 output yields an empty pattern."""

    def f(x):
        return lax.reduce(x.reshape(3, 1)[:0], 0.0, _fold, (1,))

    result = jacobian_sparsity(f, jnp.ones(3))
    assert result.shape == (0, 3)
    assert result.nnz == 0


@pytest.mark.reduction
@_needs_minmax
def test_jnp_minmax_no_axis():
    """jnp.minmax reduces all elements into both the minimum and the maximum.

    Since JAX 0.12, it lowers to a single generic ``reduce`` over two operands.
    """

    def f(x):
        return jnp.stack(jnp.minmax(x))

    x = jnp.array([3.0, 1.0, 2.0])
    result = jacobian_sparsity(f, x).todense().astype(int)
    expected = np.array(
        [
            [1, 1, 1],  # min
            [1, 1, 1],  # max
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.reduction
@_needs_minmax
def test_jnp_minmax_with_axis():
    """jnp.minmax along an axis reduces each row into its minimum and maximum."""

    def f(x):
        return jnp.concatenate(jnp.minmax(x.reshape(2, 3), axis=1))

    x = jnp.array([3.0, 1.0, 2.0, 5.0, 6.0, 4.0])
    result = jacobian_sparsity(f, x).todense().astype(int)
    expected = np.array(
        [
            [1, 1, 1, 0, 0, 0],  # min of row 0
            [0, 0, 0, 1, 1, 1],
            [1, 1, 1, 0, 0, 0],  # max of row 0
            [0, 0, 0, 1, 1, 1],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)
