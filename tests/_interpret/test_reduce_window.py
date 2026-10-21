"""Tests for windowed reductions.

References:
    https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce_window.html
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from asdex import jacobian_sparsity
from tests._utils import (
    assert_flat_jacobian_sparsity_exact,
    assert_hessian_sparsity_exact,
)


def _numerical_flat_jacobian_sparsity_fwd(f, x):
    """Compute flattened Jacobian sparsity with forward-mode AD.

    JAX 0.11 implements neither reverse-mode nor vmapped forward-mode
    differentiation for generic ``reduce_window``,
    so the shared reverse-mode helper cannot be used for that primitive.
    A JVP per input basis vector gives the same numerical reference.
    """
    x = jnp.asarray(x)
    basis = jnp.eye(x.size, dtype=x.dtype).reshape((x.size, *x.shape))
    rows = [np.asarray(jax.jvp(f, (x,), (tangent,))[1]).ravel() for tangent in basis]
    return (np.abs(np.asarray(rows).T) > 1e-10).astype(int)


@pytest.mark.parametrize(
    (
        "shape",
        "window_dimensions",
        "window_strides",
        "padding",
        "base_dilation",
        "window_dilation",
    ),
    [
        ((6,), (2,), (1,), "VALID", None, None),
        ((8,), (3,), (2,), "SAME", None, None),
        ((5, 7), (2, 3), (1, 1), "VALID", None, None),
        ((5, 7), (3, 2), (2, 1), "SAME", None, None),
        ((3, 4), (2, 2), (1, 1), ((1, 1), (0, 2)), None, None),
        ((4, 5), (2, 2), (1, 1), ((-1, 0), (0, -1)), None, None),
        ((3, 4), (2, 2), (1, 1), "VALID", None, (2, 1)),
        ((3, 4), (2, 2), (1, 1), ((1, 1), (0, 2)), (2, 1), None),
        ((2, 3, 4), (1, 2, 2), (1, 2, 1), "VALID", None, None),
    ],
)
def test_reduce_window_sum_exact(
    shape,
    window_dimensions,
    window_strides,
    padding,
    base_dilation,
    window_dilation,
):
    """Sum windows depend on every input position in each strided/dilated window."""
    x = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape) + 1.0

    def f(x):
        # A scalar init of the operand dtype keeps JAX on reduce_window_sum;
        # any other init value falls back to the generic primitive.
        return lax.reduce_window(
            x,
            x.dtype.type(0.0),
            lax.add,
            window_dimensions,
            window_strides,
            padding,
            base_dilation,
            window_dilation,
        )

    assert_flat_jacobian_sparsity_exact(f, x)


@pytest.mark.parametrize(
    ("shape", "window_dimensions", "window_strides", "padding"),
    [
        ((6,), (2,), (1,), "VALID"),
        ((5, 7), (1, 2), (1, 2), "SAME"),
    ],
)
def test_reduce_window_extrema_exact(shape, window_dimensions, window_strides, padding):
    """Max and min windows together cover both elements of each size-2 window."""
    x = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape) + 1.0

    def f(x):
        windowed = (
            window_dimensions,
            window_strides,
            padding,
        )
        return lax.reduce_window(
            x, x.dtype.type(-np.inf), lax.max, *windowed
        ) + lax.reduce_window(x, x.dtype.type(np.inf), lax.min, *windowed)

    assert_flat_jacobian_sparsity_exact(f, x)


@pytest.mark.parametrize("combiner", [lax.add, lax.mul, lax.max, lax.min])
@pytest.mark.parametrize(
    ("shape", "window_dimensions", "window_strides", "padding", "expected"),
    [
        # A 1-D window of two covers two adjacent inputs per output.
        (
            (4,),
            (2,),
            (1,),
            "VALID",
            [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]],
        ),
        # A row stride of two leaves the last input row out of every window.
        (
            (3, 4),
            (2, 2),
            (2, 1),
            "VALID",
            [
                [1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
                [0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0, 0],
                [0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 0, 0],
            ],
        ),
        # Padding widens the border windows without adding dependencies.
        (
            (2,),
            (2,),
            (1,),
            ((1, 1),),
            [[1, 0], [1, 1], [0, 1]],
        ),
    ],
)
def test_reduce_window_generic_combines_whole_window(
    combiner, shape, window_dimensions, window_strides, padding, expected
):
    """Generic reduce_window unions the windows of every operand per output.

    Add, mul, max, and min read every element of their window,
    so the union matches their numerical pattern exactly,
    even though the handler does not look at the combiner jaxpr.
    """
    x = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape) + 1.0

    def f(x):
        # A non-identity init value keeps JAX on the generic primitive
        # rather than lowering to reduce_window_sum/max/min.
        return lax.reduce_window(
            x,
            x.dtype.type(1.0),
            combiner,
            window_dimensions,
            window_strides,
            padding,
        )

    detected = jacobian_sparsity(f, x).todense().astype(int)
    np.testing.assert_array_equal(detected, np.asarray(expected))
    numerical = _numerical_flat_jacobian_sparsity_fwd(f, x)
    assert np.all(detected >= numerical), "Detected pattern must cover numerical"


@pytest.mark.fallback
def test_reduce_window_generic_accumulator_only_conservative():
    """A combiner that ignores the window element still reports the full window.

    TODO(reduce_window): the handler unions the window without looking at the
    combiner jaxpr, so it cannot tell that this combiner returns the init value
    untouched and leaves every derivative zero.
    Analyzing the combiner would give every output an empty set.
    """
    x = jnp.arange(4, dtype=jnp.float32) + 1.0

    def f(x):
        return lax.reduce_window(
            x,
            x.dtype.type(1.0),
            lambda acc, _elem: acc,
            (2,),
            (1,),
            "VALID",
        )

    detected = jacobian_sparsity(f, x).todense().astype(int)
    expected = [[1, 1, 0, 0], [0, 1, 1, 0], [0, 0, 1, 1]]
    np.testing.assert_array_equal(detected, np.asarray(expected))
    # The combiner never reads the window element, so nothing is needed.
    assert _numerical_flat_jacobian_sparsity_fwd(f, x).sum() == 0


def test_reduce_window_dynamic_init():
    """An init value computed from the input reaches every output.

    JAX cannot differentiate a non-constant init value
    (``reduce_window jvp does not support non-zero init_value_tangent``),
    so there is no numerical reference here:
    unioning the init value's index sets is what keeps the pattern covered.
    """
    x = jnp.arange(4, dtype=jnp.float32) + 1.0

    def f(x):
        return lax.reduce_window(x, x.sum(), lax.add, (2,), (1,), "VALID")

    detected = jacobian_sparsity(f, x).todense().astype(int)
    np.testing.assert_array_equal(detected, np.ones((3, 4), dtype=int))


@pytest.mark.fallback
def test_reduce_window_multiple_operands_conservative():
    """Generic reduce_window over a tuple of operands unions every operand.

    The second output accumulates only the first operand (``lhs[1]``)
    and never reads the second one,
    so its dependency on the second operand is an over-approximation.

    TODO(reduce_window): tracking the combiner would give each output
    exactly the windows of the operands it accumulates.
    """
    x = (jnp.arange(6.0) + 1.0, jnp.arange(6.0) + 2.0)

    def computation(lhs, rhs):
        return lhs[0] + rhs[0], lhs[1]

    def f(xs):
        inits = (xs[0].dtype.type(0.0), xs[1].dtype.type(1.0))
        return lax.reduce_window(xs, inits, computation, (2,), (1,))

    def f_flat(z):
        xs = (z[:6], z[6:])
        return jnp.concatenate([out.ravel() for out in f(xs)])

    # Each row of the first operand's block also carries the second operand's
    # window, and both outputs share that union,
    # so the two blocks of five output rows are identical.
    window = [
        [1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1],
    ]
    expected = np.asarray(window + window)

    detected = jacobian_sparsity(f, x).todense().astype(int)
    np.testing.assert_array_equal(detected, expected)
    numerical = _numerical_flat_jacobian_sparsity_fwd(
        f_flat, jnp.concatenate([leaf.ravel() for leaf in x])
    )
    assert np.all(detected >= numerical), "Detected pattern must cover numerical"


@pytest.mark.parametrize(
    ("shape", "padding", "window_dimensions"),
    [
        ((0,), ((2, 2),), (2,)),
        ((2,), ((1, 1),), (4,)),
        ((0,), ((0, 0),), (2,)),
        ((0, 3), ((1, 1), (0, 0)), (2, 2)),
        ((1,), ((1, 1),), (1,)),
        ((3,), ((3, 3),), (1,)),
    ],
)
def test_reduce_window_degenerate(shape, padding, window_dimensions):
    """Empty inputs, oversized windows, and all-padding windows stay dependency-free."""
    x = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape)

    def f(x):
        return lax.reduce_window(
            x,
            x.dtype.type(0.0),
            lax.add,
            window_dimensions,
            (1,) * len(shape),
            padding,
        )

    assert_flat_jacobian_sparsity_exact(f, x)


def test_reduce_window_scalar_operand():
    """A rank-0 operand is a single empty window that reads the scalar itself."""
    x = jnp.float32(2.0)

    def f(x):
        return lax.reduce_window(x, x.dtype.type(0.0), lax.add, (), ())

    assert_flat_jacobian_sparsity_exact(f, x)


def test_reduce_window_sum_hessian():
    """Hessians through reduce_window_sum reuse the window dependency structure."""
    x = jnp.arange(6.0)

    def f(x):
        windowed = lax.reduce_window(x, x.dtype.type(0.0), lax.add, (2,), (1,))
        return (windowed**2).sum()

    assert_hessian_sparsity_exact(f, x)
