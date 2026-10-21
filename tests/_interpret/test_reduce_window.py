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

    JAX 0.11.2 implements neither reverse-mode nor vmapped forward-mode
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
        return lax.reduce_window(
            x,
            jnp.asarray(0.0, x.dtype),
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
            x, jnp.asarray(-np.inf, x.dtype), lax.max, *windowed
        ) + lax.reduce_window(x, jnp.asarray(np.inf, x.dtype), lax.min, *windowed)

    assert_flat_jacobian_sparsity_exact(f, x)


@pytest.mark.parametrize(
    ("combiner", "init_value"),
    [(lax.add, 1.0), (lax.mul, 2.0)],
)
@pytest.mark.parametrize(
    ("shape", "window_dimensions", "window_strides", "padding"),
    [
        ((6,), (2,), (1,), "VALID"),
        ((3, 4), (2, 3), (2, 1), "SAME"),
        ((3, 4), (2, 2), (1, 1), ((1, 1), (0, 2))),
    ],
)
def test_reduce_window_generic_exact(
    combiner, init_value, shape, window_dimensions, window_strides, padding
):
    """Generic reduce_window with a non-identity init value is window-wide."""
    x = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape) + 1.0

    def f(x):
        return lax.reduce_window(
            x,
            jnp.asarray(init_value, x.dtype),
            combiner,
            window_dimensions,
            window_strides,
            padding,
        )

    detected = jacobian_sparsity(f, x).todense().astype(int)
    expected = _numerical_flat_jacobian_sparsity_fwd(f, x)
    np.testing.assert_array_equal(detected, expected)


@pytest.mark.parametrize(
    ("shape", "window_dimensions", "window_strides"),
    [((6,), (2,), (1,)), ((3, 4), (1, 2), (1, 2))],
)
def test_reduce_window_generic_extrema_exact(shape, window_dimensions, window_strides):
    """Generic max and min windows with non-identity init values cover whole windows."""
    x = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape) + 1.0

    def f(x):
        windowed = (window_dimensions, window_strides, "VALID")
        # reduce_window requires the init values to have the operand dtype.
        init_max = jnp.asarray(0.0, x.dtype)
        init_min = jnp.asarray(1e6, x.dtype)
        return lax.reduce_window(x, init_max, lax.max, *windowed) + lax.reduce_window(
            x, init_min, lax.min, *windowed
        )

    detected = jacobian_sparsity(f, x).todense().astype(int)
    expected = _numerical_flat_jacobian_sparsity_fwd(f, x)
    np.testing.assert_array_equal(detected, expected)


def test_reduce_window_multiple_operands_exact():
    """Generic reduce_window over a tuple of operands unions all their windows."""
    x = (jnp.arange(6.0) + 1.0, jnp.arange(6.0) + 2.0)

    def computation(lhs, rhs):
        total = lhs[0] + rhs[0] + lhs[1] + rhs[1]
        return total, total

    def f(xs):
        inits = (jnp.asarray(0.0, xs[0].dtype), jnp.asarray(1.0, xs[1].dtype))
        return lax.reduce_window(xs, inits, computation, (2,), (1,))

    def f_flat(z):
        xs = (z[:6], z[6:])
        return jnp.concatenate([out.ravel() for out in f(xs)])

    detected = jacobian_sparsity(f, x).todense().astype(int)
    expected = _numerical_flat_jacobian_sparsity_fwd(
        f_flat, jnp.concatenate([leaf.ravel() for leaf in x])
    )
    np.testing.assert_array_equal(detected, expected)


@pytest.mark.parametrize(
    ("shape", "padding", "window_dimensions"),
    [
        ((0,), ((2, 2),), (2,)),
        ((2,), ((1, 1),), (4,)),
        ((0,), ((0, 0),), (2,)),
        ((0, 3), ((1, 1), (0, 0)), (2, 2)),
    ],
)
def test_reduce_window_degenerate(shape, padding, window_dimensions):
    """Empty inputs and windows larger than the input do not crash or invent deps."""
    x = jnp.arange(np.prod(shape), dtype=jnp.float32).reshape(shape)

    def f(x):
        return lax.reduce_window(
            x,
            jnp.asarray(0.0, x.dtype),
            lax.add,
            window_dimensions,
            (1,) * len(shape),
            padding,
        )

    assert_flat_jacobian_sparsity_exact(f, x)


def test_reduce_window_sum_hessian():
    """Hessians through reduce_window_sum reuse the window dependency structure."""
    x = jnp.arange(6.0)

    def f(x):
        windowed = lax.reduce_window(x, jnp.asarray(0.0, x.dtype), lax.add, (2,), (1,))
        return (windowed**2).sum()

    assert_hessian_sparsity_exact(f, x)
