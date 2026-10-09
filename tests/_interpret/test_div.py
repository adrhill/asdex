"""Tests for division propagation.

Basic div sparsity tests (zero numerators, bounds through dynamic_slice)
live in ``test_elementwise.py``.
This file covers the integer semantics of ``lax.div``,
which truncates toward zero and differs from numpy's true division
and floor division on negative operands.

https://docs.jax.dev/en/latest/_autosummary/jax.lax.div.html
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from asdex import jacobian_sparsity
from asdex.detection._interpret._common import _PropState
from asdex.detection._interpret._div import _lax_div, _propagate_bounds_div
from tests._utils import (
    assert_jacobian_sparsity_conservative,
    assert_jacobian_sparsity_exact,
)


@pytest.mark.elementwise
def test_div_integer_const_truncation():
    """Integer div const propagation follows lax.div truncation toward zero.

    lax.div([0, 1, 2], 2) = [0, 0, 1], so the gather reads [x[0], x[0], x[2]].
    True division would give [0, 0.5, 1]
    and resolve row 1 to the wrong gather index.
    The const chain sits in a cond branch
    because top-level arithmetic on concrete arrays
    is folded away during tracing.
    """

    def f(x):
        idx = jnp.arange(3, dtype=jnp.int32)

        def true_branch(ops):
            i, values = ops
            j = lax.mul(lax.div(i, jnp.int32(2)), jnp.int32(2))  # [0, 0, 2]
            return values[j] * 1.0

        def false_branch(ops):
            _, values = ops
            return values[:3] * 0.0

        return lax.cond(x[0] > 0, true_branch, false_branch, (idx, x))

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.array(
        [
            [1, 0, 0],  # out[0] <- x[0]
            [1, 0, 0],  # out[1] <- x[0]
            [0, 0, 1],  # out[2] <- x[2]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)
    # x[0] > 0 takes the true branch, so jax.jacobian sees the gather.
    assert_jacobian_sparsity_exact(f, np.array([1.0, 2.0, 3.0]))


@pytest.mark.elementwise
@pytest.mark.parametrize("dtype", [jnp.int32, jnp.float32])
def test_lax_div_matches_lax(dtype):
    """``_lax_div`` agrees with ``lax.div`` on every sign combination.

    Guards against the numpy reimplementation drifting from JAX's semantics.
    """
    num = np.array([-7, -6, -5, -1, 0, 1, 5, 6, 7], dtype=dtype)
    den = np.array([-3, -2, 2, 3], dtype=dtype)
    num, den = np.meshgrid(num, den)

    expected = np.asarray(lax.div(num, den))
    result = _lax_div(num, den)
    assert result is not None
    assert result.dtype == expected.dtype
    np.testing.assert_array_equal(result, expected)


@pytest.mark.elementwise
def test_lax_div_float_zero_divisor_matches_lax():
    """Float division by zero is well defined and matches ``lax.div``."""
    num = np.array([-1, 0, 1], dtype=np.float32)
    den = np.zeros(3, dtype=np.float32)

    with np.errstate(divide="ignore", invalid="ignore"):
        result = _lax_div(num, den)
    assert result is not None
    np.testing.assert_array_equal(result, np.asarray(lax.div(num, den)))


@pytest.mark.elementwise
@pytest.mark.parametrize(
    ("num", "den"),
    [
        ([5, 6], [2, 0]),  # division by zero
        ([np.iinfo(np.int32).min], [-1]),  # signed overflow
    ],
    ids=["zero_divisor", "int_min_by_minus_one"],
)
def test_lax_div_integer_undefined_is_unknown(num, den):
    """Integer div results that XLA leaves implementation-defined are not guessed."""
    result = _lax_div(np.array(num, dtype=np.int32), np.array(den, dtype=np.int32))
    assert result is None


@pytest.mark.elementwise
def test_div_integer_zero_divisor_index_conservative():
    """An index computed by integer division by zero falls back to conservative.

    lax.div(i, 0) is implementation-defined in XLA,
    so the gather index is unknown
    and every output may read any input.
    """

    def f(x):
        idx = jnp.arange(3, dtype=jnp.int32)

        def true_branch(ops):
            i, values = ops
            j = lax.div(i, jnp.zeros(3, dtype=jnp.int32))
            return values[j] * 1.0

        def false_branch(ops):
            _, values = ops
            return values[:3] * 0.0

        return lax.cond(x[0] > 0, true_branch, false_branch, (idx, x))

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.ones((3, 3), dtype=int)  # out[i] <- any x[j]
    np.testing.assert_array_equal(result, expected)
    assert_jacobian_sparsity_conservative(f, np.array([1.0, 2.0, 3.0]))


@pytest.mark.elementwise
def test_div_bounds_integer_truncation():
    """Integer bounds through div follow lax.div truncation toward zero.

    lax.div(-5, 2) = -2, while flooring gives -3.
    A floored bound excludes the value the program actually computes,
    so bounded enumeration would never try the true index.
    """
    jaxpr = jax.make_jaxpr(lambda a, b: lax.div(a, b))(
        jnp.zeros(1, jnp.int32), jnp.zeros(1, jnp.int32)
    ).jaxpr
    eqn = jaxpr.eqns[0]
    numerator, denominator = eqn.invars

    state = _PropState(
        consts={denominator: np.array([2], dtype=np.int32)},
        bounds={numerator: (np.array([-5]), np.array([-5]))},
    )
    _propagate_bounds_div(eqn, state)

    lo, hi = state.bounds[eqn.outvars[0]]
    np.testing.assert_array_equal(lo, [-2])
    np.testing.assert_array_equal(hi, [-2])


@pytest.mark.elementwise
def test_lax_div_matches_lax_for_large_int64():
    """Integer const division is exact for int64 values beyond float64 precision.

    Truncating a float64 quotient loses the low bits of operands above 2**53,
    so an index computed from such a quotient would resolve to the wrong position.
    The result must match ``lax.div`` exactly, for every sign combination.
    """
    big = 2**60 + 3
    a = np.array([big, -big, big, -big, 7, -7], dtype=np.int64)
    b = np.array([3, 3, -3, -3, 2, 2], dtype=np.int64)
    with jax.enable_x64(True):
        expected = np.asarray(lax.div(a, b))
    np.testing.assert_array_equal(_lax_div(a, b), expected)
