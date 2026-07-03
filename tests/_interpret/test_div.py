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
from asdex.detection._interpret._div import _propagate_bounds_div


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

    state_consts = {denominator: np.array([2], dtype=np.int32)}
    state_bounds = {numerator: (np.array([-5]), np.array([-5]))}
    _propagate_bounds_div(eqn, state_consts, state_bounds)

    lo, hi = state_bounds[eqn.outvars[0]]
    np.testing.assert_array_equal(lo, [-2])
    np.testing.assert_array_equal(hi, [-2])
