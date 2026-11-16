"""Tests for Equinox's nonbatchable, nontraceable and nondifferentiable_backward.

All three come from ``equinox/internal/_nontraceable.py``.
They return their operand unchanged and only raise under some transformations,
so they get the pattern of a pass-through primitive.

Requires equinox to be installed.
"""

import numpy as np
import pytest

eqx = pytest.importorskip("equinox")
eqxi = eqx.internal

import jax.numpy as jnp  # noqa: E402

from asdex import jacobian_sparsity  # noqa: E402

_IDENTITIES = [eqxi.nonbatchable, eqxi.nontraceable, eqxi.nondifferentiable_backward]


@pytest.mark.array_ops
@pytest.mark.parametrize("op", _IDENTITIES, ids=lambda op: op.__name__)
def test_identity_pattern(op):
    """The operand's index sets pass through unchanged and nothing is reported."""

    def f(x):
        return op(x) * 2

    result = jacobian_sparsity(f, np.ones(3))
    np.testing.assert_array_equal(result.todense(), np.eye(3))
    assert result._fallbacks == ()


@pytest.mark.array_ops
@pytest.mark.parametrize("op", _IDENTITIES, ids=lambda op: op.__name__)
def test_non_trivial_index_sets(op):
    """Index sets with several entries per element pass through unchanged."""

    def f(x):
        return op((x[:2] + x[1:]).reshape(1, 2))

    result = jacobian_sparsity(f, np.ones(3))
    expected = np.array([[1, 1, 0], [0, 1, 1]])
    np.testing.assert_array_equal(result.todense(), expected)
    assert result._fallbacks == ()


@pytest.mark.array_ops
@pytest.mark.parametrize("op", _IDENTITIES, ids=lambda op: op.__name__)
def test_const_index_stays_resolvable(op):
    """A const index passed through keeps its value, so the gather stays precise."""

    def f(x):
        return x[op(jnp.array([2, 0]))]

    result = jacobian_sparsity(f, np.ones(3))
    expected = np.array([[0, 0, 1], [1, 0, 0]])
    np.testing.assert_array_equal(result.todense(), expected)
    assert result._fallbacks == ()
