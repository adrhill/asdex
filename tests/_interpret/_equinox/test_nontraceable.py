"""Tests for Equinox's nonbatchable, nontraceable and nondifferentiable_backward.

All three come from ``equinox/internal/_nontraceable.py``.
They return their operand unchanged and only raise under some transformations,
so they get the pattern of a pass-through primitive.

JAX differentiates through ``nondifferentiable_backward`` in forward mode only,
so its patterns are checked against ``jax.jacfwd``.
JAX cannot differentiate through ``nonbatchable`` and ``nontraceable``,
so their patterns are checked by hand.
Where JAX raises, asdex must raise as well instead of returning zeros.

Requires equinox to be installed.
"""

import numpy as np
import pytest

eqx = pytest.importorskip("equinox")
eqxi = eqx.internal

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

import asdex  # noqa: E402
from asdex import jacobian_sparsity  # noqa: E402
from tests._utils import (  # noqa: E402
    assert_flat_jacobian_sparsity_exact,
    assert_jacobian_sparsity_exact,
)

_X = np.arange(1.0, 4.0)


def _scaled(op):
    return lambda x: op(x) * 2


def _overlapping(op):
    """Index sets with several entries per element."""
    return lambda x: op((x[:2] + x[1:]).reshape(1, 2))


@pytest.mark.array_ops
@pytest.mark.parametrize("build", [_scaled, _overlapping], ids=lambda b: b.__name__)
def test_nondifferentiable_backward_pattern(build):
    """The operand's index sets pass through unchanged and nothing is reported."""
    f = build(eqxi.nondifferentiable_backward)
    result = assert_flat_jacobian_sparsity_exact(f, _X, reference=jax.jacfwd)
    assert result._fallbacks == ()


@pytest.mark.array_ops
@pytest.mark.parametrize(
    ("build", "expected"),
    [
        (_scaled, np.eye(3)),
        (_overlapping, np.array([[1, 1, 0], [0, 1, 1]])),
    ],
    ids=["_scaled", "_overlapping"],
)
@pytest.mark.parametrize(
    "op", [eqxi.nonbatchable, eqxi.nontraceable], ids=lambda op: op.__name__
)
def test_not_differentiable_pattern(op, build, expected):
    """The operand's index sets pass through unchanged and nothing is reported."""
    result = jacobian_sparsity(build(op), _X)
    np.testing.assert_array_equal(result.todense(), expected)
    assert result._fallbacks == ()


@pytest.mark.array_ops
@pytest.mark.parametrize(
    "op",
    [eqxi.nonbatchable, eqxi.nontraceable, eqxi.nondifferentiable_backward],
    ids=lambda op: op.__name__,
)
def test_const_index_stays_resolvable(op):
    """A const index passed through keeps its value, so the gather stays precise.

    The index carries a symbolic zero tangent,
    so JAX differentiates ``f`` despite the op.
    """

    def f(x):
        return x[op(jnp.array([2, 0]))]

    result = assert_jacobian_sparsity_exact(f, _X)
    assert result._fallbacks == ()


@pytest.mark.array_ops
@pytest.mark.parametrize(
    ("op", "mode", "error", "match"),
    [
        (eqxi.nonbatchable, "fwd", NotImplementedError, "Differentiation rule"),
        (eqxi.nonbatchable, "rev", NotImplementedError, "Differentiation rule"),
        (eqxi.nontraceable, "fwd", RuntimeError, "Detected differentiation"),
        (eqxi.nontraceable, "rev", RuntimeError, "Detected differentiation"),
        (eqxi.nondifferentiable_backward, "rev", RuntimeError, "Unexpected cotangent"),
    ],
    ids=[
        "nonbatchable-fwd",
        "nonbatchable-rev",
        "nontraceable-fwd",
        "nontraceable-rev",
        "nondifferentiable_backward-rev",
    ],
)
def test_raises_like_jax(op, mode, error, match):
    """Asdex raises where JAX raises instead of returning zeros.

    An empty pattern would skip AD and return an all-zero Jacobian.
    """
    f = _scaled(op)
    jax_jacobian = jax.jacfwd if mode == "fwd" else jax.jacrev
    with pytest.raises(error, match=match):
        jax_jacobian(f)(_X)
    with pytest.raises(error, match=match):
        asdex.jacobian(f, _X, mode=mode)(_X)
