"""Tests for Equinox's nonbatchable, nontraceable and nondifferentiable_backward.

All three come from ``equinox/internal/_nontraceable.py``.
They return their operand unchanged and only raise under some transformations,
so they get the pattern of a pass-through primitive.

``nondifferentiable_backward`` raises only in reverse mode,
so its patterns are checked against ``jax.jacfwd``.
JAX cannot differentiate ``nonbatchable`` and ``nontraceable`` in either mode,
so asdex must raise as well instead of returning zeros.
Integer operands carry symbolic zero tangents,
so all three can be checked against ``jax.jacobian`` for const indices.

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

_IDENTITIES = [eqxi.nonbatchable, eqxi.nontraceable, eqxi.nondifferentiable_backward]
_NOT_DIFFERENTIABLE = [
    (eqxi.nonbatchable, NotImplementedError, "Differentiation rule"),
    (eqxi.nontraceable, RuntimeError, "Detected differentiation"),
]
_X = np.arange(1.0, 4.0)


@pytest.mark.array_ops
@pytest.mark.parametrize("op", _IDENTITIES, ids=lambda op: op.__name__)
def test_identity_pattern(op):
    """The operand's index sets pass through unchanged and nothing is reported."""

    def f(x):
        return op(x) * 2

    result = jacobian_sparsity(f, _X)
    np.testing.assert_array_equal(result.todense(), np.eye(3))
    assert result._fallbacks == ()
    if op is eqxi.nondifferentiable_backward:
        assert_jacobian_sparsity_exact(f, _X, reference=jax.jacfwd)


@pytest.mark.array_ops
@pytest.mark.parametrize("op", _IDENTITIES, ids=lambda op: op.__name__)
def test_non_trivial_index_sets(op):
    """Index sets with several entries per element pass through unchanged."""

    def f(x):
        return op((x[:2] + x[1:]).reshape(1, 2))

    result = jacobian_sparsity(f, _X)
    expected = np.array([[1, 1, 0], [0, 1, 1]])
    np.testing.assert_array_equal(result.todense(), expected)
    assert result._fallbacks == ()
    if op is eqxi.nondifferentiable_backward:
        assert_flat_jacobian_sparsity_exact(f, _X, reference=jax.jacfwd)


@pytest.mark.array_ops
@pytest.mark.parametrize("op", _IDENTITIES, ids=lambda op: op.__name__)
def test_const_index_stays_resolvable(op):
    """A const index passed through keeps its value, so the gather stays precise."""

    def f(x):
        return x[op(jnp.array([2, 0]))]

    result = jacobian_sparsity(f, _X)
    expected = np.array([[0, 0, 1], [1, 0, 0]])
    np.testing.assert_array_equal(result.todense(), expected)
    assert result._fallbacks == ()
    assert_jacobian_sparsity_exact(f, _X)


@pytest.mark.array_ops
def test_nondifferentiable_backward_matches_jax_per_mode():
    """Asdex differentiates in forward mode and raises in reverse mode, like JAX.

    An empty pattern would skip AD in both modes and return zeros.
    """

    def f(x):
        return eqxi.nondifferentiable_backward(x) * 2

    jac = asdex.jacobian(f, _X, mode="fwd", output_format="dense")(_X)
    np.testing.assert_allclose(jac, jax.jacfwd(f)(_X))

    with pytest.raises(RuntimeError, match="Unexpected cotangent"):
        jax.jacrev(f)(_X)
    with pytest.raises(RuntimeError, match="Unexpected cotangent"):
        asdex.jacobian(f, _X, mode="rev")(_X)


@pytest.mark.array_ops
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize(
    ("op", "error", "match"),
    _NOT_DIFFERENTIABLE,
    ids=[op.__name__ for op, _, _ in _NOT_DIFFERENTIABLE],
)
def test_not_differentiable_raises_like_jax(op, error, match, mode):
    """Asdex raises where JAX raises instead of returning zeros.

    An empty pattern would skip AD and return an all-zero Jacobian.
    """

    def f(x):
        return op(x) * 2

    jax_jacobian = jax.jacfwd if mode == "fwd" else jax.jacrev
    with pytest.raises(error, match=match):
        jax_jacobian(f)(_X)
    with pytest.raises(error, match=match):
        asdex.jacobian(f, _X, mode=mode)(_X)
