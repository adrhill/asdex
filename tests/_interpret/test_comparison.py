"""Tests for comparison propagation (lt, le, gt, ge, eq, ne).

Comparisons have zero derivative.
``lt``, ``le``, ``gt``, and ``ge`` additionally resolve to a const boolean
when the value bounds of their operands prove the result,
so a downstream ``select_n`` only tracks the branch that can be taken.

Most tests cross-check that resolution against ``jax.jacobian``.
The operands are argmaxes of slices of ``x``,
so their bounds are known and every value in those bounds is reachable.
The reference pattern is the union of ``jax.jacobian`` over inputs
that reach every combination of operand values.
Detection must match that union exactly,
so asdex's bounds reasoning cannot drift from JAX's comparison semantics.

https://docs.jax.dev/en/latest/_autosummary/jax.lax.lt.html
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from asdex import jacobian_sparsity
from tests._utils import (
    assert_jacobian_sparsity_conservative,
    assert_jacobian_sparsity_exact,
)

# Input layout for the scalar tests:
# x[0:3] -> a = argmax in [0, 2]
# x[3:6] -> b = argmax in [0, 2]
# x[6]   -> branch taken when the comparison is true
# x[7]   -> branch taken when the comparison is false
_N_IN = 8
_TRUE_BRANCH = np.array([[0, 0, 0, 0, 0, 0, 1, 0]])
_FALSE_BRANCH = np.array([[0, 0, 0, 0, 0, 0, 0, 1]])
_BOTH_BRANCHES = _TRUE_BRANCH | _FALSE_BRANCH

_BOUNDED_OPS = [lax.lt, lax.le, lax.gt, lax.ge]

_i32 = np.int32
_f32 = np.float32


def _af(a):
    return a.astype(jnp.float32)


# (id, lhs, rhs) where lhs and rhs map the bounded operands (a, b) to an operand.
# Ids name a, b in [0, 2] and the const or expression they are compared with.
_OPERAND_CASES = [
    # Bounded vs const, with the const below, at, inside, and above [0, 2].
    *[(f"a_vs_{c}", lambda a, b: a, lambda a, b, c=c: _i32(c)) for c in range(-1, 4)],
    # Const vs bounded, so the bounds sit on the right-hand side.
    *[(f"{c}_vs_a", lambda a, b, c=c: _i32(c), lambda a, b: a) for c in range(-1, 4)],
    # Bounded vs bounded, from disjoint below through touching and overlap to disjoint above.
    *[
        (f"a_vs_b{s:+d}", lambda a, b: a, lambda a, b, s=s: b + _i32(s))
        for s in range(-3, 4)
    ],
    # Degenerate interval: a * 0 has bounds [0, 0].
    *[
        (f"a*0_vs_{c}", lambda a, b: a * _i32(0), lambda a, b, c=c: _i32(c))
        for c in range(-1, 2)
    ],
    # Float bounds: a / 2 has bounds [0.0, 1.0].
    *[
        (f"a/2_vs_{c}", lambda a, b: _af(a) * _f32(0.5), lambda a, b, c=c: _f32(c))
        for c in [-0.5, 0.0, 0.5, 1.0, 1.5]
    ],
    # Non-finite consts.
    ("a_vs_inf", lambda a, b: _af(a), lambda a, b: _f32(np.inf)),
    ("a_vs_-inf", lambda a, b: _af(a), lambda a, b: _f32(-np.inf)),
    ("a_vs_nan", lambda a, b: _af(a), lambda a, b: _f32(np.nan)),
    ("nan_vs_a", lambda a, b: _f32(np.nan), lambda a, b: _af(a)),
    # Const vs const, resolved by const propagation.
    *[
        (f"{c1}_vs_{c2}", lambda a, b, c1=c1: _i32(c1), lambda a, b, c2=c2: _i32(c2))
        for c1, c2 in [(1, 2), (2, 2), (3, 2)]
    ],
    ("nan_vs_nan", lambda a, b: _f32(np.nan), lambda a, b: _f32(np.nan)),
]


def _select_on(op, lhs, rhs):
    """Build ``f(x) = where(op(lhs, rhs), x[6], x[7])`` with bounded a and b.

    ``lax`` comparisons require equal dtypes,
    and argmax returns int64 when x64 is enabled,
    so both operands are promoted to their common dtype first.
    """

    def f(x):
        a = jnp.argmax(x[0:3])
        b = jnp.argmax(x[3:6])
        lhs_val, rhs_val = lhs(a, b), rhs(a, b)
        dtype = jnp.result_type(lhs_val, rhs_val)
        pred = op(jnp.asarray(lhs_val, dtype), jnp.asarray(rhs_val, dtype))
        return jnp.where(pred, x[6:7], x[7:8])

    return f


def _reaching_inputs():
    """Inputs that make (a, b) take every value in [0, 2] x [0, 2]."""
    xs = []
    for i, j in itertools.product(range(3), range(3)):
        x = np.zeros(_N_IN, dtype=np.float32)
        x[i] = 1.0
        x[3 + j] = 1.0
        x[6:] = [0.5, 0.7]
        xs.append(x)
    return xs


def _reachable_pattern(f, xs):
    """Union of the ``jax.jacobian`` patterns of ``f`` over ``xs``."""
    jacs = [np.abs(np.asarray(jax.jacobian(f)(x))) > 0 for x in xs]
    return np.any(jacs, axis=0).astype(int)


def _detected_pattern(f, x):
    return jacobian_sparsity(f, x).todense().astype(int)


@pytest.mark.control_flow
@pytest.mark.parametrize("op", _BOUNDED_OPS, ids=lambda op: op.__name__)
@pytest.mark.parametrize(
    ("lhs", "rhs"),
    [case[1:] for case in _OPERAND_CASES],
    ids=[case[0] for case in _OPERAND_CASES],
)
def test_bounds_resolution_matches_jax(op, lhs, rhs):
    """Bounds resolution of lt, le, gt, and ge matches ``jax.jacobian``.

    When the bounds prove the comparison, only the taken branch is tracked.
    When both outcomes are reachable, both branches are tracked.
    Either way the detected pattern equals the union over all reachable inputs.
    """
    f = _select_on(op, lhs, rhs)
    xs = _reaching_inputs()
    np.testing.assert_array_equal(
        _detected_pattern(f, xs[0]), _reachable_pattern(f, xs)
    )


@pytest.mark.control_flow
@pytest.mark.parametrize("op", [lax.eq, lax.ne], ids=lambda op: op.__name__)
@pytest.mark.parametrize(
    ("lhs", "rhs"),
    [
        (lambda a, b: a, lambda a, b: _i32(1)),
        (lambda a, b: a, lambda a, b: b),
        (lambda a, b: _i32(2), lambda a, b: _i32(2)),
        (lambda a, b: _i32(1), lambda a, b: _i32(2)),
        (lambda a, b: _f32(np.nan), lambda a, b: _f32(np.nan)),
    ],
    ids=["a_vs_1", "a_vs_b", "2_vs_2", "1_vs_2", "nan_vs_nan"],
)
def test_eq_ne_match_jax(op, lhs, rhs):
    """Eq and ne match ``jax.jacobian`` when the outcome is not decided by bounds.

    Covers bounded operands where both outcomes are reachable
    and const operands, which resolve through const propagation.
    """
    f = _select_on(op, lhs, rhs)
    xs = _reaching_inputs()
    np.testing.assert_array_equal(
        _detected_pattern(f, xs[0]), _reachable_pattern(f, xs)
    )


@pytest.mark.control_flow
@pytest.mark.fallback
@pytest.mark.parametrize(
    ("op", "lhs", "rhs", "precise"),
    [
        (lax.eq, lambda a, b: a, lambda a, b: _i32(5), _FALSE_BRANCH),
        (lax.ne, lambda a, b: a, lambda a, b: _i32(5), _TRUE_BRANCH),
        (lax.eq, lambda a, b: a * _i32(0), lambda a, b: _i32(0), _TRUE_BRANCH),
        (lax.ne, lambda a, b: a * _i32(0), lambda a, b: _i32(0), _FALSE_BRANCH),
        (lax.eq, lambda a, b: a, lambda a, b: b + _i32(3), _FALSE_BRANCH),
    ],
    ids=["a_eq_5", "a_ne_5", "a*0_eq_0", "a*0_ne_0", "a_eq_b+3"],
)
def test_eq_ne_ignore_bounds(op, lhs, rhs, precise):
    """Eq and ne do not use value bounds, so both branches are tracked.

    TODO(eq): resolve eq and ne from bounds.
    The result is always false when the bounds are disjoint,
    and always true when both operands are the same singleton.
    The precise pattern is the ``precise`` branch only.
    """
    f = _select_on(op, lhs, rhs)
    xs = _reaching_inputs()
    np.testing.assert_array_equal(_reachable_pattern(f, xs), precise)
    np.testing.assert_array_equal(_detected_pattern(f, xs[0]), _BOTH_BRANCHES)


@pytest.mark.control_flow
@pytest.mark.fallback
@pytest.mark.parametrize(
    ("op", "precise"),
    [
        (lax.lt, _FALSE_BRANCH),
        (lax.le, _TRUE_BRANCH),
        (lax.gt, _FALSE_BRANCH),
        (lax.ge, _TRUE_BRANCH),
    ],
    ids=lambda p: getattr(p, "__name__", ""),
)
def test_same_operand_unresolved(op, precise):
    """Comparing a value with itself is not resolved from bounds.

    Bounds track each operand independently,
    so ``a < a`` looks like ``[0, 2] < [0, 2]``, which can go either way.
    TODO(lt): ``a < a`` and ``a > a`` are always false,
    and ``a <= a`` and ``a >= a`` are always true.
    The precise pattern is the ``precise`` branch only.
    """
    f = _select_on(op, lambda a, b: a, lambda a, b: a)
    xs = _reaching_inputs()
    np.testing.assert_array_equal(_reachable_pattern(f, xs), precise)
    np.testing.assert_array_equal(_detected_pattern(f, xs[0]), _BOTH_BRANCHES)


# Vector operands


def _select_on_vector(op, thresholds):
    """Build ``f(x) = where(op(a, thresholds), x[6:8], x[8:10])``.

    ``a = argmax(x[0:6].reshape(2, 3), axis=1)`` has bounds [0, 2] per element.
    """

    def f(x):
        a = jnp.argmax(x[0:6].reshape(2, 3), axis=1)
        pred = op(a, jnp.asarray(thresholds, dtype=a.dtype))
        return jnp.where(pred, x[6:8], x[8:10])

    return f


def _vector_reaching_inputs():
    """Inputs that make both elements of a take every value in [0, 2]."""
    xs = []
    for i, j in itertools.product(range(3), range(3)):
        x = np.zeros(10, dtype=np.float32)
        x[i] = 1.0
        x[3 + j] = 1.0
        x[6:] = [0.2, 0.3, 0.5, 0.7]
        xs.append(x)
    return xs


_VECTOR_BOTH_BRANCHES = np.array(
    [
        [0, 0, 0, 0, 0, 0, 1, 0, 1, 0],  # out[0] = x[6] or x[8]
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],  # out[1] = x[7] or x[9]
    ]
)


@pytest.mark.control_flow
@pytest.mark.parametrize("op", _BOUNDED_OPS, ids=lambda op: op.__name__)
@pytest.mark.parametrize(
    "thresholds",
    [[-1, -1], [0, 0], [1, 1], [2, 2], [3, 3]],
    ids=["-1", "0", "1", "2", "3"],
)
def test_vector_bounds_resolution_matches_jax(op, thresholds):
    """Element-wise bounds resolve the mask when every element agrees.

    With the same threshold for every element,
    the elements are either all decided the same way or all undecided,
    so detection must match ``jax.jacobian`` exactly.
    """
    f = _select_on_vector(op, thresholds)
    xs = _vector_reaching_inputs()
    np.testing.assert_array_equal(
        _detected_pattern(f, xs[0]), _reachable_pattern(f, xs)
    )


@pytest.mark.control_flow
@pytest.mark.fallback
@pytest.mark.parametrize(
    ("op", "thresholds", "precise"),
    [
        (
            lax.lt,
            [3, -1],
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # out[0] = x[6]
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 1],  # out[1] = x[9]
                ]
            ),
        ),
        (
            lax.ge,
            [0, 1],
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # out[0] = x[6]
                    [0, 0, 0, 0, 0, 0, 0, 1, 0, 1],  # out[1] = x[7] or x[9]
                ]
            ),
        ),
    ],
    ids=["lt_decided_differently", "ge_partly_decided"],
)
def test_vector_mixed_resolution_unresolved(op, thresholds, precise):
    """A mask is only resolved when every element is decided the same way.

    ``a < [3, -1]`` is always ``[True, False]``,
    and ``a >= [0, 1]`` is always true in element 0 but undecided in element 1.
    The handler stores a const only when all elements agree,
    so both branches are tracked in every element.
    TODO(lt): resolve per element.
    The precise pattern is ``precise``.
    """
    f = _select_on_vector(op, thresholds)
    xs = _vector_reaching_inputs()
    np.testing.assert_array_equal(_reachable_pattern(f, xs), precise)
    assert_jacobian_sparsity_conservative(f, xs[0])
    np.testing.assert_array_equal(_detected_pattern(f, xs[0]), _VECTOR_BOTH_BRANCHES)


@pytest.mark.control_flow
def test_vector_broadcast_scalar_bounds():
    """A bounded scalar broadcast against a const vector resolves the whole mask.

    ``a`` in [0, 2] is less than every element of ``[3, 4]``,
    so only the true branch is tracked.
    """

    def f(x):
        a = jnp.argmax(x[0:3])
        return jnp.where(a < jnp.array([3, 4]), x[3:5], x[5:7])

    xs = []
    for i in range(3):
        x = np.zeros(7, dtype=np.float32)
        x[i] = 1.0
        xs.append(x)
    expected = np.array(
        [
            [0, 0, 0, 1, 0, 0, 0],  # out[0] = x[3]
            [0, 0, 0, 0, 1, 0, 0],  # out[1] = x[4]
        ]
    )
    np.testing.assert_array_equal(_reachable_pattern(f, xs), expected)
    np.testing.assert_array_equal(_detected_pattern(f, xs[0]), expected)


# Zero derivative without bounds


@pytest.mark.control_flow
@pytest.mark.parametrize(
    "op",
    [lax.lt, lax.le, lax.gt, lax.ge, lax.eq, lax.ne],
    ids=lambda op: op.__name__,
)
def test_zero_derivative(op):
    """Comparisons of unbounded floats have zero derivative."""

    def f(x):
        return op(x[:3], x[3:]).astype(jnp.float32)

    x = np.array([0.1, 0.5, 0.9, 0.3, 0.5, 0.7], dtype=np.float32)
    assert_jacobian_sparsity_exact(f, x)
    np.testing.assert_array_equal(_detected_pattern(f, x), np.zeros((3, 6), dtype=int))


@pytest.mark.control_flow
@pytest.mark.parametrize(
    "op",
    [lax.lt, lax.le, lax.gt, lax.ge, lax.eq, lax.ne],
    ids=lambda op: op.__name__,
)
def test_size_zero(op):
    """Comparisons of zero-element arrays produce an empty pattern."""

    def f(x):
        return jnp.where(op(x[:0], x[:0]), x[:0], -x[:0])

    result = jacobian_sparsity(f, np.zeros(3)).todense()
    assert result.shape == (0, 3)
