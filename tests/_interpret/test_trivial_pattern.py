"""Tests for primitives that get a whole-equation empty or dense pattern.

Those with a module of their own are tested there.
This file covers the rest:
``mulhi`` (integer-valued), ``debug_print`` and ``debug_callback`` (no outputs),
which get an empty pattern,
and ``pure_callback``, whose body is not visible,
which gets a dense pattern without being reported.
"""

import logging
import warnings

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from asdex import DensePatternWarning, jacobian_sparsity
from tests._utils import assert_jacobian_sparsity_conservative

# Empty pattern


@pytest.mark.elementwise
def test_mulhi_zero_derivative():
    """``mulhi`` is integer-valued, so it adds no dependencies to ``x * mulhi(u, u)``.

    The operand is fed through ``astype`` so that it carries dependencies.
    """

    def f(x):
        u = x.astype(jnp.uint32)
        return x * lax.mulhi(u, u).astype(jnp.float32)

    result = jacobian_sparsity(f, np.ones(3))
    np.testing.assert_array_equal(result.todense(), np.eye(3))
    assert result._fallbacks == ()


def _print(x):
    jax.debug.print("{}", x)


def _print_ordered(x):
    jax.debug.print("{}", x, ordered=True)


def _callback(x):
    jax.debug.callback(lambda v: None, x)


_DEBUG = pytest.mark.parametrize(
    "debug",
    [_print, _print_ordered, _callback],
    ids=["debug_print", "debug_print_ordered", "debug_callback"],
)


@_DEBUG
def test_debug_is_silent(debug, caplog):
    """Debug primitives have no outputs, so they are neither reported nor logged."""

    def f(x):
        debug(x)
        return x * 2

    with caplog.at_level(logging.INFO, logger="asdex.detection"):
        result = jacobian_sparsity(f, np.ones(3))

    np.testing.assert_array_equal(result.todense(), np.eye(3))
    assert result._fallbacks == ()
    assert caplog.records == []


@_DEBUG
def test_debug_not_blamed_for_dense_pattern(debug):
    """A dense pattern next to a debug primitive does not name it in the warning."""

    def f(x):
        debug(x)
        return jnp.sum(x) * jnp.ones(2)

    with pytest.warns(DensePatternWarning) as record:
        result = jacobian_sparsity(f, np.ones(3))

    assert result._fallbacks == ()
    message = str(record[0].message)
    assert "debug" not in message
    assert "No primitives without a precise handler were involved" in message


# Dense pattern


def _sin_callback(x):
    return jax.pure_callback(np.sin, jax.ShapeDtypeStruct(x.shape, x.dtype), x)


@jax.custom_jvp
def _sin_callback_with_jvp(x):
    return _sin_callback(x)


@_sin_callback_with_jvp.defjvp
def _sin_callback_jvp(primals, tangents):
    (x,), (t,) = primals, tangents
    return _sin_callback_with_jvp(x), jnp.cos(x) * t


@pytest.mark.parametrize(
    "f",
    [_sin_callback, _sin_callback_with_jvp],
    ids=["bare", "custom_jvp"],
)
def test_pure_callback_dense_and_silent(f, caplog):
    """``pure_callback`` gets a dense pattern without being reported.

    Its body is not visible in the jaxpr, so no tighter pattern can be derived.
    Wrapping it in ``jax.custom_jvp`` is the only form ``jax.jacobian`` accepts.
    Detection still follows the primal and meets the ``pure_callback`` inside,
    so the pattern is dense although the rule here is diagonal.
    """
    x = np.arange(1.0, 4.0)
    with (
        warnings.catch_warnings(),
        caplog.at_level(logging.INFO, logger="asdex.detection"),
    ):
        warnings.simplefilter("ignore", DensePatternWarning)
        result = jacobian_sparsity(f, x)

    np.testing.assert_array_equal(result.todense(), np.ones((3, 3)))
    assert result._fallbacks == ()
    assert caplog.records == []
    if f is _sin_callback_with_jvp:
        assert_jacobian_sparsity_conservative(f, x)
