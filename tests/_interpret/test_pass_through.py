"""Tests for the pass-through propagation handler.

``optimization_barrier`` and ``device_put`` return their operands unchanged,
so index sets, const values, and bounds pass straight through.

https://docs.jax.dev/en/latest/_autosummary/jax.lax.optimization_barrier.html
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from asdex import jacobian_sparsity
from asdex.detection._interpret import _prop_jaxpr
from tests._utils import assert_jacobian_sparsity_exact

_PASS_THROUGHS = [
    pytest.param(lax.optimization_barrier, id="optimization_barrier"),
    pytest.param(jax.device_put, id="device_put"),
]


@pytest.mark.array_ops
@pytest.mark.parametrize("pass_through", _PASS_THROUGHS)
def test_pass_through_single_operand(pass_through):
    """A single 2D operand passes through with an identity Jacobian."""

    def f(x):
        return pass_through(jnp.sin(x))

    x = np.arange(1.0, 7.0).reshape(2, 3)
    result = jacobian_sparsity(f, x).todense().astype(int)
    np.testing.assert_array_equal(result, np.eye(6, dtype=int))
    assert_jacobian_sparsity_exact(f, x)


@pytest.mark.array_ops
@pytest.mark.parametrize("pass_through", _PASS_THROUGHS)
def test_pass_through_3d_composed_with_reshape(pass_through):
    """A pass-through inside a 3D computation keeps one dependency per element."""

    def f(x):
        return (2.0 * pass_through(x.reshape(2, 3, 4)))[0].ravel()

    x = np.arange(1.0, 25.0)
    expected = np.hstack([np.eye(12, dtype=int), np.zeros((12, 12), dtype=int)])
    np.testing.assert_array_equal(
        jacobian_sparsity(f, x).todense().astype(int), expected
    )
    assert_jacobian_sparsity_exact(f, x)


@pytest.mark.array_ops
@pytest.mark.parametrize("pass_through", _PASS_THROUGHS)
def test_pass_through_aliases_index_sets(pass_through):
    """The handler aliases the input's index sets instead of copying them."""
    jaxpr = jax.make_jaxpr(pass_through)(jnp.zeros(3)).jaxpr
    input_indices = [[{0}, {1}, {2}]]

    result = _prop_jaxpr(jaxpr, input_indices)

    assert result[0] is input_indices[0]


@pytest.mark.array_ops
@pytest.mark.parametrize("pass_through", _PASS_THROUGHS)
def test_pass_through_multiple_operands(pass_through):
    """Each output keeps the index sets of its own operand, in order."""

    def f(x):
        a, b = pass_through((x[:2] * 2.0, jnp.sum(x)))
        return jnp.concatenate([a, b[None]])

    x = np.array([1.0, 2.0, 3.0])
    result = jacobian_sparsity(f, x).todense().astype(int)
    expected = np.array(
        [
            [1, 0, 0],  # a[0] = 2 * x[0]
            [0, 1, 0],  # a[1] = 2 * x[1]
            [1, 1, 1],  # b = sum(x)
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)
    assert_jacobian_sparsity_exact(f, x)


@pytest.mark.array_ops
@pytest.mark.parametrize("pass_through", _PASS_THROUGHS)
def test_pass_through_feeds_gather(pass_through):
    """A gather downstream of a pass-through still resolves its operand."""

    def f(x):
        return jnp.take(pass_through(x), jnp.array([2, 0, 2]))

    x = np.arange(1.0, 4.0)
    expected = np.array(
        [
            [0, 0, 1],  # out[0] = x[2]
            [1, 0, 0],  # out[1] = x[0]
            [0, 0, 1],  # out[2] = x[2]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(
        jacobian_sparsity(f, x).todense().astype(int), expected
    )
    assert_jacobian_sparsity_exact(f, x)


@pytest.mark.array_ops
@pytest.mark.parametrize("pass_through", _PASS_THROUGHS)
def test_pass_through_forwards_const_index(pass_through):
    """A const index array passing through still resolves a gather."""

    def f(x):
        idx, y = pass_through((jnp.array([2, 0, 1]), x))
        return y[idx]

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.array(
        [
            [0, 0, 1],  # out[0] = x[2]
            [1, 0, 0],  # out[1] = x[0]
            [0, 1, 0],  # out[2] = x[1]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
@pytest.mark.parametrize("pass_through", _PASS_THROUGHS)
def test_pass_through_scalar_and_zero_size(pass_through):
    """Scalar and zero-size operands pass through without crashing."""

    def f(x):
        s, e = pass_through((x[0], x[:0]))
        return jnp.concatenate([s[None], e])

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.array([[1, 0, 0]], dtype=int)  # out[0] = x[0]
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_device_put_explicit_device():
    """Placing an array on an explicit device keeps its pattern."""

    def f(x):
        return jax.device_put(x[::-1], jax.devices()[0])

    x = np.array([1.0, 2.0, 3.0])
    result = jacobian_sparsity(f, x).todense().astype(int)
    np.testing.assert_array_equal(result, np.eye(3, dtype=int)[::-1])
    assert_jacobian_sparsity_exact(f, x)
