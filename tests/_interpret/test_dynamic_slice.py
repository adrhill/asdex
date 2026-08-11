"""Tests for dynamic_slice and dynamic_update_slice propagation.

https://docs.jax.dev/en/latest/_autosummary/jax.lax.dynamic_slice.html
https://docs.jax.dev/en/latest/_autosummary/jax.lax.dynamic_update_slice.html
"""

import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity

# dynamic_slice


@pytest.mark.array_ops
def test_dynamic_slice_static_start():
    """dynamic_slice with static start indices tracks precise dependencies.

    This is the pattern emitted by jnp.split.
    """

    def f(x):
        return lax.dynamic_slice(x, (1,), (3,))

    result = jacobian_sparsity(f, np.zeros(5)).todense().astype(int)
    # out = x[1:4], so out[0]←{1}, out[1]←{2}, out[2]←{3}
    expected = np.array(
        [
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
@pytest.mark.fallback
def test_dynamic_slice_dynamic_start():
    """dynamic_slice with bounded dynamic start enumerates possible windows.

    TODO(dynamic_slice): the actual slice selects exactly one window,
    so each output depends on exactly one input (shifted identity).
    Detection takes the union over all possible start positions,
    reporting up to two dependencies per output element.
    """

    def f(x):
        # Start index depends on input → dynamic
        idx = jnp.argmax(x[:2])
        return lax.dynamic_slice(x, (idx,), (3,))

    result = jacobian_sparsity(f, np.zeros(5)).todense().astype(int)
    expected = np.array(
        [[1, 1, 0, 0, 0], [0, 1, 1, 0, 0], [0, 0, 1, 1, 0]],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_dynamic_slice_swap_halves():
    """Two dynamic_slice calls with static starts swap halves precisely."""

    def f(x):
        first_half = lax.dynamic_slice(x, (0,), (2,))
        second_half = lax.dynamic_slice(x, (2,), (2,))
        return jnp.concatenate([second_half, first_half])

    result = jacobian_sparsity(f, np.zeros(4)).todense().astype(int)
    expected = np.array(
        [
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [1, 0, 0, 0],
            [0, 1, 0, 0],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


# dynamic_update_slice


@pytest.mark.array_ops
def test_dynamic_update_slice_static_start():
    """dynamic_update_slice with static start replaces a sub-region precisely."""

    def f(x):
        arr = jnp.zeros(5)
        return lax.dynamic_update_slice(arr, x[:2], (1,))

    result = jacobian_sparsity(f, np.zeros(4)).todense().astype(int)
    # out = [0, x[0], x[1], 0, 0]
    expected = np.array(
        [
            [0, 0, 0, 0],  # out[0] ← constant 0
            [1, 0, 0, 0],  # out[1] ← x[0]
            [0, 1, 0, 0],  # out[2] ← x[1]
            [0, 0, 0, 0],  # out[3] ← constant 0
            [0, 0, 0, 0],  # out[4] ← constant 0
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
@pytest.mark.fallback
def test_dynamic_update_slice_dynamic_start():
    """dynamic_update_slice with bounded dynamic start enumerates possible windows.

    TODO(dynamic_update_slice): the actual update targets exactly one window,
    so each output depends on at most one update element.
    Detection takes the union over all possible start positions,
    reporting dependencies from both possible windows.
    """

    def f(x):
        idx = jnp.argmax(x[:2])
        return lax.dynamic_update_slice(x, x[2:4], (idx,))

    result = jacobian_sparsity(f, np.zeros(4)).todense().astype(int)
    expected = np.array(
        [[1, 0, 1, 0], [0, 0, 1, 1], [0, 0, 1, 1], [0, 0, 0, 1]],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_dynamic_slice_2d():
    """dynamic_slice on a 2D array with static starts.

    Slicing mat[1:3, 1:4] from a 3x4 matrix
    picks flat indices 5, 6, 7, 9, 10, 11.
    """

    def f(x):
        mat = x.reshape(3, 4)
        return lax.dynamic_slice(mat, (1, 1), (2, 3)).ravel()

    result = jacobian_sparsity(f, np.zeros(12)).todense().astype(int)
    expected = np.zeros((6, 12), dtype=int)
    expected[0, 5] = 1  # (1,1)
    expected[1, 6] = 1  # (1,2)
    expected[2, 7] = 1  # (1,3)
    expected[3, 9] = 1  # (2,1)
    expected[4, 10] = 1  # (2,2)
    expected[5, 11] = 1  # (2,3)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_dynamic_update_slice_2d():
    """dynamic_update_slice on a 2D array with static starts.

    Inserting a 2x3 update at position (0, 1) in a 3x4 matrix
    fills flat positions 1, 2, 3, 5, 6, 7.
    """

    def f(x):
        mat = jnp.zeros((3, 4))
        update = x[:6].reshape(2, 3)
        return lax.dynamic_update_slice(mat, update, (0, 1)).ravel()

    result = jacobian_sparsity(f, np.zeros(8)).todense().astype(int)
    expected = np.zeros((12, 8), dtype=int)
    expected[1, 0] = 1  # (0,1) ← x[0]
    expected[2, 1] = 1  # (0,2) ← x[1]
    expected[3, 2] = 1  # (0,3) ← x[2]
    expected[5, 3] = 1  # (1,1) ← x[3]
    expected[6, 4] = 1  # (1,2) ← x[4]
    expected[7, 5] = 1  # (1,3) ← x[5]
    np.testing.assert_array_equal(result, expected)


# Input-dependent start indices


@pytest.mark.array_ops
@pytest.mark.fallback
def test_dynamic_slice_input_dependent_start_conservative():
    """dynamic_slice with an unbounded input-dependent start goes conservative.

    The start index derives from the input without known bounds,
    so any window is possible and every output
    must depend on all operand elements and the start's own dependencies.
    Raising is wrong here, this is valid user code.

    TODO(dynamic_slice): the true pattern selects a single window,
    one dependency per output element.
    """

    def f(x):
        start = x[0].astype(jnp.int32)
        return lax.dynamic_slice(x, (start,), (2,))

    result = jacobian_sparsity(f, np.zeros(4)).todense().astype(int)
    expected = np.ones((2, 4), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
@pytest.mark.fallback
def test_dynamic_update_slice_input_dependent_start_conservative():
    """dynamic_update_slice with an unbounded input-dependent start goes conservative.

    Any window may receive the update,
    so every output must depend on the operand, the update,
    and the start's own dependencies.
    Raising is wrong here, this is valid user code.

    TODO(dynamic_update_slice): the true pattern updates a single window,
    all other positions keep their identity dependency.
    """

    def f(x):
        start = x[0].astype(jnp.int32)
        return lax.dynamic_update_slice(x, x[2:4], (start,))

    result = jacobian_sparsity(f, np.zeros(4)).todense().astype(int)
    expected = np.ones((4, 4), dtype=int)
    np.testing.assert_array_equal(result, expected)


# Size-0 dimension


@pytest.mark.array_ops
def test_dynamic_slice_zero_size():
    """Dynamic-slicing to a zero-sized output produces an empty Jacobian."""

    def f(x):
        return lax.dynamic_slice(x[:0], (0,), (0,))

    result = jacobian_sparsity(f, np.zeros(3))
    assert result.shape == (0, 3)
    assert result.nnz == 0
