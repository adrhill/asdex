"""Tests for unstack propagation.

https://docs.jax.dev/en/latest/_autosummary/jax.lax.unstack.html
"""

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity

# 1D input


@pytest.mark.array_ops
def test_unstack_1d():
    """Unstack a 1D array into scalars."""

    def f(x):
        # x has shape (3,), unstack produces 3 scalars
        parts = lax.unstack(x, axis=0)
        return jnp.stack(parts)  # back to (3,)

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_unstack_1d_jacobian_verification():
    """Verify 1D unstack sparsity matches numerical Jacobian."""

    def f(x):
        parts = lax.unstack(x, axis=0)
        return jnp.stack(parts)

    x = np.arange(4.0) + 1
    detected = jacobian_sparsity(f, x).todense().astype(int)
    numerical_full = jax.jacobian(f)(x)
    numerical = (np.abs(numerical_full.reshape(-1, x.size)) > 1e-10).astype(int)
    np.testing.assert_array_equal(detected, numerical)


# 2D inputs


@pytest.mark.array_ops
def test_unstack_2d_axis0():
    """Unstack a 2D array along axis 0."""

    def f(x):
        arr = x.reshape(2, 3)
        parts = lax.unstack(arr, axis=0)  # 2 arrays of shape (3,)
        return jnp.concatenate(parts)

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    expected = np.eye(6, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_unstack_2d_axis1():
    """Unstack a 2D array along axis 1."""

    def f(x):
        arr = x.reshape(2, 3)
        parts = lax.unstack(arr, axis=1)  # 3 arrays of shape (2,)
        return jnp.concatenate(parts)

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    # Output: [arr[0,0], arr[1,0], arr[0,1], arr[1,1], arr[0,2], arr[1,2]]
    # Input flat: [arr[0,0], arr[0,1], arr[0,2], arr[1,0], arr[1,1], arr[1,2]]
    expected = np.array(
        [
            [1, 0, 0, 0, 0, 0],  # out[0] = arr[0,0] = x[0]
            [0, 0, 0, 1, 0, 0],  # out[1] = arr[1,0] = x[3]
            [0, 1, 0, 0, 0, 0],  # out[2] = arr[0,1] = x[1]
            [0, 0, 0, 0, 1, 0],  # out[3] = arr[1,1] = x[4]
            [0, 0, 1, 0, 0, 0],  # out[4] = arr[0,2] = x[2]
            [0, 0, 0, 0, 0, 1],  # out[5] = arr[1,2] = x[5]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_unstack_2d_axis_negative():
    """Unstack along axis=-1 (same as axis=1 for 2D)."""

    def f(x):
        arr = x.reshape(2, 3)
        parts = lax.unstack(arr, axis=-1)  # 3 arrays of shape (2,)
        return jnp.concatenate(parts)

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    # Same as axis=1
    expected = np.array(
        [
            [1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_unstack_2d_nonsquare_jacobian_verification():
    """Verify 2D non-square unstack matches numerical Jacobian."""

    def f(x):
        arr = x.reshape(2, 4)
        parts = lax.unstack(arr, axis=1)
        return jnp.concatenate(parts)

    x = np.arange(8.0) + 1
    detected = jacobian_sparsity(f, x).todense().astype(int)
    numerical_full = jax.jacobian(f)(x)
    numerical = (np.abs(numerical_full.reshape(-1, x.size)) > 1e-10).astype(int)
    np.testing.assert_array_equal(detected, numerical)


# 3D inputs


@pytest.mark.array_ops
def test_unstack_3d_axis0():
    """Unstack a 3D array along axis 0."""

    def f(x):
        arr = x.reshape(2, 3, 4)
        parts = lax.unstack(arr, axis=0)  # 2 arrays of shape (3, 4)
        return jnp.concatenate([p.ravel() for p in parts])

    result = jacobian_sparsity(f, np.zeros(24)).todense().astype(int)
    expected = np.eye(24, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_unstack_3d_axis1():
    """Unstack a 3D array along axis 1."""

    def f(x):
        arr = x.reshape(2, 3, 4)
        parts = lax.unstack(arr, axis=1)  # 3 arrays of shape (2, 4)
        return jnp.concatenate([p.ravel() for p in parts])

    x = np.arange(24.0) + 1
    detected = jacobian_sparsity(f, x).todense().astype(int)
    numerical_full = jax.jacobian(f)(x)
    numerical = (np.abs(numerical_full.reshape(-1, x.size)) > 1e-10).astype(int)
    np.testing.assert_array_equal(detected, numerical)


@pytest.mark.array_ops
def test_unstack_3d_axis2():
    """Unstack a 3D array along axis 2 (last axis)."""

    def f(x):
        arr = x.reshape(2, 3, 4)
        parts = lax.unstack(arr, axis=2)  # 4 arrays of shape (2, 3)
        return jnp.concatenate([p.ravel() for p in parts])

    x = np.arange(24.0) + 1
    detected = jacobian_sparsity(f, x).todense().astype(int)
    numerical_full = jax.jacobian(f)(x)
    numerical = (np.abs(numerical_full.reshape(-1, x.size)) > 1e-10).astype(int)
    np.testing.assert_array_equal(detected, numerical)


# 4D inputs


@pytest.mark.array_ops
def test_unstack_4d_axis2():
    """Unstack a 4D array along a middle axis."""

    def f(x):
        arr = x.reshape(2, 3, 4, 5)
        parts = lax.unstack(arr, axis=2)  # 4 arrays of shape (2, 3, 5)
        return jnp.concatenate([p.ravel() for p in parts])

    x = np.arange(120.0) + 1
    detected = jacobian_sparsity(f, x).todense().astype(int)
    numerical_full = jax.jacobian(f)(x)
    numerical = (np.abs(numerical_full.reshape(-1, x.size)) > 1e-10).astype(int)
    np.testing.assert_array_equal(detected, numerical)


# Edge cases


@pytest.mark.array_ops
def test_unstack_size_one_axis():
    """Unstack along an axis of size 1 produces a single output."""

    def f(x):
        arr = x.reshape(1, 4)
        parts = lax.unstack(arr, axis=0)  # 1 array of shape (4,)
        return parts[0]

    result = jacobian_sparsity(f, np.zeros(4)).todense().astype(int)
    expected = np.eye(4, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_unstack_zero_sized():
    """Unstack where outputs are zero-sized."""

    def f(x):
        arr = x.reshape(2, 0)
        parts = lax.unstack(arr, axis=0)  # 2 arrays of shape (0,)
        return jnp.concatenate(parts)  # shape (0,)

    result = jacobian_sparsity(f, np.zeros(0)).todense().astype(int)
    expected = np.zeros((0, 0), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_unstack_constants_no_dependency():
    """Unstack of constants has no input dependency."""

    def f(x):
        arr = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        parts = lax.unstack(arr, axis=0)
        return jnp.concatenate(parts)

    result = jacobian_sparsity(f, np.zeros(2)).todense().astype(int)
    expected = np.zeros((6, 2), dtype=int)
    np.testing.assert_array_equal(result, expected)


# Const chain verification


@pytest.mark.array_ops
def test_unstack_const_propagation_with_gather():
    """Verify const values propagate through unstack for downstream gather."""

    def f(x):
        # Create index array [[0, 1], [1, 0]] and unstack
        indices = jnp.array([[0, 1], [1, 0]])
        idx_parts = lax.unstack(indices, axis=0)  # [0, 1] and [1, 0]
        # Use unstacked indices to gather from x
        return jnp.concatenate([x[idx_parts[0]], x[idx_parts[1]]])

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.array(
        [
            [1, 0, 0],  # out[0] = x[0]
            [0, 1, 0],  # out[1] = x[1]
            [0, 1, 0],  # out[2] = x[1]
            [1, 0, 0],  # out[3] = x[0]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


# Compositions


@pytest.mark.array_ops
def test_unstack_then_stack():
    """Unstack then stack should be identity."""

    def f(x):
        arr = x.reshape(2, 3)
        parts = lax.unstack(arr, axis=0)
        return jnp.stack(parts, axis=0).ravel()

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    expected = np.eye(6, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_stack_then_unstack():
    """Stack then unstack should be identity."""

    def f(x):
        a, b = x[:3], x[3:]
        stacked = jnp.stack([a, b], axis=0)  # (2, 3)
        parts = lax.unstack(stacked, axis=0)  # back to 2 arrays of (3,)
        return jnp.concatenate(parts)

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    expected = np.eye(6, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_double_unstack():
    """Unstack twice."""

    def f(x):
        arr = x.reshape(2, 3, 4)
        parts1 = lax.unstack(arr, axis=0)  # 2 arrays of (3, 4)
        all_parts = []
        for p in parts1:
            parts2 = lax.unstack(p, axis=0)  # 3 arrays of (4,)
            all_parts.extend(parts2)
        return jnp.concatenate(all_parts)

    result = jacobian_sparsity(f, np.zeros(24)).todense().astype(int)
    expected = np.eye(24, dtype=int)
    np.testing.assert_array_equal(result, expected)


# Non-contiguous patterns


@pytest.mark.array_ops
def test_unstack_non_contiguous_input():
    """Unstack input with non-contiguous dependencies (from broadcast)."""

    def f(x):
        # Broadcast x to (2, 3) then unstack
        arr = jnp.broadcast_to(x, (2, 3))  # each row depends on all of x
        parts = lax.unstack(arr, axis=0)  # 2 arrays of (3,)
        return jnp.concatenate(parts)

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.array(
        [
            [1, 0, 0],  # out[0] = x[0]
            [0, 1, 0],  # out[1] = x[1]
            [0, 0, 1],  # out[2] = x[2]
            [1, 0, 0],  # out[3] = x[0] (second row)
            [0, 1, 0],  # out[4] = x[1]
            [0, 0, 1],  # out[5] = x[2]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_unstack_after_reduction():
    """Unstack after a reduction that unions dependencies."""

    def f(x):
        arr = x.reshape(2, 3)
        sums = jnp.sum(arr, axis=1, keepdims=True)  # (2, 1) - each depends on row
        broadcast = jnp.broadcast_to(sums, (2, 3))  # (2, 3)
        parts = lax.unstack(broadcast, axis=0)  # 2 arrays of (3,)
        return jnp.concatenate(parts)

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    # First 3 outputs depend on first row (x[0:3])
    # Last 3 outputs depend on second row (x[3:6])
    expected = np.array(
        [
            [1, 1, 1, 0, 0, 0],  # out[0] depends on row 0
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 0, 0, 0],
            [0, 0, 0, 1, 1, 1],  # out[3] depends on row 1
            [0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)
