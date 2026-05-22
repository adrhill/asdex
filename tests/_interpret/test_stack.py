"""Tests for stack propagation.

https://docs.jax.dev/en/latest/_autosummary/jax.lax.stack.html
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity

# 1D inputs
# ---------


@pytest.mark.array_ops
def test_stack_1d_axis0():
    """Stack two 1D arrays along axis 0."""

    def f(x):
        a, b = x[:3], x[3:]
        return jnp.stack([a, b], axis=0)

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    expected = np.eye(6, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_stack_1d_axis1():
    """Stack two 1D arrays along axis 1 (interleaves elements)."""

    def f(x):
        a, b = x[:3], x[3:]
        return jnp.stack([a, b], axis=1)

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    # Output shape (3, 2): [[a0, b0], [a1, b1], [a2, b2]]
    # Flat output: [a0, b0, a1, b1, a2, b2] = [x0, x3, x1, x4, x2, x5]
    expected = np.array(
        [
            [1, 0, 0, 0, 0, 0],  # out[0,0] = a[0] = x[0]
            [0, 0, 0, 1, 0, 0],  # out[0,1] = b[0] = x[3]
            [0, 1, 0, 0, 0, 0],  # out[1,0] = a[1] = x[1]
            [0, 0, 0, 0, 1, 0],  # out[1,1] = b[1] = x[4]
            [0, 0, 1, 0, 0, 0],  # out[2,0] = a[2] = x[2]
            [0, 0, 0, 0, 0, 1],  # out[2,1] = b[2] = x[5]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_stack_1d_axis_negative():
    """Stack along axis=-1 (same as axis=1 for 1D inputs)."""

    def f(x):
        a, b = x[:2], x[2:]
        return jnp.stack([a, b], axis=-1)

    result = jacobian_sparsity(f, np.zeros(4)).todense().astype(int)
    # axis=-1 on 1D inputs is same as axis=1
    expected = np.array(
        [
            [1, 0, 0, 0],  # out[0,0] = a[0]
            [0, 0, 1, 0],  # out[0,1] = b[0]
            [0, 1, 0, 0],  # out[1,0] = a[1]
            [0, 0, 0, 1],  # out[1,1] = b[1]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_stack_1d_jacobian_verification():
    """Verify stack sparsity matches numerical Jacobian."""

    def f(x):
        a, b = x[:3], x[3:]
        return jnp.stack([a, b], axis=0)

    x = np.arange(6.0) + 1
    detected = jacobian_sparsity(f, x).todense().astype(int)
    numerical_full = jax.jacobian(f)(x)
    numerical = (np.abs(numerical_full.reshape(-1, x.size)) > 1e-10).astype(int)
    np.testing.assert_array_equal(detected, numerical)


# 2D inputs
# ---------


@pytest.mark.array_ops
def test_stack_2d_axis0():
    """Stack two 2D arrays along axis 0."""

    def f(x):
        a = x[:6].reshape(2, 3)
        b = x[6:].reshape(2, 3)
        return jnp.stack([a, b], axis=0)  # (2, 2, 3)

    result = jacobian_sparsity(f, np.zeros(12)).todense().astype(int)
    # Output shape (2, 2, 3): [[[a], [a]], [[b], [b]]]
    # First 6 outputs from a, next 6 from b
    expected = np.eye(12, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_stack_2d_axis1():
    """Stack two 2D arrays along axis 1."""

    def f(x):
        a = x[:6].reshape(2, 3)
        b = x[6:].reshape(2, 3)
        return jnp.stack([a, b], axis=1)  # (2, 2, 3)

    result = jacobian_sparsity(f, np.zeros(12)).todense().astype(int)
    # Output shape (2, 2, 3)
    # out[i, j, k] = inputs[j][i, k]
    # Flat: row 0 of a, row 0 of b, row 1 of a, row 1 of b
    expected = np.zeros((12, 12), dtype=int)
    # out[0,0,:] = a[0,:] = x[0:3], out[0,1,:] = b[0,:] = x[6:9]
    # out[1,0,:] = a[1,:] = x[3:6], out[1,1,:] = b[1,:] = x[9:12]
    for out_idx, in_idx in enumerate([0, 1, 2, 6, 7, 8, 3, 4, 5, 9, 10, 11]):
        expected[out_idx, in_idx] = 1
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_stack_2d_axis2():
    """Stack two 2D arrays along axis 2 (last axis)."""

    def f(x):
        a = x[:6].reshape(2, 3)
        b = x[6:].reshape(2, 3)
        return jnp.stack([a, b], axis=2)  # (2, 3, 2)

    result = jacobian_sparsity(f, np.zeros(12)).todense().astype(int)
    # Output shape (2, 3, 2)
    # out[i, j, k] = inputs[k][i, j]
    # Interleaved: a[0,0], b[0,0], a[0,1], b[0,1], ...
    expected = np.zeros((12, 12), dtype=int)
    for i in range(2):
        for j in range(3):
            out_flat_a = (i * 3 + j) * 2
            out_flat_b = (i * 3 + j) * 2 + 1
            in_flat = i * 3 + j
            expected[out_flat_a, in_flat] = 1
            expected[out_flat_b, in_flat + 6] = 1
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_stack_2d_nonsquare_jacobian_verification():
    """Verify 2D non-square stack matches numerical Jacobian."""

    def f(x):
        a = x[:6].reshape(2, 3)
        b = x[6:].reshape(2, 3)
        return jnp.stack([a, b], axis=1)

    x = np.arange(12.0) + 1
    detected = jacobian_sparsity(f, x).todense().astype(int)
    numerical_full = jax.jacobian(f)(x)
    numerical = (np.abs(numerical_full.reshape(-1, x.size)) > 1e-10).astype(int)
    np.testing.assert_array_equal(detected, numerical)


# 3D inputs
# ---------


@pytest.mark.array_ops
def test_stack_3d_axis0():
    """Stack two 3D arrays along axis 0."""

    def f(x):
        a = x[:24].reshape(2, 3, 4)
        b = x[24:].reshape(2, 3, 4)
        return jnp.stack([a, b], axis=0)  # (2, 2, 3, 4)

    result = jacobian_sparsity(f, np.zeros(48)).todense().astype(int)
    expected = np.eye(48, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_stack_3d_axis2():
    """Stack two 3D arrays along middle axis."""

    def f(x):
        a = x[:24].reshape(2, 3, 4)
        b = x[24:].reshape(2, 3, 4)
        return jnp.stack([a, b], axis=2)  # (2, 3, 2, 4)

    x = np.arange(48.0) + 1
    detected = jacobian_sparsity(f, x).todense().astype(int)
    numerical_full = jax.jacobian(f)(x)
    numerical = (np.abs(numerical_full.reshape(-1, x.size)) > 1e-10).astype(int)
    np.testing.assert_array_equal(detected, numerical)


# Multiple inputs
# ---------------


@pytest.mark.array_ops
def test_stack_three_inputs():
    """Stack three arrays."""

    def f(x):
        a, b, c = x[:2], x[2:4], x[4:]
        return jnp.stack([a, b, c], axis=0)  # (3, 2)

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    expected = np.eye(6, dtype=int)
    np.testing.assert_array_equal(result, expected)


# Edge cases
# ----------


@pytest.mark.array_ops
def test_stack_scalar_inputs():
    """Stack scalar arrays."""

    def f(x):
        a, b = x[0], x[1]
        return jnp.stack([a, b])

    result = jacobian_sparsity(f, np.zeros(2)).todense().astype(int)
    expected = np.eye(2, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_stack_single_input():
    """Stack a single array (adds dimension of size 1)."""

    def f(x):
        return jnp.stack([x], axis=0)  # (1, 3)

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_stack_zero_sized():
    """Stack zero-sized arrays."""

    def f(x):
        a = x[:0]
        b = x[:0]
        return jnp.stack([a, b], axis=0)  # (2, 0)

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    # Output is empty, so Jacobian has 0 rows
    expected = np.zeros((0, 3), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_stack_constants_no_dependency():
    """Stack of constants has no input dependency."""

    def f(x):
        a = jnp.array([1.0, 2.0])
        b = jnp.array([3.0, 4.0])
        return jnp.stack([a, b])

    result = jacobian_sparsity(f, np.zeros(2)).todense().astype(int)
    expected = np.zeros((4, 2), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_stack_mixed_constants():
    """Stack mixing input-dependent and constant arrays."""

    def f(x):
        a = x[:2]
        b = jnp.array([1.0, 2.0])
        return jnp.stack([a, b], axis=0)  # (2, 2)

    result = jacobian_sparsity(f, np.zeros(2)).todense().astype(int)
    # out[0,:] depends on x, out[1,:] is constant
    expected = np.array(
        [
            [1, 0],  # out[0,0] = a[0] = x[0]
            [0, 1],  # out[0,1] = a[1] = x[1]
            [0, 0],  # out[1,0] = b[0] (constant)
            [0, 0],  # out[1,1] = b[1] (constant)
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


# Const chain verification
# ------------------------


@pytest.mark.array_ops
def test_stack_const_propagation_with_gather():
    """Verify const values propagate through stack for downstream gather."""

    def f(x):
        # Create index arrays and stack them
        idx = jnp.array([0, 1])
        idx2 = jnp.array([1, 0])
        indices = jnp.stack([idx, idx2], axis=0)  # [[0, 1], [1, 0]]
        # Use stacked indices to gather from x
        return x[indices]  # Gathers x[0], x[1], x[1], x[0]

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.array(
        [
            [1, 0, 0],  # out[0,0] = x[0]
            [0, 1, 0],  # out[0,1] = x[1]
            [0, 1, 0],  # out[1,0] = x[1]
            [1, 0, 0],  # out[1,1] = x[0]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


# Compositions
# ------------


@pytest.mark.array_ops
def test_stack_then_unstack():
    """Stack then unstack via slicing should be identity."""

    def f(x):
        a, b = x[:3], x[3:]
        stacked = jnp.stack([a, b], axis=0)  # (2, 3)
        return jnp.concatenate([stacked[0], stacked[1]])  # back to (6,)

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    expected = np.eye(6, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_double_stack():
    """Stack of stacked arrays."""

    def f(x):
        a, b, c, d = x[:2], x[2:4], x[4:6], x[6:]
        s1 = jnp.stack([a, b], axis=0)  # (2, 2)
        s2 = jnp.stack([c, d], axis=0)  # (2, 2)
        return jnp.stack([s1, s2], axis=0)  # (2, 2, 2)

    result = jacobian_sparsity(f, np.zeros(8)).todense().astype(int)
    expected = np.eye(8, dtype=int)
    np.testing.assert_array_equal(result, expected)
