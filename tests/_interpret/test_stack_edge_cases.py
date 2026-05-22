"""Edge case tests for stack and unstack handlers.

These tests cover cases not in the main test files:
- vmap interaction
- 5D arrays
- Deeply nested stack/unstack
- lax.stack (vs jnp.stack)
- Aliased inputs (same array stacked twice)
- Prime dimensions
- Conservative fallbacks
"""

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity

# vmap interaction
# ----------------


@pytest.mark.vmap
def test_stack_inside_vmap():
    """Stack operation inside vmap."""

    def inner(x):
        a, b = x[:2], x[2:]
        return jnp.stack([a, b], axis=0)

    def f(x):
        batched = x.reshape(3, 4)
        return jax.vmap(inner)(batched).ravel()

    x = np.arange(12.0) + 1
    detected = jacobian_sparsity(f, x).todense().astype(int)
    numerical_full = jax.jacobian(f)(x)
    numerical = (np.abs(numerical_full.reshape(-1, x.size)) > 1e-10).astype(int)
    np.testing.assert_array_equal(detected, numerical)


@pytest.mark.vmap
def test_unstack_inside_vmap():
    """Unstack operation inside vmap."""

    def inner(x):
        arr = x.reshape(2, 2)
        parts = lax.unstack(arr, axis=0)
        return jnp.concatenate(parts)

    def f(x):
        batched = x.reshape(3, 4)
        return jax.vmap(inner)(batched).ravel()

    x = np.arange(12.0) + 1
    detected = jacobian_sparsity(f, x).todense().astype(int)
    numerical_full = jax.jacobian(f)(x)
    numerical = (np.abs(numerical_full.reshape(-1, x.size)) > 1e-10).astype(int)
    np.testing.assert_array_equal(detected, numerical)


# 5D arrays
# ---------


@pytest.mark.array_ops
def test_stack_5d():
    """Stack 5D arrays at middle axis."""

    def f(x):
        a = x[:120].reshape(2, 3, 4, 5, 1)
        b = x[120:].reshape(2, 3, 4, 5, 1)
        return jnp.stack([a, b], axis=2)

    x = np.arange(240.0) + 1
    detected = jacobian_sparsity(f, x).todense().astype(int)
    numerical_full = jax.jacobian(f)(x)
    numerical = (np.abs(numerical_full.reshape(-1, x.size)) > 1e-10).astype(int)
    np.testing.assert_array_equal(detected, numerical)


@pytest.mark.array_ops
def test_unstack_5d():
    """Unstack 5D array along middle axis."""

    def f(x):
        arr = x.reshape(2, 3, 4, 5, 1)
        parts = lax.unstack(arr, axis=2)  # 4 arrays of (2, 3, 5, 1)
        return jnp.concatenate([p.ravel() for p in parts])

    x = np.arange(120.0) + 1
    detected = jacobian_sparsity(f, x).todense().astype(int)
    numerical_full = jax.jacobian(f)(x)
    numerical = (np.abs(numerical_full.reshape(-1, x.size)) > 1e-10).astype(int)
    np.testing.assert_array_equal(detected, numerical)


# Deep nesting
# ------------


@pytest.mark.array_ops
def test_triple_stack():
    """Stack of stacked of stacked arrays."""

    def f(x):
        parts = [x[i * 2 : (i + 1) * 2] for i in range(8)]
        s1 = jnp.stack(parts[:4], axis=0)  # (4, 2)
        s2 = jnp.stack(parts[4:], axis=0)  # (4, 2)
        return jnp.stack([s1, s2], axis=0)  # (2, 4, 2)

    result = jacobian_sparsity(f, np.zeros(16)).todense().astype(int)
    expected = np.eye(16, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_triple_unstack():
    """Unstack three levels deep."""

    def f(x):
        arr = x.reshape(2, 4, 2)
        parts1 = lax.unstack(arr, axis=0)  # 2 arrays of (4, 2)
        all_parts = []
        for p1 in parts1:
            parts2 = lax.unstack(p1, axis=0)  # 4 arrays of (2,)
            for p2 in parts2:
                parts3 = lax.unstack(p2, axis=0)  # 2 scalars
                all_parts.extend(parts3)
        return jnp.stack(all_parts)

    result = jacobian_sparsity(f, np.zeros(16)).todense().astype(int)
    expected = np.eye(16, dtype=int)
    np.testing.assert_array_equal(result, expected)


# lax.stack direct
# ----------------


@pytest.mark.array_ops
def test_lax_stack_direct():
    """Use lax.stack directly instead of jnp.stack."""

    def f(x):
        a, b = x[:3], x[3:]
        return lax.stack([a, b], axis=0)

    x = np.arange(6.0) + 1
    detected = jacobian_sparsity(f, x).todense().astype(int)
    numerical_full = jax.jacobian(f)(x)
    numerical = (np.abs(numerical_full.reshape(-1, x.size)) > 1e-10).astype(int)
    np.testing.assert_array_equal(detected, numerical)


# Aliased inputs
# --------------


@pytest.mark.array_ops
def test_stack_same_array_twice():
    """Stack the same array with itself."""

    def f(x):
        return jnp.stack([x, x], axis=0)

    x = np.arange(3.0) + 1
    detected = jacobian_sparsity(f, x).todense().astype(int)
    # Output shape (2, 3), both rows have same dependencies as input
    expected = np.array(
        [
            [1, 0, 0],  # out[0,0] = x[0]
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 0],  # out[1,0] = x[0] (same array)
            [0, 1, 0],
            [0, 0, 1],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(detected, expected)


# Prime dimensions
# ----------------


@pytest.mark.array_ops
def test_stack_prime_dims():
    """Stack arrays with prime number dimensions (no common factors)."""

    def f(x):
        a = x[:77].reshape(7, 11)
        b = x[77:].reshape(7, 11)
        return jnp.stack([a, b], axis=1)  # (7, 2, 11)

    x = np.arange(154.0) + 1
    detected = jacobian_sparsity(f, x).todense().astype(int)
    numerical_full = jax.jacobian(f)(x)
    numerical = (np.abs(numerical_full.reshape(-1, x.size)) > 1e-10).astype(int)
    np.testing.assert_array_equal(detected, numerical)


@pytest.mark.array_ops
def test_unstack_prime_dims():
    """Unstack array with prime number dimensions."""

    def f(x):
        arr = x.reshape(7, 11)
        parts = lax.unstack(arr, axis=0)  # 7 arrays of (11,)
        return jnp.concatenate(parts)

    x = np.arange(77.0) + 1
    detected = jacobian_sparsity(f, x).todense().astype(int)
    numerical_full = jax.jacobian(f)(x)
    numerical = (np.abs(numerical_full.reshape(-1, x.size)) > 1e-10).astype(int)
    np.testing.assert_array_equal(detected, numerical)


# Conservative fallbacks
# ----------------------


@pytest.mark.control_flow
@pytest.mark.fallback
def test_stack_inside_cond():
    """Stack inside conditional branch.

    TODO(cond): Conservative fallback.
    Expected precise pattern is identity (stacking a with b in true branch).
    """

    def f(x):
        def true_fn(x):
            a, b = x[:3], x[3:]
            return jnp.stack([a, b], axis=0)

        def false_fn(x):
            a, b = x[:3], x[3:]
            return jnp.stack([b, a], axis=0)

        return lax.cond(True, true_fn, false_fn, x)

    result = jacobian_sparsity(f, np.arange(6.0) + 1).todense().astype(int)
    # Conservative: unions both branches
    expected = np.array(
        [
            [1, 0, 0, 1, 0, 0],  # out[0] could be a[0] or b[0]
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
@pytest.mark.fallback
def test_stack_mixed_const_nonconst():
    """Stack const with non-const, gather using const part.

    TODO(stack): Conservative fallback.
    When stacking const with non-const, the result is not tracked as const,
    so downstream gather falls back to conservative.
    Expected precise pattern would be [[1,0,0,0], [0,0,1,0]] (x[0] and x[2]).
    """

    def f(x):
        const_idx = jnp.array([0, 2])
        nonconst = x[:2].astype(jnp.int32)
        stacked = jnp.stack([const_idx, nonconst], axis=0)
        return x[stacked[0]]  # x[[0, 2]]

    result = jacobian_sparsity(f, np.zeros(4)).todense().astype(int)
    # Conservative: stacked is not const
    expected = np.array(
        [
            [1, 1, 1, 1],
            [1, 1, 1, 1],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


# Const propagation chains
# ------------------------


@pytest.mark.array_ops
def test_stack_unstack_const_chain():
    """Const propagates through stack then unstack for precise gather."""

    def f(x):
        idx1 = jnp.array([0, 1])
        idx2 = jnp.array([2, 3])
        stacked = jnp.stack([idx1, idx2], axis=0)  # [[0,1], [2,3]]
        parts = lax.unstack(stacked, axis=0)  # back to [0,1] and [2,3]
        return jnp.concatenate([x[parts[0]], x[parts[1]]])

    result = jacobian_sparsity(f, np.zeros(5)).todense().astype(int)
    expected = np.eye(4, 5, dtype=int)  # x[0], x[1], x[2], x[3]
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_unstack_stack_const_chain():
    """Const propagates through unstack then stack at different axis."""

    def f(x):
        indices = jnp.array([[0, 1, 2], [3, 4, 5]])  # (2, 3)
        parts = lax.unstack(indices, axis=0)  # [0,1,2] and [3,4,5]
        restacked = jnp.stack(parts, axis=1)  # [[0,3], [1,4], [2,5]]
        return x[restacked.ravel()]  # [0,3,1,4,2,5]

    result = jacobian_sparsity(f, np.zeros(7)).todense().astype(int)
    expected = np.zeros((6, 7), dtype=int)
    for i, idx in enumerate([0, 3, 1, 4, 2, 5]):
        expected[i, idx] = 1
    np.testing.assert_array_equal(result, expected)
