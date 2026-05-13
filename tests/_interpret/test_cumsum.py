"""Tests for the prop_cumsum handler.

Cumulative sum produces a lower-triangular (forward) or upper-triangular (reverse)
dependency pattern along the scan axis,
with independent lanes across other dimensions.
"""

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity


def _cumsum_jacobian(
    shape: tuple[int, ...], axis: int, reverse: bool = False
) -> np.ndarray:
    """Build the expected Jacobian for a cumsum operation.

    Lower-triangular along the scan axis (forward)
    or upper-triangular (reverse),
    with independent lanes across other dimensions.
    """
    n = int(np.prod(shape))
    expected = np.zeros((n, n), dtype=int)
    scan_len = shape[axis]

    # Build position map and organize by scan axis
    pos = np.arange(n).reshape(shape)
    pos = np.moveaxis(pos, axis, 0)
    n_lanes = pos[0].size if scan_len > 0 else 0
    pos_flat = (
        pos.reshape(scan_len, n_lanes) if scan_len > 0 else np.empty((0, 0), dtype=int)
    )

    for f in range(n_lanes):
        for k in range(scan_len):
            out_pos = pos_flat[k, f]
            if reverse:
                # out[k] depends on in[k], in[k+1], ..., in[scan_len-1]
                for j in range(k, scan_len):
                    expected[out_pos, pos_flat[j, f]] = 1
            else:
                # out[k] depends on in[0], in[1], ..., in[k]
                for j in range(k + 1):
                    expected[out_pos, pos_flat[j, f]] = 1

    return expected


# 1D tests


@pytest.mark.array_ops
def test_cumsum_1d_forward():
    """Forward cumsum: lower-triangular pattern."""

    def f(x):
        return lax.cumsum(x, axis=0)

    result = jacobian_sparsity(f, np.zeros(5)).todense().astype(int)
    expected = _cumsum_jacobian((5,), axis=0)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_cumsum_1d_reverse():
    """Reverse cumsum: upper-triangular pattern."""

    def f(x):
        return lax.cumsum(x, axis=0, reverse=True)

    result = jacobian_sparsity(f, np.zeros(5)).todense().astype(int)
    expected = _cumsum_jacobian((5,), axis=0, reverse=True)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_cumsum_1d_jacobian_values():
    """Verify detected pattern matches numerical Jacobian."""

    def f(x):
        return lax.cumsum(x, axis=0)

    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    detected = jacobian_sparsity(f, np.zeros(4)).todense().astype(int)
    numerical = (np.abs(jax.jacobian(f)(x)) > 1e-10).astype(int)
    np.testing.assert_array_equal(detected, numerical)


# 2D tests


@pytest.mark.array_ops
@pytest.mark.parametrize(
    ("shape", "axis"),
    [
        pytest.param((3, 4), 0, id="axis0"),
        pytest.param((3, 4), 1, id="axis1"),
    ],
)
def test_cumsum_2d(shape, axis):
    """2D cumsum along each axis with non-square shape."""
    n = int(np.prod(shape))

    def f(x):
        return lax.cumsum(x.reshape(shape), axis=axis).flatten()

    result = jacobian_sparsity(f, np.zeros(n)).todense().astype(int)
    expected = _cumsum_jacobian(shape, axis=axis)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
@pytest.mark.parametrize(
    ("shape", "axis"),
    [
        pytest.param((3, 4), 0, id="axis0"),
        pytest.param((3, 4), 1, id="axis1"),
    ],
)
def test_cumsum_2d_reverse(shape, axis):
    """2D reverse cumsum along each axis."""
    n = int(np.prod(shape))

    def f(x):
        return lax.cumsum(x.reshape(shape), axis=axis, reverse=True).flatten()

    result = jacobian_sparsity(f, np.zeros(n)).todense().astype(int)
    expected = _cumsum_jacobian(shape, axis=axis, reverse=True)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_cumsum_2d_jacobian_values():
    """Verify detected pattern matches numerical Jacobian for 2D."""
    shape = (3, 4)
    n = int(np.prod(shape))

    def f(x):
        return lax.cumsum(x.reshape(shape), axis=1).flatten()

    x = jnp.arange(1.0, n + 1.0)
    detected = jacobian_sparsity(f, np.zeros(n)).todense().astype(int)
    numerical = (np.abs(jax.jacobian(f)(x)) > 1e-10).astype(int)
    np.testing.assert_array_equal(detected, numerical)


# 3D tests


@pytest.mark.array_ops
@pytest.mark.parametrize(
    "axis",
    [
        pytest.param(0, id="axis0"),
        pytest.param(1, id="axis1"),
        pytest.param(2, id="axis2"),
    ],
)
def test_cumsum_3d(axis):
    """3D cumsum along each axis."""
    shape = (2, 3, 4)
    n = int(np.prod(shape))

    def f(x):
        return lax.cumsum(x.reshape(shape), axis=axis).flatten()

    result = jacobian_sparsity(f, np.zeros(n)).todense().astype(int)
    expected = _cumsum_jacobian(shape, axis=axis)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_cumsum_3d_jacobian_values():
    """Verify detected pattern matches numerical Jacobian for 3D."""
    shape = (2, 3, 4)
    n = int(np.prod(shape))

    def f(x):
        return lax.cumsum(x.reshape(shape), axis=1).flatten()

    x = jnp.arange(1.0, n + 1.0)
    detected = jacobian_sparsity(f, np.zeros(n)).todense().astype(int)
    numerical = (np.abs(jax.jacobian(f)(x)) > 1e-10).astype(int)
    np.testing.assert_array_equal(detected, numerical)


# Edge cases


@pytest.mark.array_ops
def test_cumsum_size_one():
    """Size-1 dimension: trivial scan, identity pattern."""

    def f(x):
        return lax.cumsum(x.reshape(1, 3), axis=0).flatten()

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_cumsum_size_zero():
    """Size-0 dimension: empty array."""

    def f(x):
        return lax.cumsum(jnp.zeros((0, 3)), axis=0).flatten()

    result = jacobian_sparsity(f, np.zeros(3))
    assert result.shape == (0, 3)
    assert result.nnz == 0


# Compositions


@pytest.mark.array_ops
def test_cumsum_chained():
    """Cumsum chained with cumsum: forward then reverse."""

    def f(x):
        y = lax.cumsum(x, axis=0)
        return lax.cumsum(y, axis=0, reverse=True)

    result = jacobian_sparsity(f, np.zeros(4)).todense().astype(int)
    # Forward lower-triangular then reverse upper-triangular.
    # Verify against numerical Jacobian.
    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    numerical = (np.abs(jax.jacobian(f)(x)) > 1e-10).astype(int)
    np.testing.assert_array_equal(result, numerical)


@pytest.mark.array_ops
def test_cumsum_after_broadcast():
    """Non-contiguous input patterns from a prior broadcast."""

    def f(x):
        # Broadcast (3,) -> (2, 3), then cumsum along axis 0
        y = jnp.broadcast_to(x, (2, 3))
        return lax.cumsum(y, axis=0).flatten()

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    # Row 0: [a, b, c] (broadcast of x)
    # Row 1: [a+a, b+b, c+c] (cumsum along axis 0)
    # First 3 outputs are identity, last 3 are also identity (2*x[i] depends on x[i])
    expected = np.vstack([np.eye(3, dtype=int), np.eye(3, dtype=int)])
    np.testing.assert_array_equal(result, expected)


# Conservative audit


@pytest.mark.array_ops
def test_cumsum_sparser_than_conservative():
    """Cumsum pattern is strictly sparser than conservative (all-ones)."""
    shape = (3, 4)
    n = int(np.prod(shape))

    def f(x):
        return lax.cumsum(x.reshape(shape), axis=0).flatten()

    result = jacobian_sparsity(f, np.zeros(n))
    # Conservative would be n*n = 144 nonzeros
    assert result.nnz < n * n
    # Exact count: 4 lanes * (1+2+3) = 24 nonzeros
    expected_nnz = 4 * (1 + 2 + 3)
    assert result.nnz == expected_nnz


# High-level API


@pytest.mark.array_ops
def test_jnp_cumsum():
    """jnp.cumsum lowers to the cumsum primitive."""

    def f(x):
        return jnp.cumsum(x)

    result = jacobian_sparsity(f, np.zeros(4)).todense().astype(int)
    expected = _cumsum_jacobian((4,), axis=0)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_jnp_cumsum_2d_axis():
    """jnp.cumsum with explicit axis on 2D array."""
    shape = (3, 4)
    n = int(np.prod(shape))

    def f(x):
        return jnp.cumsum(x.reshape(shape), axis=1).flatten()

    result = jacobian_sparsity(f, np.zeros(n)).todense().astype(int)
    expected = _cumsum_jacobian(shape, axis=1)
    np.testing.assert_array_equal(result, expected)
