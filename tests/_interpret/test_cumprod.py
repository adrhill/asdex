"""Tests for cumprod, cummax, and cummin handlers.

These cumulative operations share the same structural sparsity pattern as cumsum:
lower-triangular (forward) or upper-triangular (reverse) along the scan axis.

The numerical Jacobians differ:
- cumprod: ∂out[i]/∂x[j] = prod(x[k] for k≤i, k≠j)
- cummax/cummin: ∂out[i]/∂x[j] = 1 only at argmax/argmin position

But structurally, each output depends on all preceding inputs along the scan axis.
"""

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity


def _cumulative_jacobian(
    shape: tuple[int, ...], axis: int, reverse: bool = False
) -> np.ndarray:
    """Build the expected structural Jacobian for cumulative operations.

    Lower-triangular along the scan axis (forward)
    or upper-triangular (reverse),
    with independent lanes across other dimensions.
    """
    n = int(np.prod(shape))
    expected = np.zeros((n, n), dtype=int)
    scan_len = shape[axis]

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
                for j in range(k, scan_len):
                    expected[out_pos, pos_flat[j, f]] = 1
            else:
                for j in range(k + 1):
                    expected[out_pos, pos_flat[j, f]] = 1

    return expected


# =============================================================================
# cumprod tests
# =============================================================================


@pytest.mark.array_ops
def test_cumprod_1d_forward():
    """Forward cumprod: lower-triangular structural pattern."""

    def f(x):
        return lax.cumprod(x, axis=0)

    result = jacobian_sparsity(f, np.zeros(5)).todense().astype(int)
    expected = _cumulative_jacobian((5,), axis=0)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_cumprod_1d_reverse():
    """Reverse cumprod: upper-triangular structural pattern."""

    def f(x):
        return lax.cumprod(x, axis=0, reverse=True)

    result = jacobian_sparsity(f, np.zeros(5)).todense().astype(int)
    expected = _cumulative_jacobian((5,), axis=0, reverse=True)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_cumprod_1d_structural_vs_numerical():
    """Structural pattern is at least as dense as numerical Jacobian.

    For cumprod, the structural pattern is lower-triangular,
    but the numerical Jacobian may have zeros where x[j]=0.
    """

    def f(x):
        return lax.cumprod(x, axis=0)

    x = jnp.array([1.0, 2.0, 3.0, 4.0])
    structural = jacobian_sparsity(f, np.zeros(4)).todense().astype(int)
    numerical = (np.abs(jax.jacobian(f)(x)) > 1e-10).astype(int)
    # Structural should cover numerical (be at least as dense)
    assert np.all(structural >= numerical)
    # And structural is lower-triangular
    expected = _cumulative_jacobian((4,), axis=0)
    np.testing.assert_array_equal(structural, expected)


@pytest.mark.array_ops
@pytest.mark.parametrize(
    ("shape", "axis"),
    [
        pytest.param((3, 4), 0, id="axis0"),
        pytest.param((3, 4), 1, id="axis1"),
    ],
)
def test_cumprod_2d(shape, axis):
    """2D cumprod along each axis with non-square shape."""
    n = int(np.prod(shape))

    def f(x):
        return lax.cumprod(x.reshape(shape), axis=axis).flatten()

    result = jacobian_sparsity(f, np.zeros(n)).todense().astype(int)
    expected = _cumulative_jacobian(shape, axis=axis)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
@pytest.mark.parametrize("axis", [0, 1, 2])
def test_cumprod_3d(axis):
    """3D cumprod along each axis."""
    shape = (2, 3, 4)
    n = int(np.prod(shape))

    def f(x):
        return lax.cumprod(x.reshape(shape), axis=axis).flatten()

    result = jacobian_sparsity(f, np.zeros(n)).todense().astype(int)
    expected = _cumulative_jacobian(shape, axis=axis)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_cumprod_size_one():
    """Size-1 scan dimension: identity pattern."""

    def f(x):
        return lax.cumprod(x.reshape(1, 3), axis=0).flatten()

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_cumprod_size_zero():
    """Size-0 scan dimension: empty array."""

    def f(x):
        return lax.cumprod(jnp.zeros((0, 3)), axis=0).flatten()

    result = jacobian_sparsity(f, np.zeros(3))
    assert result.shape == (0, 3)
    assert result.nnz == 0


@pytest.mark.array_ops
def test_jnp_cumprod():
    """jnp.cumprod lowers to the cumprod primitive."""

    def f(x):
        return jnp.cumprod(x)

    result = jacobian_sparsity(f, np.zeros(4)).todense().astype(int)
    expected = _cumulative_jacobian((4,), axis=0)
    np.testing.assert_array_equal(result, expected)


# =============================================================================
# cummax tests
# =============================================================================


@pytest.mark.array_ops
def test_cummax_1d_forward():
    """Forward cummax: lower-triangular structural pattern."""

    def f(x):
        return lax.cummax(x, axis=0)

    result = jacobian_sparsity(f, np.zeros(5)).todense().astype(int)
    expected = _cumulative_jacobian((5,), axis=0)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_cummax_1d_reverse():
    """Reverse cummax: upper-triangular structural pattern."""

    def f(x):
        return lax.cummax(x, axis=0, reverse=True)

    result = jacobian_sparsity(f, np.zeros(5)).todense().astype(int)
    expected = _cumulative_jacobian((5,), axis=0, reverse=True)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_cummax_structural_vs_numerical():
    """Structural pattern is at least as dense as numerical Jacobian.

    For cummax, the numerical Jacobian is sparse (only argmax positions),
    but structurally we must assume any preceding element could be the max.
    """

    def f(x):
        return lax.cummax(x, axis=0)

    x = jnp.array([1.0, 3.0, 2.0, 4.0])
    structural = jacobian_sparsity(f, np.zeros(4)).todense().astype(int)
    numerical = (np.abs(jax.jacobian(f)(x)) > 1e-10).astype(int)
    # Structural should cover numerical
    assert np.all(structural >= numerical)
    # Numerical is sparser (only argmax positions have gradient)
    assert np.sum(numerical) < np.sum(structural)


@pytest.mark.array_ops
@pytest.mark.parametrize(
    ("shape", "axis"),
    [
        pytest.param((3, 4), 0, id="axis0"),
        pytest.param((3, 4), 1, id="axis1"),
    ],
)
def test_cummax_2d(shape, axis):
    """2D cummax along each axis with non-square shape."""
    n = int(np.prod(shape))

    def f(x):
        return lax.cummax(x.reshape(shape), axis=axis).flatten()

    result = jacobian_sparsity(f, np.zeros(n)).todense().astype(int)
    expected = _cumulative_jacobian(shape, axis=axis)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_cummax_size_zero():
    """Size-0 scan dimension: empty array."""

    def f(x):
        return lax.cummax(jnp.zeros((0, 3)), axis=0).flatten()

    result = jacobian_sparsity(f, np.zeros(3))
    assert result.shape == (0, 3)
    assert result.nnz == 0


# =============================================================================
# cummin tests
# =============================================================================


@pytest.mark.array_ops
def test_cummin_1d_forward():
    """Forward cummin: lower-triangular structural pattern."""

    def f(x):
        return lax.cummin(x, axis=0)

    result = jacobian_sparsity(f, np.zeros(5)).todense().astype(int)
    expected = _cumulative_jacobian((5,), axis=0)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_cummin_1d_reverse():
    """Reverse cummin: upper-triangular structural pattern."""

    def f(x):
        return lax.cummin(x, axis=0, reverse=True)

    result = jacobian_sparsity(f, np.zeros(5)).todense().astype(int)
    expected = _cumulative_jacobian((5,), axis=0, reverse=True)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_cummin_structural_vs_numerical():
    """Structural pattern is at least as dense as numerical Jacobian."""

    def f(x):
        return lax.cummin(x, axis=0)

    x = jnp.array([4.0, 2.0, 3.0, 1.0])
    structural = jacobian_sparsity(f, np.zeros(4)).todense().astype(int)
    numerical = (np.abs(jax.jacobian(f)(x)) > 1e-10).astype(int)
    # Structural should cover numerical
    assert np.all(structural >= numerical)
    # Numerical is sparser (only argmin positions have gradient)
    assert np.sum(numerical) < np.sum(structural)


@pytest.mark.array_ops
@pytest.mark.parametrize(
    ("shape", "axis"),
    [
        pytest.param((3, 4), 0, id="axis0"),
        pytest.param((3, 4), 1, id="axis1"),
    ],
)
def test_cummin_2d(shape, axis):
    """2D cummin along each axis with non-square shape."""
    n = int(np.prod(shape))

    def f(x):
        return lax.cummin(x.reshape(shape), axis=axis).flatten()

    result = jacobian_sparsity(f, np.zeros(n)).todense().astype(int)
    expected = _cumulative_jacobian(shape, axis=axis)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_cummin_size_zero():
    """Size-0 scan dimension: empty array."""

    def f(x):
        return lax.cummin(jnp.zeros((0, 3)), axis=0).flatten()

    result = jacobian_sparsity(f, np.zeros(3))
    assert result.shape == (0, 3)
    assert result.nnz == 0


# =============================================================================
# Conservative audit
# =============================================================================


@pytest.mark.array_ops
def test_cumprod_sparser_than_conservative():
    """Cumprod pattern is strictly sparser than conservative (all-ones)."""
    shape = (3, 4)
    n = int(np.prod(shape))

    def f(x):
        return lax.cumprod(x.reshape(shape), axis=0).flatten()

    result = jacobian_sparsity(f, np.zeros(n))
    assert result.nnz < n * n
    # 4 lanes * (1+2+3) = 24 nonzeros
    expected_nnz = 4 * (1 + 2 + 3)
    assert result.nnz == expected_nnz


@pytest.mark.array_ops
def test_cummax_sparser_than_conservative():
    """Cummax pattern is strictly sparser than conservative."""
    shape = (3, 4)
    n = int(np.prod(shape))

    def f(x):
        return lax.cummax(x.reshape(shape), axis=0).flatten()

    result = jacobian_sparsity(f, np.zeros(n))
    assert result.nnz < n * n
    expected_nnz = 4 * (1 + 2 + 3)
    assert result.nnz == expected_nnz


@pytest.mark.array_ops
def test_cummin_sparser_than_conservative():
    """Cummin pattern is strictly sparser than conservative."""
    shape = (3, 4)
    n = int(np.prod(shape))

    def f(x):
        return lax.cummin(x.reshape(shape), axis=0).flatten()

    result = jacobian_sparsity(f, np.zeros(n))
    assert result.nnz < n * n
    expected_nnz = 4 * (1 + 2 + 3)
    assert result.nnz == expected_nnz
