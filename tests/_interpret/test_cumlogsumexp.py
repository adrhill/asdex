"""Tests for the cumlogsumexp handler.

Cumulative log-sum-exp has the same structural sparsity pattern as cumsum:
each output depends on the prefix (or suffix when ``reverse=True``)
along the scan axis, with independent lanes across other dimensions.
"""

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity
from tests._utils import assert_jacobian_sparsity_exact


def _cumulative_jacobian(
    shape: tuple[int, ...], axis: int, reverse: bool = False
) -> np.ndarray:
    """Build the expected Jacobian for a cumulative operation.

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


# 1D tests


@pytest.mark.array_ops
@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse"])
def test_cumlogsumexp_1d(reverse):
    """1D cumlogsumexp is lower-triangular forward and upper-triangular reverse."""

    def f(x):
        return lax.cumlogsumexp(x, axis=0, reverse=reverse)

    x = jnp.array([0.5, 1.0, 2.0, 4.0])
    result = jacobian_sparsity(f, np.ones(4)).todense().astype(int)
    expected = _cumulative_jacobian((4,), axis=0, reverse=reverse)
    np.testing.assert_array_equal(result, expected)
    assert_jacobian_sparsity_exact(f, x)


@pytest.mark.array_ops
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_cumlogsumexp_dtypes(dtype):
    """Cumlogsumexp works for both supported precisions."""

    def f(x):
        return lax.cumlogsumexp(x, axis=0)

    with jax.enable_x64(dtype == np.float64):
        x = jnp.asarray([0.5, 1.0, 2.0], dtype=dtype)
        result = jacobian_sparsity(f, np.ones(3, dtype=dtype)).todense().astype(int)
        np.testing.assert_array_equal(result, _cumulative_jacobian((3,), axis=0))
        assert_jacobian_sparsity_exact(f, x)


# 2D tests


@pytest.mark.array_ops
@pytest.mark.parametrize(
    ("shape", "axis"),
    [
        pytest.param((3, 4), 0, id="axis0"),
        pytest.param((3, 4), 1, id="axis1"),
    ],
)
@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse"])
def test_cumlogsumexp_2d(shape, axis, reverse):
    """2D cumlogsumexp along each axis."""
    n = int(np.prod(shape))

    def f(x):
        return lax.cumlogsumexp(x.reshape(shape), axis=axis, reverse=reverse).flatten()

    x = jnp.arange(1.0, n + 1.0) / 4
    result = jacobian_sparsity(f, np.ones(n)).todense().astype(int)
    expected = _cumulative_jacobian(shape, axis=axis, reverse=reverse)
    np.testing.assert_array_equal(result, expected)
    assert_jacobian_sparsity_exact(f, x)


# 3D tests


@pytest.mark.array_ops
@pytest.mark.parametrize("axis", [0, 1, 2], ids=["axis0", "axis1", "axis2"])
@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse"])
def test_cumlogsumexp_3d(axis, reverse):
    """3D cumlogsumexp along each axis."""
    shape = (2, 3, 4)
    n = int(np.prod(shape))

    def f(x):
        return lax.cumlogsumexp(x.reshape(shape), axis=axis, reverse=reverse).flatten()

    result = jacobian_sparsity(f, np.ones(n)).todense().astype(int)
    expected = _cumulative_jacobian(shape, axis=axis, reverse=reverse)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_cumlogsumexp_3d_jacobian_values():
    """Verify the 3D detected pattern matches the numerical Jacobian."""
    shape = (2, 3, 4)
    n = int(np.prod(shape))

    def f(x):
        return lax.cumlogsumexp(x.reshape(shape), axis=1).flatten()

    x = jnp.arange(1.0, n + 1.0) / 5
    assert_jacobian_sparsity_exact(f, x)


# Edge cases


@pytest.mark.array_ops
@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse"])
def test_cumlogsumexp_size_one(reverse):
    """Size-1 scan axis is an identity pattern."""

    def f(x):
        return lax.cumlogsumexp(x.reshape(1, 3), axis=0, reverse=reverse).flatten()

    result = jacobian_sparsity(f, np.ones(3)).todense().astype(int)
    np.testing.assert_array_equal(result, np.eye(3, dtype=int))


@pytest.mark.array_ops
@pytest.mark.parametrize("reverse", [False, True], ids=["forward", "reverse"])
def test_cumlogsumexp_zero_size_scan_axis(reverse):
    """An empty scan axis returns an empty pattern."""

    def f(x):
        return lax.cumlogsumexp(x.reshape(0, 3), axis=0, reverse=reverse).flatten()

    result = jacobian_sparsity(f, np.zeros(0))
    assert result.shape == (0, 0)
    assert result.nnz == 0


@pytest.mark.array_ops
def test_cumlogsumexp_zero_size_other_axis():
    """A zero-length non-scan axis also returns an empty pattern."""

    def f(x):
        return lax.cumlogsumexp(x.reshape(3, 0), axis=0).flatten()

    result = jacobian_sparsity(f, np.zeros(0))
    assert result.shape == (0, 0)
    assert result.nnz == 0


# Conservative audit


@pytest.mark.array_ops
def test_cumlogsumexp_sparser_than_conservative():
    """Cumlogsumexp produces a strictly sparser result than conservative."""
    shape = (3, 4)

    def f(x):
        return lax.cumlogsumexp(x.reshape(shape), axis=0).flatten()

    result = jacobian_sparsity(f, np.ones(12)).todense().astype(int)
    n_out, n_in = result.shape
    assert 0 < result.sum() < n_out * n_in
