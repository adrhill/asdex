"""Tests for tile propagation.

https://docs.jax.dev/en/latest/_autosummary/jax.lax.tile.html
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity


@pytest.mark.array_ops
def test_tile_1d():
    """Tiling a 1D array repeats element dependencies."""

    def f(x):
        return jnp.tile(x, 2)

    result = jacobian_sparsity(f, np.zeros(2)).todense().astype(int)
    expected = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_tile_identity():
    """Tiling with reps=1 produces the identity Jacobian."""

    def f(x):
        return jnp.tile(x, 1)

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_tile_multidim():
    """Tiling a 2D array along both axes tracks modular dependencies."""

    def f(x):
        mat = x.reshape(2, 3)
        return jnp.tile(mat, (2, 2)).flatten()

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    # mat = [[0,1,2],[3,4,5]], tiled to (4,6)
    # Each output element maps to input[i%2, j%3]
    expected = np.zeros((24, 6), dtype=int)
    for out_idx in range(24):
        row, col = divmod(out_idx, 6)
        in_idx = (row % 2) * 3 + (col % 3)
        expected[out_idx, in_idx] = 1
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_tile_constant():
    """Tiling a constant array produces zero sparsity."""

    def f(_x):
        const = jnp.array([1.0, 2.0])
        return jnp.tile(const, 3)

    result = jacobian_sparsity(f, np.zeros(2)).todense().astype(int)
    expected = np.zeros((6, 2), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_tile_composition_with_sum():
    """Tiling followed by reduction unions the tiled dependencies."""

    def f(x):
        tiled = jnp.tile(x, 2)  # [x0, x1, x0, x1]
        return jnp.sum(tiled.reshape(2, 2), axis=0)  # [x0+x0, x1+x1]

    result = jacobian_sparsity(f, np.zeros(2)).todense().astype(int)
    expected = np.eye(2, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_tile_large_reps():
    """Tiling with larger repetition count."""

    def f(x):
        return jnp.tile(x, 4)

    result = jacobian_sparsity(f, np.zeros(2)).todense().astype(int)
    expected = np.array(
        [[1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1]], dtype=int
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_tile_3d():
    """Tiling a 3D array tracks modular dependencies across all axes."""

    def f(x):
        arr = x.reshape(1, 2, 3)
        return jnp.tile(arr, (2, 1, 1)).flatten()

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    # Tiling (1,2,3) by (2,1,1) -> (2,2,3), each element maps to input[i%1, j%2, k%3]
    # So output is just input repeated twice.
    expected = np.vstack([np.eye(6, dtype=int), np.eye(6, dtype=int)])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_tile_size_1_dim():
    """Tiling a size-1 dimension broadcasts that dimension."""

    def f(x):
        col = x.reshape(3, 1)
        return jnp.tile(col, (1, 4)).flatten()

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    # (3,1) tiled (1,4) -> (3,4): each row repeats the same input element
    expected = np.zeros((12, 3), dtype=int)
    for i in range(12):
        expected[i, i // 4] = 1
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_tile_non_contiguous_deps():
    """Tile preserves non-trivial dependency sets from prior operations."""

    def f(x):
        # Sum pairs to create shared state_indices, then tile
        summed = x[:2] + x[2:]  # [x0+x2, x1+x3]
        return jnp.tile(summed, 2)  # [x0+x2, x1+x3, x0+x2, x1+x3]

    result = jacobian_sparsity(f, np.zeros(4)).todense().astype(int)
    expected = np.array(
        [[1, 0, 1, 0], [0, 1, 0, 1], [1, 0, 1, 0], [0, 1, 0, 1]], dtype=int
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_tile_double():
    """Double-tiling composes correctly."""

    def f(x):
        tiled1 = jnp.tile(x, 2)  # [x0, x1, x0, x1]
        mat = tiled1.reshape(2, 2)
        return jnp.tile(mat, (1, 2)).flatten()  # [[x0,x1,x0,x1],[x0,x1,x0,x1]]

    result = jacobian_sparsity(f, np.zeros(2)).todense().astype(int)
    expected = np.array(
        [[1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1], [1, 0], [0, 1]], dtype=int
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_tile_matches_jax_jacobian():
    """Sparsity pattern matches the support of the dense JAX Jacobian."""

    def f(x):
        mat = x.reshape(2, 3)
        return jnp.tile(mat, (3, 2)).flatten()

    x = jnp.ones(6)
    dense_jac = jax.jacobian(f)(x)
    dense_pattern = (np.abs(np.array(dense_jac)) > 0).astype(int)

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    np.testing.assert_array_equal(result, dense_pattern)


# Size-0 dimension


@pytest.mark.array_ops
def test_tile_zero_size():
    """Tiling a zero-sized array produces an empty Jacobian."""

    def f(x):
        return jnp.tile(x[:0], 2)

    result = jacobian_sparsity(f, np.zeros(3))
    assert result.shape == (0, 3)
    assert result.nnz == 0
