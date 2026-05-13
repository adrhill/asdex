"""Tests for split propagation.

https://docs.jax.dev/en/latest/_autosummary/jax.lax.split.html
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity


@pytest.mark.array_ops
def test_split_identity():
    """Splitting and re-concatenating recovers the identity Jacobian."""

    def f(x):
        parts = jnp.split(x, 2)
        return jnp.concatenate(parts)

    result = jacobian_sparsity(f, np.zeros(4)).todense().astype(int)
    expected = np.eye(4, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_split_swap_halves():
    """Splitting and swapping halves produces a permutation matrix."""

    def f(x):
        parts = jnp.split(x, 2)
        return jnp.concatenate([parts[1], parts[0]])

    result = jacobian_sparsity(f, np.zeros(4)).todense().astype(int)
    expected = np.array(
        [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=int
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_split_uneven_sizes():
    """Splitting into uneven chunks tracks per-element dependencies."""

    def f(x):
        a, b = jnp.split(x, [1])
        return jnp.concatenate([b, a])

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    # x = [a, b, c] -> a=[a], b=[b,c] -> [b, c, a]
    expected = np.array([[0, 1, 0], [0, 0, 1], [1, 0, 0]], dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_split_multidim():
    """Splitting a 2D array along axis 1 tracks per-element dependencies."""

    def f(x):
        mat = x.reshape(2, 4)
        left, right = jnp.split(mat, 2, axis=1)
        return jnp.concatenate([right, left], axis=1).flatten()

    result = jacobian_sparsity(f, np.zeros(8)).todense().astype(int)
    # mat = [[0,1,2,3],[4,5,6,7]]
    # left = [[0,1],[4,5]], right = [[2,3],[6,7]]
    # concat = [[2,3,0,1],[6,7,4,5]]
    expected = np.zeros((8, 8), dtype=int)
    for out_idx, in_idx in enumerate([2, 3, 0, 1, 6, 7, 4, 5]):
        expected[out_idx, in_idx] = 1
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_split_constant():
    """Splitting a constant array produces zero sparsity."""

    def f(_x):
        const = jnp.array([1.0, 2.0, 3.0, 4.0])
        a, b = jnp.split(const, 2)
        return jnp.concatenate([b, a])

    result = jacobian_sparsity(f, np.zeros(2)).todense().astype(int)
    expected = np.zeros((4, 2), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_split_select_one():
    """Selecting a single chunk from split preserves element dependencies."""

    def f(x):
        parts = jnp.split(x, 3)
        return parts[1]  # middle third

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    expected = np.zeros((2, 6), dtype=int)
    expected[0, 2] = 1
    expected[1, 3] = 1
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_split_3d_axis0():
    """Splitting a 3D array along axis 0 tracks per-element dependencies."""

    def f(x):
        arr = x.reshape(4, 2, 3)
        top, bottom = jnp.split(arr, 2, axis=0)
        return jnp.concatenate([bottom, top], axis=0).flatten()

    result = jacobian_sparsity(f, np.zeros(24)).todense().astype(int)
    # top = arr[:2] (12 elements), bottom = arr[2:] (12 elements)
    # concat swaps: [bottom, top] -> indices 12..23, 0..11
    expected = np.zeros((24, 24), dtype=int)
    for i in range(12):
        expected[i, i + 12] = 1
        expected[i + 12, i] = 1
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_split_many_chunks():
    """Splitting into many single-element chunks preserves identity."""

    def f(x):
        parts = jnp.split(x, 4)
        return jnp.concatenate(list(reversed(parts)))

    result = jacobian_sparsity(f, np.zeros(4)).todense().astype(int)
    # Reverse: out[0]=in[3], out[1]=in[2], out[2]=in[1], out[3]=in[0]
    expected = np.array(
        [[0, 0, 0, 1], [0, 0, 1, 0], [0, 1, 0, 0], [1, 0, 0, 0]], dtype=int
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_split_non_contiguous_deps():
    """Split preserves non-trivial dependency sets from prior broadcast."""

    def f(x):
        # Broadcast creates shared dependencies, then split separates them.
        broadcasted = jnp.tile(x, 2)  # [x0, x1, x0, x1]
        a, b = jnp.split(broadcasted, 2)
        return a + b  # [x0+x0, x1+x1] = [2*x0, 2*x1]

    result = jacobian_sparsity(f, np.zeros(2)).todense().astype(int)
    expected = np.eye(2, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_split_double_split():
    """Double-splitting composes correctly."""

    def f(x):
        a, b = jnp.split(x, 2)
        a1, a2 = jnp.split(a, 2)
        return jnp.concatenate([a2, a1, b])

    result = jacobian_sparsity(f, np.zeros(8)).todense().astype(int)
    # x = [0..7], a = [0..3], b = [4..7]
    # a1 = [0,1], a2 = [2,3]
    # out = [2, 3, 0, 1, 4, 5, 6, 7]
    expected = np.zeros((8, 8), dtype=int)
    for out_idx, in_idx in enumerate([2, 3, 0, 1, 4, 5, 6, 7]):
        expected[out_idx, in_idx] = 1
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_split_matches_jax_jacobian():
    """Sparsity pattern matches the support of the dense JAX Jacobian."""

    def f(x):
        mat = x.reshape(3, 4)
        a, b, c = jnp.split(mat, [1, 3], axis=1)
        return jnp.concatenate([c, a, b], axis=1).flatten()

    x = jnp.ones(12)
    dense_jac = jax.jacobian(f)(x)
    dense_pattern = (np.abs(np.array(dense_jac)) > 0).astype(int)

    result = jacobian_sparsity(f, np.zeros(12)).todense().astype(int)
    np.testing.assert_array_equal(result, dense_pattern)


# Size-0 dimension


@pytest.mark.array_ops
def test_split_zero_size():
    """Splitting a zero-sized array produces an empty Jacobian."""

    def f(x):
        return jnp.split(x[:0], 1)[0]

    result = jacobian_sparsity(f, np.zeros(3))
    assert result.shape == (0, 3)
    assert result.nnz == 0
