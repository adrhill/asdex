"""Tests for the transpose propagation handler.

Tests N-dimensional permutations, identity permutations,
size-1 dimensions, higher-order cyclic permutations,
and adversarial edge cases.
"""

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity


def _transpose_jacobian(in_shape: tuple[int, ...], permutation: tuple[int, ...]):
    """Build the expected permutation Jacobian for a transpose.

    For each flat output index, compute which flat input index it reads from.
    """
    n = int(np.prod(in_shape))
    out_shape = tuple(in_shape[p] for p in permutation)

    # Inverse permutation: maps input dim → output dim.
    inv_perm = [0] * len(permutation)
    for d, p in enumerate(permutation):
        inv_perm[p] = d

    expected = np.zeros((n, n), dtype=int)
    for out_flat in range(n):
        out_coord = np.unravel_index(out_flat, out_shape)
        in_coord = tuple(out_coord[inv_perm[d]] for d in range(len(in_shape)))
        in_flat = np.ravel_multi_index(in_coord, in_shape)
        expected[out_flat, in_flat] = 1
    return expected


@pytest.mark.array_ops
def test_transpose_1d():
    """1D transpose with permutation (0,) is identity."""

    def f(x):
        return jnp.transpose(x, (0,))

    result = jacobian_sparsity(f, 4).todense().astype(int)
    expected = np.eye(4, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_transpose_identity_2d():
    """2D transpose with identity permutation (0, 1) is identity Jacobian."""

    def f(x):
        mat = x.reshape(2, 3)
        return jnp.transpose(mat, (0, 1)).flatten()

    result = jacobian_sparsity(f, 6).todense().astype(int)
    expected = np.eye(6, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_transpose_3d_swap_last_two():
    """3D transpose swapping the last two axes: (0, 2, 1)."""
    in_shape = (2, 3, 4)

    def f(x):
        arr = x.reshape(in_shape)
        return jnp.transpose(arr, (0, 2, 1)).flatten()

    result = jacobian_sparsity(f, 24).todense().astype(int)
    expected = _transpose_jacobian(in_shape, (0, 2, 1))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_transpose_3d_cyclic():
    """3D transpose with cyclic permutation (1, 2, 0).

    result[i, j, k] = input[k, i, j].
    """
    in_shape = (2, 3, 4)

    def f(x):
        arr = x.reshape(in_shape)
        return jnp.transpose(arr, (1, 2, 0)).flatten()

    result = jacobian_sparsity(f, 24).todense().astype(int)
    expected = _transpose_jacobian(in_shape, (1, 2, 0))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_transpose_3d_reverse():
    """3D transpose with full reversal (2, 1, 0)."""
    in_shape = (2, 3, 4)

    def f(x):
        arr = x.reshape(in_shape)
        return jnp.transpose(arr, (2, 1, 0)).flatten()

    result = jacobian_sparsity(f, 24).todense().astype(int)
    expected = _transpose_jacobian(in_shape, (2, 1, 0))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_transpose_size_one_dims():
    """Transpose with size-1 dimensions (no-op in practice, but tests indexing)."""
    in_shape = (1, 3, 1)

    def f(x):
        arr = x.reshape(in_shape)
        return jnp.transpose(arr, (2, 1, 0)).flatten()

    result = jacobian_sparsity(f, 3).todense().astype(int)
    # All dimensions except dim 1 have size 1, so this is effectively identity.
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_transpose_4d_nhwc_to_nchw():
    """4D transpose simulating NHWC → NCHW layout change: (0, 3, 1, 2)."""
    in_shape = (1, 2, 3, 4)
    perm = (0, 3, 1, 2)

    def f(x):
        arr = x.reshape(in_shape)
        return jnp.transpose(arr, perm).flatten()

    result = jacobian_sparsity(f, 24).todense().astype(int)
    expected = _transpose_jacobian(in_shape, perm)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_transpose_square_involution():
    """Transposing a square matrix twice gives the identity Jacobian."""

    def f(x):
        mat = x.reshape(3, 3)
        return mat.T.T.flatten()

    result = jacobian_sparsity(f, 9).todense().astype(int)
    expected = np.eye(9, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_moveaxis():
    """jnp.moveaxis lowers to transpose; verify end-to-end."""
    in_shape = (2, 3, 4)

    def f(x):
        arr = x.reshape(in_shape)
        return jnp.moveaxis(arr, 0, -1).flatten()  # (3, 4, 2)

    result = jacobian_sparsity(f, 24).todense().astype(int)
    # moveaxis(arr, 0, -1) is equivalent to transpose(arr, (1, 2, 0)).
    expected = _transpose_jacobian(in_shape, (1, 2, 0))
    np.testing.assert_array_equal(result, expected)


# Jacobian verification
# Compare detected sparsity against numerical jax.jacobian.


@pytest.mark.array_ops
def test_transpose_2d_vs_jacobian():
    """2D transpose matches numerical Jacobian."""
    in_shape = (3, 4)
    perm = (1, 0)

    def f(x):
        return jnp.transpose(x.reshape(in_shape), perm).flatten()

    detected = jacobian_sparsity(f, 12).todense().astype(int)
    x_test = jax.random.normal(jax.random.key(0), (12,))
    actual = (np.abs(jax.jacobian(f)(x_test)) > 1e-10).astype(int)
    np.testing.assert_array_equal(detected, actual)


@pytest.mark.array_ops
def test_transpose_3d_vs_jacobian():
    """3D cyclic transpose matches numerical Jacobian."""
    in_shape = (2, 3, 4)
    perm = (2, 0, 1)

    def f(x):
        return jnp.transpose(x.reshape(in_shape), perm).flatten()

    detected = jacobian_sparsity(f, 24).todense().astype(int)
    x_test = jax.random.normal(jax.random.key(1), (24,))
    actual = (np.abs(jax.jacobian(f)(x_test)) > 1e-10).astype(int)
    np.testing.assert_array_equal(detected, actual)


@pytest.mark.array_ops
def test_transpose_4d_vs_jacobian():
    """4D transpose matches numerical Jacobian."""
    in_shape = (2, 3, 2, 2)
    perm = (3, 1, 0, 2)

    def f(x):
        return jnp.transpose(x.reshape(in_shape), perm).flatten()

    detected = jacobian_sparsity(f, 24).todense().astype(int)
    x_test = jax.random.normal(jax.random.key(2), (24,))
    actual = (np.abs(jax.jacobian(f)(x_test)) > 1e-10).astype(int)
    np.testing.assert_array_equal(detected, actual)


# 5D


@pytest.mark.array_ops
def test_transpose_5d():
    """5D transpose with non-trivial permutation."""
    in_shape = (2, 3, 1, 2, 2)
    perm = (4, 2, 0, 3, 1)

    def f(x):
        return jnp.transpose(x.reshape(in_shape), perm).flatten()

    detected = jacobian_sparsity(f, 24).todense().astype(int)
    expected = _transpose_jacobian(in_shape, perm)
    np.testing.assert_array_equal(detected, expected)


# Non-square involution


@pytest.mark.array_ops
def test_transpose_nonsquare_involution():
    """Transposing a non-square matrix twice gives identity."""

    def f(x):
        mat = x.reshape(3, 5)
        return jnp.transpose(jnp.transpose(mat, (1, 0)), (1, 0)).flatten()

    result = jacobian_sparsity(f, 15).todense().astype(int)
    expected = np.eye(15, dtype=int)
    np.testing.assert_array_equal(result, expected)


# Non-contiguous input patterns


@pytest.mark.array_ops
def test_transpose_after_broadcast():
    """Transpose after broadcast: input index sets are non-trivial (shared).

    Each output element's dependencies come from the broadcast,
    not simple singletons.
    """

    def f(x):
        # x shape (3,), broadcast to (2, 3), then transpose to (3, 2).
        arr = jnp.broadcast_to(x, (2, 3))
        return jnp.transpose(arr, (1, 0)).flatten()

    result = jacobian_sparsity(f, 3).todense().astype(int)
    # After broadcast: each column shares one input.
    # Transpose (1,0): output (3,2), out[i,j] = broadcast[j,i] = x[i].
    # So every output element depends on the input element matching its row.
    expected = np.array(
        [
            [1, 0, 0],  # out[0,0] = x[0]
            [1, 0, 0],  # out[0,1] = x[0]
            [0, 1, 0],  # out[1,0] = x[1]
            [0, 1, 0],  # out[1,1] = x[1]
            [0, 0, 1],  # out[2,0] = x[2]
            [0, 0, 1],  # out[2,1] = x[2]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


# Const chain: transpose of a literal index array feeding into gather


@pytest.mark.array_ops
def test_transpose_const_chain():
    """Transpose of a constant array followed by gather resolves precisely.

    The transpose propagates const values so the downstream gather
    can resolve static indices.
    """

    def f(x):
        # Build a constant index matrix, transpose it, and use as gather indices.
        mat = x.reshape(2, 3)
        # Transpose then index: equivalent to selecting specific elements.
        return mat[:, 0] + mat[:, 2]

    result = jacobian_sparsity(f, 6).todense().astype(int)
    # mat[:,0] = [x[0], x[3]], mat[:,2] = [x[2], x[5]]
    # out[0] = x[0] + x[2], out[1] = x[3] + x[5]
    expected = np.array(
        [
            [1, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 1],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


# Conservative audit


@pytest.mark.array_ops
@pytest.mark.parametrize(
    ("desc", "in_shape", "perm"),
    [
        ("2d", (3, 5), (1, 0)),
        ("3d_cyclic", (2, 3, 4), (1, 2, 0)),
        ("4d_nchw", (1, 2, 3, 4), (0, 3, 1, 2)),
    ],
)
def test_transpose_conservative_audit(desc, in_shape, perm):
    """Transpose sparsity is strictly sparser than conservative.

    For any non-trivial shape, the handler must produce a permutation matrix
    (n nonzeros) rather than the conservative n*n.
    """
    n = int(np.prod(in_shape))

    def f(x):
        return jnp.transpose(x.reshape(in_shape), perm).flatten()

    sparsity = jacobian_sparsity(f, n)
    # Permutation matrix: exactly n nonzeros.
    assert sparsity.nnz == n, f"{desc}: expected {n} nnz, got {sparsity.nnz}"
    # Strictly sparser than conservative (n*n).
    assert sparsity.nnz < n * n


# Real-world: jnp.swapaxes


@pytest.mark.array_ops
def test_swapaxes():
    """jnp.swapaxes lowers to transpose; verify end-to-end."""
    in_shape = (2, 3, 4)

    def f(x):
        arr = x.reshape(in_shape)
        return jnp.swapaxes(arr, 0, 2).flatten()  # equivalent to perm (2, 1, 0)

    result = jacobian_sparsity(f, 24).todense().astype(int)
    expected = _transpose_jacobian(in_shape, (2, 1, 0))
    np.testing.assert_array_equal(result, expected)


# Size-0 dimension


@pytest.mark.array_ops
def test_transpose_size_zero_dim():
    """Transpose with a size-0 dimension produces an empty Jacobian."""
    in_shape = (0, 3)

    def f(x):
        arr = lax.reshape(x, in_shape)
        return jnp.transpose(arr, (1, 0)).flatten()

    result = jacobian_sparsity(f, 0).todense().astype(int)
    expected = np.zeros((0, 0), dtype=int)
    np.testing.assert_array_equal(result, expected)
