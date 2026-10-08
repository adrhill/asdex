"""Tests for gather propagation.

https://docs.jax.dev/en/latest/_autosummary/jax.lax.gather.html
"""

import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity


def _perm_matrix(n_out: int, n_in: int, mapping: list[int]) -> np.ndarray:
    """Build a permutation Jacobian from an output→input flat index mapping."""
    expected = np.zeros((n_out, n_in), dtype=int)
    for out_idx, in_idx in enumerate(mapping):
        expected[out_idx, in_idx] = 1
    return expected


# Existing tests


@pytest.mark.array_ops
def test_gather_fancy_indexing():
    """Fancy indexing (gather) with static indices tracks precise dependencies.

    Each output element depends on the corresponding indexed input.
    """

    def f(x):
        indices = jnp.array([2, 0, 1])
        return x[indices]

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    # Permutation: out[0] <- in[2], out[1] <- in[0], out[2] <- in[1]
    expected = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_gather_indices_through_select_n():
    """Gather stays precise when indices pass through select_n.

    JAX's negative-index normalization emits select_n between the literal
    indices and the gather.
    Const tracking through select_n keeps the indices statically known.
    """

    def f(x):
        indices = jnp.array([2, 0, 1])
        pred = indices < 0
        wrapped = indices + 3
        final_indices = lax.select(pred, wrapped, indices)
        return x[final_indices]

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    # Permutation: out[0] <- in[2], out[1] <- in[0], out[2] <- in[1]
    expected = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_gather_dynamic_indices_fallback():
    """Gather with bounded dynamic indices enumerates possible index arrays.

    idx = argmax(x[:2]) can only be 0 or 1, so indices are [0,1] or [1,2].
    The union gives each output element at most two input dependencies.
    """

    def f(x):
        # indices depend on x, so they're dynamic
        idx = jnp.argmax(x[:2])  # Dynamic index based on input
        indices = jnp.array([0, 1]) + idx
        return jnp.take(x, indices)

    result = jacobian_sparsity(f, np.zeros(4)).todense().astype(int)
    expected = np.array([[1, 1, 0, 0], [0, 1, 1, 0]], dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_gather_2d_row_select():
    """2D gather selecting rows tracks per-row dependencies.

    Each output row depends only on the corresponding selected input row.
    """

    def f(x):
        mat = x.reshape(3, 2)
        indices = jnp.array([2, 0])  # Select rows 2 and 0
        return mat[indices].flatten()

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    # Output: row 2 (indices 4,5), then row 0 (indices 0,1)
    expected = np.array(
        [
            [0, 0, 0, 0, 1, 0],  # out[0] <- in[4]
            [0, 0, 0, 0, 0, 1],  # out[1] <- in[5]
            [1, 0, 0, 0, 0, 0],  # out[2] <- in[0]
            [0, 1, 0, 0, 0, 0],  # out[3] <- in[1]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


# Precision verification


@pytest.mark.array_ops
def test_gather_embedding_precision():
    """Dim-0 gather on non-square operand: (5, 3)[indices].

    Embedding lookup with non-square shape
    so every dim has a unique size.
    """

    def f(x):
        mat = x.reshape(5, 3)
        indices = jnp.array([3, 0, 4])
        return mat[indices].flatten()

    result = jacobian_sparsity(f, np.zeros(15)).todense().astype(int)
    # rows 3,0,4 from (5,3): flat indices [9..11, 0..2, 12..14]
    expected = _perm_matrix(9, 15, [9, 10, 11, 0, 1, 2, 12, 13, 14])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_gather_dim1_precision():
    """Gather along dim 1 on a (3, 4) matrix.

    ``x[:, indices]`` selects columns.
    """

    def f(x):
        mat = x.reshape(3, 4)
        indices = jnp.array([2, 0])
        return mat[:, indices].flatten()

    result = jacobian_sparsity(f, np.zeros(12)).todense().astype(int)
    # For each row r, select cols 2,0: flat = r*4+col
    expected = _perm_matrix(6, 12, [2, 0, 6, 4, 10, 8])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_gather_middle_dim_precision():
    """Gather along the middle dim of a (2, 3, 4) tensor.

    ``x[:, indices, :]`` selects slices along dim 1.
    """

    def f(x):
        t = x.reshape(2, 3, 4)
        indices = jnp.array([2, 0])
        return t[:, indices, :].flatten()

    result = jacobian_sparsity(f, np.zeros(24)).todense().astype(int)
    # For batch b, sel s in [2,0], col c: input = b*12 + sel*4 + c
    mapping = [b * 12 + s * 4 + c for b in range(2) for s in [2, 0] for c in range(4)]
    expected = _perm_matrix(16, 24, mapping)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_gather_last_dim_precision():
    """Gather along the last dim of a (2, 3, 4) tensor.

    ``x[:, :, indices]`` selects along the trailing axis.
    """

    def f(x):
        t = x.reshape(2, 3, 4)
        indices = jnp.array([3, 1, 0])
        return t[:, :, indices].flatten()

    result = jacobian_sparsity(f, np.zeros(24)).todense().astype(int)
    # For batch b, row r, sel c in [3,1,0]: input = b*12 + r*4 + c
    mapping = [
        b * 12 + r * 4 + c for b in range(2) for r in range(3) for c in [3, 1, 0]
    ]
    expected = _perm_matrix(18, 24, mapping)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_gather_multi_index_precision():
    """Multi-index gather ``x[rows, cols]`` on (3, 4).

    Advanced integer indexing with two index arrays
    collapses both dims simultaneously.
    """

    def f(x):
        mat = x.reshape(3, 4)
        rows = jnp.array([0, 2, 1, 0])
        cols = jnp.array([3, 1, 0, 2])
        return mat[rows, cols]

    result = jacobian_sparsity(f, np.zeros(12)).todense().astype(int)
    # flat = row*4 + col: [3, 9, 4, 2]
    expected = _perm_matrix(4, 12, [3, 9, 4, 2])
    np.testing.assert_array_equal(result, expected)


# Asymmetric shapes (all dims unique)


@pytest.mark.array_ops
def test_gather_dim0_nonsquare():
    """Dim-0 gather on (5, 3) tracks per-row dependencies precisely.

    Non-square shape makes axis-ordering bugs visible.
    """

    def f(x):
        mat = x.reshape(5, 3)
        indices = jnp.array([4, 1])
        return mat[indices].flatten()

    result = jacobian_sparsity(f, np.zeros(15)).todense().astype(int)
    # rows 4,1 from (5,3): flat indices [12,13,14, 3,4,5]
    expected = _perm_matrix(6, 15, [12, 13, 14, 3, 4, 5])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_gather_dim1_nonsquare():
    """Dim-1 gather on (2, 5) selects columns precisely.

    Non-square shape where dim 0 < dim 1.
    """

    def f(x):
        mat = x.reshape(2, 5)
        indices = jnp.array([4, 0, 2])
        return mat[:, indices].flatten()

    result = jacobian_sparsity(f, np.zeros(10)).todense().astype(int)
    # For each row r, select cols 4,0,2: flat = r*5+col
    expected = _perm_matrix(6, 10, [4, 0, 2, 9, 5, 7])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_gather_3d_asymmetric():
    """Gather along dim 1 of (2, 3, 5) — all dims have unique sizes.

    Asymmetric shape exposes transposition bugs in offset_dims mapping.
    """

    def f(x):
        t = x.reshape(2, 3, 5)
        indices = jnp.array([2, 0])
        return t[:, indices, :].flatten()

    result = jacobian_sparsity(f, np.zeros(30)).todense().astype(int)
    # For batch b, sel s in [2,0], col c: input = b*15 + s*5 + c
    mapping = [b * 15 + s * 5 + c for b in range(2) for s in [2, 0] for c in range(5)]
    expected = _perm_matrix(20, 30, mapping)
    np.testing.assert_array_equal(result, expected)


# Conservative audit


@pytest.mark.array_ops
def test_gather_dim0_sparser_than_conservative():
    """Dim-0 gather result is strictly sparser than conservative."""

    def f(x):
        mat = x.reshape(5, 3)
        indices = jnp.array([3, 0])
        return mat[indices].flatten()

    result = jacobian_sparsity(f, np.zeros(15)).todense().astype(int)
    n_out, n_in = result.shape
    assert 0 < result.sum() < n_out * n_in


@pytest.mark.array_ops
def test_gather_dim1_sparser_than_conservative():
    """Dim-1 gather result is strictly sparser than conservative."""

    def f(x):
        mat = x.reshape(2, 5)
        indices = jnp.array([4, 1])
        return mat[:, indices].flatten()

    result = jacobian_sparsity(f, np.zeros(10)).todense().astype(int)
    n_out, n_in = result.shape
    assert 0 < result.sum() < n_out * n_in


@pytest.mark.array_ops
def test_gather_middle_dim_sparser_than_conservative():
    """Middle-dim gather on (2, 3, 5) is strictly sparser than conservative."""

    def f(x):
        t = x.reshape(2, 3, 5)
        indices = jnp.array([1])
        return t[:, indices, :].flatten()

    result = jacobian_sparsity(f, np.zeros(30)).todense().astype(int)
    n_out, n_in = result.shape
    assert 0 < result.sum() < n_out * n_in


@pytest.mark.array_ops
def test_gather_multi_index_sparser_than_conservative():
    """Multi-index gather ``x[rows, cols]`` is strictly sparser than conservative."""

    def f(x):
        mat = x.reshape(3, 4)
        rows = jnp.array([0, 2])
        cols = jnp.array([3, 1])
        return mat[rows, cols]

    result = jacobian_sparsity(f, np.zeros(12)).todense().astype(int)
    n_out, n_in = result.shape
    assert 0 < result.sum() < n_out * n_in


# Const chain / composition


@pytest.mark.array_ops
def test_gather_indices_through_broadcast():
    """Indices surviving broadcast_in_dim remain statically known.

    A scalar index broadcast to a 1-element array
    should still resolve precisely for gather.
    """

    def f(x):
        # broadcast_in_dim is emitted when JAX broadcasts a scalar index.
        idx = jnp.array(2)
        idx_arr = jnp.broadcast_to(idx, (1,))
        return x[idx_arr]

    result = jacobian_sparsity(f, np.zeros(5)).todense().astype(int)
    expected = np.array([[0, 0, 1, 0, 0]], dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_gather_indices_through_reshape():
    """Reshaping a constant index array preserves static tracking.

    idx = [[1,0],[2,3]] reshaped to [1,0,2,3],
    so out[i] = x[idx_flat[i]] is a permutation matrix.
    """

    def f(x):
        idx = jnp.array([[1, 0], [2, 3]])
        idx_flat = idx.reshape(4)
        return x[idx_flat]

    result = jacobian_sparsity(f, np.zeros(5)).todense().astype(int)
    expected = np.array(
        [
            [0, 1, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
            [0, 0, 0, 1, 0],
        ]
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_gather_indices_through_slice():
    """Slicing a constant index array preserves static tracking.

    idx = [3,0,2,1] sliced to [3,0,2],
    so out[0]=x[3], out[1]=x[0], out[2]=x[2].
    """

    def f(x):
        idx = jnp.array([3, 0, 2, 1])
        idx_sub = idx[:3]  # [3, 0, 2]
        return x[idx_sub]

    result = jacobian_sparsity(f, np.zeros(5)).todense().astype(int)
    expected = np.array(
        [
            [0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
        ]
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_gather_indices_through_transpose():
    """Transposing a constant index array preserves static tracking.

    idx = [[1,0],[3,2]] transposed to [[1,3],[0,2]], flattened to [1,3,0,2],
    so out[0]=x[1], out[1]=x[3], out[2]=x[0], out[3]=x[2].
    """

    def f(x):
        idx = jnp.array([[1, 0], [3, 2]])
        idx_t = jnp.transpose(idx)  # [[1, 3], [0, 2]]
        return x[idx_t.flatten()]  # [1, 3, 0, 2]

    result = jacobian_sparsity(f, np.zeros(5)).todense().astype(int)
    expected = np.array(
        [
            [0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [1, 0, 0, 0, 0],
            [0, 0, 1, 0, 0],
        ]
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_gather_indices_through_tile():
    """Tiling a constant index array preserves static tracking.

    idx = [2,0] tiled to [2,0,2,0],
    so out[0],out[2]=x[2] and out[1],out[3]=x[0].
    """

    def f(x):
        idx = jnp.array([2, 0])
        idx_rep = jnp.tile(idx, 2)  # [2, 0, 2, 0]
        return x[idx_rep]

    result = jacobian_sparsity(f, np.zeros(4)).todense().astype(int)
    expected = np.array(
        [
            [0, 0, 1, 0],
            [1, 0, 0, 0],
            [0, 0, 1, 0],
            [1, 0, 0, 0],
        ]
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
@pytest.mark.filterwarnings("ignore:Explicitly requested dtype int64:UserWarning")
def test_gather_indices_through_convert_element_type():
    """Indices surviving convert_element_type remain statically known.

    JAX sometimes emits type conversions on index arrays
    (e.g. int32 -> int64).
    """

    def f(x):
        idx = jnp.array([2, 0, 1], dtype=jnp.int32)
        # Force a type conversion via explicit cast.
        idx64 = idx.astype(jnp.int64)
        return x[idx64]

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_gather_chained_two_gathers():
    """Two gathers chained: ``x[idx1][:, idx2]``.

    First gather selects rows (dim 0),
    second gather selects columns (dim 1) from the result.
    Both should resolve precisely.
    """

    def f(x):
        mat = x.reshape(5, 3)
        idx1 = jnp.array([4, 0, 2])
        intermediate = mat[idx1]  # (3, 3) from rows 4, 0, 2
        idx2 = jnp.array([2, 0])
        return intermediate[:, idx2].flatten()

    result = jacobian_sparsity(f, np.zeros(15)).todense().astype(int)
    # Row 4 cols [2,0] → [14,12], row 0 cols [2,0] → [2,0], row 2 cols [2,0] → [8,6]
    expected = _perm_matrix(6, 15, [14, 12, 2, 0, 8, 6])
    np.testing.assert_array_equal(result, expected)


# Edge cases


@pytest.mark.array_ops
def test_gather_single_element():
    """Gather of a single element produces a 1-output, single-dependency row."""

    def f(x):
        return x[jnp.array([2])]

    result = jacobian_sparsity(f, np.zeros(5)).todense().astype(int)
    expected = np.array([[0, 0, 1, 0, 0]], dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_gather_2d_index_array():
    """2D index array gathers into a 2D output.

    ``x[[[0, 2], [1, 3]]]`` produces shape (2, 2).
    """

    def f(x):
        idx = jnp.array([[0, 3], [1, 2]])
        return x[idx].flatten()

    result = jacobian_sparsity(f, np.zeros(5)).todense().astype(int)
    # out[0] <- in[0], out[1] <- in[3], out[2] <- in[1], out[3] <- in[2]
    expected = np.array(
        [
            [1, 0, 0, 0, 0],
            [0, 0, 0, 1, 0],
            [0, 1, 0, 0, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_gather_repeated_indices():
    """Repeated indices produce duplicate rows in the Jacobian.

    ``x[[1, 1, 1]]`` selects the same element three times.
    All three output rows should depend only on input[1].
    """

    def f(x):
        idx = jnp.array([1, 1, 1])
        return x[idx]

    result = jacobian_sparsity(f, np.zeros(4)).todense().astype(int)
    expected = np.array(
        [
            [0, 1, 0, 0],
            [0, 1, 0, 0],
            [0, 1, 0, 0],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_gather_identity_permutation():
    """Identity permutation ``x[[0, 1, 2, 3]]`` produces an identity Jacobian."""

    def f(x):
        idx = jnp.array([0, 1, 2, 3])
        return x[idx]

    result = jacobian_sparsity(f, np.zeros(4)).todense().astype(int)
    expected = np.eye(4, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_gather_dim1_via_lax_gather():
    """Gather along axis 1 using direct lax.gather.

    Uses lax.gather directly to avoid the jit wrapper
    that ``jnp.take`` introduces.
    """

    def f(x):
        mat = x.reshape(3, 4)
        indices = jnp.array([3, 0])
        return mat[:, indices].flatten()

    result = jacobian_sparsity(f, np.zeros(12)).todense().astype(int)
    # For each row r, select cols 3,0: flat = r*4+col
    expected = _perm_matrix(6, 12, [3, 0, 7, 4, 11, 8])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_gather_3d_last_dim_direct():
    """Gather along last dim of (2, 3, 5) via direct indexing.

    All three dims have unique sizes to catch transposition errors.
    Uses ``x[:, :, indices]`` instead of ``jnp.take``
    to emit a bare gather without a jit wrapper.
    """

    def f(x):
        t = x.reshape(2, 3, 5)
        indices = jnp.array([4, 1, 0])
        return t[:, :, indices].flatten()

    result = jacobian_sparsity(f, np.zeros(30)).todense().astype(int)
    # For batch b, row r, sel c in [4,1,0]: input = b*15 + r*5 + c
    mapping = [
        b * 15 + r * 5 + c for b in range(2) for r in range(3) for c in [4, 1, 0]
    ]
    expected = _perm_matrix(18, 30, mapping)
    np.testing.assert_array_equal(result, expected)


# Multi-dim gather with non-collapsed dims


@pytest.mark.array_ops
def test_gather_multi_dim_with_kept_dim():
    """Multi-dim index into first two dims of a 3D array, keeping the third.

    ``arr[rows, cols]`` on shape (3, 4, 5) selects 2 slices of length 5.
    Each output row depends on exactly one input slice.
    """

    def f(x):
        t = x.reshape(3, 4, 5)
        return t[jnp.array([0, 2]), jnp.array([1, 3])].flatten()

    result = jacobian_sparsity(f, np.zeros(60)).todense().astype(int)
    # out[i, c] = t[rows[i], cols[i], c]
    # Flat input: rows[i]*20 + cols[i]*5 + c
    mapping = [r * 20 + c * 5 + k for r, c in [(0, 1), (2, 3)] for k in range(5)]
    expected = _perm_matrix(10, 60, mapping)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_gather_multi_dim_single_coordinate():
    """Multi-dim gather with a single 1D coordinate vector.

    ``lax.gather`` with 1D start_indices (a single coordinate)
    selects one scalar element from a 2D array.
    """

    def f(x):
        arr = x.reshape(3, 4)
        # Single coordinate (1, 2) → element at flat index 6.
        return lax.gather(
            arr,
            jnp.array([1, 2]),
            dimension_numbers=lax.GatherDimensionNumbers(
                offset_dims=(),
                collapsed_slice_dims=(0, 1),
                start_index_map=(0, 1),
            ),
            slice_sizes=(1, 1),
        ).reshape(1)

    result = jacobian_sparsity(f, np.zeros(12)).todense().astype(int)
    expected = _perm_matrix(1, 12, [6])
    np.testing.assert_array_equal(result, expected)


# Low-level gather patterns (lax.gather with explicit dimension_numbers)


@pytest.mark.array_ops
def test_gather_multi_dim_start_indices_single_collapse():
    """Single-dim gather with multi-dimensional start_indices.

    When start_indices has shape (N, 1) and only one dim is collapsed,
    the batch axis is expanded into multiple axes during reshape.
    If the reshaped ndim doesn't match out_ndim, the handler must bail out
    to avoid a permutation mismatch.

    This is the pattern POWERSUM triggers via hessian_sparsity.
    """

    def f(x):
        # x^2 + x^3 + x^4 summed, then grad → triggers gather
        # with multi-dim start_indices.
        mat = x.reshape(2, 3)
        idx = jnp.array([[0, 2], [1, 0]])
        return mat[:, idx].flatten()

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    # Each output reads exactly one input element.
    # mat[:, idx] has shape (2, 2, 2).
    # mat[r, idx[i,j]] = x[r*3 + idx[i,j]]
    mapping = [r * 3 + idx for r in range(2) for idx in [0, 2, 1, 0]]
    expected = _perm_matrix(8, 6, mapping)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_gather_single_dim_start_map_mismatch():
    """Single collapsed dim with start_index_map pointing elsewhere.

    collapsed_slice_dims=(0,) but start_index_map=(1,):
    indices address dim 1, not the collapsed dim.
    """

    def f(x):
        arr = x.reshape(3, 4)
        return lax.gather(
            arr,
            jnp.array([[1], [3]]),
            dimension_numbers=lax.GatherDimensionNumbers(
                offset_dims=(1,),
                collapsed_slice_dims=(0,),
                start_index_map=(1,),
            ),
            slice_sizes=(1, 1),
        ).flatten()

    result = jacobian_sparsity(f, np.zeros(12)).todense().astype(int)
    # out[0] = arr[0, 1] = x[1], out[1] = arr[0, 3] = x[3].
    expected = _perm_matrix(2, 12, [1, 3])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_gather_single_dim_partial_slice():
    """Single collapsed dim with partial non-collapsed slice.

    slice_sizes[1]=2 but operand has shape[1]=4:
    the non-collapsed dim doesn't span the full operand.
    """

    def f(x):
        arr = x.reshape(3, 4)
        return lax.gather(
            arr,
            jnp.array([[0], [2]]),
            dimension_numbers=lax.GatherDimensionNumbers(
                offset_dims=(1,),
                collapsed_slice_dims=(0,),
                start_index_map=(0,),
            ),
            slice_sizes=(1, 2),
        ).flatten()

    result = jacobian_sparsity(f, np.zeros(12)).todense().astype(int)
    # out[0] = arr[0, 0] = x[0], out[1] = arr[0, 1] = x[1],
    # out[2] = arr[2, 0] = x[8], out[3] = arr[2, 1] = x[9].
    expected = _perm_matrix(4, 12, [0, 1, 8, 9])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_gather_multi_dim_start_map_mismatch():
    """Multi-dim collapse with reversed start_index_map.

    collapsed_slice_dims=(0, 1) but start_index_map=(1, 0).
    Reversed map swaps dims: indices [0,1] address arr[1,0],
    and [2,3] addresses arr[3,2] which clamps to arr[2,2].
    """

    def f(x):
        arr = x.reshape(3, 4, 5)
        return lax.gather(
            arr,
            jnp.array([[0, 1], [2, 3]]),
            dimension_numbers=lax.GatherDimensionNumbers(
                offset_dims=(1,),
                collapsed_slice_dims=(0, 1),
                start_index_map=(1, 0),
            ),
            slice_sizes=(1, 1, 5),
        ).flatten()

    result = jacobian_sparsity(f, np.zeros(60)).todense().astype(int)
    # arr[1, 0, 0..4] = x[20..24], arr[2, 2, 0..4] = x[50..54] (dim0 clamped 3→2).
    expected = _perm_matrix(10, 60, list(range(20, 25)) + list(range(50, 55)))
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_gather_multi_dim_partial_non_collapsed():
    """Multi-dim collapse with partial non-collapsed slice.

    slice_sizes[2]=3 but operand has shape[2]=5.
    """

    def f(x):
        arr = x.reshape(3, 4, 5)
        return lax.gather(
            arr,
            jnp.array([[0, 1], [2, 3]]),
            dimension_numbers=lax.GatherDimensionNumbers(
                offset_dims=(1,),
                collapsed_slice_dims=(0, 1),
                start_index_map=(0, 1),
            ),
            slice_sizes=(1, 1, 3),
        ).flatten()

    result = jacobian_sparsity(f, np.zeros(60)).todense().astype(int)
    # out[0..2] = arr[0, 1, 0..2] = x[5..7],
    # out[3..5] = arr[2, 3, 0..2] = x[55..57].
    expected = _perm_matrix(6, 60, [5, 6, 7, 55, 56, 57])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_gather_batching_dims():
    """Gather with operand_batching_dims tracks per-batch dependencies.

    Batching dims pair operand and start_indices along dim 0,
    so each batch element gathers independently.
    """

    def f(x):
        arr = x.reshape(2, 3)
        return lax.gather(
            arr,
            jnp.array([[1], [0]]),
            dimension_numbers=lax.GatherDimensionNumbers(
                offset_dims=(),
                collapsed_slice_dims=(1,),
                start_index_map=(1,),
                operand_batching_dims=(0,),
                start_indices_batching_dims=(0,),
            ),
            slice_sizes=(1, 1),
        )

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    # out[0] = arr[0, 1] = x[1], out[1] = arr[1, 0] = x[3].
    expected = _perm_matrix(2, 6, [1, 3])
    np.testing.assert_array_equal(result, expected)


# Data-dependent indices


@pytest.mark.array_ops
def test_gather_argsort_conservative():
    """Gather with argsort indices falls back to conservative.

    When indices are computed from inputs (data-dependent),
    the output depends on all inputs.
    This is the MWE from issue #115.
    """

    def f(x):
        idx = jnp.argsort(x)
        return x[idx]

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.ones((3, 3), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_gather_data_dependent_two_inputs():
    """Gather with data-dependent indices includes both operand and index dependencies.

    When x[argsort(y)] is computed,
    each output depends on all elements of both x (the operand) and y (the index source).
    """

    def f(x, y):
        idx = jnp.argsort(y)
        return x[idx]

    result_x = jacobian_sparsity(f, np.zeros(3), np.zeros(3), argnums=0)
    result_y = jacobian_sparsity(f, np.zeros(3), np.zeros(3), argnums=1)
    # Each output depends on all elements of x (because index could point anywhere).
    expected_x = np.ones((3, 3), dtype=int)
    # Each output depends on all elements of y (because argsort depends on all of y).
    expected_y = np.ones((3, 3), dtype=int)
    np.testing.assert_array_equal(result_x.todense().astype(int), expected_x)
    np.testing.assert_array_equal(result_y.todense().astype(int), expected_y)


@pytest.mark.array_ops
def test_gather_argmax_bounded_enumeration():
    """Gather with bounded argmax indices uses bounded enumeration.

    argmax over a small axis produces bounded indices [0, 1],
    so gather uses bounded enumeration.
    argmax has zero derivative (piecewise constant),
    so it doesn't contribute index sets.
    The result is the union of all possible selection patterns.
    """

    def f(x):
        # argmax over first 2 elements → index is 0 or 1, bounded but data-dependent.
        idx = jnp.argmax(x[:2])
        # jnp.take emits gather, not dynamic_slice.
        indices = jnp.array([0, 1]) + idx
        return jnp.take(x, indices)

    result = jacobian_sparsity(f, np.zeros(4)).todense().astype(int)
    # Bounded enumeration:
    #   idx=0: indices=[0,1] → out = [x[0], x[1]] → index sets [{0}, {1}]
    #   idx=1: indices=[1,2] → out = [x[1], x[2]] → index sets [{1}, {2}]
    # Union: out[0] depends on {0,1}, out[1] depends on {1,2}.
    expected = np.array(
        [
            [1, 1, 0, 0],  # out[0] = x[idx] depends on {0, 1}
            [0, 1, 1, 0],  # out[1] = x[idx+1] depends on {1, 2}
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


# Size-0 dimension


@pytest.mark.array_ops
def test_gather_zero_size_indices():
    """Gathering from a zero-sized array produces an empty Jacobian."""

    def f(x):
        return x[:0][jnp.array([], dtype=int)]

    result = jacobian_sparsity(f, np.zeros(3))
    assert result.shape == (0, 3)
    assert result.nnz == 0


@pytest.mark.array_ops
@pytest.mark.fallback
def test_gather_fill_mode_oob():
    """Gather with mode='fill' treats out-of-bounds reads as clamped.

    TODO(gather): under ``mode='fill'`` an out-of-bounds output
    is the constant fill value with no input dependency,
    so the precise second row is empty.
    The handler hardcodes clamp semantics
    and reports a dependency on the clamped position x[2] instead.
    The result stays a safe superset.
    """

    def f(x):
        return x.at[jnp.array([0, 5])].get(mode="fill", fill_value=0.0)

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.array(
        [
            [1, 0, 0],  # out[0] <- x[0]
            [0, 0, 1],  # out[1] <- fill constant, clamped to x[2] by the handler
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)
