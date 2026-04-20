"""Tests for scatter propagation.

https://docs.jax.dev/en/latest/_autosummary/jax.lax.scatter.html
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import hessian_sparsity, jacobian_sparsity

# Existing basic tests


@pytest.mark.array_ops
def test_scatter_at_set():
    """In-place update with .at[].set() tracks precise dependencies.

    Only the updated position depends on the update value.
    """

    def f(x):
        arr = jnp.zeros(3)
        return arr.at[1].set(x[0])

    result = jacobian_sparsity(f, input_shape=(2,)).todense().astype(int)
    # Only index 1 depends on x[0], indices 0 and 2 are constant (zeros)
    expected = np.array([[0, 0], [1, 0], [0, 0]], dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_scatter_add():
    """Scatter-add unions dependencies from operand and updates.

    Positions receiving updates depend on both the original value and the update.
    """

    def f(x):
        arr = jnp.array([1.0, 2.0, 3.0])
        return arr.at[1].add(x[0])  # arr[1] += x[0]

    result = jacobian_sparsity(f, input_shape=(2,)).todense().astype(int)
    # Index 1 depends on x[0] (no dependency on arr since arr is constant)
    expected = np.array(
        [
            [0, 0],  # out[0] <- constant
            [1, 0],  # out[1] <- x[0]
            [0, 0],  # out[2] <- constant
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_scatter_multiple():
    """Scatter at multiple indices tracks each update separately."""

    def f(x):
        arr = jnp.zeros(4)
        return arr.at[jnp.array([0, 2])].set(x[:2])

    result = jacobian_sparsity(f, input_shape=(3,)).todense().astype(int)
    expected = np.array(
        [
            [1, 0, 0],  # out[0] <- x[0]
            [0, 0, 0],  # out[1] <- zeros (constant)
            [0, 1, 0],  # out[2] <- x[1]
            [0, 0, 0],  # out[3] <- zeros (constant)
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_scatter_dynamic_indices():
    """Scatter with bounded dynamic indices enumerates possible targets.

    idx = argmax(x[3:]) can only be 0 or 1 (two elements), so idx is never 2.
    out[i] = x[i] when idx != i, or x[3] when idx == i.
    """

    def f(x):
        arr = x[:3]
        idx = jnp.argmax(x[3:]).astype(int)
        return arr.at[idx].set(x[3])

    result = jacobian_sparsity(f, input_shape=(5,)).todense().astype(int)
    expected = np.array(
        [
            [1, 0, 0, 1, 0],
            [0, 1, 0, 1, 0],
            [0, 0, 1, 0, 0],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
@pytest.mark.parametrize("method", ["mul", "min", "max"])
def test_scatter_combine(method):
    """Scatter combine variants union dependencies from operand and updates.

    Targeted positions depend on both the original value and the update.
    Non-targeted positions depend only on the operand.
    """

    def f(x):
        arr = x[:3]
        return getattr(arr.at[1], method)(x[3])

    result = jacobian_sparsity(f, input_shape=(4,)).todense().astype(int)
    expected = np.array(
        [
            [1, 0, 0, 0],  # out[0] <- x[0]
            [0, 1, 0, 1],  # out[1] <- combine(x[1], x[3])
            [0, 0, 1, 0],  # out[2] <- x[2]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_segment_sum():
    """segment_sum groups elements by segment ID.

    Each output depends on all inputs in the corresponding segment.
    """

    def f(x):
        segment_ids = jnp.array([0, 0, 1, 1, 1])
        return jax.ops.segment_sum(x, segment_ids, num_segments=2)

    result = jacobian_sparsity(f, input_shape=(5,)).todense().astype(int)
    # Segment 0: inputs 0,1 -> output 0
    # Segment 1: inputs 2,3,4 -> output 1
    expected = np.array(
        [
            [1, 1, 0, 0, 0],  # out[0] <- x[0] + x[1]
            [0, 0, 1, 1, 1],  # out[1] <- x[2] + x[3] + x[4]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_scatter_2d_batched_dim0():
    """2D scatter-add along dim 0 tracks per-column dependencies precisely.

    The backward of ``features[indices]`` on 2D arrays produces scatter-add
    with ``update_window_dims=(1,)``, ``inserted_window_dims=(0,)``.
    Each update row targets an operand row,
    with trailing dimensions passed through element-wise.
    """
    indices = jnp.array([2, 0, 1])

    def f(x):
        mat = x.reshape(3, 4)
        gathered = mat[indices]  # [3, 4]: rows reordered
        return gathered.reshape(-1)

    result = jacobian_sparsity(f, input_shape=(12,)).todense().astype(int)
    # gathered[0] = mat[2], gathered[1] = mat[0], gathered[2] = mat[1]
    # Each output element depends on exactly one input element.
    expected = np.zeros((12, 12), dtype=int)
    for i in range(3):
        for j in range(4):
            out_flat = i * 4 + j
            in_flat = indices[i] * 4 + j
            expected[out_flat, in_flat] = 1
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_scatter_set_middle_dim():
    """Scatter along a middle dimension tracks precise dependencies.

    ``arr.at[:, idx, :].set(value)`` writes a 2D slice at one position
    along dim 1. Non-target positions keep their original dependencies.
    """

    def f(x):
        arr = x.reshape(2, 3, 4)
        # Zero out the last neighbor slot (dim 1 = 2).
        arr = arr.at[:, 2, :].set(0.0)
        return arr.reshape(-1)

    result = jacobian_sparsity(f, input_shape=(24,)).todense().astype(int)
    expected = np.zeros((24, 24), dtype=int)
    for i in range(24):
        # Flat index i corresponds to (a, n, h) = (i//12, (i//4)%3, i%4)
        n = (i // 4) % 3
        if n != 2:
            # Non-target positions keep identity dependency.
            expected[i, i] = 1
        # Target positions (n == 2) are set to constant 0, no dependencies.
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_scatter_add_middle_dim():
    """Scatter-add along a middle dimension unions operand and update state_indices.

    ``arr.at[:, idx, :].add(value)`` adds a 2D slice at one position
    along dim 1. Target positions depend on both operand and updates.
    """
    values = jnp.ones((2, 4))

    def f(x):
        arr = x.reshape(2, 3, 4)
        arr = arr.at[:, 1, :].add(values)
        return arr.reshape(-1)

    result = jacobian_sparsity(f, input_shape=(24,)).todense().astype(int)
    # All positions keep identity since updates are constant.
    expected = np.eye(24, dtype=int)
    np.testing.assert_array_equal(result, expected)


# Precision verification


@pytest.mark.array_ops
def test_scatter_at_set_precision():
    """1D scatter: ``arr.at[[0,2]].set(x[4:6])`` replaces positions 0 and 2.

    Positions 0 and 2 depend on x[4] and x[5] respectively.
    Positions 1 and 3 keep their original dependencies x[1] and x[3].
    """

    def f(x):
        arr = x[:4]
        return arr.at[jnp.array([0, 2])].set(x[4:6])

    result = jacobian_sparsity(f, input_shape=(6,)).todense().astype(int)
    expected = np.array(
        [
            [0, 0, 0, 0, 1, 0],  # out[0] <- x[4]
            [0, 1, 0, 0, 0, 0],  # out[1] <- x[1]
            [0, 0, 0, 0, 0, 1],  # out[2] <- x[5]
            [0, 0, 0, 1, 0, 0],  # out[3] <- x[3]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_scatter_2d_batched_precision():
    """2D batched scatter (Pattern 1): replace rows with constants.

    Reshapes into (3, 4) and replaces rows 2 and 0 with ones.
    Only row 1 (flat indices 4-7) keeps its original dependencies.
    """
    indices = jnp.array([2, 0])

    def f(x):
        mat = x.reshape(3, 4)
        updates = jnp.ones((2, 4))
        return mat.at[indices].set(updates).reshape(-1)

    result = jacobian_sparsity(f, input_shape=(12,)).todense().astype(int)
    # Rows 0 and 2 replaced with constant ones, row 1 keeps identity.
    expected = np.zeros((12, 12), dtype=int)
    for i in range(4, 8):
        expected[i, i] = 1
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_scatter_middle_dim_precision():
    """Middle-dim scatter (Pattern 2): sets ``arr.at[:, 1, :] = 0``.

    Non-target positions keep identity;
    target positions (dim 1 = 1) become constant.
    """

    def f(x):
        arr = x.reshape(2, 3, 4)
        arr = arr.at[:, 1, :].set(0.0)
        return arr.reshape(-1)

    result = jacobian_sparsity(f, input_shape=(24,)).todense().astype(int)
    expected = np.zeros((24, 24), dtype=int)
    for i in range(24):
        if (i // 4) % 3 != 1:
            expected[i, i] = 1
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_scatter_multi_index_precision():
    """Multi-index scatter (Pattern 3): zeroes specific coordinates.

    ``mat.at[rows, cols].set(zeros)`` zeroes positions (0,1), (1,3), (2,0).
    Those positions lose input state_indices; all others keep identity.
    """

    def f(x):
        mat = x.reshape(3, 4)
        rows = jnp.array([0, 1, 2])
        cols = jnp.array([1, 3, 0])
        return mat.at[rows, cols].set(jnp.zeros(3)).reshape(-1)

    result = jacobian_sparsity(f, input_shape=(12,)).todense().astype(int)
    # Zeroed positions: (0,1)=1, (1,3)=7, (2,0)=8
    expected = np.eye(12, dtype=int)
    for pos in [1, 7, 8]:
        expected[pos, pos] = 0
    np.testing.assert_array_equal(result, expected)


# Asymmetric (non-square) shapes


@pytest.mark.array_ops
def test_scatter_batched_nonsquare():
    """Batched scatter on a non-square (3, 5) operand.

    Replaces row 1 with constant updates,
    so row 1 loses input dependencies.
    """

    def f(x):
        mat = x.reshape(3, 5)
        updates = jnp.zeros((1, 5))
        return mat.at[jnp.array([1])].set(updates).reshape(-1)

    result = jacobian_sparsity(f, input_shape=(15,)).todense().astype(int)
    # Row 1 (flat 5-9) zeroed, rows 0 and 2 keep identity.
    expected = np.eye(15, dtype=int)
    for i in range(5, 10):
        expected[i, i] = 0
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_scatter_middle_dim_nonsquare():
    """Middle-dim scatter on a non-square (2, 3, 5) operand.

    Sets ``arr.at[:, 0, :] = 0`` to replace all entries along dim 1 = 0.
    """

    def f(x):
        arr = x.reshape(2, 3, 5)
        arr = arr.at[:, 0, :].set(0.0)
        return arr.reshape(-1)

    result = jacobian_sparsity(f, input_shape=(30,)).todense().astype(int)
    # dim 1 = 0 positions zeroed: (i // 5) % 3 == 0
    expected = np.zeros((30, 30), dtype=int)
    for i in range(30):
        if (i // 5) % 3 != 0:
            expected[i, i] = 1
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_scatter_multi_index_nonsquare():
    """Multi-index scatter on a non-square (3, 5) operand.

    Writes three scalars at specific coordinates.
    """

    def f(x):
        mat = x.reshape(3, 5)
        rows = jnp.array([0, 2, 1])
        cols = jnp.array([4, 0, 2])
        return mat.at[rows, cols].set(jnp.zeros(3)).reshape(-1)

    result = jacobian_sparsity(f, input_shape=(15,)).todense().astype(int)
    # Zeroed positions: (0,4)=4, (2,0)=10, (1,2)=7
    expected = np.eye(15, dtype=int)
    for pos in [4, 7, 10]:
        expected[pos, pos] = 0
    np.testing.assert_array_equal(result, expected)


# Conservative audit


@pytest.mark.array_ops
def test_scatter_batched_sparser_than_conservative():
    """Batched scatter (Pattern 1) produces a sparser result than conservative."""

    def f(x):
        mat = x.reshape(3, 4)
        return mat.at[jnp.array([0, 2])].set(jnp.zeros((2, 4))).reshape(-1)

    result = jacobian_sparsity(f, input_shape=(12,)).todense().astype(int)
    n_out, n_in = result.shape
    assert 0 < result.sum() < n_out * n_in


@pytest.mark.array_ops
def test_scatter_middle_dim_sparser_than_conservative():
    """Middle-dim scatter (Pattern 2) produces a sparser result than conservative."""

    def f(x):
        arr = x.reshape(2, 3, 4)
        arr = arr.at[:, 2, :].set(0.0)
        return arr.reshape(-1)

    result = jacobian_sparsity(f, input_shape=(24,)).todense().astype(int)
    n_out, n_in = result.shape
    assert 0 < result.sum() < n_out * n_in


@pytest.mark.array_ops
def test_scatter_multi_index_sparser_than_conservative():
    """Multi-index scatter (Pattern 3) produces a sparser result than conservative."""

    def f(x):
        mat = x.reshape(3, 5)
        rows = jnp.array([0, 2])
        cols = jnp.array([1, 3])
        return mat.at[rows, cols].set(jnp.zeros(2)).reshape(-1)

    result = jacobian_sparsity(f, input_shape=(15,)).todense().astype(int)
    n_out, n_in = result.shape
    assert 0 < result.sum() < n_out * n_in


# Composition


@pytest.mark.array_ops
def test_scatter_then_gather():
    """Scatter followed by gather: only the gathered positions matter.

    Sets position 1 to zero, then gathers positions [0, 2].
    The final output should depend only on x[0] and x[2].
    """

    def f(x):
        arr = x.at[1].set(0.0)
        return arr[jnp.array([0, 2])]

    result = jacobian_sparsity(f, input_shape=(3,)).todense().astype(int)
    expected = np.array(
        [
            [1, 0, 0],  # out[0] <- x[0]
            [0, 0, 1],  # out[1] <- x[2]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_scatter_with_const_chain_indices():
    """Scatter with indices derived from broadcast and reshape.

    Indices are built through a const-propagation chain
    (literal -> broadcast -> reshape) and should still be resolved statically.
    """

    def f(x):
        arr = x[:4]
        # Build indices through broadcast + reshape
        base = jnp.array([1])
        broadcasted = jnp.broadcast_to(base, (2,))
        idx = broadcasted + jnp.array([0, 1])  # [1, 2]
        return arr.at[idx].set(jnp.zeros(2))

    result = jacobian_sparsity(f, input_shape=(4,)).todense().astype(int)
    # Positions 1 and 2 replaced with constant zeros.
    expected = np.array(
        [
            [1, 0, 0, 0],  # out[0] <- x[0]
            [0, 0, 0, 0],  # out[1] <- constant 0
            [0, 0, 0, 0],  # out[2] <- constant 0
            [0, 0, 0, 1],  # out[3] <- x[3]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_scatter_after_reshape():
    """Scatter into a reshaped array preserves per-element tracking."""

    def f(x):
        mat = x.reshape(2, 3)
        # Replace row 0 with zeros
        mat = mat.at[0].set(jnp.zeros(3))
        return mat.reshape(-1)

    result = jacobian_sparsity(f, input_shape=(6,)).todense().astype(int)
    # Row 0 (positions 0-2) replaced with constant 0, row 1 (3-5) kept.
    expected = np.zeros((6, 6), dtype=int)
    for i in range(3, 6):
        expected[i, i] = 1
    np.testing.assert_array_equal(result, expected)


# Edge cases


@pytest.mark.array_ops
def test_scatter_duplicate_indices_set():
    """Duplicate indices with set: last write wins.

    When two updates target the same position,
    only the last update's dependencies survive (replace semantics).
    """

    def f(x):
        arr = jnp.zeros(3)
        # Both x[0] and x[1] target position 1; x[1] wins.
        return arr.at[jnp.array([1, 1])].set(x[:2])

    result = jacobian_sparsity(f, input_shape=(3,)).todense().astype(int)
    expected = np.array(
        [
            [0, 0, 0],  # out[0] <- constant 0
            [0, 1, 0],  # out[1] <- x[1] (last write wins)
            [0, 0, 0],  # out[2] <- constant 0
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_scatter_duplicate_indices_add():
    """Duplicate indices with add: both updates contribute.

    When two updates target the same position,
    the output depends on both (union semantics).
    """

    def f(x):
        arr = jnp.zeros(3)
        # Both x[0] and x[1] are added to position 1.
        return arr.at[jnp.array([1, 1])].add(x[:2])

    result = jacobian_sparsity(f, input_shape=(3,)).todense().astype(int)
    expected = np.array(
        [
            [0, 0, 0],  # out[0] <- constant 0
            [1, 1, 0],  # out[1] <- x[0] + x[1]
            [0, 0, 0],  # out[2] <- constant 0
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_scatter_oob_indices():
    """OOB indices are silently ignored by JAX, so the array is unchanged."""

    def f(x):
        arr = x[:3]
        # Index 10 is OOB for size 3; JAX silently ignores it.
        return arr.at[jnp.array([10])].set(jnp.array([0.0]))

    result = jacobian_sparsity(f, input_shape=(3,)).todense().astype(int)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_scatter_replace_all():
    """Scatter that replaces every position: output depends only on updates."""

    def f(x):
        arr = x[:3]
        return arr.at[jnp.array([0, 1, 2])].set(x[3:6])

    result = jacobian_sparsity(f, input_shape=(6,)).todense().astype(int)
    # Every position is replaced by the corresponding update.
    expected = np.array(
        [
            [0, 0, 0, 1, 0, 0],  # out[0] <- x[3]
            [0, 0, 0, 0, 1, 0],  # out[1] <- x[4]
            [0, 0, 0, 0, 0, 1],  # out[2] <- x[5]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_scatter_multi_index_duplicate_set():
    """Multi-index scatter with duplicate coordinates: last write wins.

    Two updates target ``(0, 1)``; the second update's dep survives.
    """

    def f(x):
        mat = x.reshape(2, 3)
        rows = jnp.array([0, 0])
        cols = jnp.array([1, 1])
        vals = x[:2]  # x[0] and x[1] both target (0, 1)
        return mat.at[rows, cols].set(vals).reshape(-1)

    result = jacobian_sparsity(f, input_shape=(6,)).todense().astype(int)
    # Position (0,1) = flat 1 gets x[1] (last write wins).
    # All other positions keep their original identity state_indices.
    expected = np.eye(6, dtype=int)
    expected[1, :] = 0
    expected[1, 1] = 1  # out[1] <- x[1]
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_scatter_multi_index_duplicate_add():
    """Multi-index scatter-add with duplicate coordinates: union of both.

    Two updates target ``(0, 1)``; the output unions both state_indices.
    """

    def f(x):
        mat = jnp.zeros((2, 3))
        rows = jnp.array([0, 0])
        cols = jnp.array([1, 1])
        vals = x[:2]
        return mat.at[rows, cols].add(vals).reshape(-1)

    result = jacobian_sparsity(f, input_shape=(3,)).todense().astype(int)
    # Position (0,1) = flat 1 gets x[0] + x[1].
    expected = np.array(
        [
            [0, 0, 0],  # out[0] <- constant 0
            [1, 1, 0],  # out[1] <- x[0] + x[1]
            [0, 0, 0],  # out[2] <- constant 0
            [0, 0, 0],  # out[3] <- constant 0
            [0, 0, 0],  # out[4] <- constant 0
            [0, 0, 0],  # out[5] <- constant 0
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_scatter_multi_index_oob():
    """Multi-index scatter with OOB coordinates; JAX silently ignores them."""

    def f(x):
        mat = x.reshape(2, 3)
        # (5, 10) is OOB for (2, 3); JAX silently ignores it.
        rows = jnp.array([5])
        cols = jnp.array([10])
        return mat.at[rows, cols].set(jnp.zeros(1)).reshape(-1)

    result = jacobian_sparsity(f, input_shape=(6,)).todense().astype(int)
    expected = np.eye(6, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_scatter_2d():
    """2D partial-row scatter ``mat.at[0, :2].set(updates)`` tracks precise deps.

    Only the two targeted positions depend on the corresponding update elements.
    All other positions are constant zeros.
    """

    def f(x):
        mat = jnp.zeros((2, 3))
        updates = x[:2].reshape(1, 2)
        return mat.at[0, :2].set(updates.flatten()).flatten()

    result = jacobian_sparsity(f, input_shape=(3,)).todense().astype(int)
    expected = np.array(
        [
            [1, 0, 0],  # out[0] = mat[0,0] <- x[0]
            [0, 1, 0],  # out[1] = mat[0,1] <- x[1]
            [0, 0, 0],  # out[2] = mat[0,2] <- constant 0
            [0, 0, 0],  # out[3] = mat[1,0] <- constant 0
            [0, 0, 0],  # out[4] = mat[1,1] <- constant 0
            [0, 0, 0],  # out[5] = mat[1,2] <- constant 0
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_scatter_window():
    """Scatter-add with ``update_window_dims=(1,), inserted_window_dims=()`` is precise.

    This is the config JAX emits for the VJP of 2D gather (``features[indices]``).
    Each update row is added to the operand row at the position given by the index.
    """

    def f(x):
        mat = x.reshape(3, 4)
        indices = jnp.array([[2], [0]])
        updates = jnp.ones((2, 4))
        return jax.lax.scatter_add(
            mat,
            indices,
            updates,
            jax.lax.ScatterDimensionNumbers(
                update_window_dims=(1,),
                inserted_window_dims=(0,),
                scatter_dims_to_operand_dims=(0,),
            ),
        ).reshape(-1)

    result = jacobian_sparsity(f, input_shape=(12,)).todense().astype(int)
    # Updates are constant ones, so scatter-add positions keep identity.
    expected = np.eye(12, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.hessian
def test_scatter_hessian_slicing():
    """Hessian of ``sum((x[1:] - x[:-1])**2)`` is tridiagonal.

    The gradient uses pad (VJP of slice), not scatter,
    but this validates that the full pipeline handles the slicing pattern.
    """

    def f(x):
        return jnp.sum((x[1:] - x[:-1]) ** 2)

    n = 6
    sp = hessian_sparsity(f, input_shape=(n,))
    dense = sp.todense().astype(int)

    # Tridiagonal: each diagonal and the two sub/super-diagonals.
    expected = np.zeros((n, n), dtype=int)
    for i in range(n):
        expected[i, i] = 1
    for i in range(n - 1):
        expected[i, i + 1] = 1
        expected[i + 1, i] = 1
    np.testing.assert_array_equal(dense, expected)


@pytest.mark.array_ops
def test_scatter_batching_dims():
    """Scatter with operand_batching_dims tracks per-batch dependencies.

    Each batch element scatters independently into its corresponding operand row.
    """

    def f(x):
        mat = x.reshape(2, 3)
        return jax.lax.scatter_add(
            mat,
            jnp.array([[1], [0]]),
            jnp.ones((2,)),
            jax.lax.ScatterDimensionNumbers(
                update_window_dims=(),
                inserted_window_dims=(1,),
                scatter_dims_to_operand_dims=(1,),
                operand_batching_dims=(0,),
                scatter_indices_batching_dims=(0,),
            ),
        ).reshape(-1)

    result = jacobian_sparsity(f, input_shape=(6,)).todense().astype(int)
    # Updates are constant ones, so all positions keep identity.
    expected = np.eye(6, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_scatter_dynamic_too_many_combinations():
    """Scatter with bounded but too-many-combinations indices falls back to conservative.

    When ``argmax`` over a large axis produces bounds exceeding
    the enumeration limit, the handler falls back to conservative.
    """

    def f(x):
        arr = x[:100]
        # argmax over 100 elements: bounds (0, 99), 100 > 64 max combinations.
        idx = jnp.argmax(arr).astype(int)
        return arr.at[idx].set(x[100])

    result = jacobian_sparsity(f, input_shape=(101,)).todense().astype(int)
    n_out, n_in = result.shape
    # Conservative: every output depends on every input.
    assert result.sum() == n_out * n_in


# Size-0 dimension


@pytest.mark.array_ops
def test_scatter_zero_size_update():
    """Scatter with a zero-length update leaves the operand unchanged.

    The output depends only on the original array, not the empty update.
    """

    def f(x):
        return x.at[:0].set(jnp.zeros(0))

    result = jacobian_sparsity(f, input_shape=(3,)).todense().astype(int)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)
