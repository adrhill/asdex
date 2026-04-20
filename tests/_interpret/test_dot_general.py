"""Tests for the dot_general propagation handler.

Tests matrix multiplications, batched dots, dot products,
outer products, and higher-dimensional contractions.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity


def _dot_general_jacobian(f, n):
    """Compute the expected Jacobian via jax.jacobian (nonzero pattern)."""
    x = jnp.ones(n)
    J = jax.jacobian(f)(x)
    return (np.array(J).reshape(-1, n) != 0).astype(int)


# Vector dot product
@pytest.mark.array_ops
def test_dot_1d():
    """Dot product of two copies of the same vector.

    out = x @ x = Σ_i x[i]^2, so out depends on all inputs.
    """

    def f(x):
        return jnp.dot(x, x).reshape(1)

    result = jacobian_sparsity(f, input_shape=(3,)).todense().astype(int)
    expected = np.ones((1, 3), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_dot_1d_separate():
    """Dot product x @ y where x and y are separate halves of input.

    out depends on all inputs.
    """

    def f(xy):
        x = xy[:3]
        y = xy[3:]
        return jnp.dot(x, y).reshape(1)

    result = jacobian_sparsity(f, input_shape=(6,)).todense().astype(int)
    expected = np.ones((1, 6), dtype=int)
    np.testing.assert_array_equal(result, expected)


# Matrix-vector multiply
@pytest.mark.array_ops
def test_matvec():
    """Matrix-vector multiply: out[i] = Σ_k A[i,k] * v[k].

    A and v come from the same input, so out[i] depends on row i of A and all of v.
    """

    def f(x):
        # x[:6] is A (2x3), x[6:] is v (3,)
        A = x[:6].reshape(2, 3)
        v = x[6:]
        return A @ v

    result = jacobian_sparsity(f, input_shape=(9,)).todense().astype(int)
    # out[0] depends on A row 0 ({0,1,2}) and v ({6,7,8})
    # out[1] depends on A row 1 ({3,4,5}) and v ({6,7,8})
    expected = np.array(
        [
            [1, 1, 1, 0, 0, 0, 1, 1, 1],
            [0, 0, 0, 1, 1, 1, 1, 1, 1],
        ]
    )
    np.testing.assert_array_equal(result, expected)


# Matrix-matrix multiply
@pytest.mark.array_ops
def test_matmul_2x3_3x2():
    """Matrix multiply A(2,3) @ B(3,2) → C(2,2).

    A and B are separate halves of the input.
    """

    def f(x):
        A = x[:6].reshape(2, 3)
        B = x[6:].reshape(3, 2)
        return (A @ B).flatten()

    result = jacobian_sparsity(f, input_shape=(12,)).todense().astype(int)
    # out[i,j] depends on A row i and B col j.
    # A row 0: {0,1,2}, A row 1: {3,4,5}
    # B col 0: {6,8,10}, B col 1: {7,9,11}
    expected = np.array(
        [
            [1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1, 0],  # out[0,0]
            [1, 1, 1, 0, 0, 0, 0, 1, 0, 1, 0, 1],  # out[0,1]
            [0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0],  # out[1,0]
            [0, 0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],  # out[1,1]
        ]
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_matmul_self():
    """X @ X.T where X is a 2x3 matrix.

    Both operands share the same input, so state_indices are unioned.
    """

    def f(x):
        mat = x.reshape(2, 3)
        return (mat @ mat.T).flatten()

    result = jacobian_sparsity(f, input_shape=(6,)).todense().astype(int)
    # mat = [[x0,x1,x2],[x3,x4,x5]]
    # out[0,0] = row0·row0 → {0,1,2}
    # out[0,1] = row0·row1 → {0,1,2,3,4,5}
    # out[1,0] = row1·row0 → {0,1,2,3,4,5}
    # out[1,1] = row1·row1 → {3,4,5}
    expected = np.array(
        [
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
        ]
    )
    np.testing.assert_array_equal(result, expected)


# Batched matrix multiply
@pytest.mark.array_ops
def test_batched_matmul():
    """Batched matmul: batch dim is preserved, each batch independent.

    A(3,2,4) @ B(3,4,5) → C(3,2,5).
    All dimension sizes are unique to catch axis-confusion bugs.
    """

    def f(x):
        A = x[:24].reshape(3, 2, 4)
        B = x[24:].reshape(3, 4, 5)
        return jnp.matmul(A, B).flatten()

    result = jacobian_sparsity(f, input_shape=(84,)).todense().astype(int)
    expected = _dot_general_jacobian(f, 84)
    np.testing.assert_array_equal(result, expected)


# Outer product
@pytest.mark.array_ops
def test_outer_product_via_dot_general():
    """Outer product via dot_general with no contracting dimensions.

    out[i,j] = a[i] * b[j], so each output depends on one element from each input.
    """

    def f(x):
        a = x[:3]
        b = x[3:]
        return jax.lax.dot_general(
            a, b, dimension_numbers=(([], []), ([], []))
        ).flatten()

    result = jacobian_sparsity(f, input_shape=(5,)).todense().astype(int)
    # out[i,j] depends on {i} from a and {3+j} from b.
    expected = np.zeros((6, 5), dtype=int)
    for i in range(3):
        for j in range(2):
            expected[i * 2 + j, i] = 1
            expected[i * 2 + j, 3 + j] = 1
    np.testing.assert_array_equal(result, expected)


# Size-1 dimensions
@pytest.mark.array_ops
def test_matmul_size_1_row():
    """Matrix multiply with size-1 row: (1,3) @ (3,2) → (1,2)."""

    def f(x):
        A = x[:3].reshape(1, 3)
        B = x[3:].reshape(3, 2)
        return (A @ B).flatten()

    result = jacobian_sparsity(f, input_shape=(9,)).todense().astype(int)
    # out[0,0] depends on A[0,:] = {0,1,2} and B[:,0] = {3,5,7}
    # out[0,1] depends on A[0,:] = {0,1,2} and B[:,1] = {4,6,8}
    expected = np.array(
        [
            [1, 1, 1, 1, 0, 1, 0, 1, 0],
            [1, 1, 1, 0, 1, 0, 1, 0, 1],
        ]
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_matmul_size_1_contract():
    """Contraction over size-1 dimension: (2,1) @ (1,3) → (2,3).

    Each output depends on exactly one lhs element and one rhs element.
    """

    def f(x):
        A = x[:2].reshape(2, 1)
        B = x[2:].reshape(1, 3)
        return (A @ B).flatten()

    result = jacobian_sparsity(f, input_shape=(5,)).todense().astype(int)
    expected = np.array(
        [
            [1, 0, 1, 0, 0],  # out[0,0]: A[0,0]={0}, B[0,0]={2}
            [1, 0, 0, 1, 0],  # out[0,1]: A[0,0]={0}, B[0,1]={3}
            [1, 0, 0, 0, 1],  # out[0,2]: A[0,0]={0}, B[0,2]={4}
            [0, 1, 1, 0, 0],  # out[1,0]: A[1,0]={1}, B[0,0]={2}
            [0, 1, 0, 1, 0],  # out[1,1]: A[1,0]={1}, B[0,1]={3}
            [0, 1, 0, 0, 1],  # out[1,2]: A[1,0]={1}, B[0,2]={4}
        ]
    )
    np.testing.assert_array_equal(result, expected)


# Higher-dimensional contractions
@pytest.mark.array_ops
def test_einsum_ij_jk():
    """Einstein summation ij,jk->ik (standard matmul via einsum)."""

    def f(x):
        A = x[:6].reshape(2, 3)
        B = x[6:].reshape(3, 2)
        return jnp.einsum("ij,jk->ik", A, B).flatten()

    result = jacobian_sparsity(f, input_shape=(12,)).todense().astype(int)
    expected = _dot_general_jacobian(f, 12)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_einsum_batched():
    """Batched einsum bij,bjk->bik."""

    def f(x):
        A = x[:24].reshape(3, 2, 4)
        B = x[24:].reshape(3, 4, 5)
        return jnp.einsum("bij,bjk->bik", A, B).flatten()

    result = jacobian_sparsity(f, input_shape=(84,)).todense().astype(int)
    expected = _dot_general_jacobian(f, 84)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_tensordot_axes_1():
    """Tensordot with axes=1: contracts the last axis of A with first of B."""

    def f(x):
        A = x[:6].reshape(2, 3)
        B = x[6:].reshape(3, 4)
        return jnp.tensordot(A, B, axes=1).flatten()

    result = jacobian_sparsity(f, input_shape=(18,)).todense().astype(int)
    expected = _dot_general_jacobian(f, 18)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_tensordot_axes_2():
    """Tensordot with axes=2: contracts last 2 axes of A with first 2 of B."""

    def f(x):
        A = x[:30].reshape(5, 2, 3)
        B = x[30:].reshape(2, 3, 4)
        return jnp.tensordot(A, B, axes=2).flatten()

    result = jacobian_sparsity(f, input_shape=(54,)).todense().astype(int)
    expected = _dot_general_jacobian(f, 54)
    np.testing.assert_array_equal(result, expected)


# Non-contiguous input state_indices
@pytest.mark.array_ops
def test_matmul_after_broadcast():
    """Matmul where inputs have non-trivial dep sets from a prior broadcast.

    Verifies that set merging handles non-singleton dep sets correctly.
    """

    def f(x):
        # Broadcast x (shape 3) to (2,3), then matmul with itself transposed.
        mat = jnp.broadcast_to(x, (2, 3))
        return (mat @ mat.T).flatten()

    result = jacobian_sparsity(f, input_shape=(3,)).todense().astype(int)
    # After broadcast, mat state_indices: each row is [{0},{1},{2}].
    # mat.T state_indices: each col is [{0},{1},{2}].
    # mat @ mat.T: all outputs depend on all inputs.
    expected = np.ones((4, 3), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_matmul_with_constant():
    """Matmul where one operand is a constant with value-level zeros.

    W = [[1,0,0],[0,1,0]], so out[0] depends only on x[0]
    and out[1] depends only on x[1].
    The handler skips contracting positions where the constant is zero.
    """

    def f(x):
        W = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        return W @ x

    result = jacobian_sparsity(f, input_shape=(3,)).todense().astype(int)
    expected = np.array([[1, 0, 0], [0, 1, 0]])
    np.testing.assert_array_equal(result, expected)


# Verify against jax.jacobian
@pytest.mark.array_ops
@pytest.mark.parametrize(
    ("desc", "f", "n"),
    [
        ("vecvec", lambda x: jnp.dot(x[:2], x[2:]).reshape(1), 4),
        ("matvec", lambda x: x[:6].reshape(2, 3) @ x[6:], 9),
        ("matmat", lambda x: (x[:6].reshape(2, 3) @ x[6:].reshape(3, 2)).flatten(), 12),
        (
            "outer",
            lambda x: jax.lax.dot_general(
                x[:3], x[3:], dimension_numbers=(([], []), ([], []))
            ).flatten(),
            5,
        ),
        (
            "batched_matmul",
            lambda x: jnp.matmul(
                x[:24].reshape(2, 3, 4), x[24:].reshape(2, 4, 5)
            ).flatten(),
            64,
        ),
        (
            "size_1_contract",
            lambda x: (x[:2].reshape(2, 1) @ x[2:].reshape(1, 3)).flatten(),
            5,
        ),
        (
            "multi_contract",
            lambda x: jax.lax.dot_general(
                x[:24].reshape(2, 3, 4),
                x[24:].reshape(3, 4, 5),
                dimension_numbers=(((1, 2), (0, 1)), ((), ())),
            ).flatten(),
            84,
        ),
    ],
    ids=[
        "vecvec",
        "matvec",
        "matmat",
        "outer",
        "batched_matmul",
        "size_1_contract",
        "multi_contract",
    ],
)
def test_against_jax_jacobian(desc, f, n):
    """Sparsity pattern matches the nonzero pattern of jax.jacobian."""
    result = jacobian_sparsity(f, input_shape=(n,)).todense().astype(int)
    expected = _dot_general_jacobian(f, n)
    np.testing.assert_array_equal(result, expected)


# Adversarial edge cases
@pytest.mark.array_ops
def test_multi_contract_dims():
    """Multiple contracting dimensions: contract on 2 axes simultaneously."""

    def f(x):
        A = x[:30].reshape(5, 3, 2)
        B = x[30:].reshape(3, 2, 4)
        # Contract dims (1,2) of A with (0,1) of B → shape (5, 4)
        return jax.lax.dot_general(
            A, B, dimension_numbers=(((1, 2), (0, 1)), ((), ()))
        ).flatten()

    result = jacobian_sparsity(f, input_shape=(54,)).todense().astype(int)
    expected = _dot_general_jacobian(f, 54)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_multi_batch_dims():
    """Multiple batch dimensions: batch on 2 axes."""

    def f(x):
        A = x[:24].reshape(2, 3, 4)
        B = x[24:].reshape(2, 3, 5)
        # Batch on (0,1), no contraction → shape (2, 3, 4, 5)
        return jax.lax.dot_general(
            A, B, dimension_numbers=(((), ()), ((0, 1), (0, 1)))
        ).flatten()

    result = jacobian_sparsity(f, input_shape=(54,)).todense().astype(int)
    expected = _dot_general_jacobian(f, 54)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_batch_and_contract():
    """Both batch and contracting dimensions present."""

    def f(x):
        A = x[:24].reshape(2, 3, 4)
        B = x[24:].reshape(2, 4, 5)
        # Batch on (0,), contract on (2,) of A with (1,) of B → shape (2, 3, 5)
        return jax.lax.dot_general(
            A, B, dimension_numbers=(((2,), (1,)), ((0,), (0,)))
        ).flatten()

    result = jacobian_sparsity(f, input_shape=(64,)).todense().astype(int)
    expected = _dot_general_jacobian(f, 64)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_size_1_contract():
    """Contraction over a size-1 dimension (effectively an outer product)."""

    def f(x):
        A = x[:6].reshape(3, 1, 2)
        B = x[6:].reshape(1, 4)
        # Contract dim 1 of A with dim 0 of B → shape (3, 2, 4)
        return jax.lax.dot_general(
            A, B, dimension_numbers=(((1,), (0,)), ((), ()))
        ).flatten()

    result = jacobian_sparsity(f, input_shape=(10,)).todense().astype(int)
    expected = _dot_general_jacobian(f, 10)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_double_matmul():
    """Chained matmul: (A @ B) @ C."""

    def f(x):
        A = x[:6].reshape(2, 3)
        B = x[6:18].reshape(3, 4)
        C = x[18:].reshape(4, 2)
        return ((A @ B) @ C).flatten()

    result = jacobian_sparsity(f, input_shape=(26,)).todense().astype(int)
    expected = _dot_general_jacobian(f, 26)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_vecdot():
    """jnp.vdot (flattened dot product) on matrices."""

    def f(x):
        A = x[:6].reshape(2, 3)
        B = x[6:].reshape(2, 3)
        return jnp.vdot(A, B).reshape(1)

    result = jacobian_sparsity(f, input_shape=(12,)).todense().astype(int)
    expected = np.ones((1, 12), dtype=int)
    np.testing.assert_array_equal(result, expected)


# Conservative audit
#
# The handler should produce strictly sparser patterns than conservative
# (where every output depends on the union of all inputs).


def _conservative_pattern(f, n):
    """Build the conservative pattern: every output depends on all inputs."""
    x = jnp.ones(n)
    out = f(x)
    out_size = np.array(out).size
    return np.ones((out_size, n), dtype=int)


@pytest.mark.array_ops
@pytest.mark.parametrize(
    ("desc", "f", "n"),
    [
        (
            "matmul",
            lambda x: (x[:6].reshape(2, 3) @ x[6:].reshape(3, 2)).flatten(),
            12,
        ),
        (
            "batched_matmul",
            lambda x: jnp.matmul(
                x[:24].reshape(2, 3, 4), x[24:].reshape(2, 4, 5)
            ).flatten(),
            64,
        ),
        (
            "size_1_contract",
            lambda x: (x[:2].reshape(2, 1) @ x[2:].reshape(1, 3)).flatten(),
            5,
        ),
    ],
    ids=["matmul", "batched_matmul", "size_1_contract"],
)
def test_strictly_sparser_than_conservative(desc, f, n):
    """Handler produces a strictly sparser pattern than conservative.

    Conservative makes every output depend on all inputs.
    The handler should have fewer nonzeros while still being a superset
    of the true Jacobian nonzero pattern.
    """
    result = jacobian_sparsity(f, input_shape=(n,)).todense().astype(int)
    conservative = _conservative_pattern(f, n)
    true_jac = _dot_general_jacobian(f, n)

    # Handler is a superset of the true pattern.
    assert np.all(result >= true_jac), "Handler missed a true nonzero"

    # Handler is a subset of conservative.
    assert np.all(result <= conservative), "Handler exceeds conservative"

    # Handler is strictly sparser (fewer total nonzeros).
    assert result.sum() < conservative.sum(), (
        f"Handler ({result.sum()}) is not strictly sparser "
        f"than conservative ({conservative.sum()})"
    )


# Asymmetric shapes
#
# All dimensions are unique to catch any axis-confusion bugs.


@pytest.mark.array_ops
def test_asymmetric_matmul_2x5_5x7():
    """Matmul (2,5) @ (5,7) with all unique dimension sizes."""

    def f(x):
        A = x[:10].reshape(2, 5)
        B = x[10:].reshape(5, 7)
        return (A @ B).flatten()

    n = 10 + 35
    result = jacobian_sparsity(f, input_shape=(n,)).todense().astype(int)
    expected = _dot_general_jacobian(f, n)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_asymmetric_batched_matmul_3x2x5_3x5x7():
    """Batched matmul (3,2,5) @ (3,5,7) with all unique dimension sizes."""

    def f(x):
        A = x[:30].reshape(3, 2, 5)
        B = x[30:].reshape(3, 5, 7)
        return jnp.matmul(A, B).flatten()

    n = 30 + 105
    result = jacobian_sparsity(f, input_shape=(n,)).todense().astype(int)
    expected = _dot_general_jacobian(f, n)
    np.testing.assert_array_equal(result, expected)


# Batch broadcasting
#
# jnp.matmul broadcasts leading dimensions,
# which lowers to broadcast_in_dim + dot_general.


@pytest.mark.array_ops
def test_batch_broadcast_matmul():
    """jnp.matmul with batch broadcasting: (2,3) matmul (3,2,3,4).

    The lhs (2,3) is broadcast to (3,2,2,3) before contraction.
    Each batch slice contracts independently.
    """

    def f(x):
        A = x[:6].reshape(2, 3)
        B = x[6:].reshape(3, 2, 3, 4)
        return jnp.matmul(A, B).flatten()

    n = 6 + 72
    result = jacobian_sparsity(f, input_shape=(n,)).todense().astype(int)
    expected = _dot_general_jacobian(f, n)
    np.testing.assert_array_equal(result, expected)


# Composition
#
# Elementwise → matmul → elementwise chain.
# Verifies that sparsity is preserved through multiple operations.


@pytest.mark.array_ops
def test_elementwise_matmul_elementwise_chain():
    """Elementwise → matmul → elementwise composition.

    Applies sin to inputs, matmul, then tanh to outputs.
    The sparsity structure should match a plain matmul
    since elementwise ops preserve sparsity.
    """

    def f(x):
        A = jnp.sin(x[:6].reshape(2, 3))
        B = jnp.sin(x[6:].reshape(3, 2))
        return jnp.tanh((A @ B).flatten())

    def f_plain(x):
        A = x[:6].reshape(2, 3)
        B = x[6:].reshape(3, 2)
        return (A @ B).flatten()

    result = jacobian_sparsity(f, input_shape=(12,)).todense().astype(int)
    # Elementwise ops don't change sparsity structure.
    expected = jacobian_sparsity(f_plain, input_shape=(12,)).todense().astype(int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_scalar_constant_times_vector():
    """Scalar constant dot vector via dot_general with no contracting dims.

    ``jnp.dot(scalar, vector)`` lowers to ``dot_general`` with
    ``dimension_numbers=(([], []), ([], []))``.
    The scalar operand has shape ``()`` so ``ravel_multi_index`` returns
    a scalar index that must be broadcast to match the output size.
    """

    def f(x):
        return jnp.dot(jnp.array(2.0), x)

    result = jacobian_sparsity(f, input_shape=(3,)).todense().astype(int)
    # out[i] = 2.0 * x[i], so the Jacobian is diagonal.
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_hilbert_matrix_pattern():
    """Hilbert-like pattern: sum of c[i,j] * x[i] * x[j].

    This is the pattern that triggers the bug in HILBERTA:
    constant-folded coefficients are scalar-broadcast during dot_general.
    """

    def f(x):
        n = x.shape[0]
        # Build Hilbert matrix coefficients.
        i = jnp.arange(1, n + 1, dtype=x.dtype)
        H = 1.0 / (i[:, None] + i[None, :] - 1.0)
        return x @ H @ x

    result = jacobian_sparsity(f, input_shape=(4,)).todense().astype(int)
    # Scalar output depends on all inputs.
    expected = np.ones((1, 4), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_scalar_zero_times_vector():
    """Scalar zero constant times vector produces an all-zero Jacobian.

    ``jnp.dot(jnp.array(0.0), x)`` has a scalar-broadcast constant.
    The broadcast-to-full-size path lets zero-skipping eliminate all terms.
    """

    def f(x):
        return jnp.dot(jnp.array(0.0), x)

    result = jacobian_sparsity(f, input_shape=(3,)).todense().astype(int)
    expected = np.zeros((3, 3), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_scalar_dot_zero_skipping():
    """Scalar dot product skips terms where a factor is a known zero.

    When a constant operand has zero entries,
    the corresponding contracting positions contribute nothing.
    ``jnp.dot(x, [1, 0, 1])`` has zero derivative w.r.t. ``x[1]``.
    """

    def f(x):
        w = jnp.array([1.0, 0.0, 1.0])
        return jnp.dot(x, w).reshape(1)

    result = jacobian_sparsity(f, input_shape=(3,)).todense().astype(int)
    expected = np.array([[1, 0, 1]], dtype=int)
    np.testing.assert_array_equal(result, expected)


# Size-0 dimension


@pytest.mark.array_ops
def test_dot_zero_contraction():
    """Dot product over a zero-length contraction dimension.

    Contracting over size 0 produces zero-valued output
    with no dependencies on the inputs.
    """

    def f(x):
        return jnp.dot(x[:0], jnp.zeros((0, 2)))

    result = jacobian_sparsity(f, input_shape=(3,))
    assert result.shape == (2, 3)
    assert result.nnz == 0
