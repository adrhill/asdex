"""Tests for linear algebra primitive handlers.

Linear algebra decompositions (lu, cholesky, qr, svd, eigh) use conservative
fallback since their outputs generally depend on all inputs.
"""

import jax.numpy as jnp
import jax.scipy.linalg as scipy_linalg
import numpy as np
import pytest

from asdex import jacobian_sparsity


@pytest.mark.fallback
def test_cholesky_conservative():
    """Cholesky decomposition uses conservative fallback for the cholesky primitive.

    The upper triangle of L is always zero, so those outputs have no dependencies.
    The lower triangle entries each depend on all inputs (conservative).
    """

    def f(x):
        return jnp.linalg.cholesky(x)

    # 3x3 symmetric positive definite matrix
    A = jnp.array([[4.0, 2.0, 1.0], [2.0, 5.0, 2.0], [1.0, 2.0, 6.0]])
    pattern = jacobian_sparsity(f, A)
    result = pattern.todense()

    # Output L[i,j] is non-zero only for i >= j (lower triangle)
    # Conservative fallback: each non-zero output depends on all 9 inputs
    # Output indices (row-major): 0,1,2 / 3,4,5 / 6,7,8
    # Lower triangle: (0,0)=0, (1,0)=3, (1,1)=4, (2,0)=6, (2,1)=7, (2,2)=8
    expected = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1],  # L[0,0] depends on all
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # L[0,1] = 0 (upper triangle)
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # L[0,2] = 0 (upper triangle)
            [1, 1, 1, 1, 1, 1, 1, 1, 1],  # L[1,0] depends on all
            [1, 1, 1, 1, 1, 1, 1, 1, 1],  # L[1,1] depends on all
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # L[1,2] = 0 (upper triangle)
            [1, 1, 1, 1, 1, 1, 1, 1, 1],  # L[2,0] depends on all
            [1, 1, 1, 1, 1, 1, 1, 1, 1],  # L[2,1] depends on all
            [1, 1, 1, 1, 1, 1, 1, 1, 1],  # L[2,2] depends on all
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.fallback
def test_qr_conservative():
    """QR decomposition uses conservative fallback.

    Returns flattened pattern for (Q, R) where both are (3,3).
    Q genuinely depends on all inputs.
    R is upper triangular, so lower triangle outputs have no dependencies.

    TODO(qr): R's lower triangle should have zero rows.
    """

    def f(x):
        return jnp.linalg.qr(x)

    A = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])
    pattern = jacobian_sparsity(f, A)
    result = pattern.todense()

    # Q: (3,3) -> all outputs depend on all inputs
    expected_q = np.ones((9, 9), dtype=int)

    # R: (3,3) -> upper triangular
    # Lower triangle is always 0 (no dependencies)
    expected_r = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1],  # R[0,0] depends on all
            [1, 1, 1, 1, 1, 1, 1, 1, 1],  # R[0,1]
            [1, 1, 1, 1, 1, 1, 1, 1, 1],  # R[0,2]
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # R[1,0] = 0 (lower triangle)
            [1, 1, 1, 1, 1, 1, 1, 1, 1],  # R[1,1]
            [1, 1, 1, 1, 1, 1, 1, 1, 1],  # R[1,2]
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # R[2,0] = 0 (lower triangle)
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # R[2,1] = 0 (lower triangle)
            [1, 1, 1, 1, 1, 1, 1, 1, 1],  # R[2,2]
        ],
        dtype=int,
    )

    expected = np.vstack([expected_q, expected_r])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.fallback
def test_svd_conservative():
    """SVD uses conservative fallback.

    Returns flattened pattern for (U, s, Vh) where U:(3,3), s:(3,), Vh:(3,3).
    All outputs genuinely depend on all inputs.
    """

    def f(x):
        return jnp.linalg.svd(x)  # type: ignore[return-value]

    A = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])
    pattern = jacobian_sparsity(f, A)
    result = pattern.todense()

    expected_u = np.ones((9, 9), dtype=int)  # U: (3,3)
    expected_s = np.ones((3, 9), dtype=int)  # s: (3,)
    expected_vh = np.ones((9, 9), dtype=int)  # Vh: (3,3)

    expected = np.vstack([expected_u, expected_s, expected_vh])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.fallback
def test_eigh_conservative():
    """Eigenvalue decomposition uses conservative fallback.

    Returns flattened pattern for (eigvals, eigvecs) where eigvals:(3,), eigvecs:(3,3).
    All outputs genuinely depend on all inputs.
    """

    def f(x):
        return jnp.linalg.eigh(x)

    # 3x3 symmetric matrix
    A = jnp.array([[4.0, 2.0, 1.0], [2.0, 5.0, 2.0], [1.0, 2.0, 6.0]])
    pattern = jacobian_sparsity(f, A)
    result = pattern.todense()

    expected_vals = np.ones((3, 9), dtype=int)  # eigvals: (3,)
    expected_vecs = np.ones((9, 9), dtype=int)  # eigvecs: (3,3)

    expected = np.vstack([expected_vals, expected_vecs])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.fallback
def test_lu_conservative():
    """LU decomposition uses conservative fallback for the lu primitive.

    Returns flattened pattern for (P, L, U) where all are (3,3).
    P is a permutation matrix (no gradient dependencies).
    L is unit lower triangular (1s on diagonal, zeros above).
    U is upper triangular (zeros below diagonal).
    """

    def f(x):
        return scipy_linalg.lu(x)

    A = jnp.array([[2.0, 1.0, 1.0], [4.0, 3.0, 3.0], [8.0, 7.0, 9.0]])
    pattern = jacobian_sparsity(f, A)
    result = pattern.todense()

    # P: (3,3) -> permutation matrix has no gradient dependencies
    expected_p = np.zeros((9, 9), dtype=int)

    # L: (3,3) -> unit lower triangular
    # Diagonal = 1 (constant), upper triangle = 0, lower triangle depends on all
    expected_l = np.array(
        [
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # L[0,0] = 1 (constant)
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # L[0,1] = 0 (upper triangle)
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # L[0,2] = 0 (upper triangle)
            [1, 1, 1, 1, 1, 1, 1, 1, 1],  # L[1,0] depends on all
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # L[1,1] = 1 (constant)
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # L[1,2] = 0 (upper triangle)
            [1, 1, 1, 1, 1, 1, 1, 1, 1],  # L[2,0] depends on all
            [1, 1, 1, 1, 1, 1, 1, 1, 1],  # L[2,1] depends on all
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # L[2,2] = 1 (constant)
        ],
        dtype=int,
    )

    # U: (3,3) -> upper triangular
    # Lower triangle = 0, diagonal and upper triangle depend on all
    expected_u = np.array(
        [
            [1, 1, 1, 1, 1, 1, 1, 1, 1],  # U[0,0] depends on all
            [1, 1, 1, 1, 1, 1, 1, 1, 1],  # U[0,1]
            [1, 1, 1, 1, 1, 1, 1, 1, 1],  # U[0,2]
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # U[1,0] = 0 (lower triangle)
            [1, 1, 1, 1, 1, 1, 1, 1, 1],  # U[1,1]
            [1, 1, 1, 1, 1, 1, 1, 1, 1],  # U[1,2]
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # U[2,0] = 0 (lower triangle)
            [0, 0, 0, 0, 0, 0, 0, 0, 0],  # U[2,1] = 0 (lower triangle)
            [1, 1, 1, 1, 1, 1, 1, 1, 1],  # U[2,2]
        ],
        dtype=int,
    )

    expected = np.vstack([expected_p, expected_l, expected_u])
    np.testing.assert_array_equal(result, expected)
