"""Tests for linear algebra primitive handlers.

Linear algebra decompositions (lu, cholesky, svd, eigh, eig, schur, ...)
and triangular solves share the conservative fallback,
since their outputs generally depend on all inputs.
Only `qr` has a handler that models structural zeros.
"""

import jax
import jax.lax.linalg
import jax.numpy as jnp
import jax.scipy.linalg as scipy_linalg
import numpy as np
import pytest

from asdex import hessian, hessian_sparsity, jacobian_sparsity
from tests._utils import (
    assert_hessian_sparsity_exact,
    assert_jacobian_sparsity_conservative,
)


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


def test_qr_pivoting_unsupported():
    """QR with pivoting adds a permutation output, which the handler rejects loudly.

    TODO(qr): support pivoting by modeling the pivots output
    (integer permutation, empty index sets) alongside Q and R.
    """

    def f(x):
        q, _, _ = jax.lax.linalg.qr(x, pivoting=True)
        return q

    A = jnp.array([[1.0, 2.0], [3.0, 4.0]])
    with pytest.raises(NotImplementedError, match="expects two outputs"):
        jacobian_sparsity(f, A)


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


@pytest.mark.fallback
def test_triangular_solve_conservative():
    """Triangular solve uses conservative fallback for the triangular_solve primitive.

    The detected pattern is currently fully dense:
    every solution entry depends on every entry of A and b.
    TODO(triangular_solve): forward substitution only reads A's lower triangle,
    and x[i] only depends on rows 0..i of it and on b[0..i]:
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # x[0] = b[0] / A[0,0]
        [1, 0, 0, 1, 1, 0, 0, 0, 0, 1, 1, 0],  # x[1] needs A[1,:2], b[1], x[0]
        [1, 0, 0, 1, 1, 0, 1, 1, 1, 1, 1, 1],  # x[2] needs A[2,:3], b[2], x[0], x[1]
    """

    def f(x):
        A = x[:9].reshape(3, 3)
        b = x[9:]
        return scipy_linalg.solve_triangular(A, b, lower=True)

    x = jnp.arange(1.0, 13.0)
    result = jacobian_sparsity(f, x).todense().astype(int)
    expected = np.ones((3, 12), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.hessian
@pytest.mark.skipif(
    jax.__version_info__ >= (0, 11, 1),
    reason="det of small matrices emits triangular_solve since JAX 0.11.1",
)
@pytest.mark.parametrize("n", [2, 3])
def test_det_hessian_small_matrix(n):
    """Before JAX 0.11.1, det of 2x2 and 3x3 matrices expands into products of entries.

    The Hessian pattern is therefore precise:
    ∂²det/∂A[i,j]∂A[k,l] is nonzero only for i ≠ k and j ≠ l.
    """

    def f(x):
        return jnp.linalg.det(x.reshape(n, n))

    x = jnp.arange(1.0, n * n + 1.0) + jnp.eye(n).flatten()
    assert_hessian_sparsity_exact(f, x)


@pytest.mark.fallback
@pytest.mark.hessian
@pytest.mark.skipif(
    jax.__version_info__ < (0, 11, 1),
    reason="det of small matrices emits triangular_solve since JAX 0.11.1",
)
@pytest.mark.parametrize("n", [2, 3])
def test_det_hessian_small_matrix_conservative(n):
    """The Hessian of det for 2x2 and 3x3 matrices is currently fully dense.

    Since JAX 0.11.1, ``jnp.linalg.det`` computes small determinants
    from a closed-form LU decomposition
    whose derivative emits ``triangular_solve``,
    which uses the conservative fallback.
    TODO(triangular_solve): ∂²det/∂A[i,j]∂A[k,l] is nonzero only for i ≠ k and j ≠ l.
    For A = [[a, b], [c, d]], det = a·d - b·c, so the Hessian is antidiagonal:
        [0, 0, 0, 1],  # ∂²det/∂a∂d = 1
        [0, 0, 1, 0],  # ∂²det/∂b∂c = -1
        [0, 1, 0, 0],
        [1, 0, 0, 0],
    """

    def f(x):
        return jnp.linalg.det(x.reshape(n, n))

    x = jnp.arange(1.0, n * n + 1.0) + jnp.eye(n).flatten()
    result = hessian_sparsity(f, x).todense().astype(int)
    np.testing.assert_array_equal(result, np.ones((n * n, n * n), dtype=int))

    H = hessian(f, x, output_format="dense")(x)
    np.testing.assert_allclose(H, jax.hessian(f)(x), atol=1e-5)


# A 3x3 symmetric positive definite matrix, and its flattened form.
_SPD = jnp.array([[4.0, 2.0, 1.0], [2.0, 5.0, 2.0], [1.0, 2.0, 6.0]])
_SPD_FLAT = _SPD.ravel()


@pytest.mark.fallback
@pytest.mark.parametrize(
    ("f", "x", "expected_shape", "has_ad_rule"),
    [
        pytest.param(
            # .real keeps the output real, so a numerical Jacobian exists.
            lambda x: jnp.real(jnp.linalg.eigvals(x.reshape(3, 3))),
            _SPD_FLAT,
            (3, 9),
            True,
            id="eigvals",
        ),
        pytest.param(
            # eigenvalues (3,) plus right eigenvectors (3, 3), both complex.
            lambda x: jnp.linalg.eig(x.reshape(3, 3)),
            _SPD_FLAT,
            (12, 9),
            False,
            id="eig",
        ),
        pytest.param(scipy_linalg.schur, _SPD, (18, 9), False, id="schur"),
        pytest.param(jax.lax.linalg.hessenberg, _SPD, (11, 9), False, id="hessenberg"),
        pytest.param(
            jax.lax.linalg.tridiagonal, _SPD, (16, 9), False, id="tridiagonal"
        ),
        pytest.param(
            lambda A: jax.lax.linalg.cholesky_update(A, A[:, 0]),
            _SPD,
            (9, 9),
            False,
            id="cholesky_update",
        ),
        pytest.param(
            lambda x: jax.lax.linalg.lu_pivots_to_permutation(
                jax.lax.linalg.lu(x.reshape(3, 3))[1], 3
            ).astype(x.dtype),
            _SPD_FLAT,
            (3, 9),
            True,
            id="lu_pivots_to_permutation",
        ),
        pytest.param(
            lambda x: jax.lax.linalg.symmetric_product(
                x[:9].reshape(3, 3), x[9:].reshape(3, 3)
            ),
            jnp.arange(1.0, 19.0),
            (9, 18),
            False,
            id="symmetric_product",
        ),
        pytest.param(
            lambda x: jax.lax.linalg.householder_product(x[:9].reshape(3, 3), x[9:]),
            jnp.arange(1.0, 13.0),
            (9, 12),
            False,
            id="householder_product",
        ),
        pytest.param(
            lambda x: jax.lax.linalg.ormqr(
                x[:9].reshape(3, 3), x[9:12], x[12:].reshape(3, 3)
            ),
            jnp.arange(1.0, 22.0),
            (9, 21),
            False,
            id="ormqr",
        ),
        pytest.param(
            lambda x: jax.lax.linalg.tridiagonal_solve(
                x[:3].reshape(1, 3),
                x[3:6].reshape(1, 3),
                x[6:9].reshape(1, 3),
                x[9:12].reshape(1, 3, 1),
            ).flatten(),
            jnp.arange(1.0, 13.0),
            (3, 12),
            True,
            id="tridiagonal_solve",
        ),
    ],
)
def test_linalg_decompositions_conservative(f, x, expected_shape, has_ad_rule):
    """Every element of every linalg decomposition output depends on every input.

    These primitives share the conservative fallback with `lu` and `cholesky`.
    `eig`, `schur`, `hessenberg`, and `tridiagonal` are multi-output,
    so a fallback that drops an output fails this test.

    JAX has no differentiation rule for most of these primitives,
    so `has_ad_rule` marks the three cases that also have a numerical Jacobian
    to compare against. For the others the all-ones matrix is the reference:
    the fallback unions every operand into every output, so it cannot miss.

    TODO(schur): T is quasi-triangular, so its entries below the first
    subdiagonal are always zero.
    TODO(lu_pivots_to_permutation): the permutation is piecewise constant,
    so all of its rows should be empty. Its numerical Jacobian is already zero.
    The remaining primitives have no fixed structural zeros:
    `hessenberg` and `tridiagonal` return the reflectors below the band
    in their matrix output, and the rest are generically dense.
    """
    result = jacobian_sparsity(f, x).todense().astype(int)
    np.testing.assert_array_equal(result, np.ones(expected_shape, dtype=int))
    if has_ad_rule:
        assert_jacobian_sparsity_conservative(f, x)
