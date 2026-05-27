"""Tests for linear algebra primitive handlers.

Linear algebra decompositions (lu, cholesky, qr, svd, eigh) use conservative
fallback since their outputs generally depend on all inputs.
"""

import jax.numpy as jnp
import jax.scipy.linalg as scipy_linalg
import numpy as np
import pytest

from asdex import jacobian


@pytest.mark.fallback
def test_cholesky_conservative():
    """Cholesky decomposition uses conservative fallback for the cholesky primitive."""

    def f(x):
        return jnp.linalg.cholesky(x)

    # 3x3 symmetric positive definite matrix
    A = jnp.array([[4.0, 2.0, 1.0], [2.0, 5.0, 2.0], [1.0, 2.0, 6.0]])
    J = jacobian(f, A, output_format="dense")(A)
    # Shape: (3, 3, 3, 3) = output (3,3) x input (3,3)
    assert J.shape == (3, 3, 3, 3)
    # Upper triangle of L is always 0, so 3 outputs have no dependencies
    J_flat = J.reshape(9, 9)
    assert np.sum(np.any(J_flat != 0, axis=1)) == 6


@pytest.mark.fallback
def test_qr_conservative():
    """QR decomposition uses conservative fallback."""

    def f(x):
        return jnp.linalg.qr(x)

    A = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])
    J_q, J_r = jacobian(f, A, output_format="dense")(A)
    # Q: (3,3) output, (3,3) input
    assert J_q.shape == (3, 3, 3, 3)
    # R: (3,3) output, (3,3) input
    assert J_r.shape == (3, 3, 3, 3)


@pytest.mark.fallback
def test_svd_conservative():
    """SVD uses conservative fallback."""

    def f(x):
        return jnp.linalg.svd(x)  # type: ignore[return-value]

    A = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])
    J_u, J_s, J_vh = jacobian(f, A, output_format="dense")(A)
    # U: (3,3), s: (3,), Vh: (3,3) - full_matrices=True by default
    assert J_u.shape == (3, 3, 3, 3)
    assert J_s.shape == (3, 3, 3)
    assert J_vh.shape == (3, 3, 3, 3)


@pytest.mark.fallback
def test_eigh_conservative():
    """Eigenvalue decomposition uses conservative fallback."""

    def f(x):
        return jnp.linalg.eigh(x)

    # 3x3 symmetric matrix
    A = jnp.array([[4.0, 2.0, 1.0], [2.0, 5.0, 2.0], [1.0, 2.0, 6.0]])
    J_vals, J_vecs = jacobian(f, A, output_format="dense")(A)
    # eigvals: (3,), eigvecs: (3,3)
    assert J_vals.shape == (3, 3, 3)
    assert J_vecs.shape == (3, 3, 3, 3)


@pytest.mark.fallback
def test_lu_conservative():
    """LU decomposition uses conservative fallback for the lu primitive."""

    def f(x):
        return scipy_linalg.lu(x)

    A = jnp.array([[2.0, 1.0, 1.0], [4.0, 3.0, 3.0], [8.0, 7.0, 9.0]])
    J_p, J_l, J_u = jacobian(f, A, output_format="dense")(A)
    # P, L, U all (3,3)
    assert J_p.shape == (3, 3, 3, 3)
    assert J_l.shape == (3, 3, 3, 3)
    assert J_u.shape == (3, 3, 3, 3)
