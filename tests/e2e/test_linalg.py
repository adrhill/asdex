"""End-to-end tests for linear algebra functions.

Verifies that asdex produces correct Jacobians for linalg primitives,
matching JAX's reference implementation.

These tests use relaxed tolerances (rtol=1e-5) because linalg operations
involve numerical algorithms that accumulate floating point errors.
"""

import warnings

import jax
import jax.numpy as jnp
import jax.scipy.linalg as scipy_linalg
import pytest

import asdex

warnings.filterwarnings("ignore", category=asdex.DenseColoringWarning)

RTOL = 1e-5
ATOL = 1e-6


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_qr_jacobian(mode, output_format, chunk_size, assert_trees_allclose):
    """QR decomposition Jacobian matches JAX."""

    def f(x):
        return jnp.linalg.qr(x)

    A = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])
    J = asdex.jacobian(
        f, A, mode=mode, output_format=output_format, chunk_size=chunk_size
    )(A)
    J_jax = jax.jacobian(f)(A)
    assert_trees_allclose(J, J_jax, rtol=RTOL, atol=ATOL)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_qr_non_square(mode, output_format, chunk_size, assert_trees_allclose):
    """QR decomposition works with non-square matrices."""

    def f(x):
        return jnp.linalg.qr(x)

    # Tall matrix (more rows than columns)
    A = jnp.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]])
    J = asdex.jacobian(
        f, A, mode=mode, output_format=output_format, chunk_size=chunk_size
    )(A)
    J_jax = jax.jacobian(f)(A)
    assert_trees_allclose(J, J_jax, rtol=RTOL, atol=ATOL)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_cholesky_jacobian(mode, output_format, chunk_size, assert_trees_allclose):
    """Cholesky decomposition Jacobian matches JAX."""

    def f(x):
        return jnp.linalg.cholesky(x)

    # Symmetric positive definite matrix
    A = jnp.array([[4.0, 2.0, 1.0], [2.0, 5.0, 2.0], [1.0, 2.0, 6.0]])
    J = asdex.jacobian(
        f, A, mode=mode, output_format=output_format, chunk_size=chunk_size
    )(A)
    J_jax = jax.jacobian(f)(A)
    assert_trees_allclose(J, J_jax, rtol=RTOL, atol=ATOL)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_svd_jacobian(mode, output_format, chunk_size, assert_trees_allclose):
    """SVD Jacobian matches JAX."""

    def f(x):
        return jnp.linalg.svd(x)  # type: ignore[return-value]

    A = jnp.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 10.0]])
    J = asdex.jacobian(
        f, A, mode=mode, output_format=output_format, chunk_size=chunk_size
    )(A)
    J_jax = jax.jacobian(f)(A)
    assert_trees_allclose(J, J_jax, rtol=RTOL, atol=ATOL)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_eigh_jacobian(mode, output_format, chunk_size, assert_trees_allclose):
    """Eigenvalue decomposition Jacobian matches JAX."""

    def f(x):
        return jnp.linalg.eigh(x)

    # Symmetric matrix
    A = jnp.array([[4.0, 2.0, 1.0], [2.0, 5.0, 2.0], [1.0, 2.0, 6.0]])
    J = asdex.jacobian(
        f, A, mode=mode, output_format=output_format, chunk_size=chunk_size
    )(A)
    J_jax = jax.jacobian(f)(A)
    assert_trees_allclose(J, J_jax, rtol=RTOL, atol=ATOL)


@pytest.mark.jacobian
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo"])
def test_lu_jacobian(mode, output_format, chunk_size, assert_trees_allclose):
    """LU decomposition Jacobian matches JAX."""

    def f(x):
        return scipy_linalg.lu(x)

    A = jnp.array([[2.0, 1.0, 1.0], [4.0, 3.0, 3.0], [8.0, 7.0, 9.0]])
    J = asdex.jacobian(
        f, A, mode=mode, output_format=output_format, chunk_size=chunk_size
    )(A)
    J_jax = jax.jacobian(f)(A)
    assert_trees_allclose(J, J_jax, rtol=RTOL, atol=ATOL)
