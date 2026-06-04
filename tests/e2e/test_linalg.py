"""End-to-end tests for linear algebra functions.

Verifies that asdex produces correct Jacobians for linalg primitives,
matching JAX's reference implementation.

These tests use relaxed tolerances (rtol=1e-5) because linalg operations
involve numerical algorithms that accumulate floating point errors.
"""

import warnings

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.scipy.linalg as scipy_linalg
import pytest

import asdex

warnings.filterwarnings("ignore", category=asdex.DenseColoringWarning)

RTOL = 1e-5
ATOL = 1e-6


def _random_matrix(key: jax.Array, shape: tuple[int, ...]) -> jax.Array:
    """Generate a random matrix with well-conditioned values."""
    return jr.normal(key, shape)


def _random_spd(key: jax.Array, n: int) -> jax.Array:
    """Generate a random symmetric positive definite matrix."""
    A = jr.normal(key, (n, n))
    return A @ A.T + n * jnp.eye(n)


def _random_symmetric(key: jax.Array, n: int) -> jax.Array:
    """Generate a random symmetric matrix."""
    A = jr.normal(key, (n, n))
    return (A + A.T) / 2


def _random_invertible(key: jax.Array, n: int) -> jax.Array:
    """Generate a random invertible matrix."""
    A = jr.normal(key, (n, n))
    return A + n * jnp.eye(n)


# QR decomposition


@pytest.mark.jacobian
@pytest.mark.parametrize("shape", [(2, 2), (3, 3), (4, 3), (3, 4)])
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo", "numpy_dense"])
def test_qr_jacobian(shape, mode, output_format, chunk_size, assert_trees_allclose):
    """QR decomposition Jacobian matches JAX for various shapes."""

    def f(x):
        return jnp.linalg.qr(x)

    A = _random_matrix(jr.key(0), shape)
    J = asdex.jacobian(
        f, A, mode=mode, output_format=output_format, chunk_size=chunk_size
    )(A)
    J_jax = jax.jacobian(f)(A)
    assert_trees_allclose(J, J_jax, rtol=RTOL, atol=ATOL)


# Cholesky decomposition


@pytest.mark.jacobian
@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo", "numpy_dense"])
def test_cholesky_jacobian(n, mode, output_format, chunk_size, assert_trees_allclose):
    """Cholesky decomposition Jacobian matches JAX for various sizes."""

    def f(x):
        return jnp.linalg.cholesky(x)

    A = _random_spd(jr.key(1), n)
    J = asdex.jacobian(
        f, A, mode=mode, output_format=output_format, chunk_size=chunk_size
    )(A)
    J_jax = jax.jacobian(f)(A)
    assert_trees_allclose(J, J_jax, rtol=RTOL, atol=ATOL)


# SVD


@pytest.mark.jacobian
@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo", "numpy_dense"])
def test_svd_jacobian(n, mode, output_format, chunk_size, assert_trees_allclose):
    """SVD Jacobian matches JAX for various sizes.

    Only tests square matrices because JAX's SVD JVP is not implemented
    for non-square ("full") matrices.
    """

    def f(x):
        return jnp.linalg.svd(x)  # type: ignore[return-value]

    A = _random_matrix(jr.key(2), (n, n))
    J = asdex.jacobian(
        f, A, mode=mode, output_format=output_format, chunk_size=chunk_size
    )(A)
    J_jax = jax.jacobian(f)(A)
    assert_trees_allclose(J, J_jax, rtol=RTOL, atol=ATOL)


# Eigenvalue decomposition


@pytest.mark.jacobian
@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo", "numpy_dense"])
def test_eigh_jacobian(n, mode, output_format, chunk_size, assert_trees_allclose):
    """Eigenvalue decomposition Jacobian matches JAX for various sizes."""

    def f(x):
        return jnp.linalg.eigh(x)

    A = _random_symmetric(jr.key(3), n)
    J = asdex.jacobian(
        f, A, mode=mode, output_format=output_format, chunk_size=chunk_size
    )(A)
    J_jax = jax.jacobian(f)(A)
    assert_trees_allclose(J, J_jax, rtol=RTOL, atol=ATOL)


# LU decomposition


@pytest.mark.jacobian
@pytest.mark.parametrize("n", [2, 3, 4])
@pytest.mark.parametrize("mode", ["fwd", "rev"])
@pytest.mark.parametrize("output_format", ["dense", "bcoo", "numpy_dense"])
def test_lu_jacobian(n, mode, output_format, chunk_size, assert_trees_allclose):
    """LU decomposition Jacobian matches JAX for various sizes."""

    def f(x):
        return scipy_linalg.lu(x)

    A = _random_invertible(jr.key(4), n)
    J = asdex.jacobian(
        f, A, mode=mode, output_format=output_format, chunk_size=chunk_size
    )(A)
    J_jax = jax.jacobian(f)(A)
    assert_trees_allclose(J, J_jax, rtol=RTOL, atol=ATOL)
