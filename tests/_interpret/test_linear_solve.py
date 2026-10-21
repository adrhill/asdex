"""Tests for the custom_linear_solve handler."""

import jax
import jax.numpy as jnp
import jax.scipy.sparse.linalg
import numpy as np
import pytest

from asdex import jacobian_sparsity
from tests._utils import (
    assert_flat_jacobian_sparsity_exact,
    assert_hessian_sparsity_exact,
    assert_jacobian_sparsity_conservative,
)


@pytest.mark.parametrize(
    "A",
    [
        np.zeros((0, 0)),
        np.array([[2.0]]),
        np.array([[2.0, 1.0], [3.0, 4.0]]),
        np.array([[3.0, 1.0, 2.0], [1.0, 4.0, 0.0], [2.0, 1.0, 5.0]]),
    ],
    ids=["0x0", "1x1", "2x2", "3x3"],
)
def test_solve_jacobian(A):
    """Solving a linear system has a dense Jacobian in a general matrix."""
    b = jnp.arange(1.0, A.shape[0] + 1.0)
    assert_flat_jacobian_sparsity_exact(lambda A: jnp.linalg.solve(A, b), A)


def test_solve_in_expression():
    """A solve composed with elementwise and cumulative operations stays exact."""
    A = np.array([[3.0, 1.0, 2.0], [1.0, 4.0, 0.0], [2.0, 1.0, 5.0]])

    def f(A):
        x = jnp.linalg.solve(A, jnp.array([1.0, 2.0, 3.0]))
        return jnp.concatenate([jnp.sin(x), jnp.cumsum(x**2)])

    assert_flat_jacobian_sparsity_exact(f, A)


def test_inv_jacobian():
    """jnp.linalg.inv routes through custom_linear_solve and has a dense Jacobian."""
    A = np.array([[3.0, 1.0, 2.0], [1.0, 4.0, 0.0], [2.0, 1.0, 5.0]])
    assert_flat_jacobian_sparsity_exact(jnp.linalg.inv, A)


def test_custom_linear_solve_mixed_dependencies():
    """custom_linear_solve keeps operator and right-hand side dependencies distinct."""

    def f(x):
        scale = x[:3]
        rhs = x[3:].reshape(3, 2)

        def matvec(v):
            return scale[:, None] * v

        def solve(matvec, rhs):
            return rhs / scale[:, None]

        return jax.lax.custom_linear_solve(matvec, rhs, solve, symmetric=True)

    x = np.arange(1.0, 10.0)
    assert_flat_jacobian_sparsity_exact(f, x)
    result = jacobian_sparsity(f, x).todense().astype(int)
    expected = np.array(
        [
            [1, 0, 0, 1, 0, 0, 0, 0, 0],  # x[0,0] = rhs[0,0] / scale[0]
            [1, 0, 0, 0, 1, 0, 0, 0, 0],  # x[0,1] = rhs[0,1] / scale[0]
            [0, 1, 0, 0, 0, 1, 0, 0, 0],  # x[1,0] = rhs[1,0] / scale[1]
            [0, 1, 0, 0, 0, 0, 1, 0, 0],  # x[1,1] = rhs[1,1] / scale[1]
            [0, 0, 1, 0, 0, 0, 0, 1, 0],  # x[2,0] = rhs[2,0] / scale[2]
            [0, 0, 1, 0, 0, 0, 0, 0, 1],  # x[2,1] = rhs[2,1] / scale[2]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "solver",
    [jax.scipy.sparse.linalg.cg, jax.scipy.sparse.linalg.gmres],
    ids=["cg", "gmres"],
)
def test_iterative_solver_jacobian(solver):
    """Iterative solvers lower to custom_linear_solve and have a dense solution map."""
    A = np.array([[4.0, 1.0, 0.5], [1.0, 3.0, 1.0], [0.5, 1.0, 5.0]])

    def f(b):
        return solver(lambda v: A @ v, b)[0]

    assert_flat_jacobian_sparsity_exact(f, np.ones(3))


def test_solve_hessian():
    """Differentiating through a solve exercises both the solve and transpose solve jaxprs."""

    def f(x):
        A = x[:9].reshape(3, 3) + jnp.eye(3) * 3.0
        return jnp.sum(jnp.linalg.solve(A, x[9:]))

    assert_hessian_sparsity_exact(f, np.linspace(0.3, 1.4, 12))


@pytest.mark.fallback
def test_solve_sparse_matrix_conservative():
    """A sparse matrix loses its structure because its solve runs ``lu``.

    For diagonal ``A``, ``x[i] = b[i] / A[i,i]``,
    so row ``i`` should depend only on ``A[i,i]`` and ``b[i]``.

    TODO(lu): the ``lu`` fallback makes every solution entry depend on the whole
    matrix. The precise pattern is
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # x[0] = b[0] / A[0,0]
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0],  # x[1] = b[1] / A[1,1]
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1],  # x[2] = b[2] / A[2,2]
    """

    def f(x):
        A = x[:9].reshape(3, 3)
        return jnp.linalg.solve(A, x[9:])

    x = np.concatenate([np.diag([2.0, 3.0, 4.0]).ravel(), np.arange(1.0, 4.0)])
    assert_jacobian_sparsity_conservative(f, x)
    result = jacobian_sparsity(f, x).todense().astype(int)
    np.testing.assert_array_equal(result, np.ones((3, 12), dtype=int))


@pytest.mark.fallback
def test_solve_batched_conservative():
    """Batched solves are conservative across batches because ``lu`` is conservative.

    TODO(lu): each output block ``b`` should only depend on
    ``A[b]`` and ``rhs[b]``, giving a block-diagonal pattern instead of all 48 columns.
    """

    def f(x):
        A = x[:36].reshape(4, 3, 3)
        b = x[36:].reshape(4, 3)
        return jnp.linalg.solve(A, b[..., None]).squeeze(-1).ravel()

    A = np.eye(3)[None] * 2.0 + np.arange(4)[:, None, None]
    b = np.ones((4, 3))
    x = np.concatenate([A.ravel(), b.ravel()])
    assert_jacobian_sparsity_conservative(f, x)
    result = jacobian_sparsity(f, x).todense().astype(int)
    np.testing.assert_array_equal(result, np.ones((12, 48), dtype=int))
