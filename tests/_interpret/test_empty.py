"""Tests for the empty propagation handler.

Since JAX 0.11, ``jnp.empty`` and ``jnp.empty_like`` emit the ``empty`` primitive
instead of a broadcast of zeros.
Its contents are undefined,
so it contributes no dependencies but must not be treated as known zeros.

https://docs.jax.dev/en/latest/_autosummary/jax.numpy.empty.html
"""

import jax.numpy as jnp
import numpy as np
import pytest
from jax import lax

from asdex import jacobian_sparsity
from tests._utils import assert_jacobian_sparsity_exact


@pytest.mark.array_ops
def test_empty_filled_by_set():
    """An empty buffer fully overwritten by ``.at[:].set`` follows the written values."""

    def f(x):
        out = jnp.empty((3,))
        return out.at[:].set(x[:3] * 2.0)

    x = np.array([1.0, 2.0, 3.0, 4.0])
    result = jacobian_sparsity(f, x).todense().astype(int)
    expected = np.array(
        [
            [1, 0, 0, 0],  # out[0] = 2 * x[0]
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)
    assert_jacobian_sparsity_exact(f, x)


@pytest.mark.array_ops
def test_empty_partially_filled():
    """Positions that are never written keep the empty buffer's empty index sets."""

    def f(x):
        return jnp.empty((4,)).at[1:].set(x[:-1])

    result = jacobian_sparsity(f, np.zeros(4)).todense().astype(int)
    expected = np.array(
        [
            [0, 0, 0, 0],  # out[0] = uninitialized
            [1, 0, 0, 0],  # out[1] = x[0]
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_empty_like_has_no_dependency_on_its_argument():
    """``jnp.empty_like(x)`` only reads the shape and dtype of x, not its values."""

    def f(x):
        return jnp.empty_like(x).at[:, 0].set(x[:, 1])

    result = jacobian_sparsity(f, np.zeros((2, 3))).todense().astype(int)
    expected = np.array(
        [
            [0, 1, 0, 0, 0, 0],  # out[0,0] = x[0,1]
            [0, 0, 0, 0, 0, 0],  # out[0,1] = uninitialized
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0],  # out[1,0] = x[1,1]
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_empty_is_not_treated_as_zeros_in_mul():
    """Multiplying by an empty buffer keeps the dependency on the other operand.

    Before JAX 0.11, ``jnp.empty`` lowered to zeros,
    which let ``mul`` clear the pattern.
    The buffer is now uninitialized, so its values may be nonzero.
    """

    def f(x):
        return x * jnp.empty((3, 2))

    result = jacobian_sparsity(f, np.zeros((3, 2))).todense().astype(int)
    expected = np.eye(6, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_empty_is_not_treated_as_zeros_in_dot_general():
    """A matmul with an empty matrix keeps every contraction term."""

    def f(x):
        return jnp.empty((2, 3)) @ x

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.ones((2, 3), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
@pytest.mark.parametrize(
    "shape",
    [(3,), (3, 4), (2, 3, 4)],
    ids=["1d", "2d", "3d"],
)
def test_lax_empty_has_no_dependencies(shape):
    """``lax.empty`` added to the input contributes no dependencies of its own."""
    n = int(np.prod(shape))

    def f(x):
        return lax.empty(shape, jnp.float32) + x.reshape(shape)

    result = jacobian_sparsity(f, np.zeros(n)).todense().astype(int)
    np.testing.assert_array_equal(result, np.eye(n, dtype=int))


@pytest.mark.array_ops
@pytest.mark.parametrize("shape", [(0,), (0, 3), (2, 0, 4)], ids=["1d", "2d", "3d"])
def test_lax_empty_zero_size(shape):
    """A zero-size ``lax.empty`` does not crash and produces no rows."""
    n = int(np.prod(shape))

    def f(x):
        return lax.empty(shape, jnp.float32) + x[:n].reshape(shape)

    pattern = jacobian_sparsity(f, np.zeros(5))
    assert pattern.shape == (0, 5)
    assert pattern.nnz == 0


@pytest.mark.array_ops
def test_empty_zero_size():
    """A zero-size empty buffer produces no rows."""

    def f(x):
        return jnp.concatenate([jnp.empty((0,)), x])

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_empty_scalar():
    """A scalar empty buffer added to the input keeps the diagonal pattern."""

    def f(x):
        return x + jnp.empty(())

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)
