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
from tests._utils import (
    assert_flat_jacobian_sparsity_exact,
    assert_jacobian_sparsity_conservative,
    assert_jacobian_sparsity_exact,
    numerical_flat_jacobian_sparsity,
)


@pytest.mark.array_ops
def test_empty_filled_by_set():
    """An empty buffer fully overwritten by ``.at[:].set`` follows the written values."""

    def f(x):
        out = jnp.empty((3,))
        return out.at[:].set(x[:3] * 2.0)

    assert_jacobian_sparsity_exact(f, np.array([1.0, 2.0, 3.0, 4.0]))


@pytest.mark.array_ops
def test_empty_partially_filled():
    """Positions that are never written keep the empty buffer's empty index sets."""

    def f(x):
        return jnp.empty((4,)).at[1:].set(x[:-1])

    assert_jacobian_sparsity_exact(f, np.array([1.0, 2.0, 3.0, 4.0]))


@pytest.mark.array_ops
def test_empty_like_has_no_dependency_on_its_argument():
    """``jnp.empty_like(x)`` only reads the shape and dtype of x, not its values."""

    def f(x):
        return jnp.empty_like(x).at[:, 0].set(x[:, 1])

    assert_flat_jacobian_sparsity_exact(f, np.arange(1.0, 7.0).reshape(2, 3))


@pytest.mark.array_ops
def test_empty_is_not_treated_as_zeros_in_mul():
    """Multiplying by an empty buffer keeps the dependency on the other operand.

    Before JAX 0.11, ``jnp.empty`` lowered to zeros,
    which let ``mul`` clear the pattern.
    The buffer is now uninitialized, so its values may be nonzero.
    The numerical Jacobian reads whatever the buffer holds (zeros on CPU),
    so the detected pattern can only be checked to cover it.
    """

    def f(x):
        return x * jnp.empty((3, 2))

    x = np.arange(1.0, 7.0).reshape(3, 2)
    result = jacobian_sparsity(f, x).todense().astype(int)
    expected = np.eye(6, dtype=int)  # out[i,j] = x[i,j] * empty[i,j]
    np.testing.assert_array_equal(result, expected)
    assert np.all(result >= numerical_flat_jacobian_sparsity(f, x))


@pytest.mark.array_ops
def test_empty_is_not_treated_as_zeros_in_dot_general():
    """A matmul with an empty matrix keeps every contraction term.

    As for ``mul``, the numerical Jacobian reads the uninitialized buffer,
    so the detected pattern can only be checked to cover it.
    """

    def f(x):
        return jnp.empty((2, 3)) @ x

    x = np.array([1.0, 2.0, 3.0])
    result = jacobian_sparsity(f, x).todense().astype(int)
    expected = np.ones((2, 3), dtype=int)  # out[i] = sum_j empty[i,j] * x[j]
    np.testing.assert_array_equal(result, expected)
    assert_jacobian_sparsity_conservative(f, x)


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

    assert_flat_jacobian_sparsity_exact(f, np.arange(1.0, n + 1.0))


@pytest.mark.array_ops
@pytest.mark.parametrize("shape", [(0,), (0, 3), (2, 0, 4)], ids=["1d", "2d", "3d"])
def test_lax_empty_zero_size(shape):
    """A zero-size ``lax.empty`` does not crash and produces no rows."""
    n = int(np.prod(shape))

    def f(x):
        return lax.empty(shape, jnp.float32) + x[:n].reshape(shape)

    assert_flat_jacobian_sparsity_exact(f, np.arange(1.0, 6.0))


@pytest.mark.array_ops
def test_empty_zero_size():
    """A zero-size empty buffer produces no rows."""

    def f(x):
        return jnp.concatenate([jnp.empty((0,)), x])

    assert_jacobian_sparsity_exact(f, np.array([1.0, 2.0, 3.0]))


@pytest.mark.array_ops
def test_empty_scalar():
    """A scalar empty buffer added to the input keeps the diagonal pattern."""

    def f(x):
        return x + jnp.empty(())

    assert_jacobian_sparsity_exact(f, np.array([1.0, 2.0, 3.0]))
