"""Shared test utilities."""

import jax
import numpy as np

from asdex import jacobian_sparsity


def numerical_jacobian_sparsity(f, x, atol=1e-10, holomorphic=False):
    """Compute sparsity pattern from numerical Jacobian.

    Returns a binary int array: 1 where J[i,j] is not close to zero, else 0.
    """
    J = jax.jacobian(f, holomorphic=holomorphic)(x)
    return (~np.isclose(np.asarray(J), 0, atol=atol)).astype(int)


def assert_jacobian_sparsity_exact(f, x, holomorphic=False):
    """Assert detected sparsity matches numerical Jacobian exactly."""
    detected = jacobian_sparsity(f, x).todense().astype(int)
    expected = numerical_jacobian_sparsity(f, x, holomorphic=holomorphic)
    np.testing.assert_array_equal(detected, expected)


def assert_jacobian_sparsity_conservative(f, x):
    """Assert detected sparsity is a superset of numerical Jacobian.

    When calling this function, also verify the detected pattern against
    a manually defined ``expected`` matrix using ``np.testing.assert_array_equal``.
    This ensures both correctness (covers numerical) and precision (matches intent).
    """
    detected = jacobian_sparsity(f, x).todense().astype(int)
    numerical = numerical_jacobian_sparsity(f, x)
    assert np.all(detected >= numerical), "Detected pattern must cover numerical"
