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


def assert_jacobian_sparsity_exact(f, x, input_proto=None, holomorphic=False):
    """Assert detected sparsity matches numerical Jacobian exactly.

    Args:
        f: Function to test.
        x: Input values for numerical Jacobian computation.
        input_proto: Prototype array for shape/dtype (default: zeros like x).
        holomorphic: Whether f is holomorphic (complex->complex).
    """
    if input_proto is None:
        input_proto = np.zeros_like(x)
    detected = jacobian_sparsity(f, input_proto).todense().astype(int)
    expected = numerical_jacobian_sparsity(f, x, holomorphic=holomorphic)
    np.testing.assert_array_equal(detected, expected)


def assert_jacobian_sparsity_conservative(f, x, input_proto=None, nnz_per_row=None):
    """Assert detected sparsity is a superset of numerical Jacobian.

    Args:
        f: Function to test.
        x: Input values for numerical Jacobian computation.
        input_proto: Prototype array for shape/dtype (default: zeros like x).
        nnz_per_row: Expected number of nonzeros per row (optional).
    """
    if input_proto is None:
        input_proto = np.zeros_like(x)
    detected = jacobian_sparsity(f, input_proto).todense().astype(int)
    numerical = numerical_jacobian_sparsity(f, x)
    assert np.all(detected >= numerical), "Detected pattern must cover numerical"
    if nnz_per_row is not None:
        np.testing.assert_array_equal(detected.sum(axis=1), nnz_per_row)
