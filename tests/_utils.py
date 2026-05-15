"""Shared test utilities."""

import jax
import numpy as np


def numerical_jacobian_sparsity(f, x, atol=1e-10):
    """Compute sparsity pattern from numerical Jacobian.

    Returns a binary int array: 1 where J[i,j] is not close to zero, else 0.
    """
    J = jax.jacobian(f)(x)
    return (~np.isclose(np.asarray(J), 0, atol=atol)).astype(int)
