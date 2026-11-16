"""Shared test utilities.

The assertion helpers take ``fallbacks``,
the expected names of primitives without a precise handler,
and check them unless it is ``None``.
They return the detected pattern,
so that tests can check its other fields without detecting it again.
"""

import warnings

import jax
import numpy as np

from asdex import DensePatternWarning, hessian_sparsity, jacobian_sparsity


def numerical_jacobian_sparsity(
    f, x, atol=1e-10, holomorphic=False, reference=jax.jacobian
):
    """Compute sparsity pattern from numerical Jacobian.

    Returns a binary int array: 1 where J[i,j] is not close to zero, else 0.
    ``reference`` computes the Jacobian.
    Pass ``jax.jacfwd`` for functions that JAX can only differentiate in forward mode.
    """
    J = reference(f, holomorphic=holomorphic)(x)
    return (~np.isclose(J, 0, atol=atol)).astype(int)


def numerical_flat_jacobian_sparsity(f, x, atol=1e-10, reference=jax.jacobian):
    """Compute sparsity pattern from numerical Jacobian, flattened to 2D.

    Like ``numerical_jacobian_sparsity``,
    but reshapes the Jacobian to ``(output size, input size)``,
    matching the patterns asdex detects for inputs and outputs of any shape.
    """
    J = numerical_jacobian_sparsity(f, x, atol=atol, reference=reference)
    return J.reshape(jax.eval_shape(f, x).size, np.size(x))


def _assert_fallbacks(sparsity, fallbacks):
    if fallbacks is not None:
        assert sparsity._fallbacks == tuple(fallbacks)


def assert_jacobian_sparsity_exact(
    f, x, holomorphic=False, reference=jax.jacobian, fallbacks=None
):
    """Assert detected sparsity matches numerical Jacobian exactly."""
    sparsity = jacobian_sparsity(f, x)
    expected = numerical_jacobian_sparsity(
        f, x, holomorphic=holomorphic, reference=reference
    )
    np.testing.assert_array_equal(sparsity.todense().astype(int), expected)
    _assert_fallbacks(sparsity, fallbacks)
    return sparsity


def assert_flat_jacobian_sparsity_exact(f, x, reference=jax.jacobian, fallbacks=None):
    """Assert detected sparsity matches the flattened numerical Jacobian exactly.

    Use this instead of ``assert_jacobian_sparsity_exact``
    when the input or output is not 1D.
    """
    sparsity = jacobian_sparsity(f, x)
    expected = numerical_flat_jacobian_sparsity(f, x, reference=reference)
    np.testing.assert_array_equal(sparsity.todense().astype(int), expected)
    _assert_fallbacks(sparsity, fallbacks)
    return sparsity


def assert_jacobian_sparsity_conservative(f, x, fallbacks=None):
    """Assert detected sparsity is a superset of numerical Jacobian.

    When calling this function, also verify the detected pattern against
    a manually defined ``expected`` matrix using ``np.testing.assert_array_equal``.
    This ensures both correctness (covers numerical) and precision (matches intent).
    Conservative patterns are often fully dense,
    so ``DensePatternWarning`` is ignored.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DensePatternWarning)
        sparsity = jacobian_sparsity(f, x)
    numerical = numerical_jacobian_sparsity(f, x)
    detected = sparsity.todense().astype(int)
    assert np.all(detected >= numerical), "Detected pattern must cover numerical"
    _assert_fallbacks(sparsity, fallbacks)
    return sparsity


def assert_hessian_sparsity_exact(f, x, fallbacks=None):
    """Assert detected Hessian sparsity matches the numerical Hessian exactly."""
    sparsity = hessian_sparsity(f, x)
    expected = (~np.isclose(jax.hessian(f)(x), 0, atol=1e-10)).astype(int)
    np.testing.assert_array_equal(sparsity.todense().astype(int), expected)
    _assert_fallbacks(sparsity, fallbacks)
    return sparsity
