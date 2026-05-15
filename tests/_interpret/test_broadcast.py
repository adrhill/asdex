"""Tests for broadcast_in_dim propagation.

https://docs.jax.dev/en/latest/_autosummary/jax.lax.broadcast_in_dim.html
"""

import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity


def _broadcast_jacobian(
    in_shape: tuple[int, ...],
    out_shape: tuple[int, ...],
    broadcast_dims: tuple[int, ...],
) -> np.ndarray:
    """Build expected Jacobian for broadcast_in_dim.

    Each output element depends on exactly one input element,
    determined by projecting output coordinates onto input dimensions.
    """
    n_in = int(np.prod(in_shape)) if in_shape else 1
    n_out = int(np.prod(out_shape))

    expected = np.zeros((n_out, n_in), dtype=int)
    for out_flat in range(n_out):
        out_coord = np.unravel_index(out_flat, out_shape)
        in_coord = (
            tuple(
                min(out_coord[broadcast_dims[i]], in_shape[i] - 1)
                for i in range(len(in_shape))
            )
            if in_shape
            else ()
        )
        in_flat = np.ravel_multi_index(in_coord, in_shape) if in_shape else 0
        expected[out_flat, in_flat] = 1
    return expected


_SHAPES_AND_BROADCAST = [
    # Scalar input (empty broadcast_dimensions)
    pytest.param((), (3,), (), id="scalar_to_1d"),
    pytest.param((), (2, 3), (), id="scalar_to_2d"),
    pytest.param((), (2, 3, 4), (), id="scalar_to_3d"),
    # 1D input
    pytest.param((3,), (3,), (0,), id="1d_identity"),
    pytest.param((3,), (2, 3), (1,), id="1d_to_2d_axis1"),
    pytest.param((3,), (3, 2), (0,), id="1d_to_2d_axis0"),
    pytest.param((4,), (2, 3, 4), (2,), id="1d_to_3d"),
    # 2D input
    pytest.param((3, 4), (3, 4), (0, 1), id="2d_identity"),
    pytest.param((3, 4), (2, 3, 4), (1, 2), id="2d_to_3d"),
    pytest.param((3, 1), (3, 4), (0, 1), id="2d_size_one_broadcast"),
    pytest.param((1, 4), (3, 4), (0, 1), id="2d_size_one_first"),
    # Size-1 dimensions that broadcast
    pytest.param((1,), (5,), (0,), id="1d_size_one"),
    pytest.param((1,), (3, 4), (0,), id="1d_size_one_to_2d_axis0"),
    pytest.param((1,), (3, 4), (1,), id="1d_size_one_to_2d_axis1"),
    pytest.param((1, 1), (3, 4), (0, 1), id="2d_all_ones"),
    # 3D/4D input
    pytest.param((2, 3, 4), (2, 3, 4), (0, 1, 2), id="3d_identity"),
    pytest.param((2, 3, 4), (5, 2, 3, 4), (1, 2, 3), id="3d_to_4d"),
    pytest.param((2, 3, 2, 4), (2, 3, 2, 4), (0, 1, 2, 3), id="4d_identity"),
]


# Core broadcast tests
@pytest.mark.array_ops
@pytest.mark.parametrize(
    ("in_shape", "out_shape", "broadcast_dims"), _SHAPES_AND_BROADCAST
)
def test_broadcast_in_dim(in_shape, out_shape, broadcast_dims):
    """Each output depends on exactly one input (projected from broadcast dims)."""
    n_in = int(np.prod(in_shape)) if in_shape else 1

    def f(x):
        arr = x.reshape(in_shape) if in_shape else x[0]
        return lax.broadcast_in_dim(arr, out_shape, broadcast_dims).flatten()

    result = jacobian_sparsity(f, np.zeros(n_in)).todense().astype(int)
    expected = _broadcast_jacobian(in_shape, out_shape, broadcast_dims)
    np.testing.assert_array_equal(result, expected)


# Compositions
@pytest.mark.array_ops
def test_scalar_broadcast():
    """Broadcasting a scalar preserves per-element structure."""

    def f(x):
        # Each element broadcast independently
        return jnp.array([jnp.broadcast_to(x[0], (2,)).sum(), x[1] * 2])

    result = jacobian_sparsity(f, np.zeros(2)).todense().astype(int)
    expected = np.array([[1, 0], [0, 1]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_broadcast_constant():
    """Broadcasting a constant array produces zero sparsity."""

    def f(_x):
        const = jnp.array([1.0, 2.0])  # Shape (2,)
        return jnp.broadcast_to(const, (3, 2)).flatten()  # Shape (6,)

    result = jacobian_sparsity(f, np.zeros(2)).todense().astype(int)
    expected = np.zeros((6, 2), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_broadcast_input_add_constant():
    """Broadcasting input and adding a constant preserves input structure."""

    def f(x):
        const = jnp.array([[1.0], [2.0]])  # Shape (2, 1)
        x_col = x.reshape(2, 1)  # Shape (2, 1)
        broadcasted = jnp.broadcast_to(x_col, (2, 3))  # Shape (2, 3)
        return (broadcasted + const).flatten()

    result = jacobian_sparsity(f, np.zeros(2)).todense().astype(int)
    # Each row of output depends on corresponding input element
    # Output shape (2, 3) flattened: rows 0-2 from x[0], rows 3-5 from x[1]
    expected = np.array(
        [
            [1, 0],  # out[0] <- x[0]
            [1, 0],  # out[1] <- x[0]
            [1, 0],  # out[2] <- x[0]
            [0, 1],  # out[3] <- x[1]
            [0, 1],  # out[4] <- x[1]
            [0, 1],  # out[5] <- x[1]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


# Size-0 dimension


@pytest.mark.array_ops
def test_broadcast_size_zero_dim():
    """Broadcasting a zero-sized array produces an empty Jacobian.

    Zero-sized inputs have no elements, so output index sets should be empty.
    """

    def f(x):
        return jnp.broadcast_to(x[:0].reshape(0, 1), (0, 3)).flatten()

    result = jacobian_sparsity(f, np.zeros(3))
    assert result.shape == (0, 3)
    assert result.nnz == 0


@pytest.mark.array_ops
def test_broadcast_expand_dims_zero():
    """expand_dims on a zero-sized array produces an empty Jacobian.

    Reproducer from GitHub issue #86.
    """

    def f(x):
        return jnp.expand_dims(x[:0], axis=1).flatten()

    result = jacobian_sparsity(f, np.zeros(3))
    assert result.shape == (0, 3)
    assert result.nnz == 0
