"""Tests for multi-dimensional input and output shapes.

Verifies that jacobian_sparsity, hessian_sparsity, jacobian, and
hessian work when the function takes or returns a multi-dimensional
array (e.g. a matrix or an image) rather than a flat vector.
"""

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from asdex import (
    hessian,
    hessian_sparsity,
    jacobian,
    jacobian_sparsity,
)

# Sparsity detection with tuple input_shape


@pytest.mark.array_ops
def test_row_sum_sparsity():
    """Row-sum of a (3, 4) matrix: each output depends on one row."""

    def f(x):
        return x.sum(axis=1)

    result = jacobian_sparsity(f, input_shape=((3, 4),))
    assert result.shape == (3, 12)
    dense = result.todense().astype(int)
    expected = np.zeros((3, 12), dtype=int)
    for i in range(3):
        expected[i, 4 * i : 4 * (i + 1)] = 1
    np.testing.assert_array_equal(dense, expected)


@pytest.mark.array_ops
def test_column_sum_sparsity():
    """Column-sum of a (3, 4) matrix: each output depends on one column."""

    def f(x):
        return x.sum(axis=0)

    result = jacobian_sparsity(f, input_shape=((3, 4),))
    assert result.shape == (4, 12)
    dense = result.todense().astype(int)
    # Output j depends on inputs j, j+4, j+8 (column j in row-major layout)
    expected = np.zeros((4, 12), dtype=int)
    for j in range(4):
        for i in range(3):
            expected[j, 4 * i + j] = 1
    np.testing.assert_array_equal(dense, expected)


@pytest.mark.array_ops
def test_flatten_sparsity():
    """Flattening a (3, 4) matrix: identity Jacobian in flat space."""

    def f(x):
        return x.ravel()

    result = jacobian_sparsity(f, input_shape=((3, 4),))
    assert result.shape == (12, 12)
    dense = result.todense().astype(int)
    np.testing.assert_array_equal(dense, np.eye(12, dtype=int))


@pytest.mark.array_ops
def test_3d_input_sparsity():
    """3D input shape: (2, 3, 4) flattened to n=24."""

    def f(x):
        # Sum over last axis: (2, 3, 4) -> (2, 3) -> flatten to (6,)
        return x.sum(axis=2).ravel()

    result = jacobian_sparsity(f, input_shape=((2, 3, 4),))
    assert result.shape == (6, 24)
    assert result.nnz == 24  # each of 24 inputs appears in exactly one output


@pytest.mark.hessian
def test_hessian_matrix_input_sparsity():
    """Hessian sparsity for a scalar function of a (3, 3) matrix."""

    def f(x):
        return jnp.sum(x**2)

    result = hessian_sparsity(f, input_shape=((3, 3),))
    assert result.shape == (9, 9)
    # Diagonal Hessian: each x_{ij}^2 only couples with itself
    dense = result.todense().astype(int)
    np.testing.assert_array_equal(dense, np.eye(9, dtype=int))


# Sparsity detection with multi-dimensional outputs


@pytest.mark.array_ops
def test_reshape_output_sparsity():
    """1D input, 2D output via reshape: identity in flat space."""

    def f(x):
        return (x**2).reshape(2, 3)

    result = jacobian_sparsity(f, input_shape=(6,))
    assert result.shape == (6, 6)
    dense = result.todense().astype(int)
    np.testing.assert_array_equal(dense, np.eye(6, dtype=int))


@pytest.mark.array_ops
def test_keepdims_output_sparsity():
    """2D input, 2D output via sum with keepdims."""

    def f(x):
        return x.sum(axis=1, keepdims=True)

    result = jacobian_sparsity(f, input_shape=((3, 4),))
    assert result.shape == (3, 12)
    dense = result.todense().astype(int)
    expected = np.zeros((3, 12), dtype=int)
    for i in range(3):
        expected[i, 4 * i : 4 * (i + 1)] = 1
    np.testing.assert_array_equal(dense, expected)


@pytest.mark.array_ops
def test_multidim_input_and_output_sparsity():
    """Both input and output are multi-dimensional: (2, 3) -> (3, 2)."""

    def f(x):
        # Transpose-like via slicing (avoids unsupported transpose primitive)
        rows = [x[i, :] for i in range(2)]
        cols = [jnp.array([rows[0][j], rows[1][j]]) for j in range(3)]
        return jnp.stack(cols)  # (3, 2)

    result = jacobian_sparsity(f, input_shape=((2, 3),))
    assert result.shape == (6, 6)
    dense = result.todense().astype(int)
    # Output is effectively a transpose: out[j, i] = in[i, j]
    # Flat output index j*2+i depends on flat input index i*3+j
    expected = np.zeros((6, 6), dtype=int)
    for i in range(2):
        for j in range(3):
            out_flat = j * 2 + i
            in_flat = i * 3 + j
            expected[out_flat, in_flat] = 1
    np.testing.assert_array_equal(dense, expected)


# Sparse Jacobian computation with multi-dimensional inputs and outputs


@pytest.mark.jacobian
def test_jacobian_matrix_input():
    """Jacobian with a (3, 4) matrix input matches jax.jacobian."""

    def f(x):
        return x.sum(axis=1)

    x = np.arange(12.0).reshape(3, 4)
    result = jacobian(f, input_shape=(x.shape,))(x).todense()
    expected = jax.jacobian(f)(x)
    assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.jacobian
def test_jacobian_elementwise_matrix():
    """Element-wise f(x) = x^2 on a (2, 3) matrix."""

    def f(x):
        return x**2

    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = jacobian(f, input_shape=(x.shape,))(x).todense()
    expected = jax.jacobian(f)(x)
    assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.jacobian
def test_jacobian_2d_output():
    """Jacobian where f returns a (2, 3) matrix."""

    def f(x):
        return (x**2).reshape(2, 3)

    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    result = jacobian(f, input_shape=x.shape)(x).todense()
    expected = jax.jacobian(f)(x)
    assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.jacobian
def test_jacobian_2d_input_and_output():
    """Jacobian where both input and output are 2D."""

    def f(x):
        # (3, 4) -> (3, 1) with keepdims
        return x.sum(axis=1, keepdims=True)

    x = np.arange(12.0).reshape(3, 4)
    result = jacobian(f, input_shape=(x.shape,))(x).todense()
    expected = jax.jacobian(f)(x)
    assert_allclose(result, expected, rtol=1e-5)


@pytest.mark.hessian
def test_hessian_matrix_input():
    """Hessian with a (2, 3) matrix input matches jax.hessian."""

    def f(x):
        return jnp.sum(x**2)

    x = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    result = hessian(f, input_shape=(x.shape,))(x).todense()
    expected = jax.hessian(f)(x)
    assert_allclose(result, expected, rtol=1e-5)


# LeNet-style ConvNet taking a 2D image input


class _TinyLeNet(nn.Module):
    """Minimal LeNet: two conv layers followed by a linear head."""

    @nn.compact
    def __call__(self, x):
        # x: (H, W) single-channel image
        x = x[None, :, :, None]  # (1, H, W, 1) batch + channel dims

        x = nn.Conv(features=4, kernel_size=(3, 3), padding="VALID")(x)
        x = nn.relu(x)

        x = nn.Conv(features=2, kernel_size=(3, 3), padding="VALID")(x)
        x = nn.relu(x)

        return x.ravel()


_lenet = _TinyLeNet()
_lenet_params = _lenet.init(jax.random.key(42), jnp.zeros((8, 8)))


def _lenet_fn(x):
    """Apply tiny LeNet to a single (8, 8) image."""
    return _lenet.apply(_lenet_params, x)


@pytest.mark.jacobian
def test_lenet_sparsity_detection():
    """Sparsity detection on a LeNet with 2D image input."""
    sparsity = jacobian_sparsity(_lenet_fn, input_shape=((8, 8),))

    n = 64  # 8 * 8
    assert sparsity.n == n
    assert sparsity.nnz > 0
    # Two 3x3 VALID convs: receptive field is 5x5,
    # so each output depends on at most 25 inputs
    max_deps = 25
    dense = sparsity.todense()
    assert np.all(dense.sum(axis=1) <= max_deps)


@pytest.mark.slow
@pytest.mark.jacobian
def test_lenet_jacobian_values():
    """Sparse Jacobian of LeNet matches dense jax.jacobian."""
    x = jax.random.normal(jax.random.key(0), (8, 8))

    result = jacobian(_lenet_fn, input_shape=((8, 8),))(np.asarray(x)).todense()
    # Reference: jax.jacobian gives (m, H, W), reshape to (m, n)
    expected = jax.jacobian(_lenet_fn)(x).reshape(result.shape[0], 64)

    assert_allclose(result, np.asarray(expected), rtol=1e-4)
