"""Tests for pad primitive handler."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import hessian_sparsity, jacobian_sparsity


@pytest.mark.array_ops
def test_pad_1d_constant():
    """1D padding with constant values has no dependencies on padded positions."""

    def f(x):
        return jnp.pad(x, (1, 1), constant_values=0)

    result = jacobian_sparsity(f, 2).todense().astype(int)
    expected = np.array([[0, 0], [1, 0], [0, 1], [0, 0]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_pad_1d_asymmetric():
    """Asymmetric padding: different amounts on each side."""

    def f(x):
        return jnp.pad(x, (2, 1), constant_values=0)

    result = jacobian_sparsity(f, 3).todense().astype(int)
    # 2 pad + 3 original + 1 pad = 6 output elements
    expected = np.array(
        [
            [0, 0, 0],  # pad
            [0, 0, 0],  # pad
            [1, 0, 0],  # x[0]
            [0, 1, 0],  # x[1]
            [0, 0, 1],  # x[2]
            [0, 0, 0],  # pad
        ]
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_pad_2d():
    """2D non-square padding preserves per-element structure.

    Uses non-square shape (2,3) to catch dimension-ordering bugs.
    """

    def f(x):
        mat = x.reshape(2, 3)
        return jnp.pad(mat, ((1, 0), (0, 1)), constant_values=0).flatten()

    result = jacobian_sparsity(f, 6).todense().astype(int)
    # Input: [[x0, x1, x2], [x3, x4, x5]]
    # Padded: [[0, 0, 0, 0], [x0, x1, x2, 0], [x3, x4, x5, 0]]  (3x4 = 12 outputs)
    expected = np.array(
        [
            [0, 0, 0, 0, 0, 0],  # pad (0,0)
            [0, 0, 0, 0, 0, 0],  # pad (0,1)
            [0, 0, 0, 0, 0, 0],  # pad (0,2)
            [0, 0, 0, 0, 0, 0],  # pad (0,3)
            [1, 0, 0, 0, 0, 0],  # x0  (1,0)
            [0, 1, 0, 0, 0, 0],  # x1  (1,1)
            [0, 0, 1, 0, 0, 0],  # x2  (1,2)
            [0, 0, 0, 0, 0, 0],  # pad (1,3)
            [0, 0, 0, 1, 0, 0],  # x3  (2,0)
            [0, 0, 0, 0, 1, 0],  # x4  (2,1)
            [0, 0, 0, 0, 0, 1],  # x5  (2,2)
            [0, 0, 0, 0, 0, 0],  # pad (2,3)
        ]
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_pad_negative():
    """Negative padding trims the array (used by JAX's grad)."""

    def f(x):
        # Negative padding = trimming: removes first and last element
        return jax.lax.pad(x, 0.0, [(-1, -1, 0)])

    result = jacobian_sparsity(f, 4).todense().astype(int)
    # Removes first and last: output = [x1, x2]
    expected = np.array(
        [
            [0, 1, 0, 0],
            [0, 0, 1, 0],
        ]
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_pad_interior():
    """Interior padding inserts elements between each pair of original elements."""

    def f(x):
        return jax.lax.pad(x, 0.0, [(0, 0, 1)])

    result = jacobian_sparsity(f, 3).todense().astype(int)
    # x = [a, b, c] -> [a, 0, b, 0, c] (5 elements)
    expected = np.array(
        [
            [1, 0, 0],  # a
            [0, 0, 0],  # interior pad
            [0, 1, 0],  # b
            [0, 0, 0],  # interior pad
            [0, 0, 1],  # c
        ]
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_pad_interior_with_border():
    """Interior and border padding combined."""

    def f(x):
        return jax.lax.pad(x, 0.0, [(1, 1, 2)])

    result = jacobian_sparsity(f, 2).todense().astype(int)
    # x = [a, b] -> [0, a, 0, 0, b, 0] (6 elements)
    # low=1, then a, then 2 interior pads, then b, then high=1
    expected = np.array(
        [
            [0, 0],  # low pad
            [1, 0],  # a
            [0, 0],  # interior
            [0, 0],  # interior
            [0, 1],  # b
            [0, 0],  # high pad
        ]
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_pad_noop():
    """Zero padding on all sides is an identity operation."""

    def f(x):
        return jax.lax.pad(x, 0.0, [(0, 0, 0)])

    result = jacobian_sparsity(f, 3).todense().astype(int)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_pad_negative_with_interior():
    """Negative border padding combined with interior padding.

    Interior padding dilates first,
    then negative low/high trims from edges.
    """

    def f(x):
        # Dilate [a, b, c] -> [a, 0, b, 0, c], then trim first element
        return jax.lax.pad(x, 0.0, [(-1, 0, 1)])

    result = jacobian_sparsity(f, 3).todense().astype(int)
    # Dilated: [a, 0, b, 0, c] (5 elements), trim first -> [0, b, 0, c] (4 elements)
    expected = np.array(
        [
            [0, 0, 0],  # interior pad (was between a and b)
            [0, 1, 0],  # b
            [0, 0, 0],  # interior pad
            [0, 0, 1],  # c
        ]
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_pad_value_with_deps():
    """Padding value that depends on inputs propagates to padding positions."""

    def f(x):
        # Pad with x[0] as the padding value — padding positions depend on x[0]
        return jax.lax.pad(x, x[0], [(1, 1, 0)])

    result = jacobian_sparsity(f, 3).todense().astype(int)
    # Output: [x[0], x[0], x[1], x[2], x[0]]
    # Padding positions inherit x[0]'s dependency
    expected = np.array(
        [
            [1, 0, 0],  # pad = x[0]
            [1, 0, 0],  # x[0]
            [0, 1, 0],  # x[1]
            [0, 0, 1],  # x[2]
            [1, 0, 0],  # pad = x[0]
        ]
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_pad_2d_interior():
    """2D non-square interior padding dilates both dimensions.

    Uses non-square shape (2,3) to catch dimension-ordering bugs.
    """

    def f(x):
        mat = x.reshape(2, 3)
        # Interior padding of 1 in both dims
        return jax.lax.pad(mat, 0.0, [(0, 0, 1), (0, 0, 1)]).flatten()

    result = jacobian_sparsity(f, 6).todense().astype(int)
    # Input: [[a, b, c], [d, e, f]]
    # Dilated (3x5):
    # [[a, 0, b, 0, c],
    #  [0, 0, 0, 0, 0],
    #  [d, 0, e, 0, f]]
    expected = np.array(
        [
            [1, 0, 0, 0, 0, 0],  # a
            [0, 0, 0, 0, 0, 0],  # interior
            [0, 1, 0, 0, 0, 0],  # b
            [0, 0, 0, 0, 0, 0],  # interior
            [0, 0, 1, 0, 0, 0],  # c
            [0, 0, 0, 0, 0, 0],  # interior row
            [0, 0, 0, 0, 0, 0],  # interior row
            [0, 0, 0, 0, 0, 0],  # interior row
            [0, 0, 0, 0, 0, 0],  # interior row
            [0, 0, 0, 0, 0, 0],  # interior row
            [0, 0, 0, 1, 0, 0],  # d
            [0, 0, 0, 0, 0, 0],  # interior
            [0, 0, 0, 0, 1, 0],  # e
            [0, 0, 0, 0, 0, 0],  # interior
            [0, 0, 0, 0, 0, 1],  # f
        ]
    )
    np.testing.assert_array_equal(result, expected)


# Precision verification (compare against jax.jacobian)


@pytest.mark.array_ops
def test_pad_1d_precision():
    """1D pad with interior+border matches actual Jacobian exactly."""

    def f(x):
        return jax.lax.pad(x, 0.0, [(1, 2, 1)])

    input_size = 5
    detected = jacobian_sparsity(f, input_size).todense().astype(int)

    x_test = jax.random.normal(jax.random.key(0), (input_size,))
    actual_jac = jax.jacobian(f)(x_test)
    actual_nonzero = (np.abs(actual_jac) > 1e-10).astype(int)

    np.testing.assert_array_equal(
        detected, actual_nonzero, "Detected sparsity should match actual Jacobian"
    )


@pytest.mark.array_ops
def test_pad_2d_nonsquare_precision():
    """2D non-square pad with asymmetric config matches actual Jacobian exactly."""

    def f(x):
        mat = x.reshape(2, 3)
        return jax.lax.pad(mat, 0.0, [(1, 0, 0), (0, 2, 1)]).flatten()

    input_size = 6
    detected = jacobian_sparsity(f, input_size).todense().astype(int)

    x_test = jax.random.normal(jax.random.key(1), (input_size,))
    actual_jac = jax.jacobian(f)(x_test)
    actual_nonzero = (np.abs(actual_jac) > 1e-10).astype(int)

    np.testing.assert_array_equal(
        detected, actual_nonzero, "Detected sparsity should match actual Jacobian"
    )


@pytest.mark.array_ops
def test_pad_3d_precision():
    """3D pad with mixed config matches actual Jacobian exactly."""

    def f(x):
        arr = x.reshape(2, 3, 2)
        return jax.lax.pad(arr, 0.0, [(0, 1, 0), (1, 0, 1), (0, 0, 0)]).flatten()

    input_size = 12
    detected = jacobian_sparsity(f, input_size).todense().astype(int)

    x_test = jax.random.normal(jax.random.key(2), (input_size,))
    actual_jac = jax.jacobian(f)(x_test)
    actual_nonzero = (np.abs(actual_jac) > 1e-10).astype(int)

    np.testing.assert_array_equal(
        detected, actual_nonzero, "Detected sparsity should match actual Jacobian"
    )


# Conservative audit


@pytest.mark.array_ops
def test_pad_conservative_audit():
    """Pad sparsity is strictly sparser than dense.

    The detected pattern must have fewer nonzeros than n_outputs * n_inputs,
    confirming the handler is not falling back to a conservative all-ones result.
    """

    def f(x):
        mat = x.reshape(2, 3)
        return jnp.pad(mat, ((1, 1), (1, 1)), constant_values=0).flatten()

    n_inputs = 6
    result = jacobian_sparsity(f, n_inputs).todense().astype(int)
    n_outputs = result.shape[0]

    assert result.sum() < n_outputs * n_inputs


# 3D + mixed interior/border


@pytest.mark.array_ops
def test_pad_3d_mixed():
    """3D pad with mixed border and interior config.

    Exercises multi-dimensional coordinate mapping
    where each axis has a different padding configuration.
    Uses all-unique dimension sizes (2, 3, 4) to catch transposition bugs.
    """

    def f(x):
        arr = x.reshape(2, 3, 4)
        # dim 0: border only (low=1, high=0, interior=0)
        # dim 1: interior only (low=0, high=0, interior=1)
        # dim 2: border + interior (low=0, high=1, interior=1)
        return jax.lax.pad(arr, 0.0, [(1, 0, 0), (0, 0, 1), (0, 1, 1)]).flatten()

    input_size = 24
    detected = jacobian_sparsity(f, input_size).todense().astype(int)

    # Verify against actual Jacobian
    x_test = jax.random.normal(jax.random.key(3), (input_size,))
    actual_jac = jax.jacobian(f)(x_test)
    actual_nonzero = (np.abs(actual_jac) > 1e-10).astype(int)

    np.testing.assert_array_equal(
        detected, actual_nonzero, "Detected sparsity should match actual Jacobian"
    )


# Degenerate shapes


@pytest.mark.array_ops
def test_pad_size_one_dim():
    """Pad a (1,3) array where one dimension has size 1.

    Size-1 dimensions are an edge case for interior padding
    since there are no gaps between elements to fill.
    """

    def f(x):
        mat = x.reshape(1, 3)
        # Interior padding on the size-1 dim has no effect.
        return jax.lax.pad(mat, 0.0, [(1, 1, 1), (0, 0, 1)]).flatten()

    input_size = 3
    detected = jacobian_sparsity(f, input_size).todense().astype(int)

    x_test = jax.random.normal(jax.random.key(4), (input_size,))
    actual_jac = jax.jacobian(f)(x_test)
    actual_nonzero = (np.abs(actual_jac) > 1e-10).astype(int)

    np.testing.assert_array_equal(
        detected, actual_nonzero, "Detected sparsity should match actual Jacobian"
    )


@pytest.mark.hessian
def test_pad_in_hessian():
    """Pad enables precise tridiagonal Hessian for finite-difference stencils.

    This is the motivating use case:
    JAX's grad of sliced operations emits pad primitives.
    """

    def f(x):
        return ((x[1:] - x[:-1]) ** 2).sum()

    H = hessian_sparsity(f, 4).todense().astype(int)
    expected = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 1, 0],
            [0, 1, 1, 1],
            [0, 0, 1, 1],
        ]
    )
    np.testing.assert_array_equal(H, expected)


# Size-0 dimension


@pytest.mark.array_ops
def test_pad_zero_size_input():
    """Padding a zero-sized array produces output from padding value only."""

    def f(x):
        return jnp.pad(x[:0], (1, 1))

    result = jacobian_sparsity(f, 3)
    # Output has 2 elements (low pad + high pad), none depend on input.
    assert result.shape == (2, 3)
    assert result.nnz == 0
