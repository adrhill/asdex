"""Tests for random primitive handlers.

Random primitives (random_seed, random_unwrap, random_wrap, random_bits)
generate values that don't depend on traced inputs,
so they produce empty dependency sets.

The random_bits primitive takes a key as input but has zero derivative
with respect to the key (random number generation is not differentiable).
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity


@pytest.mark.array_ops
def test_random_prngkey_diagonal():
    """Random noise added to input: diagonal Jacobian.

    The random values don't depend on x, so only the x + noise term contributes.
    """

    def f(x):
        key = jax.random.PRNGKey(0)
        noise = jax.random.normal(key, x.shape)
        return x + 0.1 * noise

    x = np.zeros(5)
    result = jacobian_sparsity(f, x).todense().astype(int)
    expected = np.eye(5, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_random_prngkey_2d():
    """Random noise with 2D input."""

    def f(x):
        key = jax.random.PRNGKey(42)
        noise = jax.random.uniform(key, x.shape)
        return x * noise

    shape = (3, 4)
    n = int(np.prod(shape))
    x = np.zeros(shape)
    result = jacobian_sparsity(f, x).todense().astype(int)
    expected = np.eye(n, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_random_split_and_use():
    """Split a key and use both subkeys."""

    def f(x):
        key = jax.random.PRNGKey(0)
        k1, k2 = jax.random.split(key)
        noise1 = jax.random.normal(k1, x.shape)
        noise2 = jax.random.normal(k2, x.shape)
        return x + 0.1 * noise1 + 0.1 * noise2

    x = np.zeros(4)
    result = jacobian_sparsity(f, x).todense().astype(int)
    expected = np.eye(4, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_random_fold_in():
    """Using fold_in to derive a key."""

    def f(x):
        key = jax.random.PRNGKey(0)
        key = jax.random.fold_in(key, 42)
        noise = jax.random.normal(key, x.shape)
        return x + noise

    x = np.zeros(3)
    result = jacobian_sparsity(f, x).todense().astype(int)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_random_different_distributions():
    """Various random distributions all produce diagonal patterns."""

    def f_uniform(x):
        key = jax.random.PRNGKey(0)
        return x + jax.random.uniform(key, x.shape)

    def f_exponential(x):
        key = jax.random.PRNGKey(0)
        return x + jax.random.exponential(key, x.shape)

    def f_bernoulli(x):
        key = jax.random.PRNGKey(0)
        mask = jax.random.bernoulli(key, 0.5, x.shape)
        return x * mask

    x = np.zeros(4)
    expected = np.eye(4, dtype=int)

    for f in [f_uniform, f_exponential, f_bernoulli]:
        result = jacobian_sparsity(f, x).todense().astype(int)
        np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_random_no_input_dependency():
    """Random output alone has no input dependencies."""

    def f(x):
        key = jax.random.PRNGKey(0)
        # Output doesn't use x at all
        return jax.random.normal(key, (3,))

    x = np.zeros(4)
    result = jacobian_sparsity(f, x)
    # 3 outputs, 4 inputs, all zeros
    assert result.shape == (3, 4)
    assert result.nnz == 0


@pytest.mark.array_ops
def test_random_size_zero():
    """Size-0 random array."""

    def f(x):
        key = jax.random.PRNGKey(0)
        return jax.random.normal(key, (0,))

    x = np.zeros(3)
    result = jacobian_sparsity(f, x)
    assert result.shape == (0, 3)
    assert result.nnz == 0


@pytest.mark.array_ops
def test_random_jacobian_numerical():
    """Verify structural pattern matches numerical Jacobian."""

    def f(x):
        key = jax.random.PRNGKey(0)
        noise = jax.random.normal(key, x.shape)
        return x + 0.1 * noise

    x = jnp.array([1.0, 2.0, 3.0])
    structural = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    numerical = (np.abs(jax.jacobian(f)(x)) > 1e-10).astype(int)
    np.testing.assert_array_equal(structural, numerical)


@pytest.mark.array_ops
def test_random_sparser_than_conservative():
    """Random + input is diagonal, not dense."""

    def f(x):
        key = jax.random.PRNGKey(0)
        noise = jax.random.normal(key, x.shape)
        return x + noise

    n = 5
    x = np.zeros(n)
    result = jacobian_sparsity(f, x)
    # Diagonal has n nonzeros, conservative would have n*n
    assert result.nnz == n
    assert result.nnz < n * n
