"""Tests for cond and switch propagation.

https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity


@pytest.mark.control_flow
def test_cond_union_of_branches():
    """Cond unions state_indices from different branches.

    One branch returns x[:2], the other returns x[1:3].
    The union gives each output state_indices from both branches.
    """

    def f(x):
        return jax.lax.cond(
            True,
            lambda operand: operand[:2],
            lambda operand: operand[1:3],
            x,
        )

    result = jacobian_sparsity(f, input_shape=(3,)).todense().astype(int)
    # Branch 0: out[0]←{0}, out[1]←{1}
    # Branch 1: out[0]←{1}, out[1]←{2}
    # Union:    out[0]←{0,1}, out[1]←{1,2}
    expected = np.array(
        [
            [1, 1, 0],
            [0, 1, 1],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_cond_identical_branches():
    """Cond with identical branches returns the same state_indices as either branch."""

    def f(x):
        return jax.lax.cond(
            True,
            lambda operand: operand * 2,
            lambda operand: operand * 3,
            x,
        )

    result = jacobian_sparsity(f, input_shape=(3,)).todense().astype(int)
    # Both branches are elementwise, so the union is still diagonal
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_cond_one_branch_constant():
    """Cond where one branch returns a constant still unions both."""

    def f(x):
        return jax.lax.cond(
            True,
            lambda operand: operand,
            lambda operand: jnp.ones_like(operand),
            x,
        )

    result = jacobian_sparsity(f, input_shape=(3,)).todense().astype(int)
    # Branch 0: identity (diagonal).  Branch 1: constant (zeros).
    # Union is just the diagonal.
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_switch_three_branches():
    """lax.switch with 3 branches unions all of them."""

    def f(x):
        return jax.lax.switch(
            0,
            [
                lambda o: o[:2],
                lambda o: o[1:3],
                lambda o: o[2:4],
            ],
            x,
        )

    result = jacobian_sparsity(f, input_shape=(4,)).todense().astype(int)
    # Union: out[0]←{0,1,2}, out[1]←{1,2,3}
    expected = np.array([[1, 1, 1, 0], [0, 1, 1, 1]], dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_cond_asymmetric_branches():
    """Cond where one branch does a reduction, the other is elementwise."""

    def f(x):
        return jax.lax.cond(
            True,
            lambda o: o * 2,
            lambda o: jnp.full_like(o, jnp.sum(o)),
            x,
        )

    result = jacobian_sparsity(f, input_shape=(3,)).todense().astype(int)
    # Union: elementwise (diagonal) ∪ all-to-all (dense) = dense
    expected = np.ones((3, 3), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_cond_closure_captured_index():
    """Cond branch with closure-captured index array resolves gather precisely.

    Without seeding state_consts from the ClosedJaxpr's captured constants,
    the gather in branch_a would be conservative (dense).
    """
    indices = jnp.array([2, 0, 1])

    def f(x):
        def branch_a(o):
            return o[indices]  # permutation via closure-captured index

        def branch_b(o):
            return o  # identity

        return jax.lax.cond(True, branch_a, branch_b, x)

    result = jacobian_sparsity(f, input_shape=(3,)).todense().astype(int)
    # branch_a: out[0]←{2}, out[1]←{0}, out[2]←{1}
    # branch_b: out[0]←{0}, out[1]←{1}, out[2]←{2}
    # Union:    out[0]←{0,2}, out[1]←{0,1}, out[2]←{1,2}
    expected = np.array(
        [
            [1, 0, 1],
            [1, 1, 0],
            [0, 1, 1],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


# Size-0 dimension


@pytest.mark.control_flow
def test_cond_zero_size_output():
    """Cond returning a zero-sized array produces an empty Jacobian."""

    def f(x):
        return jax.lax.cond(True, lambda y: y[:0], lambda y: y[:0], x)

    result = jacobian_sparsity(f, input_shape=(3,))
    assert result.shape == (0, 3)
    assert result.nnz == 0
