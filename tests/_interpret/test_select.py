"""Tests for select_n propagation (jnp.where, lax.select).

https://docs.jax.dev/en/latest/_autosummary/jax.lax.select_n.html
"""

import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import hessian_sparsity, jacobian_sparsity


@pytest.mark.control_flow
def test_ifelse_both_branches():
    """Ifelse unions both branches (global sparsity)."""

    def f(x):
        # jnp.where is the JAX equivalent of ifelse
        return jnp.array([jnp.where(x[1] < x[2], x[0] + x[1], x[2] * x[3])])

    result = jacobian_sparsity(f, 4).todense().astype(int)
    expected = np.array([[1, 1, 1, 1]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_ifelse_one_branch_constant():
    """Ifelse with one constant branch."""

    def f(x):
        return jnp.array([jnp.where(x[1] < x[2], x[0] + x[1], 1.0)])

    result = jacobian_sparsity(f, 4).todense().astype(int)
    expected = np.array([[1, 1, 0, 0]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_where_mask():
    """jnp.where with array mask is element-wise.

    Each output depends only on the corresponding input
    since both branches are element-wise and the mask has zero derivative.
    """

    def f(x):
        mask = x > 0
        return jnp.where(mask, x, -x)

    result = jacobian_sparsity(f, 3).todense().astype(int)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_ifelse_one_branch_constant_false():
    """Ifelse with constant true branch only tracks false branch."""

    def f(x):
        return jnp.array([jnp.where(x[1] < x[2], 1.0, x[2] * x[3])])

    result = jacobian_sparsity(f, 4).todense().astype(int)
    expected = np.array([[0, 0, 1, 1]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_select_n_mixed_deps():
    """select_n with a const predicate picks only the selected branch per element.

    pred=[True, False, True] selects a[0], b[1], a[2].
    """

    def f(x):
        a = x[:3]
        b = jnp.array([x[3], x[4], x[3]])
        pred = jnp.array([True, False, True])
        return jnp.where(pred, a, b)

    result = jacobian_sparsity(f, 5).todense().astype(int)
    # out[0] ← a[0] = {0}, out[1] ← b[1] = {4}, out[2] ← a[2] = {2}
    expected = np.array([[1, 0, 0, 0, 0], [0, 0, 0, 0, 1], [0, 0, 1, 0, 0]], dtype=int)
    np.testing.assert_array_equal(result, expected)


# Hessian sparsity for select_n / jnp.where


@pytest.mark.hessian
@pytest.mark.control_flow
def test_hessian_where_both_branches():
    """jnp.where unions Hessian patterns from both branches."""

    def f(x):
        return jnp.where(x[0] > 0, x[0] ** x[1], x[2] * x[3])

    H = hessian_sparsity(f, 4).todense().astype(int)
    expected = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ]
    )
    np.testing.assert_array_equal(H, expected)


@pytest.mark.hessian
@pytest.mark.control_flow
def test_hessian_where_one_constant_true():
    """jnp.where with constant false branch."""

    def f(x):
        return jnp.where(x[0] > 0, x[0] ** x[1], 1.0)

    H = hessian_sparsity(f, 4).todense().astype(int)
    expected = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ]
    )
    np.testing.assert_array_equal(H, expected)


@pytest.mark.hessian
@pytest.mark.control_flow
def test_hessian_where_one_constant_false():
    """jnp.where with constant true branch."""

    def f(x):
        return jnp.where(x[0] > 0, 1.0, x[2] * x[3])

    H = hessian_sparsity(f, 4).todense().astype(int)
    expected = np.array(
        [
            [0, 0, 0, 0],
            [0, 0, 0, 0],
            [0, 0, 0, 1],
            [0, 0, 1, 0],
        ]
    )
    np.testing.assert_array_equal(H, expected)


@pytest.mark.control_flow
def test_select_n_bounds_merge_floor_div():
    """Bounds merge through select_n enables floor division (``//``) bounds propagation.

    ``//`` lowers to a nested jaxpr with ``div``, ``rem``, ``sign``, ``select_n``.
    The ``select_n`` has a dynamic predicate (sign of remainder),
    but both branches have known bounds, so merging produces valid output bounds.
    Without the merge, bounds are lost and the result is fully dense (all 1s).

    argmax(x[:4]) ∈ {0,1,2,3}, so div gives [0,1].
    The floor-div correction branch (q-1) has bounds [-1,0],
    so the merged result is [-1,1], which dynamic_slice clamps to [0,2].
    This is slightly conservative (true range is [0,1])
    but much better than the fully-dense fallback.
    """

    def f(x):
        idx = jnp.argmax(x[:4])  # bounds: [0, 3]
        start = idx // 2  # bounds: [-1, 1] via merged select_n
        return lax.dynamic_slice(x, (start,), (3,))

    result = jacobian_sparsity(f, 5).todense().astype(int)
    # Slightly conservative: start ∈ {0,1,2} instead of true {0,1}.
    expected = np.array(
        [
            [1, 1, 1, 0, 0],
            [0, 1, 1, 1, 0],
            [0, 0, 1, 1, 1],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)
