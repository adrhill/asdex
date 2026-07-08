"""Tests for scan propagation.

https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian, jacobian_sparsity
from tests._utils import assert_jacobian_sparsity_exact


@pytest.mark.control_flow
def test_scan_cumulative_sum():
    """Cumulative sum: scalar carry accumulates over 1D xs.

    ys[t] = carry[t] = xs[0] + ... + xs[t] (lower-triangular).
    """

    def f(x):
        def body(carry, xi):
            new_carry = carry + xi
            return new_carry, new_carry

        _, ys = jax.lax.scan(body, 0.0, x)
        return ys

    result = jacobian_sparsity(f, np.zeros(4)).todense().astype(int)
    expected = np.array(
        [
            [1, 0, 0, 0],  # ys[0] = x[0]
            [1, 1, 0, 0],  # ys[1] = x[0] + x[1]
            [1, 1, 1, 0],  # ys[2] = x[0] + x[1] + x[2]
            [1, 1, 1, 1],  # ys[3] = x[0] + x[1] + x[2] + x[3]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_scan_2d_carry_and_xs():
    """2D carry and xs: elementwise accumulation preserves structure.

    Progressive block-diagonal: ys[t] depends on xs[0..t] per element.
    """

    def f(x):
        x_2d = x.reshape(3, 2)

        def body(carry, xi):
            new_carry = carry + xi
            return new_carry, new_carry

        _, ys = jax.lax.scan(body, jnp.zeros(2), x_2d)
        return ys.ravel()

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    expected = np.array(
        [
            [1, 0, 0, 0, 0, 0],  # ys[0] = xs[0]
            [0, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 0, 0],  # ys[1] = xs[0] + xs[1]
            [0, 1, 0, 1, 0, 0],
            [1, 0, 1, 0, 1, 0],  # ys[2] = xs[0] + xs[1] + xs[2]
            [0, 1, 0, 1, 0, 1],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_scan_multiple_carries():
    """Multiple carry variables: (array, scalar) carry.

    The array carry accumulates elementwise,
    the scalar carry counts iterations.
    Only the array output is returned.
    """

    def f(x):
        def body(carry, xi):
            arr, count = carry
            new_arr = arr + xi
            return (new_arr, count + 1.0), new_arr

        (arr_out, _), _ = jax.lax.scan(body, (jnp.zeros(3), 0.0), x.reshape(2, 3))
        return arr_out

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    # arr_out depends on all xs slices elementwise
    expected = np.array(
        [
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_scan_carry_only():
    """Carry-only scan with no xs (length from params).

    The body only transforms the carry without scanning over inputs.
    Identity body preserves diagonal pattern.
    """

    def f(x):
        def body(carry, _):
            return carry, None

        carry_out, _ = jax.lax.scan(body, x, None, length=5)
        return carry_out

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_scan_with_closure_const():
    """Scan body captures an external array via closure.

    Elementwise multiply with a constant preserves diagonal pattern.
    """
    weights = jnp.array([2.0, 3.0])

    def f(x):
        def body(carry, xi):
            new_carry = carry * weights
            return new_carry, new_carry

        _, ys = jax.lax.scan(body, x, None, length=3)
        return ys.ravel()

    result = jacobian_sparsity(f, np.zeros(2)).todense().astype(int)
    # Each output element depends only on the corresponding input element
    expected = np.array(
        [
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_scan_reverse():
    """reverse=True scans xs in reverse order.

    ys[t] depends on xs[t..n-1] when scanning in reverse,
    giving an upper-triangular pattern.
    """

    def f(x):
        def body(carry, xi):
            return carry + xi, carry + xi

        _, ys = jax.lax.scan(body, 0.0, x, reverse=True)
        return ys

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.array(
        [
            [1, 1, 1],  # ys[0] = x[0] + x[1] + x[2]
            [0, 1, 1],  # ys[1] = x[1] + x[2]
            [0, 0, 1],  # ys[2] = x[2]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_scan_reverse_multi_carry():
    """reverse=True with multi-element carry.

    Elementwise accumulation in reverse still produces the same
    overapproximate sparsity pattern.
    """

    def f(x):
        x_2d = x.reshape(2, 3)

        def body(carry, xi):
            return carry + xi, carry

        carry_out, _ = jax.lax.scan(body, jnp.zeros(3), x_2d, reverse=True)
        return carry_out

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    expected = np.array(
        [
            [1, 0, 0, 1, 0, 0],
            [0, 1, 0, 0, 1, 0],
            [0, 0, 1, 0, 0, 1],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_scan_identity_body():
    """Identity body: carry passes through unchanged, ys copies carry.

    Converges in one iteration. Output depends only on carry_init.
    """

    def f(x):
        def body(carry, xi):
            return carry, carry

        _, ys = jax.lax.scan(body, x, None, length=4)
        return ys.ravel()

    result = jacobian_sparsity(f, np.zeros(2)).todense().astype(int)
    expected = np.array(
        [
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1],
            [1, 0],
            [0, 1],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_scan_composition():
    """Nested scan: outer scan calls inner scan in its body.

    Inner scan accumulates elementwise,
    outer scan accumulates the inner result.
    ys[t] depends on xs[0..t] progressively.
    """

    def f(x):
        x_4d = x.reshape(2, 2, 2)

        def inner_body(carry, xi):
            return carry + xi, None

        def outer_body(carry, xi):
            inner_carry, _ = jax.lax.scan(inner_body, carry, xi)
            return inner_carry, inner_carry

        _, ys = jax.lax.scan(outer_body, jnp.zeros(2), x_4d)
        return ys.ravel()

    result = jacobian_sparsity(f, np.zeros(8)).todense().astype(int)
    expected = np.array(
        [
            [1, 0, 1, 0, 0, 0, 0, 0],  # ys[0] = inner_scan(zeros, xs[0:4])
            [0, 1, 0, 1, 0, 0, 0, 0],
            [1, 0, 1, 0, 1, 0, 1, 0],  # ys[1] = inner_scan(ys[0], xs[4:8])
            [0, 1, 0, 1, 0, 1, 0, 1],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_scan_noncontiguous_input():
    """Scan with non-contiguous input dependencies.

    Only odd-indexed inputs are used via slicing before scan.
    """

    def f(x):
        xs = x[1::2]  # select odd indices: x[1], x[3], x[5]

        def body(carry, xi):
            return carry + xi, carry + xi

        _, ys = jax.lax.scan(body, 0.0, xs)
        return ys

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    expected = np.array(
        [
            [0, 1, 0, 0, 0, 0],  # ys[0] = x[1]
            [0, 1, 0, 1, 0, 0],  # ys[1] = x[1] + x[3]
            [0, 1, 0, 1, 0, 1],  # ys[2] = x[1] + x[3] + x[5]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_scan_vs_jax_jacobian():
    """Verify scan sparsity against dense jax.jacobian for correctness.

    The detected sparsity pattern must be a superset of the true nonzero pattern.
    """

    def f(x):
        def body(carry, xi):
            return carry + xi**2, carry * xi

        carry_out, ys = jax.lax.scan(body, jnp.zeros(2), x.reshape(3, 2))
        return jnp.concatenate([carry_out, ys.ravel()])

    x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    detected = jacobian_sparsity(f, np.zeros(6)).todense().astype(bool)
    dense_jac = jax.jacobian(f)(x)
    true_nonzero = np.abs(np.array(dense_jac)) > 0

    # Detected pattern must cover all true nonzeros
    assert np.all(detected | ~true_nonzero), "Detected sparsity misses true nonzeros"


@pytest.mark.control_flow
@pytest.mark.filterwarnings("ignore::asdex.DenseColoringWarning")
def test_scan_jacobian_values():
    """Verify sparse Jacobian values match dense jax.jacobian."""

    def f(x):
        def body(carry, xi):
            return carry + xi, carry * 2.0

        carry_out, ys = jax.lax.scan(body, 0.0, x)
        return jnp.concatenate([jnp.array([carry_out]), ys])

    x = jnp.array([1.0, 2.0, 3.0])
    sparse_jac = jacobian(f, x)(x).todense()
    dense_jac = np.array(jax.jacobian(f)(x))
    np.testing.assert_allclose(sparse_jac, dense_jac)


@pytest.mark.control_flow
def test_scan_pytree_xs():
    """Pytree xs: tuple of arrays as scan inputs.

    JAX flattens leaves into separate invars,
    so this exercises multiple xs vars with different dep patterns.
    """

    def f(x):
        a = x[:3]
        b = x[3:6]

        def body(carry, xsi):
            ai, bi = xsi
            return carry + ai * bi, carry

        carry_out, _ = jax.lax.scan(body, 0.0, (a, b))
        return jnp.array([carry_out])

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    expected = np.ones((1, 6), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_scan_pytree_ys():
    """Pytree ys: body returns a tuple as the y output.

    Exercises multiple ys outvars in the body jaxpr.
    """

    def f(x):
        def body(carry, xi):
            new_carry = carry + xi
            return new_carry, (new_carry, xi * 2.0)

        _, (sums, doubled) = jax.lax.scan(body, 0.0, x)
        return jnp.concatenate([sums, doubled])

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.array(
        [
            [1, 0, 0],  # sums[0] = x[0]
            [1, 1, 0],  # sums[1] = x[0] + x[1]
            [1, 1, 1],  # sums[2] = x[0] + x[1] + x[2]
            [1, 0, 0],  # doubled[0] = 2 * x[0]
            [0, 1, 0],  # doubled[1] = 2 * x[1]
            [0, 0, 1],  # doubled[2] = 2 * x[2]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_scan_length_one():
    """Single iteration with zero carry_init."""

    def f(x):
        def body(carry, xi):
            return carry + xi, carry

        carry_out, ys = jax.lax.scan(body, jnp.zeros(2), x.reshape(1, 2))
        return jnp.concatenate([carry_out, ys.ravel()])

    result = jacobian_sparsity(f, np.zeros(2)).todense().astype(int)
    expected = np.array(
        [
            [1, 0],  # carry_out[0] = 0 + x[0]
            [0, 1],  # carry_out[1] = 0 + x[1]
            [0, 0],  # ys[0][0] = carry_init[0] = 0
            [0, 0],  # ys[0][1] = carry_init[1] = 0
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_scan_scalar_carry_scalar_xs():
    """Simplest possible scan: scalar carry, scalar xs slices."""

    def f(x):
        def body(carry, xi):
            return carry + xi, carry

        carry_out, ys = jax.lax.scan(body, 0.0, x)
        return jnp.concatenate([jnp.array([carry_out]), ys])

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.array(
        [
            [1, 1, 1],  # carry_out = x[0] + x[1] + x[2]
            [0, 0, 0],  # ys[0] = carry_init = 0
            [1, 0, 0],  # ys[1] = x[0]
            [1, 1, 0],  # ys[2] = x[0] + x[1]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_scan_unroll():
    """unroll=True only affects XLA lowering, not the jaxpr.

    Sparsity should be identical to unroll=1.
    """

    def f_unrolled(x):
        def body(carry, xi):
            return carry + xi, carry + xi

        _, ys = jax.lax.scan(body, 0.0, x, unroll=True)
        return ys

    def f_default(x):
        def body(carry, xi):
            return carry + xi, carry + xi

        _, ys = jax.lax.scan(body, 0.0, x)
        return ys

    sp_unrolled = jacobian_sparsity(f_unrolled, np.zeros(4)).todense().astype(int)
    sp_default = jacobian_sparsity(f_default, np.zeros(4)).todense().astype(int)
    np.testing.assert_array_equal(sp_unrolled, sp_default)


@pytest.mark.control_flow
def test_scan_ys_independent_of_carry():
    """Ys depend only on xs, not on carry.

    Body: y = f(xi) only, carry is a counter.
    Each ys[t] depends only on xs[t].
    """

    def f(x):
        xs = x[1:].reshape(3, 1)
        carry_init = x[0:1]

        def body(carry, xi):
            return carry + 1.0, xi * 2.0

        _, ys = jax.lax.scan(body, carry_init, xs)
        return ys.ravel()

    result = jacobian_sparsity(f, np.zeros(4)).todense().astype(int)
    expected = np.array(
        [
            [0, 1, 0, 0],  # ys[0] = 2 * x[1]
            [0, 0, 1, 0],  # ys[1] = 2 * x[2]
            [0, 0, 0, 1],  # ys[2] = 2 * x[3]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_scan_carry_only_with_mixing():
    """Carry-only scan where the body mixes carry elements.

    Tests fixed-point convergence without xs,
    analogous to while_loop dependency spreading.
    """

    def f(x):
        def body(carry, _):
            # Mix: each element depends on both
            return jnp.array([carry[0] + carry[1], carry[1]]), None

        carry_out, _ = jax.lax.scan(body, x, None, length=3)
        return carry_out

    result = jacobian_sparsity(f, np.zeros(2)).todense().astype(int)
    # carry[0] depends on both inputs after mixing, carry[1] only on itself
    expected = np.array(
        [
            [1, 1],
            [0, 1],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_scan_carry_interaction_across_tuple():
    """Carry elements interact across iterations via tuple carry.

    One carry element feeds into another,
    so index sets spread through forward propagation.
    """

    def f(x):
        def body(carry, _):
            a, b = carry
            # a gets b's value, b stays
            return (a + b, b), None

        (a_out, b_out), _ = jax.lax.scan(body, (x[:2], x[2:4]), None, length=3)
        return jnp.concatenate([a_out, b_out])

    result = jacobian_sparsity(f, np.zeros(4)).todense().astype(int)
    # a_out depends on both a_init and b_init (elementwise)
    # b_out depends only on b_init
    expected = np.array(
        [
            [1, 0, 1, 0],
            [0, 1, 0, 1],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
def test_scan_with_cond_inside():
    """Scan body contains a conditional branch.

    Exercises scan + cond interaction.
    """

    def f(x):
        def body(carry, xi):
            new_carry = jax.lax.cond(
                True,
                lambda c, v: c + v,
                lambda c, v: c * v,
                carry,
                xi,
            )
            return new_carry, new_carry

        _, ys = jax.lax.scan(body, 0.0, x)
        return ys

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.array(
        [
            [1, 0, 0],  # ys[0] = x[0]
            [1, 1, 0],  # ys[1] = x[0] + x[1]
            [1, 1, 1],  # ys[2] = x[0] + x[1] + x[2]
        ],
        dtype=int,
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.control_flow
@pytest.mark.filterwarnings("ignore::asdex.DenseColoringWarning")
def test_scan_jacobian_values_pytree_xs():
    """Verify sparse Jacobian values match dense jax.jacobian with pytree xs."""

    def f(x):
        a = x[:3]
        b = x[3:6]

        def body(carry, xsi):
            ai, bi = xsi
            return carry + ai * bi, carry

        carry_out, ys = jax.lax.scan(body, 0.0, (a, b))
        return jnp.concatenate([jnp.array([carry_out]), ys])

    x = jnp.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    sparse_jac = jacobian(f, x)(x).todense()
    dense_jac = np.array(jax.jacobian(f)(x))
    np.testing.assert_allclose(sparse_jac, dense_jac)


def saturating_cumsum_scan(length: int, *, reverse: bool = False):
    """Scan whose carry index sets saturate after two steps.

    The body maps the carry through ``cumsum`` and emits the pre-update carry:
    ys at the first simulated step have identity index sets,
    every later step has lower-triangular index sets.
    There are no xs, so the saturation early exit applies.
    """

    def f(x):
        def body(carry, _):
            return jnp.cumsum(carry), carry

        _, ys = jax.lax.scan(body, x, None, length=length, reverse=reverse)
        return ys.flatten()

    return f


@pytest.mark.control_flow
def test_scan_saturation_long_scan():
    """A long scan without xs dependencies stops once the carry saturates.

    The carry index sets stabilize after two steps,
    so detection must not simulate all 50_000 timesteps.
    The resulting pattern stays exact.
    """
    n, length = 3, 50_000
    result = jacobian_sparsity(saturating_cumsum_scan(length), np.zeros(n))
    dense = result.todense().astype(int)

    assert dense.shape == (length * n, n)
    expected_first = np.eye(n, dtype=int)  # ys[0] = carry_init = x
    np.testing.assert_array_equal(dense[:n], expected_first)
    tril = np.tril(np.ones((n, n), dtype=int))  # ys[t>=1] = cumsum deps
    np.testing.assert_array_equal(dense[n:], np.tile(tril, (length - 1, 1)))


@pytest.mark.control_flow
def test_scan_saturation_exact():
    """The saturation early exit reproduces the exact per-timestep pattern."""
    x = jax.random.normal(jax.random.key(0), (3,))
    assert_jacobian_sparsity_exact(saturating_cumsum_scan(6), x)


@pytest.mark.control_flow
def test_scan_saturation_reverse():
    """Reversed scans place the replicated ys slices at the right timesteps.

    With ``reverse=True`` the simulation runs from the last timestep down,
    so the identity slice belongs to timestep length-1
    and the replicated triangular slices fill timesteps 0 onward.
    """
    n, length = 3, 100
    result = jacobian_sparsity(
        saturating_cumsum_scan(length, reverse=True), np.zeros(n)
    )
    dense = result.todense().astype(int)

    tril = np.tril(np.ones((n, n), dtype=int))  # ys[t<length-1] = cumsum deps
    np.testing.assert_array_equal(dense[: -n * 1], np.tile(tril, (length - 1, 1)))
    expected_last = np.eye(n, dtype=int)  # ys[length-1] = carry_init = x
    np.testing.assert_array_equal(dense[-n:], expected_last)


@pytest.mark.control_flow
def test_scan_saturation_no_carry():
    """A carry-free scan over closure state saturates after one step.

    The scan consts (the closed-over ``x``) carry input dependencies,
    which the early exit must preserve in every replicated ys slice.
    """
    n, length = 3, 500

    def f(x):
        def body(carry, _):
            return carry, jnp.sum(x)

        _, ys = jax.lax.scan(body, None, None, length=length)
        return ys

    result = jacobian_sparsity(f, np.zeros(n)).todense().astype(int)
    expected = np.ones((length, n), dtype=int)  # ys[t] = x[0] + x[1] + x[2]
    np.testing.assert_array_equal(result, expected)
