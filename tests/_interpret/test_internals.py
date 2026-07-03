"""Unit tests for internal propagation functions.

https://docs.jax.dev/en/latest/jaxpr.html
"""

import jax
import jax.nn
import jax.numpy as jnp
import numpy as np
import pytest
from jax._src.core import Literal, Primitive

from asdex import jacobian_sparsity
from asdex.detection._interpret import (
    _prop_closed_jaxpr,
    _prop_dispatch,
    _prop_jaxpr,
)
from asdex.detection._interpret._common import (
    _atom_shape,
    _index_sets,
    _singleton_index_set,
)
from asdex.detection._interpret._reshape import _prop_reshape


class FakeAval:
    """Fake abstract value with shape."""

    def __init__(self, shape):
        self.shape = shape


class FakeVar:
    """Fake Var for testing with shape info."""

    def __init__(self, shape):
        self.aval = FakeAval(shape)


class FakeEqn:
    """Fake JaxprEqn for testing."""

    def __init__(self, primitive_name: str, params: dict):
        self.primitive = Primitive(primitive_name)
        self.params = params
        self.invars = []
        self.outvars = []


def test_nested_jaxpr_missing_param_raises():
    """Error is raised when nested jaxpr primitive has no 'jaxpr' parameter."""
    eqn = FakeEqn("pjit", params={})
    env = {}
    state_consts = {}

    with pytest.raises(ValueError, match="has no 'jaxpr' parameter"):
        _prop_closed_jaxpr(eqn, env, state_consts, {}, "jaxpr")  # ty: ignore[invalid-argument-type]


def test_nested_jaxpr_missing_param_error_message():
    """Error message includes primitive name and issue tracker URL."""
    eqn = FakeEqn("xla_call", params={})
    env = {}
    state_consts = {}

    with pytest.raises(ValueError, match="xla_call"):
        _prop_closed_jaxpr(eqn, env, state_consts, {}, "jaxpr")  # ty: ignore[invalid-argument-type]


def test_custom_call_missing_param_raises():
    """Error is raised when custom call primitive has no 'call_jaxpr' parameter."""
    eqn = FakeEqn("custom_jvp_call", params={})
    env = {}
    state_consts = {}

    with pytest.raises(ValueError, match="has no 'call_jaxpr' parameter"):
        _prop_closed_jaxpr(eqn, env, state_consts, {}, "call_jaxpr")  # ty: ignore[invalid-argument-type]


def test_custom_call_missing_param_error_message():
    """Error message includes primitive name and issue tracker URL."""
    eqn = FakeEqn("custom_vjp_call", params={})
    env = {}
    state_consts = {}

    with pytest.raises(ValueError, match="custom_vjp_call"):
        _prop_closed_jaxpr(eqn, env, state_consts, {}, "call_jaxpr")  # ty: ignore[invalid-argument-type]


def test_unknown_primitive_raises():
    """Unknown primitives raise NotImplementedError."""
    eqn = FakeEqn("nonexistent_op", params={})
    state_indices = {}
    state_consts = {}

    with pytest.raises(NotImplementedError, match="No handler for primitive"):
        _prop_dispatch(eqn, state_indices, state_consts, {})  # ty: ignore[invalid-argument-type]


def test_unknown_primitive_error_message():
    """Error message includes primitive name and issue tracker URL."""
    eqn = FakeEqn("fake_primitive", params={})
    state_indices = {}
    state_consts = {}

    with pytest.raises(NotImplementedError) as exc_info:
        _prop_dispatch(eqn, state_indices, state_consts, {})  # ty: ignore[invalid-argument-type]

    assert "fake_primitive" in str(exc_info.value)
    assert "https://github.com/adrhill/asdex/issues" in str(exc_info.value)


def test_prop_jaxpr_default_const_vals():
    """_prop_jaxpr works when state_consts is not provided (defaults to {})."""
    dummy = jnp.zeros(2)
    closed_jaxpr = jax.make_jaxpr(lambda x: x + 1)(dummy)
    jaxpr = closed_jaxpr.jaxpr

    input_indices = [[_singleton_index_set(0), _singleton_index_set(1)]]
    # Call without state_consts — should default to empty dict
    result = _prop_jaxpr(jaxpr, input_indices)
    assert len(result) == 1
    assert result[0] == [_singleton_index_set(0), _singleton_index_set(1)]


@pytest.mark.elementwise
def test_stop_gradient():
    """stop_gradient passes dependencies through unchanged."""

    def f(x):
        return jax.lax.stop_gradient(x)

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


def test_atom_shape_literal():
    """_atom_shape extracts shape from Literal values."""
    val = np.array([1.0, 2.0, 3.0])
    lit = Literal(val=val, aval=None)
    assert _atom_shape(lit) == (3,)

    scalar_lit = Literal(val=np.float32(1.0), aval=None)
    assert _atom_shape(scalar_lit) == ()


def test_reshape_size_mismatch_raises():
    """Reshape with input/output size mismatch raises ValueError.

    This should never occur in valid JAX code.
    """
    in_var = FakeVar(shape=(3,))
    out_var = FakeVar(shape=(2,))  # Mismatched size

    eqn = FakeEqn("reshape", params={"new_sizes": (2,), "dimensions": None})
    eqn.invars = [in_var]
    eqn.outvars = [out_var]

    state_indices = {
        in_var: [
            _singleton_index_set(0),
            _singleton_index_set(1),
            _singleton_index_set(2),
        ]
    }
    with pytest.raises(ValueError, match="Reshape size mismatch"):
        _prop_reshape(eqn, state_indices, {})  # ty: ignore[invalid-argument-type]


# Integration tests for precise handlers


@pytest.mark.array_ops
def test_transpose_2d():
    """Transpose preserves per-element dependencies with coordinate reordering.

    output[i,j] depends only on input[j,i], so the Jacobian is a permutation matrix.
    """

    def f(x):
        mat = x.reshape(2, 3)
        return mat.T.flatten()  # (3, 2) -> 6 elements

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    # Transpose of (2,3) -> (3,2): out[i,j] = in[j,i].
    # Flat mapping: out[0]=in[0], out[1]=in[3], out[2]=in[1],
    #               out[3]=in[4], out[4]=in[2], out[5]=in[5].
    expected = np.zeros((6, 6), dtype=int)
    for out_idx, in_idx in enumerate([0, 3, 1, 4, 2, 5]):
        expected[out_idx, in_idx] = 1
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_reverse():
    """jnp.flip reverses the array; output[i] depends on input[n-1-i]."""

    def f(x):
        return jnp.flip(x)

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_pad():
    """Pad inserts constant elements; original elements preserve dependencies."""

    def f(x):
        return jnp.pad(x, (1, 1), constant_values=0)

    result = jacobian_sparsity(f, np.zeros(2)).todense().astype(int)
    expected = np.array([[0, 0], [1, 0], [0, 1], [0, 0]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_tile():
    """jnp.tile tracks per-element dependencies via modular indexing."""

    def f(x):
        return jnp.tile(x, 2)

    result = jacobian_sparsity(f, np.zeros(2)).todense().astype(int)
    expected = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_split():
    """jnp.split tracks per-element dependencies through split and concat."""

    def f(x):
        parts = jnp.split(x, 2)
        return jnp.concatenate([parts[1], parts[0]])  # swap halves

    result = jacobian_sparsity(f, np.zeros(4)).todense().astype(int)
    expected = np.array(
        [[0, 0, 1, 0], [0, 0, 0, 1], [1, 0, 0, 0], [0, 1, 0, 0]], dtype=int
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_matmul():
    """Matrix multiplication (dot_general) tracks row/column dependencies.

    For f(x) = X @ X.T where X is (2, 3),
    output[i,j] depends on rows i and j of input.
    Diagonal blocks share state_indices, off-diagonal blocks union both rows.
    """

    def f(x):
        mat = x.reshape(2, 3)
        return (mat @ mat.T).flatten()

    result = jacobian_sparsity(f, np.zeros(6)).todense().astype(int)
    expected = np.array(
        [
            [1, 1, 1, 0, 0, 0],
            [1, 1, 1, 1, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [0, 0, 0, 1, 1, 1],
        ]
    )
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_iota_eye():
    """Eye @ x: dot_general skips value-level zeros in the identity matrix.

    The handler exploits that ``jnp.eye(3)`` is a known constant,
    so out[i] depends only on x[i].
    """

    def f(x):
        return jnp.eye(3) @ x

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_sort():
    """1D sort: all outputs depend on all inputs."""

    def f(x):
        return jnp.sort(x)

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.ones((3, 3), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_custom_jvp_relu():
    """jax.nn.relu uses custom_jvp but tracks element-wise dependencies.

    ReLU is element-wise: each output depends only on corresponding input.
    """

    def f(x):
        return jax.nn.relu(x)

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
@pytest.mark.bug
def test_custom_jvp_rule_adding_dependencies_missed():
    """Detection misses dependencies that only the custom JVP rule introduces.

    A straight-through estimator rounds in the primal
    but passes tangents through unchanged,
    so the true Jacobian is 2·I.
    The custom_jvp_call handler traces the primal ``call_jaxpr``,
    where ``round`` has zero derivative,
    and never sees the custom rule.
    The detected pattern is empty (missing nonzeros),
    so downstream decompression silently returns an all-zero Jacobian.

    This test pins the current broken behavior
    and must be flipped to the diagonal pattern when the handler is fixed.

    TODO(custom_jvp_call): propagate through the custom JVP rule
    (or fall back conservatively)
    so the detected pattern covers the true 2·I Jacobian.
    """

    @jax.custom_jvp
    def ste_round(x):
        return jnp.round(x)

    @ste_round.defjvp
    def ste_round_jvp(primals, tangents):
        (x,), (t,) = primals, tangents
        return jnp.round(x), t  # derivative rule: identity

    def f(x):
        return ste_round(x) * 2.0

    x = np.array([0.1, 1.6, 2.4])
    true_jacobian = np.asarray(jax.jacfwd(f)(x))
    np.testing.assert_array_equal(true_jacobian, 2 * np.eye(3))

    # Broken: the detected pattern misses all three true nonzeros.
    result = jacobian_sparsity(f, x).todense().astype(int)
    np.testing.assert_array_equal(result, np.zeros((3, 3), dtype=int))


@pytest.mark.array_ops
def test_custom_vjp_user_defined():
    """User-defined custom_vjp traces forward computation."""

    @jax.custom_vjp
    def my_square(x):
        return x**2

    def my_square_fwd(x):
        return my_square(x), x

    def my_square_bwd(res, g):
        x = res
        return (2 * x * g,)

    my_square.defvjp(my_square_fwd, my_square_bwd)

    def f(x):
        return my_square(x)

    result = jacobian_sparsity(f, np.zeros(3)).todense().astype(int)
    expected = np.eye(3, dtype=int)  # Element-wise operation
    np.testing.assert_array_equal(result, expected)


# Internal invariants


def test_index_sets_unknown_var_raises():
    """_index_sets raises for a Var that was never seeded or written.

    Every Var is either seeded (invars, constvars) or written by a handler,
    so a missing Var indicates a handler bug upstream.
    Returning a guessed default silently drops dependencies
    and gets the element count wrong.
    """
    var = jax.make_jaxpr(lambda x: x + 1)(jnp.zeros(2)).jaxpr.invars[0]

    with pytest.raises(KeyError):
        _index_sets({}, var)


# Contracts with JAX internals


def test_while_invars_layout():
    """JAX binds while_p invars as [cond_consts, body_consts, carry].

    The while handler slices eqn.invars by cond_nconsts and body_nconsts,
    so this ordering is a load-bearing contract with JAX internals.
    This test fails loudly if a JAX upgrade reorders the groups.
    """

    def f(c, b, init):
        return jax.lax.while_loop(lambda s: s[0] < c, lambda s: s + b, init)

    jaxpr = jax.make_jaxpr(f)(1.0, 2.0, jnp.zeros(2)).jaxpr
    (eqn,) = [e for e in jaxpr.eqns if e.primitive.name == "while"]
    c_var, b_var, init_var = jaxpr.invars

    assert eqn.params["cond_nconsts"] == 1
    assert eqn.params["body_nconsts"] == 1
    assert list(eqn.invars) == [c_var, b_var, init_var]
