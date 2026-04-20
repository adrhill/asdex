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
    prop_closed_jaxpr,
    prop_dispatch,
    prop_jaxpr,
)
from asdex.detection._interpret._commons import atom_shape, singleton_index_set
from asdex.detection._interpret._reshape import prop_reshape


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
        prop_closed_jaxpr(eqn, env, state_consts, {}, "jaxpr")  # ty: ignore[invalid-argument-type]


def test_nested_jaxpr_missing_param_error_message():
    """Error message includes primitive name and issue tracker URL."""
    eqn = FakeEqn("xla_call", params={})
    env = {}
    state_consts = {}

    with pytest.raises(ValueError, match="xla_call"):
        prop_closed_jaxpr(eqn, env, state_consts, {}, "jaxpr")  # ty: ignore[invalid-argument-type]


def test_custom_call_missing_param_raises():
    """Error is raised when custom call primitive has no 'call_jaxpr' parameter."""
    eqn = FakeEqn("custom_jvp_call", params={})
    env = {}
    state_consts = {}

    with pytest.raises(ValueError, match="has no 'call_jaxpr' parameter"):
        prop_closed_jaxpr(eqn, env, state_consts, {}, "call_jaxpr")  # ty: ignore[invalid-argument-type]


def test_custom_call_missing_param_error_message():
    """Error message includes primitive name and issue tracker URL."""
    eqn = FakeEqn("custom_vjp_call", params={})
    env = {}
    state_consts = {}

    with pytest.raises(ValueError, match="custom_vjp_call"):
        prop_closed_jaxpr(eqn, env, state_consts, {}, "call_jaxpr")  # ty: ignore[invalid-argument-type]


def test_unknown_primitive_raises():
    """Unknown primitives raise NotImplementedError."""
    eqn = FakeEqn("nonexistent_op", params={})
    state_indices = {}
    state_consts = {}

    with pytest.raises(NotImplementedError, match="No handler for primitive"):
        prop_dispatch(eqn, state_indices, state_consts, {})  # ty: ignore[invalid-argument-type]


def test_unknown_primitive_error_message():
    """Error message includes primitive name and issue tracker URL."""
    eqn = FakeEqn("fake_primitive", params={})
    state_indices = {}
    state_consts = {}

    with pytest.raises(NotImplementedError) as exc_info:
        prop_dispatch(eqn, state_indices, state_consts, {})  # ty: ignore[invalid-argument-type]

    assert "fake_primitive" in str(exc_info.value)
    assert "https://github.com/adrhill/asdex/issues" in str(exc_info.value)


def test_prop_jaxpr_default_const_vals():
    """prop_jaxpr works when state_consts is not provided (defaults to {})."""
    dummy = jnp.zeros(2)
    closed_jaxpr = jax.make_jaxpr(lambda x: x + 1)(dummy)
    jaxpr = closed_jaxpr.jaxpr

    input_indices = [[singleton_index_set(0), singleton_index_set(1)]]
    # Call without state_consts — should default to empty dict
    result = prop_jaxpr(jaxpr, input_indices)
    assert len(result) == 1
    assert result[0] == [singleton_index_set(0), singleton_index_set(1)]


@pytest.mark.elementwise
def test_stop_gradient():
    """stop_gradient passes dependencies through unchanged."""

    def f(x):
        return jax.lax.stop_gradient(x)

    result = jacobian_sparsity(f, 3).todense().astype(int)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


def test_atom_shape_literal():
    """atom_shape extracts shape from Literal values."""
    val = np.array([1.0, 2.0, 3.0])
    lit = Literal(val=val, aval=None)
    assert atom_shape(lit) == (3,)

    scalar_lit = Literal(val=np.float32(1.0), aval=None)
    assert atom_shape(scalar_lit) == ()


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
        in_var: [singleton_index_set(0), singleton_index_set(1), singleton_index_set(2)]
    }
    with pytest.raises(ValueError, match="Reshape size mismatch"):
        prop_reshape(eqn, state_indices, {})  # ty: ignore[invalid-argument-type]


# Integration tests for precise handlers


@pytest.mark.array_ops
def test_transpose_2d():
    """Transpose preserves per-element dependencies with coordinate reordering.

    output[i,j] depends only on input[j,i], so the Jacobian is a permutation matrix.
    """

    def f(x):
        mat = x.reshape(2, 3)
        return mat.T.flatten()  # (3, 2) -> 6 elements

    result = jacobian_sparsity(f, 6).todense().astype(int)
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

    result = jacobian_sparsity(f, 3).todense().astype(int)
    expected = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_pad():
    """Pad inserts constant elements; original elements preserve dependencies."""

    def f(x):
        return jnp.pad(x, (1, 1), constant_values=0)

    result = jacobian_sparsity(f, 2).todense().astype(int)
    expected = np.array([[0, 0], [1, 0], [0, 1], [0, 0]])
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_tile():
    """jnp.tile tracks per-element dependencies via modular indexing."""

    def f(x):
        return jnp.tile(x, 2)

    result = jacobian_sparsity(f, 2).todense().astype(int)
    expected = np.array([[1, 0], [0, 1], [1, 0], [0, 1]], dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_split():
    """jnp.split tracks per-element dependencies through split and concat."""

    def f(x):
        parts = jnp.split(x, 2)
        return jnp.concatenate([parts[1], parts[0]])  # swap halves

    result = jacobian_sparsity(f, 4).todense().astype(int)
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

    result = jacobian_sparsity(f, 6).todense().astype(int)
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

    result = jacobian_sparsity(f, 3).todense().astype(int)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_sort():
    """1D sort: all outputs depend on all inputs."""

    def f(x):
        return jnp.sort(x)

    result = jacobian_sparsity(f, 3).todense().astype(int)
    expected = np.ones((3, 3), dtype=int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_custom_jvp_relu():
    """jax.nn.relu uses custom_jvp but tracks element-wise dependencies.

    ReLU is element-wise: each output depends only on corresponding input.
    """

    def f(x):
        return jax.nn.relu(x)

    result = jacobian_sparsity(f, 3).todense().astype(int)
    expected = np.eye(3, dtype=int)
    np.testing.assert_array_equal(result, expected)


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

    result = jacobian_sparsity(f, 3).todense().astype(int)
    expected = np.eye(3, dtype=int)  # Element-wise operation
    np.testing.assert_array_equal(result, expected)
