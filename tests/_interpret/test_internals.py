"""Unit tests for internal propagation functions.

https://docs.jax.dev/en/latest/jaxpr.html
"""

import logging
import warnings

import jax
import jax.nn
import jax.numpy as jnp
import numpy as np
import pytest
from jax import dtypes
from jax._src.core import Literal, Primitive, ShapedArray
from jax._src.dtypes import ExtendedDType
from jax.interpreters import ad

from asdex import DensePatternWarning, hessian_sparsity, jacobian, jacobian_sparsity
from asdex.detection._interpret import (
    _prop_closed_jaxpr,
    _prop_jaxpr,
)
from asdex.detection._interpret._common import (
    _atom_const_val,
    _atom_shape,
    _forward_across_jaxpr_boundary,
    _has_differentiable_dtype,
    _index_sets,
    _PropState,
    _seed_const_vals,
    _singleton_index_set,
)
from asdex.detection._interpret._reshape import _prop_reshape
from tests._utils import assert_jacobian_sparsity_exact


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


def test_wrong_index_set_count_raises(monkeypatch):
    """A handler that records too few index sets raises instead of shifting rows."""

    def drop_last_index_set(eqn, state):
        state.indices[eqn.outvars[0]] = _index_sets(state, eqn.invars[0])[:-1]

    monkeypatch.setattr(
        "asdex.detection._interpret._prop_unary_elementwise", drop_last_index_set
    )
    with pytest.raises(
        RuntimeError, match=r"'sin' recorded 2 index sets .* 3 elements"
    ):
        jacobian_sparsity(jnp.sin, np.zeros(3))


def test_nested_jaxpr_missing_param_raises():
    """Error is raised when nested jaxpr primitive has no 'jaxpr' parameter."""
    eqn = FakeEqn("pjit", params={})

    with pytest.raises(ValueError, match="has no 'jaxpr' parameter"):
        _prop_closed_jaxpr(eqn, _PropState(), "jaxpr")  # ty: ignore[invalid-argument-type]


def test_nested_jaxpr_missing_param_error_message():
    """Error message includes primitive name and issue tracker URL."""
    eqn = FakeEqn("xla_call", params={})

    with pytest.raises(ValueError, match="xla_call"):
        _prop_closed_jaxpr(eqn, _PropState(), "jaxpr")  # ty: ignore[invalid-argument-type]


def test_custom_call_missing_param_raises():
    """Error is raised when custom call primitive has no 'call_jaxpr' parameter."""
    eqn = FakeEqn("custom_jvp_call", params={})

    with pytest.raises(ValueError, match="has no 'call_jaxpr' parameter"):
        _prop_closed_jaxpr(eqn, _PropState(), "call_jaxpr")  # ty: ignore[invalid-argument-type]


def test_custom_call_missing_param_error_message():
    """Error message includes primitive name and issue tracker URL."""
    eqn = FakeEqn("custom_vjp_call", params={})

    with pytest.raises(ValueError, match="custom_vjp_call"):
        _prop_closed_jaxpr(eqn, _PropState(), "call_jaxpr")  # ty: ignore[invalid-argument-type]


_unknown_p = Primitive("asdex_test_unknown")
_unknown_p.def_abstract_eval(lambda x: x)


_unknown_linear_p = Primitive("asdex_test_unknown_linear")
_unknown_linear_p.def_abstract_eval(lambda x: x)
ad.deflinear(_unknown_linear_p, lambda ct: [_unknown_linear_p.bind(ct)])


def _f_unknown_linear(x):
    return jnp.sum(jnp.sin(_unknown_linear_p.bind(x)))


def test_unknown_primitive_falls_back_conservatively(caplog):
    """Unknown primitives fall back to a conservative pattern and are logged.

    Only the outputs downstream of the unknown primitive lose their sparsity.
    """

    def f(x):
        return jnp.concatenate([_unknown_p.bind(x[:2]), x[2:] * 2])

    with caplog.at_level(logging.INFO, logger="asdex.detection"):
        sparsity = jacobian_sparsity(f, np.zeros(4))

    expected = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ]
    )
    np.testing.assert_array_equal(sparsity.todense().astype(int), expected)
    assert sparsity._fallbacks == ("asdex_test_unknown",)

    (record,) = caplog.records
    assert record.name == "asdex.detection"
    assert record.levelno == logging.INFO
    assert record.message == (
        "No precise handler for primitive 'asdex_test_unknown',"
        " falling back to a conservative pattern."
        " If this primitive is part of JAX,"
        " please help out asdex's development by reporting it at"
        " https://github.com/adrhill/asdex/issues"
    )


def test_unknown_primitive_logged_once_per_detection(caplog):
    """An unknown primitive is logged once, however often it is dispatched.

    The scan body is propagated once per iteration,
    and the primitive also appears outside of it.
    """

    def f(x):
        def body(carry, _):
            return _unknown_p.bind(carry), None

        carry, _ = jax.lax.scan(body, x[:2], None, length=3)
        return jnp.concatenate([carry, _unknown_p.bind(x[2:])])

    with caplog.at_level(logging.INFO, logger="asdex.detection"):
        sparsity = jacobian_sparsity(f, np.zeros(4))

    expected = np.array(
        [
            [1, 1, 0, 0],
            [1, 1, 0, 0],
            [0, 0, 1, 1],
            [0, 0, 1, 1],
        ]
    )
    np.testing.assert_array_equal(sparsity.todense().astype(int), expected)
    assert sparsity._fallbacks == ("asdex_test_unknown",)
    (record,) = caplog.records
    assert "primitive 'asdex_test_unknown'" in record.message


def test_fallbacks_listed_in_tracing_order(capsys):
    """``print_fallbacks`` lists every unknown primitive in the order tracing met it.

    The order here is the reverse of the alphabetical one.
    ``print(sparsity)`` leaves the fallbacks out.
    """

    def f(x):
        return jnp.concatenate([_unknown_linear_p.bind(x[:2]), _unknown_p.bind(x[2:])])

    sparsity = jacobian_sparsity(f, np.zeros(4))

    assert sparsity._fallbacks == ("asdex_test_unknown_linear", "asdex_test_unknown")
    assert "asdex_test_unknown" not in str(sparsity)

    sparsity.print_fallbacks()
    assert capsys.readouterr().out == (
        "Sparsity detection fell back to conservative patterns"
        " for 2 primitives without a precise handler:\n"
        "  - asdex_test_unknown_linear\n"
        "  - asdex_test_unknown\n"
        "These patterns assume that every output depends on every input,\n"
        "so the detected pattern may be less sparse than necessary.\n"
        "Please help out asdex's development by reporting them at\n"
        "https://github.com/adrhill/asdex/issues\n"
    )


def test_fallbacks_empty_without_unknown_primitives(capsys):
    """Detection that knows every primitive reports no fallbacks."""
    sparsity = jacobian_sparsity(lambda x: jnp.sin(x) * 2, np.zeros(3))

    assert sparsity._fallbacks == ()
    sparsity.print_fallbacks()
    assert capsys.readouterr().out == (
        "No conservative fallbacks: every primitive had a precise handler.\n"
    )


def test_unknown_primitive_is_silent_by_default(capsys):
    """Without logging configuration, a sparse fallback prints nothing and does not warn."""

    def f(x):
        return jnp.concatenate([_unknown_p.bind(x[:2]), x[2:]])

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        jacobian_sparsity(f, np.zeros(4))

    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == ""


def test_unknown_primitive_dense_pattern_warns():
    """A fully dense pattern after an unknown-primitive fallback emits a warning."""
    with pytest.warns(DensePatternWarning) as record:
        result = jacobian_sparsity(_unknown_p.bind, np.zeros(3))

    np.testing.assert_array_equal(result.todense(), np.ones((3, 3)))
    assert result._fallbacks == ("asdex_test_unknown",)
    (warning,) = record
    message = str(warning.message)
    assert "fully dense 3\u00d73" in message
    assert "for primitives without a precise handler: asdex_test_unknown." in message
    assert "https://github.com/adrhill/asdex/issues" in message


@pytest.mark.parametrize(
    "detect",
    [
        lambda: jacobian_sparsity(_unknown_p.bind, np.zeros(3)),
        lambda: hessian_sparsity(_f_unknown_linear, np.zeros(3)),
        lambda: jacobian(_unknown_p.bind, np.zeros(3)),
    ],
    ids=["jacobian_sparsity", "hessian_sparsity", "jacobian"],
)
def test_dense_pattern_warning_points_at_caller(detect):
    """The warning is attributed to user code, whichever entry point ran detection."""
    with pytest.warns(DensePatternWarning) as record:
        detect()

    (warning,) = [w for w in record if issubclass(w.category, DensePatternWarning)]
    assert warning.filename == __file__


@pytest.mark.parametrize("n", [1, 3])
def test_single_row_or_column_pattern_does_not_warn(n):
    """Patterns with a single row or column take one color, dense or not."""
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        row = jacobian_sparsity(lambda x: jnp.sum(_unknown_p.bind(x)), np.zeros(n))
        col = jacobian_sparsity(
            lambda x: _unknown_p.bind(jnp.broadcast_to(x, (n,))), np.zeros(())
        )

    np.testing.assert_array_equal(row.todense(), np.ones((1, n)))
    np.testing.assert_array_equal(col.todense(), np.ones((n, 1)))


def test_unknown_primitive_dense_pattern_warns_for_hessian():
    """The dense-pattern warning also fires through ``hessian_sparsity``.

    Detection runs on the jaxpr of the gradient,
    which here contains the unknown primitive and its transpose.
    """
    with pytest.warns(DensePatternWarning, match="asdex_test_unknown_linear"):
        result = hessian_sparsity(_f_unknown_linear, np.zeros(3))

    np.testing.assert_array_equal(result.todense(), np.ones((3, 3)))


_NO_FALLBACKS_MESSAGE = (
    "No primitives without a precise handler were involved,"
    " so the function may be dense by nature,\n"
    "or asdex's handlers may have over-approximated its sparsity.\n"
    "If you expected a sparser pattern,"
    " please help out asdex's development by reporting this at"
    " https://github.com/adrhill/asdex/issues\n"
)


def test_dense_pattern_without_fallback_warns_without_blame():
    """A pattern that is dense by nature warns without naming any primitive."""
    with pytest.warns(DensePatternWarning) as record:
        result = jacobian_sparsity(lambda x: jnp.sum(x) * x, np.ones(3))

    np.testing.assert_array_equal(result.todense(), np.ones((3, 3)))
    assert result._fallbacks == ()
    (warning,) = record
    message = str(warning.message)
    assert "fully dense 3\u00d73" in message
    assert _NO_FALLBACKS_MESSAGE in message


def test_unknown_primitive_on_constants_is_not_blamed(caplog):
    """An unknown primitive without input dependencies has no conservative fallback.

    It is logged and listed by ``print_fallbacks``,
    but not named as the cause of a pattern that is dense anyway.
    """

    def f(x):
        return jnp.sum(x) * x + _unknown_p.bind(jnp.ones(3))

    with (
        caplog.at_level(logging.INFO, logger="asdex.detection"),
        pytest.warns(DensePatternWarning) as record,
    ):
        result = jacobian_sparsity(f, np.ones(3))

    np.testing.assert_array_equal(result.todense(), np.ones((3, 3)))
    assert result._fallbacks == ("asdex_test_unknown",)
    (log_record,) = caplog.records
    assert "primitive 'asdex_test_unknown'" in log_record.message
    (warning,) = record
    assert _NO_FALLBACKS_MESSAGE in str(warning.message)
    assert "asdex_test_unknown" not in str(warning.message)


def test_dense_pattern_warning_names_only_conservative_fallbacks():
    """The warning names unknown primitives on inputs, not those on constants.

    Both are listed by ``print_fallbacks``.
    """

    def f(x):
        return _unknown_p.bind(x) + _unknown_linear_p.bind(jnp.ones(3))

    with pytest.warns(DensePatternWarning) as record:
        result = jacobian_sparsity(f, np.ones(3))

    np.testing.assert_array_equal(result.todense(), np.ones((3, 3)))
    assert result._fallbacks == ("asdex_test_unknown", "asdex_test_unknown_linear")
    (warning,) = record
    message = str(warning.message)
    assert "for primitives without a precise handler: asdex_test_unknown." in message
    assert "asdex_test_unknown_linear" not in message


def test_unknown_primitive_conservative_after_constant_use():
    """A primitive first seen on constants still counts as conservative if later seen on inputs."""

    def f(x):
        return _unknown_p.bind(jnp.ones(3)) + _unknown_p.bind(x)

    with pytest.warns(DensePatternWarning, match="precise handler: asdex_test_unknown"):
        result = jacobian_sparsity(f, np.ones(3))

    assert result._fallbacks == ("asdex_test_unknown",)


# Unknown to asdex, but JAX can differentiate them,
# so their patterns can be checked against ``jax.jacobian``.
_unknown_int_p = Primitive("asdex_test_unknown_int")
_unknown_int_p.def_impl(lambda x: jnp.floor(x).astype(jnp.int32))
_unknown_int_p.def_abstract_eval(lambda x: ShapedArray(x.shape, jnp.int32))
ad.primitive_jvps[_unknown_int_p] = lambda primals, _: (
    _unknown_int_p.bind(*primals),
    np.zeros(primals[0].shape, dtypes.float0),
)

_unknown_mixed_p = Primitive("asdex_test_unknown_mixed")
_unknown_mixed_p.multiple_results = True
_unknown_mixed_p.def_impl(
    lambda x: [jnp.sin(x) * jnp.sum(x), jnp.floor(x).astype(jnp.int32)]
)
_unknown_mixed_p.def_abstract_eval(
    lambda x: [ShapedArray(x.shape, x.dtype), ShapedArray(x.shape, jnp.int32)]
)


def _unknown_mixed_jvp(primals, tangents):
    (x,) = primals
    (t,) = tangents
    t = ad.instantiate_zeros(t)
    y, n = _unknown_mixed_p.bind(x)
    y_dot = jnp.cos(x) * jnp.sum(x) * t + jnp.sin(x) * jnp.sum(t)
    return [y, n], [y_dot, np.zeros(n.shape, dtypes.float0)]


ad.primitive_jvps[_unknown_mixed_p] = _unknown_mixed_jvp


def test_unknown_primitive_integer_output_is_empty():
    """An integer output of an unknown primitive has zero derivative.

    The primitive is still listed by ``print_fallbacks``,
    since it is unknown, but it spreads no dependencies.
    """

    def f(x):
        return x * _unknown_int_p.bind(x).astype(x.dtype)

    assert_jacobian_sparsity_exact(
        f, np.array([1.5, 2.5, 3.5]), fallbacks=("asdex_test_unknown_int",)
    )


def test_unknown_primitive_integer_output_is_not_blamed():
    """An unknown primitive with only integer outputs has no conservative fallback.

    Its inputs carry dependencies, but no inexact output receives them,
    so the warning does not name it.
    """

    def f(x):
        return jnp.sum(x) * x + _unknown_int_p.bind(x).astype(x.dtype)

    with pytest.warns(DensePatternWarning) as record:
        assert_jacobian_sparsity_exact(
            f, np.array([1.5, 2.5, 3.5]), fallbacks=("asdex_test_unknown_int",)
        )

    (warning,) = record
    assert _NO_FALLBACKS_MESSAGE in str(warning.message)
    assert "asdex_test_unknown_int" not in str(warning.message)


def test_unknown_primitive_mixed_outputs():
    """Only the inexact output of an unknown primitive gets the dense pattern."""

    def f(x):
        y, n = _unknown_mixed_p.bind(x)
        return jnp.concatenate([y, n.astype(x.dtype)])

    assert_jacobian_sparsity_exact(
        f, np.array([0.3, 0.7, 1.1]), fallbacks=("asdex_test_unknown_mixed",)
    )


class _AtomWithAval:
    """Fake atom that only carries an abstract value."""

    def __init__(self, aval):
        self.aval = aval


class _DtypeAval:
    """Fake abstract value that only carries a dtype."""

    def __init__(self, dtype):
        self.dtype = dtype


class _FloatTangentRules:
    @staticmethod
    def tangent_dtype(_):
        return np.dtype(np.float32)


class _FloatTangentDType(ExtendedDType):
    """Extended dtype that declares a float32 tangent, so it is differentiable."""

    _rules = _FloatTangentRules
    name = "float_tangent"

    @property
    def type(self):
        return dtypes.extended


@pytest.mark.parametrize(
    ("aval", "expected"),
    [
        (ShapedArray((2,), jnp.float32), True),
        (ShapedArray((), jnp.float32, weak_type=True), True),
        (ShapedArray((2,), jnp.bfloat16), True),
        (ShapedArray((2,), jnp.complex64), True),
        (ShapedArray((2,), jnp.int32), False),
        (ShapedArray((), jnp.int32, weak_type=True), False),
        (ShapedArray((2,), jnp.uint8), False),
        (ShapedArray((2,), jnp.bool_), False),
        (ShapedArray((2,), dtypes.float0), False),
        (ShapedArray((2,), jax.random.key(0).dtype), False),
        (_DtypeAval(_FloatTangentDType()), True),
        (FakeAval((2,)), True),
    ],
    ids=[
        "float32",
        "weak_float",
        "bfloat16",
        "complex64",
        "int32",
        "weak_int",
        "uint8",
        "bool",
        "float0",
        "prng_key",
        "differentiable_extended",
        "no_dtype",
    ],
)
def test_has_differentiable_dtype(aval, expected):
    """Floating and complex dtypes are differentiable, integers, booleans and keys are not.

    Extended dtypes declare it themselves through their tangent dtype.
    Atoms without a dtype, such as tokens, count as differentiable,
    which keeps their index sets and is conservative.
    """
    atom = _AtomWithAval(aval)
    assert _has_differentiable_dtype(atom) is expected  # ty: ignore[invalid-argument-type]


def test_unknown_primitive_fallback_covers_true_jacobian():
    """The fallback pattern contains every nonzero of the true Jacobian.

    The primitive has a JVP rule, so the pattern can be checked against JAX.
    Its Jacobian block is dense, so the conservative pattern is also exact.
    """
    p = Primitive("asdex_test_unknown_differentiable")
    p.def_impl(lambda x: jnp.sin(x) * jnp.sum(x))
    p.def_abstract_eval(lambda x: x)

    def p_jvp(primals, tangents):
        (x,) = primals
        (t,) = tangents
        t = ad.instantiate_zeros(t)
        return p.bind(x), jnp.cos(x) * jnp.sum(x) * t + jnp.sin(x) * jnp.sum(t)

    ad.primitive_jvps[p] = p_jvp

    def f(x):
        return jnp.concatenate([p.bind(x[:2]), x[2:] * 2])

    x = np.array([0.3, 0.7, 1.1, 1.9])
    result = jacobian_sparsity(f, x).todense().astype(int)

    # Column by column, since the primitive has no batching rule for jax.jacobian.
    columns = [jax.jvp(f, (x,), (np.eye(4)[j],))[1] for j in range(4)]
    true_nonzeros = (np.stack(columns, axis=1) != 0).astype(int)

    np.testing.assert_array_equal(result, true_nonzeros)


def test_mutable_ref_raises():
    """Primitives on mutable refs raise instead of falling back.

    A write to a ref flows into an input of the primitive,
    which the conservative fallback would miss.
    Refs are unsupported by design,
    and the error says so and suggests how to rewrite the function.
    """

    def f(x):
        ref = jax.new_ref(jnp.zeros(3))
        ref[...] = x * 2.0
        return ref[...]

    with pytest.raises(
        NotImplementedError,
        match=r"'new_ref'.*mutable array reference.*does not support by design.*\.at\[",
    ):
        jacobian_sparsity(f, np.ones(3))


def test_mutable_ref_from_closure_raises():
    """Reads and writes raise too when the ref was created outside of the function.

    The jaxpr then has no ``new_ref``, so the write (``swap``) is the first to raise.
    """
    ref = jax.new_ref(jnp.zeros(3))

    def f(x):
        ref[...] = x * 2.0
        return ref[...]

    with pytest.raises(NotImplementedError, match=r"'swap'.*mutable array reference"):
        jacobian_sparsity(f, np.ones(3))


def test_known_primitive_without_precise_handler_is_reported(caplog):
    """A known primitive without a precise handler is logged and listed like an unknown one."""
    with (
        warnings.catch_warnings(),
        caplog.at_level(logging.INFO, logger="asdex.detection"),
    ):
        warnings.simplefilter("error")
        result = jacobian_sparsity(
            lambda x: jnp.linalg.cholesky(x @ x.T + jnp.eye(2)), np.ones((2, 2))
        )

    # jnp.linalg.cholesky zeroes the strict upper triangle after the primitive.
    expected = np.ones((4, 4))
    expected[1] = 0
    np.testing.assert_array_equal(result.todense(), expected)
    assert result._fallbacks == ("cholesky",)
    (record,) = caplog.records
    assert record.message.startswith(
        "No precise handler for primitive 'cholesky',"
        " falling back to a conservative pattern."
    )


def test_prop_jaxpr_default_const_vals():
    """_prop_jaxpr works when no parent state is provided (defaults to a fresh one)."""
    dummy = jnp.zeros(2)
    closed_jaxpr = jax.make_jaxpr(lambda x: x + 1)(dummy)
    jaxpr = closed_jaxpr.jaxpr

    input_indices = [[_singleton_index_set(0), _singleton_index_set(1)]]
    # Call without a parent state — a fresh empty state is created internally
    result = _prop_jaxpr(jaxpr, input_indices)
    assert len(result) == 1
    assert result[0] == [_singleton_index_set(0), _singleton_index_set(1)]


def test_seed_const_vals_is_lazy():
    """Seeded closure constants are stored as-is and materialized on first read.

    Eager ``np.asarray`` would copy every closure constant
    (e.g. all NN weights) device-to-host,
    even when its value is never consulted during propagation.
    """
    w = jnp.arange(3.0)
    closed = jax.make_jaxpr(lambda x: x * w)(jnp.zeros(3))
    assert len(closed.jaxpr.constvars) == 1
    var = closed.jaxpr.constvars[0]

    state = _PropState()
    _seed_const_vals(state, closed.jaxpr.constvars, closed.consts)
    assert state.consts[var] is closed.consts[0]  # stored unconverted

    val = _atom_const_val(var, state)
    assert isinstance(val, np.ndarray)
    assert state.consts[var] is val  # materialization is cached
    np.testing.assert_array_equal(val, np.arange(3.0))


def test_forward_across_jaxpr_boundary_preserves_laziness():
    """Forwarding consts across a jaxpr boundary must not materialize them."""
    w = jnp.arange(4.0)
    closed = jax.make_jaxpr(lambda x: x * w)(jnp.zeros(4))
    assert len(closed.jaxpr.constvars) == 1
    outer = closed.jaxpr.constvars[0]
    inner = jax.make_jaxpr(lambda x: x + 1.0)(jnp.zeros(4)).jaxpr.invars[0]

    state = _PropState()
    _seed_const_vals(state, closed.jaxpr.constvars, closed.consts)
    _forward_across_jaxpr_boundary(state, [outer], [inner])
    assert state.consts[inner] is closed.consts[0]  # still unconverted


def test_prop_closed_jaxpr_forwards_consts_out_lazily():
    """Consts leaving a nested jaxpr must not be materialized on the way out.

    A jit-wrapped model that threads a large closure constant through
    (e.g. NN weights) would otherwise pay a device-to-host copy
    at every nested-jaxpr boundary,
    even when no handler ever reads the value.
    """
    w = jnp.arange(5.0)

    @jax.jit
    def g(x):
        return jnp.sin(x), w  # second output is the captured const

    outer = jax.make_jaxpr(g)(jnp.zeros(5))
    eqn = outer.jaxpr.eqns[0]
    assert eqn.primitive.name == "jit"
    inner = eqn.params["jaxpr"]
    assert len(inner.consts) == 1

    state = _PropState()
    state.indices[outer.jaxpr.invars[0]] = [_singleton_index_set(i) for i in range(5)]
    _prop_closed_jaxpr(eqn, state, "jaxpr")

    # The const reaches the outer scope as the original array, unconverted.
    assert state.consts[eqn.outvars[1]] is inner.consts[0]


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

    state = _PropState(
        indices={  # ty: ignore[invalid-argument-type]
            in_var: [
                _singleton_index_set(0),
                _singleton_index_set(1),
                _singleton_index_set(2),
            ]
        }
    )
    with pytest.raises(ValueError, match="Reshape size mismatch"):
        _prop_reshape(eqn, state)  # ty: ignore[invalid-argument-type]


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
    (or fall back to a conservative pattern)
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
        _index_sets(_PropState(), var)
