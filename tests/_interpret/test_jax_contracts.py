"""Contract tests pinning implicit layout assumptions about JAX internals.

The interpreter reads ``eqn.invars``, ``eqn.outvars``, and ``eqn.params``
of primitives whose layout is an undocumented contract with JAX internals
(see the private-API dependency on ``jax._src`` in ``detection/_api.py``).
A params rename fails loudly with a ``KeyError``,
but a reordering of variables or a change in parameter semantics
would silently produce wrong sparsity patterns.

Each test here traces a small function on the installed JAX
and asserts the exact structure the corresponding handler relies on,
so a JAX upgrade that breaks a contract fails loudly in this file
instead of surfacing as a wrong pattern elsewhere.

https://docs.jax.dev/en/latest/jaxpr.html
"""

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.core import ClosedJaxpr, Jaxpr
from jax._src.interpreters.partial_eval import dce_jaxpr


def _iter_eqns(jaxpr):
    """Yield all equations in ``jaxpr``, recursing into jaxprs nested in params."""
    for eqn in jaxpr.eqns:
        yield eqn
        for val in eqn.params.values():
            for sub in _sub_jaxprs(val):
                yield from _iter_eqns(sub)


def _sub_jaxprs(val):
    """Yield the jaxprs contained in a single params value, if any."""
    items = val if isinstance(val, (tuple, list)) else (val,)
    for item in items:
        if isinstance(item, ClosedJaxpr):
            yield item.jaxpr
        elif isinstance(item, Jaxpr):
            yield item


def _unique_eqn(jaxpr, primitive_name: str):
    """Return the unique equation with ``primitive_name``, searching nested jaxprs."""
    matches = [e for e in _iter_eqns(jaxpr) if e.primitive.name == primitive_name]
    assert len(matches) == 1, f"expected one '{primitive_name}' eqn, got {len(matches)}"
    return matches[0]


# Control flow


def test_while_invars_layout():
    """JAX binds while_p invars as [cond_consts, body_consts, carry].

    The while handler slices eqn.invars by cond_nconsts and body_nconsts,
    so this ordering is a load-bearing contract with JAX internals.
    This test fails loudly if a JAX upgrade reorders the groups.
    """

    def f(c, b, init):
        return jax.lax.while_loop(lambda s: s[0] < c, lambda s: s + b, init)

    jaxpr = jax.make_jaxpr(f)(1.0, 2.0, jnp.zeros(2)).jaxpr
    eqn = _unique_eqn(jaxpr, "while")
    c_var, b_var, init_var = jaxpr.invars

    assert eqn.params["cond_nconsts"] == 1
    assert eqn.params["body_nconsts"] == 1
    assert list(eqn.invars) == [c_var, b_var, init_var]


def test_while_inner_jaxpr_layouts():
    """The while_p cond and body jaxprs take [their consts..., carry...].

    ``_prop_while`` feeds the body jaxpr the index sets of
    [body_consts..., carry...] and treats the body outputs as the new carry,
    so both inner input orderings and the body output layout are load-bearing.
    """

    def f(c, b, init):
        return jax.lax.while_loop(
            lambda s: jnp.sum(s) < jnp.sum(c), lambda s: s + jnp.sum(b), init
        )

    jaxpr = jax.make_jaxpr(f)(jnp.zeros(2), jnp.zeros(3), jnp.zeros(4)).jaxpr
    eqn = _unique_eqn(jaxpr, "while")

    cond_jaxpr = eqn.params["cond_jaxpr"].jaxpr
    body_jaxpr = eqn.params["body_jaxpr"].jaxpr
    # cond consts have shape (2,), body consts (3,), the carry (4,)
    assert [v.aval.shape for v in cond_jaxpr.invars] == [(2,), (4,)]
    assert [v.aval.shape for v in body_jaxpr.invars] == [(3,), (4,)]
    assert [v.aval.shape for v in body_jaxpr.outvars] == [(4,)]


def test_scan_layout():
    """JAX binds scan_p as [consts, carry_init, xs] -> [carry_final, ys].

    ``_prop_scan`` slices eqn.invars by the consts and carry group sizes,
    slices xs along a leading axis of size ``length``,
    and feeds the body jaxpr [consts..., carry..., x_slice...],
    so every one of these orderings is load-bearing.

    JAX 0.11 replaced the ``num_consts`` / ``num_carry`` params
    with ``ft_in``, an ``FTTuple`` splitting the invars into
    ``(consts, carry, xs)`` groups whose lengths give those counts.
    """

    def f(c, init, xs):
        def body(carry, x):
            return carry + jnp.sum(c) + jnp.sum(x), jnp.sum(carry) * jnp.ones(6)

        return jax.lax.scan(body, init, xs)

    jaxpr = jax.make_jaxpr(f)(jnp.zeros(2), jnp.zeros(4), jnp.zeros((3, 5))).jaxpr
    eqn = _unique_eqn(jaxpr, "scan")
    c_var, init_var, xs_var = jaxpr.invars

    num_consts, num_carry, num_xs = (
        len(group) for group in eqn.params["ft_in"].unpack()
    )
    assert (num_consts, num_carry, num_xs) == (1, 1, 1)
    assert eqn.params["length"] == 3
    assert eqn.params["reverse"] is False
    assert list(eqn.invars) == [c_var, init_var, xs_var]

    # outvars: [carry_final..., ys...] where ys gain a leading axis of size length
    assert [v.aval.shape for v in eqn.outvars] == [(4,), (3, 6)]

    # body: [consts..., carry..., x_slice...] -> [carry_new..., y_slice...]
    body_closed = eqn.params["jaxpr"]
    assert isinstance(body_closed, ClosedJaxpr)
    assert [v.aval.shape for v in body_closed.jaxpr.invars] == [(2,), (4,), (5,)]
    assert [v.aval.shape for v in body_closed.jaxpr.outvars] == [(4,), (6,)]


def test_cond_layout():
    """JAX binds cond_p as [index_scalar, operands...] with branches[i] for index i.

    ``_prop_cond`` drops invars[0] (the selector)
    and feeds invars[1:] to every branch jaxpr,
    so the selector-first layout and the operand alignment are load-bearing.
    A boolean ``lax.cond`` lowers to index 0 for the false branch
    and index 1 for the true branch.
    """

    def f(pred, x, y):
        return jax.lax.cond(pred, lambda a, b: a + b, lambda a, b: a * b, x, y)

    jaxpr = jax.make_jaxpr(f)(True, jnp.zeros(3), jnp.zeros(3)).jaxpr
    eqn = _unique_eqn(jaxpr, "cond")
    _, x_var, y_var = jaxpr.invars

    # invars[0] is the scalar branch index (the converted predicate)
    assert eqn.invars[0].aval.shape == ()
    assert list(eqn.invars[1:]) == [x_var, y_var]

    branches = eqn.params["branches"]
    assert len(branches) == 2
    for branch in branches:
        assert [v.aval.shape for v in branch.jaxpr.invars] == [(3,), (3,)]

    # branches[0] is the false branch (mul), branches[1] the true branch (add)
    assert branches[0].jaxpr.eqns[0].primitive.name == "mul"
    assert branches[1].jaxpr.eqns[0].primitive.name == "add"


# Element selection


def test_select_n_invars_layout():
    """JAX binds select_n_p invars as [which, cases...] in case order.

    ``_prop_select_n`` reads invars[0] as the selector
    and indexes invars[1:] by the selector value when it is constant,
    so both the selector position and the case order are load-bearing.
    """

    def f(which, a, b, c):
        return jax.lax.select_n(which, a, b, c)

    jaxpr = jax.make_jaxpr(f)(
        np.zeros(3, dtype=np.int32), jnp.zeros(3), jnp.zeros(3), jnp.zeros(3)
    ).jaxpr
    eqn = _unique_eqn(jaxpr, "select_n")

    assert list(eqn.invars) == list(jaxpr.invars)


def test_select_n_case_order():
    """A boolean selector picks cases in (on_false, on_true) order.

    ``lax.select(pred, on_true, on_false)`` lowers to
    ``select_n(pred, on_false, on_true)``,
    pinning that cases[i] is selected where the selector equals i.
    """

    def f(pred, on_true, on_false):
        return jax.lax.select(pred, on_true, on_false)

    jaxpr = jax.make_jaxpr(f)(np.zeros(3, dtype=bool), jnp.zeros(3), jnp.zeros(3)).jaxpr
    eqn = _unique_eqn(jaxpr, "select_n")
    pred_var, on_true_var, on_false_var = jaxpr.invars

    assert list(eqn.invars) == [pred_var, on_false_var, on_true_var]


# Slicing and indexing


def test_dynamic_slice_invars_layout():
    """JAX binds dynamic_slice_p invars as [operand, *start_indices] in dim order.

    ``_prop_dynamic_slice`` reads invars[0] as the operand
    and invars[1:] as one start index per operand dimension,
    so the operand-first layout and the start ordering are load-bearing.
    Unsigned starts skip the negative-index normalization,
    keeping the eqn invars identical to the function arguments.
    """

    def f(x, i, j):
        return jax.lax.dynamic_slice(x, (i, j), (2, 3))

    jaxpr = jax.make_jaxpr(f)(jnp.zeros((4, 5)), np.uint32(1), np.uint32(0)).jaxpr
    eqn = _unique_eqn(jaxpr, "dynamic_slice")
    x_var, i_var, j_var = jaxpr.invars

    assert list(eqn.invars) == [x_var, i_var, j_var]
    assert eqn.params["slice_sizes"] == (2, 3)


def test_dynamic_update_slice_invars_layout():
    """JAX binds dynamic_update_slice_p invars as [operand, update, *start_indices].

    ``_prop_dynamic_update_slice`` reads invars[0] as the operand,
    invars[1] as the update,
    and invars[2:] as one start index per operand dimension.
    """

    def f(x, u, i, j):
        return jax.lax.dynamic_update_slice(x, u, (i, j))

    jaxpr = jax.make_jaxpr(f)(
        jnp.zeros((4, 5)), jnp.ones((2, 3)), np.uint32(1), np.uint32(0)
    ).jaxpr
    eqn = _unique_eqn(jaxpr, "dynamic_update_slice")
    x_var, u_var, i_var, j_var = jaxpr.invars

    assert list(eqn.invars) == [x_var, u_var, i_var, j_var]


def test_gather_layout():
    """JAX binds gather_p invars as [operand, start_indices].

    ``_prop_gather`` reads invars[0] as the array being indexed
    and invars[1] as the integer index vectors,
    and interprets the ``GatherDimensionNumbers`` fields for 1D fancy indexing
    as pinned here.
    """
    jaxpr = jax.make_jaxpr(lambda x, idx: x[idx])(
        jnp.zeros(7), np.zeros(3, dtype=np.int32)
    ).jaxpr
    eqn = _unique_eqn(jaxpr, "gather")
    x_var = jaxpr.invars[0]

    assert eqn.invars[0] == x_var
    assert np.issubdtype(eqn.invars[1].aval.dtype, np.integer)

    dim_nums = eqn.params["dimension_numbers"]
    assert dim_nums.offset_dims == ()
    assert dim_nums.collapsed_slice_dims == (0,)
    assert dim_nums.start_index_map == (0,)
    assert eqn.params["slice_sizes"] == (1,)


def test_scatter_layout():
    """JAX binds scatter_p invars as [operand, scatter_indices, updates].

    ``_prop_scatter`` reads the three positions by index,
    so a reordering would silently swap the roles of operand and updates.
    The ``ScatterDimensionNumbers`` fields for 1D index assignment
    are pinned alongside.
    """
    jaxpr = jax.make_jaxpr(lambda x, idx, u: x.at[idx].set(u))(
        jnp.zeros(7), np.zeros(3, dtype=np.int32), jnp.ones(3)
    ).jaxpr
    eqn = _unique_eqn(jaxpr, "scatter")
    x_var, _, u_var = jaxpr.invars

    assert eqn.invars[0] == x_var
    assert np.issubdtype(eqn.invars[1].aval.dtype, np.integer)
    assert eqn.invars[2] == u_var

    dim_nums = eqn.params["dimension_numbers"]
    assert dim_nums.update_window_dims == ()
    assert dim_nums.inserted_window_dims == (0,)
    assert dim_nums.scatter_dims_to_operand_dims == (0,)


# Array construction and reshaping


def test_pad_invars_layout():
    """JAX binds pad_p invars as [operand, padding_value]."""

    def f(x, v):
        return jax.lax.pad(x, v, ((1, 2, 0),))

    jaxpr = jax.make_jaxpr(f)(jnp.zeros(3), jnp.zeros(())).jaxpr
    eqn = _unique_eqn(jaxpr, "pad")
    x_var, v_var = jaxpr.invars

    assert list(eqn.invars) == [x_var, v_var]


def test_pad_padding_config_convention():
    """pad_p padding_config entries are (low, high, interior) per dimension.

    ``_prop_pad`` unpacks each entry in this order,
    so a reordering would silently shift which outputs keep their index sets.
    ``jnp.pad(x, (1, 2))`` pads 1 before and 2 after by its public contract,
    which pins the (low, high) order of the lax-level param.
    """
    jaxpr = jax.make_jaxpr(lambda x: jnp.pad(x, (1, 2)))(jnp.zeros(3)).jaxpr
    eqn = _unique_eqn(jaxpr, "pad")
    assert eqn.params["padding_config"] == ((1, 2, 0),)

    # Interior padding inserts elements between neighbors:
    # 3 elements with interior=2 give 3 + 2 * 2 = 7.
    jaxpr = jax.make_jaxpr(lambda x, v: jax.lax.pad(x, v, ((0, 0, 2),)))(
        jnp.zeros(3), jnp.zeros(())
    ).jaxpr
    eqn = _unique_eqn(jaxpr, "pad")
    assert eqn.outvars[0].aval.shape == (7,)


def test_top_k_outvars_layout():
    """JAX binds top_k_p outvars as [values, indices].

    ``_prop_top_k`` writes reduction-style index sets to outvars[0]
    and empty index sets to outvars[1],
    so swapping the outputs would silently zero out the values pattern.
    """
    jaxpr = jax.make_jaxpr(lambda x: jax.lax.top_k(x, 2))(jnp.zeros(5)).jaxpr
    eqn = _unique_eqn(jaxpr, "top_k")

    assert eqn.params["k"] == 2
    values, indices = eqn.outvars
    assert values.aval.shape == (2,)
    assert values.aval.dtype == eqn.invars[0].aval.dtype
    assert indices.aval.shape == (2,)
    assert np.issubdtype(indices.aval.dtype, np.integer)


def test_iota_dimension_semantics():
    """iota_p values increase along the axis given by the ``dimension`` param.

    ``_prop_iota`` reconstructs the concrete values from shape and dimension
    for downstream gather and scatter precision,
    so its formula must match what JAX actually computes.
    """
    jaxpr = jax.make_jaxpr(lambda: jax.lax.broadcasted_iota(jnp.int32, (2, 3), 1))()
    eqn = _unique_eqn(jaxpr.jaxpr, "iota")

    assert eqn.params["shape"] == (2, 3)
    assert eqn.params["dimension"] == 1
    assert eqn.params["dtype"] == jnp.int32

    # The handler's reconstruction formula, pinned against the real values
    expected = np.broadcast_to(np.arange(3, dtype=np.int32).reshape(1, 3), (2, 3))
    actual = np.asarray(jax.lax.broadcasted_iota(jnp.int32, (2, 3), 1))
    np.testing.assert_array_equal(actual, expected)


# Contractions


def test_dot_general_dimension_numbers_layout():
    """dot_general_p dimension_numbers nest as ((lhs_contract, rhs_contract), (lhs_batch, rhs_batch)).

    ``_prop_dot_general`` destructures exactly this nesting,
    and for square shapes a swapped nesting would go unnoticed at trace time.
    """
    jaxpr = jax.make_jaxpr(lambda a, b: a @ b)(
        jnp.zeros((2, 3)), jnp.zeros((3, 4))
    ).jaxpr
    eqn = _unique_eqn(jaxpr, "dot_general")
    assert eqn.params["dimension_numbers"] == (((1,), (0,)), ((), ()))

    jaxpr = jax.make_jaxpr(lambda a, b: a @ b)(
        jnp.zeros((5, 2, 3)), jnp.zeros((5, 3, 4))
    ).jaxpr
    eqn = _unique_eqn(jaxpr, "dot_general")
    assert eqn.params["dimension_numbers"] == (((2,), (1,)), ((0,), (0,)))


def test_conv_dimension_numbers_layout():
    """ConvDimensionNumbers specs are (batch, feature, *spatial) dimension positions.

    ``_prop_conv_general_dilated`` reads spec[0] as the batch dimension,
    spec[1] as the feature dimension, and spec[2:] as spatial dimensions
    (out_feature and in_feature for the kernel spec).
    An NHWC/HWIO layout makes the positions asymmetric,
    so any change in the spec convention fails here.
    """

    def f(lhs, rhs):
        return jax.lax.conv_general_dilated(
            lhs,
            rhs,
            window_strides=(1, 1),
            padding="VALID",
            dimension_numbers=("NHWC", "HWIO", "NHWC"),
        )

    jaxpr = jax.make_jaxpr(f)(jnp.zeros((1, 5, 5, 2)), jnp.zeros((2, 2, 2, 3))).jaxpr
    eqn = _unique_eqn(jaxpr, "conv_general_dilated")
    lhs_var, rhs_var = jaxpr.invars

    assert list(eqn.invars) == [lhs_var, rhs_var]

    dim_nums = eqn.params["dimension_numbers"]
    assert dim_nums.lhs_spec == (0, 3, 1, 2)  # N=0, C=3, spatial H,W = 1,2
    assert dim_nums.rhs_spec == (3, 2, 0, 1)  # O=3, I=2, spatial H,W = 0,1
    assert dim_nums.out_spec == (0, 3, 1, 2)

    assert eqn.params["window_strides"] == (1, 1)
    assert eqn.params["padding"] == ((0, 0), (0, 0))
    assert eqn.params["lhs_dilation"] == (1, 1)
    assert eqn.params["rhs_dilation"] == (1, 1)
    assert eqn.params["feature_group_count"] == 1
    assert eqn.params["batch_group_count"] == 1


# Nested jaxprs


def test_nested_jaxpr_param_alignment():
    """The jit/pjit ``jaxpr`` param aligns positionally with the eqn variables.

    ``_prop_closed_jaxpr`` forwards index sets, const values, and bounds
    by zipping eqn.invars with the inner jaxpr invars
    (and eqn.outvars with the inner outvars),
    so the 1:1 positional correspondence is load-bearing.
    """
    jaxpr = jax.make_jaxpr(jax.jit(lambda a, b: (a * 2.0, jnp.sum(b))))(
        jnp.zeros(2), jnp.zeros(3)
    ).jaxpr

    (eqn,) = jaxpr.eqns
    assert eqn.primitive.name in ("jit", "pjit")

    inner = eqn.params["jaxpr"]
    assert isinstance(inner, ClosedJaxpr)
    assert [v.aval for v in inner.jaxpr.invars] == [v.aval for v in eqn.invars]
    assert [v.aval for v in inner.jaxpr.outvars] == [v.aval for v in eqn.outvars]


def test_custom_jvp_call_jaxpr_alignment():
    """custom_jvp_call stores the primal function as an aligned ``call_jaxpr`` param."""

    @jax.custom_jvp
    def g(a, b):
        return jnp.sum(a) * b

    @g.defjvp
    def g_jvp(primals, tangents):
        a, b = primals
        da, db = tangents
        return g(a, b), jnp.sum(da) * b + jnp.sum(a) * db

    jaxpr = jax.make_jaxpr(g)(jnp.zeros(2), jnp.zeros(3)).jaxpr
    eqn = _unique_eqn(jaxpr, "custom_jvp_call")

    assert list(eqn.invars) == list(jaxpr.invars)
    inner = eqn.params["call_jaxpr"]
    assert [v.aval for v in inner.jaxpr.invars] == [v.aval for v in eqn.invars]
    assert len(inner.jaxpr.outvars) == len(eqn.outvars) == 1


def test_custom_vjp_call_jaxpr_alignment():
    """custom_vjp_call stores the primal function as an aligned ``call_jaxpr`` param."""

    @jax.custom_vjp
    def g(a, b):
        return jnp.sum(a) * b

    def g_fwd(a, b):
        return g(a, b), (a, b)

    def g_bwd(res, ct):
        a, b = res
        return jnp.sum(ct * b) * jnp.ones_like(a), jnp.sum(a) * ct

    g.defvjp(g_fwd, g_bwd)

    jaxpr = jax.make_jaxpr(g)(jnp.zeros(2), jnp.zeros(3)).jaxpr
    eqn = _unique_eqn(jaxpr, "custom_vjp_call")

    assert list(eqn.invars) == list(jaxpr.invars)
    inner = eqn.params["call_jaxpr"]
    assert [v.aval for v in inner.jaxpr.invars] == [v.aval for v in eqn.invars]
    assert len(inner.jaxpr.outvars) == len(eqn.outvars) == 1


# Dead code elimination


def test_dce_jaxpr_instantiate_preserves_inputs():
    """``dce_jaxpr`` with instantiate=True keeps all invars in order.

    ``_dce_closed_jaxpr`` in ``detection/_api.py`` relies on this
    to keep the seeded input index sets aligned with the jaxpr inputs
    while dead equations are removed.
    """

    def f(a, b):
        dead = jnp.sum(b)  # noqa: F841
        return a * 2.0

    jaxpr = jax.make_jaxpr(f)(jnp.zeros(2), jnp.zeros(3)).jaxpr
    assert "reduce_sum" in [e.primitive.name for e in jaxpr.eqns]

    new_jaxpr, _ = dce_jaxpr(jaxpr, [True] * len(jaxpr.outvars), instantiate=True)

    # All inputs survive in order, even though ``b`` is now unused
    assert [v.aval for v in new_jaxpr.invars] == [v.aval for v in jaxpr.invars]
    # The dead equation is removed
    assert "reduce_sum" not in [e.primitive.name for e in new_jaxpr.eqns]
