"""Propagation rule for custom_jvp_call."""

from collections.abc import Sequence
from typing import Any

from jax._src.core import Jaxpr, JaxprEqn

from ._common import (
    IndexSet,
    PropJaxprFn,
    _atom_numel,
    _empty_index_sets,
    _forget_value_info,
    _forward_across_jaxpr_boundary,
    _index_sets,
    _PropState,
    _report_issue,
    _seed_const_vals,
)


def _prop_custom_jvp_call(
    eqn: JaxprEqn,
    state: _PropState,
    _prop_jaxpr: PropJaxprFn,
) -> None:
    """A function with a user-defined JVP rule.

    JAX differentiates it with the rule, never with the primal function,
    so the index sets follow the rule's tangents.
    The primal can be far sparser than the rule,
    e.g. a straight-through estimator ``round(x)`` with JVP ``t``,
    or a bit-level ``rsqrt`` approximation whose rule is ``-y³/2 · t``.
    Both primals have a zero derivative,
    so following them would miss every nonzero of the Jacobian.

    The rule's tangent outputs are linear in its tangent inputs,
    with coefficients that depend on the primals.
    Seeding the tangents with the operands' index sets
    and the primals with empty ones
    therefore yields exactly the rule's Jacobian pattern.

    The primal function is still propagated for its values,
    so indices computed inside it stay resolvable downstream.

    A rule may apply its own function to the tangents,
    as rules of linear functions like permutations do.
    Following that call's rule again would unfold forever,
    so a rule already in progress is followed through its primal instead.
    The function is linear in that position,
    so the primal's pattern is assumed to be the rule's.

    Math:
        f(x) with rule (x, t) ↦ (f(x), Jᵣ(x) t)
        ∂f/∂x := Jᵣ(x), the rule's tangent map, not ∂(primal)/∂x

    Example: y = ste(x) with primal round(x) and rule (x, t) ↦ (round(x), t)
        Input index sets:  [{0}, {1}]
        Primal:            round has zero derivative → [{}, {}]
        Rule tangents:     t → [{0}, {1}]
        Output index sets: [{0}, {1}]

    Jaxpr:
        invars: [*closure_consts, *args] (first ``num_consts`` are closure consts)
        call_jaxpr: the primal function over all invars
        jvp_jaxpr_fun: traces the rule given one symbolic-zero flag per arg,
            returning ``(jaxpr, consts, out_zeros)``.
            The jaxpr's inputs are ``[*consts, *primals, *nonzero_tangents]``
            and its outputs are ``[*primals_out, *nonzero_tangents_out]``.

    https://docs.jax.dev/en/latest/_autosummary/jax.custom_jvp.html
    """
    jvp_jaxpr_fun = eqn.params["jvp_jaxpr_fun"]
    # Every level of a self-calling rule is traced afresh,
    # so the rule's source location is the only identity that repeats.
    rule_key = jvp_jaxpr_fun.debug_info.func_src_info
    if rule_key is not None and rule_key in state.custom_jvp_rules:
        in_indices = [_index_sets(state, v) for v in eqn.invars]
        out_indices = _prop_primal(eqn, state, _prop_jaxpr, in_indices)
        for outvar, indices in zip(eqn.outvars, out_indices, strict=True):
            state.indices[outvar] = indices
        return

    # Index sets come from the rule, so the primal only contributes values.
    empty_indices = [_empty_index_sets(_atom_numel(v)) for v in eqn.invars]
    _prop_primal(eqn, state, _prop_jaxpr, empty_indices)

    # The closure consts get no tangents in JAX's rule application,
    # so dependencies flowing through them are dropped, just as JAX drops them.
    args = eqn.invars[eqn.params["num_consts"] :]
    arg_indices = [_index_sets(state, arg) for arg in args]

    # A rule that calls its own function on the primals contains this equation again,
    # with operands that are seeded empty, so this also skips that nested call.
    if not any(any(indices) for indices in arg_indices):
        for outvar in eqn.outvars:
            state.indices[outvar] = _empty_index_sets(_atom_numel(outvar))
        return

    # Mark every tangent as nonzero, as JAX does when instantiating zeros
    # for rules that do not opt into symbolic zeros.
    jvp_jaxpr, jvp_consts, out_zeros = jvp_jaxpr_fun.call_wrapped(*[False] * len(args))
    _check_jvp_layout(eqn, jvp_jaxpr, jvp_consts, out_zeros, len(args))

    # Mirror how JAX binds the rule's consts positionally before its arguments.
    invars = jvp_jaxpr.invars
    num_rule_consts = len(jvp_consts)
    const_invars = invars[:num_rule_consts]
    primal_invars = invars[num_rule_consts : num_rule_consts + len(args)]
    tangent_invars = invars[num_rule_consts + len(args) :]

    # The rule's jaxpr is memoized, so its vars may hold value info from another call.
    _seed_const_vals(state, const_invars, jvp_consts)
    _forward_across_jaxpr_boundary(state, args, primal_invars)
    _forget_value_info(state, tangent_invars)

    input_indices: list[list[IndexSet]] = [
        *(_empty_index_sets(_atom_numel(v)) for v in const_invars),
        *(_empty_index_sets(_atom_numel(v)) for v in primal_invars),
        *arg_indices,
    ]
    if rule_key is not None:
        state.custom_jvp_rules.add(rule_key)
    try:
        output_indices = _prop_jaxpr(jvp_jaxpr, input_indices, state)
    finally:
        if rule_key is not None:
            state.custom_jvp_rules.discard(rule_key)
    nonzero_tangent_indices = iter(output_indices[len(eqn.outvars) :])

    for outvar, is_zero in zip(eqn.outvars, out_zeros, strict=True):
        if is_zero:
            state.indices[outvar] = _empty_index_sets(_atom_numel(outvar))
        else:
            state.indices[outvar] = next(nonzero_tangent_indices)


def _prop_primal(
    eqn: JaxprEqn,
    state: _PropState,
    _prop_jaxpr: PropJaxprFn,
    in_indices: list[list[IndexSet]],
) -> list[list[IndexSet]]:
    """Propagate through the primal function, forwarding its values out.

    Returns the primal's output index sets.
    """
    call_jaxpr = eqn.params["call_jaxpr"]
    _seed_const_vals(state, call_jaxpr.constvars, call_jaxpr.consts)
    _forward_across_jaxpr_boundary(state, eqn.invars, call_jaxpr.invars)
    out_indices = _prop_jaxpr(call_jaxpr, in_indices, state)
    _forward_across_jaxpr_boundary(state, call_jaxpr.outvars, eqn.outvars)
    return out_indices


def _check_jvp_layout(
    eqn: JaxprEqn,
    jvp_jaxpr: Jaxpr,
    jvp_consts: Sequence[Any],
    out_zeros: Sequence[bool],
    num_args: int,
) -> None:
    """Raise if the traced rule does not have the layout JAX documents in ``lift_jvp``.

    Reading tangents from the wrong positions would silently produce a wrong pattern.
    """
    num_outs = len(eqn.outvars)
    num_nonzero_outs = sum(not z for z in out_zeros)
    expected_ins = len(jvp_consts) + 2 * num_args
    expected_outs = num_outs + num_nonzero_outs
    if (
        jvp_jaxpr.constvars
        or len(jvp_jaxpr.invars) != expected_ins
        or len(out_zeros) != num_outs
        or len(jvp_jaxpr.outvars) != expected_outs
    ):
        msg = _report_issue(
            "Unexpected layout of the traced custom_jvp rule: "
            f"{len(jvp_jaxpr.invars)} inputs and {len(jvp_jaxpr.outvars)} outputs, "
            f"expected {expected_ins} inputs and {expected_outs} outputs."
        )
        raise ValueError(msg)
