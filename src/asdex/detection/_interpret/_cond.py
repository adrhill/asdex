"""Propagation rule for cond (conditional branching)."""

from jax._src.core import JaxprEqn

from ._common import (
    IndexSet,
    IndexSetSequence,
    PropJaxprFn,
    StateConsts,
    StateIndices,
    _copy_index_sets,
    _forward_const_vals,
    _index_sets,
    _seed_const_vals,
)


def _prop_cond(
    eqn: JaxprEqn,
    state_indices: StateIndices,
    state_consts: StateConsts,
    _prop_jaxpr: PropJaxprFn,
) -> None:
    """cond/switch selects one of several branches based on an integer index.

    Since we don't know which branch executes at trace time,
    output state_indices are the union across all branches.

    Layout:
        invars: [index_scalar, operands...]
        outvars: [results...]
        params: branches (tuple of ClosedJaxpr)

    Example: cond(pred, true_fn, false_fn, x)
        true_fn:  out = x[:2]  → state_indices [{0}, {1}]
        false_fn: out = x[1:]  → state_indices [{1}, {2}]
        union:    [{0, 1}, {1, 2}]

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html
    """
    branches = eqn.params["branches"]
    operands = eqn.invars[1:]
    operand_indices: list[IndexSetSequence] = [
        _index_sets(state_indices, v) for v in operands
    ]

    n_out = len(eqn.outvars)

    # Propagate each branch and collect per-branch output state_indices
    branch_outputs: list[list[IndexSetSequence]] = []
    for branch in branches:
        _seed_const_vals(state_consts, branch.jaxpr.constvars, branch.consts)
        _forward_const_vals(state_consts, operands, branch.jaxpr.invars)
        out = _prop_jaxpr(branch.jaxpr, operand_indices, state_consts)
        branch_outputs.append(out)

    # Union across branches for each output variable
    for i in range(n_out):
        outvar = eqn.outvars[i]
        # Start from first branch, union with the rest
        merged: list[IndexSet] = _copy_index_sets(branch_outputs[0][i])
        for branch_out in branch_outputs[1:]:
            for j in range(len(merged)):
                merged[j].update(branch_out[i][j])
        state_indices[outvar] = merged
