"""Propagation rule for cond (conditional branching)."""

from jax._src.core import JaxprEqn

from ._common import (
    IndexSet,
    PropJaxprFn,
    _copy_index_sets,
    _forward_into_jaxpr,
    _index_sets,
    _PropState,
    _seed_const_vals,
)


def _prop_cond(
    eqn: JaxprEqn,
    state: _PropState,
    _prop_jaxpr: PropJaxprFn,
) -> None:
    """cond/switch selects one of several branches based on an integer index.

    Since we don't know which branch executes at trace time,
    output index sets are the union across all branches.

    Layout:
        invars: [index_scalar, operands...]
        outvars: [results...]
        params: branches (tuple of ClosedJaxpr)

    Example: cond(pred, true_fn, false_fn, x)
        true_fn:  out = x[:2]  → index sets [{0}, {1}]
        false_fn: out = x[1:]  → index sets [{1}, {2}]
        union:    [{0, 1}, {1, 2}]

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.cond.html
    """
    branches = eqn.params["branches"]
    operands = eqn.invars[1:]
    operand_indices: list[list[IndexSet]] = [_index_sets(state, v) for v in operands]

    n_out = len(eqn.outvars)

    # Propagate each branch and collect per-branch output index sets
    branch_outputs: list[list[list[IndexSet]]] = []
    for branch in branches:
        _seed_const_vals(state, branch.jaxpr.constvars, branch.consts)
        _forward_into_jaxpr(state, operands, branch.jaxpr.invars)
        out = _prop_jaxpr(branch.jaxpr, operand_indices, state)
        branch_outputs.append(out)

    # Union across branches for each output variable
    for i in range(n_out):
        outvar = eqn.outvars[i]
        # Start from first branch, union with the rest
        merged: list[IndexSet] = _copy_index_sets(branch_outputs[0][i])
        for branch_out in branch_outputs[1:]:
            for j in range(len(merged)):
                merged[j] |= branch_out[i][j]
        state.indices[outvar] = merged
