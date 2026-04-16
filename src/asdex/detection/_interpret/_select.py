"""Propagation rule for select_n operations."""

import numpy as np
from jax._src.core import JaxprEqn

from ._commons import (
    IndexSet,
    StateBounds,
    StateConsts,
    StateIndices,
    atom_const_val,
    atom_numel,
    atom_shape,
    atom_value_bounds,
    empty_index_set,
    index_sets,
)


def prop_select_n(
    eqn: JaxprEqn,
    state_indices: StateIndices,
    state_consts: StateConsts,
    state_bounds: StateBounds | None = None,
) -> None:
    """select_n(which, *cases) picks case values element-wise.

    ``which`` is a boolean or integer selector (scalar or array).
    All cases must have identical shapes.
    The selector has zero derivative,
    so only value-case state_indices contribute to the sparsity pattern.

    Also propagates value bounds through the selected branch
    when the predicate is a known constant.

    Jaxpr:
        invars[0]: which (boolean or integer, scalar or array)
        invars[1:]: value cases (on_false, on_true, ...)

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.select_n.html
    """
    out_var = eqn.outvars[0]
    out_size = atom_numel(out_var)
    cases = eqn.invars[1:]  # value cases (which is invars[0])

    case_indices = [index_sets(state_indices, c) for c in cases]

    # When the selector is a known constant,
    # each output element takes index sets from exactly one branch.
    which_atom = eqn.invars[0]
    which_val = atom_const_val(which_atom, state_consts)

    if which_val is not None:
        flat_which = np.broadcast_to(which_val, atom_shape(out_var)).ravel().astype(int)
        out_indices = [case_indices[flat_which[i]][i] for i in range(out_size)]
    else:
        # Dynamic selector: union across all value cases.
        out_indices = []
        for i in range(out_size):
            merged: IndexSet = empty_index_set()
            for c_idx in case_indices:
                merged |= c_idx[i]
            out_indices.append(merged)

    state_indices[out_var] = out_indices

    # When all inputs are statically known, compute the concrete result
    # so state_consts tracking isn't broken by this op.
    case_vals = [atom_const_val(c, state_consts) for c in cases]
    if which_val is not None and all(v is not None for v in case_vals):
        state_consts[out_var] = np.choose(
            which_val, [v for v in case_vals if v is not None]
        )

    # Propagate value bounds.
    if state_bounds is not None:
        case_bounds = [atom_value_bounds(c, state_consts, state_bounds) for c in cases]

        # Const predicate uniformly selects one branch → use its bounds exactly.
        if which_val is not None and len(cases) == 2 and which_val.dtype == bool:
            if not np.any(which_val) and case_bounds[0] is not None:
                state_bounds[out_var] = case_bounds[0]
                return
            if np.all(which_val) and case_bounds[1] is not None:
                state_bounds[out_var] = case_bounds[1]
                return

        # Dynamic or mixed predicate → merge bounds across all branches.
        if all(b is not None for b in case_bounds):
            los, his = zip(*(b for b in case_bounds if b is not None), strict=True)
            lo = los[0]
            hi = his[0]
            for lo_i, hi_i in zip(los[1:], his[1:], strict=True):
                lo = np.minimum(lo, lo_i)
                hi = np.maximum(hi, hi_i)
            state_bounds[out_var] = (lo, hi)
