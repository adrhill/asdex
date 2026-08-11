"""Propagation rule for select_n operations."""

from collections.abc import Sequence

import numpy as np
from jax._src.core import JaxprEqn, Var

from ._common import (
    Atom,
    _atom_const_val,
    _atom_numel,
    _atom_shape,
    _atom_value_bounds,
    _index_sets,
    _PropState,
    _union_elementwise,
)


def _all_const_vals(
    atoms: Sequence[Atom], state: _PropState
) -> list[np.ndarray] | None:
    """Const values for every atom, or ``None`` as soon as one is unknown.

    Stops at the first unknown so a runtime-dependent case
    does not force materializing the remaining cases' consts.
    """
    vals: list[np.ndarray] = []
    for atom in atoms:
        val = _atom_const_val(atom, state)
        if val is None:
            return None
        vals.append(val)
    return vals


def _merged_case_bounds(
    atoms: Sequence[Atom], state: _PropState
) -> tuple[np.ndarray, np.ndarray] | None:
    """Element-wise ``(min lo, max hi)`` envelope of every atom's bounds.

    Returns ``None`` as soon as one atom has no bounds,
    since the envelope needs all of them.
    Stopping early keeps the remaining cases' consts unmaterialized.
    """
    merged: tuple[np.ndarray, np.ndarray] | None = None
    for atom in atoms:
        bounds = _atom_value_bounds(atom, state)
        if bounds is None:
            return None
        merged = (
            bounds
            if merged is None
            else (np.minimum(merged[0], bounds[0]), np.maximum(merged[1], bounds[1]))
        )
    return merged


def _prop_select_n(
    eqn: JaxprEqn,
    state: _PropState,
) -> None:
    """select_n(which, *cases) picks case values element-wise.

    ``which`` is a boolean or integer selector (scalar or array).
    All cases must have identical shapes.
    The selector has zero derivative,
    so only value-case index sets contribute to the sparsity pattern.

    Also propagates value bounds through the selected branch
    when the predicate is a known constant.

    Jaxpr:
        invars[0]: which (boolean or integer, scalar or array)
        invars[1:]: value cases (on_false, on_true, ...)

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.select_n.html
    """
    out_var = eqn.outvars[0]
    out_size = _atom_numel(out_var)
    cases = eqn.invars[1:]  # value cases (which is invars[0])

    case_indices = [_index_sets(state, c) for c in cases]

    # When the selector is a known constant,
    # each output element takes index sets from exactly one branch.
    which_atom = eqn.invars[0]
    which_val = _atom_const_val(which_atom, state)

    if which_val is not None:
        flat_which = (
            np.broadcast_to(which_val, _atom_shape(out_var)).ravel().astype(int)
        )
        out_indices = [case_indices[flat_which[i]][i] for i in range(out_size)]
    else:
        # Dynamic selector: union across all value cases.
        out_indices = _union_elementwise(case_indices, out_size)

    state.indices[out_var] = out_indices

    # When all inputs are statically known, compute the concrete result
    # so state.consts tracking isn't broken by this op.
    # A dynamic selector can never store a const result,
    # so the cases are only read once the selector is known.
    if which_val is not None:
        case_vals = _all_const_vals(cases, state)
        if case_vals is not None:
            state.consts[out_var] = np.choose(which_val, case_vals)

    # Const boolean predicate uniformly selects one branch → use its bounds exactly.
    # Only the selected branch is read.
    # Falling through to the merge when it has no bounds would be pointless:
    # the merge needs every branch's bounds, including this one,
    # so it would bail after materializing the other branches' consts.
    if which_val is not None and len(cases) == 2 and which_val.dtype == bool:
        if not np.any(which_val):
            _store_branch_bounds(state, out_var, cases[0])
            return
        if np.all(which_val):
            _store_branch_bounds(state, out_var, cases[1])
            return

    # Dynamic or mixed predicate → merge bounds across all branches.
    bounds = _merged_case_bounds(cases, state)
    if bounds is not None:
        state.bounds[out_var] = bounds


def _store_branch_bounds(state: _PropState, out_var: Var, case: Atom) -> None:
    """Store one branch's value bounds as the output's, if that branch has any."""
    bounds = _atom_value_bounds(case, state)
    if bounds is not None:
        state.bounds[out_var] = bounds
