"""Propagation rule for select_if_vmap (Equinox primitive)."""

import numpy as np
from jax._src.core import JaxprEqn

from .._common import (
    _atom_const_val,
    _atom_numel,
    _check_no_index_sets,
    _index_sets,
    _PropState,
)


def _prop_select_if_vmap(eqn: JaxprEqn, state: _PropState) -> None:
    """select_if_vmap(pred, on_true, on_false) picks values element-wise.

    Equinox emits this when vmapping ``lax.cond``.
    Both branches are traced and the result is selected element-wise,
    identical to ``select_n`` with two cases.
    The predicate has zero derivative,
    so only the branch values contribute to the sparsity pattern.

    Jaxpr:
        invars[0]: pred (boolean, scalar or array)
        invars[1]: on_true (value when pred is True)
        invars[2]: on_false (value when pred is False)

    https://github.com/patrick-kidger/equinox/blob/main/equinox/internal/_loop/common.py
    """
    _check_no_index_sets(state, eqn.invars[0], eqn.primitive.name)

    out_var = eqn.outvars[0]
    out_size = _atom_numel(out_var)
    on_true, on_false = eqn.invars[1], eqn.invars[2]
    true_indices = _index_sets(state, on_true)
    false_indices = _index_sets(state, on_false)

    state.indices[out_var] = [
        true_indices[i] | false_indices[i] for i in range(out_size)
    ]

    # Propagate concrete values when both branches are statically known.
    pred_val = _atom_const_val(eqn.invars[0], state)
    true_val = _atom_const_val(on_true, state)
    false_val = _atom_const_val(on_false, state)
    if pred_val is not None and true_val is not None and false_val is not None:
        state.consts[out_var] = np.where(pred_val, true_val, false_val)
