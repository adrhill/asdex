"""Propagation rule for stack operations."""

import numpy as np
from jax._src.core import JaxprEqn

from ._commons import (
    IndexSet,
    StateConsts,
    StateIndices,
    atom_const_val,
    atom_shape,
    index_sets,
    numel,
)


def prop_stack(
    eqn: JaxprEqn, state_indices: StateIndices, state_consts: StateConsts
) -> None:
    """Stack joins arrays along a new axis.

    Each output element comes from exactly one input element.
    For N inputs of shape S, stacking at axis d produces shape S[:d] + (N,) + S[d:].
    Output element [i_0, ..., i_{d-1}, k, i_d, ...] reads from input k at [i_0, ..., i_{d-1}, i_d, ...].

    The Jacobian is a permutation matrix.

    Example: stack([a, b], axis=0) where a, b have shape (2,)
        Input index sets:  [{0}, {1}], [{2}, {3}]
        Output shape: (2, 2)
        Output index sets: [{0}, {1}, {2}, {3}]
        (row 0 from a, row 1 from b)

    Jaxpr:
        invars: list of input arrays to stack (all same shape)
        axis: dimension along which to insert the new axis

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.stack.html
    """
    axis = eqn.params["axis"]
    out_var = eqn.outvars[0]

    if not eqn.invars:
        state_indices[out_var] = []
        return

    in_shape = atom_shape(eqn.invars[0])
    in_numel = numel(in_shape)
    n_inputs = len(eqn.invars)
    out_numel = in_numel * n_inputs

    if out_numel == 0:
        state_indices[out_var] = []
        return

    # Pool all input index sets into one list.
    # Build position arrays that map to positions in the pool.
    all_indices: list[IndexSet] = []
    index_arrays = []
    for invar in eqn.invars:
        in_indices = index_sets(state_indices, invar)
        offset = len(all_indices)
        all_indices.extend(in_indices)
        index_arrays.append(
            np.arange(offset, offset + len(in_indices)).reshape(in_shape)
        )

    # np.stack mirrors the primitive's semantics:
    # stacking along axis inserts a new dimension at that position.
    permutation_map = np.stack(index_arrays, axis=axis).ravel()
    state_indices[out_var] = [all_indices[i] for i in permutation_map]

    # Propagate const values for downstream gather/scatter.
    vals = [atom_const_val(v, state_consts) for v in eqn.invars]
    if all(v is not None for v in vals):
        state_consts[out_var] = np.stack([v for v in vals if v is not None], axis=axis)
