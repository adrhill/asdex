"""Propagation rule for concatenate operations."""

import numpy as np
from jax._src.core import JaxprEqn

from ._commons import (
    IndexSet,
    StateConsts,
    StateIndices,
    atom_const_val,
    atom_shape,
    index_sets,
)


def prop_concatenate(
    eqn: JaxprEqn, state_indices: StateIndices, state_consts: StateConsts
) -> None:
    """Concatenate joins arrays along a specified axis.

    Each output element comes from exactly one input element.

    For concat([A, B], axis=0): output = [A; B] (vertical stack).
    For concat([A, B], axis=1): output = [A | B] (horizontal stack).
    The Jacobian is a permuted identity matrix.

    Example: concat([[a,b], [c,d]], axis=0) → [a,b,c,d]
        Input state_indices:  [{0}, {1}], [{2}, {3}]
        Output state_indices: [{0}, {1}, {2}, {3}]

    Jaxpr:
        invars: list of input arrays to concatenate
        dimension: axis along which to concatenate

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.concatenate.html
    """
    dim = eqn.params["dimension"]

    # Pool every input's flat state_indices into one list.
    # For each input, build a shaped array whose values are positions in that pool.
    # np.concatenate on these index arrays mirrors the real op's element shuffling,
    # giving a flat mapping from each output element to the pool position it came from.
    all_indices: list[IndexSet] = []
    index_arrays = []
    for invar in eqn.invars:
        in_indices = index_sets(state_indices, invar)
        offset = len(all_indices)
        all_indices.extend(in_indices)
        shape = atom_shape(invar)
        index_arrays.append(np.arange(offset, offset + len(in_indices)).reshape(shape))

    permutation_map = np.concatenate(index_arrays, axis=dim).ravel()
    state_indices[eqn.outvars[0]] = [all_indices[i] for i in permutation_map]

    # Propagate state_consts so downstream gather/scatter can resolve indices.
    vals = [atom_const_val(v, state_consts) for v in eqn.invars]
    if all(v is not None for v in vals):
        state_consts[eqn.outvars[0]] = np.concatenate(
            [v for v in vals if v is not None], axis=dim
        )
