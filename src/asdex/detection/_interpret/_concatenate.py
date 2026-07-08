"""Propagation rule for concatenate operations.

Also hosts the shared join core used by ``_stack.py``,
since stack and concatenate differ only in the numpy op that mirrors them.
"""

from collections.abc import Callable

import numpy as np
from jax._src.core import JaxprEqn

from ._common import (
    IndexSet,
    _atom_const_val,
    _atom_shape,
    _index_sets,
    _PropState,
)


def _join_inputs(
    eqn: JaxprEqn,
    state: _PropState,
    join: Callable[[list[np.ndarray], int], np.ndarray],
    axis: int,
) -> None:
    """Shared core for stack and concatenate.

    Pools every input's flat index sets into one list.
    For each input, builds a shaped array whose values are positions in that pool.
    Applying ``join`` to these index arrays mirrors the real op's element shuffling,
    giving a flat mapping from each output element to the pool position it came from.
    Also joins const values so downstream gather/scatter can resolve indices.
    """
    all_indices: list[IndexSet] = []
    index_arrays = []
    for invar in eqn.invars:
        in_indices = _index_sets(state, invar)
        offset = len(all_indices)
        all_indices.extend(in_indices)
        shape = _atom_shape(invar)
        index_arrays.append(np.arange(offset, offset + len(in_indices)).reshape(shape))

    permutation_map = join(index_arrays, axis).ravel()
    state.indices[eqn.outvars[0]] = [all_indices[i] for i in permutation_map]

    vals = [_atom_const_val(v, state) for v in eqn.invars]
    if all(v is not None for v in vals):
        state.consts[eqn.outvars[0]] = join([v for v in vals if v is not None], axis)


def _prop_concatenate(eqn: JaxprEqn, state: _PropState) -> None:
    """Concatenate joins arrays along a specified axis.

    Each output element comes from exactly one input element.

    For concat([A, B], axis=0): output = [A; B] (vertical stack).
    For concat([A, B], axis=1): output = [A | B] (horizontal stack).
    The Jacobian is a permuted identity matrix.

    Example: concat([[a,b], [c,d]], axis=0) → [a,b,c,d]
        Input index sets:  [{0}, {1}], [{2}, {3}]
        Output index sets: [{0}, {1}, {2}, {3}]

    Jaxpr:
        invars: list of input arrays to concatenate
        dimension: axis along which to concatenate

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.concatenate.html
    """
    _join_inputs(eqn, state, np.concatenate, eqn.params["dimension"])
