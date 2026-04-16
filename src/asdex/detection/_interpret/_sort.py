"""Propagation rule for sort.

Sorting along one dimension mixes elements within slices along that dimension,
but not across other dimensions.
"""

import numpy as np
from jax._src.core import JaxprEqn

from ._commons import (
    IndexSet,
    StateIndices,
    atom_numel,
    atom_shape,
    empty_index_sets,
    index_sets,
    union_all,
)


def prop_sort(eqn: JaxprEqn, state_indices: StateIndices) -> None:
    """Sort reorders elements along one dimension.

    Each output element depends on all input elements in its slice
    along the sort dimension,
    since any input could end up at any position after sorting.

    For input shape (*batch, n) sorted along the last axis:
        out[*b, j] depends on all in[*b, :]

    For multiple operands (multi-key sort via ``lax.sort``),
    the permutation is determined by all key inputs jointly,
    so state_indices from all operands in the same slice are unioned.

    Example: y = sort(x, dimension=1) where x.shape = (2, 3)
        Input state_indices:  [{0}, {1}, {2}, {3}, {4}, {5}]
        Output state_indices: [{0, 1, 2}, {0, 1, 2}, {0, 1, 2},
                      {3, 4, 5}, {3, 4, 5}, {3, 4, 5}]

    Jaxpr:
        invars: one or more operand arrays (all same shape)
        dimension: axis along which to sort
        is_stable: whether sort is stable (irrelevant for sparsity)
        num_keys: number of key operands (irrelevant for sparsity)

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.sort.html
    """
    dim = eqn.params["dimension"]
    in_shape = atom_shape(eqn.invars[0])
    total = atom_numel(eqn.invars[0])

    if total == 0:
        for outvar in eqn.outvars:
            state_indices[outvar] = []
        return

    # groups[b] holds the flat indices for batch slice b.
    groups = np.moveaxis(np.arange(total).reshape(in_shape), dim, -1).reshape(
        -1, in_shape[dim]
    )

    # Union state_indices from all operands within each batch slice.
    all_indices = [index_sets(state_indices, v) for v in eqn.invars]
    group_indices = [
        union_all([ids[i] for ids in all_indices for i in g]) for g in groups
    ]

    # Broadcast each slice's state_indices back to every element in the slice.
    for outvar in eqn.outvars:
        out: list[IndexSet] = empty_index_sets(total)
        for gd, g in zip(group_indices, groups, strict=True):
            for i in g:
                out[i] = gd
        state_indices[outvar] = out
