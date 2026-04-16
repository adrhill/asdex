"""Propagation rule for reduce_sum, reduce_max, reduce_min, and reduce_prod.

All four share the same sparsity structure:
each output depends on all inputs along the reduced axes.
"""

import numpy as np
from jax._src.core import JaxprEqn

from ._commons import (
    IndexSet,
    StateIndices,
    atom_shape,
    empty_index_sets,
    index_sets,
    numel,
    union_all,
)


def prop_reduce(eqn: JaxprEqn, state_indices: StateIndices) -> None:
    """Reduction aggregates elements along specified axes.

    Each output depends on all input elements that reduce into it.
    This applies uniformly to reduce_sum, reduce_max, reduce_min, and reduce_prod:
    in all cases, changing any input in a reduction group
    can change the output.

    Full reduction (no axes or all axes):
        out = reduce(x)  →  out depends on all inputs
    Partial reduction along axis k:
        out[i] = reduce_j x[i, j]  →  out[i] depends on all x[i, :]

    Example: y = sum(x, axis=1) where x.shape = (2, 3)
        Input state_indices:  [{0}, {1}, {2}, {3}, {4}, {5}]
        Output state_indices: [{0, 1, 2}, {3, 4, 5}]  (one set per row)

    Jaxpr:
        invars[0]: input array
        axes: tuple of axes to reduce (empty = full reduction)

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce_sum.html
    https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce_max.html
    https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce_min.html
    https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce_prod.html
    """
    in_indices = index_sets(state_indices, eqn.invars[0])
    axes = eqn.params.get("axes", ())
    in_shape = atom_shape(eqn.invars[0])

    # Full reduction: single output depends on all inputs
    if not axes or len(axes) == len(in_shape):
        state_indices[eqn.outvars[0]] = [union_all(in_indices)]
        return

    # Partial reduction: group input elements by their non-reduced coordinates.
    # Build a flat map from each input element to its output group,
    # then union input state_indices into the corresponding output set.
    kept_dims = [d for d in range(len(in_shape)) if d not in axes]
    out_shape = tuple(in_shape[d] for d in kept_dims)
    out_size = numel(out_shape)

    # For each input element, project to output coordinates (drop reduced dims)
    in_coords = np.indices(in_shape)
    out_coords = tuple(in_coords[d] for d in kept_dims)
    group_map = np.ravel_multi_index(out_coords, out_shape).ravel()

    out_indices: list[IndexSet] = empty_index_sets(out_size)
    for in_flat, elem_deps in enumerate(in_indices):
        out_indices[group_map[in_flat]] |= elem_deps

    state_indices[eqn.outvars[0]] = out_indices
