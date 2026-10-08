"""Propagation rule for unstack operations."""

import numpy as np
from jax._src.core import JaxprEqn

from ._common import (
    _atom_const_val,
    _atom_shape,
    _index_sets,
    _numel,
    _permute_indices,
    _PropState,
)


def _prop_unstack(eqn: JaxprEqn, state: _PropState) -> None:
    """Unstack splits an array along an axis into multiple sub-arrays.

    Each output element maps to exactly one input element,
    so dependencies pass through unchanged.
    Output k contains the slice at position k along the unstacked axis,
    with that axis removed from the shape.

    The Jacobian is a permutation of rows of the identity matrix.

    Example: x has shape (2, 3), unstack(x, axis=0)
        Input index sets:  [{0}, {1}, {2}, {3}, {4}, {5}]
        Output 0 shape: (3,), index sets: [{0}, {1}, {2}]
        Output 1 shape: (3,), index sets: [{3}, {4}, {5}]

    Jaxpr:
        invars[0]: input array
        axis: dimension along which to unstack (default 0)

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.unstack.html
    """
    in_indices = _index_sets(state, eqn.invars[0])
    in_shape = _atom_shape(eqn.invars[0])
    axis = eqn.params.get("axis", 0)

    if not eqn.outvars:
        return

    out_numel = _numel(in_shape) // len(eqn.outvars) if eqn.outvars else 0
    if out_numel == 0:
        for out_var in eqn.outvars:
            state.indices[out_var] = []
        return

    # Build a position map and use np.moveaxis to reorder so the unstacked axis is first.
    # Then each output k gets the k-th slice along that leading dimension.
    pos_map = np.arange(_numel(in_shape)).reshape(in_shape)
    pos_map = np.moveaxis(pos_map, axis, 0)

    for k, out_var in enumerate(eqn.outvars):
        flat_map = pos_map[k].ravel()
        state.indices[out_var] = _permute_indices(in_indices, flat_map)

    # Propagate const values for downstream gather/scatter.
    in_val = _atom_const_val(eqn.invars[0], state)
    if in_val is not None:
        slices = np.moveaxis(in_val, axis, 0)
        for k, out_var in enumerate(eqn.outvars):
            state.consts[out_var] = slices[k]
