"""Propagation rule for broadcast_in_dim."""

import numpy as np
from jax._src.core import JaxprEqn

from ._common import (
    _atom_const_val,
    _atom_shape,
    _atom_value_bounds,
    _index_sets,
    _numel,
    _permute_indices,
    _PropState,
)


def _prop_broadcast_in_dim(
    eqn: JaxprEqn,
    state: _PropState,
) -> None:
    """Broadcast replicates input elements across new or expanded dimensions.

    Each output element depends on exactly one input element,
    determined by projecting output coordinates onto input dimensions.

    For broadcast_dimensions mapping input dim i → output dim d[i]:
        out[..., j, ...] = in[..., j mod in_shape[i], ...]
    Size-1 input dims are implicitly broadcast (all outputs read index 0).

    Also tracks const values: if input is a Literal or known const,
    the output value is also recorded for use in gather/scatter handlers.

    Example: x.shape = (3,), y = broadcast(x, shape=(2, 3), dims=(1,))
        Input index sets:  [{0}, {1}, {2}]
        Output index sets: [{0}, {1}, {2}, {0}, {1}, {2}]  (repeated per row)

    Jaxpr:
        invars[0]: input array
        shape: target output shape
        broadcast_dimensions: maps input dim i to output dim

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.broadcast_in_dim.html
    """
    in_atom = eqn.invars[0]
    in_indices = _index_sets(state, in_atom)
    out_shape = eqn.params["shape"]
    broadcast_dims = eqn.params["broadcast_dimensions"]
    out_var = eqn.outvars[0]

    # Gather/scatter handlers need concrete index arrays to resolve which input elements are accessed.
    # When the broadcast input is statically known (literal or traced from constants),
    # propagate its value so downstream handlers can use it
    # instead of falling back to conservative all-to-all dependencies.
    in_val = _atom_const_val(in_atom, state)
    if in_val is not None:
        intermediate_shape = [1] * len(out_shape)
        for i, out_dim in enumerate(broadcast_dims):
            intermediate_shape[out_dim] = (in_val.shape or (1,))[i]
        state.consts[out_var] = np.broadcast_to(
            np.reshape(in_val, intermediate_shape), out_shape
        )

    # Propagate value bounds by broadcasting to the output shape.
    _propagate_bounds_broadcast(eqn, state)

    out_size = _numel(out_shape)
    if out_size == 0:
        state.indices[out_var] = []
        return

    # Scalars have a single dependency set shared by all output elements,
    # so we can skip the coordinate mapping below and just replicate it.
    # Early return avoids building the np.indices grid for this common case.
    if len(in_indices) == 1:
        state.indices[out_var] = [in_indices[0]] * out_size
        return

    # General case: map each output element back to the input element it reads.
    # np.indices gives all output coordinates.
    # We select the output dim corresponding to each input dim via broadcast_dims.
    # Size-1 input dims are broadcast (every output reads index 0), so we clamp to 0.
    in_shape = _atom_shape(in_atom)
    out_coords = np.indices(out_shape)
    in_coords = tuple(
        out_coords[broadcast_dims[i]] if in_shape[i] > 1 else 0
        for i in range(len(in_shape))
    )
    flat_map = np.ravel_multi_index(in_coords, in_shape).ravel()

    state.indices[out_var] = _permute_indices(in_indices, flat_map)


def _propagate_bounds_broadcast(eqn: JaxprEqn, state: _PropState) -> None:
    """Propagate value bounds through broadcast_in_dim.

    Broadcasting replicates values without changing them,
    so bounds are broadcast to the output shape.
    """
    bounds = _atom_value_bounds(eqn.invars[0], state)
    if bounds is None:
        return
    lo, hi = bounds
    out_shape = eqn.params["shape"]
    broadcast_dims = eqn.params["broadcast_dimensions"]
    in_shape = lo.shape or (1,)

    intermediate_shape = [1] * len(out_shape)
    for i, out_dim in enumerate(broadcast_dims):
        intermediate_shape[out_dim] = in_shape[i]

    state.bounds[eqn.outvars[0]] = (
        np.broadcast_to(np.reshape(lo, intermediate_shape), out_shape),
        np.broadcast_to(np.reshape(hi, intermediate_shape), out_shape),
    )
