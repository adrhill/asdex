"""Propagation rule for tile operations."""

from functools import partial

import numpy as np
from jax._src.core import JaxprEqn

from ._commons import (
    StateConsts,
    StateIndices,
    atom_shape,
    index_sets,
    permute_indices,
    propagate_const_unary,
)


def prop_tile(
    eqn: JaxprEqn, state_indices: StateIndices, state_consts: StateConsts
) -> None:
    """Tile repeats an array along each dimension.

    Each output element depends on exactly one input element
    via modular indexing: ``out[i, j, ...] = in[i % s0, j % s1, ...]``.

    The Jacobian has exactly one 1 per row.

    Example: x = [a, b], tile(x, reps=(2,))
        Input state_indices:  [{0}, {1}]
        Output state_indices: [{0}, {1}, {0}, {1}]

    Jaxpr:
        invars[0]: input array
        reps: tuple of repetition counts per dimension

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.tile.html
    """
    in_indices = index_sets(state_indices, eqn.invars[0])
    in_shape = atom_shape(eqn.invars[0])
    reps = eqn.params["reps"]

    out_shape = tuple(s * r for s, r in zip(in_shape, reps, strict=True))

    # Build output coordinates and map back to input via modular arithmetic.
    out_coords = np.indices(out_shape)  # (ndim, *out_shape)
    in_coords = tuple(out_coords[d] % in_shape[d] for d in range(len(in_shape)))
    flat_map = np.ravel_multi_index(in_coords, in_shape).ravel()

    state_indices[eqn.outvars[0]] = permute_indices(in_indices, flat_map)

    propagate_const_unary(eqn, state_consts, partial(np.tile, reps=reps))
