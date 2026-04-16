"""Propagation rule for transpose operations."""

from functools import partial

import numpy as np
from jax._src.core import JaxprEqn

from ._commons import (
    StateConsts,
    StateIndices,
    atom_shape,
    index_sets,
    propagate_const_unary,
    transform_indices,
)


def prop_transpose(
    eqn: JaxprEqn, state_indices: StateIndices, state_consts: StateConsts
) -> None:
    """Transpose permutes the dimensions of an array.

    Each output element maps to exactly one input element
    via the inverse permutation of coordinates.
    The Jacobian is a permutation matrix.

    For permutation perm, output[i0, i1, ...] = input[inv(i0), inv(i1), ...],
    where inv_perm[perm[d]] = d.

    Example: x = [[a, b, c], [d, e, f]], transpose(x, (1, 0))
        Input state_indices:  [{0}, {1}, {2}, {3}, {4}, {5}]
        Output state_indices: [{0}, {3}, {1}, {4}, {2}, {5}]

    Jaxpr:
        invars[0]: input array
        permutation: tuple of ints specifying the dimension order

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.transpose.html
    """
    in_indices = index_sets(state_indices, eqn.invars[0])
    in_shape = atom_shape(eqn.invars[0])
    permutation = eqn.params["permutation"]

    state_indices[eqn.outvars[0]] = transform_indices(
        in_indices, in_shape, lambda p: p.transpose(permutation)
    )

    propagate_const_unary(eqn, state_consts, partial(np.transpose, axes=permutation))
