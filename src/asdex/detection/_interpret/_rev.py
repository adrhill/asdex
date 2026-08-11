"""Propagation rule for rev (reverse) operations."""

import numpy as np
from jax._src.core import JaxprEqn

from ._common import (
    _atom_shape,
    _index_sets,
    _PropState,
    _transform_indices,
)


def _prop_rev(eqn: JaxprEqn, state: _PropState) -> None:
    """Rev reverses an array along specified dimensions.

    Each output element maps to exactly one input element
    by flipping coordinates along the reversed dimensions.
    The Jacobian is a permutation matrix.

    For dimensions d in reversed_dims,
    output[..., i_d, ...] = input[..., (shape[d]-1-i_d), ...].

    Example: x = [a, b, c], rev(x, dimensions=[0])
        Input index sets:  [{0}, {1}, {2}]
        Output index sets: [{2}, {1}, {0}]

    Jaxpr:
        invars[0]: input array
        dimensions: sequence of ints specifying which axes to reverse

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.rev.html
    """
    in_indices = _index_sets(state, eqn.invars[0])
    in_shape = _atom_shape(eqn.invars[0])
    dimensions = eqn.params["dimensions"]

    state.indices[eqn.outvars[0]] = _transform_indices(
        in_indices, in_shape, lambda p: np.flip(p, axis=dimensions)
    )
