"""Propagation rule for stack operations."""

import numpy as np
from jax._src.core import JaxprEqn

from ._common import _PropState
from ._concatenate import _join_inputs


def _prop_stack(eqn: JaxprEqn, state: _PropState) -> None:
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
    # np.stack requires at least one array, so guard the degenerate case.
    if not eqn.invars:
        state.indices[eqn.outvars[0]] = []
        return

    _join_inputs(eqn, state, np.stack, eqn.params["axis"])
