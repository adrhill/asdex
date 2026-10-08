"""Propagation rule for concatenate operations."""

import numpy as np
from jax._src.core import JaxprEqn

from ._common import _join_inputs, _PropState


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
