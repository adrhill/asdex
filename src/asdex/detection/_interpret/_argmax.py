"""Propagation rule for argmax and argmin."""

import numpy as np
from jax._src.core import JaxprEqn

from ._commons import StateBounds, StateIndices, atom_shape
from ._elementwise import prop_zero_derivative


def prop_argmax(
    eqn: JaxprEqn, state_indices: StateIndices, state_bounds: StateBounds
) -> None:
    """argmax/argmin returns the index of the extreme value along an axis.

    The output has zero derivative with respect to the input
    (the index is piecewise constant).
    The output value is bounded: it lies in ``[0, axis_size - 1]``.
    These bounds are stored in ``state_bounds``
    so that downstream ``gather``/``scatter``/``dynamic_slice`` handlers
    can enumerate possible index values
    instead of falling back to conservative.

    Example: y = argmax([a, b, c]) → y ∈ {0, 1, 2}
        Input index sets:  [{0}, {1}, {2}]
        Output index sets: [{}, {}, {}]  (zero derivative)
        Value bounds: lo=0, hi=2

    Jaxpr:
        invars[0]: input array
        axes: tuple of reduction axes

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.argmax.html
    """
    prop_zero_derivative(eqn, state_indices)

    axes = eqn.params["axes"]
    in_shape = atom_shape(eqn.invars[0])
    axis_size = in_shape[axes[0]]
    out_var = eqn.outvars[0]
    lo = np.zeros(atom_shape(out_var), dtype=np.int64)
    hi = np.full(atom_shape(out_var), axis_size - 1, dtype=np.int64)
    state_bounds[out_var] = (lo, hi)
