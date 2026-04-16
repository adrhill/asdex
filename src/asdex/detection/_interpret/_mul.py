"""Propagation rule for element-wise multiplication."""

import numpy as np
from jax._src.core import JaxprEqn

from ._commons import (
    StateBounds,
    StateConsts,
    StateIndices,
    atom_value_bounds,
    clear_where_zero,
    propagate_const_binary,
)
from ._elementwise import _binary_elementwise


def prop_mul(
    eqn: JaxprEqn,
    state_indices: StateIndices,
    state_consts: StateConsts,
    state_bounds: StateBounds,
) -> None:
    """Multiplication is element-wise with a special case for known zeros.

    Like other binary element-wise ops,
    each output depends on the corresponding elements from both inputs.
    However, since d(0 * y)/dy = 0,
    output positions where either operand is a known constant zero
    have no dependency on the inputs.

    Example: z = x * [0, 1] where x = [a, b]
        Input index sets: [{0}, {1}], [{}, {}]
        Output index sets: [{}, {1}]  (first cleared by known zero)

    Jaxpr:
        invars[0]: first input array
        invars[1]: second input array
    """
    _binary_elementwise(eqn, state_indices)
    propagate_const_binary(eqn, state_consts, np.multiply)
    clear_where_zero(eqn, state_indices, state_consts, 0)
    clear_where_zero(eqn, state_indices, state_consts, 1)
    _propagate_bounds_mul(eqn, state_consts, state_bounds)


def _propagate_bounds_mul(
    eqn: JaxprEqn,
    state_consts: StateConsts,
    state_bounds: StateBounds,
) -> None:
    """Propagate value bounds through ``mul`` via interval arithmetic.

    ``[a,b] * [c,d]`` → ``[min(ac,ad,bc,bd), max(ac,ad,bc,bd)]``.
    This handles all sign combinations correctly.
    """
    in1_bounds = atom_value_bounds(eqn.invars[0], state_consts, state_bounds)
    in2_bounds = atom_value_bounds(eqn.invars[1], state_consts, state_bounds)
    if in1_bounds is None or in2_bounds is None:
        return

    lo1, hi1 = in1_bounds
    lo2, hi2 = in2_bounds

    c1 = lo1 * lo2
    c2 = lo1 * hi2
    c3 = hi1 * lo2
    c4 = hi1 * hi2

    lo = np.minimum(np.minimum(c1, c2), np.minimum(c3, c4))
    hi = np.maximum(np.maximum(c1, c2), np.maximum(c3, c4))
    state_bounds[eqn.outvars[0]] = (lo, hi)
