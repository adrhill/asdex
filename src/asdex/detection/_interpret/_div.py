"""Propagation rule for element-wise division."""

import numpy as np
from jax._src.core import JaxprEqn

from ._common import (
    _binary_value_bounds,
    _clear_where_zero,
    _propagate_const_binary,
    _PropState,
)
from ._elementwise import _binary_elementwise, _lax_div


def _prop_div(
    eqn: JaxprEqn,
    state: _PropState,
) -> None:
    """Division is element-wise with a special case for known zero numerators.

    Like other binary element-wise ops,
    each output depends on the corresponding elements from both inputs.
    However, since d(0 / y)/dy = 0,
    output positions where the numerator is a known constant zero
    have no dependency on the inputs.

    Example: z = [0, x] / [y, y]
        Input index sets: [{}, {1}], [{2}, {3}]
        Output index sets: [{}, {1, 3}]  (first cleared by known zero numerator)

    Jaxpr:
        invars[0]: numerator
        invars[1]: denominator
    """
    _binary_elementwise(eqn, state)
    _propagate_const_binary(eqn, state, _lax_div)
    _clear_where_zero(eqn, state, 0)
    _propagate_bounds_div(eqn, state)


def _propagate_bounds_div(
    eqn: JaxprEqn,
    state: _PropState,
) -> None:
    """Propagate value bounds through ``div`` via interval arithmetic.

    Only propagates when divisor bounds have constant sign (no zero crossing),
    since division by an interval spanning zero is undefined.
    Integer division matches ``lax.div``, which truncates toward zero.
    Flooring instead would exclude the value the program actually computes
    for negative intervals, and bounded enumeration would never try it.
    """
    bounds = _binary_value_bounds(eqn, state)
    if bounds is None:
        return

    (lo1, hi1), (lo2, hi2) = bounds

    # Skip if divisor bounds span zero.
    if not (np.all(lo2 > 0) or np.all(hi2 < 0)):
        return

    out_dtype = getattr(eqn.outvars[0].aval, "dtype", np.float64)
    divide = _lax_div if np.issubdtype(out_dtype, np.integer) else np.true_divide

    # All four endpoint combinations.
    c1 = divide(lo1, lo2)
    c2 = divide(lo1, hi2)
    c3 = divide(hi1, lo2)
    c4 = divide(hi1, hi2)

    lo = np.minimum(np.minimum(c1, c2), np.minimum(c3, c4))
    hi = np.maximum(np.maximum(c1, c2), np.maximum(c3, c4))
    state.bounds[eqn.outvars[0]] = (lo, hi)
