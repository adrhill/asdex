"""Propagation rules for comparison primitives (lt, le, gt, ge).

Comparisons are piecewise constant (zero derivative).
When value bounds prove the result is always True or always False,
the result is stored as a const value
so that ``select_n`` can pick the correct branch.
"""

import numpy as np
from jax._src.core import JaxprEqn

from ._common import (
    _atom_numel,
    _atom_shape,
    _atom_value_bounds,
    _empty_index_sets,
    _propagate_const_binary,
    _PropState,
)


def _get_bounds(
    eqn: JaxprEqn,
    state: _PropState,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Get (lo1, hi1, lo2, hi2) for both inputs, or None if unavailable."""
    if eqn.outvars[0] in state.consts:
        return None
    b1 = _atom_value_bounds(eqn.invars[0], state)
    b2 = _atom_value_bounds(eqn.invars[1], state)
    if b1 is None or b2 is None:
        return None
    return (*b1, *b2)


def _set_const(eqn: JaxprEqn, state: _PropState, value: bool) -> None:
    """Store a constant boolean result."""
    state.consts[eqn.outvars[0]] = np.full(_atom_shape(eqn.outvars[0]), value)


def _prop_comparison(eqn: JaxprEqn, state: _PropState, ufunc: np.ufunc) -> None:
    """Shared implementation for lt, le, gt, and ge.

    The output has empty index sets (zero derivative).
    When the input bounds separate, the result is stored as a const boolean:
    the comparison is always true when it holds between the least favorable extremes,
    and always false when it fails between the most favorable extremes.
    For ``<`` and ``<=`` the least favorable pair is ``(hi(a), lo(b))``,
    for ``>`` and ``>=`` it is ``(lo(a), hi(b))``.
    """
    state.indices[eqn.outvars[0]] = _empty_index_sets(_atom_numel(eqn.outvars[0]))
    _propagate_const_binary(eqn, state, ufunc)
    bounds = _get_bounds(eqn, state)
    if bounds is None:
        return
    lo1, hi1, lo2, hi2 = bounds
    match ufunc:
        case np.less | np.less_equal:
            worst, best = (hi1, lo2), (lo1, hi2)
        case np.greater | np.greater_equal:
            worst, best = (lo1, hi2), (hi1, lo2)
        case _:
            msg = f"Unsupported comparison ufunc: {ufunc}"
            raise ValueError(msg)
    if np.all(ufunc(*worst)):
        _set_const(eqn, state, True)
    elif not np.any(ufunc(*best)):
        _set_const(eqn, state, False)


def _prop_lt(eqn: JaxprEqn, state: _PropState) -> None:
    """Less-than comparison with bounds resolution.

    Always true when ``hi(a) < lo(b)``.
    Always false when ``lo(a) >= hi(b)``.
    """
    _prop_comparison(eqn, state, np.less)


def _prop_le(eqn: JaxprEqn, state: _PropState) -> None:
    """Less-or-equal comparison with bounds resolution.

    Always true when ``hi(a) <= lo(b)``.
    Always false when ``lo(a) > hi(b)``.
    """
    _prop_comparison(eqn, state, np.less_equal)


def _prop_gt(eqn: JaxprEqn, state: _PropState) -> None:
    """Greater-than comparison with bounds resolution.

    Always true when ``lo(a) > hi(b)``.
    Always false when ``hi(a) <= lo(b)``.
    """
    _prop_comparison(eqn, state, np.greater)


def _prop_ge(eqn: JaxprEqn, state: _PropState) -> None:
    """Greater-or-equal comparison with bounds resolution.

    Always true when ``lo(a) >= hi(b)``.
    Always false when ``hi(a) < lo(b)``.
    """
    _prop_comparison(eqn, state, np.greater_equal)
