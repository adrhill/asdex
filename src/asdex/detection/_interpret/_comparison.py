"""Propagation rules for comparison primitives (lt, le, gt, ge).

Comparisons are piecewise constant (zero derivative).
When value bounds prove the result is always True or always False,
the result is stored as a const value
so that ``select_n`` can pick the correct branch.
"""

import numpy as np
from jax._src.core import JaxprEqn

from ._commons import (
    StateBounds,
    StateConsts,
    StateIndices,
    atom_numel,
    atom_shape,
    atom_value_bounds,
    empty_index_sets,
    propagate_const_binary,
)


def _get_bounds(
    eqn: JaxprEqn,
    state_consts: StateConsts,
    state_bounds: StateBounds,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """Get (lo1, hi1, lo2, hi2) for both inputs, or None if unavailable."""
    if eqn.outvars[0] in state_consts:
        return None
    b1 = atom_value_bounds(eqn.invars[0], state_consts, state_bounds)
    b2 = atom_value_bounds(eqn.invars[1], state_consts, state_bounds)
    if b1 is None or b2 is None:
        return None
    return (*b1, *b2)


def _set_const(eqn: JaxprEqn, state_consts: StateConsts, value: bool) -> None:
    """Store a constant boolean result."""
    state_consts[eqn.outvars[0]] = np.full(atom_shape(eqn.outvars[0]), value)


def prop_lt(
    eqn: JaxprEqn,
    state_indices: StateIndices,
    state_consts: StateConsts,
    state_bounds: StateBounds,
) -> None:
    """Less-than comparison with bounds resolution.

    Always true when ``hi(a) < lo(b)``.
    Always false when ``lo(a) >= hi(b)``.
    """
    state_indices[eqn.outvars[0]] = empty_index_sets(atom_numel(eqn.outvars[0]))
    propagate_const_binary(eqn, state_consts, np.less)
    bounds = _get_bounds(eqn, state_consts, state_bounds)
    if bounds is not None:
        lo1, hi1, lo2, hi2 = bounds
        if np.all(hi1 < lo2):
            _set_const(eqn, state_consts, True)
        elif np.all(lo1 >= hi2):
            _set_const(eqn, state_consts, False)


def prop_le(
    eqn: JaxprEqn,
    state_indices: StateIndices,
    state_consts: StateConsts,
    state_bounds: StateBounds,
) -> None:
    """Less-or-equal comparison with bounds resolution.

    Always true when ``hi(a) <= lo(b)``.
    Always false when ``lo(a) > hi(b)``.
    """
    state_indices[eqn.outvars[0]] = empty_index_sets(atom_numel(eqn.outvars[0]))
    propagate_const_binary(eqn, state_consts, np.less_equal)
    bounds = _get_bounds(eqn, state_consts, state_bounds)
    if bounds is not None:
        lo1, hi1, lo2, hi2 = bounds
        if np.all(hi1 <= lo2):
            _set_const(eqn, state_consts, True)
        elif np.all(lo1 > hi2):
            _set_const(eqn, state_consts, False)


def prop_gt(
    eqn: JaxprEqn,
    state_indices: StateIndices,
    state_consts: StateConsts,
    state_bounds: StateBounds,
) -> None:
    """Greater-than comparison with bounds resolution.

    Always true when ``lo(a) > hi(b)``.
    Always false when ``hi(a) <= lo(b)``.
    """
    state_indices[eqn.outvars[0]] = empty_index_sets(atom_numel(eqn.outvars[0]))
    propagate_const_binary(eqn, state_consts, np.greater)
    bounds = _get_bounds(eqn, state_consts, state_bounds)
    if bounds is not None:
        lo1, hi1, lo2, hi2 = bounds
        if np.all(lo1 > hi2):
            _set_const(eqn, state_consts, True)
        elif np.all(hi1 <= lo2):
            _set_const(eqn, state_consts, False)


def prop_ge(
    eqn: JaxprEqn,
    state_indices: StateIndices,
    state_consts: StateConsts,
    state_bounds: StateBounds,
) -> None:
    """Greater-or-equal comparison with bounds resolution.

    Always true when ``lo(a) >= hi(b)``.
    Always false when ``hi(a) < lo(b)``.
    """
    state_indices[eqn.outvars[0]] = empty_index_sets(atom_numel(eqn.outvars[0]))
    propagate_const_binary(eqn, state_consts, np.greater_equal)
    bounds = _get_bounds(eqn, state_consts, state_bounds)
    if bounds is not None:
        lo1, hi1, lo2, hi2 = bounds
        if np.all(lo1 >= hi2):
            _set_const(eqn, state_consts, True)
        elif np.all(hi1 < lo2):
            _set_const(eqn, state_consts, False)
