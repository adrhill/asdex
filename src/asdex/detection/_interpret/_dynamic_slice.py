"""Propagation rules for dynamic_slice and dynamic_update_slice."""

import numpy as np
from jax._src.core import JaxprEqn

from ._common import (
    IndexSet,
    _atom_const_val,
    _atom_shape,
    _atom_value_bounds,
    _bounded_ranges,
    _clamp_starts,
    _conservative_indices,
    _enumerate_bounded_patterns,
    _index_sets,
    _merge_index_dependencies,
    _numel,
    _PropState,
    _transform_indices,
)


def _resolve_starts(
    eqn: JaxprEqn, start_offset: int, state: _PropState
) -> list[int] | None:
    """Try to resolve start indices as static ints.

    Returns None if any start depends on runtime values.
    """
    starts: list[int] = []
    for atom in eqn.invars[start_offset:]:
        val = _atom_const_val(atom, state)
        if val is None:
            return None
        starts.append(int(val.flat[0]))
    return starts


def _resolve_start_bounds(
    eqn: JaxprEqn,
    start_offset: int,
    state: _PropState,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Try to resolve per-dimension (lo, hi) bounds for start indices.

    Returns None if any start has no bounds information.
    """
    los: list[int] = []
    his: list[int] = []
    for atom in eqn.invars[start_offset:]:
        b = _atom_value_bounds(atom, state)
        if b is None:
            return None
        lo, hi = b
        los.append(int(lo.flat[0]))
        his.append(int(hi.flat[0]))
    return np.array(los), np.array(his)


def _prop_dynamic_slice(
    eqn: JaxprEqn,
    state: _PropState,
) -> None:
    """dynamic_slice extracts a sub-array at a potentially dynamic offset.

    With static start indices, each output element maps to exactly one input element.
    With bounded dynamic starts, enumerates all possible start positions
    and unions the resulting patterns.
    Otherwise falls back to conservative,
    including the start indices' own dependencies.

    For static starts s and slice_sizes sz:
        out[i₀, i₁, ...] = in[s₀ + i₀, s₁ + i₁, ...]
    The Jacobian is a selection matrix with exactly one 1 per row.

    Example: x = [a, b, c, d, e], dynamic_slice(x, [1], [3]) = [b, c, d]
        Input index sets:  [{0}, {1}, {2}, {3}, {4}]
        Output index sets: [{1}, {2}, {3}]

    Jaxpr:
        invars: [operand, *start_indices]
        params: slice_sizes

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.dynamic_slice.html
    """
    operand = eqn.invars[0]
    in_indices = _index_sets(state, operand)
    slice_sizes = eqn.params["slice_sizes"]

    start_index_sets: list[IndexSet] = []
    for start_atom in eqn.invars[1:]:
        start_index_sets.extend(_index_sets(state, start_atom))

    starts = _resolve_starts(eqn, 1, state)
    if starts is not None:
        in_shape = _atom_shape(operand)
        slices = tuple(
            slice(s, s + sz) for s, sz in zip(starts, slice_sizes, strict=True)
        )
        state.indices[eqn.outvars[0]] = _transform_indices(
            in_indices, in_shape, lambda p: p[slices]
        )
        return

    # Try bounded enumeration.
    start_bounds = _resolve_start_bounds(eqn, 1, state)
    if start_bounds is not None:
        in_shape = _atom_shape(operand)
        ranges = _bounded_ranges(start_bounds)

        def _make_slice(vals: tuple[int, ...]) -> list[set[int]]:
            clamped = _clamp_starts(vals, in_shape, slice_sizes)
            sl = tuple(
                slice(s, s + sz) for s, sz in zip(clamped, slice_sizes, strict=True)
            )
            return _transform_indices(in_indices, in_shape, lambda p, sl=sl: p[sl])

        result = _enumerate_bounded_patterns(ranges, _numel(slice_sizes), _make_slice)
        if result is not None:
            state.indices[eqn.outvars[0]] = _merge_index_dependencies(
                result, start_index_sets
            )
            return

    # Unresolvable starts - conservative fallback,
    # including the starts' own dependencies (mirrors gather).
    state.indices[eqn.outvars[0]] = _conservative_indices(
        in_indices + start_index_sets, _numel(slice_sizes)
    )


def _prop_dynamic_update_slice(
    eqn: JaxprEqn,
    state: _PropState,
) -> None:
    """dynamic_update_slice overwrites a sub-array at a potentially dynamic offset.

    With static start indices, updated positions get update index sets,
    the rest keep operand index sets.
    With bounded dynamic starts, enumerates all possible start positions
    and unions the resulting patterns.
    Otherwise falls back to conservative,
    including the start indices' own dependencies.

    For static starts s and update shape u_shape:
        out[i] = update[i - s]  if s ≤ i < s + u_shape
        out[i] = operand[i]     otherwise

    Example: operand = [a, b, c, d], update = [X, Y], start = [1]
        out = [a, X, Y, d]
        Output index sets: [{0}, {upd_0}, {upd_1}, {3}]

    Jaxpr:
        invars: [operand, update, *start_indices]
        params: (none relevant)

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.dynamic_update_slice.html
    """
    operand = eqn.invars[0]
    update = eqn.invars[1]
    operand_indices = _index_sets(state, operand)
    upd_indices = _index_sets(state, update)
    operand_shape = _atom_shape(operand)
    upd_shape = _atom_shape(update)

    start_index_sets: list[IndexSet] = []
    for start_atom in eqn.invars[2:]:
        start_index_sets.extend(_index_sets(state, start_atom))

    starts = _resolve_starts(eqn, 2, state)
    if starts is not None:
        state.indices[eqn.outvars[0]] = _dynamic_update_for_starts(
            starts,
            operand_indices,
            upd_indices,
            operand_shape,
            upd_shape,
        )
        return

    # Try bounded enumeration.
    start_bounds = _resolve_start_bounds(eqn, 2, state)
    if start_bounds is not None:
        ranges = _bounded_ranges(start_bounds)

        def _make_update(vals: tuple[int, ...]) -> list[set[int]]:
            clamped = _clamp_starts(vals, operand_shape, upd_shape)
            return _dynamic_update_for_starts(
                list(clamped),
                operand_indices,
                upd_indices,
                operand_shape,
                upd_shape,
            )

        result = _enumerate_bounded_patterns(
            ranges, _numel(operand_shape), _make_update
        )
        if result is not None:
            state.indices[eqn.outvars[0]] = _merge_index_dependencies(
                result, start_index_sets
            )
            return

    # Unresolvable starts - conservative fallback,
    # including the starts' own dependencies (mirrors gather).
    state.indices[eqn.outvars[0]] = _conservative_indices(
        operand_indices + upd_indices + start_index_sets, _numel(operand_shape)
    )


def _dynamic_update_for_starts(
    starts: list[int] | tuple[int, ...],
    operand_indices: list[IndexSet],
    upd_indices: list[IndexSet],
    operand_shape: tuple[int, ...],
    upd_shape: tuple[int, ...],
) -> list[IndexSet]:
    """Compute output index sets for a dynamic_update_slice with known starts."""
    # Shallow copy: entries in the window are replaced below,
    # the sets themselves are shared and never mutated.
    out_indices: list[IndexSet] = list(operand_indices)

    upd_coords = np.indices(upd_shape)
    op_coords = tuple(s + upd_coords[d] for d, s in enumerate(starts))
    flat_map = np.ravel_multi_index(op_coords, operand_shape).ravel()

    for upd_flat, op_flat in enumerate(flat_map):
        out_indices[op_flat] = upd_indices[upd_flat]

    return out_indices
