"""Propagation rules for dynamic_slice and dynamic_update_slice."""

import numpy as np
from jax._src.core import JaxprEqn

from ._commons import (
    IndexSet,
    StateBounds,
    StateConsts,
    StateIndices,
    atom_const_val,
    atom_shape,
    atom_value_bounds,
    check_no_index_sets,
    clamp_starts,
    conservative_indices,
    copy_index_sets,
    enumerate_bounded_patterns,
    index_sets,
    numel,
    transform_indices,
)


def _resolve_starts(
    eqn: JaxprEqn, start_offset: int, state_consts: StateConsts
) -> list[int] | None:
    """Try to resolve start indices as static ints.

    Returns None if any start depends on runtime values.
    """
    starts: list[int] = []
    for atom in eqn.invars[start_offset:]:
        val = atom_const_val(atom, state_consts)
        if val is None:
            return None
        starts.append(int(val.flat[0]))
    return starts


def _resolve_start_bounds(
    eqn: JaxprEqn,
    start_offset: int,
    state_consts: StateConsts,
    state_bounds: StateBounds,
) -> list[tuple[int, int]] | None:
    """Try to resolve per-dimension (lo, hi) bounds for start indices.

    Returns None if any start has no bounds information.
    """
    bounds: list[tuple[int, int]] = []
    for atom in eqn.invars[start_offset:]:
        b = atom_value_bounds(atom, state_consts, state_bounds)
        if b is None:
            return None
        lo, hi = b
        bounds.append((int(lo.flat[0]), int(hi.flat[0])))
    return bounds


def prop_dynamic_slice(
    eqn: JaxprEqn,
    state_indices: StateIndices,
    state_consts: StateConsts,
    state_bounds: StateBounds,
) -> None:
    """dynamic_slice extracts a sub-array at a potentially dynamic offset.

    With static start indices, each output element maps to exactly one input element.
    With bounded dynamic starts, enumerates all possible start positions
    and unions the resulting patterns.
    Otherwise falls back to conservative.

    For static starts s and slice_sizes sz:
        out[i₀, i₁, ...] = in[s₀ + i₀, s₁ + i₁, ...]
    The Jacobian is a selection matrix with exactly one 1 per row.

    Example: x = [a, b, c, d, e], dynamic_slice(x, [1], [3]) = [b, c, d]
        Input state_indices:  [{0}, {1}, {2}, {3}, {4}]
        Output state_indices: [{1}, {2}, {3}]

    Jaxpr:
        invars: [operand, *start_indices]
        params: slice_sizes

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.dynamic_slice.html
    """
    operand = eqn.invars[0]
    in_indices = index_sets(state_indices, operand)
    slice_sizes = eqn.params["slice_sizes"]

    # TODO: include start index sets in output dependencies.
    for start_atom in eqn.invars[1:]:
        check_no_index_sets(state_indices, start_atom, eqn.primitive.name)

    starts = _resolve_starts(eqn, 1, state_consts)
    if starts is not None:
        in_shape = atom_shape(operand)
        slices = tuple(
            slice(s, s + sz) for s, sz in zip(starts, slice_sizes, strict=True)
        )
        state_indices[eqn.outvars[0]] = transform_indices(
            in_indices, in_shape, lambda p: p[slices]
        )
        return

    # Try bounded enumeration.
    start_bounds = _resolve_start_bounds(eqn, 1, state_consts, state_bounds)
    if start_bounds is not None:
        in_shape = atom_shape(operand)
        ranges = [range(lo, hi + 1) for lo, hi in start_bounds]

        def _make_slice(vals: tuple[int, ...]) -> list[set[int]]:
            clamped = clamp_starts(vals, in_shape, slice_sizes)
            sl = tuple(
                slice(s, s + sz) for s, sz in zip(clamped, slice_sizes, strict=True)
            )
            return transform_indices(in_indices, in_shape, lambda p, sl=sl: p[sl])

        result = enumerate_bounded_patterns(ranges, numel(slice_sizes), _make_slice)
        if result is not None:
            state_indices[eqn.outvars[0]] = result
            return

    state_indices[eqn.outvars[0]] = conservative_indices(in_indices, numel(slice_sizes))


def prop_dynamic_update_slice(
    eqn: JaxprEqn,
    state_indices: StateIndices,
    state_consts: StateConsts,
    state_bounds: StateBounds,
) -> None:
    """dynamic_update_slice overwrites a sub-array at a potentially dynamic offset.

    With static start indices, updated positions get update state_indices,
    the rest keep operand state_indices.
    With bounded dynamic starts, enumerates all possible start positions
    and unions the resulting patterns.
    Otherwise falls back to conservative.

    For static starts s and update shape u_shape:
        out[i] = update[i - s]  if s ≤ i < s + u_shape
        out[i] = operand[i]     otherwise

    Example: operand = [a, b, c, d], update = [X, Y], start = [1]
        out = [a, X, Y, d]
        Output state_indices: [{0}, {upd_0}, {upd_1}, {3}]

    Jaxpr:
        invars: [operand, update, *start_indices]
        params: (none relevant)

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.dynamic_update_slice.html
    """
    operand = eqn.invars[0]
    update = eqn.invars[1]
    operand_indices = index_sets(state_indices, operand)
    upd_indices = index_sets(state_indices, update)
    operand_shape = atom_shape(operand)
    upd_shape = atom_shape(update)

    # TODO: include start index sets in output dependencies.
    for start_atom in eqn.invars[2:]:
        check_no_index_sets(state_indices, start_atom, eqn.primitive.name)

    starts = _resolve_starts(eqn, 2, state_consts)
    if starts is not None:
        state_indices[eqn.outvars[0]] = _dynamic_update_for_starts(
            starts,
            operand_indices,
            upd_indices,
            operand_shape,
            upd_shape,
        )
        return

    # Try bounded enumeration.
    start_bounds = _resolve_start_bounds(eqn, 2, state_consts, state_bounds)
    if start_bounds is not None:
        ranges = [range(lo, hi + 1) for lo, hi in start_bounds]

        def _make_update(vals: tuple[int, ...]) -> list[set[int]]:
            clamped = clamp_starts(vals, operand_shape, upd_shape)
            return _dynamic_update_for_starts(
                list(clamped),
                operand_indices,
                upd_indices,
                operand_shape,
                upd_shape,
            )

        result = enumerate_bounded_patterns(ranges, numel(operand_shape), _make_update)
        if result is not None:
            state_indices[eqn.outvars[0]] = result
            return

    state_indices[eqn.outvars[0]] = conservative_indices(
        operand_indices + upd_indices, numel(operand_shape)
    )


def _dynamic_update_for_starts(
    starts: list[int] | tuple[int, ...],
    operand_indices: list[IndexSet],
    upd_indices: list[IndexSet],
    operand_shape: tuple[int, ...],
    upd_shape: tuple[int, ...],
) -> list[IndexSet]:
    """Compute output index sets for a dynamic_update_slice with known starts."""
    out_indices: list[IndexSet] = copy_index_sets(operand_indices)

    upd_coords = np.indices(upd_shape)
    op_coords = tuple(s + upd_coords[d] for d, s in enumerate(starts))
    flat_map = np.ravel_multi_index(op_coords, operand_shape).ravel()

    for upd_flat, op_flat in enumerate(flat_map):
        out_indices[op_flat] = upd_indices[upd_flat]

    return out_indices
