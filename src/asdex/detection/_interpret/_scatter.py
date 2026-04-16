"""Propagation rule for scatter operations."""

import numpy as np
from jax._src.core import JaxprEqn

from ._commons import (
    IndexSet,
    StateBounds,
    StateConsts,
    StateIndices,
    atom_const_val,
    atom_numel,
    atom_shape,
    atom_value_bounds,
    check_no_index_sets,
    conservative_indices,
    enumerate_bounded_patterns,
    index_sets,
    numel,
)


def _scatter_flat_map(
    concrete_indices: np.ndarray,
    eqn: JaxprEqn,
    operand_shape: tuple[int, ...],
    updates_shape: tuple[int, ...],
) -> np.ndarray:
    """Compute the flat update-to-operand position map for a scatter with known indices.

    Simulates XLA scatter semantics on flat positions.
    Returns a 1-D integer array where ``flat_map[i]`` is the operand flat position
    that update element ``i`` writes to, or ``-1`` for OOB.
    """
    dim_nums = eqn.params["dimension_numbers"]
    op_ndim = len(operand_shape)
    update_window_dims = set(dim_nums.update_window_dims)
    inserted_window_dims = dim_nums.inserted_window_dims
    scatter_dims_to_operand_dims = dim_nums.scatter_dims_to_operand_dims

    operand_batching_dims = getattr(dim_nums, "operand_batching_dims", ()) or ()
    si_batching_dims = getattr(dim_nums, "scatter_indices_batching_dims", ()) or ()

    si_shape = concrete_indices.shape
    index_vector_dim = len(si_shape) - 1

    si_batch_axes = [
        d
        for d in range(len(si_shape))
        if d != index_vector_dim and d not in si_batching_dims
    ]
    si_batch_shape = tuple(si_shape[d] for d in si_batch_axes)
    batching_shape = tuple(operand_shape[d] for d in operand_batching_dims)

    # Operand dims removed from the window (inserted + batching).
    removed = set(inserted_window_dims) | set(operand_batching_dims)
    window_operand_dims = [d for d in range(op_ndim) if d not in removed]
    window_shape = tuple(updates_shape[d] for d in dim_nums.update_window_dims)

    updates_size = numel(updates_shape)
    update_ndim = len(updates_shape)
    flat_map = np.full(updates_size, -1, dtype=np.intp)

    for batch_idx in np.ndindex(*batching_shape) if batching_shape else [()]:
        for si_batch_idx in np.ndindex(*si_batch_shape) if si_batch_shape else [()]:
            # Look up index vector from scatter_indices.
            si_idx: list[int | slice] = [0 for _ in range(len(si_shape))]
            for i, d in enumerate(si_batching_dims):
                si_idx[d] = batch_idx[i]
            for i, d in enumerate(si_batch_axes):
                si_idx[d] = si_batch_idx[i]
            si_idx[index_vector_dim] = slice(None)
            index_vector = concrete_indices[tuple(si_idx)]

            # Build start position in operand.
            start = [0] * op_ndim
            for i, d in enumerate(scatter_dims_to_operand_dims):
                start[d] = int(index_vector[i])
            for i, d in enumerate(operand_batching_dims):
                start[d] = int(batch_idx[i])

            for window_idx in np.ndindex(*window_shape) if window_shape else [()]:
                # Build full operand index: start + window offset at non-removed dims.
                operand_idx = list(start)
                w_iter = iter(window_idx)
                for d in window_operand_dims:
                    operand_idx[d] += next(w_iter)

                # Scatter drops OOB updates (unlike gather which clamps).
                if any(
                    operand_idx[d] < 0 or operand_idx[d] >= operand_shape[d]
                    for d in range(op_ndim)
                ):
                    continue

                # Build update multi-index from batch and window components.
                update_multi = [0] * update_ndim
                b_iter = iter(batch_idx + si_batch_idx)
                w_iter2 = iter(window_idx)
                for d in range(update_ndim):
                    if d in update_window_dims:
                        update_multi[d] = next(w_iter2)
                    else:
                        update_multi[d] = next(b_iter)

                operand_flat = int(np.ravel_multi_index(operand_idx, operand_shape))
                update_flat = int(np.ravel_multi_index(update_multi, updates_shape))
                flat_map[update_flat] = operand_flat

    return flat_map


def _scatter_for_indices(
    concrete_indices: np.ndarray,
    eqn: JaxprEqn,
    operand_indices: list[IndexSet],
    updates_indices: list[IndexSet],
) -> list[IndexSet]:
    """Compute output index sets for a scatter with known concrete indices.

    Uses ``_scatter_flat_map`` to determine which operand position
    each update element targets,
    then combines index sets according to scatter semantics.
    """
    update_jaxpr = eqn.params.get("update_jaxpr")
    is_combine = update_jaxpr is not None

    operand_shape = atom_shape(eqn.invars[0])
    out_size = numel(operand_shape)
    updates_shape = atom_shape(eqn.invars[2])

    flat_map = _scatter_flat_map(concrete_indices, eqn, operand_shape, updates_shape)

    # Build operand_pos → list of update positions that target it.
    scatter_positions: dict[int, list[int]] = {}
    for update_flat, operand_flat in enumerate(flat_map):
        if operand_flat >= 0:
            if operand_flat not in scatter_positions:
                scatter_positions[operand_flat] = []
            scatter_positions[operand_flat].append(update_flat)

    out_indices: list[IndexSet] = []
    for i in range(out_size):
        if i in scatter_positions:
            if is_combine:
                combined = operand_indices[i].copy()
                for u_flat in scatter_positions[i]:
                    combined |= updates_indices[u_flat]
                out_indices.append(combined)
            else:
                # Replace semantics: last writer wins.
                last_u = scatter_positions[i][-1]
                out_indices.append(updates_indices[last_u].copy())
        else:
            out_indices.append(operand_indices[i].copy())

    return out_indices


def prop_scatter(
    eqn: JaxprEqn,
    state_indices: StateIndices,
    state_consts: StateConsts,
    state_bounds: StateBounds,
) -> None:
    """Scatter writes updates into operand at positions given by scatter_indices.

    For static scatter_indices (Literal or tracked const),
    simulates XLA scatter semantics on flat positions
    to determine which operand element each update element targets.
    Handles any ``ScatterDimensionNumbers`` configuration,
    including partial windows, multi-index scatters,
    and ``operand_batching_dims``.
    For bounded dynamic scatter_indices, enumerates all possible index arrays
    and unions the resulting patterns.
    For fully dynamic scatter_indices, falls back to conservative.

    For scatter (replace): out[idx[i]] = updates[i], else out[j] = operand[j]
        Positions NOT in idx: depend on corresponding operand element.
        Positions in idx: depend on corresponding updates element.

    For scatter-add/mul/min/max: out[idx[i]] = combine(operand[idx[i]], updates[i])
        Positions in idx: depend on BOTH operand AND updates.

    Example: arr = [a, b, c], arr.at[1].set(x) = [a, x, c]
        operand state_indices:  [{0}, {1}, {2}]  (from arr)
        updates state_indices:  [{3}]             (from x, assuming x is input index 3)
        Output state_indices:   [{0}, {3}, {2}]   (index 1 replaced by x)

    Example with dynamic scatter_indices: arr.at[traced_idx].set(x)
        Cannot determine which position receives the update.
        Conservative: all outputs depend on all inputs.

    Jaxpr:
        invars[0]: operand — base array
        invars[1]: scatter_indices — positions to scatter into
        invars[2]: updates — values to write
        dimension_numbers: ScatterDimensionNumbers specifying axis mapping
        update_jaxpr: combination function (e.g., add for scatter-add),
            absent for plain scatter (replace)

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.scatter.html
    """
    operand_indices = index_sets(state_indices, eqn.invars[0])
    indices_atom = eqn.invars[1]
    # TODO: include scatter_indices index sets in output dependencies.
    check_no_index_sets(state_indices, indices_atom, eqn.primitive.name)
    updates_indices = index_sets(state_indices, eqn.invars[2])

    concrete_indices = atom_const_val(indices_atom, state_consts)

    if concrete_indices is not None:
        state_indices[eqn.outvars[0]] = _scatter_for_indices(
            concrete_indices,
            eqn,
            operand_indices,
            updates_indices,
        )
        return

    # Try bounded enumeration.
    bounds = atom_value_bounds(indices_atom, state_consts, state_bounds)
    if bounds is not None:
        lo, hi = bounds
        lo_flat, hi_flat = lo.flatten(), hi.flatten()
        si_shape = atom_shape(indices_atom)
        out_size = atom_numel(eqn.outvars[0])
        ranges = [
            range(int(lo_flat[i]), int(hi_flat[i]) + 1) for i in range(len(lo_flat))
        ]

        def _make(vals: tuple[int, ...]) -> list[set[int]]:
            candidate = np.array(vals, dtype=lo.dtype).reshape(si_shape)
            return _scatter_for_indices(
                candidate, eqn, operand_indices, updates_indices
            )

        result = enumerate_bounded_patterns(ranges, out_size, _make)
        if result is not None:
            state_indices[eqn.outvars[0]] = result
            return

    # Dynamic indices - conservative fallback.
    state_indices[eqn.outvars[0]] = conservative_indices(
        operand_indices + updates_indices, atom_numel(eqn.outvars[0])
    )
