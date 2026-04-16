"""Propagation rule for gather operations."""

import numpy as np
from jax._src.core import JaxprEqn

from ._commons import (
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
    permute_indices,
    position_map,
)


def _gather_flat_map(
    concrete_indices: np.ndarray,
    eqn: JaxprEqn,
    operand_shape: tuple[int, ...],
) -> np.ndarray:
    """Compute the flat output→input position map for a gather with known indices.

    Simulates XLA gather semantics on a position map.
    Returns a 1-D integer array where ``flat_map[i]`` is the flat input position
    that output element ``i`` reads from.
    """
    dim_nums = eqn.params["dimension_numbers"]
    slice_sizes = eqn.params["slice_sizes"]
    op_ndim = len(operand_shape)
    offset_dims = dim_nums.offset_dims
    collapsed = dim_nums.collapsed_slice_dims
    start_index_map = dim_nums.start_index_map

    operand_batching_dims = getattr(dim_nums, "operand_batching_dims", ()) or ()
    si_batching_dims = getattr(dim_nums, "start_indices_batching_dims", ()) or ()

    si_shape = concrete_indices.shape
    index_vector_dim = len(si_shape) - 1

    si_batch_axes = [
        d
        for d in range(len(si_shape))
        if d != index_vector_dim and d not in si_batching_dims
    ]
    si_batch_shape = tuple(si_shape[d] for d in si_batch_axes)
    batching_shape = tuple(operand_shape[d] for d in operand_batching_dims)

    removed = set(collapsed) | set(operand_batching_dims)
    offset_operand_dims = [d for d in range(op_ndim) if d not in removed]
    offset_shape = tuple(slice_sizes[d] for d in offset_operand_dims)

    op_pos = position_map(operand_shape)

    slices = []
    for batch_idx in np.ndindex(*batching_shape) if batching_shape else [()]:
        for si_batch_idx in np.ndindex(*si_batch_shape) if si_batch_shape else [()]:
            si_idx: list[int | slice] = [0 for _ in range(len(si_shape))]
            for i, d in enumerate(si_batching_dims):
                si_idx[d] = batch_idx[i]
            for i, d in enumerate(si_batch_axes):
                si_idx[d] = si_batch_idx[i]
            si_idx[index_vector_dim] = slice(None)
            index_vector = concrete_indices[tuple(si_idx)]

            start = [0] * op_ndim
            for i, d in enumerate(start_index_map):
                start[d] = int(index_vector[i])
            for i, d in enumerate(operand_batching_dims):
                start[d] = int(batch_idx[i])

            # JAX clamps OOB indices to valid bounds.
            for d in range(op_ndim):
                start[d] = max(0, min(start[d], operand_shape[d] - slice_sizes[d]))

            sl = tuple(
                slice(start[d], start[d] + slice_sizes[d]) for d in range(op_ndim)
            )
            result = op_pos[sl]

            for d in sorted(removed, reverse=True):
                result = np.squeeze(result, axis=d)

            slices.append(result.flatten())

    all_results = np.stack(slices)
    intermediate_shape = batching_shape + si_batch_shape + offset_shape
    assembled = all_results.reshape(intermediate_shape)

    out_ndim = len(atom_shape(eqn.outvars[0]))
    n_batch = len(batching_shape) + len(si_batch_shape)

    perm = [0] * out_ndim
    batch_iter = iter(range(n_batch))
    offset_iter = iter(range(n_batch, n_batch + len(offset_shape)))
    for i in range(out_ndim):
        if i in offset_dims:
            perm[i] = next(offset_iter)
        else:
            perm[i] = next(batch_iter)

    return assembled.transpose(perm).flatten()


def prop_gather(
    eqn: JaxprEqn,
    state_indices: StateIndices,
    state_consts: StateConsts,
    state_bounds: StateBounds,
) -> None:
    """Gather extracts slices from operand at positions given by start_indices.

    For static start_indices (Literal or tracked const),
    simulates XLA gather semantics on a position map
    to determine which input element each output element reads from.
    Handles any ``GatherDimensionNumbers`` configuration,
    including mismatched ``start_index_map``,
    partial slices, and ``operand_batching_dims``.
    For bounded dynamic start_indices, enumerates all possible index arrays
    and unions the resulting patterns.
    For fully dynamic start_indices, falls back to conservative.

    The Jacobian is a selection/permutation matrix:
    each output element reads exactly one input element.

    Example: x = [a, b, c], idx = [2, 0, 1], y = x[idx] = [c, a, b]
        Input index sets:  [{0}, {1}, {2}]
        Output index sets: [{2}, {0}, {1}]  (permuted by index array)

    Example: x.shape = (3, 4), y = x[:, idx] where idx = [2, 0]
        Each output row selects columns 2 and 0 from the corresponding input row.

    Example with dynamic start_indices: x[traced_idx]
        Cannot determine which inputs each output depends on.
        Conservative: all outputs depend on all inputs.

    Jaxpr:
        invars[0]: operand — array to gather from
        invars[1]: start_indices — positions at which slices begin
        dimension_numbers: GatherDimensionNumbers specifying axis mapping
        slice_sizes: shape of each extracted slice (length = ndim(operand))

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.gather.html
    """
    operand_indices = index_sets(state_indices, eqn.invars[0])
    # TODO: include start_indices index sets in output dependencies.
    check_no_index_sets(state_indices, eqn.invars[1], eqn.primitive.name)
    operand_shape = atom_shape(eqn.invars[0])
    out_size = atom_numel(eqn.outvars[0])

    if out_size == 0:
        state_indices[eqn.outvars[0]] = []
        return

    concrete_indices = atom_const_val(eqn.invars[1], state_consts)
    if concrete_indices is not None:
        flat_map = _gather_flat_map(concrete_indices, eqn, operand_shape)
        state_indices[eqn.outvars[0]] = permute_indices(operand_indices, flat_map)
        return

    # Try bounded enumeration.
    bounds = atom_value_bounds(eqn.invars[1], state_consts, state_bounds)
    if bounds is not None:
        lo, hi = bounds
        lo_flat, hi_flat = lo.flatten(), hi.flatten()
        si_shape = atom_shape(eqn.invars[1])
        ranges = [
            range(int(lo_flat[i]), int(hi_flat[i]) + 1) for i in range(len(lo_flat))
        ]

        def _make(vals: tuple[int, ...]) -> list[set[int]]:
            candidate = np.array(vals, dtype=lo.dtype).reshape(si_shape)
            return permute_indices(
                operand_indices, _gather_flat_map(candidate, eqn, operand_shape)
            )

        result = enumerate_bounded_patterns(ranges, out_size, _make)
        if result is not None:
            state_indices[eqn.outvars[0]] = result
            return

    # Conservative fallback: every output depends on every input.
    state_indices[eqn.outvars[0]] = conservative_indices(operand_indices, out_size)
