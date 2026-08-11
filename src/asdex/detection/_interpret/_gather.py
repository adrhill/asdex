"""Propagation rule for gather operations.

Naming: ``si_`` is short for ``start_indices``, the second input to ``lax.gather``.
Also hosts the index-vector iteration machinery shared with ``_scatter.py``.
"""

from collections.abc import Iterator, Sequence

import numpy as np
from jax._src.core import JaxprEqn

from ._common import (
    _atom_const_val,
    _atom_numel,
    _atom_shape,
    _atom_value_bounds,
    _bounded_ranges,
    _clamp_starts,
    _conservative_indices,
    _enumerate_bounded_patterns,
    _index_sets,
    _merge_index_dependencies,
    _permute_indices,
    _position_map,
    _PropState,
)


def _si_batch_axes(
    si_shape: tuple[int, ...], si_batching_dims: tuple[int, ...]
) -> list[int]:
    """Start-indices axes that enumerate the batch.

    Excludes the trailing index-vector dim,
    which holds the components of one index vector rather than a batch position,
    and the explicit batching dims, which pair with operand batch positions.
    """
    index_vector_dim = len(si_shape) - 1
    return [
        d
        for d in range(len(si_shape))
        if d != index_vector_dim and d not in si_batching_dims
    ]


def _si_batch_shapes(
    concrete_indices: np.ndarray,
    operand_shape: tuple[int, ...],
    operand_batching_dims: tuple[int, ...],
    si_batching_dims: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """The operand batch shape and the start-indices batch shape.

    Together these are the leading axes of the intermediate gather result,
    in the order ``_iter_si_starts`` walks them.
    """
    batching_shape = tuple(operand_shape[d] for d in operand_batching_dims)
    si_shape = concrete_indices.shape
    si_batch_shape = tuple(
        si_shape[d] for d in _si_batch_axes(si_shape, si_batching_dims)
    )
    return batching_shape, si_batch_shape


def _iter_si_starts(
    concrete_indices: np.ndarray,
    operand_shape: tuple[int, ...],
    operand_batching_dims: tuple[int, ...],
    si_batching_dims: tuple[int, ...],
    index_map: Sequence[int],
) -> Iterator[tuple[tuple[int, ...], tuple[int, ...], list[int]]]:
    """Shared index-vector iteration for gather and scatter.

    Walks every combination of operand batch position and start-indices batch position,
    extracts the corresponding index vector from ``concrete_indices``,
    and assembles the operand start position it addresses.
    ``index_map`` maps index-vector components to operand dims
    (``start_index_map`` for gather, ``scatter_dims_to_operand_dims`` for scatter).

    Yields ``(batch_idx, si_batch_idx, start)`` triples
    in row-major order over both batch spaces.
    Starts are not clamped;
    gather (always) and scatter (mode='clip') apply their own OOB policy.
    """
    op_ndim = len(operand_shape)
    si_shape = concrete_indices.shape
    index_vector_dim = len(si_shape) - 1
    si_batch_axes = _si_batch_axes(si_shape, si_batching_dims)
    batching_shape, si_batch_shape = _si_batch_shapes(
        concrete_indices, operand_shape, operand_batching_dims, si_batching_dims
    )

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
            for i, d in enumerate(index_map):
                start[d] = int(index_vector[i])
            for i, d in enumerate(operand_batching_dims):
                start[d] = int(batch_idx[i])
            yield batch_idx, si_batch_idx, start


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

    operand_batching_dims = getattr(dim_nums, "operand_batching_dims", ()) or ()
    si_batching_dims = getattr(dim_nums, "start_indices_batching_dims", ()) or ()

    removed = set(collapsed) | set(operand_batching_dims)
    offset_operand_dims = [d for d in range(op_ndim) if d not in removed]
    offset_shape = tuple(slice_sizes[d] for d in offset_operand_dims)

    op_pos = _position_map(operand_shape)

    batching_shape, si_batch_shape = _si_batch_shapes(
        concrete_indices, operand_shape, operand_batching_dims, si_batching_dims
    )
    starts = _iter_si_starts(
        concrete_indices,
        operand_shape,
        operand_batching_dims,
        si_batching_dims,
        dim_nums.start_index_map,
    )

    slices = []
    for _, _, raw_start in starts:
        # JAX clamps OOB indices to valid bounds.
        start = _clamp_starts(raw_start, operand_shape, slice_sizes)

        sl = tuple(slice(start[d], start[d] + slice_sizes[d]) for d in range(op_ndim))
        result = op_pos[sl]

        for d in sorted(removed, reverse=True):
            result = np.squeeze(result, axis=d)

        slices.append(result.flatten())

    all_results = np.stack(slices)
    intermediate_shape = batching_shape + si_batch_shape + offset_shape
    assembled = all_results.reshape(intermediate_shape)

    out_ndim = len(_atom_shape(eqn.outvars[0]))
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


def _prop_gather(
    eqn: JaxprEqn,
    state: _PropState,
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

    Example with data-dependent start_indices: y = x[argsort(x)]
        The indices depend on x, so each output depends on all inputs
        (both from the operand and from the index computation).
        Conservative fallback: all outputs depend on all inputs.

    Jaxpr:
        invars[0]: operand — array to gather from
        invars[1]: start_indices — positions at which slices begin
        dimension_numbers: GatherDimensionNumbers specifying axis mapping
        slice_sizes: shape of each extracted slice (length = ndim(operand))

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.gather.html
    """
    operand_indices = _index_sets(state, eqn.invars[0])
    si_index_sets = _index_sets(state, eqn.invars[1])
    operand_shape = _atom_shape(eqn.invars[0])
    out_size = _atom_numel(eqn.outvars[0])

    if out_size == 0:
        state.indices[eqn.outvars[0]] = []
        return

    concrete_indices = _atom_const_val(eqn.invars[1], state)
    if concrete_indices is not None:
        flat_map = _gather_flat_map(concrete_indices, eqn, operand_shape)
        state.indices[eqn.outvars[0]] = _permute_indices(operand_indices, flat_map)
        return

    # Try bounded enumeration.
    bounds = _atom_value_bounds(eqn.invars[1], state)
    if bounds is not None:
        lo = bounds[0]
        si_shape = _atom_shape(eqn.invars[1])
        ranges = _bounded_ranges(bounds)

        def _make(vals: tuple[int, ...]) -> list[set[int]]:
            candidate = np.array(vals, dtype=lo.dtype).reshape(si_shape)
            return _permute_indices(
                operand_indices, _gather_flat_map(candidate, eqn, operand_shape)
            )

        result = _enumerate_bounded_patterns(ranges, out_size, _make)
        if result is not None:
            state.indices[eqn.outvars[0]] = _merge_index_dependencies(
                result, si_index_sets
            )
            return

    # Conservative fallback: every output depends on all inputs.
    # Include both operand and start_indices dependencies.
    state.indices[eqn.outvars[0]] = _conservative_indices(
        operand_indices + si_index_sets, out_size
    )
