"""Propagation rule for gather operations.

Naming: ``si_`` is short for ``start_indices``, the second input to ``lax.gather``.
Also hosts the index-vector iteration machinery shared with ``_scatter.py``.

Start-indices layout:
``start_indices`` is an array of index vectors.
Its trailing dim holds the components of one index vector,
and each index vector gives the operand position where one gathered slice starts.
Its other dims fall into two groups:

- Batching dims (``start_indices_batching_dims``) come from ``vmap``.
  Each pairs with an operand batching dim (``operand_batching_dims``) of the same size,
  and the index vectors at position ``b`` along it only address operand slice ``b``.
- All remaining dims, the "si batch axes", enumerate index vectors,
  like the dims of ``idx`` in ``x[idx]``.

For example, ``vmap(lambda row, i: row[i])(x, idx)``
with ``x.shape = (2, 5)`` and ``idx.shape = (2, 3)``
gathers with ``start_indices`` of shape (2, 3, 1).
Dim 0 is the vmapped axis, a batching dim paired with operand dim 0,
so row ``b`` of ``idx`` only indexes into row ``b`` of ``x``.
Dim 1 is an si batch axis enumerating the three index vectors of each row.
Dim 2 holds the index vectors, each with a single component.
"""

from collections.abc import Iterator, Sequence

import numpy as np
from jax._src.core import JaxprEqn

from ._common import (
    IndexSet,
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
    """Start-indices dims that enumerate index vectors.

    These are all dims except the trailing index-vector dim
    and the batching dims, which ``vmap`` pairs with operand dims.
    See the module docstring for the start-indices layout.

    Args:
        si_shape: Shape of the start indices.
        si_batching_dims: Start-indices dims that pair with operand batching dims.

    Example: ``x[idx]`` with ``idx.shape = (2,)``
        si_shape = (2, 1), si_batching_dims = ()
        Dim 1 holds the index vectors, so dim 0 enumerates them.
        Returns [0].

    Example: ``x[idx]`` with ``idx.shape = (2, 2)``
        si_shape = (2, 2, 1), si_batching_dims = ()
        Dim 2 holds the index vectors, so dims 0 and 1 enumerate them.
        Returns [0, 1].

    Example: ``vmap(lambda row, i: row[i])(x, idx)`` with ``idx.shape = (2, 3)``
        si_shape = (2, 3, 1), si_batching_dims = (0,)
        Dim 0 is the vmapped axis and dim 2 holds the index vectors.
        Returns [1].
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
    """Shapes of the two nested loops in ``_iter_si_starts``.

    The outer loop runs over the batching dims (from ``vmap``),
    the inner loop over the index vectors within one batch.
    Together, these shapes are the leading axes of the intermediate gather result.
    See the module docstring for the start-indices layout.

    Args:
        concrete_indices: Start indices, with the index vectors in the trailing dim.
        operand_shape: Shape of the operand that is gathered from or scattered into.
        operand_batching_dims: Operand dims that pair with ``si_batching_dims``.
        si_batching_dims: Start-indices dims that pair with ``operand_batching_dims``.

    Returns:
        ``(batching_shape, si_batch_shape)``.
        ``batching_shape`` holds the sizes of the batching dims,
        and ``si_batch_shape`` the sizes of the ``_si_batch_axes``.

    Example: ``x[idx]`` with ``x.shape = (5, 4)`` and ``idx.shape = (2,)``
        concrete_indices.shape = (2, 1), no batching dims
        Without ``vmap`` there are no batching dims, so batching_shape = ().
        ``idx`` holds two index vectors, so si_batch_shape = (2,).
        Returns ((), (2,)).

    Example: ``x[idx]`` with ``x.shape = (5, 4)`` and ``idx.shape = (2, 2)``
        concrete_indices.shape = (2, 2, 1), no batching dims
        ``idx`` holds a (2, 2) grid of index vectors.
        Returns ((), (2, 2)).

    Example: ``vmap(lambda row, i: row[i])(x, idx)``
        with ``x.shape = (2, 5)`` and ``idx.shape = (2, 3)``
        concrete_indices.shape = (2, 3, 1),
        operand_batching_dims = (0,), si_batching_dims = (0,)
        The vmapped axis has size 2, so batching_shape = (2,).
        Each row of ``idx`` holds three index vectors, so si_batch_shape = (3,).
        Returns ((2,), (3,)).
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
    """Operand start position of every index vector, for gather and scatter.

    Loops over the batching dims (from ``vmap``) on the outside
    and over the index vectors within one batch on the inside,
    with the loop shapes given by ``_si_batch_shapes``.
    See the module docstring for the start-indices layout.

    Args:
        concrete_indices: Start indices, with the index vectors in the trailing dim.
        operand_shape: Shape of the operand that is gathered from or scattered into.
        operand_batching_dims: Operand dims that pair with ``si_batching_dims``.
        si_batching_dims: Start-indices dims that pair with ``operand_batching_dims``.
        index_map: Operand dim addressed by each index-vector component
            (``start_index_map`` for gather,
            ``scatter_dims_to_operand_dims`` for scatter).

    Yields:
        ``(batch_idx, si_batch_idx, start)`` triples in row-major order.
        ``batch_idx`` is the position along the batching dims,
        which is ``()`` without ``vmap``.
        ``si_batch_idx`` is the index vector's position along ``_si_batch_axes``.
        ``start`` is the operand position the index vector addresses.
        Its operand batching dims are set to ``batch_idx``,
        and dims addressed by neither are 0.
        Starts are not clamped, since gather (always) and scatter (mode='clip')
        apply their own out-of-bounds policy.

    Example: ``x[idx]`` with ``x.shape = (5, 4)`` and ``idx = [1, 3]``
        concrete_indices = [[1], [3]], index_map = (0,), no batching dims
        Each index vector has one component, which addresses operand dim 0.
        The index vector [1] selects row 1 of ``x``, which starts at [1, 0].
        Yields ((), (0,), [1, 0]) and ((), (1,), [3, 0]).

    Example: ``vmap(lambda row, i: row[i])(x, idx)``
        with ``x.shape = (2, 5)`` and ``idx = [[4, 0, 2], [1, 1, 3]]``
        concrete_indices = [[[4], [0], [2]], [[1], [1], [3]]], index_map = (1,),
        operand_batching_dims = (0,), si_batching_dims = (0,)
        ``batch_idx = (b,)`` selects row ``b`` of both ``x`` and ``idx``,
        so ``start[0] = b`` and ``start[1]`` comes from the index vector.
        For b = 0, the index vectors [4], [0], [2] yield
            ((0,), (0,), [0, 4]), ((0,), (1,), [0, 0]), ((0,), (2,), [0, 2]).
        For b = 1, the index vectors [1], [1], [3] yield
            ((1,), (0,), [1, 1]), ((1,), (1,), [1, 1]), ((1,), (2,), [1, 3]).
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

        def _make(vals: tuple[int, ...]) -> list[IndexSet]:
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
