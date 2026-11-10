"""Propagation rules for reductions.

``reduce_sum``, ``reduce_max``, ``reduce_min``, and ``reduce_prod``
share the same sparsity structure:
each output depends on all inputs along the reduced axes.
The generic ``reduce`` applies a user-supplied computation to several operands at once.
"""

import numpy as np
from jax._src.core import JaxprEqn

from ._common import (
    IndexSet,
    _atom_shape,
    _empty_index_sets,
    _index_sets,
    _numel,
    _PropState,
    _union_all,
    _union_elementwise,
)


def _prop_reduce(eqn: JaxprEqn, state: _PropState) -> None:
    """Reduction aggregates elements along specified axes.

    Each output depends on all input elements that reduce into it.
    This applies uniformly to reduce_sum, reduce_max, reduce_min, and reduce_prod:
    in all cases, changing any input in a reduction group
    can change the output.

    Full reduction (no axes or all axes):
        out = reduce(x)  →  out depends on all inputs
    Partial reduction along axis k:
        out[i] = reduce_j x[i, j]  →  out[i] depends on all x[i, :]

    Example: y = sum(x, axis=1) where x.shape = (2, 3)
        Input index sets:  [{0}, {1}, {2}, {3}, {4}, {5}]
        Output index sets: [{0, 1, 2}, {3, 4, 5}]  (one set per row)

    Jaxpr:
        invars[0]: input array
        axes: tuple of axes to reduce

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce_sum.html
    https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce_max.html
    https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce_min.html
    https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce_prod.html
    """
    state.indices[eqn.outvars[0]] = _reduce_index_sets(
        _index_sets(state, eqn.invars[0]),
        _atom_shape(eqn.invars[0]),
        eqn.params.get("axes", ()),
    )


def _prop_reduce_generic(eqn: JaxprEqn, state: _PropState) -> None:
    """Generic reduction folds a user-supplied computation over several operands.

    ``lax.reduce`` reduces ``N`` operands of equal shape along the same axes at once.
    The computation maps ``N`` accumulators and ``N`` elements to ``N`` new accumulators,
    so it may mix the operands,
    and each accumulator starts from its init value.
    Each output element therefore depends on the elements of all operands
    that reduce into it, and on all init values.

    For ``N`` operands reduced along axis k:
        outs[n][i] = fold_j computation(acc, (x_1[i, j], ..., x_N[i, j]))[n]
        outs[n][i] depends on all x_m[i, :] and all init_m

    Example: lo, hi = jnp.minmax(x, axis=1) where x.shape = (2, 3)
        JAX passes x as both operands, with inits inf and -inf.
        Input index sets:  [{0}, {1}, {2}, {3}, {4}, {5}]  (for both operands)
        Output index sets: [{0, 1, 2}, {3, 4, 5}]  (for both lo and hi)

    The computation is not inspected,
    so outputs that only use some of the operands get a conservative pattern.

    Jaxpr:
        invars[:N]: operands, all of the same shape
        invars[N:]: scalar init values, one per operand
        outvars: N reduced outputs
        dimensions: tuple of axes to reduce
        jaxpr: the computation, mapping (accs..., elements...) to new accs

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce.html
    """
    num_operands = len(eqn.invars) // 2
    operands = eqn.invars[:num_operands]
    inits = eqn.invars[num_operands:]
    axes = eqn.params["dimensions"]

    reduced = [
        _reduce_index_sets(_index_sets(state, operand), _atom_shape(operand), axes)
        for operand in operands
    ]
    # Scalar inits broadcast over the output via the modular indexing of _union_elementwise.
    init_indices = [_index_sets(state, init) for init in inits]
    out_indices = _union_elementwise(reduced + init_indices, len(reduced[0]))
    for outvar in eqn.outvars:
        state.indices[outvar] = out_indices


def _reduce_index_sets(
    in_indices: list[IndexSet], in_shape: tuple[int, ...], axes: tuple[int, ...]
) -> list[IndexSet]:
    """Union the index sets of all input elements that reduce into each output element."""
    # Full reduction: single output depends on all inputs
    if len(axes) == len(in_shape):
        return [_union_all(in_indices)]

    # Partial reduction: group input elements by their non-reduced coordinates.
    # Build a flat map from each input element to its output group,
    # then union input index sets into the corresponding output set.
    kept_dims = [d for d in range(len(in_shape)) if d not in axes]
    out_shape = tuple(in_shape[d] for d in kept_dims)
    out_size = _numel(out_shape)

    # For each input element, project to output coordinates (drop reduced dims)
    in_coords = np.indices(in_shape)
    out_coords = tuple(in_coords[d] for d in kept_dims)
    group_map = np.ravel_multi_index(out_coords, out_shape).ravel()

    out_indices: list[IndexSet] = _empty_index_sets(out_size)
    for in_flat, elem_deps in enumerate(in_indices):
        out_indices[group_map[in_flat]] |= elem_deps

    return out_indices
