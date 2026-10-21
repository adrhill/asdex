"""Propagation rules for windowed reductions."""

import numpy as np
from jax._src.core import JaxprEqn

from ._common import (
    _atom_shape,
    _index_sets,
    _numel,
    _PropState,
    _row_strides,
    _union_all,
)


def _window_targets(
    out_shape: tuple[int, ...],
    in_shape: tuple[int, ...],
    window_dimensions: tuple[int, ...],
    window_strides: tuple[int, ...],
    window_dilation: tuple[int, ...],
    base_dilation: tuple[int, ...],
    padding: tuple[tuple[int, int], ...],
) -> list[list[int]]:
    """Map each output position to the flat input positions inside its window.

    For output coordinate ``o`` and tap coordinate ``k`` along a dimension,
    the position in the base-dilated, padded input is
    ``o * stride + k * window_dilation - padding_lo``.
    A tap is in the window when that position is nonnegative,
    lands on an actual input element rather than in a base-dilation gap,
    and is smaller than the input size.
    Padding and base-dilation gaps contribute no dependencies,
    so they are masked out.
    """
    ndim = len(in_shape)
    out_size = _numel(out_shape)
    window_size = _numel(window_dimensions)

    out_coords = np.indices(out_shape).reshape(ndim, out_size)
    tap_coords = np.indices(window_dimensions).reshape(ndim, window_size)
    strides = np.asarray(window_strides, dtype=np.intp)[:, None, None]
    dilation = np.asarray(window_dilation, dtype=np.intp)[:, None, None]
    base = np.asarray(base_dilation, dtype=np.intp)[:, None, None]
    low = np.asarray([lo for lo, _ in padding], dtype=np.intp)[:, None, None]
    sizes = np.asarray(in_shape, dtype=np.intp)[:, None, None]

    pos = out_coords[:, :, None] * strides + tap_coords[:, None, :] * dilation - low
    in_coords = pos // base
    valid = ((pos >= 0) & (pos % base == 0) & (in_coords < sizes)).all(axis=0)
    strides_flat = np.asarray(_row_strides(in_shape), dtype=np.intp)
    offsets = (in_coords * strides_flat[:, None, None]).sum(axis=0)
    return [offsets[o, valid[o]].tolist() for o in range(out_size)]


def _prop_reduce_window(eqn: JaxprEqn, state: _PropState) -> None:
    """Windowed reductions aggregate a window of input elements per output.

    For sum, max, min, and product, every input position inside an output's
    window can affect that output, so the Jacobian row is the union of the
    index sets at those positions.
    Windows can be strided, dilated, padded, and based on a dilated input.

    Example: y = reduce_window(x, 0, add, (2,), (1,)) with x = [a, b, c]
        Input index sets:  [{0}, {1}, {2}]
        Output index sets: [{0, 1}, {1, 2}]

    Generic ``reduce_window`` unions the whole window for every operand,
    even when the combiner could drop some of them (e.g. a select-style
    combiner), and includes the init values in every output.
    Multi-operand reductions similarly union all operands conservatively.

    Jaxpr:
        invars: operand arrays, followed by one scalar init value per output
        outvars: one output array per operand
        jaxpr: combiner over the scalar window elements (generic only)
        consts, window_dimensions, window_strides, padding, base_dilation,
        window_dilation: window geometry

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce_window.html
    """
    out_shape = _atom_shape(eqn.outvars[0])
    out_size = _numel(out_shape)
    if out_size == 0:
        for outvar in eqn.outvars:
            state.indices[outvar] = []
        return

    window_dimensions = eqn.params["window_dimensions"]
    window_strides = eqn.params["window_strides"]
    window_dilation = eqn.params["window_dilation"]
    base_dilation = eqn.params["base_dilation"]
    padding = eqn.params["padding"]

    # The generic primitive takes one scalar init value per output operand.
    num_outputs = len(eqn.outvars)
    operands = eqn.invars[:num_outputs]
    inits = eqn.invars[num_outputs:]
    operand_indices = [_index_sets(state, atom) for atom in operands]
    init_indices = [_index_sets(state, atom) for atom in inits]
    init_dep = _union_all([s for indices in init_indices for s in indices])

    if len(operands) == 1:
        in_indices = operand_indices[0]
        targets = _window_targets(
            out_shape,
            _atom_shape(operands[0]),
            window_dimensions,
            window_strides,
            window_dilation,
            base_dilation,
            padding,
        )
        out_indices = [
            in_indices[window[0]]
            if len(window) == 1
            else _union_all([in_indices[t] for t in window])
            for window in targets
        ]
    else:
        target_lists = [
            _window_targets(
                out_shape,
                _atom_shape(atom),
                window_dimensions,
                window_strides,
                window_dilation,
                base_dilation,
                padding,
            )
            for atom in operands
        ]
        out_indices = [
            _union_all(
                [
                    indices[t]
                    for indices, targets in zip(
                        operand_indices, target_lists, strict=True
                    )
                    for t in targets[o]
                ]
            )
            for o in range(out_size)
        ]

    if init_dep:
        out_indices = [s | init_dep for s in out_indices]
    for outvar in eqn.outvars:
        state.indices[outvar] = out_indices
