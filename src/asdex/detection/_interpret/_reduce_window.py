"""Propagation rule for reduce_window and its sum, max, and min specializations."""

from jax._src.core import JaxprEqn

from ._common import (
    IndexSet,
    _atom_shape,
    _index_sets,
    _numel,
    _PropState,
    _union_all,
    _window_target_lists,
)


def _prop_reduce_window(eqn: JaxprEqn, state: _PropState) -> None:
    """Windowed reductions aggregate a window of input elements per output.

    For ``reduce_window_sum``, ``reduce_window_max``, and ``reduce_window_min``,
    the Jacobian row of an output is the union of the index sets
    of the input positions inside its window:
        out[o] = sum/max/min_{k in window(o)} x[o * window_strides + k * window_dilation]
    Patterns are global, so every position in the window matters:
    any of them can be the element that changes the reduction.

    Windows can be strided, dilated, padded, and applied to a base-dilated input.

    Example: y = reduce_window(x, 0, add, (2,), (1,)) with x = [a, b, c]
        Input index sets:  [{0}, {1}, {2}]
        Output index sets: [{0, 1}, {1, 2}]

    Generic ``reduce_window`` reduces with an arbitrary combiner jaxpr,
    which may ignore part of its input:
    a ``select``-style combiner returns one of its two arguments,
    so not every element of the window reaches the output.
    Without analyzing the combiner, the handler unions
    the whole window of every operand into every output,
    and adds the dependencies of the init values.
    That is exact for combiners that read their whole window (add, mul, max, min)
    and conservative otherwise, since a missed nonzero is not an option.

    Jaxpr:
        invars: one operand per output, followed by one scalar init value per output
        outvars: one output array per operand
        jaxpr, consts: combiner over the scalar window elements
            (only the generic primitive has them;
            the specializations inline their identity init)
        window_dimensions, window_strides, padding, base_dilation,
        window_dilation: window geometry

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.reduce_window.html
    """
    out_shape = _atom_shape(eqn.outvars[0])
    out_size = _numel(out_shape)
    if out_size == 0:
        for outvar in eqn.outvars:
            state.indices[outvar] = []
        return

    # The generic primitive appends one scalar init value per operand,
    # while the sum/max/min specializations carry operands only.
    num_operands = len(eqn.outvars)
    operands = eqn.invars[:num_operands]
    init_indices = _union_all(
        [
            in_set
            for atom in eqn.invars[num_operands:]
            for in_set in _index_sets(state, atom)
        ]
    )

    # JAX derives the output shape from the first operand only,
    # so every operand is mapped against its own input shape.
    target_lists = [
        _window_target_lists(
            out_shape,
            eqn.params["window_dimensions"],
            _atom_shape(atom),
            eqn.params["window_strides"],
            eqn.params["base_dilation"],
            eqn.params["window_dilation"],
            eqn.params["padding"],
        )
        for atom in operands
    ]
    operand_indices = [_index_sets(state, atom) for atom in operands]

    out_indices: list[IndexSet] = [
        _union_all(
            [
                indices[t]
                for indices, targets in zip(operand_indices, target_lists, strict=True)
                for t in targets[o]
            ]
        )
        for o in range(out_size)
    ]
    if init_indices:
        out_indices = [out_set | init_indices for out_set in out_indices]
    for outvar in eqn.outvars:
        state.indices[outvar] = out_indices
