"""Propagation rule for the ``fft`` primitive."""

from jax._src.core import JaxprEqn

from ._common import (
    IndexSet,
    _atom_shape,
    _empty_index_sets,
    _index_sets,
    _numel,
    _PropState,
)


def _prop_fft(eqn: JaxprEqn, state: _PropState) -> None:
    """A discrete Fourier transform mixes every element along the transformed axes.

    The primitive always transforms the trailing ``len(fft_lengths)`` axes,
    so the array splits into one block per position of the leading untransformed axes,
    and the Jacobian is block diagonal: each output element of a block
    is a weighted sum of every element of the same input block,
    so the block is dense and the untransformed axes stay independent.
    All four ``fft_type`` values share this structure,
    since they only differ in the size of the blocks
    (``RFFT`` halves and ``IRFFT`` doubles the last transformed axis),
    so the handler reads shapes but not ``fft_type``.

    Example: y = fft(x) where x.shape = (2, 3)
        Input index sets:  [{0}, {1}, {2}, {3}, {4}, {5}]
        The trailing transformed axis has length 3,
        so every row of 3 is one block.
        Output index sets: [{0, 1, 2}, {0, 1, 2}, {0, 1, 2},
                            {3, 4, 5}, {3, 4, 5}, {3, 4, 5}]

    Jaxpr:
        invars[0]: input array
        fft_lengths: lengths of the transformed trailing axes
        fft_type: FFT, IFFT, RFFT, or IRFFT

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.fft.html
    """
    in_indices = _index_sets(state, eqn.invars[0])
    in_shape = _atom_shape(eqn.invars[0])
    out_shape = _atom_shape(eqn.outvars[0])
    out_size = _numel(out_shape)
    if out_size == 0:
        state.indices[eqn.outvars[0]] = []
        return

    n_transformed = len(eqn.params["fft_lengths"])
    in_block_size = _numel(in_shape[-n_transformed:])
    out_block_size = _numel(out_shape[-n_transformed:])

    block_indices: list[IndexSet] = _empty_index_sets(out_size // out_block_size)
    for flat, index_set in enumerate(in_indices):
        block_indices[flat // in_block_size] |= index_set

    state.indices[eqn.outvars[0]] = [
        block_indices[flat // out_block_size] for flat in range(out_size)
    ]
