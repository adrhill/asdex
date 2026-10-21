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

    The primitive always transforms the trailing ``len(fft_lengths)`` axes.
    Every output element along those axes is a weighted sum
    of all input elements along the same axes,
    while the leading batch axes are independent.
    This holds for ``FFT``, ``IFFT``, ``RFFT``, and ``IRFFT``:
    each output in a batch block depends on every input in that block,
    including both real and imaginary parts of a complex input.

    For a 1D transform with input [a, b, c, d]:
        out[k] = Σ_n aₙ · exp(-2πikn/N)  →  out[k] depends on {0, 1, 2, 3}

    Example: y = fft(x) where x.shape = (2, 3)
        Input index sets:  [{0}, {1}, {2}, {3}, {4}, {5}]
        Output index sets: [{0, 1, 2}, {0, 1, 2}, {0, 1, 2},
                            {3, 4, 5}, {3, 4, 5}, {3, 4, 5}]
        The two rows are independent batch blocks.

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
    n_blocks = out_size // out_block_size

    block_indices: list[IndexSet] = _empty_index_sets(n_blocks)
    if in_block_size:
        for flat, index_set in enumerate(in_indices):
            block_indices[flat // in_block_size] |= index_set

    state.indices[eqn.outvars[0]] = [
        block_indices[flat // out_block_size] for flat in range(out_size)
    ]
