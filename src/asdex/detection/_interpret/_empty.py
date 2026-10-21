"""Propagation rule for empty."""

from jax._src.core import JaxprEqn

from ._common import _empty_index_sets, _numel, _PropState


def _prop_empty(eqn: JaxprEqn, state: _PropState) -> None:
    """Empty allocates an uninitialized array with no input dependencies.

    The output is not computed from any input,
    so ∂y/∂x = 0 for every input x and all dependency sets are empty.
    Unlike ``iota``, no const value is recorded:
    the contents are undefined, so treating them as known values
    (e.g. as zeros that let ``mul`` or ``dot_general`` skip entries)
    could drop nonzeros once the array is filled.
    JAX emits this primitive for ``jnp.empty`` and ``jnp.empty_like`` since 0.11,
    which used to lower to a broadcast of zeros.

    Example: y = jnp.empty((2,))
        Output index sets: [{}, {}]

    Jaxpr:
        invars: [] (no inputs, even for ``jnp.empty_like``)
        shape: output shape
        dtype: output dtype

    https://docs.jax.dev/en/latest/_autosummary/jax.numpy.empty.html
    """
    state.indices[eqn.outvars[0]] = _empty_index_sets(_numel(eqn.params["shape"]))
