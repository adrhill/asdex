"""Propagation rules for the empty array primitives."""

from jax._src.core import JaxprEqn

from ._common import _atom_numel, _empty_index_sets, _PropState


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

    The internal ``empty2`` primitive allocates the same kind of buffer,
    a scalar of the given dtype,
    and AD emits it for the undefined residuals of a ``cond`` branch
    (``ad_util.empty_like_aval``),
    so it shows up when differentiating a ``cond``
    and therefore in Hessian detection.

    Example: y = jnp.empty((2,))
        Output index sets: [{}, {}]

    Jaxpr:
        invars: [] (no inputs, even for ``jnp.empty_like``)
        outvars[0]: the allocated array
        params: ``shape`` and ``dtype`` for ``empty``,
            only ``dtype`` and ``memory_space`` for ``empty2``

    https://docs.jax.dev/en/latest/_autosummary/jax.numpy.empty.html
    """
    # The element count comes from the output variable because ``empty2``
    # carries no ``shape`` parameter, only a dtype.
    state.indices[eqn.outvars[0]] = _empty_index_sets(_atom_numel(eqn.outvars[0]))
