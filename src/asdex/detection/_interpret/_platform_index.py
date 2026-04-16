"""Handler for the ``platform_index`` primitive.

``platform_index`` returns a scalar i32 indicating which platform is active.
It takes no inputs and its output is fully determined at runtime,
so all dependency sets are empty.

No ``state_consts`` tracking is needed,
because the value is platform-dependent and unknown at trace time.

Jaxpr:
    invars: [] (no inputs)
    outvars: [scalar i32]
    params: platforms (tuple of platform strings)

https://docs.jax.dev/en/latest/_autosummary/jax.lax.platform_dependent.html
"""

from jax._src.core import JaxprEqn

from ._commons import StateIndices, empty_index_sets


def prop_platform_index(eqn: JaxprEqn, state_indices: StateIndices) -> None:
    """Platform index produces a constant scalar with no input dependencies.

    The output is a scalar integer selecting the active platform.
    Since it depends on no inputs,
    the single-element dependency list contains an empty set.

    Math:
        J = [] (no inputs, so no Jacobian)

    Example:
        invars: []
        outvars: [c]
        state_indices[c] = [{}]

    Jaxpr:
        invars: [] (no inputs)
        outvars: [scalar i32]
        params: platforms

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.platform_dependent.html
    """
    state_indices[eqn.outvars[0]] = empty_index_sets(1)
