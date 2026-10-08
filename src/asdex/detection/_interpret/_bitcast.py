"""Propagation rule for bitcast_convert_type."""

import numpy as np
from jax._src.core import JaxprEqn

from ._common import (
    _atom_const_val,
    _atom_numel,
    _atom_shape,
    _empty_index_sets,
    _PropState,
)


def _prop_bitcast_convert_type(eqn: JaxprEqn, state: _PropState) -> None:
    """Bitcast reinterprets the bits of each element as another dtype.

    JAX defines its derivative as zero,
    so the output has no dependencies on the input.

    Changing the element width changes the shape.
    Narrowing to a dtype ``r`` times smaller appends a trailing axis of size ``r``,
    and widening by ``r`` consumes a trailing axis of size ``r``.

    Example: y = bitcast_convert_type(x, int8) where x is f32[2]
        Input index sets:  [{0}, {1}]
        Output index sets: [{}, {}, {}, {}, {}, {}, {}, {}]  (i8[2, 4])

    Also propagates const values,
    reinterpreting their bytes rather than converting their values.
    Bounds are not propagated because reinterpretation is not monotone.

    Jaxpr:
        invars[0]: input array
        new_dtype: target dtype

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.bitcast_convert_type.html
    """
    out_var = eqn.outvars[0]
    state.indices[out_var] = _empty_index_sets(_atom_numel(out_var))

    in_val = _atom_const_val(eqn.invars[0], state)
    in_dtype = getattr(eqn.invars[0].aval, "dtype", None)
    if in_val is not None and in_dtype is not None:
        # Literals and seeded consts may not have the aval's dtype yet,
        # and the bytes are only meaningful once they do.
        in_val = np.ascontiguousarray(in_val, dtype=in_dtype)
        state.consts[out_var] = in_val.view(eqn.params["new_dtype"]).reshape(
            _atom_shape(out_var)
        )
