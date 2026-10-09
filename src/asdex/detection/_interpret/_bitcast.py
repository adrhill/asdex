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

    The output is not a smooth function of the input values.
    A tiny change to a float can flip its exponent bits
    and change the reinterpreted value by orders of magnitude.
    JAX therefore registers a zero JVP (``ad.defjvp_zero``),
    and ``jax.jacobian`` of a bitcast is all zeros.
    Detection mirrors that, so the output has no dependencies on the input.

    Math:
        ∂y/∂x = 0

    Changing the element width changes the shape,
    so the output can have more or fewer elements than the input.
    Let ``r`` be the ratio of the wider to the narrower bit width.
    Narrowing (e.g. f32 to i8, ``r = 4``) splits each input element
    into ``r`` narrower ones along a new trailing axis,
    so ``x[..., i]`` becomes ``y[..., i, 0:r]``.
    Widening (e.g. i8 to f32, ``r = 4``) requires a trailing axis of size ``r``
    and joins it into one wider element,
    so ``x[..., i, 0:r]`` becomes ``y[..., i]``.
    Since every output element still gets an empty index set,
    the handler sizes the output from the outvar, not from the input.

    Example: x = [a, b] is f32[2], y = bitcast_convert_type(x, int8)
        y = [[a0, a1, a2, a3], [b0, b1, b2, b3]] is i8[2, 4],
        where a0 is the least significant byte of a.
        Input index sets:  [{0}, {1}]  (x is the function input, a is input 0)
        Output index sets: [{}, {}, {}, {}, {}, {}, {}, {}]
        Each byte of a comes from a, but none depends on input 0, since ∂y/∂x = 0.

    Example: x = [[a0, a1], [b0, b1], [c0, c1]] is f16[3, 2],
             y = bitcast_convert_type(x, float32)
        y = [a, b, c] is f32[3], where a joins the halves a0 and a1.
        Input index sets:  [{0}, {1}, {2}, {3}, {4}, {5}]  (one per half)
        Output index sets: [{}, {}, {}]

    Also propagates const values,
    since a bitcast const can feed a gather index.
    The value must keep its bits rather than be converted,
    e.g. the float32 subnormal 1.4e-45 has the bit pattern of int32 1, not 0.
    Narrowing splits each element into bytes in host order,
    which is little-endian (least significant first) on the hardware JAX targets.
    numpy's ``view`` reinterprets the same way,
    so ``x.view(new_dtype)`` reshaped to the outvar's shape matches XLA.
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
