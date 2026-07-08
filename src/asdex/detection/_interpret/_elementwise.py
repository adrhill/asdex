"""Propagation rules for element-wise operations."""

from collections.abc import Callable

import numpy as np
from jax._src.core import JaxprEqn

from ._common import (
    _atom_const_val,
    _atom_numel,
    _atom_shape,
    _atom_value_bounds,
    _broadcast_flat_map,
    _clear_where_zero,
    _empty_index_set,
    _empty_index_sets,
    _index_sets,
    _numel,
    _propagate_const_binary,
    _propagate_const_unary,
    _PropState,
    _union_elementwise,
)


def _lax_div(in1_val: np.ndarray, in2_val: np.ndarray) -> np.ndarray:
    """Divide with ``lax.div`` semantics.

    ``lax.div`` truncates toward zero for integer inputs,
    while ``np.divide`` is true division and returns floats.
    Using numpy semantics on integer index arithmetic
    would resolve gather/scatter indices to the wrong positions.
    """
    result_dtype = np.result_type(in1_val, in2_val)
    if np.issubdtype(result_dtype, np.integer):
        return np.trunc(np.true_divide(in1_val, in2_val)).astype(result_dtype)
    return np.true_divide(in1_val, in2_val)


# Functions for evaluating constant values during tracing.
# Used to propagate static index values through arithmetic to gather/scatter.
# Entries must match lax semantics, which differ from numpy for integer div/rem.
_BINARY_CONST_UFUNCS: dict[str, Callable[[np.ndarray, np.ndarray], np.ndarray]] = {
    # arithmetic
    "add": np.add,
    "add_any": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "div": _lax_div,
    "pow": np.power,
    "max": np.maximum,
    "min": np.minimum,
    "atan2": np.arctan2,
    # lax.rem takes the dividend's sign like C fmod,
    # while np.remainder takes the divisor's sign.
    "rem": np.fmod,
    "nextafter": np.nextafter,
    # comparison
    "eq": np.equal,
    "ne": np.not_equal,
    # bitwise
    "and": np.bitwise_and,
    "or": np.bitwise_or,
    "xor": np.bitwise_xor,
}

# Unary zero-derivative primitives whose const values feed integer index arithmetic
# (e.g. ``jnp.floor_divide`` expands to div/sign/rem/select_n).
# ``round`` is excluded: ``lax.round`` rounding methods differ from
# numpy's round-half-to-even, and a mismatched const yields a wrong pattern.
_UNARY_CONST_UFUNCS: dict[str, Callable[[np.ndarray], np.ndarray]] = {
    "sign": np.sign,
    "floor": np.floor,
    "ceil": np.ceil,
    # matches lax: bitwise for integers, logical for booleans
    "not": np.bitwise_not,
}


def _propagate_const(eqn: JaxprEqn, state: _PropState) -> None:
    """Propagate a const value through a binary elementwise op.

    If both inputs are statically known,
    apply the matching numpy ufunc and store the result.
    Without this, downstream handlers (e.g. ``gather``, ``scatter``) cannot resolve
    static index arrays and fall back to conservative.
    """
    ufunc = _BINARY_CONST_UFUNCS.get(eqn.primitive.name)
    if ufunc is not None:
        _propagate_const_binary(eqn, state, ufunc)


# Building blocks (private)


def _zero_derivative(eqn: JaxprEqn, state: _PropState) -> None:
    """Set empty index sets for zero-derivative outputs."""
    for outvar in eqn.outvars:
        state.indices[outvar] = _empty_index_sets(_atom_numel(outvar))


def _binary_elementwise(
    eqn: JaxprEqn,
    state: _PropState,
    *,
    is_der1_zero_globally: bool = False,
    is_der2_zero_globally: bool = False,
) -> None:
    """Union per-element index sets from two inputs.

    If ``is_der1_zero_globally`` is True, ∂f/∂x₁ = 0 everywhere,
    so the first input doesn't contribute to output dependencies.
    Likewise for ``is_der2_zero_globally`` and the second input.
    """
    in1 = _index_sets(state, eqn.invars[0])
    in2 = _index_sets(state, eqn.invars[1])
    out_size = 0 if len(in1) == 0 or len(in2) == 0 else max(len(in1), len(in2))

    in1_shape = _atom_shape(eqn.invars[0])
    in2_shape = _atom_shape(eqn.invars[1])

    # Fast path: same shape or scalar.
    # Modular indexing handles both correctly:
    # i % len == i for same size, i % 1 == 0 for scalar.
    if in1_shape == in2_shape or len(in1) <= 1 or len(in2) <= 1:
        state.indices[eqn.outvars[0]] = [
            _union_with_zero_derivs(
                in1[i % len(in1)],
                in2[i % len(in2)],
                is_der1_zero_globally,
                is_der2_zero_globally,
            )
            for i in range(out_size)
        ]
        return

    # General broadcast: map output positions to input positions
    # respecting numpy-style broadcasting (size-1 dims read index 0).
    # Example: mul of (16,16) * (16,1) → (16,16).
    # out[p,d] depends on in1[p,d] and in2[p,0], not in2[d].
    out_shape = _atom_shape(eqn.outvars[0])
    out_size = _numel(out_shape)
    in1_flat = _broadcast_flat_map(in1_shape, out_shape)
    in2_flat = _broadcast_flat_map(in2_shape, out_shape)

    state.indices[eqn.outvars[0]] = [
        _union_with_zero_derivs(
            in1[in1_flat[i]],
            in2[in2_flat[i]],
            is_der1_zero_globally,
            is_der2_zero_globally,
        )
        for i in range(out_size)
    ]


def _union_with_zero_derivs(
    s1: set[int],
    s2: set[int],
    is_der1_zero: bool,
    is_der2_zero: bool,
) -> set[int]:
    """Union index sets, excluding inputs with zero derivatives.

    The result may alias an input set,
    which is safe since index sets are never mutated.
    Aliasing instead of unioning when one side is empty
    avoids an allocation per element
    for ops with a constant operand (e.g. ``x * 2.0``).
    """
    if is_der1_zero and is_der2_zero:
        return _empty_index_set()
    if is_der1_zero or not s1:
        return s2
    if is_der2_zero or not s2:
        return s1
    return s1 | s2


def _propagate_bounds_add(
    eqn: JaxprEqn,
    state: _PropState,
) -> None:
    """Propagate value bounds through ``add`` or ``add_any`` via interval arithmetic."""
    b1 = _atom_value_bounds(eqn.invars[0], state)
    b2 = _atom_value_bounds(eqn.invars[1], state)
    if b1 is None or b2 is None:
        return
    lo1, hi1 = b1
    lo2, hi2 = b2
    state.bounds[eqn.outvars[0]] = (lo1 + lo2, hi1 + hi2)


def _propagate_bounds_sub(
    eqn: JaxprEqn,
    state: _PropState,
) -> None:
    """Propagate value bounds through ``sub`` via interval arithmetic."""
    b1 = _atom_value_bounds(eqn.invars[0], state)
    b2 = _atom_value_bounds(eqn.invars[1], state)
    if b1 is None or b2 is None:
        return
    lo1, hi1 = b1
    lo2, hi2 = b2
    state.bounds[eqn.outvars[0]] = (lo1 - hi2, hi1 - lo2)


# Composite handlers (public)
# Each corresponds to exactly one dispatch case in _prop_dispatch.


def _prop_zero_derivative(eqn: JaxprEqn, state: _PropState) -> None:
    """Zero-derivative primitives (floor, ceil, sign, ...).

    Operations with zero derivative almost everywhere.
    Their outputs are piecewise constant,
    so infinitesimal input changes don't affect outputs.

    Mathematically, for f in {floor, ceil, sign, ...}:
        ∂f/∂x = 0  (almost everywhere)
    Therefore, output elements have no dependencies on input elements.

    Example: y = floor(x) where x = [1.7, 2.3, 3.9]
        Input index sets:  [{0}, {1}, {2}]
        Output index sets: [{}, {}, {}]  (empty sets, no dependence)
    """
    _zero_derivative(eqn, state)


def _prop_zero_derivative_const(eqn: JaxprEqn, state: _PropState) -> None:
    """Zero-derivative primitives that also propagate const values.

    Used for comparisons (eq, ne) and bitwise ops (and, or, xor)
    where the output is zero-derivative
    but the concrete result may be needed by downstream handlers.
    """
    _zero_derivative(eqn, state)
    _propagate_const(eqn, state)


def _prop_zero_derivative_unary_const(eqn: JaxprEqn, state: _PropState) -> None:
    """Unary zero-derivative primitives that also propagate const values.

    Used for sign, floor, ceil, and not,
    which appear in integer index arithmetic
    (e.g. inside the ``jnp.floor_divide`` expansion).
    Without const propagation here the chain breaks
    and downstream gather/scatter falls back to conservative.
    """
    _zero_derivative(eqn, state)
    ufunc = _UNARY_CONST_UFUNCS.get(eqn.primitive.name)
    if ufunc is not None:
        _propagate_const_unary(eqn, state, ufunc)


def _prop_ternary_elementwise(eqn: JaxprEqn, state: _PropState) -> None:
    """Ternary elementwise operation where each output depends on all three inputs.

    Used for `regularized_incomplete_beta(a, b, x)` where each output element
    depends on the corresponding elements from all three input arrays.
    Handles broadcasting via modular indexing.

    Example: z = betainc(a, b, x) where a, b, x are arrays of shape (3,)
        Input index sets:  [{0}, {1}, {2}], [{3}, {4}, {5}], [{6}, {7}, {8}]
        Output index sets: [{0, 3, 6}, {1, 4, 7}, {2, 5, 8}]

    Jaxpr:
        invars[0]: first input (a)
        invars[1]: second input (b)
        invars[2]: third input (x)
    """
    inputs = [_index_sets(state, invar) for invar in eqn.invars]
    out_size = _atom_numel(eqn.outvars[0])
    state.indices[eqn.outvars[0]] = _union_elementwise(inputs, out_size)


def _prop_binary_const(
    eqn: JaxprEqn,
    state: _PropState,
    *,
    is_der1_zero_globally: bool = False,
    is_der2_zero_globally: bool = False,
) -> None:
    """Binary elementwise primitives (div, pow, max, min, ...) with const propagation.

    Each output element depends on the corresponding elements from both inputs.
    Also propagates const values for downstream index resolution.

    For f(x, y) element-wise:
        ∂f/∂x[i] and ∂f/∂y[i] are generally nonzero.
    So out[i] depends on {x[i], y[i]} (union of dependencies).

    If ``is_der1_zero_globally`` is True, ∂f/∂x₁ = 0 everywhere,
    so only the second input contributes.
    Likewise for ``is_der2_zero_globally``.

    Example: z = x + y where x = [a, b], y = [c, d]
        Input index sets:  [{0}, {1}], [{2}, {3}]
        Output index sets: [{0, 2}, {1, 3}]

    Jaxpr:
        invars[0]: first input array
        invars[1]: second input array
    """
    _binary_elementwise(
        eqn,
        state,
        is_der1_zero_globally=is_der1_zero_globally,
        is_der2_zero_globally=is_der2_zero_globally,
    )
    _propagate_const(eqn, state)


def _prop_add(
    eqn: JaxprEqn,
    state: _PropState,
) -> None:
    """Add / add_any: binary elementwise with interval arithmetic bounds.

    ``[a,b] + [c,d] = [a+c, b+d]``.
    """
    _binary_elementwise(eqn, state)
    _propagate_const(eqn, state)
    _propagate_bounds_add(eqn, state)


def _prop_sub(
    eqn: JaxprEqn,
    state: _PropState,
) -> None:
    """Sub: binary elementwise with interval arithmetic bounds.

    ``[a,b] - [c,d] = [a-d, b-c]``.
    """
    _binary_elementwise(eqn, state)
    _propagate_const(eqn, state)
    _propagate_bounds_sub(eqn, state)


def _prop_integer_pow(
    eqn: JaxprEqn,
    state: _PropState,
) -> None:
    """Integer power x^n is element-wise.

    Each output depends only on the corresponding input element.
    Special cases:
    - x^0 = 1 has zero derivative, so no dependencies.
    - 0^n = 0 for n > 0, so d(0^n)/dx = 0 and no dependencies.

    ∂(x^n)/∂x = n·x^(n-1), which is zero iff n = 0 or (x = 0 and n > 1).

    Example: y = x^2 where x = [a, b, c]
        Input index sets:  [{0}, {1}, {2}]
        Output index sets: [{0}, {1}, {2}]  (or [{}, {}, {}] if n=0)

    Jaxpr:
        invars[0]: input array
        y: the integer exponent

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.integer_pow.html
    """
    y = eqn.params.get("y", 1)
    in_indices = _index_sets(state, eqn.invars[0])

    if y == 0:
        state.indices[eqn.outvars[0]] = _empty_index_sets(len(in_indices))
    else:
        # Aliasing the input list is safe: index sets are never mutated,
        # and _clear_where_zero below builds a new list.
        state.indices[eqn.outvars[0]] = in_indices

    # Const propagation.
    in_val = _atom_const_val(eqn.invars[0], state)
    if in_val is not None:
        state.consts[eqn.outvars[0]] = np.power(in_val, y)

    # Zero-skipping: d(0^n)/dx = n * 0^(n-1) = 0 for n > 1.
    # For n = 1, d(x)/dx = 1 even at x = 0, so no skipping.
    if y > 1:
        _clear_where_zero(eqn, state, 0)

    # Bounds propagation for [a,b]^n.
    _propagate_bounds_integer_pow(eqn, y, state)


def _propagate_bounds_integer_pow(
    eqn: JaxprEqn,
    y: int,
    state: _PropState,
) -> None:
    """Propagate value bounds through ``integer_pow``.

    - n < 0: no bounds propagated.
      Negative powers are decreasing (not increasing) on positive inputs,
      so the monotone mapping below would invert (lo, hi),
      and they are undefined at zero.
    - n == 0: bounds are (1, 1).
    - n even: [0, max(|a|,|b|)^n] if interval spans zero,
      else [min(|a|,|b|)^n, max(|a|,|b|)^n].
    - n odd (increasing): [a^n, b^n].
    """
    in_bounds = _atom_value_bounds(eqn.invars[0], state)
    if in_bounds is None:
        return

    lo, hi = in_bounds

    if y < 0:
        return
    if y == 0:
        ones = np.ones_like(lo)
        state.bounds[eqn.outvars[0]] = (ones, ones)
    elif y % 2 == 1:
        # Odd power is monotone.
        state.bounds[eqn.outvars[0]] = (np.power(lo, y), np.power(hi, y))
    else:
        # Even power: x^n is not monotone over intervals spanning zero.
        abs_lo = np.abs(lo)
        abs_hi = np.abs(hi)
        max_abs = np.maximum(abs_lo, abs_hi)
        min_abs = np.minimum(abs_lo, abs_hi)

        spans_zero = (lo <= 0) & (hi >= 0)
        out_lo = np.where(spans_zero, np.zeros_like(lo), np.power(min_abs, y))
        out_hi = np.power(max_abs, y)
        state.bounds[eqn.outvars[0]] = (out_lo, out_hi)


def _prop_unary_elementwise(eqn: JaxprEqn, state: _PropState) -> None:
    """Unary element-wise ops (exp, sin, etc.) apply a function to each element.

    Each output depends only on the corresponding input element.
    The Jacobian is diagonal.

    For f(x) element-wise:
        ∂f[i]/∂x[j] = f'(x[i]) if i = j, else 0

    Example: y = exp(x) where x = [a, b, c]
        Input index sets:  [{0}, {1}, {2}]
        Output index sets: [{0}, {1}, {2}]

    Jaxpr:
        invars[0]: input array
    """
    state.indices[eqn.outvars[0]] = _index_sets(state, eqn.invars[0])


def _prop_convert_element_type(
    eqn: JaxprEqn,
    state: _PropState,
) -> None:
    """Type conversion (e.g., float32 → float64) changes dtype without changing values.

    Dependencies pass through unchanged.
    The Jacobian is the identity matrix.

    Also propagates const values and value bounds with the new dtype
    so downstream gather/scatter can resolve static indices.
    JAX inserts ``convert_element_type`` for index dtype changes
    (e.g. int64 → int32) before gather/scatter;
    without const propagation here the chain breaks
    and gathers fall back to conservative.

    Example: y = x.astype(float64) where x = [a, b, c]
        Input index sets:  [{0}, {1}, {2}]
        Output index sets: [{0}, {1}, {2}]

    Jaxpr:
        invars[0]: input array
        new_dtype: target dtype

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.convert_element_type.html
    """
    state.indices[eqn.outvars[0]] = _index_sets(state, eqn.invars[0])

    in_val = _atom_const_val(eqn.invars[0], state)
    if in_val is not None:
        new_dtype = eqn.params.get("new_dtype")
        if new_dtype is not None:
            state.consts[eqn.outvars[0]] = in_val.astype(new_dtype)
        else:
            # stop_gradient, bitcast_convert_type, etc. — pass through as-is.
            state.consts[eqn.outvars[0]] = in_val

    # Propagate value bounds with dtype cast.
    bounds = _atom_value_bounds(eqn.invars[0], state)
    if bounds is not None:
        lo, hi = bounds
        new_dtype = eqn.params.get("new_dtype")
        if new_dtype is not None:
            state.bounds[eqn.outvars[0]] = (
                lo.astype(new_dtype),
                hi.astype(new_dtype),
            )
        else:
            state.bounds[eqn.outvars[0]] = (lo, hi)


def _prop_clamp(eqn: JaxprEqn, state: _PropState) -> None:
    """Clamp(lo, x, hi) returns lo when x < lo, hi when x > hi, else x.

    All three operands can contribute to the output depending on runtime values:
        ∂clamp/∂lo = 1 if x < lo, else 0
        ∂clamp/∂x  = 1 if lo ≤ x ≤ hi, else 0
        ∂clamp/∂hi = 1 if x > hi, else 0

    Since we compute global sparsity patterns (not runtime-dependent),
    we conservatively propagate dependencies from all three operands.

    Example: y = clamp(x[0], x[1], x[2]) where x = [lo, val, hi]
        Input index sets: lo=[{0}], val=[{1}], hi=[{2}]
        Output index sets: [{0, 1, 2}]

    Jaxpr:
        invars[0]: lo (lower bound)
        invars[1]: x (value to clamp)
        invars[2]: hi (upper bound)

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.clamp.html
    """
    lo = _index_sets(state, eqn.invars[0])
    x = _index_sets(state, eqn.invars[1])
    hi = _index_sets(state, eqn.invars[2])

    state.indices[eqn.outvars[0]] = _union_elementwise(
        [lo, x, hi], _atom_numel(eqn.outvars[0])
    )
