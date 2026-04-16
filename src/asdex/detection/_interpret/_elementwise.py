"""Propagation rules for element-wise operations."""

import numpy as np
from jax._src.core import JaxprEqn

from ._commons import (
    StateBounds,
    StateConsts,
    StateIndices,
    atom_const_val,
    atom_numel,
    atom_shape,
    atom_value_bounds,
    clear_where_zero,
    copy_index_sets,
    empty_index_sets,
    index_sets,
    numel,
    propagate_const_binary,
)

# Ufuncs for evaluating constant values during tracing.
# Used to propagate static index values through arithmetic to gather/scatter.
_BINARY_CONST_UFUNCS: dict[str, np.ufunc] = {
    # arithmetic
    "add": np.add,
    "add_any": np.add,
    "sub": np.subtract,
    "mul": np.multiply,
    "div": np.divide,
    "pow": np.power,
    "max": np.maximum,
    "min": np.minimum,
    "atan2": np.arctan2,
    "rem": np.remainder,
    "nextafter": np.nextafter,
    # comparison
    "eq": np.equal,
    "ne": np.not_equal,
    # bitwise
    "and": np.bitwise_and,
    "or": np.bitwise_or,
    "xor": np.bitwise_xor,
}


def _propagate_const(eqn: JaxprEqn, state_consts: StateConsts) -> None:
    """Propagate a const value through a binary elementwise op.

    If both inputs are statically known,
    apply the matching numpy ufunc and store the result.
    Without this, downstream handlers (e.g. ``gather``, ``scatter``) cannot resolve
    static index arrays and fall back to conservative.
    """
    ufunc = _BINARY_CONST_UFUNCS.get(eqn.primitive.name)
    if ufunc is not None:
        propagate_const_binary(eqn, state_consts, ufunc)


# Building blocks (private)


def _zero_derivative(eqn: JaxprEqn, state_indices: StateIndices) -> None:
    """Set empty index sets for zero-derivative outputs."""
    for outvar in eqn.outvars:
        state_indices[outvar] = empty_index_sets(atom_numel(outvar))


def _binary_elementwise(eqn: JaxprEqn, state_indices: StateIndices) -> None:
    """Union per-element index sets from two inputs."""
    in1 = index_sets(state_indices, eqn.invars[0])
    in2 = index_sets(state_indices, eqn.invars[1])
    out_size = 0 if len(in1) == 0 or len(in2) == 0 else max(len(in1), len(in2))

    in1_shape = atom_shape(eqn.invars[0])
    in2_shape = atom_shape(eqn.invars[1])

    # Fast path: same shape or scalar.
    # Modular indexing handles both correctly:
    # i % len == i for same size, i % 1 == 0 for scalar.
    if in1_shape == in2_shape or len(in1) <= 1 or len(in2) <= 1:
        state_indices[eqn.outvars[0]] = [
            in1[i % len(in1)] | in2[i % len(in2)] for i in range(out_size)
        ]
        return

    # General broadcast: map output coordinates to input coordinates
    # respecting numpy-style broadcasting (size-1 dims read index 0).
    # Example: mul of (16,16) * (16,1) → (16,16).
    # out[p,d] depends on in1[p,d] and in2[p,0], not in2[d].
    out_shape = atom_shape(eqn.outvars[0])
    out_size = numel(out_shape)
    out_coords = np.indices(out_shape)
    ndim = len(out_shape)

    def _broadcast_flat(in_shape: tuple[int, ...]) -> np.ndarray:
        # Left-pad with 1s to match output ndim (numpy broadcasting rule).
        pad = ndim - len(in_shape)
        padded = (1,) * pad + in_shape
        coords = tuple(out_coords[d] if padded[d] > 1 else 0 for d in range(ndim))
        return np.ravel_multi_index(coords, padded).ravel()

    in1_flat = _broadcast_flat(in1_shape)
    in2_flat = _broadcast_flat(in2_shape)

    state_indices[eqn.outvars[0]] = [
        in1[in1_flat[i]] | in2[in2_flat[i]] for i in range(out_size)
    ]


def _propagate_bounds_add(
    eqn: JaxprEqn,
    state_consts: StateConsts,
    state_bounds: StateBounds,
) -> None:
    """Propagate value bounds through ``add`` or ``add_any`` via interval arithmetic."""
    b1 = atom_value_bounds(eqn.invars[0], state_consts, state_bounds)
    b2 = atom_value_bounds(eqn.invars[1], state_consts, state_bounds)
    if b1 is None or b2 is None:
        return
    lo1, hi1 = b1
    lo2, hi2 = b2
    state_bounds[eqn.outvars[0]] = (lo1 + lo2, hi1 + hi2)


def _propagate_bounds_sub(
    eqn: JaxprEqn,
    state_consts: StateConsts,
    state_bounds: StateBounds,
) -> None:
    """Propagate value bounds through ``sub`` via interval arithmetic."""
    b1 = atom_value_bounds(eqn.invars[0], state_consts, state_bounds)
    b2 = atom_value_bounds(eqn.invars[1], state_consts, state_bounds)
    if b1 is None or b2 is None:
        return
    lo1, hi1 = b1
    lo2, hi2 = b2
    state_bounds[eqn.outvars[0]] = (lo1 - hi2, hi1 - lo2)


# Composite handlers (public)
# Each corresponds to exactly one dispatch case in prop_dispatch.


def prop_zero_derivative(eqn: JaxprEqn, state_indices: StateIndices) -> None:
    """Zero-derivative primitives (floor, ceil, sign, ...).

    Operations with zero derivative almost everywhere.
    Their outputs are piecewise constant,
    so infinitesimal input changes don't affect outputs.

    Mathematically, for f in {floor, ceil, sign, ...}:
        ∂f/∂x = 0  (almost everywhere)
    Therefore, output elements have no dependencies on input elements.

    Example: y = floor(x) where x = [1.7, 2.3, 3.9]
        Input state_indices:  [{0}, {1}, {2}]
        Output state_indices: [{}, {}, {}]  (empty sets, no dependence)
    """
    _zero_derivative(eqn, state_indices)


def prop_zero_derivative_const(
    eqn: JaxprEqn, state_indices: StateIndices, state_consts: StateConsts
) -> None:
    """Zero-derivative primitives that also propagate const values.

    Used for comparisons (eq, ne) and bitwise ops (and, or, xor)
    where the output is zero-derivative
    but the concrete result may be needed by downstream handlers.
    """
    _zero_derivative(eqn, state_indices)
    _propagate_const(eqn, state_consts)


def prop_binary_const(
    eqn: JaxprEqn, state_indices: StateIndices, state_consts: StateConsts
) -> None:
    """Binary elementwise primitives (div, pow, max, min, ...) with const propagation.

    Each output element depends on the corresponding elements from both inputs.
    Also propagates const values for downstream index resolution.

    For f(x, y) element-wise:
        ∂f/∂x[i] and ∂f/∂y[i] are generally nonzero.
    So out[i] depends on {x[i], y[i]} (union of dependencies).

    Example: z = x + y where x = [a, b], y = [c, d]
        Input state_indices:  [{0}, {1}], [{2}, {3}]
        Output state_indices: [{0, 2}, {1, 3}]

    Jaxpr:
        invars[0]: first input array
        invars[1]: second input array
    """
    _binary_elementwise(eqn, state_indices)
    _propagate_const(eqn, state_consts)


def prop_add(
    eqn: JaxprEqn,
    state_indices: StateIndices,
    state_consts: StateConsts,
    state_bounds: StateBounds,
) -> None:
    """Add / add_any: binary elementwise with interval arithmetic bounds.

    ``[a,b] + [c,d] = [a+c, b+d]``.
    """
    _binary_elementwise(eqn, state_indices)
    _propagate_const(eqn, state_consts)
    _propagate_bounds_add(eqn, state_consts, state_bounds)


def prop_sub(
    eqn: JaxprEqn,
    state_indices: StateIndices,
    state_consts: StateConsts,
    state_bounds: StateBounds,
) -> None:
    """Sub: binary elementwise with interval arithmetic bounds.

    ``[a,b] - [c,d] = [a-d, b-c]``.
    """
    _binary_elementwise(eqn, state_indices)
    _propagate_const(eqn, state_consts)
    _propagate_bounds_sub(eqn, state_consts, state_bounds)


def prop_integer_pow(
    eqn: JaxprEqn,
    state_indices: StateIndices,
    state_consts: StateConsts | None = None,
    state_bounds: StateBounds | None = None,
) -> None:
    """Integer power x^n is element-wise.

    Each output depends only on the corresponding input element.
    Special cases:
    - x^0 = 1 has zero derivative, so no dependencies.
    - 0^n = 0 for n > 0, so d(0^n)/dx = 0 and no dependencies.

    ∂(x^n)/∂x = n·x^(n-1), which is zero iff n = 0 or (x = 0 and n > 1).

    Example: y = x^2 where x = [a, b, c]
        Input state_indices:  [{0}, {1}, {2}]
        Output state_indices: [{0}, {1}, {2}]  (or [{}, {}, {}] if n=0)

    Jaxpr:
        invars[0]: input array
        y: the integer exponent

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.integer_pow.html
    """
    y = eqn.params.get("y", 1)
    in_indices = index_sets(state_indices, eqn.invars[0])

    if y == 0:
        state_indices[eqn.outvars[0]] = empty_index_sets(len(in_indices))
    else:
        state_indices[eqn.outvars[0]] = copy_index_sets(in_indices)

    # Const propagation.
    if state_consts is not None:
        in_val = atom_const_val(eqn.invars[0], state_consts)
        if in_val is not None:
            state_consts[eqn.outvars[0]] = np.power(in_val, y)

    # Zero-skipping: d(0^n)/dx = n * 0^(n-1) = 0 for n > 1.
    # For n = 1, d(x)/dx = 1 even at x = 0, so no skipping.
    if state_consts is not None and y > 1:
        clear_where_zero(eqn, state_indices, state_consts, 0)

    # Bounds propagation for [a,b]^n.
    if state_bounds is not None:
        _propagate_bounds_integer_pow(eqn, y, state_consts or {}, state_bounds)


def _propagate_bounds_integer_pow(
    eqn: JaxprEqn,
    y: int,
    state_consts: StateConsts,
    state_bounds: StateBounds,
) -> None:
    """Propagate value bounds through ``integer_pow``.

    - n == 0: bounds are (1, 1).
    - n even: [0, max(|a|,|b|)^n] if interval spans zero,
      else [min(|a|,|b|)^n, max(|a|,|b|)^n].
    - n odd (monotone): [a^n, b^n].
    """
    in_bounds = atom_value_bounds(eqn.invars[0], state_consts, state_bounds)
    if in_bounds is None:
        return

    lo, hi = in_bounds

    if y == 0:
        ones = np.ones_like(lo)
        state_bounds[eqn.outvars[0]] = (ones, ones)
    elif y % 2 == 1:
        # Odd power is monotone.
        state_bounds[eqn.outvars[0]] = (np.power(lo, y), np.power(hi, y))
    else:
        # Even power: x^n is not monotone over intervals spanning zero.
        abs_lo = np.abs(lo)
        abs_hi = np.abs(hi)
        max_abs = np.maximum(abs_lo, abs_hi)
        min_abs = np.minimum(abs_lo, abs_hi)

        spans_zero = (lo <= 0) & (hi >= 0)
        out_lo = np.where(spans_zero, np.zeros_like(lo), np.power(min_abs, y))
        out_hi = np.power(max_abs, y)
        state_bounds[eqn.outvars[0]] = (out_lo, out_hi)


def prop_unary_elementwise(eqn: JaxprEqn, state_indices: StateIndices) -> None:
    """Unary element-wise ops (exp, sin, etc.) apply a function to each element.

    Each output depends only on the corresponding input element.
    The Jacobian is diagonal.

    For f(x) element-wise:
        ∂f[i]/∂x[j] = f'(x[i]) if i = j, else 0

    Example: y = exp(x) where x = [a, b, c]
        Input state_indices:  [{0}, {1}, {2}]
        Output state_indices: [{0}, {1}, {2}]

    Jaxpr:
        invars[0]: input array
    """
    state_indices[eqn.outvars[0]] = copy_index_sets(
        index_sets(state_indices, eqn.invars[0])
    )


def prop_convert_element_type(
    eqn: JaxprEqn,
    state_indices: StateIndices,
    state_consts: StateConsts,
    state_bounds: StateBounds | None = None,
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
        Input state_indices:  [{0}, {1}, {2}]
        Output state_indices: [{0}, {1}, {2}]

    Jaxpr:
        invars[0]: input array
        new_dtype: target dtype

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.convert_element_type.html
    """
    state_indices[eqn.outvars[0]] = copy_index_sets(
        index_sets(state_indices, eqn.invars[0])
    )

    in_val = atom_const_val(eqn.invars[0], state_consts)
    if in_val is not None:
        new_dtype = eqn.params.get("new_dtype")
        if new_dtype is not None:
            state_consts[eqn.outvars[0]] = in_val.astype(new_dtype)
        else:
            # stop_gradient, bitcast_convert_type, etc. — pass through as-is.
            state_consts[eqn.outvars[0]] = in_val

    # Propagate value bounds with dtype cast.
    if state_bounds is not None:
        bounds = atom_value_bounds(eqn.invars[0], state_consts, state_bounds)
        if bounds is not None:
            lo, hi = bounds
            new_dtype = eqn.params.get("new_dtype")
            if new_dtype is not None:
                state_bounds[eqn.outvars[0]] = (
                    lo.astype(new_dtype),
                    hi.astype(new_dtype),
                )
            else:
                state_bounds[eqn.outvars[0]] = (lo, hi)
