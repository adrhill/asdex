"""Propagate index sets through a jaxpr to determine sparsity.

Each JAX primitive has a handler that maps input index sets to output index sets.
For example, element-wise ops preserve per-element dependencies, while reductions
union all input dependencies into a single output.

The main entry point is `prop_jaxpr`, which walks the computation graph
and applies the appropriate handler for each equation.
"""

import numpy as np
from jax._src.core import Jaxpr, JaxprEqn, Var

from ._argmax import prop_argmax
from ._broadcast import prop_broadcast_in_dim
from ._commons import (
    IndexSet,
    StateBounds,
    StateConsts,
    StateIndices,
    atom_numel,
    conservative_indices,
    empty_index_sets,
    forward_const_vals,
    forward_value_bounds,
    index_sets,
    seed_const_vals,
)
from ._comparison import prop_ge, prop_gt, prop_le, prop_lt
from ._concatenate import prop_concatenate
from ._cond import prop_cond
from ._conv import prop_conv_general_dilated
from ._cumsum import prop_cumsum
from ._div import prop_div
from ._dot_general import prop_dot_general
from ._dynamic_slice import prop_dynamic_slice, prop_dynamic_update_slice
from ._elementwise import (
    prop_add,
    prop_binary_const,
    prop_convert_element_type,
    prop_integer_pow,
    prop_sub,
    prop_unary_elementwise,
    prop_zero_derivative,
    prop_zero_derivative_const,
)
from ._equinox._select_if_vmap import prop_select_if_vmap
from ._gather import prop_gather
from ._mul import prop_mul
from ._pad import prop_pad
from ._platform_index import prop_platform_index
from ._reduce import prop_reduce
from ._reshape import prop_reshape
from ._rev import prop_rev
from ._scan import prop_scan
from ._scatter import prop_scatter
from ._select import prop_select_n
from ._slice import prop_slice
from ._sort import prop_sort
from ._split import prop_split
from ._squeeze import prop_squeeze
from ._tile import prop_tile
from ._top_k import prop_top_k
from ._transpose import prop_transpose
from ._while import prop_while


def prop_jaxpr(
    jaxpr: Jaxpr,
    input_indices: list[list[IndexSet]],
    state_consts: StateConsts | None = None,
    state_bounds: StateBounds | None = None,
) -> list[list[IndexSet]]:
    """Propagate index sets through a jaxpr.

    Args:
        jaxpr: The jaxpr to analyze
        input_indices: List of per-element index set lists, one per input variable
        state_consts: Optional mapping of constant variables to their values.
            Used for precise tracking of static indices in gather/scatter.
        state_bounds: Optional pre-seeded value bounds from an outer scope.
            Used to forward bounded-but-not-constant values into nested jaxprs.

    Returns:
        List of per-element index set lists, one per output variable
    """
    state_indices: StateIndices = {}
    if state_consts is None:
        state_consts = {}
    if state_bounds is None:
        state_bounds = {}

    # Initialize input variables
    for var, indices in zip(jaxpr.invars, input_indices, strict=False):
        state_indices[var] = indices

    # Initialize constant variables (no input dependencies)
    for var in jaxpr.constvars:
        state_indices[var] = empty_index_sets(atom_numel(var))

    # Process each equation
    for eqn in jaxpr.eqns:
        prop_dispatch(eqn, state_indices, state_consts, state_bounds)

    # Return output dependencies
    return [index_sets(state_indices, outvar) for outvar in jaxpr.outvars]


def prop_closed_jaxpr(
    eqn: JaxprEqn,
    state_indices: StateIndices,
    state_consts: StateConsts,
    state_bounds: StateBounds,
    param_key: str,
) -> None:
    """Recursively trace a closed jaxpr stored in ``eqn.params[param_key]``.

    Shared implementation for ``prop_nested_jaxpr`` (param ``"jaxpr"``)
    and ``prop_custom_call`` (param ``"call_jaxpr"``).
    """
    closed = eqn.params.get(param_key)
    if closed is None:
        msg = (
            f"Primitive '{eqn.primitive.name}' has no '{param_key}' parameter. "
            "Please help out asdex's development by reporting this at "
            "https://github.com/adrhill/asdex/issues"
        )
        raise ValueError(msg)

    # Unwrap ClosedJaxpr, seeding state_consts for captured constants
    if hasattr(closed, "jaxpr"):
        seed_const_vals(state_consts, closed.jaxpr.constvars, closed.consts)
        closed = closed.jaxpr

    forward_const_vals(state_consts, eqn.invars, closed.invars)
    forward_value_bounds(state_bounds, eqn.invars, closed.invars)
    input_indices = [index_sets(state_indices, invar) for invar in eqn.invars]
    output_indices = prop_jaxpr(closed, input_indices, state_consts, state_bounds)

    for outvar, indices, inner_outvar in zip(
        eqn.outvars,
        output_indices,
        closed.outvars,
        strict=False,
    ):
        state_indices[outvar] = indices
        if isinstance(inner_outvar, Var) and inner_outvar in state_bounds:
            state_bounds[outvar] = state_bounds[inner_outvar]


def prop_dispatch(
    eqn: JaxprEqn,
    state_indices: StateIndices,
    state_consts: StateConsts,
    state_bounds: StateBounds,
) -> None:
    """Propagate dependencies through a single equation."""
    match eqn.primitive.name:
        case "argmax" | "argmin":
            prop_argmax(eqn, state_indices, state_bounds)
        case (
            "floor"
            | "ceil"
            | "round"
            | "sign"
            | "is_finite"
            | "clz"
            | "clamp"
            | "population_count"
            | "reduce_and"
            | "reduce_or"
            | "reduce_xor"
            | "not"
        ):
            prop_zero_derivative(eqn, state_indices)
        case "eq" | "ne" | "lt_to" | "le_to":
            prop_zero_derivative_const(eqn, state_indices, state_consts)
        case "lt":
            prop_lt(eqn, state_indices, state_consts, state_bounds)
        case "le":
            prop_le(eqn, state_indices, state_consts, state_bounds)
        case "gt":
            prop_gt(eqn, state_indices, state_consts, state_bounds)
        case "ge":
            prop_ge(eqn, state_indices, state_consts, state_bounds)
        case "and" | "or" | "xor":
            prop_zero_derivative_const(eqn, state_indices, state_consts)
        case "jit" | "pjit" | "xla_call" | "named_call":
            prop_closed_jaxpr(eqn, state_indices, state_consts, state_bounds, "jaxpr")
        case "slice":
            prop_slice(eqn, state_indices, state_consts)
        case "pad":
            prop_pad(eqn, state_indices)
        case "squeeze":
            prop_squeeze(eqn, state_indices, state_consts)
        case "broadcast_in_dim":
            prop_broadcast_in_dim(eqn, state_indices, state_consts, state_bounds)
        case "concatenate":
            prop_concatenate(eqn, state_indices, state_consts)
        case "reshape":
            prop_reshape(eqn, state_indices, state_consts)
        case "transpose":
            prop_transpose(eqn, state_indices, state_consts)
        case "rev":
            prop_rev(eqn, state_indices)
        case "integer_pow":
            prop_integer_pow(eqn, state_indices, state_consts, state_bounds)
        case "mul":
            prop_mul(eqn, state_indices, state_consts, state_bounds)
        case "add" | "add_any":
            prop_add(eqn, state_indices, state_consts, state_bounds)
        case "sub":
            prop_sub(eqn, state_indices, state_consts, state_bounds)
        case "div":
            prop_div(eqn, state_indices, state_consts, state_bounds)
        case "pow" | "max" | "min" | "atan2" | "rem" | "nextafter" | "complex":
            prop_binary_const(eqn, state_indices, state_consts)
        case (
            "neg"
            | "exp"
            | "log"
            | "sin"
            | "cos"
            | "tan"
            | "sqrt"
            | "abs"
            | "sinh"
            | "cosh"
            | "tanh"
            | "log1p"
            | "expm1"
            | "acos"
            | "acosh"
            | "asin"
            | "asinh"
            | "atan"
            | "atanh"
            | "cbrt"
            | "conj"
            | "copy"
            | "exp2"
            | "logistic"
            | "real"
            | "imag"
            | "rsqrt"
            | "erf"
            | "square"
        ):
            prop_unary_elementwise(eqn, state_indices)
        case "reduce_sum" | "reduce_max" | "reduce_min" | "reduce_prod":
            prop_reduce(eqn, state_indices)
        case (
            "convert_element_type"
            | "bitcast_convert_type"
            | "reduce_precision"
            | "stop_gradient"
        ):
            prop_convert_element_type(eqn, state_indices, state_consts, state_bounds)
        case "conv_general_dilated":
            prop_conv_general_dilated(eqn, state_indices)
        case "custom_jvp_call" | "custom_vjp_call":
            prop_closed_jaxpr(
                eqn, state_indices, state_consts, state_bounds, "call_jaxpr"
            )
        case "gather":
            prop_gather(eqn, state_indices, state_consts, state_bounds)
        case "scatter" | "scatter-add" | "scatter-mul" | "scatter-min" | "scatter-max":
            prop_scatter(eqn, state_indices, state_consts, state_bounds)
        case "select_n":
            prop_select_n(eqn, state_indices, state_consts, state_bounds)
        case "select_if_vmap":
            prop_select_if_vmap(eqn, state_indices, state_consts)
        case "iota":
            _prop_iota(eqn, state_indices, state_consts)
        case "while":
            prop_while(eqn, state_indices, state_consts, prop_jaxpr)
        case "cond":
            prop_cond(eqn, state_indices, state_consts, prop_jaxpr)
        case "platform_index":
            prop_platform_index(eqn, state_indices)
        case "dynamic_slice":
            prop_dynamic_slice(eqn, state_indices, state_consts, state_bounds)
        case "dynamic_update_slice":
            prop_dynamic_update_slice(eqn, state_indices, state_consts, state_bounds)
        case "top_k":
            prop_top_k(eqn, state_indices)
        # TODO: add precise handlers for remaining control flow operators.
        # https://docs.jax.dev/en/latest/jax.lax.html#control-flow-operators
        case "scan":
            prop_scan(eqn, state_indices, state_consts, prop_jaxpr)
        case "dot_general":
            prop_dot_general(eqn, state_indices, state_consts)
        case "split":
            prop_split(eqn, state_indices)
        case "tile":
            prop_tile(eqn, state_indices, state_consts)
        case "sort":
            prop_sort(eqn, state_indices)
        case "cumsum":
            prop_cumsum(eqn, state_indices)
        # Conservative fallback: all outputs depend on all inputs.
        case (
            "nonbatchable"
            | "unvmap_any"  # from Equinox
            | "unvmap_max"  # from Equinox
            | "pure_callback"
        ):
            prop_conservative_fallback(eqn, state_indices)
        case _:
            prop_throw_error(eqn, state_indices)


def _prop_iota(
    eqn: JaxprEqn, state_indices: StateIndices, state_consts: StateConsts
) -> None:
    """Iota generates a constant index array with no input dependencies.

    The output is fully determined by the parameters (shape, dtype, dimension),
    so all dependency sets are empty.
    We also track the concrete values for downstream gather/scatter precision.

    Jaxpr:
        invars: [] (no inputs)
        shape: output shape
        dtype: output dtype
        dimension: axis along which indices increase
    """
    shape = eqn.params["shape"]
    numel = int(np.prod(shape))
    state_indices[eqn.outvars[0]] = empty_index_sets(numel)

    dtype = eqn.params["dtype"]
    dim = eqn.params["dimension"]
    state_consts[eqn.outvars[0]] = np.broadcast_to(
        np.arange(shape[dim], dtype=dtype).reshape(
            [shape[dim] if i == dim else 1 for i in range(len(shape))]
        ),
        shape,
    )


def prop_conservative_fallback(eqn: JaxprEqn, state_indices: StateIndices) -> None:
    """Conservative fallback for primitives without precise handlers.

    Assumes worst-case: every output element may depend on every input element.
    This is correct but may overestimate sparsity (more nonzeros than necessary).

    Used for primitives without precise handlers.
    """
    all_inputs: list[IndexSet] = []
    for invar in eqn.invars:
        all_inputs.extend(index_sets(state_indices, invar))
    for outvar in eqn.outvars:
        state_indices[outvar] = conservative_indices(all_inputs, atom_numel(outvar))


def prop_throw_error(eqn: JaxprEqn, state_indices: StateIndices) -> None:
    """Raise an error for unknown primitives.

    This ensures we don't silently produce incorrect sparsity patterns.
    """
    msg = (
        f"No handler for primitive '{eqn.primitive.name}'. "
        "Please help out asdex's development by reporting this at "
        "https://github.com/adrhill/asdex/issues"
    )
    raise NotImplementedError(msg)
