"""Propagate index sets through a jaxpr to determine sparsity.

Each JAX primitive has a handler that maps input index sets to output index sets.
For example, element-wise ops preserve per-element dependencies, while reductions
union all input dependencies into a single output.

The main entry point is `_prop_jaxpr`, which walks the computation graph
and applies the appropriate handler for each equation.
"""

import numpy as np
from jax._src.core import Jaxpr, JaxprEqn, Var

from ._argmax import _prop_argmax
from ._broadcast import _prop_broadcast_in_dim
from ._common import (
    IndexSet,
    StateBounds,
    StateConsts,
    StateIndices,
    _atom_const_val,
    _atom_numel,
    _conservative_indices,
    _empty_index_sets,
    _forward_const_vals,
    _forward_value_bounds,
    _index_sets,
    _seed_const_vals,
)
from ._comparison import _prop_ge, _prop_gt, _prop_le, _prop_lt
from ._concatenate import _prop_concatenate
from ._cond import _prop_cond
from ._conv import _prop_conv_general_dilated
from ._cumsum import _prop_cumsum
from ._div import _prop_div
from ._dot_general import _prop_dot_general
from ._dynamic_slice import _prop_dynamic_slice, _prop_dynamic_update_slice
from ._elementwise import (
    _prop_add,
    _prop_binary_const,
    _prop_clamp,
    _prop_convert_element_type,
    _prop_integer_pow,
    _prop_sub,
    _prop_ternary_elementwise,
    _prop_unary_elementwise,
    _prop_zero_derivative,
    _prop_zero_derivative_const,
    _prop_zero_derivative_unary_const,
)
from ._equinox._select_if_vmap import _prop_select_if_vmap
from ._gather import _prop_gather
from ._linalg import _prop_qr
from ._mul import _prop_mul
from ._pad import _prop_pad
from ._platform_index import _prop_platform_index
from ._random import _prop_random
from ._reduce import _prop_reduce
from ._reshape import _prop_reshape
from ._rev import _prop_rev
from ._scan import _prop_scan
from ._scatter import _prop_scatter
from ._select import _prop_select_n
from ._slice import _prop_slice
from ._sort import _prop_sort
from ._split import _prop_split
from ._squeeze import _prop_squeeze
from ._stack import _prop_stack
from ._tile import _prop_tile
from ._top_k import _prop_top_k
from ._transpose import _prop_transpose
from ._unstack import _prop_unstack
from ._while import _prop_while


def _prop_jaxpr(
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
        state_indices[var] = _empty_index_sets(_atom_numel(var))

    # Process each equation
    for eqn in jaxpr.eqns:
        _prop_dispatch(eqn, state_indices, state_consts, state_bounds)

    # Return output dependencies
    return [_index_sets(state_indices, outvar) for outvar in jaxpr.outvars]


def _prop_closed_jaxpr(
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
        _seed_const_vals(state_consts, closed.jaxpr.constvars, closed.consts)
        closed = closed.jaxpr

    _forward_const_vals(state_consts, eqn.invars, closed.invars)
    _forward_value_bounds(state_bounds, eqn.invars, closed.invars)
    input_indices = [_index_sets(state_indices, invar) for invar in eqn.invars]
    output_indices = _prop_jaxpr(closed, input_indices, state_consts, state_bounds)

    for outvar, indices, inner_outvar in zip(
        eqn.outvars,
        output_indices,
        closed.outvars,
        strict=False,
    ):
        state_indices[outvar] = indices
        if isinstance(inner_outvar, Var) and inner_outvar in state_bounds:
            state_bounds[outvar] = state_bounds[inner_outvar]
        # Forward const values symmetrically to bounds,
        # so indices computed inside the nested jaxpr stay resolvable outside.
        val = _atom_const_val(inner_outvar, state_consts)
        if val is not None:
            state_consts[outvar] = val


def _prop_dispatch(
    eqn: JaxprEqn,
    state_indices: StateIndices,
    state_consts: StateConsts,
    state_bounds: StateBounds,
) -> None:
    """Propagate dependencies through a single equation."""
    match eqn.primitive.name:
        case "argmax" | "argmin":
            _prop_argmax(eqn, state_indices, state_bounds)
        # Zero derivative (piecewise constant, ∂f/∂x = 0 a.e.)
        # with const propagation for downstream index resolution
        case (
            "floor"  # ∂⌊x⌋/∂x = 0
            | "ceil"  # ∂⌈x⌉/∂x = 0
            | "sign"  # ∂sign(x)/∂x = 0
            | "not"
        ):
            _prop_zero_derivative_unary_const(eqn, state_indices, state_consts)
        # Zero derivative (piecewise constant, ∂f/∂x = 0 a.e.)
        case (
            "round"  # ∂round(x)/∂x = 0
            | "is_finite"
            | "clz"
            | "population_count"
            | "reduce_and"
            | "reduce_or"
            | "reduce_xor"
            | "shift_left"
            | "shift_right_arithmetic"
            | "shift_right_logical"
        ):
            _prop_zero_derivative(eqn, state_indices)
        case "clamp":
            _prop_clamp(eqn, state_indices)
        case "eq" | "ne" | "lt_to" | "le_to":
            _prop_zero_derivative_const(eqn, state_indices, state_consts)
        case "lt":
            _prop_lt(eqn, state_indices, state_consts, state_bounds)
        case "le":
            _prop_le(eqn, state_indices, state_consts, state_bounds)
        case "gt":
            _prop_gt(eqn, state_indices, state_consts, state_bounds)
        case "ge":
            _prop_ge(eqn, state_indices, state_consts, state_bounds)
        case "and" | "or" | "xor":
            _prop_zero_derivative_const(eqn, state_indices, state_consts)
        case "jit" | "pjit" | "xla_call" | "named_call" | "remat2":
            _prop_closed_jaxpr(eqn, state_indices, state_consts, state_bounds, "jaxpr")
        case "slice":
            _prop_slice(eqn, state_indices, state_consts)
        case "pad":
            _prop_pad(eqn, state_indices)
        case "squeeze":
            _prop_squeeze(eqn, state_indices, state_consts)
        case "broadcast_in_dim":
            _prop_broadcast_in_dim(eqn, state_indices, state_consts, state_bounds)
        case "concatenate":
            _prop_concatenate(eqn, state_indices, state_consts)
        case "reshape":
            _prop_reshape(eqn, state_indices, state_consts)
        case "transpose":
            _prop_transpose(eqn, state_indices, state_consts)
        case "rev":
            _prop_rev(eqn, state_indices)
        case "integer_pow":
            _prop_integer_pow(eqn, state_indices, state_consts, state_bounds)
        case "mul":
            _prop_mul(eqn, state_indices, state_consts, state_bounds)
        case "add" | "add_any":
            _prop_add(eqn, state_indices, state_consts, state_bounds)
        case "sub":
            _prop_sub(eqn, state_indices, state_consts, state_bounds)
        case "div":
            _prop_div(eqn, state_indices, state_consts, state_bounds)
        # Binary elementwise with nonzero partials wrt both operands
        case (
            "pow"  # ∂(x^y)/∂x = y·x^(y-1), ∂(x^y)/∂y = x^y·ln(x)
            | "max"  # ∂max/∂x = 1 if x>y else 0, ∂max/∂y = 1 if y>x else 0
            | "min"  # ∂min/∂x = 1 if x<y else 0, ∂min/∂y = 1 if y<x else 0
            | "atan2"  # ∂atan2(y,x)/∂y = x/(x²+y²), ∂/∂x = -y/(x²+y²)
            | "rem"  # ∂(x mod y)/∂x = 1, ∂(x mod y)/∂y = -⌊x/y⌋
            | "nextafter"
            | "complex"
        ):
            _prop_binary_const(eqn, state_indices, state_consts)
        case "polygamma":
            # ∂ψₙ/∂n = 0 (n is integer order), ∂ψₙ/∂x = ψₙ₊₁(x).
            _prop_binary_const(
                eqn, state_indices, state_consts, is_der1_zero_globally=True
            )
        # Unary elementwise with nonzero derivative (diagonal Jacobian)
        case (
            "neg"  # ∂(-x)/∂x = -1
            | "exp"  # ∂eˣ/∂x = eˣ
            | "log"  # ∂log(x)/∂x = 1/x
            | "sin"  # ∂sin(x)/∂x = cos(x)
            | "cos"  # ∂cos(x)/∂x = -sin(x)
            | "tan"  # ∂tan(x)/∂x = sec²(x)
            | "sqrt"  # ∂√x/∂x = 1/(2√x)
            | "abs"  # ∂|x|/∂x = sign(x)
            | "sinh"  # ∂sinh(x)/∂x = cosh(x)
            | "cosh"  # ∂cosh(x)/∂x = sinh(x)
            | "tanh"  # ∂tanh(x)/∂x = sech²(x)
            | "log1p"  # ∂log(1+x)/∂x = 1/(1+x)
            | "expm1"  # ∂(eˣ-1)/∂x = eˣ
            | "acos"  # ∂arccos(x)/∂x = -1/√(1-x²)
            | "acosh"  # ∂arccosh(x)/∂x = 1/√(x²-1)
            | "asin"  # ∂arcsin(x)/∂x = 1/√(1-x²)
            | "asinh"  # ∂arcsinh(x)/∂x = 1/√(x²+1)
            | "atan"  # ∂arctan(x)/∂x = 1/(1+x²)
            | "atanh"  # ∂arctanh(x)/∂x = 1/(1-x²)
            | "cbrt"  # ∂x^(1/3)/∂x = 1/(3x^(2/3))
            | "conj"  # ∂conj(z)/∂z = 1 (Wirtinger)
            | "copy"  # ∂x/∂x = 1
            | "exp2"  # ∂2ˣ/∂x = 2ˣ·ln(2)
            | "logistic"  # ∂σ(x)/∂x = σ(x)(1-σ(x))
            | "real"  # ∂Re(z)/∂z = 1/2
            | "imag"  # ∂Im(z)/∂z = -i/2
            | "rsqrt"  # ∂(1/√x)/∂x = -1/(2x^(3/2))
            | "erf"  # ∂erf(x)/∂x = 2e^(-x²)/√π
            | "erfc"  # ∂erfc(x)/∂x = -2e^(-x²)/√π
            | "erf_inv"  # ∂erf⁻¹(x)/∂x = (√π/2)·exp(erf⁻¹(x)²)
            | "square"  # ∂x²/∂x = 2x
            | "digamma"  # ∂ψ(x)/∂x = ψ₁(x)
            | "lgamma"  # ∂log(Γ(x))/∂x = ψ(x)
            | "bessel_i0e"  # nonzero derivative
            | "bessel_i1e"  # nonzero derivative
        ):
            _prop_unary_elementwise(eqn, state_indices)
        case "regularized_incomplete_beta":
            _prop_ternary_elementwise(eqn, state_indices)
        case "reduce_sum" | "reduce_max" | "reduce_min" | "reduce_prod":
            _prop_reduce(eqn, state_indices)
        case (
            "convert_element_type"
            | "bitcast_convert_type"
            | "reduce_precision"
            | "stop_gradient"
        ):
            _prop_convert_element_type(eqn, state_indices, state_consts, state_bounds)
        case "conv_general_dilated":
            _prop_conv_general_dilated(eqn, state_indices)
        case "custom_jvp_call" | "custom_vjp_call":
            _prop_closed_jaxpr(
                eqn, state_indices, state_consts, state_bounds, "call_jaxpr"
            )
        case "gather":
            _prop_gather(eqn, state_indices, state_consts, state_bounds)
        case "scatter" | "scatter-add" | "scatter-mul" | "scatter-min" | "scatter-max":
            _prop_scatter(eqn, state_indices, state_consts, state_bounds)
        case "select_n":
            _prop_select_n(eqn, state_indices, state_consts, state_bounds)
        case "select_if_vmap":
            _prop_select_if_vmap(eqn, state_indices, state_consts)
        case "iota":
            _prop_iota(eqn, state_indices, state_consts)
        case (
            "random_seed"
            | "random_unwrap"
            | "random_wrap"
            | "random_split"
            | "random_fold_in"
            | "random_bits"
        ):
            _prop_random(eqn, state_indices)
        case "while":
            _prop_while(eqn, state_indices, state_consts, state_bounds, _prop_jaxpr)
        case "cond":
            _prop_cond(eqn, state_indices, state_consts, state_bounds, _prop_jaxpr)
        case "platform_index":
            _prop_platform_index(eqn, state_indices)
        case "dynamic_slice":
            _prop_dynamic_slice(eqn, state_indices, state_consts, state_bounds)
        case "dynamic_update_slice":
            _prop_dynamic_update_slice(eqn, state_indices, state_consts, state_bounds)
        case "top_k":
            _prop_top_k(eqn, state_indices)
        # TODO: add precise handlers for remaining control flow operators.
        # https://docs.jax.dev/en/latest/jax.lax.html#control-flow-operators
        case "scan":
            _prop_scan(eqn, state_indices, state_consts, state_bounds, _prop_jaxpr)
        case "dot_general":
            _prop_dot_general(eqn, state_indices, state_consts)
        case "split":
            _prop_split(eqn, state_indices)
        case "stack":
            _prop_stack(eqn, state_indices, state_consts)
        case "unstack":
            _prop_unstack(eqn, state_indices, state_consts)
        case "tile":
            _prop_tile(eqn, state_indices, state_consts)
        case "sort":
            _prop_sort(eqn, state_indices)
        case "cumsum" | "cumprod" | "cummax" | "cummin":
            _prop_cumsum(eqn, state_indices)
        case "qr":
            _prop_qr(eqn, state_indices)
        # Conservative fallback: all outputs depend on all inputs.
        case (
            "nonbatchable"
            | "unvmap_any"  # from Equinox
            | "unvmap_max"  # from Equinox
            | "pure_callback"
            | "lu"
            | "cholesky"
            | "svd"
            | "eigh"
        ):
            _prop_conservative_fallback(eqn, state_indices)
        case _:
            _prop_throw_error(eqn, state_indices)


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
    state_indices[eqn.outvars[0]] = _empty_index_sets(numel)

    dtype = eqn.params["dtype"]
    dim = eqn.params["dimension"]
    state_consts[eqn.outvars[0]] = np.broadcast_to(
        np.arange(shape[dim], dtype=dtype).reshape(
            [shape[dim] if i == dim else 1 for i in range(len(shape))]
        ),
        shape,
    )


def _prop_conservative_fallback(eqn: JaxprEqn, state_indices: StateIndices) -> None:
    """Conservative fallback for primitives without precise handlers.

    Assumes worst-case: every output element may depend on every input element.
    This is correct but may overestimate sparsity (more nonzeros than necessary).

    Used for primitives without precise handlers.
    """
    all_inputs: list[IndexSet] = []
    for invar in eqn.invars:
        all_inputs.extend(_index_sets(state_indices, invar))
    for outvar in eqn.outvars:
        state_indices[outvar] = _conservative_indices(all_inputs, _atom_numel(outvar))


def _prop_throw_error(eqn: JaxprEqn, state_indices: StateIndices) -> None:
    """Raise an error for unknown primitives.

    This ensures we don't silently produce incorrect sparsity patterns.
    """
    msg = (
        f"No handler for primitive '{eqn.primitive.name}'. "
        "Please help out asdex's development by reporting this at "
        "https://github.com/adrhill/asdex/issues"
    )
    raise NotImplementedError(msg)
