"""Propagate index sets through a jaxpr to determine sparsity.

Each JAX primitive has a handler that maps input index sets to output index sets.
For example, element-wise ops preserve per-element dependencies, while reductions
union all input dependencies into a single output.

The main entry point is `_prop_jaxpr`, which walks the computation graph
and applies the appropriate handler for each equation.
"""

from jax._src.core import Jaxpr, JaxprEqn, Var

from ._argmax import _prop_argmax
from ._broadcast import _prop_broadcast_in_dim
from ._common import (
    IndexSet,
    _atom_const_val,
    _atom_numel,
    _conservative_indices,
    _empty_index_sets,
    _forward_into_jaxpr,
    _index_sets,
    _PropState,
    _report_issue,
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
from ._iota import _prop_iota
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
    parent: _PropState | None = None,
) -> list[list[IndexSet]]:
    """Propagate index sets through a jaxpr.

    Runs in a fresh scope:
    index sets are tracked in a new dict so they can be freed when the scope ends,
    while const values and value bounds are shared with the parent scope by aliasing.

    Args:
        jaxpr: The jaxpr to analyze
        input_indices: List of per-element index set lists, one per input variable
        parent: Optional propagation state of the enclosing scope.
            Its const values and value bounds carry over into this jaxpr,
            so handlers can resolve indices seeded outside it.

    Returns:
        List of per-element index set lists, one per output variable
    """
    if parent is None:
        state = _PropState()
    else:
        state = _PropState(consts=parent.consts, bounds=parent.bounds)

    # Initialize input variables
    for var, indices in zip(jaxpr.invars, input_indices, strict=False):
        state.indices[var] = indices

    # Initialize constant variables (no input dependencies)
    for var in jaxpr.constvars:
        state.indices[var] = _empty_index_sets(_atom_numel(var))

    # Process each equation
    for eqn in jaxpr.eqns:
        _prop_dispatch(eqn, state)

    # Return output dependencies
    return [_index_sets(state, outvar) for outvar in jaxpr.outvars]


def _prop_closed_jaxpr(
    eqn: JaxprEqn,
    state: _PropState,
    param_key: str,
) -> None:
    """Recursively trace a closed jaxpr stored in ``eqn.params[param_key]``.

    Shared implementation for ``prop_nested_jaxpr`` (param ``"jaxpr"``)
    and ``prop_custom_call`` (param ``"call_jaxpr"``).
    """
    closed = eqn.params.get(param_key)
    if closed is None:
        msg = _report_issue(
            f"Primitive '{eqn.primitive.name}' has no '{param_key}' parameter."
        )
        raise ValueError(msg)

    # Unwrap ClosedJaxpr, seeding state.consts for captured constants
    if hasattr(closed, "jaxpr"):
        _seed_const_vals(state, closed.jaxpr.constvars, closed.consts)
        closed = closed.jaxpr

    _forward_into_jaxpr(state, eqn.invars, closed.invars)
    input_indices = [_index_sets(state, invar) for invar in eqn.invars]
    output_indices = _prop_jaxpr(closed, input_indices, state)

    for outvar, indices, inner_outvar in zip(
        eqn.outvars,
        output_indices,
        closed.outvars,
        strict=False,
    ):
        state.indices[outvar] = indices
        if isinstance(inner_outvar, Var) and inner_outvar in state.bounds:
            state.bounds[outvar] = state.bounds[inner_outvar]
        # Forward const values symmetrically to bounds,
        # so indices computed inside the nested jaxpr stay resolvable outside.
        val = _atom_const_val(inner_outvar, state)
        if val is not None:
            state.consts[outvar] = val


def _prop_dispatch(eqn: JaxprEqn, state: _PropState) -> None:
    """Propagate dependencies through a single equation."""
    match eqn.primitive.name:
        case "argmax" | "argmin":
            _prop_argmax(eqn, state)
        # Zero derivative (piecewise constant, ∂f/∂x = 0 a.e.)
        # with const propagation for downstream index resolution
        case (
            "floor"  # ∂⌊x⌋/∂x = 0
            | "ceil"  # ∂⌈x⌉/∂x = 0
            | "sign"  # ∂sign(x)/∂x = 0
            | "not"
        ):
            _prop_zero_derivative_unary_const(eqn, state)
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
            _prop_zero_derivative(eqn, state)
        case "clamp":
            _prop_clamp(eqn, state)
        case "eq" | "ne" | "lt_to" | "le_to":
            _prop_zero_derivative_const(eqn, state)
        case "lt":
            _prop_lt(eqn, state)
        case "le":
            _prop_le(eqn, state)
        case "gt":
            _prop_gt(eqn, state)
        case "ge":
            _prop_ge(eqn, state)
        case "and" | "or" | "xor":
            _prop_zero_derivative_const(eqn, state)
        case "jit" | "pjit" | "xla_call" | "named_call" | "remat2":
            _prop_closed_jaxpr(eqn, state, "jaxpr")
        case "slice":
            _prop_slice(eqn, state)
        case "pad":
            _prop_pad(eqn, state)
        case "squeeze":
            _prop_squeeze(eqn, state)
        case "broadcast_in_dim":
            _prop_broadcast_in_dim(eqn, state)
        case "concatenate":
            _prop_concatenate(eqn, state)
        case "reshape":
            _prop_reshape(eqn, state)
        case "transpose":
            _prop_transpose(eqn, state)
        case "rev":
            _prop_rev(eqn, state)
        case "integer_pow":
            _prop_integer_pow(eqn, state)
        case "mul":
            _prop_mul(eqn, state)
        case "add" | "add_any":
            _prop_add(eqn, state)
        case "sub":
            _prop_sub(eqn, state)
        case "div":
            _prop_div(eqn, state)
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
            _prop_binary_const(eqn, state)
        case "polygamma":
            # ∂ψₙ/∂n = 0 (n is integer order), ∂ψₙ/∂x = ψₙ₊₁(x).
            _prop_binary_const(eqn, state, is_der1_zero_globally=True)
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
            _prop_unary_elementwise(eqn, state)
        case "regularized_incomplete_beta":
            _prop_ternary_elementwise(eqn, state)
        case "reduce_sum" | "reduce_max" | "reduce_min" | "reduce_prod":
            _prop_reduce(eqn, state)
        case (
            "convert_element_type"
            | "bitcast_convert_type"
            | "reduce_precision"
            | "stop_gradient"
        ):
            _prop_convert_element_type(eqn, state)
        case "conv_general_dilated":
            _prop_conv_general_dilated(eqn, state)
        case "custom_jvp_call" | "custom_vjp_call":
            _prop_closed_jaxpr(eqn, state, "call_jaxpr")
        case "gather":
            _prop_gather(eqn, state)
        case "scatter" | "scatter-add" | "scatter-mul" | "scatter-min" | "scatter-max":
            _prop_scatter(eqn, state)
        case "select_n":
            _prop_select_n(eqn, state)
        case "select_if_vmap":
            _prop_select_if_vmap(eqn, state)
        case "iota":
            _prop_iota(eqn, state)
        case (
            "random_seed"
            | "random_unwrap"
            | "random_wrap"
            | "random_split"
            | "random_fold_in"
            | "random_bits"
        ):
            _prop_random(eqn, state)
        case "while":
            _prop_while(eqn, state, _prop_jaxpr)
        case "cond":
            _prop_cond(eqn, state, _prop_jaxpr)
        case "platform_index":
            _prop_platform_index(eqn, state)
        case "dynamic_slice":
            _prop_dynamic_slice(eqn, state)
        case "dynamic_update_slice":
            _prop_dynamic_update_slice(eqn, state)
        case "top_k":
            _prop_top_k(eqn, state)
        # TODO: add precise handlers for remaining control flow operators.
        # https://docs.jax.dev/en/latest/jax.lax.html#control-flow-operators
        case "scan":
            _prop_scan(eqn, state, _prop_jaxpr)
        case "dot_general":
            _prop_dot_general(eqn, state)
        case "split":
            _prop_split(eqn, state)
        case "stack":
            _prop_stack(eqn, state)
        case "unstack":
            _prop_unstack(eqn, state)
        case "tile":
            _prop_tile(eqn, state)
        case "sort":
            _prop_sort(eqn, state)
        case "cumsum" | "cumprod" | "cummax" | "cummin":
            _prop_cumsum(eqn, state)
        case "qr":
            _prop_qr(eqn, state)
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
            _prop_conservative_fallback(eqn, state)
        case _:
            _prop_throw_error(eqn, state)


def _prop_conservative_fallback(eqn: JaxprEqn, state: _PropState) -> None:
    """Conservative fallback for primitives without precise handlers.

    Assumes worst-case: every output element may depend on every input element.
    This is correct but may overestimate sparsity (more nonzeros than necessary).

    Used for primitives without precise handlers.
    """
    all_inputs: list[IndexSet] = []
    for invar in eqn.invars:
        all_inputs.extend(_index_sets(state, invar))
    for outvar in eqn.outvars:
        state.indices[outvar] = _conservative_indices(all_inputs, _atom_numel(outvar))


def _prop_throw_error(eqn: JaxprEqn, state: _PropState) -> None:
    """Raise an error for unknown primitives.

    This ensures we don't silently produce incorrect sparsity patterns.
    """
    msg = _report_issue(f"No handler for primitive '{eqn.primitive.name}'.")
    raise NotImplementedError(msg)
