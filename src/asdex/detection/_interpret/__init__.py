"""Propagate index sets through a jaxpr to determine sparsity.

Each JAX primitive has a handler that maps input index sets to output index sets.
For example, element-wise ops preserve per-element dependencies, while reductions
union all input dependencies into a single output.

The main entry point is `_prop_jaxpr`, which walks the computation graph
and applies the appropriate handler for each equation.
"""

import logging
from collections.abc import Sequence

from jax._src.core import Jaxpr, JaxprEqn, Var
from jax._src.state.types import AbstractRef

from asdex._errors import _ISSUES_URL

from ._argmax import _prop_argmax
from ._bitcast import _prop_bitcast_convert_type
from ._broadcast import _prop_broadcast_in_dim
from ._common import (
    IndexSet,
    _atom_numel,
    _empty_index_sets,
    _forget_value_info,
    _forward_across_jaxpr_boundary,
    _has_differentiable_dtype,
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
    _prop_max,
    _prop_min,
    _prop_sub,
    _prop_ternary_elementwise,
    _prop_unary_elementwise,
    _prop_zero_derivative_const,
    _prop_zero_derivative_unary_const,
)
from ._equinox._select_if_vmap import _prop_select_if_vmap
from ._gather import _prop_gather
from ._iota import _prop_iota
from ._linalg import _prop_qr
from ._mul import _prop_mul
from ._pad import _prop_pad
from ._pass_through import _prop_pass_through
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
from ._trivial_pattern import _prop_dense_pattern, _prop_empty_pattern
from ._unstack import _prop_unstack
from ._while import _prop_while

_logger = logging.getLogger("asdex.detection")
"""Logger for sparsity detection.

Named after the public subpackage rather than this private module,
so users can configure it without depending on internals
and other parts of asdex can get loggers of their own.
"""


def _prop_jaxpr(
    jaxpr: Jaxpr,
    input_indices: list[list[IndexSet]],
    parent: _PropState | None = None,
) -> list[list[IndexSet]]:
    """Propagate index sets through a jaxpr.

    Runs in a fresh scope:
    index sets are tracked in a new dict so they can be freed when the scope ends,
    while const values, value bounds, and the record of unknown primitives
    are shared with the parent scope by aliasing.

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
        state = _PropState(
            consts=parent.consts,
            bounds=parent.bounds,
            fallbacks=parent.fallbacks,
        )

    # Initialize input variables
    for var, indices in zip(jaxpr.invars, input_indices, strict=False):
        state.indices[var] = indices
    _empty_nondifferentiable_indexsets(jaxpr.invars, state)

    # Initialize constant variables (no input dependencies)
    for var in jaxpr.constvars:
        state.indices[var] = _empty_index_sets(_atom_numel(var))

    # Process each equation.
    # Handlers only record value info they can derive,
    # so clear what a reused jaxpr's outvars kept from an earlier call site.
    for eqn in jaxpr.eqns:
        _forget_value_info(state, eqn.outvars)
        _prop_dispatch(eqn, state)
        _check_index_set_counts(eqn, state)
        _empty_nondifferentiable_indexsets(eqn.outvars, state)

    # Return output dependencies
    return [_index_sets(state, outvar) for outvar in jaxpr.outvars]


def _empty_nondifferentiable_indexsets(
    variables: Sequence[Var], state: _PropState
) -> None:
    """Empty the index sets of variables without a differentiable dtype.

    Values of integer, boolean or PRNG key dtype have zero derivative,
    as in JAX, which gives them a ``float0`` tangent.
    ``_prop_jaxpr`` applies this to the inputs of every jaxpr
    and to the outputs of every equation, whatever their handler wrote.
    Integer inputs differentiated with ``allow_int`` therefore carry no dependencies,
    and handlers need no dtype checks.

    Example: (s, i) = sort(x, iota) where x has index sets [{0}, {1}, {2}]
        The sort handler writes [{0, 1, 2}, ...] for the integer permutation i.
        Output index sets of i: [{}, {}, {}]
    """
    for var in variables:
        if not _has_differentiable_dtype(var) and any(state.indices.get(var, ())):
            state.indices[var] = _empty_index_sets(_atom_numel(var))


def _check_index_set_counts(eqn: JaxprEqn, state: _PropState) -> None:
    """Raise if a handler recorded the wrong number of index sets for an output.

    Each output element needs exactly one index set.
    A wrong count would silently shift or drop rows of the sparsity pattern,
    since rows are assigned by position.

    Example: y = sin(x) where x is f32[3], handler records [{0}, {1}]
        y has 3 elements but only 2 index sets.
        Without this check, y[2] would get no row,
        and every later row would move up by one.
        Raises RuntimeError naming 'sin'.
    """
    for outvar in eqn.outvars:
        if outvar not in state.indices:
            continue
        actual = len(state.indices[outvar])
        expected = _atom_numel(outvar)
        if actual != expected:
            msg = _report_issue(
                f"Handler for '{eqn.primitive.name}' recorded {actual} index sets "
                f"for an output with {expected} elements."
            )
            raise RuntimeError(msg)


def _prop_closed_jaxpr(
    eqn: JaxprEqn,
    state: _PropState,
    param_key: str,
) -> None:
    """Recursively trace a closed jaxpr stored in ``eqn.params[param_key]``.

    Used for call-like primitives such as ``jit`` and ``remat2`` (param ``"jaxpr"``)
    and ``custom_jvp_call`` / ``custom_vjp_call`` (param ``"call_jaxpr"``).
    """
    closed = eqn.params.get(param_key)
    if closed is None:
        msg = _report_issue(
            f"Primitive '{eqn.primitive.name}' has no '{param_key}' parameter."
        )
        raise ValueError(msg)
    _prop_call(eqn, state, closed)


def _prop_call(eqn: JaxprEqn, state: _PropState, closed) -> None:
    """Trace ``closed`` as the body of ``eqn``.

    The jaxpr's invars and outvars must align with the equation's.
    """
    # Unwrap ClosedJaxpr, seeding state.consts for captured constants
    if hasattr(closed, "jaxpr"):
        _seed_const_vals(state, closed.jaxpr.constvars, closed.consts)
        closed = closed.jaxpr

    _forward_across_jaxpr_boundary(state, eqn.invars, closed.invars)
    input_indices = [_index_sets(state, invar) for invar in eqn.invars]
    output_indices = _prop_jaxpr(closed, input_indices, state)

    # Forward consts and bounds back out,
    # so indices computed inside the nested jaxpr stay resolvable outside.
    _forward_across_jaxpr_boundary(state, closed.outvars, eqn.outvars)
    for outvar, indices in zip(eqn.outvars, output_indices, strict=False):
        state.indices[outvar] = indices


def _prop_call_hi_primitive(eqn: JaxprEqn, state: _PropState) -> None:
    """Propagate through a HiJax primitive, identified by its class.

    ``call_hi_primitive`` stores the actual operation as a HiJax primitive object
    (a ``HiPrim``, called ``VJPHiPrimitive`` before JAX 0.11.2)
    in ``eqn.params["_prim"]``.
    Any HiJax primitive without a case falls back to a conservative pattern,
    since its semantics are unknown.
    It is named ``hijax:<module>.<class>`` in logs and ``SparsityPattern.print_fallbacks``,
    which tells it apart from regular primitives and points to its definition.

    Jaxpr:
        invars: operands, aligned with ``_prim.jaxpr.invars``
            (closure consts hoisted to the front)
        _prim: the HiJax primitive, whose ``jaxpr`` maps invars to outvars

    https://docs.jax.dev/en/latest/_autosummary/jax.checkpoint.html
    """
    prim = eqn.params["_prim"]
    match type(prim).__name__:
        # jax.checkpoint under jax_remat3 emits RematTraced, and jax.custom_remat emits CustomRemat.
        # Both evaluate their wrapped jaxpr on the operands, so the pattern is that of the jaxpr.
        # Like custom_vjp_call, custom_remat follows its primal, not its custom backward rule.
        case "RematTraced" | "CustomRemat":
            _prop_call(eqn, state, prim.jaxpr)
        # checkpoint_name under jax_remat3 emits CheckpointName, which returns its operand unchanged.
        case "CheckpointName":
            _prop_pass_through(eqn, state)
        case _:
            prim_type = type(prim)
            name = f"hijax:{prim_type.__module__}.{prim_type.__name__}"
            _prop_fallback(eqn, state, name)


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
            | "round"  # ∂round(x)/∂x = 0
            | "sign"  # ∂sign(x)/∂x = 0
            | "not"
        ):
            _prop_zero_derivative_unary_const(eqn, state)
        # Zero derivative (piecewise constant, ∂f/∂x = 0 a.e.)
        case (
            "is_finite"
            | "clz"
            | "population_count"
            | "reduce_and"
            | "reduce_or"
            | "reduce_xor"
            | "shift_left"
            | "shift_right_arithmetic"
            | "shift_right_logical"
            | "lu_pivots_to_permutation"  # integer permutation of the pivots
            # mulhi(a, b) is the high half of the double-width product of two integers.
            | "mulhi"
            # Equinox's unvmap_any, unvmap_all and unvmap_max reduce a boolean
            # or integer array to a scalar over all elements and all vmap batch members.
            # Equinox uses them in error_if and while loops
            # to decide for a whole batch at once.
            | "unvmap_any"
            | "unvmap_all"
            | "unvmap_max"
        ):
            _prop_empty_pattern(eqn, state)
        # Debug primitives have no outputs, so nothing can depend on them.
        case "debug_print" | "debug_callback":
            _prop_empty_pattern(eqn, state)
        case "clamp":
            _prop_clamp(eqn, state)
        # eq_to, lt_to and le_to come from JAX's lexicographic sort comparators.
        case "eq" | "ne" | "eq_to" | "lt_to" | "le_to":
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
        case "call_hi_primitive":
            _prop_call_hi_primitive(eqn, state)
        case (
            "optimization_barrier"
            | "device_put"
            | "name"
            # Equinox's nonbatchable, nontraceable and nondifferentiable_backward
            # are identities that raise under some transformations.
            # nonbatchable raises under vmap and has no JVP rule,
            # nontraceable raises under every transformation,
            # and nondifferentiable_backward raises only in reverse mode.
            # Its forward-mode Jacobian is the identity.
            # An empty pattern would be wrong for it,
            # and for the other two would make asdex skip AD and return zeros
            # where JAX raises.
            | "nonbatchable"
            | "nontraceable"
            | "nondifferentiable_backward"
        ):
            _prop_pass_through(eqn, state)
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
        # ∂max/∂x = 1 if x>y else 0, ∂max/∂y = 1 if y>x else 0
        case "max":
            _prop_max(eqn, state)
        # ∂min/∂x = 1 if x<y else 0, ∂min/∂y = 1 if y<x else 0
        case "min":
            _prop_min(eqn, state)
        # Binary elementwise with nonzero partials wrt both operands
        case (
            "pow"  # ∂(x^y)/∂x = y·x^(y-1), ∂(x^y)/∂y = x^y·ln(x)
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
            | "log2"  # ∂log₂(x)/∂x = 1/(x·ln(2))
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
            | "one_minus_square"  # ∂(1-x²)/∂x = -2x
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
        case "convert_element_type" | "stop_gradient":
            _prop_convert_element_type(eqn, state)
        case "reduce_precision":
            # Rounding changes const values,
            # so only the elementwise dependencies carry over.
            _prop_unary_elementwise(eqn, state)
        case "bitcast_convert_type":
            _prop_bitcast_convert_type(eqn, state)
        case "conv_general_dilated":
            _prop_conv_general_dilated(eqn, state)
        case "custom_jvp_call" | "custom_vjp_call":
            _prop_closed_jaxpr(eqn, state, "call_jaxpr")
        case "gather":
            _prop_gather(eqn, state)
        case (
            "scatter"
            | "scatter-add"
            | "scatter-mul"
            | "scatter-min"
            | "scatter-max"
            | "scatter-sub"
        ):
            _prop_scatter(eqn, state)
        case "select_n":
            _prop_select_n(eqn, state)
        case "select_if_vmap":
            _prop_select_if_vmap(eqn, state)
        case "iota":
            _prop_iota(eqn, state)
        # Uninitialized buffers have no input dependencies.
        # Unlike iota, no const is recorded:
        # reading undefined contents as zeros could drop nonzeros once the array is filled.
        # empty2 holds the undefined residuals of a cond branch under AD,
        # so it appears in Hessian detection.
        case "empty" | "empty2":
            _prop_empty_pattern(eqn, state)
        # Random number generation is not differentiable:
        # random bits and derived keys have zero derivative with respect to the key,
        # even for the raw uint32 key of rng_bit_generator.
        case (
            "random_seed"
            | "random_unwrap"
            | "random_wrap"
            | "random_split"
            | "random_fold_in"
            | "random_bits"
            | "random_clone"
            | "rng_bit_generator"
        ):
            _prop_empty_pattern(eqn, state)
        case "while":
            _prop_while(eqn, state, _prop_jaxpr)
        case "cond":
            _prop_cond(eqn, state, _prop_jaxpr)
        # Selects the active platform, an input-free scalar.
        # Its value is only known at runtime, so no const is recorded.
        case "platform_index":
            _prop_empty_pattern(eqn, state)
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
        case "cumsum" | "cumprod" | "cummax" | "cummin" | "cumlogsumexp":
            _prop_cumsum(eqn, state)
        case "qr":
            _prop_qr(eqn, state)
        # The body of a callback is not visible in the jaxpr,
        # so the dense pattern is the tightest derivable and not a fallback.
        # jax.pure_callback and io_callback call a Python function on the host that JAX does not trace,
        # so the jaxpr only records the shapes and dtypes of its outputs.
        # The side effects of io_callback do not make the dense pattern less conservative:
        # JAX does not differentiate through the host,
        # so a value that one callback stores and another returns has zero derivative.
        case "pure_callback" | "io_callback":
            _prop_dense_pattern(eqn, state)
        # Known primitives without a precise handler fall back like unknown ones.
        # All of them could treat batch members independently.
        # TODO(lu): the pivot choice has zero derivative,
        # so entry (i, j) of the packed output depends on every row,
        # but only on columns 0 to i - 1 and j for i <= j,
        # and on columns 0 to j for i > j.
        # TODO(cholesky): the strict upper triangle of the output is empty,
        # and L[i, j] depends only on lower-triangle entries A[k, l]
        # with l <= j and either k <= j or k == i.
        # TODO(triangular_solve): for a lower-triangular a,
        # row i of the solution depends only on rows 0 to i of a and b.
        # TODO(schur): the entries of T below its subdiagonal are empty.
        # Most other outputs are generically dense,
        # e.g. hessenberg and tridiagonal pack their reflectors below the band.
        case (
            "lu"
            | "cholesky"
            | "triangular_solve"
            | "svd"
            | "eigh"
            | "eig"
            | "schur"
            | "hessenberg"
            | "tridiagonal"
            | "cholesky_update"
            | "symmetric_product"
            | "householder_product"
            | "ormqr"
            | "tridiagonal_solve"
        ) as name:
            _prop_fallback(eqn, state, name)
        case name:
            _prop_fallback(eqn, state, name)


def _prop_fallback(eqn: JaxprEqn, state: _PropState, name: str) -> None:
    """Fall back to the dense pattern for a primitive without a precise handler.

    The fallback is conservative whenever every dependency that JAX differentiates
    flows from ``eqn.invars`` to ``eqn.outvars``
    (see https://adrianhill.de/asdex/explanation/global-sparsity/#why-conservative-patterns-are-safe).
    Side effects on the host do not break this, since JAX does not differentiate through them.
    Mutable refs do, since a write stores dependencies in the ref, which is an input.
    Primitives on mutable refs therefore raise instead.
    This is by design:
    supporting refs would make conservativeness rest
    on every handler with a nested jaxpr tracking writes,
    rather than on this one argument.

    Each such primitive is logged once per detection
    and recorded in ``state.fallbacks``,
    along with whether any of its fallbacks was conservative,
    meaning that its inputs carried dependencies to an output of differentiable dtype.
    Only conservative fallbacks can cost sparsity,
    so only their primitives are named if the final pattern turns out dense.
    """
    # A write to a ref (``jax.new_ref``) is a dependency that flows into an input,
    # and a later read picks it up from there.
    # The fallback only writes index sets for outvars, so it would miss nonzeros.
    # This limitation is deliberate, so the message does not ask for an issue report.
    if any(isinstance(atom.aval, AbstractRef) for atom in (*eqn.invars, *eqn.outvars)):
        msg = (
            f"Primitive '{name}' operates on a mutable ref (`jax.new_ref`)."
            " Mutable refs are not supported by design:"
            " a write stores dependencies in the ref instead of in an output,"
            " so sparsity detection cannot guarantee a conservative pattern."
            " Rewrite the function with functional updates such as `x.at[idx].set(y)`."
        )
        raise NotImplementedError(msg)

    if name not in state.fallbacks:
        state.fallbacks[name] = False
        _logger.info(
            "No precise handler for primitive '%s', falling back to a conservative pattern."
            " If this primitive is part of JAX,"
            " please help out asdex's development by reporting it at %s",
            name,
            _ISSUES_URL,
        )
    _prop_dense_pattern(eqn, state)
    # The fallback is only conservative if it adds nonzeros to the pattern,
    # i.e. if an output keeps the dependencies the dense pattern gave it.
    # ``_prop_jaxpr`` empties outputs without a differentiable dtype after this returns,
    # so only the others count.
    if any(
        indices
        for outvar in eqn.outvars
        if _has_differentiable_dtype(outvar)
        for indices in state.indices[outvar]
    ):
        state.fallbacks[name] = True
