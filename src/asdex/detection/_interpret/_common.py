"""Types, constants, and utilities for dependency tracking."""

import itertools
import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from jax._src.core import Jaxpr, JaxprEqn, Literal, Var
from numpy.typing import ArrayLike

IndexSet = set[int]
"""A single per-element dependency set.

Backed by Python's built-in set.
Benchmarked against pyroaring.BitMap and int bitmasks;
set[int] wins for the typical workload (small sparse sets, large universe).
"""


def _empty_index_set() -> IndexSet:
    """Create an empty dependency set."""
    return set()


def _singleton_index_set(i: int) -> IndexSet:
    """Create a dependency set containing a single index."""
    return {i}


def _empty_index_sets(n: int) -> list[IndexSet]:
    """Create n empty dependency sets."""
    return [_empty_index_set() for _ in range(n)]


def _identity_index_sets(n: int) -> list[IndexSet]:
    """Create identity sets where element i depends on index i."""
    return [_singleton_index_set(i) for i in range(n)]


StateIndices = dict[Var, list[IndexSet]]
"""Maps each variable to its per-element dependency index sets."""

StateConsts = dict[Var, ArrayLike]
"""Maps variables to their concrete array values (for static index tracking).

Handlers write computed ``np.ndarray`` values.
Seeded closure constants stay in their original array type
(e.g. JAX device arrays)
until ``_atom_const_val`` materializes them to numpy on first read,
so constants that are never read as values are never copied to host.
"""

ValueBounds = tuple[np.ndarray, np.ndarray]
"""Per-element inclusive ``(lo, hi)`` bounds for the elements of one array."""

StateBounds = dict[Var, ValueBounds]
"""Maps variables to per-element inclusive (lo, hi) integer bounds.

Used to track bounded-but-not-constant values
(e.g. output of ``argmax`` over a small axis)
so that dynamic index handlers can enumerate all possible values
instead of falling back to conservative.
"""

Atom = Var | Literal
"""Atomic elements in jaxpressions: named intermediates (Var) or constants (Literal)."""


def _report_issue(msg: str) -> str:
    """Append the standard report-an-issue request to an error message."""
    return (
        f"{msg} "
        "Please help out asdex's development by reporting this at "
        "https://github.com/adrhill/asdex/issues"
    )


@dataclass(slots=True)
class _PropState:
    """Propagation state threaded through every handler.

    ``indices`` is scoped to a single jaxpr:
    each nested jaxpr (cond branch, while body, jit call) gets a fresh dict,
    so intermediate index sets can be freed when the scope ends.
    ``consts`` and ``bounds`` are shared across nested scopes by aliasing.
    JAX caches traced jaxprs, so one inner jaxpr (and its ``Var`` objects)
    can be propagated from several call sites.
    Value info on a jaxpr's vars is therefore overwritten on every entry:
    ``_forward_across_jaxpr_boundary`` clears what it cannot forward,
    ``_forget_value_info`` clears inputs that are never forwarded,
    and ``_prop_jaxpr`` clears each equation's outvars before dispatch.
    """

    indices: StateIndices = field(default_factory=dict)
    consts: StateConsts = field(default_factory=dict)
    bounds: StateBounds = field(default_factory=dict)


PropJaxprFn = Callable[
    [Jaxpr, list[list[IndexSet]], _PropState],
    list[list[IndexSet]],
]
"""Signature of ``_prop_jaxpr``, passed as callback to break circular imports."""


_MAX_ENUM_COMBINATIONS = 64
"""Maximum number of index combinations to enumerate for bounded dynamic indices.

When ``gather``, ``scatter``, ``dynamic_slice``, or ``dynamic_update_slice``
receive indices that are not statically known but have bounded value ranges
(e.g. from ``argmax`` over a small axis),
we enumerate all possible index arrays and union the resulting sparsity patterns.
This yields a tighter pattern than the conservative all-to-all fallback.

The cap prevents combinatorial blowup for multi-element index arrays:
an index with *k* elements where each has *r* possible values
gives *r^k* combinations.
If this exceeds the cap, the handler falls back to conservative.

The value 64 is chosen to keep enumeration fast
while covering the common cases
(e.g. one ``argmax`` index with up to 64 possible values,
or two indices each with up to 8 possible values).
"""


def _bounded_ranges(bounds: ValueBounds) -> list[range]:
    """Build per-element inclusive candidate ranges from (lo, hi) bounds.

    Feeds ``_enumerate_bounded_patterns``:
    element ``i`` of the flattened bounds may take any value in ``range(lo[i], hi[i] + 1)``.
    """
    lo, hi = bounds
    return [
        range(int(lo_i), int(hi_i) + 1)
        for lo_i, hi_i in zip(np.ravel(lo), np.ravel(hi), strict=True)
    ]


def _enumerate_bounded_patterns(
    ranges: Sequence[range],
    out_size: int,
    make_pattern: Callable[[tuple[int, ...]], list[IndexSet] | None],
) -> list[IndexSet] | None:
    """Enumerate all candidate index combinations and union the resulting patterns.

    Used by ``gather``, ``scatter``, ``dynamic_slice``, and ``dynamic_update_slice``
    when indices are bounded but not statically known.
    Each call site builds its own ``ranges`` (from ``_atom_value_bounds``
    or ``_resolve_start_bounds``) and provides a ``make_pattern`` callback
    that computes the sparsity pattern for one concrete index combination.

    Returns ``None`` if the total number of combinations exceeds
    ``_MAX_ENUM_COMBINATIONS`` or if ``make_pattern`` returns ``None``
    (indicating an unrecognized pattern, as in scatter).
    """
    if math.prod(len(r) for r in ranges) > _MAX_ENUM_COMBINATIONS:
        return None

    accumulated: list[IndexSet] | None = None
    for candidate_values in itertools.product(*ranges):
        pattern = make_pattern(candidate_values)
        if pattern is None:
            return None
        if accumulated is None:
            accumulated = pattern
        else:
            for i in range(out_size):
                accumulated[i] = accumulated[i] | pattern[i]

    return accumulated


def _merge_index_dependencies(
    result: list[IndexSet], index_sets: list[IndexSet]
) -> list[IndexSet]:
    """Union the index operand's own index sets into every enumerated pattern.

    ``_enumerate_bounded_patterns`` resolves which operand positions
    a bounded dynamic index may read or write,
    but the index operand itself may depend on the function's inputs.
    Those dependencies reach every position the enumeration produced,
    so they are unioned in afterwards.

    Builds a new list rather than mutating ``result`` in place,
    which is required since enumerated patterns may alias input index sets.
    """
    if not any(index_sets):
        return result
    combined = _union_all(index_sets)
    return [iset | combined for iset in result]


# Shape and size


def _numel(shape: Sequence[int]) -> int:
    """Compute the total number of elements from a shape tuple."""
    return math.prod(shape) if shape else 1


def _atom_shape(atom: Atom) -> tuple[int, ...]:
    """Get the shape of a variable or literal."""
    if isinstance(atom, Literal):
        return tuple(getattr(atom.val, "shape", ()))
    return tuple(getattr(atom.aval, "shape", ()))


def _atom_numel(atom: Atom) -> int:
    """Get the total number of elements in a variable or literal."""
    return _numel(_atom_shape(atom))


# Atom value access


def _index_sets(state: _PropState, atom: Atom) -> list[IndexSet]:
    """Get the index sets for a variable or literal.

    Every ``Var`` is either seeded (invars, constvars) or written by a handler,
    so a missing ``Var`` indicates a handler bug upstream.
    Guessing a default here would silently drop dependencies
    and get the element count wrong,
    so we raise instead.
    """
    if isinstance(atom, Literal):
        return _empty_index_sets(_atom_numel(atom))
    if atom not in state.indices:
        msg = _report_issue(f"No index sets recorded for variable '{atom}'.")
        raise KeyError(msg)
    return state.indices[atom]


def _copy_index_set(src: IndexSet) -> IndexSet:
    """Copy a single index set.

    Used by handlers that accumulate into a set with ``|=``
    and therefore need to own it.
    """
    return src.copy()


def _copy_index_sets(src: list[IndexSet]) -> list[IndexSet]:
    """Deep-copy a list of index sets.

    Inlines the copy rather than calling ``_copy_index_set`` per element,
    since this runs once per carry element in the ``cond`` and ``while`` loops.
    Both are backend-specific, which is why both live here.
    """
    return [s.copy() for s in src]


def _atom_const_val(atom: Atom, state: _PropState) -> np.ndarray | None:
    """Get the concrete value of an atom, if statically known.

    The value is known in two cases:
    - **Literals**: constants embedded directly in the jaxpr.
    - **Tracked vars**: variables in ``state.consts``, whose values were
      computed from constants through earlier operations.

    Lazily seeded consts (see ``_seed_const_vals``)
    are materialized to numpy on first read and cached.

    Returns ``None`` when the value depends on runtime inputs.
    """
    if isinstance(atom, Literal):
        return np.asarray(atom.val)
    if isinstance(atom, Var) and atom in state.consts:
        val = state.consts[atom]
        if not isinstance(val, np.ndarray):
            val = np.asarray(val)
            state.consts[atom] = val
        return val
    return None


def _atom_value_bounds(
    atom: Atom,
    state: _PropState,
) -> ValueBounds | None:
    """Get per-element inclusive (lo, hi) bounds for an atom.

    Returns exact ``(val, val)`` for constants,
    tracked bounds for bounded variables,
    or ``None`` when no information is available.
    """
    val = _atom_const_val(atom, state)
    if val is not None:
        return (val, val)
    if isinstance(atom, Var) and atom in state.bounds:
        return state.bounds[atom]
    return None


def _exact_ints(a: np.ndarray) -> np.ndarray:
    """Widen an integer array to Python ints so bounds arithmetic cannot wrap.

    Interval arithmetic in the operand dtype silently wraps on overflow
    (e.g. int8 ``120 + 15`` gives ``-121``),
    yielding bounds that exclude values the program computes.
    Arbitrary-precision ints keep the endpoints exact,
    and ``_set_value_bounds`` then drops any interval
    that does not fit the output dtype.
    Non-integer arrays are returned unchanged.
    """
    a = np.asarray(a)
    if np.issubdtype(a.dtype, np.integer):
        return a.astype(object)
    return a


def _set_value_bounds(
    state: _PropState, var: Var, lo: np.ndarray, hi: np.ndarray
) -> None:
    """Store ``(lo, hi)`` as the value bounds of ``var`` if they are sound.

    Every bounds write goes through here,
    so stored bounds always satisfy ``lo <= hi``
    and, for integer outputs, lie within the output dtype's range.
    Bounds outside that range mean the computation may wrap around,
    so the real values are not an interval and nothing is stored.
    """
    lo, hi = np.asarray(lo), np.asarray(hi)
    if not np.all(lo <= hi):
        return
    aval_dtype = getattr(var.aval, "dtype", None)
    if aval_dtype is None:
        return
    dtype = np.dtype(aval_dtype)
    if np.issubdtype(dtype, np.integer):
        info = np.iinfo(dtype.type)
        if not (np.all(lo >= info.min) and np.all(hi <= info.max)):
            return
        lo = lo.astype(dtype, copy=False)
        hi = hi.astype(dtype, copy=False)
    state.bounds[var] = (lo, hi)


def _binary_value_bounds(
    eqn: JaxprEqn,
    state: _PropState,
) -> tuple[ValueBounds, ValueBounds] | None:
    """Get value bounds for both operands of a binary op, or ``None`` if either is unknown.

    The first operand is checked before the second is read,
    so an input-dependent first operand (e.g. ``x + bias``)
    does not force materializing a large second-operand const
    whose bounds would be discarded anyway.
    Mirrors the same early return in ``_propagate_const_binary``.
    """
    b1 = _atom_value_bounds(eqn.invars[0], state)
    if b1 is None:
        return None
    b2 = _atom_value_bounds(eqn.invars[1], state)
    if b2 is None:
        return None
    return b1, b2


def _ternary_value_bounds(
    eqn: JaxprEqn,
    state: _PropState,
) -> tuple[ValueBounds, ValueBounds, ValueBounds] | None:
    """Get value bounds for all three operands of a ternary op.

    Returns ``None`` as soon as an operand's bounds are unknown,
    for the same reason as in ``_binary_value_bounds``:
    a later operand's const is never materialized
    once the result is known to be discarded.
    """
    b1 = _atom_value_bounds(eqn.invars[0], state)
    if b1 is None:
        return None
    b2 = _atom_value_bounds(eqn.invars[1], state)
    if b2 is None:
        return None
    b3 = _atom_value_bounds(eqn.invars[2], state)
    if b3 is None:
        return None
    return b1, b2, b3


def _propagate_const_unary(
    eqn: JaxprEqn,
    state: _PropState,
    transform: Callable[[np.ndarray], np.ndarray],
) -> None:
    """Propagate a const value through a unary op.

    If the input is statically known,
    apply ``transform`` and store the result.
    Without this, downstream handlers (e.g. ``gather``, ``scatter``) cannot resolve
    static index arrays and fall back to conservative.
    """
    in_val = _atom_const_val(eqn.invars[0], state)
    if in_val is not None:
        state.consts[eqn.outvars[0]] = transform(in_val)


def _propagate_const_binary(
    eqn: JaxprEqn,
    state: _PropState,
    transform: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> None:
    """Propagate a const value through a binary op.

    If both inputs are statically known,
    apply ``transform`` and store the result.
    Without this, downstream handlers (e.g. ``gather``, ``scatter``) cannot resolve
    static index arrays and fall back to conservative.
    """
    in1 = _atom_const_val(eqn.invars[0], state)
    if in1 is None:
        # Skip reading the second operand,
        # so an input-dependent operand (e.g. x + bias)
        # does not force materializing a large const.
        return
    in2 = _atom_const_val(eqn.invars[1], state)
    if in2 is not None:
        state.consts[eqn.outvars[0]] = transform(in1, in2)


# Zero-skipping


def _clear_where_zero(
    eqn: JaxprEqn,
    state: _PropState,
    invar_idx: int,
) -> None:
    """Clear output index sets at positions where an input is a known constant zero.

    Used by ``mul``, ``div``, and ``integer_pow`` for zero-skipping:
    ``d(0 * y)/dy = 0``, ``d(0 / y)/dy = 0``, ``d(0^n)/dx = 0`` for ``n > 1``.

    Builds a new output list instead of mutating in place,
    so it is safe on output lists that alias an input's list.
    """
    val = _atom_const_val(eqn.invars[invar_idx], state)
    if val is None:
        return
    out_shape = _atom_shape(eqn.outvars[0])
    in_shape = _atom_shape(eqn.invars[invar_idx])
    flat = np.ravel(val)[_broadcast_flat_map(in_shape, out_shape)]

    out_indices = state.indices[eqn.outvars[0]]
    empty = _empty_index_set()
    state.indices[eqn.outvars[0]] = [
        empty if flat[i] == 0 else s for i, s in enumerate(out_indices)
    ]


# Index set operations


def _union_all(sets: Sequence[IndexSet]) -> IndexSet:
    """Union all sets together, returning a new set."""
    if not sets:
        return _empty_index_set()
    result: IndexSet = _empty_index_set()
    for s in sets:
        result |= s
    return result


def _union_elementwise(
    inputs: Sequence[list[IndexSet]], out_size: int
) -> list[IndexSet]:
    """Union multiple index set lists element-wise with scalar broadcasting.

    Each input list represents per-element index sets for one operand.
    Scalars (length 1) broadcast to match the output size via modular indexing.
    """
    return [_union_all([inp[i % len(inp)] for inp in inputs]) for i in range(out_size)]


def _check_no_index_sets(state: _PropState, atom: Atom, primitive_name: str) -> None:
    """Verify that an atom carries no input dependencies.

    Some handlers assume that auxiliary inputs
    (index arrays, kernel weights, selectors)
    are constants with empty dependency sets.
    This function validates that assumption
    and raises an informative error when it is violated.
    """
    if any(_index_sets(state, atom)):
        msg = _report_issue(
            f"'{primitive_name}' handler assumes an auxiliary input "
            "has no dependency on the function's inputs, "
            "but found non-empty index sets."
        )
        raise ValueError(msg)


def _conservative_indices(all_indices: list[IndexSet], out_size: int) -> list[IndexSet]:
    """Build conservative output index sets where every element depends on the union of all inputs."""
    combined = _union_all(all_indices)
    return [combined] * out_size


# Index clamping


def _clamp_starts(
    starts: Sequence[int], in_shape: Sequence[int], slice_sizes: Sequence[int]
) -> tuple[int, ...]:
    """Clamp start indices to valid bounds.

    Matches JAX's ``dynamic_slice`` and ``gather`` semantics,
    which silently clamp out-of-bounds starts
    rather than raising an error.
    """
    return tuple(
        max(0, min(s, dim - sz))
        for s, dim, sz in zip(starts, in_shape, slice_sizes, strict=True)
    )


# Position maps


def _position_map(shape: Sequence[int]) -> np.ndarray:
    """Build an array where each element holds its own flat position.

    For shape ``(2, 3)``, returns ``[[0, 1, 2], [3, 4, 5]]``.
    Applying operations (transpose, slice, etc.) to this array
    reveals which input position each output position reads from.
    """
    return np.arange(_numel(shape)).reshape(shape)


def _broadcast_flat_map(
    in_shape: tuple[int, ...], out_shape: tuple[int, ...]
) -> np.ndarray:
    """Map each output position to the input position it reads under broadcasting.

    Follows numpy broadcasting rules:
    the input shape is left-padded with 1s to the output ndim,
    and size-1 dims always read index 0.
    Returns a flat integer array of length ``numel(out_shape)``.
    For const values, ``np.ravel(val)[flat_map]`` broadcasts the value itself.
    """
    padded = (1,) * (len(out_shape) - len(in_shape)) + tuple(in_shape)
    return np.broadcast_to(_position_map(padded), out_shape).ravel()


def _permute_indices(
    in_indices: list[IndexSet], flat_map: Sequence[int] | np.ndarray
) -> list[IndexSet]:
    """Build output index sets by looking up input positions from a flat map.

    Each output element copies its index set from ``in_indices[flat_map[i]]``.
    Used by handlers that already have a precomputed flat integer map
    (broadcast, tile, gather).
    """
    return [in_indices[j] for j in flat_map]


def _transform_indices(
    in_indices: list[IndexSet],
    in_shape: Sequence[int],
    transform: Callable[[np.ndarray], np.ndarray] = lambda p: p,
) -> list[IndexSet]:
    """Build output index sets by transforming a position map.

    Creates a position map for ``in_shape``
    (an array where element ``i`` holds value ``i``),
    applies ``transform``,
    and uses the result to look up index sets from ``in_indices``.

    Each output element copies its index set from the input position
    determined by the transformed position map.
    This is the common pattern for permutation-like ops
    (transpose, rev, slice, reshape, split, dynamic_slice)
    where each output reads exactly one input element.
    """
    flat_map = transform(_position_map(in_shape)).ravel()
    return _permute_indices(in_indices, flat_map)


def _join_inputs(
    eqn: JaxprEqn,
    state: _PropState,
    join: Callable[[list[np.ndarray], int], np.ndarray],
    axis: int,
) -> None:
    """Shared core for stack and concatenate.

    Pools every input's flat index sets into one list.
    For each input, builds a shaped array whose values are positions in that pool.
    Applying ``join`` to these index arrays mirrors the real op's element shuffling,
    giving a flat mapping from each output element to the pool position it came from.
    Also joins const values so downstream gather/scatter can resolve indices.
    """
    all_indices: list[IndexSet] = []
    index_arrays = []
    for invar in eqn.invars:
        in_indices = _index_sets(state, invar)
        offset = len(all_indices)
        all_indices.extend(in_indices)
        shape = _atom_shape(invar)
        index_arrays.append(np.arange(offset, offset + len(in_indices)).reshape(shape))

    permutation_map = join(index_arrays, axis).ravel()
    state.indices[eqn.outvars[0]] = [all_indices[i] for i in permutation_map]

    vals = [_atom_const_val(v, state) for v in eqn.invars]
    if all(v is not None for v in vals):
        state.consts[eqn.outvars[0]] = join([v for v in vals if v is not None], axis)


# Coordinate helpers


def _row_strides(shape: Sequence[int]) -> tuple[int, ...]:
    """Compute row-major strides for multi-dimensional index tracking.

    Used to build flat indices from coordinates
    when a handler tracks dependencies per dimension (pad, conv, dot_general).
    Each stride tells how many flat elements to skip
    when incrementing one coordinate position.

    For shape (2, 3, 4): _row_strides = (12, 4, 1) since moving one step in dim 0
    skips 3*4=12 elements, dim 1 skips 4 elements, and dim 2 skips 1 element.
    """
    result: list[int] = []
    stride = 1
    for dim in reversed(shape):
        result.append(stride)
        stride *= dim
    return tuple(reversed(result))


def _dim_offsets(shape: Sequence[int], dims: Sequence[int]) -> np.ndarray:
    """Flat positions of every coordinate over ``dims``, with all other dims at zero.

    The coordinates over ``dims`` are enumerated in C order,
    taking ``dims`` in the given order rather than sorted.
    Adding the offsets of complementary dims yields a full flat position,
    which is how dot_general and conv split positions into a base and an offset.

    For shape (2, 3), the flat positions are [[0, 1, 2], [3, 4, 5]], so

    - dims (0,) gives [0, 3], down the first column.
    - dims (1,) gives [0, 1, 2], along the first row.
    - dims (1, 0) gives [0, 3, 1, 4, 2, 5], the whole array in transposed order.
    """
    strides = _row_strides(shape)
    sizes = tuple(shape[d] for d in dims)
    coords = (
        np.indices(sizes, dtype=np.int64).reshape(len(dims), -1)
        if sizes
        else np.zeros((0, 1), dtype=np.int64)
    )
    offsets = np.zeros(_numel(sizes), dtype=np.int64)
    for i, d in enumerate(dims):
        offsets += coords[i] * strides[d]
    return offsets


# Const value propagation


def _seed_const_vals(state: _PropState, constvars, consts) -> None:
    """Populate ``state.consts`` for the captured constants of a ClosedJaxpr.

    Without this, gather/scatter inside nested jaxprs (cond branches,
    while bodies, jit-wrapped calls) cannot resolve closure-captured
    index arrays and fall back to conservative.

    Values are stored as-is rather than converted with ``np.asarray``:
    conversion copies device arrays to host and keeps the copies alive
    for the whole analysis,
    which is wasted work for constants that are never read as values
    (e.g. convolution kernels, whose values do not affect the pattern).
    ``_atom_const_val`` materializes on first read and caches.
    """
    for var, val in zip(constvars, consts, strict=True):
        state.consts[var] = val


def _forward_across_jaxpr_boundary(
    state: _PropState, src_atoms: Sequence[Atom], dst_vars
) -> None:
    """Transfer const values and value bounds across a nested-jaxpr boundary.

    At a nested jaxpr (cond branch, while body, jit call),
    the outer equation's atoms and the inner jaxpr's variables are different
    ``Var`` objects representing the same values.
    This copies any concrete values and value bounds from the source atoms
    to the corresponding destination vars so that downstream handlers
    (gather, scatter, dynamic_slice) can resolve indices precisely.
    Consts and bounds are forwarded together
    so a call site cannot forward one and forget the other.

    The direction is set by the caller:
    inward maps outer invars to inner invars,
    outward maps inner outvars to outer outvars.
    Both are the same operation, atoms mapped to fresh vars.

    Tracked const values are forwarded as stored, without materializing:
    reading them here would force the host copies
    that ``_seed_const_vals`` deliberately defers
    at every nested-jaxpr boundary.

    Destinations whose source has no const value or bounds are cleared,
    so a reused jaxpr does not keep value info from an earlier call site.
    """
    for src, dst in zip(src_atoms, dst_vars, strict=False):
        _forget_value_info(state, [dst])
        if isinstance(src, Literal):
            state.consts[dst] = np.asarray(src.val)
        elif isinstance(src, Var):
            if src in state.consts:
                state.consts[dst] = state.consts[src]
            if src in state.bounds:
                state.bounds[dst] = state.bounds[src]


def _forget_value_info(state: _PropState, variables: Sequence[Var]) -> None:
    """Drop any const values and value bounds stored on ``variables``.

    Used for nested-jaxpr inputs that are never forwarded
    (e.g. loop carries, whose values change every iteration)
    and for equation outputs before they are recomputed.
    Without it, a jaxpr reused from an earlier call site
    would still carry that call's value info.
    """
    for var in variables:
        state.consts.pop(var, None)
        state.bounds.pop(var, None)
