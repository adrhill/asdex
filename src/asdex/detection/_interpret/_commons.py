"""Types, constants, and utilities for dependency tracking."""

import itertools
import math
from collections.abc import Callable, Sequence

import numpy as np
from jax._src.core import Jaxpr, JaxprEqn, Literal, Var

IndexSet = set[int]
"""A single per-element dependency set.

Backed by Python's built-in set.
Benchmarked against pyroaring.BitMap and int bitmasks;
set[int] wins for the typical workload (small sparse sets, large universe).
"""


def empty_index_set() -> IndexSet:
    """Create an empty dependency set."""
    return set()


def singleton_index_set(i: int) -> IndexSet:
    """Create a dependency set containing a single index."""
    return {i}


def empty_index_sets(n: int) -> list[IndexSet]:
    """Create n empty dependency sets."""
    return [empty_index_set() for _ in range(n)]


def identity_index_sets(n: int) -> list[IndexSet]:
    """Create identity sets where element i depends on index i."""
    return [singleton_index_set(i) for i in range(n)]


StateIndices = dict[Var, list[IndexSet]]
"""Maps each variable to its per-element dependency index sets."""

StateConsts = dict[Var, np.ndarray]
"""Maps variables to their concrete numpy array values (for static index tracking)."""

StateBounds = dict[Var, tuple[np.ndarray, np.ndarray]]
"""Maps variables to per-element inclusive (lo, hi) integer bounds.

Used to track bounded-but-not-constant values
(e.g. output of ``argmax`` over a small axis)
so that dynamic index handlers can enumerate all possible values
instead of falling back to conservative.
"""

Atom = Var | Literal
"""Atomic elements in jaxpressions: named intermediates (Var) or constants (Literal)."""

PropJaxprFn = Callable[
    [Jaxpr, list[list[IndexSet]], StateConsts | None], list[list[IndexSet]]
]
"""Signature of ``prop_jaxpr``, passed as callback to break circular imports."""


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


def enumerate_bounded_patterns(
    ranges: Sequence[range],
    out_size: int,
    make_pattern: Callable[[tuple[int, ...]], list[IndexSet] | None],
) -> list[IndexSet] | None:
    """Enumerate all candidate index combinations and union the resulting patterns.

    Used by ``gather``, ``scatter``, ``dynamic_slice``, and ``dynamic_update_slice``
    when indices are bounded but not statically known.
    Each call site builds its own ``ranges`` (from ``atom_value_bounds``
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


# Shape and size


def numel(shape: Sequence[int]) -> int:
    """Compute the total number of elements from a shape tuple."""
    return math.prod(shape) if shape else 1


def atom_shape(atom: Atom) -> tuple[int, ...]:
    """Get the shape of a variable or literal."""
    if isinstance(atom, Literal):
        return tuple(getattr(atom.val, "shape", ()))
    return tuple(getattr(atom.aval, "shape", ()))


def atom_numel(atom: Atom) -> int:
    """Get the total number of elements in a variable or literal."""
    if isinstance(atom, Literal):
        shape = getattr(atom.val, "shape", ())
        return numel(tuple(shape)) if shape else 1
    shape = getattr(atom.aval, "shape", ())
    return numel(tuple(shape)) if shape else 1


# Atom value access


def index_sets(state_indices: StateIndices, atom: Atom) -> list[IndexSet]:
    """Get the index sets for a variable or literal."""
    if isinstance(atom, Literal):
        return empty_index_sets(atom_numel(atom))
    return state_indices.get(atom, [empty_index_set()])


def copy_index_sets(src: list[IndexSet]) -> list[IndexSet]:
    """Deep-copy a list of index sets."""
    return [s.copy() for s in src]


def atom_const_val(atom: Atom, state_consts: StateConsts) -> np.ndarray | None:
    """Get the concrete value of an atom, if statically known.

    The value is known in two cases:
    - **Literals**: constants embedded directly in the jaxpr.
    - **Tracked vars**: variables in ``state_consts``, whose values were
      computed from constants through earlier operations.

    Returns ``None`` when the value depends on runtime inputs.
    """
    if isinstance(atom, Literal):
        return np.asarray(atom.val)
    if isinstance(atom, Var) and atom in state_consts:
        return state_consts[atom]
    return None


def atom_value_bounds(
    atom: Atom,
    state_consts: StateConsts,
    state_bounds: StateBounds,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Get per-element inclusive (lo, hi) bounds for an atom.

    Returns exact ``(val, val)`` for constants,
    tracked bounds for bounded variables,
    or ``None`` when no information is available.
    """
    val = atom_const_val(atom, state_consts)
    if val is not None:
        return (val, val)
    if isinstance(atom, Var) and atom in state_bounds:
        return state_bounds[atom]
    return None


def propagate_const_unary(
    eqn: JaxprEqn,
    state_consts: StateConsts,
    transform: Callable[[np.ndarray], np.ndarray],
) -> None:
    """Propagate a const value through a unary op.

    If the input is statically known,
    apply ``transform`` and store the result.
    Without this, downstream handlers (e.g. ``gather``, ``scatter``) cannot resolve
    static index arrays and fall back to conservative.
    """
    in_val = atom_const_val(eqn.invars[0], state_consts)
    if in_val is not None:
        state_consts[eqn.outvars[0]] = transform(in_val)


def propagate_const_binary(
    eqn: JaxprEqn,
    state_consts: StateConsts,
    transform: Callable[[np.ndarray, np.ndarray], np.ndarray],
) -> None:
    """Propagate a const value through a binary op.

    If both inputs are statically known,
    apply ``transform`` and store the result.
    Without this, downstream handlers (e.g. ``gather``, ``scatter``) cannot resolve
    static index arrays and fall back to conservative.
    """
    in1 = atom_const_val(eqn.invars[0], state_consts)
    in2 = atom_const_val(eqn.invars[1], state_consts)
    if in1 is not None and in2 is not None:
        state_consts[eqn.outvars[0]] = transform(in1, in2)


# Zero-skipping


def broadcast_to_output(
    val: np.ndarray, in_shape: tuple[int, ...], out_shape: tuple[int, ...]
) -> np.ndarray:
    """Broadcast a const value from input shape to output shape, returning a flat array.

    Handles numpy-style broadcasting: left-pads with 1s then expands.
    """
    ndim = len(out_shape)
    arr = np.asarray(val).reshape(in_shape) if in_shape else np.asarray(val)
    pad = ndim - len(in_shape)
    padded_shape = (1,) * pad + in_shape
    return np.broadcast_to(arr.reshape(padded_shape), out_shape).ravel()


def clear_where_zero(
    eqn: JaxprEqn,
    state_indices: StateIndices,
    state_consts: StateConsts,
    invar_idx: int,
) -> None:
    """Clear output index sets at positions where an input is a known constant zero.

    Used by ``mul``, ``div``, and ``integer_pow`` for zero-skipping:
    ``d(0 * y)/dy = 0``, ``d(0 / y)/dy = 0``, ``d(0^n)/dx = 0`` for ``n > 1``.
    """
    val = atom_const_val(eqn.invars[invar_idx], state_consts)
    if val is None:
        return
    out_shape = atom_shape(eqn.outvars[0])
    in_shape = atom_shape(eqn.invars[invar_idx])
    flat = broadcast_to_output(val, in_shape, out_shape)

    out_indices = state_indices[eqn.outvars[0]]
    for i in range(len(out_indices)):
        if flat[i] == 0:
            out_indices[i] = empty_index_set()


# Index set operations


def union_all(sets: Sequence[IndexSet]) -> IndexSet:
    """Union all sets together, returning a new set."""
    if not sets:
        return empty_index_set()
    result: IndexSet = empty_index_set()
    for s in sets:
        result |= s
    return result


def check_no_index_sets(
    state_indices: StateIndices, atom: Atom, primitive_name: str
) -> None:
    """Verify that an atom carries no input dependencies.

    Some handlers assume that auxiliary inputs
    (index arrays, kernel weights, selectors)
    are constants with empty dependency sets.
    This function validates that assumption
    and raises an informative error when it is violated.
    """
    if any(index_sets(state_indices, atom)):
        msg = (
            f"'{primitive_name}' handler assumes an auxiliary input "
            "has no dependency on the function's inputs, "
            "but found non-empty index sets. "
            "Please help out asdex's development by reporting this at https://github.com/adrhill/asdex/issues"
        )
        raise ValueError(msg)


def conservative_indices(all_indices: list[IndexSet], out_size: int) -> list[IndexSet]:
    """Build conservative output index sets where every element depends on the union of all inputs."""
    combined = union_all(all_indices)
    return [combined] * out_size


# Index clamping


def clamp_starts(
    starts: tuple[int, ...], in_shape: Sequence[int], slice_sizes: Sequence[int]
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


def position_map(shape: Sequence[int]) -> np.ndarray:
    """Build an array where each element holds its own flat position.

    For shape ``(2, 3)``, returns ``[[0, 1, 2], [3, 4, 5]]``.
    Applying operations (transpose, slice, etc.) to this array
    reveals which input position each output position reads from.
    """
    return np.arange(numel(shape)).reshape(shape)


def permute_indices(
    in_indices: list[IndexSet], flat_map: Sequence[int] | np.ndarray
) -> list[IndexSet]:
    """Build output index sets by looking up input positions from a flat map.

    Each output element copies its index set from ``in_indices[flat_map[i]]``.
    Used by handlers that already have a precomputed flat integer map
    (broadcast, tile, gather).
    """
    return [in_indices[j] for j in flat_map]


def transform_indices(
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
    flat_map = transform(position_map(in_shape)).ravel()
    return permute_indices(in_indices, flat_map)


# Coordinate helpers


def row_strides(shape: Sequence[int]) -> tuple[int, ...]:
    """Compute row-major strides for multi-dimensional index tracking.

    Used to convert between flat indices and coordinates when propagating
    dependencies through slice and broadcast_in_dim.
    Each stride tells how many flat elements to skip
    when incrementing one coordinate position.

    For shape (2, 3, 4): row_strides = (12, 4, 1) since moving one step in dim 0
    skips 3*4=12 elements, dim 1 skips 4 elements, and dim 2 skips 1 element.
    """
    result: list[int] = []
    stride = 1
    for dim in reversed(shape):
        result.append(stride)
        stride *= dim
    return tuple(reversed(result))


def flat_to_coords(flat: int, strides: tuple[int, ...]) -> list[int]:
    """Convert a flat index to multi-dimensional coordinates using row-major strides."""
    coord = []
    remaining = flat
    for s in strides:
        coord.append(remaining // s)
        remaining %= s
    return coord


# Const value propagation


def seed_const_vals(state_consts: StateConsts, constvars, consts) -> None:
    """Populate state_consts for the captured constants of a ClosedJaxpr.

    Without this, gather/scatter inside nested jaxprs (cond branches,
    while bodies, jit-wrapped calls) cannot resolve closure-captured
    index arrays and fall back to conservative.
    """
    for var, val in zip(constvars, consts, strict=True):
        state_consts[var] = np.asarray(val)


def forward_value_bounds(
    state_bounds: StateBounds, outer_atoms: Sequence[Atom], inner_vars
) -> None:
    """Transfer known value bounds from outer-scope atoms to inner jaxpr variables.

    Same idea as ``forward_const_vals`` but for value bounds.
    """
    for outer, inner in zip(outer_atoms, inner_vars, strict=False):
        if isinstance(outer, Var) and outer in state_bounds:
            state_bounds[inner] = state_bounds[outer]


def forward_const_vals(
    state_consts: StateConsts, outer_atoms: Sequence[Atom], inner_vars
) -> None:
    """Transfer known state_consts from outer-scope atoms to inner jaxpr variables.

    When entering a nested jaxpr (cond branch, while body, jit call),
    the outer equation's invars and the inner jaxpr's invars are different
    ``Var`` objects representing the same values.
    This copies any concrete values from the outer atoms
    to the corresponding inner vars so that downstream handlers
    (gather, scatter, dynamic_slice) can resolve indices precisely.
    """
    for outer, inner in zip(outer_atoms, inner_vars, strict=False):
        val = atom_const_val(outer, state_consts)
        if val is not None:
            state_consts[inner] = val
