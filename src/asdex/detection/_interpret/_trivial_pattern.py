"""Whole-equation patterns that ignore the structure of the primitive.

Both handlers write the same pattern for every output of an equation,
derived from its inputs alone.
They live in their own module so that other handlers
can call them without importing from ``__init__.py``.
"""

from jax._src.core import JaxprEqn

from ._common import (
    IndexSet,
    _atom_numel,
    _empty_index_sets,
    _index_sets,
    _is_inexact,
    _PropState,
    _union_all,
)


def _prop_empty_pattern(eqn: JaxprEqn, state: _PropState) -> None:
    """Every output has zero derivative with respect to every input.

    Applies to piecewise-constant primitives (``floor``, ``sign``, comparisons),
    integer-valued results (``argmax``),
    and primitives without inputs.

    Mathematically, ∂yⱼ/∂xᵢ = 0 almost everywhere, for all i, j.

    Example: y = floor(x) where x has index sets [{0}, {1}, {2}]
        Output index sets: [{}, {}, {}]

    Jaxpr:
        invars: any, not read
        outvars: any number
    """
    for outvar in eqn.outvars:
        state.indices[outvar] = _empty_index_sets(_atom_numel(outvar))


def _prop_dense_pattern(eqn: JaxprEqn, state: _PropState) -> None:
    """Every inexact output element depends on every input element.

    Applies to primitives whose body is not visible in the jaxpr,
    to handlers whose operands are not known at trace time,
    and to primitives without a precise handler, which ``_prop_fallback`` reports first.
    "Dense" is relative to the equation's inputs,
    so the pattern of the whole function can remain sparse.
    One union is shared across all output positions,
    which is safe since index sets in state are never mutated.

    Outputs of non-inexact dtype (integers, booleans, PRNG keys)
    have zero derivative and get empty index sets.

    Mathematically, ∂yⱼ[k]/∂xᵢ[l] may be nonzero for all i, j, k, l
    with yⱼ inexact.

    Example: (u, s, vh) = svd(x) where x has index sets [{0}, {1}, {2}, {3}]
        Output index sets: {0, 1, 2, 3} for every element of u, s and vh

    Example: (lu, pivots, perm) = lu(x) where x has index sets [{0}, {1}, {2}, {3}]
        Output index sets: {0, 1, 2, 3} for every element of lu,
        {} for every element of the integer pivots and perm

    Jaxpr:
        invars: any number, all read
        outvars: any number, each inexact one gets the union of all input index sets
    """
    all_inputs: list[IndexSet] = []
    for invar in eqn.invars:
        all_inputs.extend(_index_sets(state, invar))
    combined = _union_all(all_inputs)
    for outvar in eqn.outvars:
        if _is_inexact(outvar):
            state.indices[outvar] = [combined] * _atom_numel(outvar)
        else:
            state.indices[outvar] = _empty_index_sets(_atom_numel(outvar))
