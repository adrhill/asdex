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
    _PropState,
    _union_all,
)


def _prop_empty_pattern(eqn: JaxprEqn, state: _PropState) -> None:
    """Every output has zero derivative with respect to every input.

    Use it for primitives whose outputs cannot change
    under infinitesimal changes of their inputs:
    piecewise-constant ops (``floor``, ``sign``, comparisons, bitwise ops),
    integer-valued results (``argmax``, ``lu_pivots_to_permutation``),
    and primitives without inputs or without outputs.
    A primitive without outputs makes this a no-op.

    Mathematically, for f in {floor, ceil, sign, ...}:
        ∂f/∂x = 0  (almost everywhere)
    Therefore, output elements have no dependencies on input elements.

    Example: y = floor(x) where x = [1.7, 2.3, 3.9]
        Input index sets:  [{0}, {1}, {2}]
        Output index sets: [{}, {}, {}]  (empty sets, no dependence)

    Jaxpr:
        invars: any, not read
        outvars: any number, each gets one empty set per element
    """
    for outvar in eqn.outvars:
        state.indices[outvar] = _empty_index_sets(_atom_numel(outvar))


def _prop_dense_pattern(eqn: JaxprEqn, state: _PropState) -> None:
    """Every output element depends on every input element.

    Use it for known primitives
    whose intended pattern is the union of all inputs,
    e.g. because their outputs are generically dense
    or because their body is not visible in the jaxpr.
    It is also the pattern for unknown primitives,
    which ``_prop_unknown_primitive`` reports before calling this handler.

    "Dense" is relative to the equation's own inputs:
    an output only depends on what the inputs depend on.
    Unlike ``DensePatternWarning``, which is about the final pattern of the whole function,
    a dense pattern here can still leave the final pattern sparse.

    The union is computed once per equation
    and shared across every position of every output,
    which is safe since index sets read from state are never mutated.

    Mathematically, for outputs yⱼ and inputs xᵢ:
        ∂yⱼ[k]/∂xᵢ[l] may be nonzero for all i, j, k, l

    Example: (u, s, vh) = svd(x) where x has index sets [{0}, {1}, {2}, {3}]
        Output index sets: {0, 1, 2, 3} for every element of u, s and vh

    Jaxpr:
        invars: any number, all read
        outvars: any number, each gets the union of all input index sets
    """
    all_inputs: list[IndexSet] = []
    for invar in eqn.invars:
        all_inputs.extend(_index_sets(state, invar))
    combined = _union_all(all_inputs)
    for outvar in eqn.outvars:
        state.indices[outvar] = [combined] * _atom_numel(outvar)
