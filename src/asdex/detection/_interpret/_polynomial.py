"""Propagation rule for polynomial."""

from jax._src.core import JaxprEqn

from ._common import (
    IndexSet,
    _atom_shape,
    _broadcast_flat_map,
    _empty_index_sets,
    _index_sets,
    _numel,
    _permute_indices,
    _PropState,
)


def _prop_polynomial(eqn: JaxprEqn, state: _PropState) -> None:
    """Polynomial evaluates ``a_0 + a_1 x + ... + a_N x^N`` elementwise.

    ``x`` and the coefficients broadcast against each other,
    so each output element depends on the corresponding element of every operand.
    JAX only emits the primitive for ``N >= 1``, so ``x`` always contributes.

    For P(x) = Σ_k a_k x^k element-wise:
        ∂P/∂x = Σ_k k·a_k·x^(k-1)
        ∂P/∂a_k = x^k

    Example: y = polynomial(x, [a, 2.0, 3.0]) where x.shape = (2,) and a.shape = (2,)
        Input index sets:  x = [{0}, {1}], a = [{2}, {3}]
        Output index sets: [{0, 2}, {1, 3}]
        The literal coefficients contribute nothing.

    Jaxpr:
        invars[0]: x, the point of evaluation
        invars[1:]: coefficients a_0, ..., a_N, from lowest to highest degree

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.polynomial.html
    """
    out_shape = _atom_shape(eqn.outvars[0])
    out_indices: list[IndexSet] | None = None
    for invar in eqn.invars:
        in_indices = _index_sets(state, invar)
        # Coefficients are mostly literals, which contribute nothing.
        if not any(in_indices):
            continue
        flat_map = _broadcast_flat_map(_atom_shape(invar), out_shape)
        broadcast = _permute_indices(in_indices, flat_map)
        if out_indices is None:
            out_indices = broadcast
        else:
            out_indices = [
                s1 | s2 for s1, s2 in zip(out_indices, broadcast, strict=True)
            ]

    if out_indices is None:
        out_indices = _empty_index_sets(_numel(out_shape))
    state.indices[eqn.outvars[0]] = out_indices
