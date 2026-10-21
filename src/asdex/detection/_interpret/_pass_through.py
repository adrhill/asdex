"""Propagation rule for primitives that return their operands unchanged."""

from jax._src.core import JaxprEqn

from ._common import _forward_across_jaxpr_boundary, _index_sets, _PropState


def _prop_pass_through(eqn: JaxprEqn, state: _PropState) -> None:
    """Pass-through primitives return each operand unchanged.

    ``optimization_barrier`` only stops XLA from moving computations across it,
    ``device_put`` only changes where an array lives,
    and ``name``, which ``checkpoint_name`` emits outside of ``jax_remat3``,
    only tags its operand for ``jax.checkpoint`` policies,
    so each output equals its operand: ∂yᵢ/∂xᵢ = I and ∂yᵢ/∂xⱼ = 0 for i ≠ j.
    Index sets are aliased, and const values and bounds are forwarded,
    so indices that pass through stay resolvable downstream.
    JAX inserts both in the gradients of ``jax.checkpoint`` under ``jax_remat3``,
    the latter for policies that offload residuals to host memory.

    Example: (y1, y2) = optimization_barrier((x1, x2)) where x1 = [a, b], x2 = [c]
        Input index sets:  [{0}, {1}], [{2}]
        Output index sets: [{0}, {1}], [{2}]

    Jaxpr:
        invars: operands, flattened
        outvars: one per operand, in the same order
        params: placement metadata for ``device_put`` and the tag for ``name``, unused here

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.optimization_barrier.html
    """
    for invar, outvar in zip(eqn.invars, eqn.outvars, strict=True):
        state.indices[outvar] = _index_sets(state, invar)
    _forward_across_jaxpr_boundary(state, eqn.invars, eqn.outvars)
