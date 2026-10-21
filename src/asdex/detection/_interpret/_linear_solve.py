"""Propagation rule for custom_linear_solve."""

from jax._src.core import JaxprEqn

from ._common import (
    PropJaxprFn,
    _forward_across_jaxpr_boundary,
    _index_sets,
    _PropState,
    _seed_const_vals,
)


def _prop_custom_linear_solve(
    eqn: JaxprEqn,
    state: _PropState,
    _prop_jaxpr: PropJaxprFn,
) -> None:
    """custom_linear_solve returns ``x = solve(solve_consts, b)``.

    The primitive stores four jaxprs (``matvec``, ``vecmat``, ``solve``,
    ``transpose_solve``), each with its own leading constants in ``invars``.
    Only the ``solve`` jaxpr computes the primal output,
    so it is propagated with its own constants and the right-hand side ``b``.
    ``matvec`` and ``vecmat`` describe the operator and its transpose,
    while ``transpose_solve`` computes the output of the new equation
    that JAX's transpose rule creates.

    The right-hand side is therefore as precise as the solve allows.
    Operator dependencies are whatever the ``solve`` jaxpr uses;
    for ``jnp.linalg.solve`` that is ``lu``,
    whose conservative fallback makes the matrix operand dense.

    Example: the solve jaxpr computes ``rhs / scale``.
        solve constants (scale): [{0}, {1}]
        b:                       [{2}, {3}]
        output:                  [{0, 2}, {1, 3}]

    Layout:
        invars: [matvec_consts..., vecmat_consts..., solve_consts...,
                 transpose_solve_consts..., b...]
        outvars: [solve outputs...]
        params: const_lengths, jaxprs

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.custom_linear_solve.html
    """
    const_lengths = eqn.params["const_lengths"]
    closed = eqn.params["jaxprs"].solve

    # Offset of the solve and transpose_solve constant groups.
    matvec_and_vecmat_length = const_lengths.matvec + const_lengths.vecmat
    solve_start = matvec_and_vecmat_length
    solve_end = solve_start + const_lengths.solve
    b_start = solve_end + const_lengths.transpose_solve
    solve_invars = eqn.invars[solve_start:solve_end] + eqn.invars[b_start:]

    if hasattr(closed, "jaxpr"):
        _seed_const_vals(state, closed.jaxpr.constvars, closed.consts)
        closed = closed.jaxpr

    _forward_across_jaxpr_boundary(state, solve_invars, closed.invars)
    input_indices = [_index_sets(state, invar) for invar in solve_invars]
    output_indices = _prop_jaxpr(closed, input_indices, state)

    _forward_across_jaxpr_boundary(state, closed.outvars, eqn.outvars)
    for outvar, indices in zip(eqn.outvars, output_indices, strict=True):
        state.indices[outvar] = indices
