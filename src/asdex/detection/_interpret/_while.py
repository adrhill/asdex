"""Propagation rule for while_loop."""

from collections.abc import Callable

from jax._src.core import JaxprEqn

from ._common import (
    IndexSet,
    PropJaxprFn,
    StateBounds,
    StateConsts,
    StateIndices,
    _copy_index_sets,
    _forward_const_vals,
    _forward_value_bounds,
    _index_sets,
    _seed_const_vals,
)

_MAX_FIXED_POINT_ITERS = 500
"""Safety bound for fixed-point iteration."""


def _prop_while(
    eqn: JaxprEqn,
    state_indices: StateIndices,
    state_consts: StateConsts,
    state_bounds: StateBounds,
    _prop_jaxpr: PropJaxprFn,
) -> None:
    """while_loop iterates a body until a condition becomes false.

    The carry variables may accumulate dependencies across iterations,
    so we iterate propagation to a fixed point.

    The cond jaxpr only produces a boolean and doesn't contribute to carry state_indices.

    Layout:
        invars: [cond_consts..., body_consts..., carry_init...]
        outvars: [carry_final...]
        params: body_jaxpr, body_nconsts, cond_jaxpr, cond_nconsts

    Example: carry = carry + const (accumulation)
        Input state_indices:  carry=[{0}, {1}], const=[{}, {}]
        After 1 iter: carry=[{0}, {1}] (stable immediately since const state_indices are empty)

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.while_loop.html
    """
    body_closed = eqn.params["body_jaxpr"]
    body_jaxpr = body_closed.jaxpr
    body_nconsts = eqn.params["body_nconsts"]
    cond_nconsts = eqn.params["cond_nconsts"]

    # Split invars: [cond_consts | body_consts | carry_init]
    n_carry = len(eqn.outvars)
    body_consts = eqn.invars[cond_nconsts : cond_nconsts + body_nconsts]
    carry_init = eqn.invars[cond_nconsts + body_nconsts :]
    assert len(carry_init) == n_carry

    _seed_const_vals(state_consts, body_jaxpr.constvars, body_closed.consts)
    # Only forward state_consts for body consts, not carry (carry changes each iteration)
    _forward_const_vals(state_consts, body_consts, body_jaxpr.invars[:body_nconsts])
    _forward_value_bounds(state_bounds, body_consts, body_jaxpr.invars[:body_nconsts])

    # Initialize carry state_indices from the initial values
    carry_indices: list[list[IndexSet]] = [
        _index_sets(state_indices, v) for v in carry_init
    ]

    # body_jaxpr invars: [body_consts..., carry...]
    const_inputs: list[list[IndexSet]] = [
        _index_sets(state_indices, v) for v in body_consts
    ]

    def iterate(carry: list[list[IndexSet]]) -> list[list[IndexSet]]:
        return _prop_jaxpr(body_jaxpr, const_inputs + carry, state_consts, state_bounds)

    _fixed_point_loop(iterate, carry_indices, n_carry)

    # Write final carry state_indices to outvars
    for outvar, out_indices in zip(eqn.outvars, carry_indices, strict=True):
        state_indices[outvar] = out_indices


def _fixed_point_loop(
    iterate_fn: Callable[[list[list[IndexSet]]], list[list[IndexSet]]],
    carry: list[list[IndexSet]],
    n_carry: int,
) -> None:
    """Run ``iterate_fn`` on carry index sets until they stabilize.

    Since index sets only grow and are bounded in size
    (i.e., monotone on a finite lattice),
    this always converges.

    Mutates ``carry`` in place.
    """
    # Carry sets may alias (shared objects from upstream handlers),
    # so copy them before in-place mutation via |=.
    for i in range(n_carry):
        carry[i] = _copy_index_sets(carry[i])

    for _iteration in range(_MAX_FIXED_POINT_ITERS):
        body_output = iterate_fn(carry)

        changed = False
        for i in range(n_carry):
            for j in range(len(carry[i])):
                before = len(carry[i][j])
                carry[i][j] |= body_output[i][j]
                if len(carry[i][j]) > before:
                    changed = True

        if not changed:
            return
    else:  # pragma: no cover
        msg = (
            f"Fixed-point iteration did not converge after "
            f"{_MAX_FIXED_POINT_ITERS} iterations. "
            "Please help out asdex's development by reporting this at "
            "https://github.com/adrhill/asdex/issues"
        )
        raise RuntimeError(msg)
