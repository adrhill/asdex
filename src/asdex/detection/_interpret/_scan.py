"""Propagation rule for scan."""

from jax._src.core import JaxprEqn

from ._commons import (
    IndexSet,
    PropJaxprFn,
    StateConsts,
    StateIndices,
    atom_shape,
    forward_const_vals,
    index_sets,
    seed_const_vals,
)


def prop_scan(
    eqn: JaxprEqn,
    state_indices: StateIndices,
    state_consts: StateConsts,
    prop_jaxpr: PropJaxprFn,
) -> None:
    """Scan applies a body jaxpr iteratively, threading carry across iterations.

    Unlike ``while_loop`` (unknown iteration count, same inputs each iteration),
    scan has a known ``length`` and different ``xs[t]`` per timestep.
    Dependencies are propagated via forward simulation:
    one ``prop_jaxpr`` call per timestep, threading carry deps forward.

    Layout:
        invars:  [consts..., carry_init..., xs...]
        outvars: [carry_final..., ys...]
        body jaxpr invars:  [consts..., carry..., x_slice...]
        body jaxpr outvars: [carry_new..., y_slice...]
        params: jaxpr, num_consts, num_carry, length, reverse, linear, unroll

    xs arrays have an extra leading dimension of size ``length``
    compared to their body counterparts x_slice.
    Similarly for ys vs y_slice.

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html
    """
    body_closed = eqn.params["jaxpr"]
    body_jaxpr = body_closed.jaxpr
    num_consts = eqn.params["num_consts"]
    num_carry = eqn.params["num_carry"]
    length = eqn.params["length"]
    reverse = eqn.params["reverse"]

    # Split invars: [consts | carry_init | xs]
    consts = eqn.invars[:num_consts]
    carry_init = eqn.invars[num_consts : num_consts + num_carry]
    xs = eqn.invars[num_consts + num_carry :]

    # Split outvars: [carry_final | ys]
    carry_final = eqn.outvars[:num_carry]
    ys = eqn.outvars[num_carry:]

    seed_const_vals(state_consts, body_jaxpr.constvars, body_closed.consts)
    forward_const_vals(state_consts, consts, body_jaxpr.invars[:num_consts])

    # Prepare const index sets for the body
    const_inputs: list[list[IndexSet]] = [index_sets(state_indices, v) for v in consts]

    # Initialize carry from carry_init
    carry_indices: list[list[IndexSet]] = [
        index_sets(state_indices, v) for v in carry_init
    ]

    # Pre-compute xs index sets and per-slice sizes
    xs_all_indices: list[list[IndexSet]] = [index_sets(state_indices, v) for v in xs]
    xs_slice_numels: list[int] = []
    for i, x_var in enumerate(xs):
        x_shape = atom_shape(x_var)
        if len(x_shape) == 0:
            raise AssertionError("scan xs must have a leading length dim")
        xs_slice_numels.append(len(xs_all_indices[i]) // x_shape[0])

    # Determine iteration length from xs or params
    iter_length: int = atom_shape(xs[0])[0] if xs else length

    # Validate ys shapes
    for y_var in ys:
        if len(atom_shape(y_var)) == 0:
            raise AssertionError("scan ys must have a leading length dim")

    # Forward simulation: one prop_jaxpr call per timestep,
    # threading carry forward and collecting per-timestep ys.
    num_ys = len(ys)
    ys_per_step: list[list[list[IndexSet]]] = [[] for _ in range(num_ys)]

    time_range = range(iter_length - 1, -1, -1) if reverse else range(iter_length)
    for t in time_range:
        # Extract xs slice for this timestep
        xs_slice_inputs: list[list[IndexSet]] = []
        for i in range(len(xs)):
            sn = xs_slice_numels[i]
            xs_slice_inputs.append(xs_all_indices[i][t * sn : (t + 1) * sn])

        body_output = prop_jaxpr(
            body_jaxpr, const_inputs + carry_indices + xs_slice_inputs, state_consts
        )

        # Thread carry forward
        carry_indices = body_output[:num_carry]

        # Collect per-timestep ys slices (in iteration order, not time order)
        y_slice_outputs = body_output[num_carry:]
        for i in range(num_ys):
            ys_per_step[i].append(y_slice_outputs[i])

    # Write carry_final
    for outvar, out_indices in zip(carry_final, carry_indices, strict=True):
        state_indices[outvar] = out_indices

    # Write ys by concatenating per-timestep slices in time order.
    # When reverse=True, iteration order is [n-1, n-2, ..., 0],
    # so we reverse to get time order [0, 1, ..., n-1].
    for i, outvar in enumerate(ys):
        slices = ys_per_step[i]
        if reverse:
            slices = slices[::-1]
        full_indices: list[IndexSet] = []
        for s in slices:
            full_indices.extend(s)
        state_indices[outvar] = full_indices
