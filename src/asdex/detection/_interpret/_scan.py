"""Propagation rule for scan."""

from jax._src.core import JaxprEqn

from ._common import (
    IndexSet,
    PropJaxprFn,
    _forward_across_jaxpr_boundary,
    _index_sets,
    _PropState,
    _seed_const_vals,
)


def _prop_scan(
    eqn: JaxprEqn,
    state: _PropState,
    _prop_jaxpr: PropJaxprFn,
) -> None:
    """Scan applies a body jaxpr iteratively, threading carry across iterations.

    Unlike ``while_loop`` (unknown iteration count, same inputs each iteration),
    scan has a known ``length`` and different ``xs[t]`` per timestep.
    Dependencies are propagated via forward simulation:
    one ``_prop_jaxpr`` call per timestep, threading carry deps forward.

    When no xs slice carries input dependencies,
    every timestep sees identical inputs apart from the carry,
    so once the carry index sets repeat between consecutive steps
    all remaining steps reproduce the same carry and ys slices.
    The simulation then stops early and replicates the last ys slice,
    which keeps e.g. a solver loop with ``length=100_000``
    at a handful of body propagations without losing exactness.

    Layout:
        invars:  [consts..., carry_init..., xs...]
        outvars: [carry_final..., ys...]
        body jaxpr invars:  [consts..., carry..., x_slice...]
        body jaxpr outvars: [carry_new..., y_slice...]
        params: jaxpr, ft_in, ft_out, length, reverse, unroll

    ``ft_in`` is a ``jax._src.flattree.FTTuple`` splitting the invars into
    ``(consts, carry, xs)`` groups; its per-group lengths give the
    ``num_consts`` / ``num_carry`` counts.

    xs arrays have an extra leading dimension of size ``length``
    compared to their body counterparts x_slice.
    Similarly for ys vs y_slice.

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.scan.html
    """
    body_closed = eqn.params["jaxpr"]
    body_jaxpr = body_closed.jaxpr
    num_consts, num_carry, _num_xs = (
        len(group) for group in eqn.params["ft_in"].unpack()
    )
    length = eqn.params["length"]
    reverse = eqn.params["reverse"]

    # Split invars: [consts | carry_init | xs]
    consts = eqn.invars[:num_consts]
    carry_init = eqn.invars[num_consts : num_consts + num_carry]
    xs = eqn.invars[num_consts + num_carry :]

    # Split outvars: [carry_final | ys]
    carry_final = eqn.outvars[:num_carry]
    ys = eqn.outvars[num_carry:]

    _seed_const_vals(state, body_jaxpr.constvars, body_closed.consts)
    _forward_across_jaxpr_boundary(state, consts, body_jaxpr.invars[:num_consts])

    # Prepare const index sets for the body
    const_inputs: list[list[IndexSet]] = [_index_sets(state, v) for v in consts]

    # Initialize carry from carry_init
    carry_indices: list[list[IndexSet]] = [_index_sets(state, v) for v in carry_init]

    # Pre-compute xs index sets and per-slice sizes.
    # xs arrays carry a leading dim of size ``length``,
    # so each per-timestep slice has ``numel // length`` elements.
    xs_all_indices: list[list[IndexSet]] = [_index_sets(state, v) for v in xs]
    xs_slice_numels: list[int] = [len(ind) // length for ind in xs_all_indices]

    # The saturation early exit is only sound when every timestep
    # sees the same xs index sets,
    # which holds in particular when no xs slice carries dependencies at all
    # (xs are constants, or the scan has no xs).
    # Body propagation never reads xs values, only their index sets,
    # so it is then a deterministic function of the carry alone.
    xs_stationary = all(not any(sets) for sets in xs_all_indices)

    # Forward simulation: one _prop_jaxpr call per timestep,
    # threading carry forward and collecting per-timestep ys.
    num_ys = len(ys)
    ys_per_step: list[list[list[IndexSet]]] = [[] for _ in range(num_ys)]

    steps_run = 0
    time_range = range(length - 1, -1, -1) if reverse else range(length)
    for t in time_range:
        # Extract xs slice for this timestep
        xs_slice_inputs: list[list[IndexSet]] = []
        for i in range(len(xs)):
            sn = xs_slice_numels[i]
            xs_slice_inputs.append(xs_all_indices[i][t * sn : (t + 1) * sn])

        body_output = _prop_jaxpr(
            body_jaxpr,
            const_inputs + carry_indices + xs_slice_inputs,
            state,
        )

        new_carry = body_output[:num_carry]

        # Collect per-timestep ys slices (in iteration order, not time order)
        y_slice_outputs = body_output[num_carry:]
        for i in range(num_ys):
            ys_per_step[i].append(y_slice_outputs[i])
        steps_run += 1

        # Thread carry forward, stopping once it saturates
        saturated = xs_stationary and _carry_saturated(new_carry, carry_indices)
        carry_indices = new_carry
        if saturated:
            break

    # Replicate the last ys slice for the steps skipped after saturation.
    # Aliasing the same slice is safe because handlers never mutate index sets.
    if steps_run < length:
        for i in range(num_ys):
            ys_per_step[i].extend([ys_per_step[i][-1]] * (length - steps_run))

    # Write carry_final
    for outvar, out_indices in zip(carry_final, carry_indices, strict=True):
        state.indices[outvar] = out_indices

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
        state.indices[outvar] = full_indices


def _carry_saturated(
    new_carry: list[list[IndexSet]],
    prev_carry: list[list[IndexSet]],
) -> bool:
    """Check whether the carry index sets are unchanged between consecutive steps.

    Identity is checked before equality
    because pass-through bodies alias the very same set objects.
    """
    return all(
        new_set is prev_set or new_set == prev_set
        for new_sets, prev_sets in zip(new_carry, prev_carry, strict=True)
        for new_set, prev_set in zip(new_sets, prev_sets, strict=True)
    )
