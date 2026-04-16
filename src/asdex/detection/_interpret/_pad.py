"""Propagation rule for pad operations."""

from jax._src.core import JaxprEqn

from ._commons import (
    IndexSet,
    StateIndices,
    atom_shape,
    empty_index_set,
    flat_to_coords,
    index_sets,
    numel,
    row_strides,
)


def prop_pad(eqn: JaxprEqn, state_indices: StateIndices) -> None:
    """Padding inserts constant-valued elements around an array.

    Each output element either maps back to exactly one input element
    (preserving its dependencies) or is a padding position
    (inheriting the padding value's dependencies, usually empty).

    For padding_config (low, high, interior) per dimension:
        out[i] maps to input[(i - low) / (interior + 1)]
        when (i - low) >= 0, (i - low) % (interior + 1) == 0,
        and the resulting index is in bounds.

    The Jacobian is a selection matrix with at most one 1 per row.

    Example: x = [a, b, c], pad(x, (1, 1), constant=0)
        Input state_indices:  [{0}, {1}, {2}]
        Output state_indices: [{}, {0}, {1}, {2}, {}]

    Jaxpr:
        invars[0]: input array
        invars[1]: padding value (scalar)
        padding_config: tuple of (low, high, interior) per dimension

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.pad.html
    """
    in_indices = index_sets(state_indices, eqn.invars[0])
    pad_indices = index_sets(state_indices, eqn.invars[1])

    in_shape = atom_shape(eqn.invars[0])
    padding_config = eqn.params["padding_config"]
    ndim = len(in_shape)

    # Compute output shape from padding config.
    out_shape = tuple(
        low + high + max(in_shape[d] + (in_shape[d] - 1) * interior, 0)
        if in_shape[d] > 0
        else low + high
        for d, (low, high, interior) in enumerate(padding_config)
    )

    in_strides = row_strides(in_shape)
    out_strides = row_strides(out_shape)
    out_size = numel(out_shape)

    # The padding value is a scalar; use its first (only) dep set.
    pad_dep = pad_indices[0] if pad_indices else empty_index_set()

    out_indices: list[IndexSet] = []
    for out_flat in range(out_size):
        out_coord = flat_to_coords(out_flat, out_strides)

        # Reverse-map each output coordinate to an input coordinate.
        in_flat = 0
        is_pad = False
        for d in range(ndim):
            low, _, interior = padding_config[d]
            pos = out_coord[d] - low

            if pos < 0 or pos >= in_shape[d] + (in_shape[d] - 1) * interior:
                # Outside the input region (low/high padding).
                is_pad = True
                break

            stride = interior + 1
            if interior > 0 and pos % stride != 0:
                # Falls on an interior padding slot.
                is_pad = True
                break

            in_idx = pos // stride if interior > 0 else pos
            in_flat += in_idx * in_strides[d]

        if is_pad:
            out_indices.append(pad_dep)
        else:
            out_indices.append(in_indices[in_flat])

    state_indices[eqn.outvars[0]] = out_indices
