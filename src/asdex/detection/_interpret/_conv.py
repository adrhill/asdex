"""Propagation rules for convolution operations."""

from itertools import product

from jax._src.core import JaxprEqn

from ._commons import (
    IndexSet,
    StateIndices,
    atom_shape,
    check_no_index_sets,
    empty_index_set,
    flat_to_coords,
    index_sets,
    numel,
    row_strides,
)


def prop_conv_general_dilated(eqn: JaxprEqn, state_indices: StateIndices) -> None:
    """Convolution slides a kernel over the input, computing weighted sums.

    Each output element depends on a local spatial window of input elements
    across the input channels in the corresponding feature group.
    When ``feature_group_count == 1`` (the common case),
    every output channel depends on all input channels.
    For grouped or depthwise convolutions (``feature_group_count > 1``),
    each output channel group only depends on
    the corresponding input channel group.

    When ``batch_group_count > 1`` (mainly in JAX backprop internals),
    each output batch aggregates multiple input batches
    within the same batch group.

    For 2D conv with kernel size (kH, kW), stride s, and C_in input channels:
        out[n, h, w, c_out] = Σ_{kh, kw, c_in} in[n, h·s + kh, w·s + kw, c_in] · W[...]
    So out[n, h, w, :] depends on in[n, h·s : h·s+kH, w·s : w·s+kW, :].

    Example: 1D conv, kernel size 2, input [a, b, c, d]
        out[0] = a·w0 + b·w1  →  state_indices {0, 1}
        out[1] = b·w0 + c·w1  →  state_indices {1, 2}
        out[2] = c·w0 + d·w1  →  state_indices {2, 3}

    Jaxpr:
        invars[0]: lhs — rank n+2 input array
        invars[1]: rhs — rank n+2 kernel weights
        dimension_numbers: ConvDimensionNumbers (batch, feature, spatial dims)
        window_strides, padding, lhs_dilation, rhs_dilation: conv parameters
        feature_group_count, batch_group_count: grouping parameters

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.conv_general_dilated.html
    """
    lhs_indices = index_sets(state_indices, eqn.invars[0])  # Input image dependencies
    # TODO: include kernel (rhs) index sets in output dependencies.
    check_no_index_sets(state_indices, eqn.invars[1], eqn.primitive.name)

    out_shape = atom_shape(eqn.outvars[0])
    out_size = numel(out_shape)

    batch_group_count = eqn.params.get("batch_group_count", 1)

    # Get shapes from avals
    lhs_shape = atom_shape(eqn.invars[0])
    rhs_shape = atom_shape(eqn.invars[1])

    # Parse dimension numbers
    dim_nums = eqn.params["dimension_numbers"]
    lhs_spec, rhs_spec, out_spec = (
        dim_nums.lhs_spec,
        dim_nums.rhs_spec,
        dim_nums.out_spec,
    )

    # Extract dimension indices
    lhs_batch_dim, lhs_feature_dim = lhs_spec[0], lhs_spec[1]
    lhs_spatial_dims = lhs_spec[2:]
    out_batch_dim, out_feature_dim = out_spec[0], out_spec[1]
    out_spatial_dims = out_spec[2:]
    rhs_spatial_dims = rhs_spec[2:]

    # Get parameters
    n_spatial = len(lhs_spatial_dims)
    window_strides = eqn.params.get("window_strides", (1,) * n_spatial)
    lhs_dilation = eqn.params.get("lhs_dilation", (1,) * n_spatial)
    rhs_dilation = eqn.params.get("rhs_dilation", (1,) * n_spatial)
    padding = eqn.params.get("padding", ((0, 0),) * n_spatial)
    feature_group_count = eqn.params.get("feature_group_count", 1)
    # JAX requires at most one of these to be > 1 at a time.

    lhs_strides = row_strides(lhs_shape)
    out_strides = row_strides(out_shape)

    # Get spatial sizes
    lhs_spatial_sizes = [lhs_shape[d] for d in lhs_spatial_dims]
    kernel_spatial_sizes = [rhs_shape[d] for d in rhs_spatial_dims]
    n_in_features = lhs_shape[lhs_feature_dim]
    n_out_features = out_shape[out_feature_dim]

    # Compute per-group channel ranges.
    # When feature_group_count == 1, this covers all input channels.
    group_size_in = n_in_features // feature_group_count
    group_size_out = n_out_features // feature_group_count

    # With batch_group_count > 1, output features are split into G groups
    # and each group reads from a shifted input batch:
    # in_batch = out_batch + group * n_out_batches.
    n_out_batches = lhs_shape[lhs_batch_dim] // batch_group_count
    channels_per_batch_group = n_out_features // batch_group_count

    out_indices: list[IndexSet] = []

    for out_flat in range(out_size):
        out_coord = flat_to_coords(out_flat, out_strides)

        out_batch_idx = out_coord[out_batch_dim]
        out_feature_idx = out_coord[out_feature_dim]
        out_spatial_coord = [out_coord[d] for d in out_spatial_dims]

        # Only iterate over input channels in the same feature group.
        feature_group_idx = out_feature_idx // group_size_out
        in_feature_start = feature_group_idx * group_size_in
        in_feature_end = in_feature_start + group_size_in

        batch_group = out_feature_idx // channels_per_batch_group
        in_batch_idx = out_batch_idx + batch_group * n_out_batches

        # Collect dependencies from input
        elem_deps: IndexSet = empty_index_set()

        # For each position in the kernel window
        for kernel_offsets in product(*[range(k) for k in kernel_spatial_sizes]):
            # Compute input spatial coordinates
            in_spatial_coord = []
            valid = True
            for i in range(n_spatial):
                in_c = (
                    out_spatial_coord[i] * window_strides[i]
                    + kernel_offsets[i] * rhs_dilation[i]
                    - padding[i][0]
                )
                if in_c < 0 or in_c >= lhs_spatial_sizes[i] * lhs_dilation[i]:
                    valid = False
                    break
                if lhs_dilation[i] > 1 and in_c % lhs_dilation[i] != 0:
                    valid = False
                    break
                in_spatial_coord.append(in_c // lhs_dilation[i])

            if not valid:
                continue

            # For each input feature channel in this group
            for in_feature_idx in range(in_feature_start, in_feature_end):
                in_coord = [0] * len(lhs_shape)
                in_coord[lhs_batch_dim] = in_batch_idx
                in_coord[lhs_feature_dim] = in_feature_idx
                for i, d in enumerate(lhs_spatial_dims):
                    in_coord[d] = in_spatial_coord[i]

                in_flat = sum(c * s for c, s in zip(in_coord, lhs_strides, strict=True))
                if in_flat < len(lhs_indices):
                    elem_deps |= lhs_indices[in_flat]

        out_indices.append(elem_deps)

    state_indices[eqn.outvars[0]] = out_indices
