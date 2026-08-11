"""Propagation rules for convolution operations."""

import numpy as np
from jax._src.core import JaxprEqn

from ._common import (
    IndexSet,
    _atom_shape,
    _conservative_indices,
    _index_sets,
    _numel,
    _PropState,
    _row_strides,
    _union_all,
)


def _window_target_lists(
    out_spatial_sizes: list[int],
    kernel_spatial_sizes: list[int],
    lhs_spatial_sizes: list[int],
    window_strides: tuple[int, ...],
    lhs_dilation: tuple[int, ...],
    rhs_dilation: tuple[int, ...],
    padding: tuple[tuple[int, int], ...],
) -> list[list[int]]:
    """Map each output spatial position to the input spatial positions in its window.

    Vectorized over (output position, kernel tap):
    for output position ``o``, kernel tap ``k``, and spatial dimension ``i``,
    the position in the lhs-dilated input is
    ``o[i] * stride[i] + k[i] * rhs_dilation[i] - padding_lo[i]``.
    A tap is valid when that position is in bounds
    and lands on an actual input element rather than in a dilation gap.

    Returns one list of flat input spatial positions per flat output spatial position,
    both row-major over the respective spatial sizes.
    """
    n_spatial = len(out_spatial_sizes)
    out_spatial_size = _numel(out_spatial_sizes)
    kernel_size = _numel(kernel_spatial_sizes)

    def per_dim(values) -> np.ndarray:
        return np.asarray(list(values), dtype=np.intp).reshape(n_spatial, 1, 1)

    out_coords = np.indices(out_spatial_sizes).reshape(n_spatial, out_spatial_size)
    tap_coords = np.indices(kernel_spatial_sizes).reshape(n_spatial, kernel_size)

    pos = (
        out_coords[:, :, None] * per_dim(window_strides)
        + tap_coords[:, None, :] * per_dim(rhs_dilation)
        - per_dim(lo for lo, _ in padding)
    )
    in_coords = pos // per_dim(lhs_dilation)
    valid = (
        (pos >= 0)
        & (in_coords < per_dim(lhs_spatial_sizes))
        & (pos % per_dim(lhs_dilation) == 0)
    ).all(axis=0)

    targets = (in_coords * per_dim(_row_strides(lhs_spatial_sizes))).sum(axis=0)
    return [targets[o, valid[o]].tolist() for o in range(out_spatial_size)]


def _prop_conv_general_dilated(eqn: JaxprEqn, state: _PropState) -> None:
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
        out[0] = a·w0 + b·w1  →  index set {0, 1}
        out[1] = b·w0 + c·w1  →  index set {1, 2}
        out[2] = c·w0 + d·w1  →  index set {2, 3}

    Because set union is associative and commutative,
    the per-output union factors instead of being recomputed per element:
    the group's channels are pre-unioned once per input spatial position,
    those sets are unioned once per output spatial position,
    and every output channel of the same group aliases the resulting set.
    This costs O(input size + output windows) set unions
    instead of O(output size * window size * channels).

    Jaxpr:
        invars[0]: lhs — rank n+2 input array
        invars[1]: rhs — rank n+2 kernel weights
        dimension_numbers: ConvDimensionNumbers (batch, feature, spatial dims)
        window_strides, padding, lhs_dilation, rhs_dilation: conv parameters
        feature_group_count, batch_group_count: grouping parameters

    https://docs.jax.dev/en/latest/_autosummary/jax.lax.conv_general_dilated.html
    """
    lhs_indices = _index_sets(state, eqn.invars[0])  # Input image dependencies
    rhs_indices = _index_sets(state, eqn.invars[1])  # Kernel dependencies

    out_shape = _atom_shape(eqn.outvars[0])
    out_size = _numel(out_shape)

    # Convolution is bilinear in (data, kernel),
    # so an input-dependent kernel (e.g. a hypernetwork) is valid user code.
    # Without precise bilinear tracking, fall back to conservative:
    # every output depends on the data and kernel dependencies.
    # TODO: track the precise pattern (data window plus full kernel per output).
    if any(rhs_indices):
        state.indices[eqn.outvars[0]] = _conservative_indices(
            lhs_indices + rhs_indices, out_size
        )
        return

    # Neither operand carries dependencies, so every output set is empty.
    if not any(lhs_indices):
        state.indices[eqn.outvars[0]] = _conservative_indices([], out_size)
        return

    if out_size == 0:
        state.indices[eqn.outvars[0]] = []
        return

    batch_group_count = eqn.params.get("batch_group_count", 1)

    # Get shapes from avals
    lhs_shape = _atom_shape(eqn.invars[0])
    rhs_shape = _atom_shape(eqn.invars[1])

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

    lhs_strides = _row_strides(lhs_shape)

    # Get spatial sizes
    lhs_spatial_sizes = [lhs_shape[d] for d in lhs_spatial_dims]
    kernel_spatial_sizes = [rhs_shape[d] for d in rhs_spatial_dims]
    out_spatial_sizes = [out_shape[d] for d in out_spatial_dims]
    n_in_features = lhs_shape[lhs_feature_dim]
    n_out_features = out_shape[out_feature_dim]
    n_lhs_batches = lhs_shape[lhs_batch_dim]

    # Compute per-group channel ranges.
    # When feature_group_count == 1, this covers all input channels.
    group_size_in = n_in_features // feature_group_count
    group_size_out = n_out_features // feature_group_count

    # With batch_group_count > 1, output features are split into G groups
    # and each group reads from a shifted input batch:
    # in_batch = out_batch + group * n_out_batches.
    n_out_batches = n_lhs_batches // batch_group_count
    channels_per_batch_group = n_out_features // batch_group_count

    window_targets = _window_target_lists(
        out_spatial_sizes,
        kernel_spatial_sizes,
        lhs_spatial_sizes,
        window_strides,
        lhs_dilation,
        rhs_dilation,
        padding,
    )

    # Flat lhs offsets of the input spatial positions, in spatial row-major order.
    in_spatial_size = _numel(lhs_spatial_sizes)
    in_coords = np.indices(lhs_spatial_sizes).reshape(n_spatial, in_spatial_size)
    spatial_strides = np.asarray(
        [lhs_strides[d] for d in lhs_spatial_dims], dtype=np.intp
    ).reshape(n_spatial, 1)
    spatial_offsets = (in_coords * spatial_strides).sum(axis=0)

    def window_union_table(in_batch: int, group: int) -> list[IndexSet]:
        """Per-output-spatial-position sets for one (input batch, feature group)."""
        base = in_batch * lhs_strides[lhs_batch_dim]
        channels = range(group * group_size_in, (group + 1) * group_size_in)
        chan_offsets = base + np.asarray(channels) * lhs_strides[lhs_feature_dim]
        flat = chan_offsets[:, None] + spatial_offsets[None, :]
        # Positions are in bounds by construction;
        # a violation here means a coordinate-math bug.
        assert flat.size == 0 or flat.max() < len(lhs_indices)

        # Pre-union the group's channels once per input spatial position.
        # A single-channel group aliases the input sets instead of copying them.
        cols = flat.T.tolist()
        if group_size_in == 1:
            channel_sets = [lhs_indices[col[0]] for col in cols]
        else:
            channel_sets = [_union_all([lhs_indices[f] for f in col]) for col in cols]

        # Union each window's taps.
        # A single-tap window aliases the channel set instead of copying it.
        out_sets: list[IndexSet] = []
        for targets in window_targets:
            if len(targets) == 1:
                out_sets.append(channel_sets[targets[0]])
            else:
                out_sets.append(_union_all([channel_sets[t] for t in targets]))
        return out_sets

    tables = [
        window_union_table(in_batch, group)
        for in_batch in range(n_lhs_batches)
        for group in range(feature_group_count)
    ]

    # Assemble the output by table lookup.
    # All output channels of the same (input batch, feature group)
    # alias the same set objects.
    out_coords = np.indices(out_shape).reshape(len(out_shape), out_size)
    out_batch_idx = out_coords[out_batch_dim]
    out_feature_idx = out_coords[out_feature_dim]
    out_spatial_strides = np.asarray(
        _row_strides(out_spatial_sizes), dtype=np.intp
    ).reshape(n_spatial, 1)
    spatial_idx = (out_coords[list(out_spatial_dims)] * out_spatial_strides).sum(axis=0)

    feature_group_idx = out_feature_idx // group_size_out
    batch_group_idx = out_feature_idx // channels_per_batch_group
    in_batch_idx = out_batch_idx + batch_group_idx * n_out_batches

    table_idx = in_batch_idx * feature_group_count + feature_group_idx
    state.indices[eqn.outvars[0]] = [
        tables[t][s]
        for t, s in zip(table_idx.tolist(), spatial_idx.tolist(), strict=True)
    ]
