"""Test asdex on convolutional layers.

References:
    https://docs.jax.dev/en/latest/_autosummary/jax.lax.conv_general_dilated.html
"""

import jax
import jax.lax as lax
import jax.numpy as jnp
import numpy as np
import pytest
from flax import nnx

from asdex import jacobian_sparsity


def check_conv_sparsity(
    input_hw: tuple[int, int],
    kernel_size: tuple[int, int],
    channels: tuple[int, int],
    *,
    verify_exact: bool = True,
) -> None:
    """Test conv sparsity detection for given parameters.

    Args:
        input_hw: (height, width) of input image
        kernel_size: (kH, kW) kernel dimensions
        channels: (C_in, C_out) input and output channels
        verify_exact: If True, verify exact match with actual Jacobian.
                      Set False for large inputs where Jacobian is expensive.
    """
    H, W = input_hw
    C_in, C_out = channels
    input_size = H * W * C_in

    conv = nnx.Conv(C_in, C_out, kernel_size, padding="VALID", rngs=nnx.Rngs(0))

    def f(x_flat):
        x = x_flat.reshape(1, H, W, C_in)
        y = conv(x)
        return y.flatten()

    sparsity = jacobian_sparsity(f, np.zeros(input_size))

    # Expected output shape with VALID padding
    out_h = H - kernel_size[0] + 1
    out_w = W - kernel_size[1] + 1
    expected_out_size = out_h * out_w * C_out

    assert sparsity.shape == (expected_out_size, input_size)
    assert sparsity.nnz > 0
    # Conservative audit: detected sparsity must be strictly sparser than dense.
    assert sparsity.nnz < expected_out_size * input_size

    if verify_exact:
        x_test = jax.random.normal(jax.random.key(42), (input_size,))
        actual_jac = jax.jacobian(f)(x_test)
        actual_nonzero = (np.abs(actual_jac) > 1e-10).astype(int)
        detected = sparsity.todense().astype(int)

        np.testing.assert_array_equal(
            detected, actual_nonzero, "Detected sparsity should match actual Jacobian"
        )


@pytest.mark.parametrize(
    ("input_hw", "kernel_size", "channels"),
    [
        ((4, 4), (2, 2), (1, 1)),
        ((4, 4), (3, 3), (1, 1)),
        ((6, 6), (2, 2), (1, 1)),
        ((6, 6), (3, 3), (2, 3)),
        ((8, 8), (3, 3), (3, 2)),
        ((8, 8), (5, 5), (1, 1)),
    ],
)
def test_conv_exact(input_hw, kernel_size, channels):
    """Verify exact sparsity match for small convolutions."""
    check_conv_sparsity(input_hw, kernel_size, channels, verify_exact=True)


def test_conv_sct_example():
    """28x28 input, 3x3 kernel, 3 in channels, 2 out channels."""
    check_conv_sparsity((28, 28), (3, 3), (3, 2), verify_exact=False)


def test_conv_with_padding():
    """Convolution with SAME padding exercises boundary coordinate checks."""
    H, W = 4, 4
    C_in, C_out = 1, 1
    input_size = H * W * C_in

    # SAME padding adds zeros around borders, so kernel windows at edges
    # will have coordinates that fall outside the valid input region
    conv = nnx.Conv(C_in, C_out, (3, 3), padding="SAME", rngs=nnx.Rngs(0))

    def f(x_flat):
        x = x_flat.reshape(1, H, W, C_in)
        y = conv(x)
        return y.flatten()

    sparsity = jacobian_sparsity(f, np.zeros(input_size))

    # With SAME padding, output size equals input size
    expected_out_size = H * W * C_out
    assert sparsity.shape == (expected_out_size, input_size)
    assert sparsity.nnz > 0

    # Verify against actual Jacobian
    x_test = jax.random.normal(jax.random.key(42), (input_size,))
    actual_jac = jax.jacobian(f)(x_test)
    actual_nonzero = (np.abs(actual_jac) > 1e-10).astype(int)
    detected = sparsity.todense().astype(int)

    np.testing.assert_array_equal(
        detected, actual_nonzero, "Detected sparsity should match actual Jacobian"
    )


def test_conv_with_strides():
    """Convolution with strides > 1."""
    H, W = 6, 6
    C_in, C_out = 1, 1
    input_size = H * W * C_in

    conv = nnx.Conv(
        C_in, C_out, (3, 3), strides=(2, 2), padding="VALID", rngs=nnx.Rngs(0)
    )

    def f(x_flat):
        x = x_flat.reshape(1, H, W, C_in)
        y = conv(x)
        return y.flatten()

    sparsity = jacobian_sparsity(f, np.zeros(input_size))

    # Output size with stride 2 and VALID padding
    out_h = (H - 3) // 2 + 1  # 2
    out_w = (W - 3) // 2 + 1  # 2
    expected_out_size = out_h * out_w * C_out

    assert sparsity.shape == (expected_out_size, input_size)
    assert sparsity.nnz > 0

    # Verify against actual Jacobian
    x_test = jax.random.normal(jax.random.key(42), (input_size,))
    actual_jac = jax.jacobian(f)(x_test)
    actual_nonzero = (np.abs(actual_jac) > 1e-10).astype(int)
    detected = sparsity.todense().astype(int)

    np.testing.assert_array_equal(
        detected, actual_nonzero, "Detected sparsity should match actual Jacobian"
    )


def test_conv_transpose():
    """Transposed convolution exercises lhs_dilation code path."""
    H, W = 4, 4
    C_in, C_out = 1, 1
    input_size = H * W * C_in

    # ConvTranspose uses lhs_dilation internally
    conv_t = nnx.ConvTranspose(
        C_in, C_out, (3, 3), strides=(2, 2), padding="VALID", rngs=nnx.Rngs(0)
    )

    def f(x_flat):
        x = x_flat.reshape(1, H, W, C_in)
        y = conv_t(x)
        return y.flatten()

    sparsity = jacobian_sparsity(f, np.zeros(input_size))

    assert sparsity.nnz > 0

    # Verify against actual Jacobian
    x_test = jax.random.normal(jax.random.key(42), (input_size,))
    actual_jac = jax.jacobian(f)(x_test)
    actual_nonzero = (np.abs(actual_jac) > 1e-10).astype(int)
    detected = sparsity.todense().astype(int)

    np.testing.assert_array_equal(
        detected, actual_nonzero, "Detected sparsity should match actual Jacobian"
    )


# Non-square shapes


@pytest.mark.parametrize(
    ("input_hw", "kernel_size", "channels"),
    [
        # Non-square input + square kernel
        ((5, 7), (3, 3), (1, 1)),
        # Non-square input + non-square kernel
        ((6, 4), (2, 3), (1, 1)),
        # Both non-square + multi-channel
        ((5, 7), (3, 2), (2, 3)),
    ],
)
def test_conv_non_square(input_hw, kernel_size, channels):
    """Non-square spatial dimensions exercise asymmetric stride arithmetic."""
    check_conv_sparsity(input_hw, kernel_size, channels, verify_exact=True)


# 1D convolution


def _check_conv1d_sparsity(
    length: int,
    kernel_size: int,
    channels: tuple[int, int],
    *,
    strides: tuple[int] = (1,),
    padding: str = "VALID",
) -> None:
    """Verify 1D conv sparsity via ``jax.lax.conv_general_dilated``.

    Uses layout ``(N, C, L)`` to exercise the 1D code path.
    """
    C_in, C_out = channels
    input_size = length * C_in
    # Kernel layout IOH: (C_in, C_out, kernel_size)
    kernel = jax.random.normal(jax.random.key(0), (C_in, C_out, kernel_size))

    def f(x_flat):
        # Layout: (N, C_in, L)
        x = x_flat.reshape(1, C_in, length)
        y = lax.conv_general_dilated(
            x,
            kernel,
            window_strides=strides,
            padding=padding,
            dimension_numbers=("NCH", "IOH", "NCH"),
        )
        return y.flatten()

    sparsity = jacobian_sparsity(f, np.zeros(input_size))

    assert sparsity.nnz > 0
    out_size = sparsity.shape[0]
    # Conservative audit
    assert sparsity.nnz < out_size * input_size

    x_test = jax.random.normal(jax.random.key(42), (input_size,))
    actual_jac = jax.jacobian(f)(x_test)
    actual_nonzero = (np.abs(actual_jac) > 1e-10).astype(int)
    detected = sparsity.todense().astype(int)

    np.testing.assert_array_equal(
        detected, actual_nonzero, "1D conv sparsity should match actual Jacobian"
    )


def test_conv1d_single_channel():
    """1D convolution with a single input and output channel."""
    _check_conv1d_sparsity(10, 3, (1, 1))


def test_conv1d_multi_channel():
    """1D convolution with multiple channels."""
    _check_conv1d_sparsity(8, 3, (2, 3))


# Grouped convolution


def _check_grouped_conv_sparsity(
    input_hw: tuple[int, int],
    kernel_size: tuple[int, int],
    channels: tuple[int, int],
    feature_group_count: int,
    *,
    strides: tuple[int, int] = (1, 1),
    padding: str = "VALID",
) -> None:
    """Verify grouped conv sparsity via ``jax.lax.conv_general_dilated``.

    Checks exact match against ``jax.jacobian`` and verifies
    the pattern is block-diagonal (sparser than ungrouped).
    """
    H, W = input_hw
    C_in, C_out = channels
    assert C_in % feature_group_count == 0
    assert C_out % feature_group_count == 0

    input_size = H * W * C_in
    # Kernel layout IOHW: (C_in // groups, C_out, kH, kW)
    kernel = jax.random.normal(
        jax.random.key(0),
        (C_in // feature_group_count, C_out, *kernel_size),
    )

    def f(x_flat):
        # NCHW layout
        x = x_flat.reshape(1, C_in, H, W)
        y = lax.conv_general_dilated(
            x,
            kernel,
            window_strides=strides,
            padding=padding,
            dimension_numbers=("NCHW", "IOHW", "NCHW"),
            feature_group_count=feature_group_count,
        )
        return y.flatten()

    sparsity = jacobian_sparsity(f, np.zeros(input_size))

    assert sparsity.nnz > 0
    out_size = sparsity.shape[0]
    # Conservative audit
    assert sparsity.nnz < out_size * input_size

    # Exact match against jax.jacobian
    x_test = jax.random.normal(jax.random.key(42), (input_size,))
    actual_jac = jax.jacobian(f)(x_test)
    actual_nonzero = (np.abs(actual_jac) > 1e-10).astype(int)
    detected = sparsity.todense().astype(int)

    np.testing.assert_array_equal(
        detected,
        actual_nonzero,
        "Grouped conv sparsity should match actual Jacobian",
    )


def test_grouped_conv_2_groups():
    """Grouped convolution with 2 groups shows block-diagonal sparsity."""
    _check_grouped_conv_sparsity((4, 4), (3, 3), (4, 4), feature_group_count=2)


def test_grouped_conv_3_groups():
    """Grouped convolution with 3 groups."""
    _check_grouped_conv_sparsity((4, 4), (3, 3), (6, 6), feature_group_count=3)


def test_grouped_conv_asymmetric_channels():
    """Grouped convolution with more output channels than input channels."""
    _check_grouped_conv_sparsity((4, 4), (3, 3), (4, 8), feature_group_count=2)


# Depthwise convolution


def test_depthwise_conv():
    """Depthwise convolution: ``feature_group_count == C_in``, kernel input dim = 1.

    Each output channel depends only on the corresponding input channel,
    producing a clear block-diagonal pattern.
    """
    H, W = 5, 5
    C_in = 3
    C_out = 3  # Same as C_in for standard depthwise
    input_size = H * W * C_in

    # Depthwise: groups == C_in, kernel layout IOHW: (1, C_out, kH, kW)
    kernel = jax.random.normal(jax.random.key(0), (1, C_out, 3, 3))

    def f(x_flat):
        x = x_flat.reshape(1, C_in, H, W)
        y = lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(1, 1),
            padding="VALID",
            dimension_numbers=("NCHW", "IOHW", "NCHW"),
            feature_group_count=C_in,
        )
        return y.flatten()

    sparsity = jacobian_sparsity(f, np.zeros(input_size))

    assert sparsity.nnz > 0
    out_size = sparsity.shape[0]
    assert sparsity.nnz < out_size * input_size

    # Exact match
    x_test = jax.random.normal(jax.random.key(42), (input_size,))
    actual_jac = jax.jacobian(f)(x_test)
    actual_nonzero = (np.abs(actual_jac) > 1e-10).astype(int)
    detected = sparsity.todense().astype(int)

    np.testing.assert_array_equal(
        detected,
        actual_nonzero,
        "Depthwise conv sparsity should match actual Jacobian",
    )


# Combinations


def test_grouped_conv_with_strides():
    """Grouped convolution combined with stride > 1."""
    _check_grouped_conv_sparsity(
        (6, 6), (3, 3), (4, 4), feature_group_count=2, strides=(2, 2)
    )


def test_grouped_conv_with_same_padding():
    """Grouped convolution combined with SAME padding."""
    _check_grouped_conv_sparsity(
        (5, 5), (3, 3), (4, 4), feature_group_count=2, padding="SAME"
    )


# Dilated kernel (rhs_dilation)


def test_rhs_dilation():
    """Direct rhs_dilation test exercises dilated (atrous) convolution.

    Previously only tested indirectly via ConvTranspose (which uses lhs_dilation).
    """
    H, W = 7, 7
    C_in, C_out = 1, 1
    input_size = H * W * C_in

    # Kernel layout IOHW: (C_in, C_out, kH, kW)
    kernel = jax.random.normal(jax.random.key(0), (C_in, C_out, 3, 3))

    def f(x_flat):
        x = x_flat.reshape(1, C_in, H, W)
        y = lax.conv_general_dilated(
            x,
            kernel,
            window_strides=(1, 1),
            padding="VALID",
            dimension_numbers=("NCHW", "IOHW", "NCHW"),
            rhs_dilation=(2, 2),
        )
        return y.flatten()

    sparsity = jacobian_sparsity(f, np.zeros(input_size))

    assert sparsity.nnz > 0
    out_size = sparsity.shape[0]
    assert sparsity.nnz < out_size * input_size

    # Exact match
    x_test = jax.random.normal(jax.random.key(42), (input_size,))
    actual_jac = jax.jacobian(f)(x_test)
    actual_nonzero = (np.abs(actual_jac) > 1e-10).astype(int)
    detected = sparsity.todense().astype(int)

    np.testing.assert_array_equal(
        detected,
        actual_nonzero,
        "Dilated kernel conv sparsity should match actual Jacobian",
    )


@pytest.mark.array_ops
def test_conv_batch_group_count_one_is_identity():
    """Explicit batch_group_count=1 produces the same result as omitting it."""
    input_size = 2 * 1 * 3 * 3  # (N=2, C=1, H=3, W=3)

    def with_bgc(x):
        lhs = x.reshape(2, 1, 3, 3)
        rhs = jnp.ones((1, 1, 2, 2))
        return lax.conv_general_dilated(
            lhs,
            rhs,
            window_strides=(1, 1),
            padding="VALID",
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
            batch_group_count=1,
        ).flatten()

    def without_bgc(x):
        lhs = x.reshape(2, 1, 3, 3)
        rhs = jnp.ones((1, 1, 2, 2))
        return lax.conv_general_dilated(
            lhs,
            rhs,
            window_strides=(1, 1),
            padding="VALID",
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        ).flatten()

    sp_with = jacobian_sparsity(with_bgc, np.zeros(input_size))
    sp_without = jacobian_sparsity(without_bgc, np.zeros(input_size))
    np.testing.assert_array_equal(sp_with.todense(), sp_without.todense())


@pytest.mark.array_ops
def test_conv_batch_group_count():
    """Conv with batch_group_count > 1 is block-diagonal.

    Each batch group's outputs depend only on that group's inputs.
    """
    input_size = 4 * 1 * 3 * 3  # (N=4, C=1, H=3, W=3)

    def f(x):
        lhs = x.reshape(4, 1, 3, 3)
        rhs = jnp.ones((2, 1, 2, 2))
        return lax.conv_general_dilated(
            lhs,
            rhs,
            window_strides=(1, 1),
            padding="VALID",
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
            batch_group_count=2,
        ).flatten()

    result = jacobian_sparsity(f, np.zeros(input_size)).todense().astype(int)
    expected = np.abs(jax.jacobian(f)(jnp.ones(input_size))) > 1e-10
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_conv_batch_group_count_multi_channel():
    """Conv with batch_group_count > 1 and multiple input channels.

    Input (6,2,4,4) with batch_group_count=3 and kernel (6,2,3,3).
    Output (2,6,2,2): each output feature pair reads from a different
    input batch pair, with full channel mixing within each pair.

    Batch group mapping: in_batch = out_batch + group * 2,
    where group = out_feature // 2 and n_out_batches = 6 // 3 = 2.
    """
    input_size = 6 * 2 * 4 * 4  # (N=6, C_in=2, H=4, W=4)

    def f(x):
        return lax.conv_general_dilated(
            x.reshape(6, 2, 4, 4),
            jnp.ones((6, 2, 3, 3)),
            window_strides=(1, 1),
            padding="VALID",
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
            batch_group_count=3,
        ).flatten()

    # Spatial block: each of 2×2 output positions reads a 3×3 window
    # across both input channels (2 channels × 4×4 = 32 elements per batch).
    spatial_block = np.zeros((4, 32), dtype=int)
    for h, w in np.ndindex(2, 2):
        for ch, kh, kw in np.ndindex(2, 3, 3):
            spatial_block[h * 2 + w, ch * 16 + (h + kh) * 4 + (w + kw)] = 1

    # Place blocks: out (n, c, :, :) reads from input batch n + (c // 2) * 2.
    expected = np.zeros((48, 192), dtype=int)
    for n in range(2):
        for c in range(6):
            in_batch = n + (c // 2) * 2
            out_start = n * 24 + c * 4
            in_start = in_batch * 32
            expected[out_start : out_start + 4, in_start : in_start + 32] = (
                spatial_block
            )

    result = jacobian_sparsity(f, np.zeros(input_size)).todense().astype(int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
def test_conv_batch_group_count_equals_n():
    """Conv with batch_group_count = N (degenerate: every batch is its own group).

    Input (6,1,3,3) with batch_group_count=6 and kernel (6,1,2,2).
    Output (1,6,2,2): each output feature reads from exactly one input batch.
    The pattern is block-diagonal with 6 identical 2x2-conv-on-3x3 blocks.
    """
    input_size = 6 * 1 * 3 * 3  # (N=6, C_in=1, H=3, W=3)

    def f(x):
        return lax.conv_general_dilated(
            x.reshape(6, 1, 3, 3),
            jnp.ones((6, 1, 2, 2)),
            window_strides=(1, 1),
            padding="VALID",
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
            batch_group_count=6,
        ).flatten()

    # Single block: 2×2 conv with 2×2 kernel on 3×3 input.
    block = np.zeros((4, 9), dtype=int)
    for h, w in np.ndindex(2, 2):
        for kh, kw in np.ndindex(2, 2):
            block[h * 2 + w, (h + kh) * 3 + (w + kw)] = 1

    # 6 identical blocks along the diagonal (one per batch/feature).
    expected = np.zeros((24, 54), dtype=int)
    for c in range(6):
        expected[c * 4 : (c + 1) * 4, c * 9 : (c + 1) * 9] = block

    result = jacobian_sparsity(f, np.zeros(input_size)).todense().astype(int)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.array_ops
@pytest.mark.fallback
def test_conv_input_dependent_kernel_conservative():
    """Conv with a kernel derived from the function input goes conservative.

    Convolution is bilinear in (data, kernel),
    so an input-dependent kernel (e.g., a hypernetwork) is valid user code
    and must not be rejected.
    Without precise bilinear tracking,
    every output must depend on both the data and the kernel dependencies.

    TODO(conv_general_dilated): the precise pattern is
    the data window plus the full kernel per output element.
    Any conservative superset that does not raise is acceptable.
    """

    def f(x):
        kernel = x[:4].reshape(1, 1, 2, 2)
        input_data = x[4:].reshape(1, 1, 3, 3)
        return lax.conv_general_dilated(
            input_data,
            kernel,
            window_strides=(1, 1),
            padding="VALID",
            dimension_numbers=("NCHW", "OIHW", "NCHW"),
        ).flatten()

    result = jacobian_sparsity(f, np.zeros(13)).todense().astype(int)
    expected = np.ones((4, 13), dtype=int)
    np.testing.assert_array_equal(result, expected)
