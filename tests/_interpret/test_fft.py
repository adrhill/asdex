"""Tests for the _prop_fft handler.

Every FFT mixes all elements along the transformed trailing axes,
leaving the leading batch axes independent.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity
from tests._utils import (
    assert_flat_jacobian_sparsity_exact,
    assert_jacobian_sparsity_exact,
)


@pytest.mark.array_ops
@pytest.mark.parametrize(
    ("transform", "n", "complex_input"),
    [
        pytest.param(lambda x: jnp.fft.fft(x), 8, False, id="fft"),
        pytest.param(lambda x: jnp.fft.fft(x), 7, False, id="fft-non-power-of-two"),
        pytest.param(lambda x: jnp.fft.ifft(x), 8, False, id="ifft"),
        pytest.param(lambda x: jnp.fft.rfft(x), 8, False, id="rfft"),
        pytest.param(lambda x: jnp.fft.rfft(x), 7, False, id="rfft-non-power-of-two"),
        pytest.param(lambda x: jnp.fft.irfft(x, n=8), 5, True, id="irfft"),
    ],
)
def test_fft_1d(transform, n, complex_input):
    """1D transforms couple every input element to every output element."""
    rng = np.random.default_rng(0)
    x = rng.normal(size=n)
    if complex_input:
        x = (x + 1j * rng.normal(size=n)).astype(np.complex64)

    assert_jacobian_sparsity_exact(lambda x: jnp.abs(transform(x)), x)


@pytest.mark.array_ops
@pytest.mark.parametrize(
    "transform",
    [
        pytest.param(lambda z: jnp.fft.fft(z), id="fft"),
        pytest.param(lambda z: jnp.fft.ifft(z), id="ifft"),
    ],
)
def test_fft_complex_input(transform):
    """Complex inputs couple every real and imaginary component in a batch block."""
    rng = np.random.default_rng(0)
    z = (rng.normal(size=8) + 1j * rng.normal(size=8)).astype(np.complex64)

    assert_jacobian_sparsity_exact(transform, z, holomorphic=True)


@pytest.mark.array_ops
@pytest.mark.parametrize(
    ("transform", "shape"),
    [
        pytest.param(
            lambda x: jnp.fft.fft2(x, axes=(0, 2)),
            (2, 3, 4),
            id="fft2-mixed-axes",
        ),
        pytest.param(
            lambda x: jnp.fft.ifft2(x, axes=(0, 2)),
            (2, 3, 4),
            id="ifft2-mixed-axes",
        ),
        pytest.param(
            lambda x: jnp.fft.fft2(x, axes=(0, 1)),
            (3, 5),
            id="fft2-non-power-of-two",
        ),
    ],
)
def test_fft2_explicit_axes(transform, shape):
    """2D transforms couple only the transformed axes and keep batch axes separate."""
    x = np.random.default_rng(0).normal(size=shape)

    assert_flat_jacobian_sparsity_exact(lambda x: jnp.abs(transform(x)), x)


@pytest.mark.array_ops
@pytest.mark.parametrize(
    "transform",
    [
        pytest.param(lambda x: jnp.fft.fftn(x), id="fftn"),
        pytest.param(lambda x: jnp.fft.fftn(x, axes=(1, 2)), id="fftn-leading-batch"),
        pytest.param(lambda x: jnp.fft.rfftn(x, axes=(1, 2)), id="rfftn-leading-batch"),
    ],
)
def test_fftn_3d(transform):
    """3D transforms mix all transformed axes while leaving a leading batch axis."""
    x = np.random.default_rng(0).normal(size=(2, 3, 4))

    assert_flat_jacobian_sparsity_exact(lambda x: jnp.abs(transform(x)), x)


@pytest.mark.array_ops
def test_fft_composed_abs():
    """An absolute value keeps the dense block pattern of the underlying FFT."""
    x = np.random.default_rng(0).normal(size=(2, 3, 4))

    def f(x):
        return jnp.abs(jnp.fft.fft2(x, axes=(1, 2)))

    assert_flat_jacobian_sparsity_exact(f, x)


@pytest.mark.array_ops
def test_fft_size_one():
    """A length-one transform is the identity."""
    assert_jacobian_sparsity_exact(lambda x: jnp.real(jnp.fft.fft(x)), np.ones(1))


@pytest.mark.array_ops
def test_fft_empty_batch():
    """A zero-sized batch axis produces an empty pattern without crashing."""
    sparsity = jacobian_sparsity(
        lambda x: jnp.fft.fft2(x, axes=(1, 2)), np.empty((0, 3, 4))
    )

    assert sparsity.shape == (0, 0)
