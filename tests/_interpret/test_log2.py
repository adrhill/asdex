"""Tests for the log2 elementwise propagation."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from asdex import jacobian_sparsity
from tests._utils import assert_jacobian_sparsity_exact


@pytest.mark.elementwise
@pytest.mark.parametrize(
    "shape",
    [
        pytest.param((4,), id="1d"),
        pytest.param((3, 4), id="2d"),
        pytest.param((2, 3, 4), id="3d"),
    ],
)
@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_log2_matches_jacobian(shape, dtype):
    """log2 has a diagonal Jacobian on non-square 1D/2D/3D shapes."""
    n = int(np.prod(shape))

    def f(x):
        return jnp.log2(x.reshape(shape)).flatten()

    with jax.enable_x64(dtype == np.float64):
        x = jnp.arange(1.0, n + 1.0, dtype=dtype) / 4
        assert_jacobian_sparsity_exact(f, x)


@pytest.mark.elementwise
def test_log2_zero_size():
    """log2 on a zero-sized array returns an empty pattern."""

    def f(x):
        return jnp.log2(x.reshape(0, 3)).flatten()

    result = jacobian_sparsity(f, np.zeros(0))
    assert result.shape == (0, 0)
    assert result.nnz == 0
