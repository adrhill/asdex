"""Tests for Equinox's unvmap_any, unvmap_all and unvmap_max.

They reduce a boolean or integer array to a scalar across all elements and batch members,
so their outputs are boolean or integer and have zero derivative.
The operands are fed through ``astype`` so that they carry dependencies,
since comparisons already write empty index sets.

Requires equinox to be installed.
"""

import numpy as np
import pytest

eqx = pytest.importorskip("equinox")
eqxi = eqx.internal

import jax.numpy as jnp  # noqa: E402

from tests._utils import assert_jacobian_sparsity_exact  # noqa: E402


@pytest.mark.reduction
@pytest.mark.parametrize(
    ("op", "dtype"),
    [
        (eqxi.unvmap_any, jnp.bool_),
        (eqxi.unvmap_all, jnp.bool_),
        (eqxi.unvmap_max, jnp.int32),
    ],
    ids=["unvmap_any", "unvmap_all", "unvmap_max"],
)
def test_zero_derivative(op, dtype):
    """The scalar output adds no dependencies, so only the diagonal of ``x * c`` remains."""

    def f(x):
        return x * op(x.astype(dtype))

    result = assert_jacobian_sparsity_exact(f, np.arange(1.0, 4.0))
    assert result._fallbacks == ()
