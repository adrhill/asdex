"""Pytest configuration and fixtures for asdex tests."""

import jax
import pytest
from jax.experimental.sparse import BCOO
from numpy.testing import assert_allclose


def _to_dense(x):
    """Convert BCOO or scipy sparse to dense, pass through other arrays."""
    if isinstance(x, BCOO):
        return x.todense()
    if hasattr(x, "toarray"):
        return x.toarray()
    return x


def _densify_tree(tree):
    """Convert every BCOO leaf in a pytree to dense, leaving other leaves untouched."""
    return jax.tree.map(_to_dense, tree, is_leaf=lambda x: isinstance(x, BCOO))


def _assert_trees_allclose(actual, expected, *, rtol=1e-7, atol=0):
    """Assert two pytrees have matching structure and allclose leaves.

    Automatically converts BCOO leaves to dense on both sides.
    Structure is compared after conversion to handle BCOO's custom pytree node.
    """
    actual_dense = _densify_tree(actual)
    expected_dense = _densify_tree(expected)
    assert jax.tree.structure(actual_dense) == jax.tree.structure(expected_dense)
    jax.tree.map(
        lambda a, e: assert_allclose(a, e, rtol=rtol, atol=atol),
        actual_dense,
        expected_dense,
    )


@pytest.fixture
def assert_trees_allclose():
    """Fixture providing the assert_trees_allclose helper."""
    return _assert_trees_allclose


@pytest.fixture
def to_dense():
    """Fixture providing the _to_dense helper."""
    return _to_dense


@pytest.fixture(params=["dense", "bcoo"])
def output_format(request):
    """Parametrize over output formats."""
    return request.param


@pytest.fixture(
    params=["dense", "bcoo", "numpy_dense", "scipy_coo", "scipy_csr", "scipy_csc"]
)
def all_output_format(request):
    """Parametrize over all output formats."""
    return request.param


@pytest.fixture(params=["fwd", "rev"])
def jacobian_mode(request):
    """Parametrize over Jacobian AD modes."""
    return request.param


@pytest.fixture(params=["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
def hessian_mode(request):
    """Parametrize over Hessian AD modes."""
    return request.param


@pytest.fixture(params=[None, 2, 3])
def chunk_size(request):
    """Parametrize over chunk sizes for decompression."""
    return request.param
