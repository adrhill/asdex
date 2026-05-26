"""Pytest configuration and fixtures for asdex tests."""

import jax
import pytest
from jax.experimental.sparse import BCOO
from numpy.testing import assert_allclose


def _to_dense(x):
    """Convert BCOO to dense, pass through other arrays."""
    return x.todense() if isinstance(x, BCOO) else x


def _assert_trees_allclose(actual, expected, *, rtol=1e-7, atol=0):
    """Assert two pytrees have matching structure and allclose leaves.

    Automatically converts BCOO leaves to dense for comparison.
    Structure is compared after conversion to handle BCOO's custom pytree node.
    """
    actual_dense = jax.tree.map(
        _to_dense, actual, is_leaf=lambda x: isinstance(x, BCOO)
    )
    assert jax.tree.structure(actual_dense) == jax.tree.structure(expected)
    jax.tree.map(
        lambda a, e: assert_allclose(a, e, rtol=rtol, atol=atol), actual_dense, expected
    )


@pytest.fixture
def assert_trees_allclose():
    """Fixture providing the assert_trees_allclose helper."""
    return _assert_trees_allclose


@pytest.fixture(params=["dense", "bcoo"])
def output_format(request):
    """Parametrize over output formats."""
    return request.param


@pytest.fixture(params=["fwd", "rev"])
def jacobian_mode(request):
    """Parametrize over Jacobian AD modes."""
    return request.param


@pytest.fixture(params=["fwd_over_rev", "rev_over_fwd", "rev_over_rev"])
def hessian_mode(request):
    """Parametrize over Hessian AD modes."""
    return request.param


@pytest.fixture(params=[None, 3])
def chunk_size(request):
    """Parametrize over chunk sizes for decompression."""
    return request.param


def pytest_report_teststatus(report, config):
    """Suppress progress dots for passing tests to keep output concise."""
    if report.when == "call" and report.passed:
        return "passed", "", ""
    return None


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "elementwise: simple element-wise operations")
    config.addinivalue_line(
        "markers", "array_ops: array manipulation (slice, concat, reshape, etc.)"
    )
    config.addinivalue_line(
        "markers", "control_flow: conditional operations (where, select)"
    )
    config.addinivalue_line(
        "markers", "reduction: reduction operations (sum, max, prod)"
    )
    config.addinivalue_line("markers", "vmap: batched/vmapped operations")
    config.addinivalue_line(
        "markers", "fallback: documents conservative fallback behavior (TODO)"
    )
    config.addinivalue_line("markers", "bug: documents known bugs")
    config.addinivalue_line("markers", "coloring: row coloring algorithm tests")
    config.addinivalue_line("markers", "jacobian: sparse Jacobian computation tests")
    config.addinivalue_line(
        "markers", "hessian: Hessian sparsity detection and computation"
    )
    config.addinivalue_line(
        "markers", "dashboard: benchmarks tracked in the GitHub Pages dashboard"
    )
