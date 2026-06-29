"""Tests for the public compressed API and standalone decompression.

The internal compress-then-decompress numerics are already covered end-to-end
by the one-shot ``jacobian``/``hessian`` tests, which share the same compress
core and gather primitive.
These tests target the public surface the e2e path never invokes:
``compressed_*`` / ``value_and_compressed_*`` returning a standalone ``B``,
and ``decompress`` / ``decompress_data`` called directly on a caller-supplied
``B`` (the raw array, the public gather, validation, the full format set).
"""

import builtins
import sys

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental.sparse import BCOO
from jax.flatten_util import ravel_pytree
from numpy.testing import assert_allclose

from asdex import (
    compressed_hessian,
    compressed_hessian_from_coloring,
    compressed_jacobian,
    compressed_jacobian_from_coloring,
    decompress,
    decompress_data,
    hessian_coloring,
    jacobian_coloring,
    value_and_compressed_hessian,
    value_and_compressed_hessian_from_coloring,
    value_and_compressed_jacobian,
    value_and_compressed_jacobian_from_coloring,
)

pytestmark = pytest.mark.filterwarnings("ignore::asdex.DenseColoringWarning")


def _expected_dim(coloring):
    """Second axis of ``B``: output size ``m`` in fwd mode, input size ``n`` otherwise."""
    return coloring.sparsity.m if coloring.mode == "fwd" else coloring.sparsity.n


def _flat_reference_jacobian(f, x):
    """Dense ``(m, n)`` Jacobian via a raveled function, matching asdex's flattening."""
    x_flat, unravel = ravel_pytree(x)

    def f_flat(xf):
        return ravel_pytree(f(unravel(xf)))[0]

    return jax.jacobian(f_flat)(x_flat)


def _flat_reference_hessian(f, x):
    """Dense ``(n, n)`` Hessian via a raveled function, matching asdex's flattening."""
    x_flat, unravel = ravel_pytree(x)

    def f_flat(xf):
        return f(unravel(xf))

    return jax.hessian(f_flat)(x_flat)


# Sample functions


def _jac_f(x):
    """Bidiagonal Jacobian: output i depends on x[i] and x[i+1]."""
    return (x[1:] - x[:-1]) ** 2


def _hess_f(x):
    """Tridiagonal Hessian from a chained quadratic."""
    return jnp.sum((x[1:] - x[:-1]) ** 2) + jnp.sum(x**3)


# Compressed Jacobian: shape contract


@pytest.mark.jacobian
def test_compressed_jacobian_shape(jacobian_mode):
    """B has shape (num_colors, dim): dim = n in rev, m in fwd."""
    x = jnp.arange(1.0, 11.0)
    coloring = jacobian_coloring(_jac_f, x, mode=jacobian_mode)
    B = compressed_jacobian_from_coloring(_jac_f, coloring)(x)
    assert B.shape == (coloring.num_colors, _expected_dim(coloring))


@pytest.mark.jacobian
def test_compressed_jacobian_one_shot_matches_from_coloring(jacobian_mode):
    """The one-shot compressed_jacobian agrees with the from_coloring variant."""
    x = jnp.arange(1.0, 11.0)
    coloring = jacobian_coloring(_jac_f, x, mode=jacobian_mode)
    B_one = compressed_jacobian(_jac_f, x, mode=jacobian_mode)(x)
    B_col = compressed_jacobian_from_coloring(_jac_f, coloring)(x)
    assert_allclose(B_one, B_col)


# Compressed Jacobian: round-trip through every format


@pytest.mark.jacobian
def test_compressed_jacobian_roundtrip(jacobian_mode, all_output_format, to_dense):
    """compressed_jacobian -> decompress matches jax.jacobian for every format."""
    x = jnp.arange(1.0, 11.0)
    coloring = jacobian_coloring(_jac_f, x, mode=jacobian_mode)
    B = compressed_jacobian_from_coloring(_jac_f, coloring)(x)
    J = decompress(coloring, B, output_format=all_output_format)
    assert_allclose(to_dense(J), jax.jacobian(_jac_f)(x), rtol=1e-6)


@pytest.mark.jacobian
def test_compressed_jacobian_chunk_size_invariant(jacobian_mode, chunk_size):
    """chunk_size changes only memory and timing, not the compressed result."""
    x = jnp.arange(1.0, 11.0)
    coloring = jacobian_coloring(_jac_f, x, mode=jacobian_mode)
    B_full = compressed_jacobian_from_coloring(_jac_f, coloring)(x)
    B_chunked = compressed_jacobian_from_coloring(
        _jac_f, coloring, chunk_size=chunk_size
    )(x)
    assert_allclose(B_full, B_chunked)


# Compressed Hessian: shape contract and round-trip


@pytest.mark.hessian
def test_compressed_hessian_shape(hessian_mode):
    """Hessian B has shape (num_colors, n)."""
    x = jnp.arange(1.0, 9.0)
    coloring = hessian_coloring(_hess_f, x, mode=hessian_mode)
    B = compressed_hessian_from_coloring(_hess_f, coloring)(x)
    assert B.shape == (coloring.num_colors, coloring.sparsity.n)


@pytest.mark.hessian
def test_compressed_hessian_roundtrip(hessian_mode, all_output_format, to_dense):
    """compressed_hessian -> decompress matches jax.hessian for every format."""
    x = jnp.arange(1.0, 9.0)
    coloring = hessian_coloring(_hess_f, x, mode=hessian_mode)
    B = compressed_hessian_from_coloring(_hess_f, coloring)(x)
    H = decompress(coloring, B, output_format=all_output_format)
    assert_allclose(to_dense(H), jax.hessian(_hess_f)(x), rtol=1e-6, atol=1e-9)


@pytest.mark.hessian
def test_compressed_hessian_one_shot_matches_from_coloring(hessian_mode):
    """The one-shot compressed_hessian agrees with the from_coloring variant."""
    x = jnp.arange(1.0, 9.0)
    coloring = hessian_coloring(_hess_f, x, mode=hessian_mode)
    B_one = compressed_hessian(_hess_f, x, mode=hessian_mode)(x)
    B_col = compressed_hessian_from_coloring(_hess_f, coloring)(x)
    assert_allclose(B_one, B_col)


# decompress_data: the jittable gather primitive


@pytest.mark.jacobian
def test_decompress_data_shape_and_dtype(jacobian_mode):
    """decompress_data returns a (nnz,) jax.Array whose dtype matches B."""
    x = jnp.arange(1.0, 11.0)
    coloring = jacobian_coloring(_jac_f, x, mode=jacobian_mode)
    B = compressed_jacobian_from_coloring(_jac_f, coloring)(x)
    data = decompress_data(coloring, B)
    assert isinstance(data, jax.Array)
    assert data.shape == (coloring.sparsity.nnz,)
    assert data.dtype == B.dtype


@pytest.mark.jacobian
def test_decompress_data_feeds_public_to_bcoo(jacobian_mode):
    """The gathered data densifies through the public to_bcoo to the reference.

    Uses only the public sparsity.to_bcoo, not the private rows/cols scatter,
    so the test does not duplicate the internal implementation.
    """
    x = jnp.arange(1.0, 11.0)
    coloring = jacobian_coloring(_jac_f, x, mode=jacobian_mode)
    B = compressed_jacobian_from_coloring(_jac_f, coloring)(x)
    data = decompress_data(coloring, B)
    J = coloring.sparsity.to_bcoo(data).todense()
    assert_allclose(J, jax.jacobian(_jac_f)(x), rtol=1e-6)


@pytest.mark.jacobian
def test_decompress_data_jittable(jacobian_mode):
    """decompress_data composes inside jax.jit."""
    x = jnp.arange(1.0, 11.0)
    coloring = jacobian_coloring(_jac_f, x, mode=jacobian_mode)
    compress = compressed_jacobian_from_coloring(_jac_f, coloring)
    B = compress(x)

    @jax.jit
    def pipeline(args):
        return decompress_data(coloring, compress(args))

    assert_allclose(pipeline(x), decompress_data(coloring, B))


@pytest.mark.jacobian
def test_decompress_data_custom_coo_triple(jacobian_mode):
    """rows/cols pair with decompress_data to build a custom COO triple."""
    x = jnp.arange(1.0, 11.0)
    coloring = jacobian_coloring(_jac_f, x, mode=jacobian_mode)
    B = compressed_jacobian_from_coloring(_jac_f, coloring)(x)
    data = decompress_data(coloring, B)
    rows, cols = coloring.sparsity.rows, coloring.sparsity.cols
    dense = np.zeros(coloring.sparsity.shape)
    dense[rows, cols] = np.asarray(data)
    assert_allclose(dense, jax.jacobian(_jac_f)(x), rtol=1e-6)


# Value-and-compressed variants


@pytest.mark.jacobian
def test_value_and_compressed_jacobian(jacobian_mode):
    """value_and_compressed_jacobian returns (value, B) matching the parts."""
    x = jnp.arange(1.0, 11.0)
    coloring = jacobian_coloring(_jac_f, x, mode=jacobian_mode)
    value, B = value_and_compressed_jacobian(_jac_f, x, mode=jacobian_mode)(x)
    assert_allclose(value, _jac_f(x))
    assert_allclose(B, compressed_jacobian_from_coloring(_jac_f, coloring)(x))


@pytest.mark.jacobian
def test_value_and_compressed_jacobian_from_coloring(jacobian_mode):
    """value_and_compressed_jacobian_from_coloring returns (value, B)."""
    x = jnp.arange(1.0, 11.0)
    coloring = jacobian_coloring(_jac_f, x, mode=jacobian_mode)
    value, B = value_and_compressed_jacobian_from_coloring(_jac_f, coloring)(x)
    assert_allclose(value, _jac_f(x))
    assert_allclose(B, compressed_jacobian_from_coloring(_jac_f, coloring)(x))


@pytest.mark.hessian
def test_value_and_compressed_hessian(hessian_mode):
    """value_and_compressed_hessian returns (value, B) matching the parts."""
    x = jnp.arange(1.0, 9.0)
    coloring = hessian_coloring(_hess_f, x, mode=hessian_mode)
    value, B = value_and_compressed_hessian(_hess_f, x, mode=hessian_mode)(x)
    assert_allclose(value, _hess_f(x), rtol=1e-6)
    assert_allclose(B, compressed_hessian_from_coloring(_hess_f, coloring)(x))


@pytest.mark.hessian
def test_value_and_compressed_hessian_from_coloring(hessian_mode):
    """value_and_compressed_hessian_from_coloring returns (value, B)."""
    x = jnp.arange(1.0, 9.0)
    coloring = hessian_coloring(_hess_f, x, mode=hessian_mode)
    value, B = value_and_compressed_hessian_from_coloring(_hess_f, coloring)(x)
    assert_allclose(value, _hess_f(x), rtol=1e-6)
    assert_allclose(B, compressed_hessian_from_coloring(_hess_f, coloring)(x))


# Auxiliary outputs


@pytest.mark.jacobian
def test_compressed_jacobian_has_aux():
    """has_aux returns (B, aux); value_and nests ((value, aux), B)."""

    def f(x):
        y = (x[1:] - x[:-1]) ** 2
        return y, {"mean": jnp.mean(y)}

    x = jnp.arange(1.0, 6.0)
    B, aux = compressed_jacobian(f, x, has_aux=True)(x)
    assert isinstance(B, jax.Array)
    assert set(aux) == {"mean"}

    (value, aux2), B2 = value_and_compressed_jacobian(f, x, has_aux=True)(x)
    assert_allclose(value, f(x)[0])
    assert_allclose(B2, B)
    assert_allclose(aux2["mean"], aux["mean"])


@pytest.mark.hessian
def test_compressed_hessian_has_aux():
    """has_aux returns (B, aux); value_and nests ((value, aux), B)."""

    def f(x):
        return jnp.sum(x**3), {"n": x.shape[0]}

    x = jnp.arange(1.0, 6.0)
    B, aux = compressed_hessian(f, x, has_aux=True)(x)
    assert isinstance(B, jax.Array)
    assert aux == {"n": 5}

    (value, aux2), B2 = value_and_compressed_hessian(f, x, has_aux=True)(x)
    assert_allclose(value, f(x)[0], rtol=1e-6)
    assert_allclose(B2, B)
    assert aux2 == {"n": 5}


# PyTree inputs and outputs: B stays a flat 2-D array


@pytest.mark.jacobian
def test_compressed_jacobian_pytree_input(jacobian_mode, to_dense):
    """Dict input keeps B a flat 2-D array; decompress is flat (m, n)."""

    def f(params):
        return params["a"] * jnp.sin(params["b"])

    params = {"a": jnp.arange(1.0, 4.0), "b": jnp.linspace(0.0, 1.0, 3)}
    coloring = jacobian_coloring(f, params, mode=jacobian_mode)
    B = compressed_jacobian_from_coloring(f, coloring)(params)
    assert B.ndim == 2
    assert B.shape == (coloring.num_colors, _expected_dim(coloring))

    J = decompress(coloring, B, output_format="dense")
    assert J.shape == coloring.sparsity.shape
    assert_allclose(to_dense(J), _flat_reference_jacobian(f, params), rtol=1e-6)


@pytest.mark.jacobian
def test_compressed_jacobian_pytree_output(jacobian_mode, to_dense):
    """PyTree output keeps B a flat 2-D array; decompress is flat (m, n)."""

    def f(x):
        return {"sq": x**2, "sum": jnp.sum(x)}

    x = jnp.arange(1.0, 5.0)
    coloring = jacobian_coloring(f, x, mode=jacobian_mode)
    B = compressed_jacobian_from_coloring(f, coloring)(x)
    assert B.ndim == 2

    J = decompress(coloring, B, output_format="dense")
    assert J.shape == coloring.sparsity.shape
    assert_allclose(to_dense(J), _flat_reference_jacobian(f, x), rtol=1e-6)


@pytest.mark.hessian
def test_compressed_hessian_pytree_input(hessian_mode, to_dense):
    """Dict input keeps Hessian B flat; decompress is flat (n, n)."""

    def f(params):
        return jnp.sum(params["a"] ** 2 * params["b"])

    params = {"a": jnp.arange(1.0, 4.0), "b": jnp.arange(2.0, 5.0)}
    coloring = hessian_coloring(f, params, mode=hessian_mode)
    B = compressed_hessian_from_coloring(f, coloring)(params)
    assert B.ndim == 2

    H = decompress(coloring, B, output_format="dense")
    assert H.shape == coloring.sparsity.shape
    assert_allclose(
        to_dense(H), _flat_reference_hessian(f, params), rtol=1e-6, atol=1e-9
    )


# Validation


@pytest.mark.jacobian
def test_decompress_data_wrong_num_colors_raises(jacobian_mode):
    """A compressed matrix with the wrong number of colors raises ValueError."""
    x = jnp.arange(1.0, 11.0)
    coloring = jacobian_coloring(_jac_f, x, mode=jacobian_mode)
    bad = jnp.zeros((coloring.num_colors + 1, _expected_dim(coloring)))
    with pytest.raises(ValueError, match="num_colors"):
        decompress_data(coloring, bad)


@pytest.mark.jacobian
def test_decompress_data_wrong_dim_raises(jacobian_mode):
    """A compressed matrix with the wrong second axis raises ValueError."""
    x = jnp.arange(1.0, 11.0)
    coloring = jacobian_coloring(_jac_f, x, mode=jacobian_mode)
    bad = jnp.zeros((coloring.num_colors, _expected_dim(coloring) + 1))
    with pytest.raises(ValueError, match="num_colors"):
        decompress_data(coloring, bad)


@pytest.mark.jacobian
def test_decompress_propagates_validation(jacobian_mode):
    """The shape check propagates from decompress_data into decompress."""
    x = jnp.arange(1.0, 11.0)
    coloring = jacobian_coloring(_jac_f, x, mode=jacobian_mode)
    bad = jnp.zeros((coloring.num_colors, _expected_dim(coloring) + 1))
    with pytest.raises(ValueError, match="num_colors"):
        decompress(coloring, bad)


@pytest.mark.jacobian
def test_decompress_1d_compressed_raises():
    """A 1-D compressed argument raises rather than reading garbage."""
    x = jnp.arange(1.0, 11.0)
    coloring = jacobian_coloring(_jac_f, x)
    with pytest.raises(ValueError, match="num_colors"):
        decompress_data(coloring, jnp.zeros(coloring.num_colors))


@pytest.mark.jacobian
def test_decompress_unknown_output_format_raises():
    """An unknown output_format raises ValueError from decompress."""
    x = jnp.arange(1.0, 11.0)
    coloring = jacobian_coloring(_jac_f, x)
    B = compressed_jacobian_from_coloring(_jac_f, coloring)(x)
    with pytest.raises(ValueError, match="Unknown output_format"):
        decompress(coloring, B, output_format="scipy_cooo")  # ty: ignore[invalid-argument-type]


@pytest.mark.jacobian
def test_decompress_scipy_without_scipy_raises(monkeypatch):
    """Requesting a scipy format without scipy installed raises ImportError."""
    x = jnp.arange(1.0, 11.0)
    coloring = jacobian_coloring(_jac_f, x)
    B = compressed_jacobian_from_coloring(_jac_f, coloring)(x)

    real_import = builtins.__import__

    def blocked_import(name, *args, **kwargs):
        if name.startswith("scipy"):
            raise ImportError(f"No module named {name!r}")
        return real_import(name, *args, **kwargs)

    for mod in list(sys.modules):
        if mod.startswith("scipy"):
            monkeypatch.delitem(sys.modules, mod)
    monkeypatch.setattr(builtins, "__import__", blocked_import)

    with pytest.raises(ImportError, match=r"pip install 'asdex\[scipy\]'"):
        decompress(coloring, B, output_format="scipy_coo")


# Output format types


@pytest.mark.jacobian
def test_decompress_returns_expected_types():
    """Each output_format yields the documented container type."""
    x = jnp.arange(1.0, 11.0)
    coloring = jacobian_coloring(_jac_f, x)
    B = compressed_jacobian_from_coloring(_jac_f, coloring)(x)

    assert isinstance(decompress(coloring, B, output_format="bcoo"), BCOO)
    assert isinstance(decompress(coloring, B, output_format="dense"), jax.Array)
    assert isinstance(decompress(coloring, B, output_format="numpy_dense"), np.ndarray)

    from scipy.sparse import coo_array, csc_array, csr_array  # noqa: PLC0415

    assert isinstance(decompress(coloring, B, output_format="scipy_coo"), coo_array)
    assert isinstance(decompress(coloring, B, output_format="scipy_csr"), csr_array)
    assert isinstance(decompress(coloring, B, output_format="scipy_csc"), csc_array)


# Empty / zero patterns


@pytest.mark.jacobian
def test_compressed_jacobian_zero_pattern(to_dense):
    """A function with an all-zero Jacobian compresses and decompresses cleanly."""

    def f(x):
        return jnp.zeros(3)

    x = jnp.arange(1.0, 4.0)
    coloring = jacobian_coloring(f, x)
    B = compressed_jacobian_from_coloring(f, coloring)(x)
    assert B.shape == (coloring.num_colors, _expected_dim(coloring))
    J = decompress(coloring, B, output_format="bcoo")
    assert to_dense(J).shape == coloring.sparsity.shape
    assert_allclose(to_dense(J), np.zeros((3, 3)))


@pytest.mark.jacobian
def test_value_and_compressed_jacobian_zero_pattern():
    """value_and_compressed_jacobian returns the value on an empty pattern."""

    def f(x):
        return jnp.zeros(3)

    x = jnp.arange(1.0, 4.0)
    value, B = value_and_compressed_jacobian(f, x)(x)
    assert_allclose(value, f(x))
    assert B.shape[0] == jacobian_coloring(f, x).num_colors
