"""Tests that the public differentiation functions return jit- and vmap-able callables.

Every public function in the ``jacobian`` / ``hessian`` / ``value_and_*`` /
``compressed_*`` families (and their ``*_from_coloring`` variants) returns a
plain callable over arrays.
This file pins the promise the docstrings make ("the returned function is
jit-able by the caller") and extends it to ``jax.vmap``:
under ``jax.jit``, ``jax.vmap``, or a ``jit``-of-``vmap`` composition,
each returned callable produces the same result as calling it eagerly.

The reference is deliberately the eager call itself, not an independent JAX
Jacobian: eager correctness is already covered in ``test_decompression.py`` and
``test_compression.py``, so the claim under test here is precisely that the
transforms preserve that result.
"""

import jax
import jax.numpy as jnp
import pytest
from jax.experimental.sparse import BCOO

from asdex import (
    compressed_hessian,
    compressed_hessian_from_coloring,
    compressed_jacobian,
    compressed_jacobian_from_coloring,
    hessian,
    hessian_coloring,
    hessian_from_coloring,
    jacobian,
    jacobian_coloring,
    jacobian_from_coloring,
    value_and_compressed_hessian,
    value_and_compressed_hessian_from_coloring,
    value_and_compressed_jacobian,
    value_and_compressed_jacobian_from_coloring,
    value_and_hessian,
    value_and_hessian_from_coloring,
    value_and_jacobian,
    value_and_jacobian_from_coloring,
)
from tests.conftest import _densify_tree

pytestmark = pytest.mark.vmap


# Sample functions and shared inputs


def _f_jac(x):
    """Bidiagonal Jacobian: output i depends on x[i] and x[i+1]."""
    return (x[1:] - x[:-1]) ** 2


def _f_hess(x):
    """Tridiagonal Hessian from a chained quadratic plus a cubic diagonal."""
    return jnp.sum((x[1:] - x[:-1]) ** 2) + jnp.sum(x**3)


_X = jnp.arange(1.0, 6.0)
# Scale (never shift) each sample: both derivatives here are translation-invariant
# in part, so scaling is what guarantees a distinct Jacobian and Hessian per row,
# which in turn makes a broadcast-instead-of-map vmap bug observable.
_BATCH = jnp.stack([_X, 2.0 * _X, 3.0 * _X, -_X])

_JAC_COLORING = jacobian_coloring(_f_jac, _X)
_HESS_COLORING = hessian_coloring(_f_hess, _X)


# Test cases
#
# Each case is a (make_fn, f, arg) triple such that make_fn(f, arg) builds
# the callable under test.
# arg is the sample input for the one-call APIs
# or the precomputed coloring for the *_from_coloring variants.
# The sparse-matrix functions additionally take an output_format,
# while the compressed ones return the plain array B and take none.


def _cases(triples):
    """Tag each (make_fn, f, arg) case with its family marker and the function name."""
    return [
        pytest.param(
            make_fn,
            f,
            arg,
            marks=pytest.mark.hessian
            if "hessian" in make_fn.__name__
            else pytest.mark.jacobian,
            id=make_fn.__name__,
        )
        for make_fn, f, arg in triples
    ]


_SPARSE_CASES = _cases(
    [
        (jacobian, _f_jac, _X),
        (jacobian_from_coloring, _f_jac, _JAC_COLORING),
        (value_and_jacobian, _f_jac, _X),
        (value_and_jacobian_from_coloring, _f_jac, _JAC_COLORING),
        (hessian, _f_hess, _X),
        (hessian_from_coloring, _f_hess, _HESS_COLORING),
        (value_and_hessian, _f_hess, _X),
        (value_and_hessian_from_coloring, _f_hess, _HESS_COLORING),
    ]
)

_COMPRESSED_CASES = _cases(
    [
        (compressed_jacobian, _f_jac, _X),
        (compressed_jacobian_from_coloring, _f_jac, _JAC_COLORING),
        (value_and_compressed_jacobian, _f_jac, _X),
        (value_and_compressed_jacobian_from_coloring, _f_jac, _JAC_COLORING),
        (compressed_hessian, _f_hess, _X),
        (compressed_hessian_from_coloring, _f_hess, _HESS_COLORING),
        (value_and_compressed_hessian, _f_hess, _X),
        (value_and_compressed_hessian_from_coloring, _f_hess, _HESS_COLORING),
    ]
)


# Reference helper


def _stack_batch(fn):
    """Eager per-sample reference for a vmap over ``_BATCH``, stacked and densified.

    Densification must happen before stacking:
    ``jax.tree.map`` would otherwise descend into the BCOO leaves
    and stack their data and indices into an invalid BCOO.
    """
    dense = [_densify_tree(fn(x)) for x in _BATCH]
    return jax.tree.map(lambda *leaves: jnp.stack(leaves), *dense)


# Sparse-matrix functions


@pytest.mark.parametrize(("make_fn", "f", "arg"), _SPARSE_CASES)
def test_sparse_fn_is_jittable(make_fn, f, arg, output_format, assert_trees_allclose):
    """jit(fn) reproduces the eager result for both dense and BCOO outputs."""
    fn = make_fn(f, arg, output_format=output_format)
    assert_trees_allclose(jax.jit(fn)(_X), fn(_X), rtol=1e-6)


@pytest.mark.parametrize(("make_fn", "f", "arg"), _SPARSE_CASES)
def test_sparse_fn_is_vmappable(make_fn, f, arg, output_format, assert_trees_allclose):
    """vmap(fn) over a batch matches the stacked eager per-sample results."""
    fn = make_fn(f, arg, output_format=output_format)
    assert_trees_allclose(jax.vmap(fn)(_BATCH), _stack_batch(fn), rtol=1e-6)


@pytest.mark.parametrize(("make_fn", "f", "arg"), _SPARSE_CASES)
def test_sparse_fn_jit_of_vmap_composes(
    make_fn, f, arg, output_format, assert_trees_allclose
):
    """jit(vmap(fn)) composes for both dense and BCOO outputs."""
    fn = make_fn(f, arg, output_format=output_format)
    assert_trees_allclose(jax.jit(jax.vmap(fn))(_BATCH), _stack_batch(fn), rtol=1e-6)


# Compressed functions


@pytest.mark.parametrize(("make_fn", "f", "arg"), _COMPRESSED_CASES)
def test_compressed_fn_is_jittable(make_fn, f, arg, assert_trees_allclose):
    """jit(fn) reproduces the eager compressed matrix B."""
    fn = make_fn(f, arg)
    assert_trees_allclose(jax.jit(fn)(_X), fn(_X), rtol=1e-6)


@pytest.mark.parametrize(("make_fn", "f", "arg"), _COMPRESSED_CASES)
def test_compressed_fn_is_vmappable(make_fn, f, arg, assert_trees_allclose):
    """vmap(fn) over a batch matches the stacked eager per-sample B."""
    fn = make_fn(f, arg)
    assert_trees_allclose(jax.vmap(fn)(_BATCH), _stack_batch(fn), rtol=1e-6)


@pytest.mark.parametrize(("make_fn", "f", "arg"), _COMPRESSED_CASES)
def test_compressed_fn_jit_of_vmap_composes(make_fn, f, arg, assert_trees_allclose):
    """jit(vmap(fn)) composes for the compressed functions."""
    fn = make_fn(f, arg)
    assert_trees_allclose(jax.jit(jax.vmap(fn))(_BATCH), _stack_batch(fn), rtol=1e-6)


# The output format survives the transform


@pytest.mark.parametrize(
    ("make_fn", "f"),
    [
        pytest.param(jacobian, _f_jac, marks=pytest.mark.jacobian, id="jacobian"),
        pytest.param(hessian, _f_hess, marks=pytest.mark.hessian, id="hessian"),
    ],
)
def test_vmap_preserves_output_format(make_fn, f, output_format):
    """Vmap keeps the requested JAX format: BCOO stays a batched BCOO, dense stays dense."""
    fn = make_fn(f, _X, output_format=output_format)
    out = jax.vmap(fn)(_BATCH)
    expected_shape = (_BATCH.shape[0], *fn(_X).shape)
    match output_format:
        case "bcoo":
            assert isinstance(out, BCOO)
            assert out.n_batch == 1
            assert out.shape == expected_shape
        case "dense":
            assert isinstance(out, jax.Array)
            assert out.shape == expected_shape
        case _:
            pytest.fail(f"Unknown output_format {output_format!r}")
