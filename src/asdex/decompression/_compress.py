"""Compression: validate inputs, run the AD engine, stop at the compressed ``B``.

Stage 1 of ``detect -> color -> decompress``,
where decompress is itself compress (this module) then decompress (gather).
The evaluators here validate the call arguments and dtypes,
short-circuit empty patterns, then call the batched-AD engine in
``differentiation.py`` and return the compressed matrix ``B`` of shape
``(num_colors, dim)`` (plus the forward value and aux).

It also holds the input-prep helpers shared with the composition layer
(``_validate_args``, the cached ``out_struct``/scalar-wrapper memos),
so the one-shot path and the public ``compressed_*`` path prepare inputs the
same way.
This module never imports the decompress side: it stops at ``B``.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from asdex._api_utils import (
    _selected_args,
    _selected_dtype,
    validate_input_dtypes,
    validate_output_dtypes,
)
from asdex.decompression._common import _expected_compressed_dim
from asdex.detection._api import _ensure_scalar, _strip_aux
from asdex.differentiation import _hessian_compressed, _jacobian_compressed
from asdex.pattern import ColoredPattern, SparsityPattern


def _assert_chunk_size(chunk_size: int | None) -> None:
    """Validate chunk_size parameter."""
    if chunk_size is not None and chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")


def _validate_args(args: tuple[Any, ...], sparsity: SparsityPattern) -> None:
    """Check ``args`` match the pattern's declared positional-arg structure."""
    if len(args) != len(sparsity.input_avals):
        raise ValueError(
            f"Expected {len(sparsity.input_avals)} positional argument(s), "
            f"got {len(args)}."
        )
    for i, (arg, aval) in enumerate(zip(args, sparsity.input_avals, strict=True)):
        user_tree = jax.tree_util.tree_structure(arg)
        aval_tree = jax.tree_util.tree_structure(aval)
        if user_tree != aval_tree:
            raise ValueError(
                f"Argument {i} pytree structure {user_tree} does not match "
                f"the colored pattern, which expects {aval_tree}."
            )
        user_leaves = jax.tree_util.tree_leaves(arg)
        aval_leaves = jax.tree_util.tree_leaves(aval)
        for k, (leaf, expected) in enumerate(
            zip(user_leaves, aval_leaves, strict=True)
        ):
            leaf_shape = tuple(getattr(leaf, "shape", ()))
            if leaf_shape != tuple(expected.shape):
                raise ValueError(
                    f"Argument {i} leaf {k} shape {leaf_shape} does not match "
                    f"expected {tuple(expected.shape)}."
                )


# Per-closure call cache
#
# Each public entry point creates one cache dict shared across calls.
# It memoizes work that only depends on the avals of the call arguments:
# output structures from jax.eval_shape (whose cost grows with model size),
# the _ensure_scalar wrapper, and the jitted core for host output formats.
# The cache is bypassed (None) when call-time kwargs or non-traceable
# positional args were bound into f:
# those can change the output structure between calls with identical avals,
# so nothing derived from f may be reused.


def _aval_key(args: tuple[Any, ...]) -> Any:
    """Hashable aval description of ``args``."""
    leaves, treedef = jax.tree_util.tree_flatten(args)
    return treedef, tuple(jax.typeof(leaf) for leaf in leaves)


def _cached_out_struct(
    f_out: Callable[..., Any],
    args: tuple[Any, ...],
    cache: dict[Any, Any] | None,
) -> Any:
    """``jax.eval_shape(f_out, *args)``, memoized on the avals of ``args``."""
    if cache is None:
        return jax.eval_shape(f_out, *args)
    key = ("out_struct", _aval_key(args))
    out_struct = cache.get(key)
    if out_struct is None:
        out_struct = jax.eval_shape(f_out, *args)
        cache[key] = out_struct
    return out_struct


def _cached_scalar_fn(
    f_out: Callable[..., Any],
    sparsity: SparsityPattern,
    cache: dict[Any, Any] | None,
) -> Callable[..., Any]:
    """Memoized ``_ensure_scalar(f_out, sparsity.input_avals)``.

    ``_ensure_scalar`` traces ``f_out`` via ``eval_shape``,
    so the wrapper is reused across calls when caching is allowed.
    """
    if cache is None:
        return _ensure_scalar(f_out, sparsity.input_avals)
    f_scalar = cache.get("f_scalar")
    if f_scalar is None:
        f_scalar = _ensure_scalar(f_out, sparsity.input_avals)
        cache["f_scalar"] = f_scalar
    return f_scalar


def _scalar_with_aux(f: Callable[..., Any]) -> Callable[..., Any]:
    """Aux-preserving counterpart of ``_ensure_scalar``.

    Wraps a ``has_aux=True`` function ``f`` returning ``(out, aux)``
    so the primary output is squeezed to shape ``()``.
    Assumes scalar-squeezability was already validated
    by ``_ensure_scalar`` on the aux-stripped function
    (``jnp.squeeze`` is a no-op for outputs that are already scalar).
    """

    def f_aux(*xs: Any) -> tuple[jax.Array, Any]:
        out, aux = f(*xs)
        return jnp.squeeze(out), aux

    return f_aux


def _cached_scalar_aux_fn(
    f: Callable[..., Any],
    cache: dict[Any, Any] | None,
) -> Callable[..., Any]:
    """Memoized ``_scalar_with_aux(f)``.

    A stable wrapper identity keeps jax's trace caches warm across calls.
    """
    if cache is None:
        return _scalar_with_aux(f)
    f_aux = cache.get("f_scalar_aux")
    if f_aux is None:
        f_aux = _scalar_with_aux(f)
        cache["f_scalar_aux"] = f_aux
    return f_aux


# Empty-pattern compressed matrix


def _empty_compressed(coloring: ColoredPattern, args: tuple[Any, ...]) -> jax.Array:
    """All-zero compressed matrix for empty patterns, dtype-matched to the function.

    Mirrors the non-empty path,
    where the compressed values inherit the selected input leaves' dtype.
    Non-float inputs (``allow_int=True``) map to the default float dtype,
    like the float0 cotangent replacement in ``_flatten_selected_cotangents``.
    The shape ``(num_colors, dim)`` matches the non-empty path,
    possibly with a zero axis when there are no colors or no preserved dimension.
    """
    sparsity = coloring.sparsity
    dtype = _selected_dtype(args, sparsity)
    if not jnp.issubdtype(dtype, jnp.inexact):
        dtype = jnp.float_
    dim = _expected_compressed_dim(coloring)
    return jnp.zeros((coloring.num_colors, dim), dtype=dtype)


# Compress evaluators: stop at B
#
# These back the public ``compressed_*`` / ``value_and_compressed_*`` factories.
# They call the same engine functions the one-shot ``_eval_*`` path uses,
# so there is a single implementation of compression.


def _compress_jacobian(
    f: Callable[..., Any],
    args: tuple[Any, ...],
    coloring: ColoredPattern,
    *,
    has_aux: bool,
    holomorphic: bool,
    allow_int: bool,
    chunk_size: int | None,
    call_cache: dict[Any, Any] | None,
) -> tuple[jax.Array, Any, Any]:
    """Compress the Jacobian of ``f`` at ``args``, returning ``(B, value, aux)``.

    The forward value rides the compression pass, so it is free;
    ``aux`` is ``None`` unless ``has_aux=True``.
    """
    sparsity = coloring.sparsity
    _validate_args(args, sparsity)
    selected = _selected_args(args, sparsity)
    validate_input_dtypes(selected, coloring.mode, holomorphic, allow_int)

    m = sparsity.m
    f_out = _strip_aux(f) if has_aux else f
    out_struct = _cached_out_struct(f_out, args, call_cache)

    if m == 0 or sparsity.nnz == 0:
        compressed = _empty_compressed(coloring, args)
        if has_aux:
            value, aux = f(*args)
            return compressed, value, aux
        return compressed, f(*args), None

    compressed, y, aux = _jacobian_compressed(
        f, args, coloring, out_struct, has_aux=has_aux, chunk_size=chunk_size
    )
    validate_output_dtypes(y, coloring.mode, holomorphic)
    return compressed, y, aux


def _compress_hessian(
    f: Callable[..., Any],
    args: tuple[Any, ...],
    coloring: ColoredPattern,
    *,
    has_aux: bool,
    holomorphic: bool,
    allow_int: bool,
    chunk_size: int | None,
    call_cache: dict[Any, Any] | None,
) -> tuple[jax.Array, Any, Any]:
    """Compress the Hessian of a scalar-valued ``f`` at ``args``.

    Mirrors ``_compress_jacobian``: a single function serves both the value and
    value-free callers, returning ``(B, value, aux)``.
    The value rides the HVP forward pass where the mode allows
    (``rev_over_fwd`` costs one extra ``f`` call),
    so value-free callers simply discard it.
    ``aux`` is ``None`` unless ``has_aux=True``.
    """
    sparsity = coloring.sparsity
    _validate_args(args, sparsity)
    selected = _selected_args(args, sparsity)
    validate_input_dtypes(selected, coloring.mode, holomorphic, allow_int)

    f_scalar_raw = _strip_aux(f) if has_aux else f
    f_scalar = _cached_scalar_fn(f_scalar_raw, sparsity, call_cache)
    out_struct = _cached_out_struct(f_scalar, args, call_cache)
    validate_output_dtypes(out_struct, coloring.mode, holomorphic)

    if sparsity.nnz == 0:
        compressed = _empty_compressed(coloring, args)
        # Compute the value through the squeezing wrappers
        # so it has shape (), consistent with the non-empty path.
        if has_aux:
            out, aux = _cached_scalar_aux_fn(f, call_cache)(*args)
            return compressed, jnp.asarray(out), aux
        return compressed, jnp.asarray(f_scalar(*args)), None

    f_aux = _cached_scalar_aux_fn(f, call_cache) if has_aux else None
    compressed, value, aux = _hessian_compressed(
        f_scalar, args, coloring, chunk_size, f_aux=f_aux
    )
    return compressed, value, aux
