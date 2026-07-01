"""Composition of compress and decompress for the high-level functions.

One worker per AD mode (``_jacobian_with_value``, ``_hessian_with_value``) glues
the two stages together for the one-shot ``jacobian``/``hessian``/``value_and_*``
family:
validate, compute ``B`` via the same engine the public ``compressed_*`` use,
gather it with ``_decompress_data``, then build the pytree/tensor output with
``_build_jacobian``/``_build_hessian``.
Each worker always produces ``(value, aux, matrix)``;
the four public ``_eval_*`` entry points project that triple into the shape their
caller expects (value-free or value-and-matrix).

This glue lives in its own module because it depends on *both* stages,
and folding it into ``_compress.py`` or ``_decompress.py`` would force those two
to import each other and lose their independence.
It also owns the host-format jit-core hack (``_cached_jit_core``):
numpy/scipy outputs cannot be wrapped in a caller-side ``jax.jit``,
so the array-valued core (compress + gather) is jitted internally instead.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp

from asdex._arguments import (
    _selected_args,
    _selected_dtype,
    _validate_args,
    validate_input_dtypes,
    validate_output_dtypes,
)
from asdex._differentiation import _hessian_compressed, _jacobian_compressed
from asdex._pattern import ColoredPattern, SparsityPattern
from asdex._types import _HOST_FORMATS, OutputFormat
from asdex.decompression._compress import (
    _cached_out_struct,
    _cached_scalar_aux_fn,
    _cached_scalar_fn,
    _CallCache,
    _strip_aux,
)
from asdex.decompression._decompress import (
    _build_hessian,
    _build_jacobian,
    _decompress_data,
)


def _empty_data(args: tuple[Any, ...], sparsity: SparsityPattern) -> jax.Array:
    """All-zero data vector for empty patterns, dtype-matched to the function.

    Mirrors the non-empty path,
    where derivative data inherits the selected input leaves' dtype.
    Non-float inputs (``allow_int=True``) map to the default float dtype,
    like the float0 cotangent replacement in ``_flatten_selected_cotangents``.
    """
    dtype = _selected_dtype(args, sparsity)
    if not jnp.issubdtype(dtype, jnp.inexact):
        dtype = jnp.float_
    return jnp.zeros(sparsity.nnz, dtype=dtype)


def _cached_jit_core(
    cache: _CallCache | None,
    output_format: OutputFormat,
    has_aux: bool,
    build: Callable[[], Callable[..., Any]],
) -> Callable[..., Any] | None:
    """Jitted array-valued core for host output formats, memoized per closure.

    Host formats (numpy/scipy) cannot be wrapped in user-side ``jax.jit``,
    so without an internal jit they pay a full re-trace of ``f`` on every call.

    Returns ``None`` when jitting is unsafe:
    call-time kwargs or static args were bound into ``f``
    (``cache is None`` — a fresh closure per call would defeat jit's trace cache),
    or ``has_aux`` is set
    (aux may contain non-JAX types, which cannot be jit outputs).
    """
    if cache is None or has_aux or output_format not in _HOST_FORMATS:
        return None
    core = cache.get("jit_core")
    if core is None:
        core = jax.jit(build())
        cache["jit_core"] = core
    return core


def _build_jacobian_core(
    f: Callable[..., Any],
    coloring: ColoredPattern,
    chunk_size: int | None,
) -> Callable[..., Any]:
    """Array-valued Jacobian core ``args -> (data, y)`` for the internal jit.

    Self-contained so jit re-traces it correctly for new input avals:
    the output structure is recomputed at trace time.
    Only used with ``has_aux=False``.
    """

    def core(*args: Any) -> tuple[jax.Array, Any]:
        out_struct = jax.eval_shape(f, *args)
        compressed, y, _ = _jacobian_compressed(
            f, args, coloring, out_struct, has_aux=False, chunk_size=chunk_size
        )
        return _decompress_data(compressed, coloring), y

    return core


def _build_hessian_core(
    f_scalar: Callable[..., Any],
    coloring: ColoredPattern,
    chunk_size: int | None,
) -> Callable[..., Any]:
    """Array-valued Hessian core ``args -> (data, value)`` for the internal jit.

    Mirrors ``_build_jacobian_core``.
    The value rides the HVP forward pass for free in every mode,
    so the core always returns it and value-free callers discard it.
    """

    def core(*args: Any) -> tuple[jax.Array, jax.Array]:
        compressed, value, _ = _hessian_compressed(f_scalar, args, coloring, chunk_size)
        return _decompress_data(compressed, coloring), value

    return core


# Unified evaluation
#
# Each AD mode has one worker that always produces ``(value, aux, matrix)``;
# the public ``_eval_*`` wrappers below project that triple into the shape their
# caller expects.
# ``need_value`` lets the value-free wrappers skip the forward ``f`` call on the
# empty path, where the value would otherwise be computed only to be discarded
# (the non-empty path computes it for free either way).
# ``aux`` is ``None`` unless ``has_aux``.


def _jacobian_with_value(
    f: Callable[..., Any],
    args: tuple[Any, ...],
    coloring: ColoredPattern,
    output_format: OutputFormat,
    *,
    need_value: bool,
    has_aux: bool,
    holomorphic: bool,
    allow_int: bool,
    chunk_size: int | None,
    call_cache: _CallCache | None,
) -> tuple[Any, Any, Any]:
    """Compute ``(value, aux, jac)`` for the sparse Jacobian of ``f`` at ``args``."""
    sparsity = coloring.sparsity
    _validate_args(args, sparsity)
    selected = _selected_args(args, sparsity)
    validate_input_dtypes(selected, coloring.mode, holomorphic, allow_int)

    m = sparsity.m
    f_out = _strip_aux(f) if has_aux else f
    out_struct = _cached_out_struct(f_out, args, call_cache)

    if m == 0 or sparsity.nnz == 0:
        jac = _build_jacobian(
            _empty_data(args, sparsity), coloring, output_format, out_struct
        )
        if has_aux:
            value, aux = f(*args)
            return value, aux, jac
        value = f(*args) if need_value else None
        return value, None, jac

    core = _cached_jit_core(
        call_cache,
        output_format,
        has_aux,
        lambda: _build_jacobian_core(f, coloring, chunk_size),
    )
    if core is not None:
        data, y = core(*args)
        aux = None
    else:
        compressed, y, aux = _jacobian_compressed(
            f, args, coloring, out_struct, has_aux=has_aux, chunk_size=chunk_size
        )
        data = _decompress_data(compressed, coloring)

    validate_output_dtypes(y, coloring.mode, holomorphic)
    jac = _build_jacobian(data, coloring, output_format, out_struct)
    return y, aux, jac


def _hessian_with_value(
    f: Callable[..., Any],
    args: tuple[Any, ...],
    coloring: ColoredPattern,
    output_format: OutputFormat,
    *,
    need_value: bool,
    has_aux: bool,
    holomorphic: bool,
    allow_int: bool,
    chunk_size: int | None,
    call_cache: _CallCache | None,
) -> tuple[Any, Any, Any]:
    """Compute ``(value, aux, hess)`` for the sparse Hessian of scalar ``f`` at ``args``."""
    sparsity = coloring.sparsity
    _validate_args(args, sparsity)
    selected = _selected_args(args, sparsity)
    validate_input_dtypes(selected, coloring.mode, holomorphic, allow_int)

    f_scalar_raw = _strip_aux(f) if has_aux else f
    f_scalar = _cached_scalar_fn(f_scalar_raw, sparsity, call_cache)
    out_struct = _cached_out_struct(f_scalar, args, call_cache)
    validate_output_dtypes(out_struct, coloring.mode, holomorphic)

    if sparsity.nnz == 0:
        hess = _build_hessian(_empty_data(args, sparsity), coloring, output_format)
        # Compute the value through the squeezing wrappers
        # so it has shape (), consistent with the non-empty path.
        if has_aux:
            out, aux = _cached_scalar_aux_fn(f, call_cache)(*args)
            return (jnp.asarray(out) if need_value else None), aux, hess
        value = jnp.asarray(f_scalar(*args)) if need_value else None
        return value, None, hess

    # On the jitted path, aux is computed by a separate f call below
    # (aux may contain non-JAX types, which cannot be jit outputs),
    # so has_aux does not block the internal jit here.
    core = _cached_jit_core(
        call_cache,
        output_format,
        False,
        lambda: _build_hessian_core(f_scalar, coloring, chunk_size),
    )
    if core is not None:
        data, value = core(*args)
        aux = f(*args)[1] if has_aux else None
    else:
        f_aux = _cached_scalar_aux_fn(f, call_cache) if has_aux else None
        compressed, value, aux = _hessian_compressed(
            f_scalar, args, coloring, chunk_size, f_aux=f_aux
        )
        data = _decompress_data(compressed, coloring)
    hess = _build_hessian(data, coloring, output_format)
    return value, aux, hess


# Public entry points: project the worker's ``(value, aux, matrix)`` triple


def _eval_jacobian(
    f: Callable[..., Any],
    args: tuple[Any, ...],
    coloring: ColoredPattern,
    output_format: OutputFormat,
    *,
    has_aux: bool,
    holomorphic: bool,
    allow_int: bool,
    chunk_size: int | None,
    call_cache: _CallCache | None,
) -> Any:
    """Evaluate the sparse Jacobian of ``f`` at ``args``.

    Returns the block structure by default, ``(jac, aux)`` with ``has_aux=True``.
    """
    _, aux, jac = _jacobian_with_value(
        f,
        args,
        coloring,
        output_format,
        need_value=False,
        has_aux=has_aux,
        holomorphic=holomorphic,
        allow_int=allow_int,
        chunk_size=chunk_size,
        call_cache=call_cache,
    )
    if has_aux:
        return jac, aux
    return jac


def _eval_value_and_jacobian(
    f: Callable[..., Any],
    args: tuple[Any, ...],
    coloring: ColoredPattern,
    output_format: OutputFormat,
    *,
    has_aux: bool,
    holomorphic: bool,
    allow_int: bool,
    chunk_size: int | None,
    call_cache: _CallCache | None,
) -> Any:
    """Evaluate ``f(*args)`` and the sparse Jacobian of ``f`` at ``args``.

    Returns ``(value, jac)`` by default; ``((value, aux), jac)`` when ``has_aux=True``,
    matching ``jax.value_and_grad``'s ordering.
    """
    value, aux, jac = _jacobian_with_value(
        f,
        args,
        coloring,
        output_format,
        need_value=True,
        has_aux=has_aux,
        holomorphic=holomorphic,
        allow_int=allow_int,
        chunk_size=chunk_size,
        call_cache=call_cache,
    )
    if has_aux:
        return (value, aux), jac
    return value, jac


def _eval_hessian(
    f: Callable[..., Any],
    args: tuple[Any, ...],
    coloring: ColoredPattern,
    output_format: OutputFormat,
    *,
    has_aux: bool,
    holomorphic: bool,
    allow_int: bool,
    chunk_size: int | None,
    call_cache: _CallCache | None,
) -> Any:
    """Evaluate the sparse Hessian of a scalar-valued ``f`` at ``args``."""
    _, aux, hess = _hessian_with_value(
        f,
        args,
        coloring,
        output_format,
        need_value=False,
        has_aux=has_aux,
        holomorphic=holomorphic,
        allow_int=allow_int,
        chunk_size=chunk_size,
        call_cache=call_cache,
    )
    if has_aux:
        return hess, aux
    return hess


def _eval_value_and_hessian(
    f: Callable[..., Any],
    args: tuple[Any, ...],
    coloring: ColoredPattern,
    output_format: OutputFormat,
    *,
    has_aux: bool,
    holomorphic: bool,
    allow_int: bool,
    chunk_size: int | None,
    call_cache: _CallCache | None,
) -> Any:
    """Evaluate ``f(*args)`` and the sparse Hessian of a scalar-valued ``f`` at ``args``."""
    value, aux, hess = _hessian_with_value(
        f,
        args,
        coloring,
        output_format,
        need_value=True,
        has_aux=has_aux,
        holomorphic=holomorphic,
        allow_int=allow_int,
        chunk_size=chunk_size,
        call_cache=call_cache,
    )
    if has_aux:
        return (value, aux), hess
    return value, hess
