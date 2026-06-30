"""Composition of compress and decompress for the high-level functions.

The four ``_eval_*`` functions glue the two stages together for the one-shot
``jacobian``/``hessian``/``value_and_*`` family:
validate, compute ``B`` via the same engine the public ``compressed_*`` use,
gather it with ``_decompress_data``, then build the pytree/tensor output with
``_build_jacobian``/``_build_hessian``.

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

from asdex._api_utils import (
    _selected_args,
    _selected_dtype,
    validate_input_dtypes,
    validate_output_dtypes,
)
from asdex.decompression._compress import (
    _cached_out_struct,
    _cached_scalar_aux_fn,
    _cached_scalar_fn,
    _CallCache,
    _strip_aux,
    _validate_args,
)
from asdex.decompression._decompress import (
    _build_hessian,
    _build_jacobian,
    _decompress_data,
)
from asdex.differentiation import _hessian_compressed, _jacobian_compressed
from asdex.modes import OutputFormat
from asdex.pattern import ColoredPattern, SparsityPattern

_HOST_FORMATS = ("numpy_dense", "scipy_coo", "scipy_csr", "scipy_csc")


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
        return _decompress_data(coloring, compressed), y

    return core


def _build_hessian_core(
    f_scalar: Callable[..., Any],
    coloring: ColoredPattern,
    chunk_size: int | None,
) -> Callable[..., Any]:
    """Array-valued Hessian core ``args -> (data, value)`` for the internal jit.

    Mirrors ``_build_jacobian_core``: the value always rides along,
    so value and value-free callers share one core (the latter discard it).
    """

    def core(*args: Any) -> tuple[jax.Array, jax.Array]:
        compressed, value, _ = _hessian_compressed(f_scalar, args, coloring, chunk_size)
        return _decompress_data(coloring, compressed), value

    return core


# Unified evaluation


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
    sparsity = coloring.sparsity
    _validate_args(args, sparsity)
    selected = _selected_args(args, sparsity)
    validate_input_dtypes(selected, coloring.mode, holomorphic, allow_int)

    m = sparsity.m
    f_out = _strip_aux(f) if has_aux else f
    out_struct = _cached_out_struct(f_out, args, call_cache)

    if m == 0 or sparsity.nnz == 0:
        jac = _build_jacobian(
            coloring, _empty_data(args, sparsity), output_format, out_struct
        )
        if has_aux:
            _, aux = f(*args)
            return jac, aux
        return jac

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
        data = _decompress_data(coloring, compressed)

    validate_output_dtypes(y, coloring.mode, holomorphic)
    jac = _build_jacobian(coloring, data, output_format, out_struct)
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
    sparsity = coloring.sparsity
    _validate_args(args, sparsity)
    selected = _selected_args(args, sparsity)
    validate_input_dtypes(selected, coloring.mode, holomorphic, allow_int)

    m = sparsity.m
    f_out = _strip_aux(f) if has_aux else f
    out_struct = _cached_out_struct(f_out, args, call_cache)

    if m == 0 or sparsity.nnz == 0:
        empty = _build_jacobian(
            coloring, _empty_data(args, sparsity), output_format, out_struct
        )
        if has_aux:
            value, aux = f(*args)
            return (value, aux), empty
        value = f(*args)
        return value, empty

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
        data = _decompress_data(coloring, compressed)

    validate_output_dtypes(y, coloring.mode, holomorphic)
    jac = _build_jacobian(coloring, data, output_format, out_struct)
    if has_aux:
        return (y, aux), jac
    return y, jac


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
    sparsity = coloring.sparsity
    _validate_args(args, sparsity)
    selected = _selected_args(args, sparsity)
    validate_input_dtypes(selected, coloring.mode, holomorphic, allow_int)

    f_scalar_raw = _strip_aux(f) if has_aux else f
    f_scalar = _cached_scalar_fn(f_scalar_raw, sparsity, call_cache)
    out_struct = _cached_out_struct(f_scalar, args, call_cache)
    validate_output_dtypes(out_struct, coloring.mode, holomorphic)

    if sparsity.nnz == 0:
        hess = _build_hessian(coloring, _empty_data(args, sparsity), output_format)
        if has_aux:
            _, aux = f(*args)
            return hess, aux
        return hess

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
        data, _value = core(*args)
        aux = f(*args)[1] if has_aux else None
    else:
        f_aux = _cached_scalar_aux_fn(f, call_cache) if has_aux else None
        compressed, _value, aux = _hessian_compressed(
            f_scalar, args, coloring, chunk_size, f_aux=f_aux
        )
        data = _decompress_data(coloring, compressed)
    hess = _build_hessian(coloring, data, output_format)
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
    """Evaluate ``f(*args)`` and the sparse Hessian of ``f`` at ``args``."""
    sparsity = coloring.sparsity
    _validate_args(args, sparsity)
    selected = _selected_args(args, sparsity)
    validate_input_dtypes(selected, coloring.mode, holomorphic, allow_int)

    f_scalar_raw = _strip_aux(f) if has_aux else f
    f_scalar = _cached_scalar_fn(f_scalar_raw, sparsity, call_cache)
    out_struct = _cached_out_struct(f_scalar, args, call_cache)
    validate_output_dtypes(out_struct, coloring.mode, holomorphic)

    if sparsity.nnz == 0:
        empty = _build_hessian(coloring, _empty_data(args, sparsity), output_format)
        # Compute the value through the squeezing wrappers
        # so it has shape (), consistent with the non-empty path.
        if has_aux:
            out, aux = _cached_scalar_aux_fn(f, call_cache)(*args)
            return (jnp.asarray(out), aux), empty
        value = jnp.asarray(f_scalar(*args))
        return value, empty

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
        data = _decompress_data(coloring, compressed)
    hess = _build_hessian(coloring, data, output_format)
    if has_aux:
        return (value, aux), hess
    return value, hess
