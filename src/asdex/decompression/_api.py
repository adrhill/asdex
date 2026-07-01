"""Public API for sparse and compressed Jacobian and Hessian computation.

This module is the user-facing surface of the ``decompression`` package:
the one-shot ``jacobian``/``hessian``/``value_and_*`` family and their
``*_from_coloring`` variants, the ``compressed_*`` and ``value_and_compressed_*``
factories that stop at the compressed matrix ``B``, and the
``decompress``/``decompress_data`` consumers that turn ``B`` back into a sparse
matrix.

Each function is a thin wrapper: it normalizes inputs and delegates the numerics
to the compress, decompress, and evaluate stages.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax

from asdex._arguments import (
    _assert_chunk_size,
    _ensure_index,
    merge_args_kwargs,
    merge_sample_inputs,
)
from asdex._defaults import (
    _DEFAULT_ALLOW_INT,
    _DEFAULT_ARGNUMS,
    _DEFAULT_CHUNK_SIZE,
    _DEFAULT_HAS_AUX,
    _DEFAULT_HOLOMORPHIC,
    _DEFAULT_MODE,
    _DEFAULT_OUTPUT_FORMAT,
    _DEFAULT_SYMMETRIC_HESSIAN,
    _DEFAULT_SYMMETRIC_JACOBIAN,
)
from asdex._docstrings import _fill_doc
from asdex._modes import (
    HessianMode,
    JacobianMode,
    OutputFormat,
    _assert_output_format,
)
from asdex._pattern import ColoredPattern
from asdex.coloring import hessian_coloring as _hessian_coloring
from asdex.coloring import jacobian_coloring as _jacobian_coloring
from asdex.decompression._compress import (
    _CallCache,
    _compress_hessian,
    _compress_jacobian,
)
from asdex.decompression._decompress import (
    _decompress_data,
    _decompress_to_format,
    _validate_compressed,
)
from asdex.decompression._evaluate import (
    _eval_hessian,
    _eval_jacobian,
    _eval_value_and_hessian,
    _eval_value_and_jacobian,
)

# Public API: one-shot entry points


@_fill_doc
def jacobian(
    f: Callable[..., Any],
    *sample_args: Any,
    argnums: int | Sequence[int] = _DEFAULT_ARGNUMS,
    has_aux: bool = _DEFAULT_HAS_AUX,
    holomorphic: bool = _DEFAULT_HOLOMORPHIC,
    allow_int: bool = _DEFAULT_ALLOW_INT,
    mode: JacobianMode | None = _DEFAULT_MODE,
    symmetric: bool = _DEFAULT_SYMMETRIC_JACOBIAN,
    output_format: OutputFormat = _DEFAULT_OUTPUT_FORMAT,
    chunk_size: int | None = _DEFAULT_CHUNK_SIZE,
    **sample_kwargs: Any,
) -> Callable[..., Any]:
    """Detect sparsity, color, and return a function computing sparse Jacobians.

    Combines [`jacobian_coloring`][asdex.jacobian_coloring]
    and [`jacobian_from_coloring`][asdex.jacobian_from_coloring]
    in one call.

    {jit}

    Args:
        f: {f_jac}
        *sample_args: {sample_args}
        argnums: {argnums}
        has_aux: Whether ``f`` returns ``(output, auxiliary_data)``,
            mirroring ``jax.jacrev``.
            When True, the returned function yields ``(jac, aux)``.
        holomorphic: Whether ``f`` is promised to be holomorphic,
            mirroring ``jax.jacrev``.
            Validates dtype compatibility at call time.
        allow_int: Whether to allow differentiating with respect to
            integer-valued inputs, mirroring ``jax.jacrev``.
        mode: AD mode.
            ``"fwd"`` uses JVPs (forward-mode AD),
            ``"rev"`` uses VJPs (reverse-mode AD).
            ``None`` picks whichever of fwd/rev needs fewer colors.
        symmetric: Whether to use symmetric (star) coloring.
            Requires a square Jacobian.
        output_format: {format_jac}
        chunk_size: {chunk_size}
        **sample_kwargs: {sample_kwargs}

    Returns:
        A function that takes the same positional args as ``f`` and returns
            a pytree of Jacobian blocks matching ``argnums``, with each leaf
            shaped ``(*out_shape, *in_leaf_shape)``.
            The block type depends on ``output_format``
            (``jax.experimental.sparse.BCOO`` by default, or ``jax.Array``
            when ``"dense"``).
    """
    _assert_output_format(output_format)
    _assert_chunk_size(chunk_size)
    argnums = _ensure_index(argnums)
    args, f_detect, remapped_argnums = merge_sample_inputs(
        f, sample_args, sample_kwargs, argnums
    )
    coloring = _jacobian_coloring(
        f_detect,
        *args,
        argnums=remapped_argnums,
        has_aux=has_aux,
        mode=mode,
        symmetric=symmetric,
    )

    call_cache: _CallCache = {}

    def jac_fn(*call_args: Any, **kwargs: Any) -> Any:
        expected_nargs = len(coloring.sparsity.input_avals)
        merged_args, f_bound = merge_args_kwargs(f, call_args, kwargs, expected_nargs)
        return _eval_jacobian(
            f_bound,
            merged_args,
            coloring,
            output_format,
            has_aux=has_aux,
            holomorphic=holomorphic,
            allow_int=allow_int,
            chunk_size=chunk_size,
            call_cache=call_cache if f_bound is f else None,
        )

    return jac_fn


@_fill_doc
def value_and_jacobian(
    f: Callable[..., Any],
    *sample_args: Any,
    argnums: int | Sequence[int] = _DEFAULT_ARGNUMS,
    has_aux: bool = _DEFAULT_HAS_AUX,
    holomorphic: bool = _DEFAULT_HOLOMORPHIC,
    allow_int: bool = _DEFAULT_ALLOW_INT,
    mode: JacobianMode | None = _DEFAULT_MODE,
    symmetric: bool = _DEFAULT_SYMMETRIC_JACOBIAN,
    output_format: OutputFormat = _DEFAULT_OUTPUT_FORMAT,
    chunk_size: int | None = _DEFAULT_CHUNK_SIZE,
    **sample_kwargs: Any,
) -> Callable[..., Any]:
    """Detect sparsity, color, and return a function computing value and sparse Jacobian.

    Like [`jacobian`][asdex.jacobian],
    but also returns the primal value ``f(*args)``
    without an extra forward pass.

    {jit}

    Returns:
        A function that takes the same positional args as ``f`` and returns
            ``(value, jac)`` — or ``((value, aux), jac)`` when ``has_aux=True``,
            matching ``jax.value_and_grad`` ordering.
    """
    _assert_output_format(output_format)
    _assert_chunk_size(chunk_size)
    argnums = _ensure_index(argnums)
    args, f_detect, remapped_argnums = merge_sample_inputs(
        f, sample_args, sample_kwargs, argnums
    )
    coloring = _jacobian_coloring(
        f_detect,
        *args,
        argnums=remapped_argnums,
        has_aux=has_aux,
        mode=mode,
        symmetric=symmetric,
    )

    call_cache: _CallCache = {}

    def val_jac_fn(*call_args: Any, **kwargs: Any) -> Any:
        expected_nargs = len(coloring.sparsity.input_avals)
        merged_args, f_bound = merge_args_kwargs(f, call_args, kwargs, expected_nargs)
        return _eval_value_and_jacobian(
            f_bound,
            merged_args,
            coloring,
            output_format,
            has_aux=has_aux,
            holomorphic=holomorphic,
            allow_int=allow_int,
            chunk_size=chunk_size,
            call_cache=call_cache if f_bound is f else None,
        )

    return val_jac_fn


@_fill_doc
def hessian(
    f: Callable[..., Any],
    *sample_args: Any,
    argnums: int | Sequence[int] = _DEFAULT_ARGNUMS,
    has_aux: bool = _DEFAULT_HAS_AUX,
    holomorphic: bool = _DEFAULT_HOLOMORPHIC,
    allow_int: bool = _DEFAULT_ALLOW_INT,
    mode: HessianMode | None = _DEFAULT_MODE,
    symmetric: bool = _DEFAULT_SYMMETRIC_HESSIAN,
    output_format: OutputFormat = _DEFAULT_OUTPUT_FORMAT,
    chunk_size: int | None = _DEFAULT_CHUNK_SIZE,
    **sample_kwargs: Any,
) -> Callable[..., Any]:
    """Detect sparsity, color, and return a function computing sparse Hessians.

    If ``f`` returns a squeezable shape like ``(1,)`` or ``(1, 1)``,
    it is automatically squeezed to scalar.

    {jit}

    Args:
        f: {f_hess}
        *sample_args: {sample_args}
        argnums: {argnums}
        has_aux: {has_aux}
        holomorphic: {holomorphic}
        allow_int: {allow_int_hess}
        mode: {mode_hess}
        symmetric: {symmetric}
        output_format: {format_hess}
        chunk_size: {chunk_size}
        **sample_kwargs: {sample_kwargs}

    Returns:
        A function that takes the same positional args as ``f`` and returns
            the sparse Hessian.
    """
    _assert_output_format(output_format)
    _assert_chunk_size(chunk_size)
    argnums = _ensure_index(argnums)
    args, f_detect, remapped_argnums = merge_sample_inputs(
        f, sample_args, sample_kwargs, argnums
    )
    coloring = _hessian_coloring(
        f_detect,
        *args,
        argnums=remapped_argnums,
        has_aux=has_aux,
        mode=mode,
        symmetric=symmetric,
    )

    call_cache: _CallCache = {}

    def hess_fn(*call_args: Any, **kwargs: Any) -> Any:
        expected_nargs = len(coloring.sparsity.input_avals)
        merged_args, f_bound = merge_args_kwargs(f, call_args, kwargs, expected_nargs)
        return _eval_hessian(
            f_bound,
            merged_args,
            coloring,
            output_format,
            has_aux=has_aux,
            holomorphic=holomorphic,
            allow_int=allow_int,
            chunk_size=chunk_size,
            call_cache=call_cache if f_bound is f else None,
        )

    return hess_fn


@_fill_doc
def value_and_hessian(
    f: Callable[..., Any],
    *sample_args: Any,
    argnums: int | Sequence[int] = _DEFAULT_ARGNUMS,
    has_aux: bool = _DEFAULT_HAS_AUX,
    holomorphic: bool = _DEFAULT_HOLOMORPHIC,
    allow_int: bool = _DEFAULT_ALLOW_INT,
    mode: HessianMode | None = _DEFAULT_MODE,
    symmetric: bool = _DEFAULT_SYMMETRIC_HESSIAN,
    output_format: OutputFormat = _DEFAULT_OUTPUT_FORMAT,
    chunk_size: int | None = _DEFAULT_CHUNK_SIZE,
    **sample_kwargs: Any,
) -> Callable[..., Any]:
    """Detect sparsity, color, and return a function computing value and sparse Hessian.

    Like [`hessian`][asdex.hessian], but also returns the primal value
    ``f(*args)`` without an extra forward pass.

    {jit}

    Args:
        f: {f_hess}
        *sample_args: {sample_args}
        argnums: {argnums}
        has_aux: {has_aux}
        holomorphic: {holomorphic}
        allow_int: {allow_int_hess}
        mode: {mode_hess}
        symmetric: {symmetric}
        output_format: {format_hess}
        chunk_size: {chunk_size}
        **sample_kwargs: {sample_kwargs}

    Returns:
        A function that takes the same positional args as ``f`` and returns
            ``(value, hessian)``.
    """
    _assert_output_format(output_format)
    _assert_chunk_size(chunk_size)
    argnums = _ensure_index(argnums)
    args, f_detect, remapped_argnums = merge_sample_inputs(
        f, sample_args, sample_kwargs, argnums
    )
    coloring = _hessian_coloring(
        f_detect,
        *args,
        argnums=remapped_argnums,
        has_aux=has_aux,
        mode=mode,
        symmetric=symmetric,
    )

    call_cache: _CallCache = {}

    def val_hess_fn(*call_args: Any, **kwargs: Any) -> Any:
        expected_nargs = len(coloring.sparsity.input_avals)
        merged_args, f_bound = merge_args_kwargs(f, call_args, kwargs, expected_nargs)
        return _eval_value_and_hessian(
            f_bound,
            merged_args,
            coloring,
            output_format,
            has_aux=has_aux,
            holomorphic=holomorphic,
            allow_int=allow_int,
            chunk_size=chunk_size,
            call_cache=call_cache if f_bound is f else None,
        )

    return val_hess_fn


# Public API: ``*_from_coloring`` entry points


@_fill_doc
def jacobian_from_coloring(
    f: Callable[..., Any],
    coloring: ColoredPattern,
    output_format: OutputFormat = _DEFAULT_OUTPUT_FORMAT,
    *,
    has_aux: bool = _DEFAULT_HAS_AUX,
    holomorphic: bool = _DEFAULT_HOLOMORPHIC,
    allow_int: bool = _DEFAULT_ALLOW_INT,
    chunk_size: int | None = _DEFAULT_CHUNK_SIZE,
) -> Callable[..., Any]:
    """Build a sparse Jacobian function from a pre-computed coloring.

    Uses row coloring + VJPs or column coloring + JVPs,
    depending on which needs fewer colors.

    The returned callable accepts ``*args, **kwargs``; kwargs are forwarded
    to ``f`` at call time (matching ``jax.jacfwd`` / ``jax.jacrev``).

    {jit}

    Args:
        f: {f_jac}
        coloring: {coloring}
        output_format: {format_jac}
        has_aux: {has_aux}
        holomorphic: {holomorphic}
        allow_int: {allow_int_jac}
        chunk_size: {chunk_size}
    """
    _assert_output_format(output_format)
    _assert_chunk_size(chunk_size)

    call_cache: _CallCache = {}

    def jac_fn(*args: Any, **kwargs: Any) -> Any:
        expected_nargs = len(coloring.sparsity.input_avals)
        merged_args, f_bound = merge_args_kwargs(f, args, kwargs, expected_nargs)
        return _eval_jacobian(
            f_bound,
            merged_args,
            coloring,
            output_format,
            has_aux=has_aux,
            holomorphic=holomorphic,
            allow_int=allow_int,
            chunk_size=chunk_size,
            call_cache=call_cache if f_bound is f else None,
        )

    return jac_fn


@_fill_doc
def hessian_from_coloring(
    f: Callable[..., Any],
    coloring: ColoredPattern,
    output_format: OutputFormat = _DEFAULT_OUTPUT_FORMAT,
    *,
    has_aux: bool = _DEFAULT_HAS_AUX,
    holomorphic: bool = _DEFAULT_HOLOMORPHIC,
    allow_int: bool = _DEFAULT_ALLOW_INT,
    chunk_size: int | None = _DEFAULT_CHUNK_SIZE,
) -> Callable[..., Any]:
    """Build a sparse Hessian function from a pre-computed coloring.

    Uses symmetric (star) coloring and Hessian-vector products by default.

    {jit}

    Args:
        f: {f_hess}
        coloring: {coloring}
        output_format: {format_hess}
        has_aux: {has_aux}
        holomorphic: {holomorphic}
        allow_int: {allow_int_hess}
        chunk_size: {chunk_size}
    """
    _assert_output_format(output_format)
    _assert_chunk_size(chunk_size)

    call_cache: _CallCache = {}

    def hess_fn(*args: Any, **kwargs: Any) -> Any:
        expected_nargs = len(coloring.sparsity.input_avals)
        merged_args, f_bound = merge_args_kwargs(f, args, kwargs, expected_nargs)
        return _eval_hessian(
            f_bound,
            merged_args,
            coloring,
            output_format,
            has_aux=has_aux,
            holomorphic=holomorphic,
            allow_int=allow_int,
            chunk_size=chunk_size,
            call_cache=call_cache if f_bound is f else None,
        )

    return hess_fn


@_fill_doc
def value_and_jacobian_from_coloring(
    f: Callable[..., Any],
    coloring: ColoredPattern,
    output_format: OutputFormat = _DEFAULT_OUTPUT_FORMAT,
    *,
    has_aux: bool = _DEFAULT_HAS_AUX,
    holomorphic: bool = _DEFAULT_HOLOMORPHIC,
    allow_int: bool = _DEFAULT_ALLOW_INT,
    chunk_size: int | None = _DEFAULT_CHUNK_SIZE,
) -> Callable[..., Any]:
    """Build a function computing value and sparse Jacobian from a pre-computed coloring.

    {jit}

    Args:
        f: {f_jac}
        coloring: {coloring}
        output_format: {format_jac}
        has_aux: {has_aux}
        holomorphic: {holomorphic}
        allow_int: {allow_int_jac}
        chunk_size: {chunk_size}
    """
    _assert_output_format(output_format)
    _assert_chunk_size(chunk_size)

    call_cache: _CallCache = {}

    def val_jac_fn(*args: Any, **kwargs: Any) -> Any:
        expected_nargs = len(coloring.sparsity.input_avals)
        merged_args, f_bound = merge_args_kwargs(f, args, kwargs, expected_nargs)
        return _eval_value_and_jacobian(
            f_bound,
            merged_args,
            coloring,
            output_format,
            has_aux=has_aux,
            holomorphic=holomorphic,
            allow_int=allow_int,
            chunk_size=chunk_size,
            call_cache=call_cache if f_bound is f else None,
        )

    return val_jac_fn


@_fill_doc
def value_and_hessian_from_coloring(
    f: Callable[..., Any],
    coloring: ColoredPattern,
    output_format: OutputFormat = _DEFAULT_OUTPUT_FORMAT,
    *,
    has_aux: bool = _DEFAULT_HAS_AUX,
    holomorphic: bool = _DEFAULT_HOLOMORPHIC,
    allow_int: bool = _DEFAULT_ALLOW_INT,
    chunk_size: int | None = _DEFAULT_CHUNK_SIZE,
) -> Callable[..., Any]:
    """Build a function computing value and sparse Hessian from a pre-computed coloring.

    {jit}

    Args:
        f: {f_hess}
        coloring: {coloring}
        output_format: {format_hess}
        has_aux: {has_aux}
        holomorphic: {holomorphic}
        allow_int: {allow_int_hess}
        chunk_size: {chunk_size}
    """
    _assert_output_format(output_format)
    _assert_chunk_size(chunk_size)

    call_cache: _CallCache = {}

    def val_hess_fn(*args: Any, **kwargs: Any) -> Any:
        expected_nargs = len(coloring.sparsity.input_avals)
        merged_args, f_bound = merge_args_kwargs(f, args, kwargs, expected_nargs)
        return _eval_value_and_hessian(
            f_bound,
            merged_args,
            coloring,
            output_format,
            has_aux=has_aux,
            holomorphic=holomorphic,
            allow_int=allow_int,
            chunk_size=chunk_size,
            call_cache=call_cache if f_bound is f else None,
        )

    return val_hess_fn


# Public API: compressed entry points
#
# These stop at the compressed matrix B of shape (num_colors, dim),
# one VJP/JVP/HVP per color, before decompression scatters B into the pattern.
# They take no output_format: formatting is the job of decompress.


def compressed_jacobian(
    f: Callable[..., Any],
    *sample_args: Any,
    argnums: int | Sequence[int] = _DEFAULT_ARGNUMS,
    has_aux: bool = _DEFAULT_HAS_AUX,
    holomorphic: bool = _DEFAULT_HOLOMORPHIC,
    allow_int: bool = _DEFAULT_ALLOW_INT,
    mode: JacobianMode | None = _DEFAULT_MODE,
    symmetric: bool = _DEFAULT_SYMMETRIC_JACOBIAN,
    chunk_size: int | None = _DEFAULT_CHUNK_SIZE,
    **sample_kwargs: Any,
) -> Callable[..., Any]:
    """Detect sparsity, color, and return a function computing the compressed Jacobian.

    Runs the same detect-and-color steps as [`jacobian`][asdex.jacobian],
    but stops at the dense compressed matrix ``B`` of shape ``(num_colors, dim)``:
    one VJP/JVP per color, before decompression scatters ``B`` into the pattern.
    Recover the sparse matrix with [`decompress`][asdex.decompress] or
    [`decompress_data`][asdex.decompress_data],
    or work with ``B`` directly (custom solvers, cross-checks, debugging).

    The returned ``B`` is a plain ``jax.Array``,
    so the returned function is jit-able by the caller.

    See [`jacobian`][asdex.jacobian] for the shared arguments
    (``argnums``, ``has_aux``, ``holomorphic``, ``allow_int``, ``mode``,
    ``symmetric``, ``chunk_size``, and the sample inputs).
    Unlike [`jacobian`][asdex.jacobian], it takes no ``output_format``:
    formatting is the job of [`decompress`][asdex.decompress].

    Returns:
        A function that takes the same positional args as ``f`` and returns
            the compressed matrix ``B`` of shape ``(num_colors, dim)``,
            or ``(B, aux)`` when ``has_aux=True``.
            ``dim`` is the flattened size of the differentiated inputs
            (the leaves selected by ``argnums``) in ``"rev"`` mode,
            and the flattened size of ``f``'s output in ``"fwd"`` mode.
    """
    _assert_chunk_size(chunk_size)
    argnums = _ensure_index(argnums)
    args, f_detect, remapped_argnums = merge_sample_inputs(
        f, sample_args, sample_kwargs, argnums
    )
    coloring = _jacobian_coloring(
        f_detect,
        *args,
        argnums=remapped_argnums,
        has_aux=has_aux,
        mode=mode,
        symmetric=symmetric,
    )

    call_cache: _CallCache = {}

    def compressed_fn(*call_args: Any, **kwargs: Any) -> Any:
        expected_nargs = len(coloring.sparsity.input_avals)
        merged_args, f_bound = merge_args_kwargs(f, call_args, kwargs, expected_nargs)
        compressed, _value, aux = _compress_jacobian(
            f_bound,
            merged_args,
            coloring,
            has_aux=has_aux,
            holomorphic=holomorphic,
            allow_int=allow_int,
            chunk_size=chunk_size,
            call_cache=call_cache if f_bound is f else None,
            need_value=False,
        )
        return (compressed, aux) if has_aux else compressed

    return compressed_fn


def compressed_jacobian_from_coloring(
    f: Callable[..., Any],
    coloring: ColoredPattern,
    *,
    has_aux: bool = _DEFAULT_HAS_AUX,
    holomorphic: bool = _DEFAULT_HOLOMORPHIC,
    allow_int: bool = _DEFAULT_ALLOW_INT,
    chunk_size: int | None = _DEFAULT_CHUNK_SIZE,
) -> Callable[..., Any]:
    """Build a compressed Jacobian function from a pre-computed coloring.

    Like [`jacobian_from_coloring`][asdex.jacobian_from_coloring],
    but stops at the compressed matrix ``B`` of shape ``(num_colors, dim)``
    instead of materializing the sparse matrix.
    See [`compressed_jacobian`][asdex.compressed_jacobian] for ``B``'s layout
    and [`jacobian_from_coloring`][asdex.jacobian_from_coloring]
    for the shared arguments.

    Returns:
        A function returning ``B`` of shape ``(num_colors, dim)``,
            or ``(B, aux)`` when ``has_aux=True``.
    """
    _assert_chunk_size(chunk_size)

    call_cache: _CallCache = {}

    def compressed_fn(*args: Any, **kwargs: Any) -> Any:
        expected_nargs = len(coloring.sparsity.input_avals)
        merged_args, f_bound = merge_args_kwargs(f, args, kwargs, expected_nargs)
        compressed, _value, aux = _compress_jacobian(
            f_bound,
            merged_args,
            coloring,
            has_aux=has_aux,
            holomorphic=holomorphic,
            allow_int=allow_int,
            chunk_size=chunk_size,
            call_cache=call_cache if f_bound is f else None,
            need_value=False,
        )
        return (compressed, aux) if has_aux else compressed

    return compressed_fn


def compressed_hessian(
    f: Callable[..., Any],
    *sample_args: Any,
    argnums: int | Sequence[int] = _DEFAULT_ARGNUMS,
    has_aux: bool = _DEFAULT_HAS_AUX,
    holomorphic: bool = _DEFAULT_HOLOMORPHIC,
    allow_int: bool = _DEFAULT_ALLOW_INT,
    mode: HessianMode | None = _DEFAULT_MODE,
    symmetric: bool = _DEFAULT_SYMMETRIC_HESSIAN,
    chunk_size: int | None = _DEFAULT_CHUNK_SIZE,
    **sample_kwargs: Any,
) -> Callable[..., Any]:
    """Detect sparsity, color, and return a function computing the compressed Hessian.

    Runs the same detect-and-color steps as [`hessian`][asdex.hessian],
    but stops at the dense compressed matrix ``B`` of shape ``(num_colors, n)``:
    one HVP per color, before decompression scatters ``B`` into the pattern.
    ``n`` is the flattened size of the differentiated inputs
    (the leaves selected by ``argnums``), so the Hessian is ``(n, n)``.
    Recover the sparse matrix with [`decompress`][asdex.decompress] or
    [`decompress_data`][asdex.decompress_data],
    or work with ``B`` directly.

    The returned ``B`` is a plain ``jax.Array``,
    so the returned function is jit-able by the caller.

    See [`hessian`][asdex.hessian] for the shared arguments
    (``argnums``, ``has_aux``, ``holomorphic``, ``allow_int``, ``mode``,
    ``symmetric``, ``chunk_size``, and the sample inputs).
    Unlike [`hessian`][asdex.hessian], it takes no ``output_format``:
    formatting is the job of [`decompress`][asdex.decompress].

    Returns:
        A function that takes the same positional args as ``f`` and returns
            the compressed matrix ``B`` of shape ``(num_colors, n)``,
            or ``(B, aux)`` when ``has_aux=True``.
    """
    _assert_chunk_size(chunk_size)
    argnums = _ensure_index(argnums)
    args, f_detect, remapped_argnums = merge_sample_inputs(
        f, sample_args, sample_kwargs, argnums
    )
    coloring = _hessian_coloring(
        f_detect,
        *args,
        argnums=remapped_argnums,
        has_aux=has_aux,
        mode=mode,
        symmetric=symmetric,
    )

    call_cache: _CallCache = {}

    def compressed_fn(*call_args: Any, **kwargs: Any) -> Any:
        expected_nargs = len(coloring.sparsity.input_avals)
        merged_args, f_bound = merge_args_kwargs(f, call_args, kwargs, expected_nargs)
        compressed, _value, aux = _compress_hessian(
            f_bound,
            merged_args,
            coloring,
            has_aux=has_aux,
            holomorphic=holomorphic,
            allow_int=allow_int,
            chunk_size=chunk_size,
            call_cache=call_cache if f_bound is f else None,
            need_value=False,
        )
        return (compressed, aux) if has_aux else compressed

    return compressed_fn


def compressed_hessian_from_coloring(
    f: Callable[..., Any],
    coloring: ColoredPattern,
    *,
    has_aux: bool = _DEFAULT_HAS_AUX,
    holomorphic: bool = _DEFAULT_HOLOMORPHIC,
    allow_int: bool = _DEFAULT_ALLOW_INT,
    chunk_size: int | None = _DEFAULT_CHUNK_SIZE,
) -> Callable[..., Any]:
    """Build a compressed Hessian function from a pre-computed coloring.

    Like [`hessian_from_coloring`][asdex.hessian_from_coloring],
    but stops at the compressed matrix ``B`` of shape ``(num_colors, n)``
    instead of materializing the sparse matrix.
    See [`compressed_hessian`][asdex.compressed_hessian] for ``B``'s layout
    and [`hessian_from_coloring`][asdex.hessian_from_coloring]
    for the shared arguments.

    Returns:
        A function returning ``B`` of shape ``(num_colors, n)``,
            or ``(B, aux)`` when ``has_aux=True``.
    """
    _assert_chunk_size(chunk_size)

    call_cache: _CallCache = {}

    def compressed_fn(*args: Any, **kwargs: Any) -> Any:
        expected_nargs = len(coloring.sparsity.input_avals)
        merged_args, f_bound = merge_args_kwargs(f, args, kwargs, expected_nargs)
        compressed, _value, aux = _compress_hessian(
            f_bound,
            merged_args,
            coloring,
            has_aux=has_aux,
            holomorphic=holomorphic,
            allow_int=allow_int,
            chunk_size=chunk_size,
            call_cache=call_cache if f_bound is f else None,
            need_value=False,
        )
        return (compressed, aux) if has_aux else compressed

    return compressed_fn


def value_and_compressed_jacobian(
    f: Callable[..., Any],
    *sample_args: Any,
    argnums: int | Sequence[int] = _DEFAULT_ARGNUMS,
    has_aux: bool = _DEFAULT_HAS_AUX,
    holomorphic: bool = _DEFAULT_HOLOMORPHIC,
    allow_int: bool = _DEFAULT_ALLOW_INT,
    mode: JacobianMode | None = _DEFAULT_MODE,
    symmetric: bool = _DEFAULT_SYMMETRIC_JACOBIAN,
    chunk_size: int | None = _DEFAULT_CHUNK_SIZE,
    **sample_kwargs: Any,
) -> Callable[..., Any]:
    """Like [`compressed_jacobian`][asdex.compressed_jacobian], also returning the value.

    The primal value ``f(*args)`` rides the compression forward pass,
    so it is nearly free.
    See [`compressed_jacobian`][asdex.compressed_jacobian] for ``B``'s layout
    and [`jacobian`][asdex.jacobian] for the shared arguments.

    Returns:
        A function that takes the same positional args as ``f`` and returns
            ``(value, B)`` — or ``((value, aux), B)`` when ``has_aux=True``,
            matching ``jax.value_and_grad`` ordering.
    """
    _assert_chunk_size(chunk_size)
    argnums = _ensure_index(argnums)
    args, f_detect, remapped_argnums = merge_sample_inputs(
        f, sample_args, sample_kwargs, argnums
    )
    coloring = _jacobian_coloring(
        f_detect,
        *args,
        argnums=remapped_argnums,
        has_aux=has_aux,
        mode=mode,
        symmetric=symmetric,
    )

    call_cache: _CallCache = {}

    def compressed_fn(*call_args: Any, **kwargs: Any) -> Any:
        expected_nargs = len(coloring.sparsity.input_avals)
        merged_args, f_bound = merge_args_kwargs(f, call_args, kwargs, expected_nargs)
        compressed, value, aux = _compress_jacobian(
            f_bound,
            merged_args,
            coloring,
            has_aux=has_aux,
            holomorphic=holomorphic,
            allow_int=allow_int,
            chunk_size=chunk_size,
            call_cache=call_cache if f_bound is f else None,
        )
        return ((value, aux), compressed) if has_aux else (value, compressed)

    return compressed_fn


def value_and_compressed_jacobian_from_coloring(
    f: Callable[..., Any],
    coloring: ColoredPattern,
    *,
    has_aux: bool = _DEFAULT_HAS_AUX,
    holomorphic: bool = _DEFAULT_HOLOMORPHIC,
    allow_int: bool = _DEFAULT_ALLOW_INT,
    chunk_size: int | None = _DEFAULT_CHUNK_SIZE,
) -> Callable[..., Any]:
    """Value and compressed Jacobian from a pre-computed coloring.

    Like [`value_and_jacobian_from_coloring`][asdex.value_and_jacobian_from_coloring],
    but stops at the compressed matrix ``B``.
    See [`compressed_jacobian`][asdex.compressed_jacobian] for ``B``'s layout.

    Returns:
        A function returning ``(value, B)``,
            or ``((value, aux), B)`` when ``has_aux=True``.
    """
    _assert_chunk_size(chunk_size)

    call_cache: _CallCache = {}

    def compressed_fn(*args: Any, **kwargs: Any) -> Any:
        expected_nargs = len(coloring.sparsity.input_avals)
        merged_args, f_bound = merge_args_kwargs(f, args, kwargs, expected_nargs)
        compressed, value, aux = _compress_jacobian(
            f_bound,
            merged_args,
            coloring,
            has_aux=has_aux,
            holomorphic=holomorphic,
            allow_int=allow_int,
            chunk_size=chunk_size,
            call_cache=call_cache if f_bound is f else None,
        )
        return ((value, aux), compressed) if has_aux else (value, compressed)

    return compressed_fn


def value_and_compressed_hessian(
    f: Callable[..., Any],
    *sample_args: Any,
    argnums: int | Sequence[int] = _DEFAULT_ARGNUMS,
    has_aux: bool = _DEFAULT_HAS_AUX,
    holomorphic: bool = _DEFAULT_HOLOMORPHIC,
    allow_int: bool = _DEFAULT_ALLOW_INT,
    mode: HessianMode | None = _DEFAULT_MODE,
    symmetric: bool = _DEFAULT_SYMMETRIC_HESSIAN,
    chunk_size: int | None = _DEFAULT_CHUNK_SIZE,
    **sample_kwargs: Any,
) -> Callable[..., Any]:
    """Like [`compressed_hessian`][asdex.compressed_hessian], also returning the value.

    The primal value ``f(*args)`` rides the HVP forward pass,
    so it is nearly free.
    See [`compressed_hessian`][asdex.compressed_hessian] for ``B``'s layout
    and [`hessian`][asdex.hessian] for the shared arguments.

    Returns:
        A function that takes the same positional args as ``f`` and returns
            ``(value, B)`` — or ``((value, aux), B)`` when ``has_aux=True``.
    """
    _assert_chunk_size(chunk_size)
    argnums = _ensure_index(argnums)
    args, f_detect, remapped_argnums = merge_sample_inputs(
        f, sample_args, sample_kwargs, argnums
    )
    coloring = _hessian_coloring(
        f_detect,
        *args,
        argnums=remapped_argnums,
        has_aux=has_aux,
        mode=mode,
        symmetric=symmetric,
    )

    call_cache: _CallCache = {}

    def compressed_fn(*call_args: Any, **kwargs: Any) -> Any:
        expected_nargs = len(coloring.sparsity.input_avals)
        merged_args, f_bound = merge_args_kwargs(f, call_args, kwargs, expected_nargs)
        compressed, value, aux = _compress_hessian(
            f_bound,
            merged_args,
            coloring,
            has_aux=has_aux,
            holomorphic=holomorphic,
            allow_int=allow_int,
            chunk_size=chunk_size,
            call_cache=call_cache if f_bound is f else None,
        )
        return ((value, aux), compressed) if has_aux else (value, compressed)

    return compressed_fn


def value_and_compressed_hessian_from_coloring(
    f: Callable[..., Any],
    coloring: ColoredPattern,
    *,
    has_aux: bool = _DEFAULT_HAS_AUX,
    holomorphic: bool = _DEFAULT_HOLOMORPHIC,
    allow_int: bool = _DEFAULT_ALLOW_INT,
    chunk_size: int | None = _DEFAULT_CHUNK_SIZE,
) -> Callable[..., Any]:
    """Value and compressed Hessian from a pre-computed coloring.

    Like [`value_and_hessian_from_coloring`][asdex.value_and_hessian_from_coloring],
    but stops at the compressed matrix ``B``.
    See [`compressed_hessian`][asdex.compressed_hessian] for ``B``'s layout.

    Returns:
        A function returning ``(value, B)``,
            or ``((value, aux), B)`` when ``has_aux=True``.
    """
    _assert_chunk_size(chunk_size)

    call_cache: _CallCache = {}

    def compressed_fn(*args: Any, **kwargs: Any) -> Any:
        expected_nargs = len(coloring.sparsity.input_avals)
        merged_args, f_bound = merge_args_kwargs(f, args, kwargs, expected_nargs)
        compressed, value, aux = _compress_hessian(
            f_bound,
            merged_args,
            coloring,
            has_aux=has_aux,
            holomorphic=holomorphic,
            allow_int=allow_int,
            chunk_size=chunk_size,
            call_cache=call_cache if f_bound is f else None,
        )
        return ((value, aux), compressed) if has_aux else (value, compressed)

    return compressed_fn


# Public API: decompression


@_fill_doc
def decompress_data(compressed: jax.Array, coloring: ColoredPattern) -> jax.Array:
    """Gather a compressed matrix ``B`` into sparse values in pattern order.

    Returns a plain ``jax.Array`` of shape ``(coloring.sparsity.nnz,)``
    holding the sparse values in ``coloring.sparsity`` order,
    so ``data[k]`` is the entry at
    ``(coloring.sparsity.rows[k], coloring.sparsity.cols[k])``.

    This is the jittable numeric core of decompression:
    it always returns a ``jax.Array``, so it composes inside ``jax.jit``
    and can feed a custom solver or sparse format,
    whereas [`decompress`][asdex.decompress] may return host
    (``numpy``/``scipy``) objects that cannot.
    Pair it with [`to_bcoo`][asdex.SparsityPattern.to_bcoo] for a BCOO,
    or with ``coloring.sparsity.rows`` / ``coloring.sparsity.cols``
    to assemble a custom format.

    Args:
        compressed: The compressed matrix ``B`` of shape ``(num_colors, dim)``,
            as returned by [`compressed_jacobian`][asdex.compressed_jacobian] or
            [`compressed_hessian`][asdex.compressed_hessian].
        coloring: {coloring_compressed}

    Returns:
        A ``jax.Array`` of shape ``(nnz,)`` with the sparse values in pattern order,
            matching ``compressed``'s dtype.

    Raises:
        ValueError: If ``compressed`` does not have shape ``(num_colors, dim)``
            for ``coloring``
            (see [`compressed_jacobian`][asdex.compressed_jacobian]
            for the per-mode ``dim``).
    """
    _validate_compressed(compressed, coloring)
    return _decompress_data(compressed, coloring)


@_fill_doc
def decompress(
    compressed: jax.Array,
    coloring: ColoredPattern,
    output_format: OutputFormat = _DEFAULT_OUTPUT_FORMAT,
) -> Any:
    """Decompress a compressed matrix ``B`` into a 2-D sparse matrix.

    Composes [`decompress_data`][asdex.decompress_data] with format dispatch:
    it gathers ``B`` into the sparse values,
    then materializes the flat ``(m, n)`` matrix in the requested format.

    Unlike the matrices returned by [`jacobian`][asdex.jacobian] /
    [`hessian`][asdex.hessian],
    this is always the flat 2-D matrix regardless of input/output pytree structure:
    ``B``'s natural domain is the 2-D compressed matrix.

    Args:
        compressed: The compressed matrix ``B`` of shape ``(num_colors, dim)``.
        coloring: {coloring_compressed}
        output_format: {format_flat}

    Returns:
        The sparse matrix of shape ``(m, n)`` in the requested format.

    Raises:
        ValueError: If ``compressed`` does not match ``coloring``'s expected shape,
            or ``output_format`` is unknown.
        ImportError: If a scipy ``output_format`` is requested but scipy is
            not installed.
    """
    _assert_output_format(output_format)
    data = decompress_data(compressed, coloring)
    return _decompress_to_format(data, coloring, output_format)
