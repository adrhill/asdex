"""Sparse Jacobian and Hessian computation using coloring and AD."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, assert_never

import jax
import jax.numpy as jnp
import numpy as np
from jax import dtypes
from jax.experimental.sparse import BCOO

from asdex._api_utils import (
    _ensure_index,
    flatten_pytree,
    merge_args_kwargs,
    merge_sample_inputs,
    unflatten_to_pytree,
    validate_input_dtypes,
    validate_output_dtypes,
)
from asdex.coloring import hessian_coloring as _hessian_coloring
from asdex.coloring import jacobian_coloring as _jacobian_coloring
from asdex.detection._api import _ensure_scalar, _strip_aux
from asdex.modes import (
    HessianMode,
    JacobianMode,
    OutputFormat,
    _assert_hessian_mode,
    _assert_jacobian_mode,
    _assert_output_format,
)
from asdex.pattern import ColoredPattern, SparsityPattern


def _to_scipy_sparse(
    data: jax.Array | np.ndarray,
    indices: np.ndarray,
    shape: tuple[int, int],
    fmt: str,
) -> Any:
    """Convert to scipy sparse array.

    Raises:
        ImportError: If scipy is not installed.
    """
    try:
        from scipy.sparse import coo_array  # noqa: PLC0415
    except ImportError as e:
        raise ImportError(
            f"scipy is required for output_format={fmt!r}. "
            "Install it with: pip install asdex[scipy]"
        ) from e

    coo = coo_array((np.asarray(data), (indices[:, 0], indices[:, 1])), shape=shape)
    if fmt == "scipy_coo":
        return coo
    if fmt == "scipy_csr":
        return coo.tocsr()
    return coo.tocsc()


def _convert_leaf_to_format(leaf: jax.Array | BCOO, output_format: str) -> Any:
    """Convert a single JAX array or BCOO leaf to the target format.

    For scipy formats, the leaf must be 2D (scipy sparse arrays only support 2D).
    Higher-dimensional blocks are converted by treating leading dimensions as rows.
    """
    if output_format == "numpy_dense":
        if isinstance(leaf, BCOO):
            return np.asarray(leaf.todense())
        return np.asarray(leaf)

    # scipy sparse formats - reshape to 2D for conversion
    arr = np.asarray(leaf.todense()) if isinstance(leaf, BCOO) else np.asarray(leaf)
    if arr.ndim == 0:
        flat_shape = (1, 1)
    elif arr.ndim == 1:
        flat_shape = (arr.shape[0], 1)
    else:
        flat_shape = (int(np.prod(arr.shape[:-1])), arr.shape[-1])

    flat = arr.reshape(flat_shape)
    nonzero = np.nonzero(flat)
    if len(nonzero[0]) == 0:
        indices_2d = np.zeros((0, 2), dtype=np.intp)
        data = np.array([], dtype=arr.dtype)
    else:
        indices_2d = np.column_stack(nonzero)
        data = flat[nonzero]

    return _to_scipy_sparse(data, indices_2d, flat_shape, output_format)


def _convert_pytree_to_format(pytree: Any, output_format: str) -> Any:
    """Convert each leaf in a pytree to the target numpy/scipy format."""

    def is_leaf(x: Any) -> bool:
        return isinstance(x, (jax.Array, np.ndarray, BCOO))

    return jax.tree_util.tree_map(
        lambda leaf: _convert_leaf_to_format(leaf, output_format),
        pytree,
        is_leaf=is_leaf,
    )


class _BCOOLeaf:
    """Wrapper to hide BCOO's internal pytree structure from tree operations.

    BCOO is registered as a pytree in JAX, which causes tree_transpose to descend
    into its internal structure.
    By wrapping BCOO in a plain class (not registered as a pytree), we can use
    tree_transpose normally and then unwrap afterwards.
    """

    __slots__ = ("array",)

    def __init__(self, array: BCOO) -> None:
        self.array = array


def _assert_chunk_size(chunk_size: int | None) -> None:
    """Validate chunk_size parameter."""
    if chunk_size is not None and chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")


def _chunked_vmap(
    fn: Callable[..., Any],
    seeds: jax.Array,
    chunk_size: int | None,
) -> jax.Array:
    """Vmap over seeds with bounded parallelism via sequential chunk processing.

    When ``chunk_size`` is ``None`` or exceeds the number of seeds, falls back to
    regular ``jax.vmap``. Otherwise, processes ``chunk_size`` seeds in parallel
    per chunk, with chunks processed sequentially via ``jax.lax.map``.

    Args:
        fn: Function to vmap over, taking a single seed vector.
        seeds: 2D array of shape ``(n_seeds, seed_dim)`` to process.
        chunk_size: Maximum seeds per parallel batch.
    """
    n = seeds.shape[0]
    if chunk_size is None or chunk_size >= n:
        return jax.vmap(fn)(seeds)

    n_chunks = (n + chunk_size - 1) // chunk_size
    padded_n = n_chunks * chunk_size

    # Pad to multiple of chunk_size
    seeds_padded = jnp.pad(seeds, ((0, padded_n - n), (0, 0)))
    chunks = seeds_padded.reshape((n_chunks, chunk_size, seeds.shape[1]))

    def process_chunk(chunk: jax.Array) -> jax.Array:
        return jax.vmap(fn)(chunk)

    results = jax.lax.map(process_chunk, chunks)
    return results.reshape((padded_n, results.shape[2]))[:n]


# Public API: one-shot entry points


def jacobian(
    f: Callable[..., Any],
    *sample_args: Any,
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    mode: JacobianMode | None = None,
    symmetric: bool = False,
    output_format: OutputFormat = "bcoo",
    chunk_size: int | None = None,
    **sample_kwargs: Any,
) -> Callable[..., Any]:
    """Detect sparsity, color, and return a function computing sparse Jacobians.

    Combines [`jacobian_coloring`][asdex.jacobian_coloring]
    and [`jacobian_from_coloring`][asdex.jacobian_from_coloring]
    in one call.

    Args:
        f: Function whose Jacobian is to be computed.
        *sample_args: Sample arguments of ``f``.
            Only structure and dtypes are used, values are ignored.
        argnums: Specifies which positional argument(s) to differentiate
            with respect to (default ``0``).
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
        output_format: Type of the output matrix.
            ``"bcoo"`` returns ``jax.experimental.sparse.BCOO`` (default),
            ``"dense"`` returns ``jax.Array``,
            ``"numpy_dense"`` returns ``numpy.ndarray``,
            ``"scipy_coo"`` returns ``scipy.sparse.coo_array``,
            ``"scipy_csr"`` returns ``scipy.sparse.csr_array``,
            ``"scipy_csc"`` returns ``scipy.sparse.csc_array``.
            SciPy formats require scipy to be installed.
        chunk_size: Maximum number of colors to process in parallel.
            When ``None`` (default), all colors are processed in a single vmapped batch.
            When specified, colors are processed in chunks of this size to reduce
            peak memory usage.
        **sample_kwargs: Sample keyword arguments of ``f``.
            Merged with ``sample_args`` based on ``f``'s signature.

    Returns:
        A function that takes the same positional args as ``f`` and returns
            a pytree of Jacobian blocks matching ``argnums``, with each leaf
            shaped ``(*out_shape, *in_leaf_shape)``.
            The block type depends on ``output_format``
            (``jax.experimental.sparse.BCOO`` by default, or ``jax.Array``
            when ``"dense"``).
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
        )

    return jac_fn


def value_and_jacobian(
    f: Callable[..., Any],
    *sample_args: Any,
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    mode: JacobianMode | None = None,
    symmetric: bool = False,
    output_format: OutputFormat = "bcoo",
    chunk_size: int | None = None,
    **sample_kwargs: Any,
) -> Callable[..., Any]:
    """Detect sparsity, color, and return a function computing value and sparse Jacobian.

    Like [`jacobian`][asdex.jacobian],
    but also returns the primal value ``f(*args)``
    without an extra forward pass.

    Returns:
        A function that takes the same positional args as ``f`` and returns
            ``(value, jac)`` — or ``((value, aux), jac)`` when ``has_aux=True``,
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
        )

    return val_jac_fn


def hessian(
    f: Callable[..., Any],
    *sample_args: Any,
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    mode: HessianMode | None = None,
    symmetric: bool = True,
    output_format: OutputFormat = "bcoo",
    chunk_size: int | None = None,
    **sample_kwargs: Any,
) -> Callable[..., Any]:
    """Detect sparsity, color, and return a function computing sparse Hessians.

    If ``f`` returns a squeezable shape like ``(1,)`` or ``(1, 1)``,
    it is automatically squeezed to scalar.

    Args:
        f: Scalar-valued function whose Hessian is to be computed.
        *sample_args: Sample arguments of ``f``.
            Only structure and dtypes are used, values are ignored.
        argnums: Specifies which positional argument(s) to differentiate
            with respect to (default ``0``).
        has_aux: Whether ``f`` returns ``(output, auxiliary_data)``.
        holomorphic: Whether ``f`` is promised to be holomorphic.
        allow_int: Whether to allow differentiating with respect to integer inputs.
        mode: AD mode for Hessian computation.
        symmetric: Whether to use symmetric (star) coloring.
        output_format: Type of the output matrix.
            ``"bcoo"`` returns ``jax.experimental.sparse.BCOO`` (default),
            ``"dense"`` returns ``jax.Array``,
            ``"numpy_dense"`` returns ``numpy.ndarray``,
            ``"scipy_coo"`` returns ``scipy.sparse.coo_array``,
            ``"scipy_csr"`` returns ``scipy.sparse.csr_array``,
            ``"scipy_csc"`` returns ``scipy.sparse.csc_array``.
            SciPy formats require scipy to be installed.
        chunk_size: Maximum number of colors to process in parallel.
            When ``None`` (default), all colors are processed in a single vmapped batch.
            When specified, colors are processed in chunks of this size to reduce
            peak memory usage.
        **sample_kwargs: Sample keyword arguments of ``f``.
            Merged with ``sample_args`` based on ``f``'s signature.

    Returns:
        A function that takes the same positional args as ``f`` and returns
            the sparse Hessian.
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
        )

    return hess_fn


def value_and_hessian(
    f: Callable[..., Any],
    *sample_args: Any,
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    mode: HessianMode | None = None,
    symmetric: bool = True,
    output_format: OutputFormat = "bcoo",
    chunk_size: int | None = None,
    **sample_kwargs: Any,
) -> Callable[..., Any]:
    """Detect sparsity, color, and return a function computing value and sparse Hessian.

    Like [`hessian`][asdex.hessian], but also returns the primal value
    ``f(*args)`` without an extra forward pass.

    Args:
        f: Scalar-valued function whose Hessian is to be computed.
        *sample_args: Sample arguments of ``f``.
            Only structure and dtypes are used, values are ignored.
        argnums: Specifies which positional argument(s) to differentiate
            with respect to (default ``0``).
        has_aux: Whether ``f`` returns ``(output, auxiliary_data)``.
        holomorphic: Whether ``f`` is promised to be holomorphic.
        allow_int: Whether to allow differentiating with respect to integer inputs.
        mode: AD mode for Hessian computation.
        symmetric: Whether to use symmetric (star) coloring.
        output_format: Type of the output matrix.
            ``"bcoo"`` returns ``jax.experimental.sparse.BCOO`` (default),
            ``"dense"`` returns ``jax.Array``,
            ``"numpy_dense"`` returns ``numpy.ndarray``,
            ``"scipy_coo"`` returns ``scipy.sparse.coo_array``,
            ``"scipy_csr"`` returns ``scipy.sparse.csr_array``,
            ``"scipy_csc"`` returns ``scipy.sparse.csc_array``.
            SciPy formats require scipy to be installed.
        chunk_size: Maximum number of colors to process in parallel.
            When ``None`` (default), all colors are processed in a single vmapped batch.
            When specified, colors are processed in chunks of this size to reduce
            peak memory usage.
        **sample_kwargs: Sample keyword arguments of ``f``.
            Merged with ``sample_args`` based on ``f``'s signature.

    Returns:
        A function that takes the same positional args as ``f`` and returns
            ``(value, hessian)``.
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
        )

    return val_hess_fn


# Public API: ``*_from_coloring`` entry points


def jacobian_from_coloring(
    f: Callable[..., Any],
    coloring: ColoredPattern,
    output_format: OutputFormat = "bcoo",
    *,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    chunk_size: int | None = None,
) -> Callable[..., Any]:
    """Build a sparse Jacobian function from a pre-computed coloring.

    Uses row coloring + VJPs or column coloring + JVPs,
    depending on which needs fewer colors.

    The returned callable accepts ``*args, **kwargs``; kwargs are forwarded
    to ``f`` at call time (matching ``jax.jacfwd`` / ``jax.jacrev``).

    Args:
        f: Function whose Jacobian is to be computed.
        coloring: Pre-computed colored sparsity pattern.
        output_format: Type of the output matrix.
            ``"bcoo"`` returns ``jax.experimental.sparse.BCOO`` (default),
            ``"dense"`` returns ``jax.Array``,
            ``"numpy_dense"`` returns ``numpy.ndarray``,
            ``"scipy_coo"`` returns ``scipy.sparse.coo_array``,
            ``"scipy_csr"`` returns ``scipy.sparse.csr_array``,
            ``"scipy_csc"`` returns ``scipy.sparse.csc_array``.
            SciPy formats require scipy to be installed.
        has_aux: Whether ``f`` returns ``(output, auxiliary_data)``.
        holomorphic: Whether ``f`` is promised to be holomorphic.
        allow_int: Whether to allow differentiating with respect to integer inputs.
        chunk_size: Maximum number of colors to process in parallel.
    """
    _assert_output_format(output_format)
    _assert_chunk_size(chunk_size)

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
        )

    return jac_fn


def hessian_from_coloring(
    f: Callable[..., Any],
    coloring: ColoredPattern,
    output_format: OutputFormat = "bcoo",
    *,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    chunk_size: int | None = None,
) -> Callable[..., Any]:
    """Build a sparse Hessian function from a pre-computed coloring.

    Uses symmetric (star) coloring and Hessian-vector products by default.

    Args:
        f: Scalar-valued function whose Hessian is to be computed.
        coloring: Pre-computed colored sparsity pattern.
        output_format: Type of the output matrix.
            ``"bcoo"`` returns ``jax.experimental.sparse.BCOO`` (default),
            ``"dense"`` returns ``jax.Array``,
            ``"numpy_dense"`` returns ``numpy.ndarray``,
            ``"scipy_coo"`` returns ``scipy.sparse.coo_array``,
            ``"scipy_csr"`` returns ``scipy.sparse.csr_array``,
            ``"scipy_csc"`` returns ``scipy.sparse.csc_array``.
            SciPy formats require scipy to be installed.
        has_aux: Whether ``f`` returns ``(output, auxiliary_data)``.
        holomorphic: Whether ``f`` is promised to be holomorphic.
        allow_int: Whether to allow differentiating with respect to integer inputs.
        chunk_size: Maximum number of colors to process in parallel.
    """
    _assert_output_format(output_format)
    _assert_chunk_size(chunk_size)

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
        )

    return hess_fn


def value_and_jacobian_from_coloring(
    f: Callable[..., Any],
    coloring: ColoredPattern,
    output_format: OutputFormat = "bcoo",
    *,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    chunk_size: int | None = None,
) -> Callable[..., Any]:
    """Build a function computing value and sparse Jacobian from a pre-computed coloring.

    Args:
        f: Function whose Jacobian is to be computed.
        coloring: Pre-computed colored sparsity pattern.
        output_format: Type of the output matrix.
            ``"bcoo"`` returns ``jax.experimental.sparse.BCOO`` (default),
            ``"dense"`` returns ``jax.Array``,
            ``"numpy_dense"`` returns ``numpy.ndarray``,
            ``"scipy_coo"`` returns ``scipy.sparse.coo_array``,
            ``"scipy_csr"`` returns ``scipy.sparse.csr_array``,
            ``"scipy_csc"`` returns ``scipy.sparse.csc_array``.
            SciPy formats require scipy to be installed.
        has_aux: Whether ``f`` returns ``(output, auxiliary_data)``.
        holomorphic: Whether ``f`` is promised to be holomorphic.
        allow_int: Whether to allow differentiating with respect to integer inputs.
        chunk_size: Maximum number of colors to process in parallel.
    """
    _assert_output_format(output_format)
    _assert_chunk_size(chunk_size)

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
        )

    return val_jac_fn


def value_and_hessian_from_coloring(
    f: Callable[..., Any],
    coloring: ColoredPattern,
    output_format: OutputFormat = "bcoo",
    *,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    chunk_size: int | None = None,
) -> Callable[..., Any]:
    """Build a function computing value and sparse Hessian from a pre-computed coloring.

    Args:
        f: Scalar-valued function whose Hessian is to be computed.
        coloring: Pre-computed colored sparsity pattern.
        output_format: Type of the output matrix.
            ``"bcoo"`` returns ``jax.experimental.sparse.BCOO`` (default),
            ``"dense"`` returns ``jax.Array``,
            ``"numpy_dense"`` returns ``numpy.ndarray``,
            ``"scipy_coo"`` returns ``scipy.sparse.coo_array``,
            ``"scipy_csr"`` returns ``scipy.sparse.csr_array``,
            ``"scipy_csc"`` returns ``scipy.sparse.csc_array``.
            SciPy formats require scipy to be installed.
        has_aux: Whether ``f`` returns ``(output, auxiliary_data)``.
        holomorphic: Whether ``f`` is promised to be holomorphic.
        allow_int: Whether to allow differentiating with respect to integer inputs.
        chunk_size: Maximum number of colors to process in parallel.
    """
    _assert_output_format(output_format)
    _assert_chunk_size(chunk_size)

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
        )

    return val_hess_fn


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
) -> Any:
    """Evaluate the sparse Jacobian of ``f`` at ``args``.

    Returns the block structure by default, ``(jac, aux)`` with ``has_aux=True``.
    """
    sparsity = coloring.sparsity
    _validate_args(args, sparsity)
    selected = _selected_args(args, sparsity)
    validate_input_dtypes(selected, coloring.mode, holomorphic, allow_int)

    m = sparsity.m
    n_selected = sparsity.n
    f_out = _strip_aux(f) if has_aux else f
    out_struct = jax.eval_shape(f_out, *args)

    if m == 0 or sparsity.nnz == 0:
        dense = jnp.zeros((m, n_selected))
        jac = _assemble_jacobian(dense, sparsity, output_format, out_struct)
        if has_aux:
            _, aux = f(*args)
            return jac, aux
        return jac

    _assert_jacobian_mode(coloring.mode)
    match coloring.mode:
        case "rev":
            compressed, y, aux = _jacobian_rows(
                f, args, coloring, out_struct, has_aux=has_aux, chunk_size=chunk_size
            )
        case "fwd":
            compressed, y, aux = _jacobian_cols(
                f, args, coloring, has_aux=has_aux, chunk_size=chunk_size
            )
        case _ as unreachable:
            assert_never(unreachable)  # ty: ignore[type-assertion-failure]

    validate_output_dtypes(y, coloring.mode, holomorphic)
    data = _decompress_data(coloring, compressed)
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
    n_selected = sparsity.n
    f_out = _strip_aux(f) if has_aux else f
    out_struct = jax.eval_shape(f_out, *args)

    if m == 0 or sparsity.nnz == 0:
        dense = jnp.zeros((m, n_selected))
        empty = _assemble_jacobian(dense, sparsity, output_format, out_struct)
        if has_aux:
            value, aux = f(*args)
            return (value, aux), empty
        value = f(*args)
        return value, empty

    _assert_jacobian_mode(coloring.mode)
    match coloring.mode:
        case "rev":
            compressed, y, aux = _jacobian_rows(
                f, args, coloring, out_struct, has_aux=has_aux, chunk_size=chunk_size
            )
        case "fwd":
            compressed, y, aux = _jacobian_cols(
                f, args, coloring, has_aux=has_aux, chunk_size=chunk_size
            )
        case _ as unreachable:
            assert_never(unreachable)  # ty: ignore[type-assertion-failure]

    validate_output_dtypes(y, coloring.mode, holomorphic)
    data = _decompress_data(coloring, compressed)
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
) -> Any:
    """Evaluate the sparse Hessian of a scalar-valued ``f`` at ``args``."""
    sparsity = coloring.sparsity
    _validate_args(args, sparsity)
    selected = _selected_args(args, sparsity)
    validate_input_dtypes(selected, coloring.mode, holomorphic, allow_int)

    f_scalar_raw = _strip_aux(f) if has_aux else f
    f_scalar = _ensure_scalar(f_scalar_raw, sparsity.input_avals)
    validate_output_dtypes(jax.eval_shape(f_scalar, *args), coloring.mode, holomorphic)

    n_selected = sparsity.n

    if sparsity.nnz == 0:
        dense = jnp.zeros((n_selected, n_selected))
        hess = _assemble_hessian(dense, sparsity, output_format)
        if has_aux:
            _, aux = f(*args)
            return hess, aux
        return hess

    compressed = _compute_hvps(f_scalar, args, coloring, chunk_size)
    data = _decompress_data(coloring, compressed)
    hess = _build_hessian(coloring, data, output_format)
    if has_aux:
        _, aux = f(*args)
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
) -> Any:
    """Evaluate ``f(*args)`` and the sparse Hessian of ``f`` at ``args``."""
    sparsity = coloring.sparsity
    _validate_args(args, sparsity)
    selected = _selected_args(args, sparsity)
    validate_input_dtypes(selected, coloring.mode, holomorphic, allow_int)

    f_scalar_raw = _strip_aux(f) if has_aux else f
    f_scalar = _ensure_scalar(f_scalar_raw, sparsity.input_avals)
    validate_output_dtypes(jax.eval_shape(f_scalar, *args), coloring.mode, holomorphic)

    n_selected = sparsity.n

    if sparsity.nnz == 0:
        dense = jnp.zeros((n_selected, n_selected))
        empty = _assemble_hessian(dense, sparsity, output_format)
        if has_aux:
            value, aux = f(*args)
            return (jnp.asarray(value), aux), empty
        value = jnp.asarray(f_scalar(*args))
        return value, empty

    value, compressed = _value_and_compute_hvps(f_scalar, args, coloring, chunk_size)
    data = _decompress_data(coloring, compressed)
    hess = _build_hessian(coloring, data, output_format)
    if has_aux:
        _, aux = f(*args)
        return (value, aux), hess
    return value, hess


# PyTree output helpers
#
# These mirror JAX's internal helpers for handling PyTree outputs in jacrev/jacfwd.
# See jax/_src/api.py: _std_basis, _jacrev_unravel, _unravel_array_into_pytree.


def _output_dtype(pytree: Any) -> jnp.dtype:
    """Get the result dtype for a PyTree of arrays."""
    leaves = jax.tree_util.tree_leaves(pytree)
    return dtypes.result_type(*leaves)


# Jacobian rows / cols over the selected input space


def _jacobian_rows(
    f: Callable[..., Any],
    args: tuple[Any, ...],
    coloring: ColoredPattern,
    out_struct: Any,
    *,
    has_aux: bool,
    chunk_size: int | None,
) -> tuple[jax.Array, Any, Any]:
    """Row-coloring VJPs over the combined selected input space.

    Returns ``(compressed, y, aux)``; ``aux`` is ``None`` when ``has_aux=False``.
    """
    sparsity = coloring.sparsity
    if has_aux:
        y, vjp_fn, aux = jax.vjp(f, *args, has_aux=True)
    else:
        y, vjp_fn = jax.vjp(f, *args)
        aux = None
    dtype = _output_dtype(y)
    seeds = jnp.asarray(coloring._seed_matrix, dtype=dtype)

    def single_vjp(seed: jax.Array) -> jax.Array:
        cotangent = unflatten_to_pytree(seed, out_struct)
        grads = vjp_fn(cotangent)
        return _flatten_selected_cotangents(grads, sparsity)

    J_compressed = _chunked_vmap(single_vjp, seeds, chunk_size)
    return J_compressed, y, aux


def _jacobian_cols(
    f: Callable[..., Any],
    args: tuple[Any, ...],
    coloring: ColoredPattern,
    *,
    has_aux: bool,
    chunk_size: int | None,
) -> tuple[jax.Array, Any, Any]:
    """Column-coloring JVPs over the combined selected input space.

    Returns ``(compressed, y, aux)``; ``aux`` is ``None`` when ``has_aux=False``.
    """
    sparsity = coloring.sparsity
    dtype = _selected_dtype(args, sparsity)
    if has_aux:
        y, jvp_fn, aux = jax.linearize(f, *args, has_aux=True)
    else:
        y, jvp_fn = jax.linearize(f, *args)
        aux = None
    seeds = jnp.asarray(coloring._seed_matrix, dtype=dtype)

    def single_jvp(seed: jax.Array) -> jax.Array:
        tangents = _build_tangents_from_seed(seed, args, sparsity)
        return flatten_pytree(jvp_fn(*tangents))

    J_compressed = _chunked_vmap(single_jvp, seeds, chunk_size)
    return J_compressed, y, aux


# HVPs over the selected input space


def _compute_hvps(
    f: Callable[..., Any],
    args: tuple[Any, ...],
    coloring: ColoredPattern,
    chunk_size: int | None,
) -> jax.Array:
    """One HVP per color for a scalar-valued multi-positional ``f``."""
    sparsity = coloring.sparsity
    dtype = _selected_dtype(args, sparsity)
    grad_argnums = sparsity.argnums

    seeds = jnp.asarray(coloring._seed_matrix, dtype=dtype)
    _assert_hessian_mode(coloring.mode)
    match coloring.mode:
        case "fwd_over_rev":
            grad_fn = jax.grad(f, argnums=grad_argnums)
            _, hvp_fn = jax.linearize(grad_fn, *args)

            def single_hvp(v: jax.Array) -> jax.Array:
                tangents = _build_tangents_from_seed(v, args, sparsity)
                tangent_out = hvp_fn(*tangents)
                return _flatten_grad_output(tangent_out)

        case "rev_over_fwd":

            def single_hvp(v: jax.Array) -> jax.Array:
                tangents = _build_tangents_from_seed(v, args, sparsity)

                def inner(*primals: Any) -> jax.Array:
                    _, out_tangent = jax.jvp(f, primals, tangents)
                    return out_tangent

                grads = jax.grad(inner, argnums=grad_argnums)(*args)
                return _flatten_grad_output(grads)

        case "rev_over_rev":
            grad_fn = jax.grad(f, argnums=grad_argnums)
            _, hvp_fn = jax.vjp(grad_fn, *args)

            def single_hvp(v: jax.Array) -> jax.Array:
                cotangent_out = _build_grad_output_from_seed(v, sparsity)
                cotangents = hvp_fn(cotangent_out)
                return _flatten_selected_cotangents(cotangents, sparsity)

        case _ as unreachable:
            assert_never(unreachable)  # ty: ignore[type-assertion-failure]

    H_compressed = _chunked_vmap(single_hvp, seeds, chunk_size)
    return H_compressed  # noqa: RET504


def _value_and_compute_hvps(
    f: Callable[..., Any],
    args: tuple[Any, ...],
    coloring: ColoredPattern,
    chunk_size: int | None,
) -> tuple[jax.Array, jax.Array]:
    """``f(*args)`` and one HVP per color for a scalar-valued ``f``.

    Primal is free for ``fwd_over_rev``; ``rev_over_fwd`` / ``rev_over_rev``
    compute it with an extra ``f(*args)`` call.
    """
    sparsity = coloring.sparsity
    dtype = _selected_dtype(args, sparsity)
    grad_argnums = sparsity.argnums

    seeds = jnp.asarray(coloring._seed_matrix, dtype=dtype)
    _assert_hessian_mode(coloring.mode)
    match coloring.mode:
        case "fwd_over_rev":
            val_and_grad = jax.value_and_grad(f, argnums=grad_argnums)
            (value, _g), hvp_fn = jax.linearize(val_and_grad, *args)

            def single_hvp(v: jax.Array) -> jax.Array:
                tangents = _build_tangents_from_seed(v, args, sparsity)
                _value_tangent, tangent_out = hvp_fn(*tangents)
                return _flatten_grad_output(tangent_out)

        case "rev_over_fwd":
            value = jnp.asarray(f(*args))

            def single_hvp(v: jax.Array) -> jax.Array:
                tangents = _build_tangents_from_seed(v, args, sparsity)

                def inner(*primals: Any) -> jax.Array:
                    _, out_tangent = jax.jvp(f, primals, tangents)
                    return out_tangent

                grads = jax.grad(inner, argnums=grad_argnums)(*args)
                return _flatten_grad_output(grads)

        case "rev_over_rev":
            # TODO: f(x) is redundant with the forward pass inside grad(f).
            # Using value_and_grad + vjp would avoid it, but inflates every
            # VJP application with dead zero-cotangents for the value path.
            # Revisit if XLA reliably DCEs the zero branch.
            value = jnp.asarray(f(*args))
            grad_fn = jax.grad(f, argnums=grad_argnums)
            _, hvp_fn = jax.vjp(grad_fn, *args)

            def single_hvp(v: jax.Array) -> jax.Array:
                cotangent_out = _build_grad_output_from_seed(v, sparsity)
                cotangents = hvp_fn(cotangent_out)
                return _flatten_selected_cotangents(cotangents, sparsity)

        case _ as unreachable:
            assert_never(unreachable)  # ty: ignore[type-assertion-failure]

    H_compressed = _chunked_vmap(single_hvp, seeds, chunk_size)
    return value, H_compressed


# Decompression


def _decompress_data(coloring: ColoredPattern, compressed: jax.Array) -> jax.Array:
    """Extract sparse data values from compressed gradient rows.

    Uses pre-computed gather indices on the ``ColoredPattern``
    to vectorize the decompression step
    (no Python loop over nnz entries).
    """
    return jax.lax.gather(
        compressed,
        coloring._gather_indices,
        dimension_numbers=jax.lax.GatherDimensionNumbers(
            offset_dims=(),
            collapsed_slice_dims=(0, 1),
            start_index_map=(0, 1),
        ),
        slice_sizes=(1, 1),
        unique_indices=True,
        mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
    )


def _scatter_dense(coloring: ColoredPattern, data: jax.Array) -> jax.Array:
    """Scatter sparse data values into a dense zero array of the full shape."""
    sparsity = coloring.sparsity
    indices = sparsity._bcoo_indices  # (nnz, 2)
    # jax.vjp returns float0 cotangents for integer inputs (allow_int=True).
    # float0 cannot back a real array, so fall back to a plain zero result.
    if data.dtype == dtypes.float0:
        return jnp.zeros(sparsity.shape, dtype=jnp.float_)
    result = jnp.zeros(sparsity.shape, dtype=data.dtype)
    return result.at[indices[:, 0], indices[:, 1]].set(data)


def _is_simple_input(sparsity: SparsityPattern) -> bool:
    """Check if input has a single leaf with trivial pytree structure."""
    if len(sparsity.leaf_shapes) != 1:
        return False
    if not isinstance(sparsity.argnums, int):
        return False
    in_aval = sparsity.input_avals[sparsity.argnums]
    in_treedef = jax.tree_util.tree_structure(in_aval)
    return in_treedef.num_leaves == 1 and in_treedef.num_nodes == 1


def _is_simple_output(out_struct: Any, sparsity: SparsityPattern) -> bool:
    """Check if output and input are both single flat arrays with trivial structure."""
    out_leaves = jax.tree_util.tree_leaves(out_struct)
    if len(out_leaves) != 1:
        return False
    out_size = int(np.prod(out_leaves[0].shape))
    if out_size != sparsity.m:
        return False
    out_treedef = jax.tree_util.tree_structure(out_struct)
    if out_treedef.num_leaves != 1 or out_treedef.num_nodes != 1:
        return False
    return _is_simple_input(sparsity)


def _build_jacobian(
    coloring: ColoredPattern,
    data: jax.Array,
    output_format: OutputFormat,
    out_struct: Any,
) -> Any:
    """Build Jacobian output from sparse data, avoiding BCOO.fromdense under JIT.

    For simple single-array outputs, constructs BCOO directly from known indices.
    For PyTree outputs, scatters to dense then assembles blocks.
    """
    sparsity = coloring.sparsity

    # Fast path: single flat array output with BCOO format.
    # Use the known sparsity pattern indices directly, avoiding fromdense.
    # numpy/scipy formats involve host transfer anyway, so fromdense cost is negligible.
    if output_format == "bcoo" and _is_simple_output(out_struct, sparsity):
        if data.dtype == dtypes.float0:
            data = jnp.zeros(sparsity.nnz, dtype=jnp.float_)
        out_shape = jax.tree_util.tree_leaves(out_struct)[0].shape
        in_shape = sparsity.leaf_shapes[0]
        return sparsity.to_bcoo(data=data).reshape((*out_shape, *in_shape))

    # General path: scatter to dense, then assemble blocks.
    dense = _scatter_dense(coloring, data)
    jac = _assemble_jacobian(dense, sparsity, output_format, out_struct)

    # Convert to numpy/scipy formats after assembly
    if output_format in ("numpy_dense", "scipy_coo", "scipy_csr", "scipy_csc"):
        return _convert_pytree_to_format(jac, output_format)
    return jac


def _build_hessian(
    coloring: ColoredPattern,
    data: jax.Array,
    output_format: OutputFormat,
) -> Any:
    """Build Hessian output from sparse data, avoiding BCOO.fromdense under JIT.

    For simple single-input cases, constructs BCOO directly from known indices.
    For PyTree inputs, scatters to dense then assembles blocks.
    """
    sparsity = coloring.sparsity

    # Fast path: single input leaf with BCOO format and trivial pytree structure.
    # numpy/scipy formats involve host transfer anyway, so fromdense cost is negligible.
    if output_format == "bcoo" and _is_simple_input(sparsity):
        in_shape = sparsity.leaf_shapes[0]
        return sparsity.to_bcoo(data=data).reshape((*in_shape, *in_shape))

    # General path: scatter to dense, then assemble blocks.
    dense = _scatter_dense(coloring, data)
    hess = _assemble_hessian(dense, sparsity, output_format)

    # Convert to numpy/scipy formats after assembly
    if output_format in ("numpy_dense", "scipy_coo", "scipy_csr", "scipy_csc"):
        return _convert_pytree_to_format(hess, output_format)
    return hess


# Argument handling and flattening


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


def _selected_args(args: tuple[Any, ...], sparsity: SparsityPattern) -> tuple[Any, ...]:
    """Sub-tuple of ``args`` at positions named by ``sparsity.argnums``."""
    return tuple(args[i] for i in sparsity._argnums_tuple)


def _selected_dtype(args: tuple[Any, ...], sparsity: SparsityPattern) -> Any:
    """Dtype for seed arrays, taken from the first selected leaf we can find."""
    for leaf in jax.tree_util.tree_leaves(_selected_args(args, sparsity)):
        dtype = getattr(leaf, "dtype", None)
        if dtype is not None:
            return dtype
    return jnp.float_


def _build_tangents_from_seed(
    seed: jax.Array,
    args: tuple[Any, ...],
    sparsity: SparsityPattern,
) -> tuple[Any, ...]:
    """Split a ``(n_selected,)`` seed into a per-positional-arg tangent pytree.

    Selected positions get chunks reshaped into their aval leaves; non-selected
    positions get zero tangents so they have no effect on the JVP.
    """
    leaf_sizes = sparsity.leaf_sizes
    leaf_shapes = sparsity.leaf_shapes
    chunks: list[jax.Array] = []
    offset = 0
    for size in leaf_sizes:
        chunks.append(seed[offset : offset + size])
        offset += size

    # Map position -> chunk offset. Chunks are in argnums order, not position order.
    pos_to_chunk_offset: dict[int, int] = {}
    chunk_offset = 0
    for pos in sparsity._argnums_tuple:
        pos_to_chunk_offset[pos] = chunk_offset
        aval_leaves = jax.tree_util.tree_leaves(sparsity.input_avals[pos])
        chunk_offset += len(aval_leaves)

    tangents: list[Any] = []
    for pos_idx, (arg, aval) in enumerate(zip(args, sparsity.input_avals, strict=True)):
        del arg
        aval_leaves = jax.tree_util.tree_leaves(aval)
        aval_tree = jax.tree_util.tree_structure(aval)
        if pos_idx in pos_to_chunk_offset:
            chunk_idx = pos_to_chunk_offset[pos_idx]
            leaf_tangents = [
                chunks[chunk_idx + k].reshape(leaf_shapes[chunk_idx + k])
                for k in range(len(aval_leaves))
            ]
        else:
            leaf_tangents = [
                jnp.zeros(tuple(leaf.shape), dtype=seed.dtype) for leaf in aval_leaves
            ]
        tangents.append(jax.tree_util.tree_unflatten(aval_tree, leaf_tangents))
    return tuple(tangents)


def _flatten_selected_cotangents(
    cotangents: Any, sparsity: SparsityPattern
) -> jax.Array:
    """Flatten cotangent leaves at selected positions into a ``(n_selected,)`` vector.

    ``jax.vjp(f, *xs)`` returns a tuple of cotangents matching the primals.
    Non-selected positions are ignored; selected positions contribute all leaves.
    Float0 leaves (from integer inputs with allow_int=True) are replaced with zeros.
    """
    selected = tuple(cotangents[i] for i in sparsity._argnums_tuple)
    leaves = jax.tree_util.tree_leaves(selected)
    if not leaves:
        return jnp.zeros((0,))
    raveled = []
    for leaf in leaves:
        if leaf.dtype == dtypes.float0:
            raveled.append(jnp.zeros(leaf.shape, dtype=jnp.float_).ravel())
        else:
            raveled.append(leaf.ravel())
    return jnp.concatenate(raveled)


def _flatten_grad_output(out: Any) -> jax.Array:
    """Flatten a gradient output into ``(n_selected,)``.

    ``jax.grad(f, argnums=...)`` already restricts its output to the selected
    positions, so every leaf contributes to the flat vector.
    """
    leaves = jax.tree_util.tree_leaves(out)
    if not leaves:
        return jnp.zeros((0,))
    return jnp.concatenate([leaf.ravel() for leaf in leaves])


def _build_grad_output_from_seed(
    seed: jax.Array,
    sparsity: SparsityPattern,
) -> Any:
    """Build a gradient-shaped pytree from a ``(n_selected,)`` seed.

    Mirrors ``_flatten_grad_output`` in reverse: used as the seed cotangent
    passed into the outer VJP in ``rev_over_rev`` Hessian mode.
    The output matches ``sparsity.example_input`` (structure of ``dyn_avals``
    when ``argnums`` is a tuple, or the single aval when it is an int).
    """
    leaf_shapes = sparsity.leaf_shapes
    leaf_sizes = sparsity.leaf_sizes
    chunks: list[jax.Array] = []
    offset = 0
    for size, shape in zip(leaf_sizes, leaf_shapes, strict=True):
        chunks.append(seed[offset : offset + size].reshape(shape))
        offset += size

    if isinstance(sparsity.argnums, int):
        # Single selected position: unflatten into that position's pytree.
        aval = sparsity.input_avals[sparsity.argnums]
        treedef = jax.tree_util.tree_structure(aval)
        return jax.tree_util.tree_unflatten(treedef, chunks)

    # Tuple of positions: one pytree per selected position, then a tuple.
    groups: list[Any] = []
    idx = 0
    for pos in sparsity.argnums:
        aval = sparsity.input_avals[pos]
        aval_leaves = jax.tree_util.tree_leaves(aval)
        group = chunks[idx : idx + len(aval_leaves)]
        idx += len(aval_leaves)
        treedef = jax.tree_util.tree_structure(aval)
        groups.append(jax.tree_util.tree_unflatten(treedef, group))
    return tuple(groups)


# Block packing


def _assemble_jacobian(
    dense: jax.Array,
    sparsity: SparsityPattern,
    output_format: OutputFormat,
    out_struct: Any,
) -> Any:
    """Split a ``(m, n_selected)`` dense matrix into per-leaf Jacobian blocks.

    Each block is reshaped to ``(*out_leaf_shape, *in_leaf_shape)`` to match
    ``jax.jacfwd`` / ``jax.jacrev`` output layout.

    For PyTree outputs, the result has structure ``(output_tree, input_tree)``,
    mirroring ``jax.jacobian``.
    """
    in_leaf_shapes = sparsity.leaf_shapes
    in_leaf_sizes = sparsity.leaf_sizes

    out_leaves, out_treedef = jax.tree_util.tree_flatten(out_struct)
    out_leaf_shapes = [tuple(leaf.shape) for leaf in out_leaves]
    out_leaf_sizes = [int(np.prod(shape)) for shape in out_leaf_shapes]

    # Build (input_leaf_idx, output_leaf_idx) -> block
    # Then transpose to (output_tree, input_tree) structure
    in_col_offset = 0
    per_input_blocks: list[list[jax.Array | BCOO]] = []

    for in_size, in_shape in zip(in_leaf_sizes, in_leaf_shapes, strict=True):
        out_row_offset = 0
        out_blocks: list[jax.Array | BCOO] = []

        for out_size, out_shape in zip(out_leaf_sizes, out_leaf_shapes, strict=True):
            chunk = dense[
                out_row_offset : out_row_offset + out_size,
                in_col_offset : in_col_offset + in_size,
            ]
            block: jax.Array | BCOO = chunk.reshape((*out_shape, *in_shape))
            if output_format == "bcoo":
                block = BCOO.fromdense(block)
            out_blocks.append(block)
            out_row_offset += out_size

        per_input_blocks.append(out_blocks)
        in_col_offset += in_size

    # per_input_blocks[in_idx][out_idx] -> need (out_tree, in_tree) structure
    # First rebuild as (in_tree, out_tree), then transpose.
    # This mirrors JAX's approach: always build both tree structures and transpose,
    # even for single-leaf cases where the structure may still be nested.
    out_trees_per_in_leaf = [
        jax.tree_util.tree_unflatten(out_treedef, out_blocks)
        for out_blocks in per_input_blocks
    ]
    in_tree_of_out_trees = _group_blocks_by_argnums(out_trees_per_in_leaf, sparsity)
    return _transpose_in_out_trees(in_tree_of_out_trees, out_treedef, output_format)


def _assemble_hessian(
    dense: jax.Array,
    sparsity: SparsityPattern,
    output_format: OutputFormat,
) -> Any:
    """Split a ``(n_sel, n_sel)`` dense matrix into a nested block grid.

    For each outer leaf, pack the inner axis into the full input-structured
    pytree using the same rules as Jacobian packing, then pack those rows
    again on the outer axis. The result mirrors what
    ``jax.hessian(f, argnums=...)`` returns.
    """
    leaf_shapes = sparsity.leaf_shapes
    leaf_sizes = sparsity.leaf_sizes

    leaf_blocks: list[list[jax.Array | BCOO]] = []
    row_offset = 0
    for row_size, row_shape in zip(leaf_sizes, leaf_shapes, strict=True):
        col_offset = 0
        row_blocks: list[jax.Array | BCOO] = []
        for col_size, col_shape in zip(leaf_sizes, leaf_shapes, strict=True):
            chunk = dense[
                row_offset : row_offset + row_size,
                col_offset : col_offset + col_size,
            ]
            block: jax.Array | BCOO = chunk.reshape(row_shape + col_shape)
            if output_format == "bcoo":
                block = BCOO.fromdense(block)
            row_blocks.append(block)
            col_offset += col_size
        leaf_blocks.append(row_blocks)
        row_offset += row_size

    inner_packed = [_group_blocks_by_argnums(row, sparsity) for row in leaf_blocks]
    return _group_blocks_by_argnums(inner_packed, sparsity)


def _group_blocks_by_argnums(
    blocks: Sequence[Any],
    sparsity: SparsityPattern,
) -> Any:
    """Group per-leaf blocks by selected position and wrap according to ``argnums``.

    When ``argnums`` is a tuple, returns a tuple of per-position pytrees;
    when it is an int, returns the single per-position pytree directly
    (matching ``jax.jacfwd`` return shape).
    """
    grouped: list[Any] = []
    idx = 0
    for pos in sparsity._argnums_tuple:
        aval = sparsity.input_avals[pos]
        aval_leaves = jax.tree_util.tree_leaves(aval)
        group = list(blocks[idx : idx + len(aval_leaves)])
        idx += len(aval_leaves)
        treedef = jax.tree_util.tree_structure(aval)
        grouped.append(jax.tree_util.tree_unflatten(treedef, group))
    if isinstance(sparsity.argnums, int):
        assert len(grouped) == 1
        return grouped[0]
    return tuple(grouped)


def _transpose_in_out_trees(
    in_tree_of_out_trees: Any,
    out_treedef: jax.tree_util.PyTreeDef,
    output_format: OutputFormat,
) -> Any:
    """Transpose (in_tree, out_tree) structure to (out_tree, in_tree).

    For dense output, uses jax.tree_util.tree_transpose directly.
    For BCOO output, wraps BCOO arrays in _BCOOLeaf to hide their internal pytree
    structure, transposes normally, then unwraps.
    """

    def is_bcoo(x: Any) -> bool:
        return isinstance(x, BCOO)

    def is_bcoo_leaf(x: Any) -> bool:
        return isinstance(x, _BCOOLeaf)

    def is_out_tree(x: Any) -> bool:
        is_leaf = is_bcoo_leaf if output_format == "bcoo" else is_bcoo
        return jax.tree_util.tree_structure(x, is_leaf=is_leaf) == out_treedef

    if output_format == "bcoo":
        in_tree_of_out_trees = jax.tree_util.tree_map(
            _BCOOLeaf, in_tree_of_out_trees, is_leaf=is_bcoo
        )

    in_treedef = jax.tree_util.tree_structure(
        jax.tree_util.tree_map(lambda _: 0, in_tree_of_out_trees, is_leaf=is_out_tree)
    )

    transposed = jax.tree_util.tree_transpose(
        in_treedef, out_treedef, in_tree_of_out_trees
    )

    if output_format == "bcoo":
        return jax.tree_util.tree_map(
            lambda x: x.array, transposed, is_leaf=is_bcoo_leaf
        )
    return transposed
