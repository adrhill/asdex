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
    JaxOutputFormat,
    OutputFormat,
    ScipyOutputFormat,
    _assert_hessian_mode,
    _assert_jacobian_mode,
    _assert_output_format,
    _import_scipy_coo_array,
)
from asdex.pattern import ColoredPattern, SparsityPattern


def _sparsity_to_scipy(
    sparsity: SparsityPattern, data: jax.Array, fmt: ScipyOutputFormat
) -> Any:
    """Build a scipy sparse array with the pattern's structure and the given data.

    Structural non-zeros that are numerically zero are kept as explicit entries,
    so the output structure always matches the detected sparsity pattern.

    Raises:
        ImportError: If scipy is not installed.
    """
    coo_array = _import_scipy_coo_array(fmt)
    coo = coo_array(
        (np.asarray(data), (sparsity.rows, sparsity.cols)), shape=sparsity.shape
    )
    match fmt:
        case "scipy_coo":
            return coo
        case "scipy_csr":
            return coo.tocsr()
        case "scipy_csc":
            return coo.tocsc()
        case _ as unreachable:
            assert_never(unreachable)


def _assert_scipy_supported_jacobian(
    out_struct: Any, sparsity: SparsityPattern, fmt: ScipyOutputFormat
) -> None:
    """Raise ``ValueError`` unless the Jacobian is a single 2D matrix.

    SciPy sparse arrays are 2D-only,
    so the input and output must each be a single flat (1D) array.
    """
    out_leaves = jax.tree_util.tree_leaves(out_struct)
    if (
        _is_simple_output(out_struct, sparsity)
        and len(out_leaves[0].shape) == 1
        and len(sparsity.leaf_shapes[0]) == 1
    ):
        return
    raise ValueError(
        "SciPy sparse formats only support 2D Jacobians: "
        f"output_format={fmt!r} requires the input and output "
        "to each be a single flat (1D) array."
    )


def _assert_scipy_supported_hessian(
    sparsity: SparsityPattern, fmt: ScipyOutputFormat
) -> None:
    """Raise ``ValueError`` unless the Hessian is a single 2D matrix.

    SciPy sparse arrays are 2D-only,
    so the input must be a single flat (1D) array.
    """
    if _is_simple_input(sparsity) and len(sparsity.leaf_shapes[0]) == 1:
        return
    raise ValueError(
        "SciPy sparse formats only support 2D Hessians: "
        f"output_format={fmt!r} requires the input to be a single flat (1D) array."
    )


def _to_numpy_pytree(pytree: Any) -> Any:
    """Convert each JAX array leaf in a pytree to ``numpy.ndarray``."""
    return jax.tree_util.tree_map(np.asarray, pytree)


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
    return jax.lax.map(fn, seeds, batch_size=chunk_size)


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

    For repeated evaluation, wrap the returned function in ``jax.jit``:
    each unjitted call re-traces ``f``,
    which can cost far more than the differentiation itself.
    The ``"numpy_dense"`` and scipy output formats cannot be jitted
    since they produce non-JAX arrays.

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
            SciPy formats require scipy and only support 2D Jacobians:
            the input and output must each be a single flat (1D) array
            (scalar outputs are not supported).
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

    call_cache: dict[Any, Any] = {}

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

    For repeated evaluation, wrap the returned function in ``jax.jit``:
    each unjitted call re-traces ``f``,
    which can cost far more than the differentiation itself.
    The ``"numpy_dense"`` and scipy output formats cannot be jitted
    since they produce non-JAX arrays.

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

    call_cache: dict[Any, Any] = {}

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

    For repeated evaluation, wrap the returned function in ``jax.jit``:
    each unjitted call re-traces ``f``,
    which can cost far more than the differentiation itself.
    The ``"numpy_dense"`` and scipy output formats cannot be jitted
    since they produce non-JAX arrays.

    Args:
        f: Scalar-valued function whose Hessian is to be computed.
        *sample_args: Sample arguments of ``f``.
            Only structure and dtypes are used, values are ignored.
        argnums: Specifies which positional argument(s) to differentiate
            with respect to (default ``0``).
        has_aux: Whether ``f`` returns ``(output, auxiliary_data)``.
        holomorphic: Whether ``f`` is promised to be holomorphic.
        allow_int: Unsupported for Hessians; passing ``True`` raises ``TypeError``
            (integer inputs cannot be differentiated twice, matching ``jax.hessian``).
        mode: AD mode for Hessian computation.
        symmetric: Whether to use symmetric (star) coloring.
        output_format: Type of the output matrix.
            ``"bcoo"`` returns ``jax.experimental.sparse.BCOO`` (default),
            ``"dense"`` returns ``jax.Array``,
            ``"numpy_dense"`` returns ``numpy.ndarray``,
            ``"scipy_coo"`` returns ``scipy.sparse.coo_array``,
            ``"scipy_csr"`` returns ``scipy.sparse.csr_array``,
            ``"scipy_csc"`` returns ``scipy.sparse.csc_array``.
            SciPy formats require scipy and only support 2D Hessians:
            the input must be a single flat (1D) array.
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

    call_cache: dict[Any, Any] = {}

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

    For repeated evaluation, wrap the returned function in ``jax.jit``:
    each unjitted call re-traces ``f``,
    which can cost far more than the differentiation itself.
    The ``"numpy_dense"`` and scipy output formats cannot be jitted
    since they produce non-JAX arrays.

    Args:
        f: Scalar-valued function whose Hessian is to be computed.
        *sample_args: Sample arguments of ``f``.
            Only structure and dtypes are used, values are ignored.
        argnums: Specifies which positional argument(s) to differentiate
            with respect to (default ``0``).
        has_aux: Whether ``f`` returns ``(output, auxiliary_data)``.
        holomorphic: Whether ``f`` is promised to be holomorphic.
        allow_int: Unsupported for Hessians; passing ``True`` raises ``TypeError``
            (integer inputs cannot be differentiated twice, matching ``jax.hessian``).
        mode: AD mode for Hessian computation.
        symmetric: Whether to use symmetric (star) coloring.
        output_format: Type of the output matrix.
            ``"bcoo"`` returns ``jax.experimental.sparse.BCOO`` (default),
            ``"dense"`` returns ``jax.Array``,
            ``"numpy_dense"`` returns ``numpy.ndarray``,
            ``"scipy_coo"`` returns ``scipy.sparse.coo_array``,
            ``"scipy_csr"`` returns ``scipy.sparse.csr_array``,
            ``"scipy_csc"`` returns ``scipy.sparse.csc_array``.
            SciPy formats require scipy and only support 2D Hessians:
            the input must be a single flat (1D) array.
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

    call_cache: dict[Any, Any] = {}

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

    For repeated evaluation, wrap the returned function in ``jax.jit``:
    each unjitted call re-traces ``f``,
    which can cost far more than the differentiation itself.
    The ``"numpy_dense"`` and scipy output formats cannot be jitted
    since they produce non-JAX arrays.

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
            SciPy formats require scipy and only support 2D Jacobians:
            the input and output must each be a single flat (1D) array
            (scalar outputs are not supported).
        has_aux: Whether ``f`` returns ``(output, auxiliary_data)``.
        holomorphic: Whether ``f`` is promised to be holomorphic.
        allow_int: Whether to allow differentiating with respect to integer inputs.
        chunk_size: Maximum number of colors to process in parallel.
    """
    _assert_output_format(output_format)
    _assert_chunk_size(chunk_size)

    call_cache: dict[Any, Any] = {}

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

    For repeated evaluation, wrap the returned function in ``jax.jit``:
    each unjitted call re-traces ``f``,
    which can cost far more than the differentiation itself.
    The ``"numpy_dense"`` and scipy output formats cannot be jitted
    since they produce non-JAX arrays.

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
            SciPy formats require scipy and only support 2D Hessians:
            the input must be a single flat (1D) array.
        has_aux: Whether ``f`` returns ``(output, auxiliary_data)``.
        holomorphic: Whether ``f`` is promised to be holomorphic.
        allow_int: Unsupported for Hessians; passing ``True`` raises ``TypeError``
            (integer inputs cannot be differentiated twice, matching ``jax.hessian``).
        chunk_size: Maximum number of colors to process in parallel.
    """
    _assert_output_format(output_format)
    _assert_chunk_size(chunk_size)

    call_cache: dict[Any, Any] = {}

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

    For repeated evaluation, wrap the returned function in ``jax.jit``:
    each unjitted call re-traces ``f``,
    which can cost far more than the differentiation itself.
    The ``"numpy_dense"`` and scipy output formats cannot be jitted
    since they produce non-JAX arrays.

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
            SciPy formats require scipy and only support 2D Jacobians:
            the input and output must each be a single flat (1D) array
            (scalar outputs are not supported).
        has_aux: Whether ``f`` returns ``(output, auxiliary_data)``.
        holomorphic: Whether ``f`` is promised to be holomorphic.
        allow_int: Whether to allow differentiating with respect to integer inputs.
        chunk_size: Maximum number of colors to process in parallel.
    """
    _assert_output_format(output_format)
    _assert_chunk_size(chunk_size)

    call_cache: dict[Any, Any] = {}

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

    For repeated evaluation, wrap the returned function in ``jax.jit``:
    each unjitted call re-traces ``f``,
    which can cost far more than the differentiation itself.
    The ``"numpy_dense"`` and scipy output formats cannot be jitted
    since they produce non-JAX arrays.

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
            SciPy formats require scipy and only support 2D Hessians:
            the input must be a single flat (1D) array.
        has_aux: Whether ``f`` returns ``(output, auxiliary_data)``.
        holomorphic: Whether ``f`` is promised to be holomorphic.
        allow_int: Unsupported for Hessians; passing ``True`` raises ``TypeError``
            (integer inputs cannot be differentiated twice, matching ``jax.hessian``).
        chunk_size: Maximum number of colors to process in parallel.
    """
    _assert_output_format(output_format)
    _assert_chunk_size(chunk_size)

    call_cache: dict[Any, Any] = {}

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


_HOST_FORMATS = ("numpy_dense", "scipy_coo", "scipy_csr", "scipy_csc")


def _cached_jit_core(
    cache: dict[Any, Any] | None,
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
    """Array-valued Hessian core ``args -> data`` for the internal jit."""

    def core(*args: Any) -> jax.Array:
        compressed, _ = _compute_hvps(f_scalar, args, coloring, chunk_size)
        return _decompress_data(coloring, compressed)

    return core


def _build_value_and_hessian_core(
    f_scalar: Callable[..., Any],
    coloring: ColoredPattern,
    chunk_size: int | None,
) -> Callable[..., Any]:
    """Array-valued Hessian core ``args -> (value, data)`` for the internal jit."""

    def core(*args: Any) -> tuple[jax.Array, jax.Array]:
        value, compressed, _ = _value_and_compute_hvps(
            f_scalar, args, coloring, chunk_size
        )
        return value, _decompress_data(coloring, compressed)

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
    call_cache: dict[Any, Any] | None,
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
    call_cache: dict[Any, Any] | None,
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
    call_cache: dict[Any, Any] | None,
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
        data = core(*args)
        aux = f(*args)[1] if has_aux else None
    else:
        f_aux = _cached_scalar_aux_fn(f, call_cache) if has_aux else None
        compressed, aux = _compute_hvps(
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
    call_cache: dict[Any, Any] | None,
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
        lambda: _build_value_and_hessian_core(f_scalar, coloring, chunk_size),
    )
    if core is not None:
        value, data = core(*args)
        aux = f(*args)[1] if has_aux else None
    else:
        f_aux = _cached_scalar_aux_fn(f, call_cache) if has_aux else None
        value, compressed, aux = _value_and_compute_hvps(
            f_scalar, args, coloring, chunk_size, f_aux=f_aux
        )
        data = _decompress_data(coloring, compressed)
    hess = _build_hessian(coloring, data, output_format)
    if has_aux:
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


def _jacobian_compressed(
    f: Callable[..., Any],
    args: tuple[Any, ...],
    coloring: ColoredPattern,
    out_struct: Any,
    *,
    has_aux: bool,
    chunk_size: int | None,
) -> tuple[jax.Array, Any, Any]:
    """Compress the Jacobian via VJPs (``rev``) or JVPs (``fwd``) by mode.

    Returns ``(compressed, y, aux)``; ``aux`` is ``None`` when ``has_aux=False``.
    """
    _assert_jacobian_mode(coloring.mode)
    match coloring.mode:
        case "rev":
            return _jacobian_rows(
                f, args, coloring, out_struct, has_aux=has_aux, chunk_size=chunk_size
            )
        case "fwd":
            return _jacobian_cols(
                f, args, coloring, has_aux=has_aux, chunk_size=chunk_size
            )
        case _ as unreachable:
            assert_never(unreachable)  # ty: ignore[type-assertion-failure]


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
    seeds = coloring._device_seeds(dtype)

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
    dtype = _uniform_selected_dtype(args, sparsity)
    if has_aux:
        y, jvp_fn, aux = jax.linearize(f, *args, has_aux=True)
    else:
        y, jvp_fn = jax.linearize(f, *args)
        aux = None
    seeds = coloring._device_seeds(dtype)

    def single_jvp(seed: jax.Array) -> jax.Array:
        tangents = _build_tangents_from_seed(seed, args, sparsity)
        return flatten_pytree(jvp_fn(*tangents))

    J_compressed = _chunked_vmap(single_jvp, seeds, chunk_size)
    return J_compressed, y, aux


# HVPs over the selected input space


def _grad_with_aux(
    f: Callable[..., Any],
    f_aux: Callable[..., Any] | None,
    grad_argnums: int | tuple[int, ...],
) -> Callable[..., Any]:
    """``jax.grad`` returning ``(grads, aux)``, with ``aux=None`` when no ``f_aux``.

    Normalizing the aux output lets callers thread it through
    ``jax.linearize(..., has_aux=True)`` / ``jax.vjp(..., has_aux=True)``
    uniformly, so aux rides along with the existing forward pass
    instead of costing an extra ``f`` call.
    """
    if f_aux is not None:
        return jax.grad(f_aux, argnums=grad_argnums, has_aux=True)
    grad_fn = jax.grad(f, argnums=grad_argnums)
    return lambda *primals: (grad_fn(*primals), None)


def _compute_hvps(
    f: Callable[..., Any],
    args: tuple[Any, ...],
    coloring: ColoredPattern,
    chunk_size: int | None,
    *,
    f_aux: Callable[..., Any] | None = None,
) -> tuple[jax.Array, Any]:
    """One HVP per color for a scalar-valued multi-positional ``f``.

    ``f_aux`` is the aux-preserving variant of ``f`` (returns ``(out, aux)``).
    When given, aux is extracted from the forward pass of the HVPs
    where the mode allows and returned alongside the compressed rows;
    otherwise the returned aux is ``None``.
    """
    sparsity = coloring.sparsity
    dtype = _uniform_selected_dtype(args, sparsity)
    grad_argnums = sparsity.argnums

    seeds = coloring._device_seeds(dtype)
    _assert_hessian_mode(coloring.mode)
    match coloring.mode:
        case "fwd_over_rev":
            grad_fn = _grad_with_aux(f, f_aux, grad_argnums)
            _, hvp_fn, aux = jax.linearize(grad_fn, *args, has_aux=True)

            def single_hvp(v: jax.Array) -> jax.Array:
                tangents = _build_tangents_from_seed(v, args, sparsity)
                tangent_out = hvp_fn(*tangents)
                return _flatten_grad_output(tangent_out)

        case "rev_over_fwd":
            # The forward passes happen inside the vmapped HVPs below,
            # so aux needs one dedicated ``f`` call.
            aux = f_aux(*args)[1] if f_aux is not None else None

            def single_hvp(v: jax.Array) -> jax.Array:
                tangents = _build_tangents_from_seed(v, args, sparsity)

                def inner(*primals: Any) -> jax.Array:
                    _, out_tangent = jax.jvp(f, primals, tangents)
                    return out_tangent

                grads = jax.grad(inner, argnums=grad_argnums)(*args)
                return _flatten_grad_output(grads)

        case "rev_over_rev":
            grad_fn = _grad_with_aux(f, f_aux, grad_argnums)
            _, hvp_fn, aux = jax.vjp(grad_fn, *args, has_aux=True)

            def single_hvp(v: jax.Array) -> jax.Array:
                cotangent_out = _build_grad_output_from_seed(v, sparsity)
                cotangents = hvp_fn(cotangent_out)
                return _flatten_selected_cotangents(cotangents, sparsity)

        case _ as unreachable:
            assert_never(unreachable)  # ty: ignore[type-assertion-failure]

    H_compressed = _chunked_vmap(single_hvp, seeds, chunk_size)
    return H_compressed, aux


def _grad_with_value_and_aux(
    f: Callable[..., Any],
    f_aux: Callable[..., Any] | None,
    grad_argnums: int | tuple[int, ...],
) -> Callable[..., Any]:
    """``jax.grad`` returning ``(grads, (value, aux))``, ``aux=None`` without ``f_aux``.

    Returning the primal value as the aux output of
    ``jax.linearize`` / ``jax.vjp`` keeps it out of the differentiated outputs,
    so it rides along with the forward pass inside ``grad``
    without inflating HVP applications with dead value (co)tangents.
    """
    if f_aux is not None:
        val_and_grad_aux = jax.value_and_grad(f_aux, argnums=grad_argnums, has_aux=True)

        def wrapped(*primals: Any) -> tuple[Any, tuple[jax.Array, Any]]:
            (value, aux), grads = val_and_grad_aux(*primals)
            return grads, (value, aux)
    else:
        val_and_grad = jax.value_and_grad(f, argnums=grad_argnums)

        def wrapped(*primals: Any) -> tuple[Any, tuple[jax.Array, Any]]:
            value, grads = val_and_grad(*primals)
            return grads, (value, None)

    return wrapped


def _value_and_compute_hvps(
    f: Callable[..., Any],
    args: tuple[Any, ...],
    coloring: ColoredPattern,
    chunk_size: int | None,
    *,
    f_aux: Callable[..., Any] | None = None,
) -> tuple[jax.Array, jax.Array, Any]:
    """``f(*args)``, one HVP per color, and aux for a scalar-valued ``f``.

    Value (and aux, when ``f_aux`` is given) is free for
    ``fwd_over_rev`` / ``rev_over_rev``;
    ``rev_over_fwd`` computes both with one extra ``f`` call.
    Returns ``(value, compressed, aux)``; ``aux`` is ``None`` without ``f_aux``.
    """
    sparsity = coloring.sparsity
    dtype = _uniform_selected_dtype(args, sparsity)
    grad_argnums = sparsity.argnums

    seeds = coloring._device_seeds(dtype)
    _assert_hessian_mode(coloring.mode)
    match coloring.mode:
        case "fwd_over_rev":
            grad_fn = _grad_with_value_and_aux(f, f_aux, grad_argnums)
            _, hvp_fn, (value, aux) = jax.linearize(grad_fn, *args, has_aux=True)

            def single_hvp(v: jax.Array) -> jax.Array:
                tangents = _build_tangents_from_seed(v, args, sparsity)
                tangent_out = hvp_fn(*tangents)
                return _flatten_grad_output(tangent_out)

        case "rev_over_fwd":
            # The forward passes happen inside the vmapped HVPs below,
            # so value and aux need one dedicated ``f`` call.
            if f_aux is not None:
                out, aux = f_aux(*args)
                value = jnp.asarray(out)
            else:
                value, aux = jnp.asarray(f(*args)), None

            def single_hvp(v: jax.Array) -> jax.Array:
                tangents = _build_tangents_from_seed(v, args, sparsity)

                def inner(*primals: Any) -> jax.Array:
                    _, out_tangent = jax.jvp(f, primals, tangents)
                    return out_tangent

                grads = jax.grad(inner, argnums=grad_argnums)(*args)
                return _flatten_grad_output(grads)

        case "rev_over_rev":
            grad_fn = _grad_with_value_and_aux(f, f_aux, grad_argnums)
            _, hvp_fn, (value, aux) = jax.vjp(grad_fn, *args, has_aux=True)

            def single_hvp(v: jax.Array) -> jax.Array:
                cotangent_out = _build_grad_output_from_seed(v, sparsity)
                cotangents = hvp_fn(cotangent_out)
                return _flatten_selected_cotangents(cotangents, sparsity)

        case _ as unreachable:
            assert_never(unreachable)  # ty: ignore[type-assertion-failure]

    H_compressed = _chunked_vmap(single_hvp, seeds, chunk_size)
    return value, H_compressed, aux


# Decompression


def _decompress_data(coloring: ColoredPattern, compressed: jax.Array) -> jax.Array:
    """Extract sparse data values from compressed gradient rows.

    Uses pre-computed gather indices on the ``ColoredPattern``
    to vectorize the decompression step
    (no Python loop over nnz entries).
    """
    # Symmetric (star) colorings map both (i, j) and (j, i)
    # to the same (color, element) gather pair,
    # so the indices are not unique.
    # The transpose of gather is scatter-add,
    # where unique_indices=True with duplicates is undefined behavior:
    # differentiating through the decompressed data
    # could silently produce wrong gradients on GPU/TPU.
    return jax.lax.gather(
        compressed,
        coloring._gather_indices,
        dimension_numbers=jax.lax.GatherDimensionNumbers(
            offset_dims=(),
            collapsed_slice_dims=(0, 1),
            start_index_map=(0, 1),
        ),
        slice_sizes=(1, 1),
        unique_indices=not coloring.symmetric,
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

    For simple single-array outputs, constructs BCOO and scipy outputs directly
    from known indices.
    For PyTree outputs, assembles per-leaf blocks:
    BCOO blocks directly from the pattern indices,
    dense blocks by scattering to a dense matrix first.
    """
    sparsity = coloring.sparsity

    # Fast path: single flat array output with BCOO format.
    # Use the known sparsity pattern indices directly, avoiding fromdense.
    if output_format == "bcoo" and _is_simple_output(out_struct, sparsity):
        if data.dtype == dtypes.float0:
            data = jnp.zeros(sparsity.nnz, dtype=jnp.float_)
        out_shape = jax.tree_util.tree_leaves(out_struct)[0].shape
        in_shape = sparsity.leaf_shapes[0]
        return sparsity.to_bcoo(data=data).reshape((*out_shape, *in_shape))

    match output_format:
        case "scipy_coo" | "scipy_csr" | "scipy_csc":
            # Build directly from the pattern indices.
            # This keeps structural zeros as explicit entries
            # and avoids materializing a dense intermediate.
            _assert_scipy_supported_jacobian(out_struct, sparsity, output_format)
            return _sparsity_to_scipy(sparsity, data, output_format)
        case "bcoo" | "dense":
            return _assemble_jacobian(coloring, data, output_format, out_struct)
        case "numpy_dense":
            jac = _assemble_jacobian(coloring, data, "dense", out_struct)
            return _to_numpy_pytree(jac)
        case _ as unreachable:
            assert_never(unreachable)


def _build_hessian(
    coloring: ColoredPattern,
    data: jax.Array,
    output_format: OutputFormat,
) -> Any:
    """Build Hessian output from sparse data, avoiding BCOO.fromdense under JIT.

    For simple single-input cases, constructs BCOO and scipy outputs directly
    from known indices.
    For PyTree inputs, assembles per-leaf blocks:
    BCOO blocks directly from the pattern indices,
    dense blocks by scattering to a dense matrix first.
    """
    sparsity = coloring.sparsity

    # Fast path: single input leaf with BCOO format and trivial pytree structure.
    if output_format == "bcoo" and _is_simple_input(sparsity):
        in_shape = sparsity.leaf_shapes[0]
        return sparsity.to_bcoo(data=data).reshape((*in_shape, *in_shape))

    match output_format:
        case "scipy_coo" | "scipy_csr" | "scipy_csc":
            # Build directly from the pattern indices.
            # This keeps structural zeros as explicit entries
            # and avoids materializing a dense intermediate.
            _assert_scipy_supported_hessian(sparsity, output_format)
            return _sparsity_to_scipy(sparsity, data, output_format)
        case "bcoo" | "dense":
            return _assemble_hessian(coloring, data, output_format)
        case "numpy_dense":
            hess = _assemble_hessian(coloring, data, "dense")
            return _to_numpy_pytree(hess)
        case _ as unreachable:
            assert_never(unreachable)


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


def _uniform_selected_dtype(args: tuple[Any, ...], sparsity: SparsityPattern) -> Any:
    """Shared dtype of all selected leaves, for input-space seeds.

    Forward-mode tangents and HVP seeds are sliced from one flat seed vector,
    so every selected leaf must share its dtype;
    mixed dtypes would otherwise fail deep inside ``jvp``/``vjp``.
    Leaves without a ``dtype`` (Python scalars) are weakly typed and skipped.
    """
    found = {
        jnp.dtype(leaf.dtype)
        for leaf in jax.tree_util.tree_leaves(_selected_args(args, sparsity))
        if getattr(leaf, "dtype", None) is not None
    }
    if len(found) > 1:
        names = sorted(d.name for d in found)
        raise TypeError(
            f"Differentiated inputs have mixed dtypes {names}, "
            "which forward and Hessian modes do not support. "
            "Cast the inputs to a common dtype, "
            "or use `mode='rev'` for Jacobians."
        )
    return found.pop() if found else jnp.float_


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


def _make_block_builder(
    coloring: ColoredPattern,
    data: jax.Array,
    output_format: JaxOutputFormat,
) -> Callable[[int, int, int, int], jax.Array | BCOO]:
    """Return a function extracting one ``(row, col)`` index window as a block.

    For BCOO output, blocks are built from the pattern entries in the window,
    all kept as explicit values
    so the block structure matches the detected sparsity pattern
    independent of the evaluation point.
    The window's entry indices are cached on the pattern,
    so repeated evaluations skip the pattern scan.

    For dense output, blocks are sliced from the scattered dense matrix.
    """
    if output_format == "bcoo":
        sparsity = coloring.sparsity

        def build_block(
            row_offset: int, row_size: int, col_offset: int, col_size: int
        ) -> BCOO:
            entry_idx, indices = sparsity._block_indices(
                row_offset, row_size, col_offset, col_size
            )
            return BCOO(
                (data[entry_idx], indices),
                shape=(row_size, col_size),
                unique_indices=True,
            )

        return build_block

    dense = _scatter_dense(coloring, data)

    def slice_block(
        row_offset: int, row_size: int, col_offset: int, col_size: int
    ) -> jax.Array:
        return dense[
            row_offset : row_offset + row_size,
            col_offset : col_offset + col_size,
        ]

    return slice_block


def _assemble_jacobian(
    coloring: ColoredPattern,
    data: jax.Array,
    output_format: JaxOutputFormat,
    out_struct: Any,
) -> Any:
    """Split the flat Jacobian data into per-leaf Jacobian blocks.

    Each block is reshaped to ``(*out_leaf_shape, *in_leaf_shape)`` to match
    ``jax.jacfwd`` / ``jax.jacrev`` output layout.

    For PyTree outputs, the result has structure ``(output_tree, input_tree)``,
    mirroring ``jax.jacobian``.

    BCOO blocks are built directly from the pattern indices,
    keeping structural zeros as explicit entries
    so the block structure is independent of the evaluation point.
    Dense blocks are sliced from the scattered dense matrix.
    """
    sparsity = coloring.sparsity
    build_block = _make_block_builder(coloring, data, output_format)

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
            block = build_block(out_row_offset, out_size, in_col_offset, in_size)
            out_blocks.append(block.reshape((*out_shape, *in_shape)))
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
    coloring: ColoredPattern,
    data: jax.Array,
    output_format: JaxOutputFormat,
) -> Any:
    """Split the flat Hessian data into a nested block grid.

    For each outer leaf, pack the inner axis into the full input-structured
    pytree using the same rules as Jacobian packing, then pack those rows
    again on the outer axis. The result mirrors what
    ``jax.hessian(f, argnums=...)`` returns.

    BCOO blocks are built directly from the pattern indices,
    keeping structural zeros as explicit entries
    so the block structure is independent of the evaluation point.
    Dense blocks are sliced from the scattered dense matrix.
    """
    sparsity = coloring.sparsity
    build_block = _make_block_builder(coloring, data, output_format)

    leaf_shapes = sparsity.leaf_shapes
    leaf_sizes = sparsity.leaf_sizes

    leaf_blocks: list[list[jax.Array | BCOO]] = []
    row_offset = 0
    for row_size, row_shape in zip(leaf_sizes, leaf_shapes, strict=True):
        col_offset = 0
        row_blocks: list[jax.Array | BCOO] = []
        for col_size, col_shape in zip(leaf_sizes, leaf_shapes, strict=True):
            block = build_block(row_offset, row_size, col_offset, col_size)
            row_blocks.append(block.reshape(row_shape + col_shape))
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
