"""Sparse Jacobian and Hessian computation using coloring and AD."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, assert_never

import jax
import jax.numpy as jnp
from jax import dtypes
from jax.experimental.sparse import BCOO

from asdex._input import bind_kwargs, validate_input_dtypes, validate_output_dtypes
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

# Public API: one-shot entry points


def jacobian(
    f: Callable[..., Any],
    input_shape: Any,
    *,
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    mode: JacobianMode | None = None,
    symmetric: bool = False,
    output_format: OutputFormat = "bcoo",
) -> Callable[..., Any]:
    """Detect sparsity, color, and return a function computing sparse Jacobians.

    Combines [`jacobian_coloring`][asdex.jacobian_coloring]
    and [`jacobian_from_coloring`][asdex.jacobian_from_coloring]
    in one call.

    Args:
        f: Function taking one or more positional arrays and returning an array.
        input_shape: A sequence with one entry per positional argument of ``f``,
            specifying the shape and dtype of that argument
            (see [`jacobian_sparsity`][asdex.jacobian_sparsity]).
        argnums: Positions of the positional arguments to differentiate with
            respect to, mirroring ``jax.jacfwd`` / ``jax.jacrev``.
            Defaults to ``0``.
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
            ``"bcoo"`` returns a sparse matrix of type ``jax.experimental.sparse.BCOO`` (default),
            ``"dense"`` returns a dense matrix of type ``jax.Array``.

    Returns:
        A function that takes the same positional args as ``f`` and returns
            a pytree of Jacobian blocks matching ``argnums``, with each leaf
            shaped ``(*out_shape, *in_leaf_shape)``.
            The block type depends on ``output_format``
            (``jax.experimental.sparse.BCOO`` by default, or ``jax.Array``
            when ``"dense"``).
    """
    coloring = _jacobian_coloring(
        f,
        input_shape,
        argnums=argnums,
        has_aux=has_aux,
        mode=mode,
        symmetric=symmetric,
    )
    return jacobian_from_coloring(
        f,
        coloring,
        output_format=output_format,
        has_aux=has_aux,
        holomorphic=holomorphic,
        allow_int=allow_int,
    )


def value_and_jacobian(
    f: Callable[..., Any],
    input_shape: Any,
    *,
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    mode: JacobianMode | None = None,
    symmetric: bool = False,
    output_format: OutputFormat = "bcoo",
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
    coloring = _jacobian_coloring(
        f,
        input_shape,
        argnums=argnums,
        has_aux=has_aux,
        mode=mode,
        symmetric=symmetric,
    )
    return value_and_jacobian_from_coloring(
        f,
        coloring,
        output_format=output_format,
        has_aux=has_aux,
        holomorphic=holomorphic,
        allow_int=allow_int,
    )


def hessian(
    f: Callable[..., Any],
    input_shape: Any,
    *,
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    mode: HessianMode | None = None,
    symmetric: bool = True,
    output_format: OutputFormat = "bcoo",
) -> Callable[..., Any]:
    """Detect sparsity, color, and return a function computing sparse Hessians.

    If ``f`` returns a squeezable shape like ``(1,)`` or ``(1, 1)``,
    it is automatically squeezed to scalar.
    """
    coloring = _hessian_coloring(
        f,
        input_shape,
        argnums=argnums,
        has_aux=has_aux,
        mode=mode,
        symmetric=symmetric,
    )
    return hessian_from_coloring(
        f,
        coloring,
        output_format=output_format,
        has_aux=has_aux,
        holomorphic=holomorphic,
        allow_int=allow_int,
    )


def value_and_hessian(
    f: Callable[..., Any],
    input_shape: Any,
    *,
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
    mode: HessianMode | None = None,
    symmetric: bool = True,
    output_format: OutputFormat = "bcoo",
) -> Callable[..., Any]:
    """Detect sparsity, color, and return a function computing value and sparse Hessian.

    Like [`hessian`][asdex.hessian], but also returns the primal value
    ``f(*args)`` without an extra forward pass.
    """
    coloring = _hessian_coloring(
        f,
        input_shape,
        argnums=argnums,
        has_aux=has_aux,
        mode=mode,
        symmetric=symmetric,
    )
    return value_and_hessian_from_coloring(
        f,
        coloring,
        output_format=output_format,
        has_aux=has_aux,
        holomorphic=holomorphic,
        allow_int=allow_int,
    )


# Public API: ``*_from_coloring`` entry points


def jacobian_from_coloring(
    f: Callable[..., Any],
    coloring: ColoredPattern,
    output_format: OutputFormat = "bcoo",
    *,
    has_aux: bool = False,
    holomorphic: bool = False,
    allow_int: bool = False,
) -> Callable[..., Any]:
    """Build a sparse Jacobian function from a pre-computed coloring.

    Uses row coloring + VJPs or column coloring + JVPs,
    depending on which needs fewer colors.

    The returned callable accepts ``*args, **kwargs``; kwargs are forwarded
    to ``f`` at call time (matching ``jax.jacfwd`` / ``jax.jacrev``).
    """
    _assert_output_format(output_format)

    def jac_fn(*args: Any, **kwargs: Any) -> Any:
        f_bound = bind_kwargs(f, kwargs)
        return _eval_jacobian(
            f_bound,
            args,
            coloring,
            output_format,
            has_aux=has_aux,
            holomorphic=holomorphic,
            allow_int=allow_int,
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
) -> Callable[..., Any]:
    """Build a sparse Hessian function from a pre-computed coloring.

    Uses symmetric (star) coloring and Hessian-vector products by default.
    """
    _assert_output_format(output_format)

    def hess_fn(*args: Any, **kwargs: Any) -> Any:
        f_bound = bind_kwargs(f, kwargs)
        return _eval_hessian(
            f_bound,
            args,
            coloring,
            output_format,
            has_aux=has_aux,
            holomorphic=holomorphic,
            allow_int=allow_int,
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
) -> Callable[..., Any]:
    """Build a function computing value and sparse Jacobian from a pre-computed coloring."""
    _assert_output_format(output_format)

    def val_jac_fn(*args: Any, **kwargs: Any) -> Any:
        f_bound = bind_kwargs(f, kwargs)
        return _eval_value_and_jacobian(
            f_bound,
            args,
            coloring,
            output_format,
            has_aux=has_aux,
            holomorphic=holomorphic,
            allow_int=allow_int,
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
) -> Callable[..., Any]:
    """Build a function computing value and sparse Hessian from a pre-computed coloring."""
    _assert_output_format(output_format)

    def val_hess_fn(*args: Any, **kwargs: Any) -> Any:
        f_bound = bind_kwargs(f, kwargs)
        return _eval_value_and_hessian(
            f_bound,
            args,
            coloring,
            output_format,
            has_aux=has_aux,
            holomorphic=holomorphic,
            allow_int=allow_int,
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
    out_shape = jax.eval_shape(f_out, *args).shape

    if m == 0 or sparsity.nnz == 0:
        dense = jnp.zeros((m, n_selected))
        jac = _assemble_jacobian(dense, sparsity, output_format, out_shape)
        if has_aux:
            _, aux = f(*args)
            return jac, aux
        return jac

    _assert_jacobian_mode(coloring.mode)
    match coloring.mode:
        case "rev":
            compressed, y, aux = _jacobian_rows(
                f, args, coloring, out_shape, has_aux=has_aux
            )
        case "fwd":
            compressed, y, aux = _jacobian_cols(f, args, coloring, has_aux=has_aux)
        case _ as unreachable:
            assert_never(unreachable)  # ty: ignore[type-assertion-failure]

    validate_output_dtypes(y, coloring.mode, holomorphic)
    data = _decompress_data(coloring, compressed)
    dense = _scatter_dense(coloring, data)
    jac = _assemble_jacobian(dense, sparsity, output_format, out_shape)
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
    out_shape = jax.eval_shape(f_out, *args).shape

    if m == 0 or sparsity.nnz == 0:
        dense = jnp.zeros((m, n_selected))
        empty = _assemble_jacobian(dense, sparsity, output_format, out_shape)
        if has_aux:
            value, aux = f(*args)
            return (jnp.asarray(value), aux), empty
        value = jnp.asarray(f(*args))
        return value, empty

    _assert_jacobian_mode(coloring.mode)
    match coloring.mode:
        case "rev":
            compressed, y, aux = _jacobian_rows(
                f, args, coloring, out_shape, has_aux=has_aux
            )
        case "fwd":
            compressed, y, aux = _jacobian_cols(f, args, coloring, has_aux=has_aux)
        case _ as unreachable:
            assert_never(unreachable)  # ty: ignore[type-assertion-failure]

    validate_output_dtypes(y, coloring.mode, holomorphic)
    data = _decompress_data(coloring, compressed)
    dense = _scatter_dense(coloring, data)
    jac = _assemble_jacobian(dense, sparsity, output_format, out_shape)
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

    compressed = _compute_hvps(f_scalar, args, coloring)
    data = _decompress_data(coloring, compressed)
    dense = _scatter_dense(coloring, data)
    hess = _assemble_hessian(dense, sparsity, output_format)
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

    value, compressed = _value_and_compute_hvps(f_scalar, args, coloring)
    data = _decompress_data(coloring, compressed)
    dense = _scatter_dense(coloring, data)
    hess = _assemble_hessian(dense, sparsity, output_format)
    if has_aux:
        _, aux = f(*args)
        return (value, aux), hess
    return value, hess


# Jacobian rows / cols over the selected input space


def _jacobian_rows(
    f: Callable[..., Any],
    args: tuple[Any, ...],
    coloring: ColoredPattern,
    out_shape: tuple[int, ...],
    *,
    has_aux: bool,
) -> tuple[jax.Array, jax.Array, Any]:
    """Row-coloring VJPs over the combined selected input space.

    Returns ``(compressed, y, aux)``; ``aux`` is ``None`` when ``has_aux=False``.
    """
    sparsity = coloring.sparsity
    if has_aux:
        y, vjp_fn, aux = jax.vjp(f, *args, has_aux=True)
    else:
        y, vjp_fn = jax.vjp(f, *args)
        aux = None
    dtype = y.dtype
    seeds = jnp.asarray(coloring._seed_matrix, dtype=dtype)

    def single_vjp(seed: jax.Array) -> jax.Array:
        cotangents = vjp_fn(seed.reshape(out_shape))
        return _flatten_selected_cotangents(cotangents, sparsity)

    return jax.vmap(single_vjp)(seeds), y, aux


def _jacobian_cols(
    f: Callable[..., Any],
    args: tuple[Any, ...],
    coloring: ColoredPattern,
    *,
    has_aux: bool,
) -> tuple[jax.Array, jax.Array, Any]:
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
        return jvp_fn(*tangents).ravel()

    return jax.vmap(single_jvp)(seeds), y, aux


# HVPs over the selected input space


def _compute_hvps(
    f: Callable[..., Any],
    args: tuple[Any, ...],
    coloring: ColoredPattern,
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

    return jax.vmap(single_hvp)(seeds)


def _value_and_compute_hvps(
    f: Callable[..., Any],
    args: tuple[Any, ...],
    coloring: ColoredPattern,
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

    return value, jax.vmap(single_hvp)(seeds)


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


def _empty_result(
    shape: tuple[int, ...], output_format: OutputFormat
) -> BCOO | jax.Array:
    """Return an all-zero matrix in the requested format."""
    match output_format:
        case "bcoo":
            return BCOO(
                (jnp.array([]), jnp.zeros((0, 2), dtype=jnp.int32)), shape=shape
            )
        case "dense":
            return jnp.zeros(shape)
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

    selected_positions = set(sparsity._argnums_tuple)
    tangents: list[Any] = []
    chunk_idx = 0
    for pos_idx, (arg, aval) in enumerate(zip(args, sparsity.input_avals, strict=True)):
        del arg
        aval_leaves = jax.tree_util.tree_leaves(aval)
        aval_tree = jax.tree_util.tree_structure(aval)
        if pos_idx in selected_positions:
            leaf_tangents = [
                chunks[chunk_idx + k].reshape(leaf_shapes[chunk_idx + k])
                for k in range(len(aval_leaves))
            ]
            chunk_idx += len(aval_leaves)
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
    """
    selected = tuple(cotangents[i] for i in sparsity._argnums_tuple)
    leaves = jax.tree_util.tree_leaves(selected)
    if not leaves:
        return jnp.zeros((0,))
    return jnp.concatenate([leaf.ravel() for leaf in leaves])


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
    out_shape: tuple[int, ...],
) -> Any:
    """Split a ``(m, n_selected)`` dense matrix into per-leaf Jacobian blocks.

    Each block is reshaped to ``(*out_shape, *in_leaf_shape)`` to match
    ``jax.jacfwd`` / ``jax.jacrev`` output layout.
    The result mirrors ``sparsity.example_input`` (single pytree when
    ``argnums`` is an int, tuple of pytrees when it is a tuple).
    """
    leaf_shapes = sparsity.leaf_shapes
    leaf_sizes = sparsity.leaf_sizes

    blocks: list[jax.Array | BCOO] = []
    offset = 0
    for size, shape in zip(leaf_sizes, leaf_shapes, strict=True):
        chunk = dense[:, offset : offset + size]
        block: jax.Array | BCOO = chunk.reshape((*out_shape, *shape))
        if output_format == "bcoo":
            block = BCOO.fromdense(block)
        blocks.append(block)
        offset += size

    return _group_blocks_by_argnums(blocks, sparsity)


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
