"""Jacobian and Hessian sparsity detection via jaxpr graph analysis."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from asdex._pytree_shapes import (
    _is_shape_leaf,
    compute_argnums_mask,
    flatten_shapes,
    is_multi_positional,
)
from asdex.detection._interpret import prop_jaxpr
from asdex.detection._interpret._commons import (
    empty_index_sets,
    identity_index_sets,
)
from asdex.pattern import SparsityPattern


def jacobian_sparsity(
    f: Callable,
    input_shape: int | tuple[int, ...] | None = None,
    *,
    input_shapes: Any = None,
    argnums: int | Sequence[int] | None = None,
) -> SparsityPattern:
    """Detect global Jacobian sparsity pattern for ``f``.

    Analyzes the computation graph structure directly,
    without evaluating any derivatives.
    The result is valid for all inputs.

    Args:
        f: Function taking one or more arrays and returning an array.
            In the single-input case, ``f(x)`` takes a single array.
            In the multi-input case, ``f(*xs)`` takes multiple positional
            arrays when ``input_shapes`` is a top-level tuple, or ``f(pytree)``
            takes a single pytree when ``input_shapes`` is itself a pytree
            (e.g. a dict).
        input_shape: Shape of the single input array (single-input mode).
            Mutually exclusive with ``input_shapes``.
            An integer is treated as a 1D length.
        input_shapes: Pytree of shapes, one per input leaf (multi-input mode).
            Mutually exclusive with ``input_shape``.
        argnums: Which positional arguments to differentiate with respect to,
            mirroring ``jax.grad(fun, argnums=...)``.
            Defaults to ``None`` (all positional args).
            Only supported for multi-positional ``input_shapes``.

    Returns:
        SparsityPattern of shape ``(m, n_selected)``
            where ``m = prod(output_shape)`` and ``n_selected`` is the total
            flat size of the selected inputs.
    """
    _assert_single_or_multi(input_shape, input_shapes)

    if input_shapes is None:
        # Single-input path.
        shape = input_shape
        assert shape is not None
        n = shape if isinstance(shape, int) else math.prod(shape)
        shape_tuple = (shape,) if isinstance(shape, int) else tuple(shape)

        dummy_input = jnp.zeros(shape_tuple)
        closed_jaxpr = jax.make_jaxpr(f)(dummy_input)
        m = int(jax.eval_shape(f, dummy_input).size)

        input_indices = [identity_index_sets(n)]
        out_indices = _run_prop(closed_jaxpr, input_indices)

        rows, cols = _coo_from_index_sets(out_indices)
        return SparsityPattern.from_coo(rows, cols, (m, n), input_shape=shape_tuple)

    # Multi-input path.
    multi_positional = is_multi_positional(input_shapes)
    leaf_shapes, _, leaf_sizes = flatten_shapes(input_shapes)
    selected_mask = compute_argnums_mask(
        argnums, input_shapes, multi_positional=multi_positional
    )
    if not any(selected_mask):
        raise ValueError("`argnums` selects no inputs; nothing to differentiate.")

    input_indices = _build_multi_input_indices(leaf_sizes, selected_mask)
    n_selected = sum(s for s, sel in zip(leaf_sizes, selected_mask, strict=True) if sel)

    dummy_pytree = _dummy_from_shapes(input_shapes, leaf_shapes)
    if multi_positional:
        closed_jaxpr = jax.make_jaxpr(f)(*dummy_pytree)
        out_aval = jax.eval_shape(f, *dummy_pytree)
    else:
        closed_jaxpr = jax.make_jaxpr(f)(dummy_pytree)
        out_aval = jax.eval_shape(f, dummy_pytree)
    m = sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(out_aval))

    out_indices = _run_prop(closed_jaxpr, input_indices)
    rows, cols = _coo_from_index_sets(out_indices)
    return SparsityPattern.from_coo(
        rows,
        cols,
        (m, n_selected),
        input_shape=_canonicalize_shapes(input_shapes, leaf_shapes),
        selected_mask=tuple(selected_mask),
        argnums=_canonicalize_argnums(argnums),
    )


def _canonicalize_argnums(
    argnums: int | Sequence[int] | None,
) -> int | tuple[int, ...] | None:
    """Normalize argnums for storage (preserving int vs tuple distinction)."""
    if argnums is None or isinstance(argnums, int):
        return argnums
    return tuple(argnums)


def hessian_sparsity(
    f: Callable,
    input_shape: int | tuple[int, ...] | None = None,
    *,
    input_shapes: Any = None,
    argnums: int | Sequence[int] | None = None,
) -> SparsityPattern:
    """Detect global Hessian sparsity pattern for a scalar-valued ``f``.

    Analyzes the Jacobian sparsity of the gradient function,
    without evaluating any derivatives.
    The result is valid for all inputs.

    If ``f`` returns a squeezable shape like ``(1,)`` or ``(1, 1)``,
    it is automatically squeezed to scalar.

    Args:
        f: Scalar-valued function taking one or more arrays.
        input_shape: Shape of the single input array (single-input mode).
            Mutually exclusive with ``input_shapes``.
        input_shapes: Pytree of shapes (multi-input mode).
            Mutually exclusive with ``input_shape``.
        argnums: Which positional arguments to differentiate with respect to,
            mirroring ``jax.grad(fun, argnums=...)``.
            Only supported for multi-positional ``input_shapes``.

    Returns:
        Square SparsityPattern over the combined, selected input space.
    """
    _assert_single_or_multi(input_shape, input_shapes)

    if input_shapes is None:
        assert input_shape is not None
        f = _ensure_scalar(f, input_shape)
        return jacobian_sparsity(jax.grad(f), input_shape)

    # Multi-input Hessian: grad w.r.t. selected argnums, then Jacobian of that.
    multi_positional = is_multi_positional(input_shapes)
    _ = compute_argnums_mask(argnums, input_shapes, multi_positional=multi_positional)

    # Resolve argnums for jax.grad: it accepts a tuple of ints (multi-positional)
    # or a single int, same as this API.
    if multi_positional:
        if argnums is None:
            positions = tuple(range(len(input_shapes)))
        elif isinstance(argnums, int):
            positions = (argnums,)
        else:
            positions = tuple(argnums)
        grad_argnums: int | tuple[int, ...] = (
            positions[0] if len(positions) == 1 else positions
        )
    else:
        grad_argnums = 0  # single pytree argument

    f_scalar = _ensure_scalar_multi(f, input_shapes, multi_positional)
    grad_fn = jax.grad(f_scalar, argnums=grad_argnums)

    # The gradient returns a pytree matching the selected inputs.
    # Re-run jacobian_sparsity on this gradient, restricting input_shapes/argnums
    # to the same selected subset so both axes match.
    return jacobian_sparsity(
        grad_fn,
        input_shapes=input_shapes,
        argnums=argnums,
    )


# Internal helpers


def _run_prop(closed_jaxpr, input_indices: list[list]) -> list:
    """Run ``prop_jaxpr`` and concatenate index sets across all output leaves.

    JAX flattens pytree-structured outputs into one ``outvar`` per leaf, so
    concatenating preserves the row ordering used by ``jax.make_jaxpr``.
    """
    jaxpr = closed_jaxpr.jaxpr
    state_consts = {
        var: np.asarray(val)
        for var, val in zip(jaxpr.constvars, closed_jaxpr.consts, strict=False)
    }
    output_indices_list = prop_jaxpr(jaxpr, input_indices, state_consts)
    flat: list = []
    for out_deps in output_indices_list:
        flat.extend(out_deps)
    return flat


def _coo_from_index_sets(
    out_indices: list,
) -> tuple[list[int], list[int]]:
    """Flatten per-output dependency sets into COO rows/cols."""
    rows: list[int] = []
    cols: list[int] = []
    for i, deps in enumerate(out_indices):
        for j in deps:
            rows.append(i)
            cols.append(j)
    return rows, cols


def _build_multi_input_indices(
    leaf_sizes: Sequence[int], selected_mask: Sequence[bool]
) -> list[list]:
    """Seed index sets per leaf: identity over selected leaves, empty otherwise.

    The selected leaves share a contiguous column space in the order they
    appear in the flat leaf list; non-selected leaves get empty sets so any
    dependency that flows through them never shows up in the pattern.
    """
    input_indices: list[list] = []
    offset = 0
    for size, selected in zip(leaf_sizes, selected_mask, strict=True):
        if selected:
            input_indices.append([{offset + j} for j in range(size)])
            offset += size
        else:
            input_indices.append(empty_index_sets(size))
    return input_indices


def _dummy_from_shapes(
    input_shapes: Any, leaf_shapes: Sequence[tuple[int, ...]]
) -> Any:
    """Build a pytree of zero arrays matching ``input_shapes``."""
    del leaf_shapes
    return jax.tree_util.tree_map(
        lambda s: jnp.zeros(tuple(s) if isinstance(s, tuple) else (int(s),)),
        input_shapes,
        is_leaf=_is_shape_leaf,
    )


def _canonicalize_shapes(
    input_shapes: Any, leaf_shapes: Sequence[tuple[int, ...]]
) -> Any:
    """Normalize ints in the spec to 1-tuples so leaves are always shape tuples."""
    del leaf_shapes
    return jax.tree_util.tree_map(
        lambda s: tuple(s) if isinstance(s, tuple) else (int(s),),
        input_shapes,
        is_leaf=_is_shape_leaf,
    )


def _assert_single_or_multi(
    input_shape: int | tuple[int, ...] | None,
    input_shapes: Any,
) -> None:
    """Ensure exactly one of ``input_shape`` / ``input_shapes`` is given."""
    if input_shape is None and input_shapes is None:
        raise TypeError(
            "Must pass either `input_shape` (single-input) "
            "or `input_shapes` (multi-input)."
        )
    if input_shape is not None and input_shapes is not None:
        raise TypeError(
            "`input_shape` (singular) and `input_shapes` (plural) are mutually "
            "exclusive; pass one or the other."
        )


def _ensure_scalar(f: Callable, input_shape: int | tuple[int, ...]) -> Callable:
    """Ensure ``f`` returns a scalar, auto-squeezing if possible.

    If ``f`` already returns shape ``()``, it is returned unchanged.
    If squeezing the output yields a scalar (e.g. shape ``(1,)``),
    a wrapped version is returned.
    Otherwise, raises ``ValueError``.
    """
    out = jax.eval_shape(f, jnp.zeros(input_shape))
    if out.shape == ():
        return f
    squeezed = jax.eval_shape(lambda x: jnp.squeeze(f(x)), jnp.zeros(input_shape))
    if squeezed.shape != ():
        raise ValueError(
            f"Expected scalar-valued function, but f has output shape {out.shape}."
        )
    return lambda x: jnp.squeeze(f(x))


def _ensure_scalar_multi(
    f: Callable, input_shapes: Any, multi_positional: bool
) -> Callable:
    """Multi-input variant of :func:`_ensure_scalar`."""
    leaf_shapes, _, _ = flatten_shapes(input_shapes)
    dummy = _dummy_from_shapes(input_shapes, leaf_shapes)

    out = jax.eval_shape(f, *dummy) if multi_positional else jax.eval_shape(f, dummy)

    if out.shape == ():
        return f

    if multi_positional:
        squeezed_shape = jax.eval_shape(lambda *xs: jnp.squeeze(f(*xs)), *dummy).shape
    else:
        squeezed_shape = jax.eval_shape(lambda x: jnp.squeeze(f(x)), dummy).shape

    if squeezed_shape != ():
        raise ValueError(
            f"Expected scalar-valued function, but f has output shape {out.shape}."
        )

    if multi_positional:
        return lambda *xs: jnp.squeeze(f(*xs))
    return lambda x: jnp.squeeze(f(x))
