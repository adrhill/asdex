"""Jacobian and Hessian sparsity detection via jaxpr graph analysis."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax._src.core import ClosedJaxpr
from jax._src.interpreters.partial_eval import dce_jaxpr

from asdex._api_utils import (
    _ensure_inbounds,
    _ensure_index,
    avals_from_args,
    merge_sample_inputs,
)
from asdex._defaults import _DEFAULT_ARGNUMS, _DEFAULT_HAS_AUX
from asdex._docstrings import _fill_doc
from asdex._pattern import SparsityPattern
from asdex.detection._interpret import prop_jaxpr
from asdex.detection._interpret._common import empty_index_sets


@_fill_doc
def jacobian_sparsity(
    f: Callable,
    *args: Any,
    argnums: int | Sequence[int] = _DEFAULT_ARGNUMS,
    has_aux: bool = _DEFAULT_HAS_AUX,
    **kwargs: Any,
) -> SparsityPattern:
    """Detect global Jacobian sparsity pattern for ``f``.

    Analyzes the computation graph structure directly,
    without evaluating any derivatives.
    The result is valid for all inputs.

    Args:
        f: Function whose Jacobian sparsity pattern is to be detected.
        *args: {sample_args}
        argnums: {argnums}
        has_aux: {has_aux_detect}
        **kwargs: {sample_kwargs_detect}

    Returns:
        SparsityPattern of shape ``(m, n_selected)``
            where ``m = prod(output_shape)`` and ``n_selected`` is the total
            flat size of the selected inputs.
    """
    argnums = _ensure_index(argnums)
    args, f, argnums = merge_sample_inputs(f, args, kwargs, argnums)
    avals = avals_from_args(args)
    selected = _argnums_tuple(argnums, len(args))

    # Resolve negative indices while preserving int-vs-tuple distinction
    argnums_resolved = selected[0] if isinstance(argnums, int) else selected

    f_out = _strip_aux(f) if has_aux else f

    closed_jaxpr = jax.make_jaxpr(f_out)(*args)
    closed_jaxpr = _dce_closed_jaxpr(closed_jaxpr)
    out_aval = jax.eval_shape(f_out, *args)
    m = sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(out_aval))

    input_indices, n_selected = _build_input_indices(avals, selected)
    out_indices = _run_prop(closed_jaxpr, input_indices)
    rows, cols = _coo_from_index_sets(out_indices)
    return SparsityPattern.from_coo(
        rows,
        cols,
        (m, n_selected),
        input_avals=avals,
        argnums=argnums_resolved,
    )


@_fill_doc
def hessian_sparsity(
    f: Callable,
    *args: Any,
    argnums: int | Sequence[int] = _DEFAULT_ARGNUMS,
    has_aux: bool = _DEFAULT_HAS_AUX,
    **kwargs: Any,
) -> SparsityPattern:
    """Detect global Hessian sparsity pattern for a scalar-valued ``f``.

    Analyzes the Jacobian sparsity of the gradient function,
    without evaluating any derivatives.
    The result is valid for all inputs.

    If ``f`` returns a squeezable shape like ``(1,)`` or ``(1, 1)``,
    it is automatically squeezed to scalar.

    Args:
        f: Scalar-valued function taking one or more positional arrays.
        *args: {sample_args}
        argnums: {argnums}
        has_aux: {has_aux_detect}
        **kwargs: {sample_kwargs_detect}

    Returns:
        Square SparsityPattern over the combined, selected input space.
    """
    argnums = _ensure_index(argnums)
    args, f, argnums = merge_sample_inputs(f, args, kwargs, argnums)

    f_out = _strip_aux(f) if has_aux else f
    f_scalar = _ensure_scalar(f_out, args)
    grad_fn = jax.grad(f_scalar, argnums=argnums)
    return jacobian_sparsity(grad_fn, *args, argnums=argnums)


# Internal helpers


def _argnums_tuple(argnums: int | tuple[int, ...], num_args: int) -> tuple[int, ...]:
    """Normalize argnums to a tuple and resolve negative indices."""
    tup = (argnums,) if isinstance(argnums, int) else argnums
    return _ensure_inbounds(num_args, tup)


def _strip_aux(f: Callable) -> Callable:
    """Drop the aux output of a ``has_aux=True`` function."""
    return lambda *xs: f(*xs)[0]


def _dce_closed_jaxpr(closed_jaxpr: ClosedJaxpr) -> ClosedJaxpr:
    """Remove equations unused by outputs while preserving all inputs.

    Uses ``instantiate=True`` so DCE keeps all input variables even if unused.
    This is needed because input_indices must align with the original inputs.
    """
    jaxpr = closed_jaxpr.jaxpr
    used_outputs = [True] * len(jaxpr.outvars)
    new_jaxpr, _ = dce_jaxpr(jaxpr, used_outputs, instantiate=True)
    return ClosedJaxpr(new_jaxpr, closed_jaxpr.consts)


def _build_input_indices(
    avals: tuple[Any, ...], selected: tuple[int, ...]
) -> tuple[list[list], int]:
    """Seed per-leaf index sets in ``jax.make_jaxpr`` leaf order.

    Selected positions get identity index sets over a contiguous column
    space; non-selected positions get empty index sets so dependencies
    flowing through them do not appear in the pattern.

    Column indices are assigned in ``selected`` (argnums) order so that
    the sparsity pattern columns match the order expected by decompression.

    Returns ``(input_indices, n_selected)``.
    """
    # First pass: assign column offsets in argnums order
    col_offsets: dict[int, int] = {}
    offset = 0
    for pos_idx in selected:
        col_offsets[pos_idx] = offset
        leaves = jax.tree_util.tree_leaves(avals[pos_idx])
        offset += sum(int(leaf.size) for leaf in leaves)
    n_selected = offset

    # Second pass: build input_indices in jaxpr input order
    input_indices: list[list] = []
    for pos_idx, pos_aval in enumerate(avals):
        leaves = jax.tree_util.tree_leaves(pos_aval)
        if pos_idx in col_offsets:
            col_offset = col_offsets[pos_idx]
            for leaf in leaves:
                size = int(leaf.size)
                input_indices.append([{col_offset + j} for j in range(size)])
                col_offset += size
        else:
            for leaf in leaves:
                input_indices.append(empty_index_sets(int(leaf.size)))  # noqa: PERF401
    return input_indices, n_selected


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
    """Flatten per-output dependency sets into COO rows/cols.

    Columns are sorted within each row,
    so detected patterns are row-major sorted and deterministic
    (set iteration order is not guaranteed)
    and ``SparsityPattern.to_bcoo`` can mark its output as sorted.
    """
    rows: list[int] = []
    cols: list[int] = []
    for i, deps in enumerate(out_indices):
        for j in sorted(deps):
            rows.append(i)
            cols.append(j)
    return rows, cols


def _ensure_scalar(f: Callable, args: tuple[Any, ...]) -> Callable:
    """Ensure ``f`` returns a scalar, auto-squeezing if possible.

    If ``f`` already returns shape ``()``, it is returned unchanged.
    If squeezing the output yields a scalar (e.g. shape ``(1,)``),
    a wrapped version is returned.
    Otherwise, raises ``ValueError``.
    """
    out = jax.eval_shape(f, *args)
    if not hasattr(out, "shape"):
        raise ValueError(
            f"Expected scalar-valued function, but f returns a PyTree: {type(out).__name__}."
        )
    if out.shape == ():
        return f
    squeezed_shape = jax.eval_shape(lambda *xs: jnp.squeeze(f(*xs)), *args).shape
    if squeezed_shape != ():
        raise ValueError(
            f"Expected scalar-valued function, but f has output shape {out.shape}."
        )
    return lambda *xs: jnp.squeeze(f(*xs))
