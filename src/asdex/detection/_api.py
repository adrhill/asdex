"""Jacobian and Hessian sparsity detection via jaxpr graph analysis."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from asdex._input import normalize_argnums, normalize_input_shape
from asdex.detection._interpret import prop_jaxpr
from asdex.detection._interpret._commons import empty_index_sets
from asdex.pattern import SparsityPattern


def jacobian_sparsity(
    f: Callable,
    input_shape: Any,
    *,
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
) -> SparsityPattern:
    """Detect global Jacobian sparsity pattern for ``f``.

    Analyzes the computation graph structure directly,
    without evaluating any derivatives.
    The result is valid for all inputs.

    Args:
        f: Function taking one or more positional arrays and returning an array.
            Each positional argument may itself be a pytree.
        input_shape: A sequence with one entry per positional argument of ``f``,
            specifying the shape and dtype of that argument.
            Each entry is a pytree whose leaves are
            ``jax.ShapeDtypeStruct``, a shape tuple (e.g. ``(3, 4)``), or a
            bare ``int``.
            The shape-tuple and bare-int forms default to ``jnp.float_``.
        argnums: Positions of the positional arguments to differentiate with
            respect to, mirroring ``jax.grad`` / ``jax.jacfwd``.
            Negative indices are resolved via ``i % len(input_shape)``.
            Defaults to ``0``.
        has_aux: Whether ``f`` returns ``(output, auxiliary_data)``.
            When True, only ``output`` is analyzed for sparsity;
            the auxiliary branch of the computation is not traced.

    Returns:
        SparsityPattern of shape ``(m, n_selected)``
            where ``m = prod(output_shape)`` and ``n_selected`` is the total
            flat size of the selected inputs.
    """
    avals = normalize_input_shape(input_shape)
    argnums = normalize_argnums(argnums, len(avals))
    selected = _as_tuple(argnums)

    f_out = _strip_aux(f) if has_aux else f
    dummy_args = tuple(_dummy_from_avals(pos) for pos in avals)

    closed_jaxpr = jax.make_jaxpr(f_out)(*dummy_args)
    out_aval = jax.eval_shape(f_out, *dummy_args)
    m = sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(out_aval))

    input_indices, n_selected = _build_input_indices(avals, selected)
    out_indices = _run_prop(closed_jaxpr, input_indices)
    rows, cols = _coo_from_index_sets(out_indices)
    return SparsityPattern.from_coo(
        rows,
        cols,
        (m, n_selected),
        input_avals=avals,
        argnums=argnums,
    )


def hessian_sparsity(
    f: Callable,
    input_shape: Any,
    *,
    argnums: int | Sequence[int] = 0,
    has_aux: bool = False,
) -> SparsityPattern:
    """Detect global Hessian sparsity pattern for a scalar-valued ``f``.

    Analyzes the Jacobian sparsity of the gradient function,
    without evaluating any derivatives.
    The result is valid for all inputs.

    If ``f`` returns a squeezable shape like ``(1,)`` or ``(1, 1)``,
    it is automatically squeezed to scalar.

    Args:
        f: Scalar-valued function taking one or more positional arrays.
        input_shape: A sequence with one entry per positional argument of ``f``,
            specifying the shape and dtype of that argument
            (see :func:`jacobian_sparsity`).
        argnums: Positions of the positional arguments to differentiate with
            respect to, mirroring ``jax.grad``.
        has_aux: Whether ``f`` returns ``(scalar_output, auxiliary_data)``.
            When True, aux is stripped before detection.

    Returns:
        Square SparsityPattern over the combined, selected input space.
    """
    avals = normalize_input_shape(input_shape)
    argnums = normalize_argnums(argnums, len(avals))

    f_out = _strip_aux(f) if has_aux else f
    f_scalar = _ensure_scalar(f_out, avals)
    grad_fn = jax.grad(f_scalar, argnums=argnums)
    return jacobian_sparsity(grad_fn, avals, argnums=argnums)


# Internal helpers


def _as_tuple(argnums: int | tuple[int, ...]) -> tuple[int, ...]:
    """``argnums`` as a tuple for indexing."""
    if isinstance(argnums, int):
        return (argnums,)
    return argnums


def _strip_aux(f: Callable) -> Callable:
    """Drop the aux output of a ``has_aux=True`` function."""
    return lambda *xs: f(*xs)[0]


def _dummy_from_avals(aval_tree: Any) -> Any:
    """Build a pytree of ``jnp.zeros`` matching a pytree of ``ShapeDtypeStruct``."""
    return jax.tree_util.tree_map(lambda s: jnp.zeros(s.shape, s.dtype), aval_tree)


def _build_input_indices(
    avals: tuple[Any, ...], selected: tuple[int, ...]
) -> tuple[list[list], int]:
    """Seed per-leaf index sets in ``jax.make_jaxpr`` leaf order.

    Selected positions get identity index sets over a contiguous column
    space; non-selected positions get empty index sets so dependencies
    flowing through them do not appear in the pattern.
    Returns ``(input_indices, n_selected)``.
    """
    input_indices: list[list] = []
    offset = 0
    for pos_idx, pos_aval in enumerate(avals):
        leaves = jax.tree_util.tree_leaves(pos_aval)
        if pos_idx in selected:
            for leaf in leaves:
                size = int(leaf.size)
                input_indices.append([{offset + j} for j in range(size)])
                offset += size
        else:
            for leaf in leaves:
                input_indices.append(empty_index_sets(int(leaf.size)))  # noqa: PERF401
    return input_indices, offset


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


def _ensure_scalar(f: Callable, avals: tuple[Any, ...]) -> Callable:
    """Ensure ``f`` returns a scalar, auto-squeezing if possible.

    If ``f`` already returns shape ``()``, it is returned unchanged.
    If squeezing the output yields a scalar (e.g. shape ``(1,)``),
    a wrapped version is returned.
    Otherwise, raises ``ValueError``.
    """
    dummy = tuple(_dummy_from_avals(pos) for pos in avals)
    out = jax.eval_shape(f, *dummy)
    if out.shape == ():
        return f
    squeezed_shape = jax.eval_shape(lambda *xs: jnp.squeeze(f(*xs)), *dummy).shape
    if squeezed_shape != ():
        raise ValueError(
            f"Expected scalar-valued function, but f has output shape {out.shape}."
        )
    return lambda *xs: jnp.squeeze(f(*xs))
