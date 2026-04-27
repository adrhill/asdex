"""Jacobian and Hessian sparsity detection via jaxpr graph analysis."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from asdex._input import _ensure_inbounds, _ensure_index, avals_from_args
from asdex.detection._interpret import prop_jaxpr
from asdex.detection._interpret._commons import empty_index_sets
from asdex.pattern import SparsityPattern


def jacobian_sparsity(
    f: Callable,
    *args: Any,
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
        *args: Sample inputs specifying the structure, shape, and dtype of each
            positional argument of ``f``.
            Only structure is used; values are ignored.
        argnums: Positions of the positional arguments to differentiate with
            respect to, mirroring ``jax.grad`` / ``jax.jacfwd``.
            Negative indices are resolved via ``i % len(args)``.
            Defaults to ``0``.
        has_aux: Whether ``f`` returns ``(output, auxiliary_data)``.
            When True, only ``output`` is analyzed for sparsity;
            the auxiliary branch of the computation is not traced.

    Returns:
        SparsityPattern of shape ``(m, n_selected)``
            where ``m = prod(output_shape)`` and ``n_selected`` is the total
            flat size of the selected inputs.
    """
    argnums = _ensure_index(argnums)
    avals = avals_from_args(args)
    selected = _argnums_tuple(argnums, len(args))

    f_out = _strip_aux(f) if has_aux else f

    closed_jaxpr = jax.make_jaxpr(f_out)(*args)
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
        argnums=argnums,
    )


def hessian_sparsity(
    f: Callable,
    *args: Any,
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
        *args: Sample inputs specifying the structure, shape, and dtype of each
            positional argument of ``f``.
            Only structure is used; values are ignored.
        argnums: Positions of the positional arguments to differentiate with
            respect to, mirroring ``jax.grad``.
        has_aux: Whether ``f`` returns ``(scalar_output, auxiliary_data)``.
            When True, aux is stripped before detection.

    Returns:
        Square SparsityPattern over the combined, selected input space.
    """
    argnums = _ensure_index(argnums)

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


def _ensure_scalar(f: Callable, args: tuple[Any, ...]) -> Callable:
    """Ensure ``f`` returns a scalar, auto-squeezing if possible.

    If ``f`` already returns shape ``()``, it is returned unchanged.
    If squeezing the output yields a scalar (e.g. shape ``(1,)``),
    a wrapped version is returned.
    Otherwise, raises ``ValueError``.
    """
    out = jax.eval_shape(f, *args)
    if out.shape == ():
        return f
    squeezed_shape = jax.eval_shape(lambda *xs: jnp.squeeze(f(*xs)), *args).shape
    if squeezed_shape != ():
        raise ValueError(
            f"Expected scalar-valued function, but f has output shape {out.shape}."
        )
    return lambda *xs: jnp.squeeze(f(*xs))
