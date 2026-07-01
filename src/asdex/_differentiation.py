"""Batched-AD engine for compressing Jacobians and Hessians.

This module holds the raw automatic-differentiation numerics:
one batched VJP/JVP/HVP per color, driven by the ``ColoredPattern`` it is handed.
It reads the seed matrix (``coloring._device_seeds``) and the
differentiated-input structure (``input_avals``/``argnums`` and the
``leaf_shapes``/``leaf_sizes`` derived from them) off the pattern,
runs the AD, and returns the compressed derivative ``B`` of shape
``(num_colors, dim)`` plus the forward value and aux.

It reads only the input structure and the seeds,
never the nonzeros (``rows``/``cols``/``nnz``) or ``OutputFormat``,
so the dependency on the pattern stays one-way and the engine is
agnostic to how ``B`` is later decompressed.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any, assert_never

import jax
import jax.numpy as jnp
from jax import dtypes

from asdex._arguments import _uniform_selected_dtype
from asdex._modes import _assert_hessian_mode, _assert_jacobian_mode
from asdex._pattern import ColoredPattern, SparsityPattern
from asdex._pytree import flatten_pytree, pytree_dtype, unflatten_to_pytree


def _chunked_vmap(
    fn: Callable[..., Any],
    seeds: jax.Array,
    chunk_size: int | None,
) -> Any:
    """Vmap over seeds with bounded parallelism via sequential chunk processing.

    When ``chunk_size`` is ``None`` or exceeds the number of seeds, falls back to
    regular ``jax.vmap``. Otherwise, processes ``chunk_size`` seeds in parallel
    per chunk, with chunks processed sequentially via ``jax.lax.map``.

    Returns whatever ``fn`` returns, batched along a leading seed axis.
    ``fn`` may return a pytree (e.g. a ``(row, primal)`` pair),
    in which case every leaf is batched.

    Args:
        fn: Function to vmap over, taking a single seed vector.
        seeds: 2D array of shape ``(n_seeds, seed_dim)`` to process.
        chunk_size: Maximum seeds per parallel batch.
    """
    n = seeds.shape[0]
    if chunk_size is None or chunk_size >= n:
        return jax.vmap(fn)(seeds)
    return jax.lax.map(fn, seeds, batch_size=chunk_size)


# Jacobian over the selected input space


def _transform_with_aux(
    transform: Callable[..., Any],
    f: Callable[..., Any],
    args: tuple[Any, ...],
    *,
    has_aux: bool,
) -> tuple[Any, Any, Any]:
    """Run ``jax.vjp`` / ``jax.linearize``, always returning ``(primal, lin_fn, aux)``.

    ``aux`` is ``None`` when ``has_aux=False``,
    mirroring ``_grad_with_value_and_aux`` on the Hessian side
    so the Jacobian engine never branches on ``has_aux``.
    Requesting aux from the transform itself lets it ride the forward pass
    instead of costing an extra ``f`` call.
    """
    if has_aux:
        return transform(f, *args, has_aux=True)
    primal, lin_fn = transform(f, *args)
    return primal, lin_fn, None


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
            return _jacobian_compressed_vjp(
                f, args, coloring, out_struct, has_aux=has_aux, chunk_size=chunk_size
            )
        case "fwd":
            return _jacobian_compressed_jvp(
                f, args, coloring, has_aux=has_aux, chunk_size=chunk_size
            )
        case _ as unreachable:
            assert_never(unreachable)  # ty: ignore[type-assertion-failure]


def _jacobian_compressed_vjp(
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
    y, vjp_fn, aux = _transform_with_aux(jax.vjp, f, args, has_aux=has_aux)
    dtype = pytree_dtype(y)
    seeds = coloring._device_seeds(dtype)

    def single_vjp(seed: jax.Array) -> jax.Array:
        cotangent = unflatten_to_pytree(seed, out_struct)
        grads = vjp_fn(cotangent)
        return _flatten_selected_cotangents(grads, sparsity)

    J_compressed = _chunked_vmap(single_vjp, seeds, chunk_size)
    return J_compressed, y, aux


def _jacobian_compressed_jvp(
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
    y, jvp_fn, aux = _transform_with_aux(jax.linearize, f, args, has_aux=has_aux)
    seeds = coloring._device_seeds(dtype)

    def single_jvp(seed: jax.Array) -> jax.Array:
        tangents = _build_tangents_from_seed(seed, args, sparsity)
        return flatten_pytree(jvp_fn(*tangents))

    J_compressed = _chunked_vmap(single_jvp, seeds, chunk_size)
    return J_compressed, y, aux


# Hessian over the selected input space


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


def _hessian_compressed(
    f: Callable[..., Any],
    args: tuple[Any, ...],
    coloring: ColoredPattern,
    chunk_size: int | None,
    *,
    f_aux: Callable[..., Any] | None = None,
) -> tuple[jax.Array, jax.Array, Any]:
    """One HVP per color for a scalar-valued ``f``, by mode.

    Mirrors ``_jacobian_compressed``: one batched engine per mode.
    The primal value rides the AD forward pass for free in every mode and is
    always returned, so the engine never needs a dedicated value call
    (``rev_over_fwd`` lifts it out of the vmapped HVPs as the ``jax.grad`` aux).
    ``f_aux`` is the aux-preserving variant of ``f`` (returns ``(out, aux)``),
    used only to recover ``aux``.
    The returned aux is ``None`` when it is not given.
    Returns ``(compressed, value, aux)``.
    """
    _assert_hessian_mode(coloring.mode)
    match coloring.mode:
        case "fwd_over_rev":
            return _hessian_compressed_fwd_over_rev(
                f, args, coloring, chunk_size, f_aux
            )
        case "rev_over_fwd":
            return _hessian_compressed_rev_over_fwd(
                f, args, coloring, chunk_size, f_aux
            )
        case "rev_over_rev":
            return _hessian_compressed_rev_over_rev(
                f, args, coloring, chunk_size, f_aux
            )
        case _ as unreachable:
            assert_never(unreachable)  # ty: ignore[type-assertion-failure]


def _hessian_compressed_fwd_over_rev(
    f: Callable[..., Any],
    args: tuple[Any, ...],
    coloring: ColoredPattern,
    chunk_size: int | None,
    f_aux: Callable[..., Any] | None,
) -> tuple[jax.Array, jax.Array, Any]:
    """Forward-over-reverse HVPs; value and aux ride the linearize forward pass."""
    sparsity = coloring.sparsity
    dtype = _uniform_selected_dtype(args, sparsity)
    seeds = coloring._device_seeds(dtype)
    grad_fn = _grad_with_value_and_aux(f, f_aux, sparsity.argnums)
    _, hvp_fn, (value, aux) = jax.linearize(grad_fn, *args, has_aux=True)

    def single_hvp(v: jax.Array) -> jax.Array:
        tangents = _build_tangents_from_seed(v, args, sparsity)
        return _flatten_grad_output(hvp_fn(*tangents))

    H_compressed = _chunked_vmap(single_hvp, seeds, chunk_size)
    return H_compressed, value, aux


def _hessian_compressed_rev_over_fwd(
    f: Callable[..., Any],
    args: tuple[Any, ...],
    coloring: ColoredPattern,
    chunk_size: int | None,
    f_aux: Callable[..., Any] | None,
) -> tuple[jax.Array, jax.Array, Any]:
    """Reverse-over-forward HVPs that lift the value from the HVP forward pass.

    Each inner ``jax.jvp`` already evaluates ``f`` at ``args`` to build its
    tangent, so returning that primal as the ``jax.grad`` aux lifts the value
    out of the vmapped HVPs for free, matching the other two Hessian modes.
    The primal does not depend on the seed, so every batched copy is identical
    and the first one is the value.
    ``aux`` cannot ride along, since the HVP differentiates the aux-stripped
    ``f``, so ``has_aux`` spends one dedicated ``f_aux`` call to recover it.
    """
    sparsity = coloring.sparsity
    dtype = _uniform_selected_dtype(args, sparsity)
    seeds = coloring._device_seeds(dtype)
    grad_argnums = sparsity.argnums

    def single_hvp(v: jax.Array) -> tuple[jax.Array, jax.Array]:
        tangents = _build_tangents_from_seed(v, args, sparsity)

        def inner(*primals: Any) -> tuple[jax.Array, jax.Array]:
            primal, out_tangent = jax.jvp(f, primals, tangents)
            return out_tangent, primal

        grads, primal = jax.grad(inner, argnums=grad_argnums, has_aux=True)(*args)
        return _flatten_grad_output(grads), primal

    H_compressed, primals = _chunked_vmap(single_hvp, seeds, chunk_size)
    aux = f_aux(*args)[1] if f_aux is not None else None
    return H_compressed, primals[0], aux


def _hessian_compressed_rev_over_rev(
    f: Callable[..., Any],
    args: tuple[Any, ...],
    coloring: ColoredPattern,
    chunk_size: int | None,
    f_aux: Callable[..., Any] | None,
) -> tuple[jax.Array, jax.Array, Any]:
    """Reverse-over-reverse HVPs; value and aux ride the outer vjp forward pass."""
    sparsity = coloring.sparsity
    dtype = _uniform_selected_dtype(args, sparsity)
    seeds = coloring._device_seeds(dtype)
    grad_fn = _grad_with_value_and_aux(f, f_aux, sparsity.argnums)
    _, hvp_fn, (value, aux) = jax.vjp(grad_fn, *args, has_aux=True)

    def single_hvp(v: jax.Array) -> jax.Array:
        cotangent_out = _build_grad_output_from_seed(v, sparsity)
        cotangents = hvp_fn(cotangent_out)
        return _flatten_selected_cotangents(cotangents, sparsity)

    H_compressed = _chunked_vmap(single_hvp, seeds, chunk_size)
    return H_compressed, value, aux


# Seed / tangent / cotangent plumbing over the selected input space


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
    return flatten_pytree(out)


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
