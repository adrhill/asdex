"""Helpers for multi-input (pytree-structured) shape specs.

A *shape* is a ``tuple[int, ...]`` (or a bare ``int``), the same thing that
``input_shape=`` accepts.
A *shape spec* is either a single shape (single-input mode) or a pytree whose
leaves are shapes (multi-input mode).
When the top-level of the spec is a tuple of shapes, the returned function is
called as ``f(*xs)``; otherwise the spec is a pytree and the function is
called as ``f(pytree)``.

This module pins down the boundary between the user-facing pytree of
arrays/shapes and the flat, concatenated input vector the detection and
decompression pipelines operate on.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import jax
import jax.numpy as jnp
from jax.tree_util import PyTreeDef

ShapeLeaf = tuple[int, ...]
"""A single array shape.

Bare ``int`` inputs are normalized to ``(int,)`` before any helper sees them.
"""

ShapeSpec = Any
"""User-facing spec: either a single shape or a pytree of shapes."""


def _normalize_shape(shape: int | tuple[int, ...]) -> ShapeLeaf:
    """Turn an ``int`` into a 1D shape tuple; leave tuples as-is."""
    if isinstance(shape, int):
        return (shape,)
    return tuple(shape)


def _is_shape_leaf(x: Any) -> bool:
    """Tell pytree traversal to stop at array shapes.

    ``tuple[int, ...]`` is a pytree node by default; this ``is_leaf`` predicate
    keeps a shape together so ``flatten_shapes`` sees one leaf per array.
    """
    if isinstance(x, tuple):
        return all(isinstance(i, int) for i in x)
    return False


def is_multi_positional(shapes_spec: ShapeSpec) -> bool:
    """Whether the spec describes multiple positional arguments to ``f``.

    The spec is treated as multi-positional iff its top level is a tuple and
    it is not itself a single shape (``(3,)`` is one array, not two).
    """
    if not isinstance(shapes_spec, tuple):
        return False
    return not _is_shape_leaf(shapes_spec)


def flatten_shapes(
    shapes_spec: ShapeSpec,
) -> tuple[list[ShapeLeaf], PyTreeDef, list[int]]:
    """Flatten a shape spec into a list of leaf shapes plus a treedef.

    Returns ``(leaf_shapes, treedef, leaf_sizes)`` where each leaf is a single
    array's shape, ``treedef`` captures the pytree structure (so results can be
    un-flattened back), and ``leaf_sizes[k]`` is ``prod(leaf_shapes[k])``.
    """
    leaves, treedef = jax.tree_util.tree_flatten(shapes_spec, is_leaf=_is_shape_leaf)
    leaf_shapes = [_normalize_shape(s) for s in leaves]
    leaf_sizes = [math.prod(s) if s else 1 for s in leaf_shapes]
    return leaf_shapes, treedef, leaf_sizes


def total_size(shapes_spec: ShapeSpec) -> int:
    """Sum of ``prod(shape)`` over all leaves."""
    _, _, sizes = flatten_shapes(shapes_spec)
    return sum(sizes)


def compute_argnums_mask(
    argnums: int | Sequence[int] | None,
    num_leaves: int,
    *,
    multi_positional: bool,
) -> list[bool]:
    """Resolve ``argnums`` to a per-leaf selection mask.

    ``argnums=None`` selects everything.
    For multi-positional specs, ``argnums`` is an int or tuple of ints indexing
    into the top-level tuple positions, matching ``jax.grad``'s convention.
    For single-pytree specs, ``argnums`` is not supported in this release
    (non-trivial key-path resolution is deferred).
    """
    if argnums is None:
        return [True] * num_leaves

    if not multi_positional:
        msg = (
            "`argnums` is only supported when `input_shapes` is a top-level "
            "tuple of shapes (multi-positional functions). "
            "For single-pytree inputs, close over or pre-select inputs manually."
        )
        raise NotImplementedError(msg)

    indices = (argnums,) if isinstance(argnums, int) else tuple(argnums)

    for i in indices:
        if not isinstance(i, int):
            raise TypeError(
                f"argnums entries must be ints for multi-positional input_shapes, got {i!r}"
            )
        if i < 0 or i >= num_leaves:
            raise ValueError(
                f"argnums={argnums} out of bounds for {num_leaves} positional inputs"
            )

    mask = [False] * num_leaves
    for i in indices:
        mask[i] = True
    return mask


def selected_offsets(
    leaf_sizes: Sequence[int], selected_mask: Sequence[bool]
) -> list[int]:
    """Cumulative offsets of the selected leaves in the combined flat space.

    ``offsets[k]`` is where selected leaf ``k`` begins and ``offsets[-1]`` is
    the total selected size.
    """
    offsets = [0]
    for size, selected in zip(leaf_sizes, selected_mask, strict=True):
        if selected:
            offsets.append(offsets[-1] + size)
    return offsets


def split_flat(
    flat: jax.Array,
    leaf_shapes: Sequence[ShapeLeaf],
    *,
    axis: int = -1,
) -> list[jax.Array]:
    """Split a flat array along ``axis`` into one array per leaf shape.

    The resulting leaf has the size-``prod(shape)`` slice reshaped to
    ``(..., *shape)``.
    """
    if not leaf_shapes:
        return []
    sizes = [math.prod(s) if s else 1 for s in leaf_shapes]
    splits = jnp.split(flat, list(_cumsum(sizes))[:-1], axis=axis)

    leading = tuple(flat.shape[:axis] if axis >= 0 else flat.shape[: flat.ndim + axis])
    result = []
    for arr, shape in zip(splits, leaf_shapes, strict=True):
        result.append(arr.reshape(leading + tuple(shape)))
    return result


def unflatten_to_pytree(leaves: Sequence[Any], treedef: PyTreeDef) -> Any:
    """Unflatten a list of leaves back into the original pytree structure."""
    return jax.tree_util.tree_unflatten(treedef, list(leaves))


def select_subtree(
    shapes_spec: ShapeSpec,
    leaves: Sequence[Any],
    selected_mask: Sequence[bool],
) -> Any:
    """Build a pytree containing only the selected leaves.

    ``leaves`` is the full flat leaf list (one entry per leaf in the original
    spec). For multi-positional specs, the selected leaves are returned as a
    tuple (or a single value when exactly one is selected, matching
    ``jax.grad``'s ``argnums=int`` convention: this wrapper is the caller's
    responsibility; ``select_subtree`` always returns a tuple for
    multi-positional and a pytree for single-pytree specs).
    """
    if is_multi_positional(shapes_spec):
        return tuple(
            leaf
            for leaf, selected in zip(leaves, selected_mask, strict=True)
            if selected
        )
    # Single-pytree: structure must be preserved, so all leaves are selected.
    _, treedef, _ = flatten_shapes(shapes_spec)
    return unflatten_to_pytree(list(leaves), treedef)


def _cumsum(xs: Sequence[int]) -> list[int]:
    out = []
    total = 0
    for x in xs:
        total += x
        out.append(total)
    return out
