"""Generic PyTree plumbing: flatten, unflatten, size, dtype, numpy conversion.

The pure ``pytree <-> flat array`` layer shared across the package.
These functions carry no domain knowledge:
they know nothing about sparsity patterns, argnums, seeds, or output formats,
only how to ravel a PyTree of arrays into one flat vector and back.

The differentiation engine, the verifier, and the decompression side
each build their domain-specific wrangling on top of these leaves,
so the plumbing lives here once instead of being re-derived in each.

``_OpaqueLeaf`` rounds out the layer:
it hides a value that is itself a registered pytree from tree operations,
so those operations treat it as a single leaf.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import dtypes


def _output_size(pytree: Any) -> int:
    """Compute the total number of elements in a PyTree of arrays.

    Used to determine the number of output dimensions for Jacobian computation,
    mirroring JAX's approach of flattening PyTree outputs.
    """
    leaves = jax.tree_util.tree_leaves(pytree)
    return sum(np.size(leaf) for leaf in leaves)


def _flatten_pytree(pytree: Any) -> jax.Array:
    """Flatten a PyTree of arrays into a single 1D array.

    An empty PyTree flattens to a length-zero vector,
    rather than tripping ``jnp.concatenate`` on an empty list.
    """
    leaves = jax.tree_util.tree_leaves(pytree)
    if not leaves:
        return jnp.zeros((0,))
    return jnp.concatenate([jnp.asarray(leaf).ravel() for leaf in leaves])


def _unflatten_to_pytree(flat: jax.Array, struct: Any) -> Any:
    """Unflatten a 1D array into a PyTree matching the given structure.

    Mirrors JAX's _unravel_array_into_pytree for cotangent construction.
    """
    leaves, treedef = jax.tree_util.tree_flatten(struct)
    sizes = [np.size(leaf) for leaf in leaves]
    splits = np.cumsum(sizes[:-1])
    parts = jnp.split(flat, splits)
    reshaped = [
        part.reshape(leaf.shape) for part, leaf in zip(parts, leaves, strict=True)
    ]
    return jax.tree_util.tree_unflatten(treedef, reshaped)


def _pytree_dtype(pytree: Any) -> jnp.dtype:
    """Get the promoted result dtype for a PyTree of arrays."""
    leaves = jax.tree_util.tree_leaves(pytree)
    return dtypes.result_type(*leaves)


def _to_numpy_pytree(pytree: Any) -> Any:
    """Convert each JAX array leaf in a PyTree to ``numpy.ndarray``."""
    return jax.tree_util.tree_map(np.asarray, pytree)


class _OpaqueLeaf:
    """Wrap a value so pytree operations treat it as a single opaque leaf.

    Some values are themselves registered pytrees
    (``jax.experimental.sparse.BCOO`` is one),
    so ``tree_map`` and ``tree_transpose`` descend into their internal structure
    instead of stopping at them.
    Wrapping such a value in this plain, unregistered class hides that structure:
    tree operations see one leaf, and the value is recovered through ``.value``.
    """

    __slots__ = ("value",)

    def __init__(self, value: Any) -> None:
        self.value = value
