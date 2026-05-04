"""Input-side API helpers: argnums normalization, dtype validation, kwargs binding.

Handles everything that runs at the public API boundary before AD kicks in:

- ``_ensure_index`` / ``_ensure_inbounds`` normalize ``argnums`` exactly once
  at definition time, mirroring ``jax._src.api_util``.
  The int-vs-tuple distinction is load-bearing: it determines whether the
  returned Jacobian is a single pytree or a tuple of pytrees.
- ``dyn_args_from_argnums`` extracts the differentiated args, mirroring
  ``jax._src.api_util.argnums_partial``'s ``dyn_args`` extraction.
- ``avals_from_args`` extracts ``ShapeDtypeStruct`` pytrees from sample inputs.
- ``_validate_input_dtypes`` / ``_validate_output_dtypes`` gate AD on dtype
  compatibility, honoring the ``holomorphic`` and ``allow_int`` kwargs.
  The checks mirror ``jax._src.api._check_{input,output}_dtype_{jacrev,jacfwd}``
  but are reimplemented here to avoid coupling to jax's private API.
- ``output_size`` computes the total number of elements in a PyTree output.
"""

from __future__ import annotations

import operator
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import ShapeDtypeStruct, dtypes
from jax.tree_util import tree_map

# Argnums normalization


def _ensure_index(x: Any) -> int | tuple[int, ...]:
    """Ensure x is either an index or a tuple of indices.

    Mirrors jax._src.api_util._ensure_index.
    Preserves int-vs-tuple distinction (load-bearing for return shape).
    """
    try:
        return operator.index(x)
    except TypeError:
        return tuple(map(operator.index, x))


def _ensure_inbounds(num_args: int, argnums: tuple[int, ...]) -> tuple[int, ...]:
    """Validate bounds and resolve negative indices.

    Mirrors jax._src.api_util._ensure_inbounds.
    """
    result = []
    for i in argnums:
        if not -num_args <= i < num_args:
            raise ValueError(
                "Positional argument indices must have value >= -len(args) "
                f"and < len(args), but got {i} for len(args) == {num_args}."
            )
        result.append(i % num_args)
    return tuple(result)


def dyn_args_from_argnums(
    args: tuple[Any, ...], argnums: int | tuple[int, ...]
) -> tuple[Any, ...]:
    """Extract dynamic args at positions specified by argnums.

    Mirrors jax._src.api_util.argnums_partial's dyn_args extraction.
    """
    argnums_tuple = (argnums,) if isinstance(argnums, int) else argnums
    argnums_tuple = _ensure_inbounds(len(args), argnums_tuple)
    return tuple(args[i] for i in argnums_tuple)


# Aval extraction from sample inputs


def _to_aval(x: Any) -> ShapeDtypeStruct:
    """Convert a leaf to ShapeDtypeStruct, handling Python scalars."""
    arr = jnp.asarray(x)
    return ShapeDtypeStruct(arr.shape, arr.dtype)


def avals_from_args(args: tuple[Any, ...]) -> tuple[Any, ...]:
    """Extract ShapeDtypeStruct pytrees from sample inputs."""
    if len(args) == 0:
        raise TypeError("Expected at least one sample input.")
    return tuple(tree_map(_to_aval, arg) for arg in args)


# Dtype validation
#
# The rules below mirror ``jax._src.api._check_{input,output}_dtype_{jacrev,jacfwd}``
# but are reimplemented so asdex stays decoupled from jax's private API
# and the error messages speak in asdex's voice.


def _check_input_dtype_rev(holomorphic: bool, allow_int: bool, x: Any) -> None:
    """Validate a single reverse-mode input leaf."""
    aval = jax.typeof(x)
    if holomorphic and not dtypes.issubdtype(aval.dtype, np.complexfloating):
        raise TypeError(
            "`holomorphic=True` requires inputs with a complex dtype, "
            f"got {aval.dtype.name}."
        )
    if (
        dtypes.issubdtype(aval.dtype, dtypes.extended)
        or dtypes.issubdtype(aval.dtype, np.integer)
        or dtypes.issubdtype(aval.dtype, np.bool_)
    ):
        if not allow_int:
            raise TypeError(
                "Reverse-mode sparse differentiation requires real- or "
                "complex-valued inputs (a sub-dtype of `np.inexact`), "
                f"got {aval.dtype.name}. "
                "Pass `allow_int=True` to differentiate through "
                "boolean- or integer-valued inputs."
            )
    elif not dtypes.issubdtype(aval.dtype, np.inexact):
        raise TypeError(
            "Reverse-mode sparse differentiation requires numerical-valued "
            "inputs (a sub-dtype of `np.bool_` or `np.number`), "
            f"got {aval.dtype.name}."
        )


def _check_output_dtype_rev(holomorphic: bool, y: Any) -> None:
    """Validate a single reverse-mode output leaf."""
    aval = jax.typeof(y)
    if dtypes.issubdtype(aval.dtype, dtypes.extended):
        raise TypeError(
            f"Unsupported output element type for reverse-mode sparse "
            f"differentiation: {aval.dtype.name}."
        )
    if holomorphic:
        if not dtypes.issubdtype(aval.dtype, np.complexfloating):
            raise TypeError(
                "`holomorphic=True` requires outputs with a complex dtype, "
                f"got {aval.dtype.name}."
            )
    elif dtypes.issubdtype(aval.dtype, np.complexfloating):
        raise TypeError(
            "Reverse-mode sparse differentiation requires real-valued outputs "
            f"(a sub-dtype of `np.floating`), got {aval.dtype.name}. "
            "Pass `holomorphic=True` for holomorphic differentiation."
        )
    elif not dtypes.issubdtype(aval.dtype, np.floating):
        raise TypeError(
            "Reverse-mode sparse differentiation requires real-valued outputs "
            f"(a sub-dtype of `np.floating`), got {aval.dtype.name}."
        )


def _check_input_dtype_fwd(holomorphic: bool, x: Any) -> None:
    """Validate a single forward-mode input leaf."""
    aval = jax.typeof(x)
    if dtypes.issubdtype(aval.dtype, dtypes.extended):
        raise TypeError(
            f"Unsupported input element type for forward-mode sparse "
            f"differentiation: {aval.dtype.name}."
        )
    if holomorphic:
        if not dtypes.issubdtype(aval.dtype, np.complexfloating):
            raise TypeError(
                "`holomorphic=True` requires inputs with a complex dtype, "
                f"got {aval.dtype.name}."
            )
    elif not dtypes.issubdtype(aval.dtype, np.floating):
        raise TypeError(
            "Forward-mode sparse differentiation requires real-valued inputs "
            f"(a sub-dtype of `np.floating`), got {aval.dtype.name}. "
            "Pass `holomorphic=True` for holomorphic differentiation."
        )


def _check_output_dtype_fwd(holomorphic: bool, y: Any) -> None:
    """Validate a single forward-mode output leaf."""
    aval = jax.typeof(y)
    if holomorphic and not dtypes.issubdtype(aval.dtype, np.complexfloating):
        raise TypeError(
            "`holomorphic=True` requires outputs with a complex dtype, "
            f"got {aval.dtype.name}."
        )


def validate_input_dtypes(
    selected: tuple[Any, ...], mode: str, holomorphic: bool, allow_int: bool
) -> None:
    """Run input-dtype validation over every leaf of the selected args."""
    if mode == "fwd":
        check = lambda a: _check_input_dtype_fwd(holomorphic, a)  # noqa: E731
    else:
        # "rev" and all Hessian modes use the reverse-mode / grad-style check.
        check = lambda a: _check_input_dtype_rev(holomorphic, allow_int, a)  # noqa: E731
    for leaf in jax.tree_util.tree_leaves(selected):
        check(leaf)


def validate_output_dtypes(y: Any, mode: str, holomorphic: bool) -> None:
    """Run output-dtype validation over every leaf of ``y`` for the given ``mode``."""
    if mode == "fwd":
        check = lambda a: _check_output_dtype_fwd(holomorphic, a)  # noqa: E731
    else:
        check = lambda a: _check_output_dtype_rev(holomorphic, a)  # noqa: E731
    for leaf in jax.tree_util.tree_leaves(y):
        check(leaf)


# Kwargs binding


def bind_kwargs(f: Callable[..., Any], kwargs: dict[str, Any]) -> Callable[..., Any]:
    """Close over runtime ``**kwargs`` so downstream AD only sees positional args.

    Matches ``jax/_src/api.py:731`` (``f = lu.wrap_init(fun, kwargs, ...)``).
    """
    if not kwargs:
        return f
    return lambda *xs: f(*xs, **kwargs)


# Output PyTree utilities


def output_size(pytree: Any) -> int:
    """Compute the total number of elements in a PyTree of arrays.

    Used to determine the number of output dimensions for Jacobian computation,
    mirroring JAX's approach of flattening PyTree outputs.
    """
    leaves = jax.tree_util.tree_leaves(pytree)
    return sum(np.size(leaf) for leaf in leaves)


def flatten_pytree(pytree: Any) -> jax.Array:
    """Flatten a PyTree of arrays into a single 1D array."""
    leaves = jax.tree_util.tree_leaves(pytree)
    return jnp.concatenate([jnp.asarray(leaf).ravel() for leaf in leaves])


def unflatten_to_pytree(flat: jax.Array, struct: Any) -> Any:
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
