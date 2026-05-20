"""Input-side API helpers: argnums normalization, dtype validation, kwargs binding.

Handles everything that runs at the public API boundary before AD kicks in:

- ``_ensure_index`` / ``_ensure_inbounds`` normalize ``argnums`` exactly once
  at definition time, mirroring ``jax._src.api_util``.
  The int-vs-tuple distinction is load-bearing: it determines whether the
  returned Jacobian is a single pytree or a tuple of pytrees.
- ``avals_from_args`` extracts ``ShapeDtypeStruct`` pytrees from sample inputs.
- ``_validate_input_dtypes`` / ``_validate_output_dtypes`` gate AD on dtype
  compatibility, honoring the ``holomorphic`` and ``allow_int`` kwargs.
  The checks mirror ``jax._src.api._check_{input,output}_dtype_{jacrev,jacfwd}``
  but are reimplemented here to avoid coupling to jax's private API.
- ``output_size`` computes the total number of elements in a PyTree output.
"""

from __future__ import annotations

import inspect
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
    Preserves int-vs-tuple distinction: argnums=0 returns a single array,
    while argnums=(0,) returns a length-1 tuple.
    """
    try:
        return operator.index(x)
    except TypeError:
        return tuple(map(operator.index, x))


def _ensure_inbounds(num_args: int, argnums: tuple[int, ...]) -> tuple[int, ...]:
    """Validate bounds and resolve negative indices to positive ones.

    For example, argnums=(-1,) with 3 args becomes (2,).
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


# Aval extraction from sample inputs


def _to_aval(x: Any) -> ShapeDtypeStruct:
    """Convert a pytree leaf (array or scalar) to ShapeDtypeStruct.

    ShapeDtypeStruct holds shape and dtype metadata without actual array data.
    """
    arr = jnp.asarray(x)
    return ShapeDtypeStruct(arr.shape, arr.dtype)


def avals_from_args(args: tuple[Any, ...]) -> tuple[Any, ...]:
    """Extract ShapeDtypeStruct pytrees from sample inputs.

    Returns pytrees with the same structure, but leaves replaced by shape+dtype metadata.
    """
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
        if allow_int:
            # Match JAX: jacfwd doesn't support allow_int, only jacrev does.
            raise TypeError(
                "`allow_int=True` is not supported in forward mode. "
                "Use `mode='rev'` for differentiating with respect to integer inputs."
            )
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


# Kwargs handling


def merge_args_kwargs(
    f: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    expected_nargs: int,
) -> tuple[tuple[Any, ...], Callable[..., Any]]:
    """Merge kwargs that correspond to expected positional args, bind the rest.

    Uses the function's signature to map keyword arguments to their
    corresponding positional indices up to ``expected_nargs``.
    Remaining kwargs are bound to the function.

    This matches JAX's behavior where ``jax.jacrev``, ``jax.jacfwd``, etc.
    accept keyword arguments at call time that get mapped to positional
    parameters.

    Mirrors ``jax/_src/api.py`` which uses ``functools.partial(f, **kwargs)``
    to bind kwargs (see ``jax/_src/linear_util.py:wrap_init``).

    Returns:
        A tuple of ``(merged_args, f_bound)`` where ``merged_args`` has exactly
        ``expected_nargs`` elements, and ``f_bound`` has any extra kwargs bound.
    """
    if not kwargs:
        return args, f

    try:
        sig = inspect.signature(f)
        bound = sig.bind(*args, **kwargs)
    except (ValueError, TypeError) as e:
        raise TypeError(
            f"Cannot bind arguments: {e}. "
            f"Got {len(args)} positional argument(s) and "
            f"keyword argument(s) {set(kwargs.keys())}."
        ) from None

    # Extract positional args and remaining kwargs, handling VAR_POSITIONAL/VAR_KEYWORD
    positional_args: list[Any] = []
    extra_kwargs: dict[str, Any] = {}
    param_count = 0

    for name, value in bound.arguments.items():
        param = sig.parameters[name]
        match param.kind:
            case inspect.Parameter.VAR_POSITIONAL:
                # *args: expand the tuple into positional args
                positional_args.extend(value)
                param_count += len(value)
            case inspect.Parameter.VAR_KEYWORD:
                # **kwargs: merge into extra_kwargs (never positional)
                extra_kwargs.update(value)
            case _:
                # Regular parameter
                if param_count < expected_nargs:
                    positional_args.append(value)
                    param_count += 1
                else:
                    extra_kwargs[name] = value

    merged_args = tuple(positional_args[:expected_nargs])

    if extra_kwargs:

        def f_bound(*xs: Any) -> Any:
            return f(*xs, **extra_kwargs)
    else:
        f_bound = f

    return merged_args, f_bound


def _is_jax_traceable(x: Any) -> bool:
    """Check if a value should be traced by JAX (array-like) vs bound statically.

    Returns True only if ALL leaves are array-like.
    Returns False if any leaf is non-traceable (bool, int, str, None), since
    these cannot be traced and will cause errors if used in Python control flow.
    """
    leaves = jax.tree_util.tree_leaves(x)
    if not leaves:
        return False
    for leaf in leaves:
        if isinstance(leaf, bool | int | str | type(None)):
            return False
        if not (hasattr(leaf, "shape") and hasattr(leaf, "dtype")):
            return False
    return True


def merge_sample_inputs(
    f: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> tuple[tuple[Any, ...], Callable[..., Any]]:
    """Merge sample inputs, resolving kwargs to positions for traceable values.

    Uses ``inspect.signature(f).bind()`` to resolve kwargs to signature positions.
    JAX-traceable values (arrays, pytrees of arrays) are passed positionally for
    tracing by ``make_jaxpr``. Non-traceable values (bools, strings, ints) are
    bound to the function statically.

    This matches JAX's behavior where ``jacrev(f)(x, flag=True)`` works even if
    ``flag`` controls a Python ``if`` branch - the flag is not traced.

    Returns:
        A tuple of ``(positional_args, f_bound)`` where ``positional_args``
        contains traceable values in signature order, and ``f_bound`` has
        non-traceable values pre-bound.
    """
    if not kwargs:
        return args, f

    try:
        sig = inspect.signature(f)
        bound = sig.bind(*args, **kwargs)
    except (ValueError, TypeError) as e:
        raise TypeError(
            f"Cannot bind sample arguments: {e}. "
            f"Got {len(args)} positional and {set(kwargs.keys())} keyword."
        ) from None

    # Split into traceable (positional) vs non-traceable (bind statically)
    positional_args: list[Any] = []
    bind_kwargs: dict[str, Any] = {}

    for name, value in bound.arguments.items():
        param = sig.parameters[name]
        match param.kind:
            case inspect.Parameter.VAR_POSITIONAL:
                # *args: expand, trace each traceable element
                positional_args.extend(v for v in value if _is_jax_traceable(v))
            case inspect.Parameter.VAR_KEYWORD:
                # **kwargs: bind all (static values like scale=2.0)
                bind_kwargs.update(value)
            case inspect.Parameter.KEYWORD_ONLY:
                # Keyword-only: always bind
                bind_kwargs[name] = value
            case _:
                # POSITIONAL_ONLY or POSITIONAL_OR_KEYWORD
                if _is_jax_traceable(value):
                    positional_args.append(value)
                else:
                    # Non-traceable (bool, int, string): bind statically
                    bind_kwargs[name] = value

    if bind_kwargs:

        def f_bound(*xs: Any) -> Any:
            return f(*xs, **bind_kwargs)
    else:
        f_bound = f

    return tuple(positional_args), f_bound


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
