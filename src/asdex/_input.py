"""Input-side API helpers: aval / argnums normalization and dtype validation.

Handles everything that runs at the public API boundary before AD kicks in:

- ``normalize_avals`` / ``normalize_argnums`` coerce user-facing input shapes
  (``ShapeDtypeStruct``, shape tuples, bare ints) and ``argnums`` into a
  uniform internal form so downstream code can skip ``is_leaf`` predicates
  and negative-index arithmetic.
- ``_validate_input_dtypes`` / ``_validate_output_dtypes`` gate AD on dtype
  compatibility, honoring the ``holomorphic`` and ``allow_int`` kwargs.
  The checks mirror ``jax._src.api._check_{input,output}_dtype_{jacrev,jacfwd}``
  but are reimplemented here to avoid coupling to jax's private API.

Each positional ``in_aval`` passed by a user is a pytree whose leaves are any of:

- ``jax.ShapeDtypeStruct`` (the canonical jax leaf, see ``jax.eval_shape``);
- a shape tuple ``(3, 4)`` (asdex sugar);
- a bare ``int`` (asdex sugar, treated as a 1D length).

``normalize_avals`` coerces all leaves to ``jax.ShapeDtypeStruct`` exactly once
at the API boundary so every internal site can work with a uniform leaf type
and no custom ``is_leaf`` predicates.

``normalize_argnums`` mirrors ``jax._src.api_util._ensure_index`` followed by
``_ensure_inbounds`` (see ``jax/_src/api_util.py``): it accepts
``int | Sequence[int]``, resolves negatives via ``i % num_args``, raises on
out-of-bounds indices with jax-style wording, and preserves the int-vs-sequence
form (``int`` stays ``int``; a sequence becomes ``tuple[int, ...]``).
That int-vs-tuple distinction is load-bearing downstream — it selects whether
``example_input`` is ``dyn_avals[0]`` or ``dyn_avals``, mirroring
``jax/_src/api.py:746``.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import ShapeDtypeStruct, dtypes
from jax.tree_util import tree_map

# Aval normalization


def _is_aval_leaf(x: Any) -> bool:
    """Return ``True`` iff ``x`` is a leaf that ``normalize_avals`` should coerce.

    A leaf is a bare ``int``, a ``tuple[int, ...]``, or a ``ShapeDtypeStruct``.
    Used as the ``is_leaf`` predicate so shape tuples like ``(3, 4)`` are not
    descended into as length-2 pytree nodes.
    """
    if isinstance(x, ShapeDtypeStruct):
        return True
    if isinstance(x, int) and not isinstance(x, bool):
        return True
    return isinstance(x, tuple) and all(
        isinstance(i, int) and not isinstance(i, bool) for i in x
    )


def _to_aval(x: Any) -> ShapeDtypeStruct:
    """Turn a single leaf into a ``ShapeDtypeStruct``.

    Shape-tuple and bare-int forms default to ``jnp.float_`` (x64-aware),
    matching ``jax``'s default dtype conventions.
    """
    if isinstance(x, ShapeDtypeStruct):
        return x
    if isinstance(x, int):
        return ShapeDtypeStruct((x,), jnp.float_)
    return ShapeDtypeStruct(tuple(x), jnp.float_)


def normalize_input_shape(input_shape: Any) -> tuple[Any, ...]:
    """Normalize the ``input_shape`` parameter into a tuple of aval pytrees.

    ``input_shape`` must be a sequence (tuple or list) with one element per
    positional argument of ``f``.
    Each element is a pytree whose leaves are ``jax.ShapeDtypeStruct``,
    a shape tuple (e.g. ``(3, 4)``), or a bare ``int``.
    """
    return normalize_avals(tuple(input_shape))


def normalize_avals(in_avals: tuple[Any, ...]) -> tuple[Any, ...]:
    """Normalize ``*in_avals`` into a tuple of pytrees of ``ShapeDtypeStruct``.

    One entry per positional argument of ``f``.
    After this call, every leaf is a ``ShapeDtypeStruct`` and every internal
    site can rely on the default ``jax.tree_util`` leaf recognition.
    """
    if len(in_avals) == 0:
        raise TypeError(
            "Expected at least one positional `in_aval` describing the input "
            "structure of `f`, got none."
        )
    return tuple(tree_map(_to_aval, a, is_leaf=_is_aval_leaf) for a in in_avals)


def normalize_argnums(
    argnums: int | Sequence[int], num_args: int
) -> int | tuple[int, ...]:
    """Normalize ``argnums`` at the boundary exactly once.

    Preserves the int-vs-sequence distinction (``int`` stays ``int``,
    any ``Sequence[int]`` becomes ``tuple[int, ...]``).
    Resolves negatives via ``i % num_args`` and raises ``ValueError`` on
    out-of-bounds indices, matching ``jax/_src/api_util.py:_ensure_inbounds``.
    """
    if isinstance(argnums, int) and not isinstance(argnums, bool):
        return _resolve_index(argnums, num_args)
    try:
        seq = tuple(int(i) for i in argnums)  # ty: ignore[not-iterable]
    except TypeError as exc:
        raise TypeError(
            f"argnums must be an int or a sequence of ints, got {argnums!r}."
        ) from exc
    return tuple(_resolve_index(i, num_args) for i in seq)


def _resolve_index(i: int, num_args: int) -> int:
    """Resolve a single index against ``num_args``, mirroring jax's wording."""
    if not -num_args <= i < num_args:
        raise ValueError(
            "Positional argument indices, e.g. for `static_argnums`, must have "
            "value greater than or equal to -len(args) and less than len(args), "
            f"but got value {i} for len(args) == {num_args}."
        )
    return i % num_args


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
