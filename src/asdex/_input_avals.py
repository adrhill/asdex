"""Input-aval and ``argnums`` normalization at the public API boundary.

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

from collections.abc import Sequence
from typing import Any

import jax.numpy as jnp
from jax import ShapeDtypeStruct
from jax.tree_util import tree_map


def _is_aval_leaf(x: Any) -> bool:
    """Return True iff ``x`` is a leaf that ``normalize_avals`` should coerce.

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
