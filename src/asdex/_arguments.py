"""Argument handling and validation at the public API boundary.

Everything that reconciles the caller's positional args, keyword args,
and ``argnums`` against the traced function,
then checks the differentiated arguments are well-formed,
before AD kicks in:

- ``_ensure_index`` / ``_ensure_inbounds`` normalize ``argnums`` exactly once
  at definition time, mirroring ``jax._src.api_util``.
  The int-vs-tuple distinction is load-bearing: it determines whether the
  returned Jacobian is a single pytree or a tuple of pytrees.
- ``avals_from_args`` extracts ``ShapeDtypeStruct`` pytrees from sample inputs.
- ``merge_sample_inputs`` / ``merge_args_kwargs`` resolve keyword arguments to
  positions and bind non-traceable arguments statically.
- ``validate_input_dtypes`` / ``validate_output_dtypes`` gate AD on dtype
  compatibility, honoring the ``holomorphic`` and ``allow_int`` kwargs.
  The checks mirror ``jax._src.api._check_{input,output}_dtype_{jacrev,jacfwd}``
  but are reimplemented here to avoid coupling to jax's private API.
- ``_validate_args`` / ``_assert_chunk_size`` check call-time arguments against
  the pattern's declared structure and shapes.
- ``_selected_args`` and friends slice out the differentiated input subspace.

The generic ``pytree <-> flat array`` plumbing these build on
lives in ``asdex._pytree``.
"""

from __future__ import annotations

import functools
import inspect
import operator
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from jax import ShapeDtypeStruct, dtypes
from jax.tree_util import tree_map

from asdex._pattern import SparsityPattern

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


def _check_input_dtype_hessian(holomorphic: bool, x: Any) -> None:
    """Validate a single Hessian-mode input leaf."""
    aval = jax.typeof(x)
    if (
        dtypes.issubdtype(aval.dtype, dtypes.extended)
        or dtypes.issubdtype(aval.dtype, np.integer)
        or dtypes.issubdtype(aval.dtype, np.bool_)
    ):
        raise TypeError(
            "Sparse Hessians require real- or complex-valued inputs "
            f"(a sub-dtype of `np.inexact`), got {aval.dtype.name}. "
            "Differentiating twice with respect to boolean- or integer-valued "
            "inputs is not supported, matching `jax.hessian`."
        )
    _check_input_dtype_rev(holomorphic, False, x)


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
    elif mode == "rev":
        check = lambda a: _check_input_dtype_rev(holomorphic, allow_int, a)  # noqa: E731
    else:
        # Hessian modes: the gradient inside each HVP requires float inputs,
        # so integer differentiation can never work here.
        if allow_int:
            raise TypeError(
                "`allow_int=True` is not supported for Hessian computation: "
                "the gradient inside each Hessian-vector product requires "
                "float inputs (`jax.hessian` has no `allow_int` either)."
            )
        check = lambda a: _check_input_dtype_hessian(holomorphic, a)  # noqa: E731
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


# Call-argument structure validation


def _assert_chunk_size(chunk_size: int | None) -> None:
    """Validate chunk_size parameter."""
    if chunk_size is not None and chunk_size <= 0:
        raise ValueError(f"chunk_size must be positive, got {chunk_size}")


def _validate_args(args: tuple[Any, ...], sparsity: SparsityPattern) -> None:
    """Check ``args`` match the pattern's declared positional-arg structure."""
    if len(args) != len(sparsity.input_avals):
        raise ValueError(
            f"Expected {len(sparsity.input_avals)} positional argument(s), "
            f"got {len(args)}."
        )
    for i, (arg, aval) in enumerate(zip(args, sparsity.input_avals, strict=True)):
        user_tree = jax.tree_util.tree_structure(arg)
        aval_tree = jax.tree_util.tree_structure(aval)
        if user_tree != aval_tree:
            raise ValueError(
                f"Argument {i} pytree structure {user_tree} does not match "
                f"the colored pattern, which expects {aval_tree}."
            )
        user_leaves = jax.tree_util.tree_leaves(arg)
        aval_leaves = jax.tree_util.tree_leaves(aval)
        for k, (leaf, expected) in enumerate(
            zip(user_leaves, aval_leaves, strict=True)
        ):
            leaf_shape = tuple(getattr(leaf, "shape", ()))
            if leaf_shape != tuple(expected.shape):
                raise ValueError(
                    f"Argument {i} leaf {k} shape {leaf_shape} does not match "
                    f"expected {tuple(expected.shape)}."
                )


# Kwargs handling


def merge_args_kwargs(
    f: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    expected_nargs: int,
) -> tuple[tuple[Any, ...], Callable[..., Any]]:
    """Merge call-time args/kwargs, filtering to only traceable values.

    Mirrors ``merge_sample_inputs`` to ensure consistent handling:
    1. Kwargs that were present at detection time are resolved to positions
    2. Kwargs only present at call time (not detection) are bound statically
    3. Non-traceable positional args are bound preserving their positions

    The heuristic: if resolving all kwargs gives more traceable args than
    expected, the extra kwargs weren't at detection time - bind them statically.

    Returns:
        A tuple of ``(traceable_args, f_bound)`` where ``traceable_args``
        has exactly ``expected_nargs`` elements (only JAX-traceable values),
        and ``f_bound`` has kwargs and non-traceables pre-bound.
    """
    if kwargs:
        # Try resolving kwargs to positions (like merge_sample_inputs)
        try:
            sig = inspect.signature(f)
            bound = sig.bind(*args, **kwargs)
        except (ValueError, TypeError) as e:
            raise TypeError(
                f"Cannot bind arguments: {e}. "
                f"Got {len(args)} positional and {set(kwargs.keys())} keyword."
            ) from None

        # Reconstruct full positional args list and collect extra kwargs
        all_args: list[Any] = []
        extra_kwargs: dict[str, Any] = {}

        for name, value in bound.arguments.items():
            param = sig.parameters[name]
            match param.kind:
                case inspect.Parameter.VAR_POSITIONAL:
                    all_args.extend(value)
                case inspect.Parameter.VAR_KEYWORD:
                    extra_kwargs.update(value)
                case inspect.Parameter.KEYWORD_ONLY:
                    extra_kwargs[name] = value
                case _:
                    all_args.append(value)

        resolved_args = tuple(all_args)
        resolved_traceable = sum(1 for a in resolved_args if _is_jax_traceable(a))

        if resolved_traceable > expected_nargs:
            # More traceable args than expected: some kwargs weren't at detection.
            # Fall back to binding ALL kwargs statically
            f = functools.partial(f, **kwargs)
            if extra_kwargs:
                f = functools.partial(f, **extra_kwargs)
            # Continue with just positional args
        else:
            # Count matches: kwargs were at detection time, use resolved version
            args = resolved_args
            if extra_kwargs:
                f = functools.partial(f, **extra_kwargs)

    if not args:
        if expected_nargs != 0:
            raise ValueError(
                f"Expected {expected_nargs} positional argument(s), got 0."
            )
        return (), f

    # Separate traceable vs non-traceable positional args
    traceable_args: list[Any] = []
    static_positions: dict[int, Any] = {}

    for i, arg in enumerate(args):
        if _is_jax_traceable(arg):
            traceable_args.append(arg)
        else:
            static_positions[i] = arg

    # Validate count matches detection time
    if len(traceable_args) != expected_nargs:
        raise ValueError(
            f"Expected {expected_nargs} positional argument(s), got {len(traceable_args)}. "
            f"(Total args: {len(args)}, non-traceable: {len(static_positions)})"
        )

    if not static_positions:
        return tuple(traceable_args), f

    # Create wrapper that injects static args at original positions
    total_nargs = len(args)
    f_original = f

    def f_bound(*xs: Any) -> Any:
        full_args: list[Any] = [None] * total_nargs
        for pos, val in static_positions.items():
            full_args[pos] = val
        xs_iter = iter(xs)
        for i in range(total_nargs):
            if full_args[i] is None:
                full_args[i] = next(xs_iter)
        return f_original(*full_args)

    return tuple(traceable_args), f_bound


def _is_jax_traceable(x: Any) -> bool:
    """Check if a value should be traced by JAX vs bound statically.

    Returns True for values that JAX can trace: arrays, pytrees of arrays,
    and numeric scalars (Python floats, numpy floats).

    Returns False for non-traceable values (bool, int, str, None) which
    cannot be traced and would cause errors if used in Python control flow.
    Python ints are excluded because they're commonly used in control flow
    (array indexing, loop bounds), not as numeric computation inputs.
    """
    leaves = jax.tree_util.tree_leaves(x)
    if not leaves:
        return False
    for leaf in leaves:
        # Non-traceables: bools, ints (control flow), strings, None
        if isinstance(leaf, bool | int | str | type(None)):
            return False
        # Accept floats (Python float, numpy float) and arrays
        if isinstance(leaf, float):
            continue
        if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
            continue
        # numpy scalar types (np.float64, etc.) have dtype but not shape
        if hasattr(leaf, "dtype"):
            continue
        return False
    return True


def merge_sample_inputs(
    f: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    argnums: int | tuple[int, ...],
) -> tuple[tuple[Any, ...], Callable[..., Any], int | tuple[int, ...]]:
    """Merge sample inputs, resolving kwargs to positions and binding non-traceables.

    Mirrors JAX's approach with two additions:
    1. Uses ``inspect.signature().bind()`` to resolve kwargs to their
       signature positions (so ``f(a=x, b=y)`` becomes ``f(x, y)``)
    2. Non-traceable positional args (bools, ints, strings) are bound
       statically, preserving their positions (like ``argnums_partial``)

    Args:
        f: The function to be traced.
        args: Positional arguments to merge.
        kwargs: Keyword arguments to resolve to positions.
        argnums: Already-validated argnums (via ``_ensure_index``).

    Returns:
        A tuple of ``(traceable_args, f_bound, remapped_argnums)`` where:
        - ``traceable_args`` contains only JAX-traceable values in signature order
        - ``f_bound`` has non-traceable values pre-bound at their positions
        - ``remapped_argnums`` maps original argnums to new positions
    """
    if not kwargs and not args:
        return (), f, argnums

    # Step 1: Use signature binding to resolve kwargs to positional order
    if kwargs:
        try:
            sig = inspect.signature(f)
            bound = sig.bind(*args, **kwargs)
        except (ValueError, TypeError) as e:
            raise TypeError(
                f"Cannot bind sample arguments: {e}. "
                f"Got {len(args)} positional and {set(kwargs.keys())} keyword."
            ) from None

        # Reconstruct full positional args list and collect extra kwargs
        all_args: list[Any] = []
        extra_kwargs: dict[str, Any] = {}

        for name, value in bound.arguments.items():
            param = sig.parameters[name]
            match param.kind:
                case inspect.Parameter.VAR_POSITIONAL:
                    # *args: expand into positional
                    all_args.extend(value)
                case inspect.Parameter.VAR_KEYWORD:
                    # **kwargs: bind statically
                    extra_kwargs.update(value)
                case inspect.Parameter.KEYWORD_ONLY:
                    # Keyword-only: bind statically
                    extra_kwargs[name] = value
                case _:
                    # POSITIONAL_ONLY or POSITIONAL_OR_KEYWORD
                    all_args.append(value)

        args = tuple(all_args)
        if extra_kwargs:
            f = functools.partial(f, **extra_kwargs)

    if not args:
        return (), f, argnums

    # Step 2: Separate traceable vs non-traceable positional args
    traceable_args: list[Any] = []
    static_positions: dict[int, Any] = {}
    old_to_new: dict[int, int] = {}

    for i, arg in enumerate(args):
        if _is_jax_traceable(arg):
            old_to_new[i] = len(traceable_args)
            traceable_args.append(arg)
        else:
            static_positions[i] = arg

    # Step 3: Remap argnums from original indices to new indices
    # First ensure bounds (this also resolves negative indices)
    num_args = len(args)
    argnums_tup = (argnums,) if isinstance(argnums, int) else argnums
    resolved_tup = _ensure_inbounds(num_args, argnums_tup)

    remapped_argnums: int | tuple[int, ...]
    if isinstance(argnums, int):
        resolved = resolved_tup[0]
        if resolved not in old_to_new:
            raise ValueError(
                f"argnums={argnums} refers to a non-traceable argument "
                f"(bool, int, str, or None). Cannot differentiate with respect "
                f"to non-traceable arguments."
            )
        remapped_argnums = old_to_new[resolved]
    else:
        remapped = []
        for orig_idx, resolved in zip(argnums, resolved_tup, strict=True):
            if resolved not in old_to_new:
                raise ValueError(
                    f"argnums index {orig_idx} refers to a non-traceable argument "
                    f"(bool, int, str, or None). Cannot differentiate with respect "
                    f"to non-traceable arguments."
                )
            remapped.append(old_to_new[resolved])
        remapped_argnums = tuple(remapped)

    if not static_positions:
        return tuple(traceable_args), f, remapped_argnums

    # Step 4: Create wrapper that injects static args at original positions
    total_nargs = len(args)
    f_original = f

    def f_bound(*xs: Any) -> Any:
        full_args: list[Any] = [None] * total_nargs
        for pos, val in static_positions.items():
            full_args[pos] = val
        xs_iter = iter(xs)
        for i in range(total_nargs):
            if full_args[i] is None:
                full_args[i] = next(xs_iter)
        return f_original(*full_args)

    return tuple(traceable_args), f_bound, remapped_argnums


# Selected-input helpers
#
# The differentiation engine and the compress/decompress evaluators all need
# the sub-tuple of arguments named by ``argnums`` and its dtype.
# They live here, in the leaf module, so both reach them without an import cycle.


def _selected_args(args: tuple[Any, ...], sparsity: SparsityPattern) -> tuple[Any, ...]:
    """Sub-tuple of ``args`` at positions named by ``sparsity.argnums``."""
    return tuple(args[i] for i in sparsity._argnums_tuple)


def _selected_dtype(args: tuple[Any, ...], sparsity: SparsityPattern) -> Any:
    """Dtype for seed arrays, taken from the first selected leaf we can find."""
    for leaf in jax.tree_util.tree_leaves(_selected_args(args, sparsity)):
        dtype = getattr(leaf, "dtype", None)
        if dtype is not None:
            return dtype
    return jnp.float_


def _uniform_selected_dtype(args: tuple[Any, ...], sparsity: SparsityPattern) -> Any:
    """Shared dtype of all selected leaves, for input-space seeds.

    Forward-mode tangents and HVP seeds are sliced from one flat seed vector,
    so every selected leaf must share its dtype;
    mixed dtypes would otherwise fail deep inside ``jvp``/``vjp``.
    Leaves without a ``dtype`` (Python scalars) are weakly typed and skipped.
    """
    found = {
        jnp.dtype(leaf.dtype)
        for leaf in jax.tree_util.tree_leaves(_selected_args(args, sparsity))
        if getattr(leaf, "dtype", None) is not None
    }
    if len(found) > 1:
        names = sorted(d.name for d in found)
        raise TypeError(
            f"Differentiated inputs have mixed dtypes {names}, "
            "which forward and Hessian modes do not support. "
            "Cast the inputs to a common dtype, "
            "or use `mode='rev'` for Jacobians."
        )
    return found.pop() if found else jnp.float_
