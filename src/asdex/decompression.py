"""Sparse Jacobian and Hessian computation using coloring and AD."""

from collections.abc import Callable, Sequence
from typing import Any, Literal, assert_never, overload

import jax
import jax.numpy as jnp
from jax.experimental.sparse import BCOO
from numpy.typing import ArrayLike

from asdex.coloring import hessian_coloring as _hessian_coloring
from asdex.coloring import jacobian_coloring as _jacobian_coloring
from asdex.detection._api import _ensure_scalar, _ensure_scalar_multi
from asdex.modes import (
    HessianMode,
    JacobianMode,
    OutputFormat,
    _assert_hessian_mode,
    _assert_jacobian_mode,
    _assert_output_format,
)
from asdex.pattern import ColoredPattern, SparsityPattern

# Public API


@overload
def jacobian(
    f: Callable[[ArrayLike], ArrayLike],
    input_shape: int | tuple[int, ...],
    *,
    mode: JacobianMode | None = None,
    symmetric: bool = False,
    output_format: Literal["bcoo"] = ...,
) -> Callable[[ArrayLike], BCOO]: ...
@overload
def jacobian(
    f: Callable[[ArrayLike], ArrayLike],
    input_shape: int | tuple[int, ...],
    *,
    mode: JacobianMode | None = None,
    symmetric: bool = False,
    output_format: Literal["dense"],
) -> Callable[[ArrayLike], jax.Array]: ...
@overload
def jacobian(
    f: Callable[..., Any],
    *,
    input_shapes: Any,
    argnums: int | Sequence[int] | None = None,
    mode: JacobianMode | None = None,
    symmetric: bool = False,
    output_format: OutputFormat = ...,
) -> Callable[..., Any]: ...
def jacobian(
    f: Callable[..., Any],
    input_shape: int | tuple[int, ...] | None = None,
    *,
    input_shapes: Any = None,
    argnums: int | Sequence[int] | None = None,
    mode: JacobianMode | None = None,
    symmetric: bool = False,
    output_format: OutputFormat = "bcoo",
) -> Callable[..., Any]:
    """Detect sparsity, color, and return a function computing sparse Jacobians.

    Combines [`jacobian_coloring`][asdex.jacobian_coloring]
    and [`jacobian_from_coloring`][asdex.jacobian_from_coloring]
    in one call.

    Args:
        f: Function taking an array and returning an array.
            Input and output may be multi-dimensional.
        input_shape: Shape of the input array (single-input mode).
            Mutually exclusive with ``input_shapes``.
        input_shapes: Pytree of shapes (multi-input mode).
            Mutually exclusive with ``input_shape``.
        argnums: Positional arguments to differentiate with respect to,
            mirroring ``jax.grad``.
            Only supported with multi-positional ``input_shapes``.
        mode: AD mode.
            ``"fwd"`` uses JVPs (forward-mode AD),
            ``"rev"`` uses VJPs (reverse-mode AD).
            ``None`` picks whichever of fwd/rev needs fewer colors.
        symmetric: Whether to use symmetric (star) coloring.
            Requires a square Jacobian.
        output_format: Type of the output matrix.
            ``"bcoo"`` returns a sparse matrix of type ``jax.experimental.sparse.BCOO`` (default),
            ``"dense"`` returns a dense matrix of type ``jax.Array``.

    Returns:
        A function that takes an input array and returns
            the Jacobian of shape ``(m, n)``
            where ``n = x.size`` and ``m = prod(output_shape)``.
            The output type depends on ``output_format``:
            a sparse ``jax.experimental.sparse.BCOO`` (default)
            or a dense ``jax.Array``.
            For multi-input functions, returns a pytree of Jacobian blocks
            matching the selected input structure.
    """
    coloring = _jacobian_coloring(
        f,
        input_shape,
        input_shapes=input_shapes,
        argnums=argnums,
        mode=mode,
        symmetric=symmetric,
    )
    return jacobian_from_coloring(f, coloring, output_format=output_format)


@overload
def value_and_jacobian(
    f: Callable[[ArrayLike], ArrayLike],
    input_shape: int | tuple[int, ...],
    *,
    mode: JacobianMode | None = None,
    symmetric: bool = False,
    output_format: Literal["bcoo"] = ...,
) -> Callable[[ArrayLike], tuple[jax.Array, BCOO]]: ...
@overload
def value_and_jacobian(
    f: Callable[[ArrayLike], ArrayLike],
    input_shape: int | tuple[int, ...],
    *,
    mode: JacobianMode | None = None,
    symmetric: bool = False,
    output_format: Literal["dense"],
) -> Callable[[ArrayLike], tuple[jax.Array, jax.Array]]: ...
@overload
def value_and_jacobian(
    f: Callable[..., Any],
    *,
    input_shapes: Any,
    argnums: int | Sequence[int] | None = None,
    mode: JacobianMode | None = None,
    symmetric: bool = False,
    output_format: OutputFormat = ...,
) -> Callable[..., tuple[Any, Any]]: ...
def value_and_jacobian(
    f: Callable[..., Any],
    input_shape: int | tuple[int, ...] | None = None,
    *,
    input_shapes: Any = None,
    argnums: int | Sequence[int] | None = None,
    mode: JacobianMode | None = None,
    symmetric: bool = False,
    output_format: OutputFormat = "bcoo",
) -> Callable[..., tuple[Any, Any]]:
    """Detect sparsity, color, and return a function computing value and sparse Jacobian.

    Like [`jacobian`][asdex.jacobian],
    but also returns the primal value ``f(x)``
    without an extra forward pass.

    Args:
        f: Function taking an array and returning an array.
            Input and output may be multi-dimensional.
        input_shape: Shape of the input array (single-input mode).
            Mutually exclusive with ``input_shapes``.
        input_shapes: Pytree of shapes (multi-input mode).
            Mutually exclusive with ``input_shape``.
        argnums: Positional arguments to differentiate with respect to,
            mirroring ``jax.grad``.
            Only supported with multi-positional ``input_shapes``.
        mode: AD mode.
            ``"fwd"`` uses JVPs (forward-mode AD),
            ``"rev"`` uses VJPs (reverse-mode AD).
            ``None`` picks whichever of fwd/rev needs fewer colors.
        symmetric: Whether to use symmetric (star) coloring.
            Requires a square Jacobian.
        output_format: Type of the output matrix.
            ``"bcoo"`` returns a sparse matrix of type ``jax.experimental.sparse.BCOO`` (default),
            ``"dense"`` returns a dense matrix of type ``jax.Array``.

    Returns:
        A function that takes an input array and returns
            ``(f(x), J)`` where ``J`` is the Jacobian
            of shape ``(m, n)``
            where ``n = x.size`` and ``m = prod(output_shape)``.
            The output type depends on ``output_format``:
            a sparse ``jax.experimental.sparse.BCOO`` (default)
            or a dense ``jax.Array``.
    """
    coloring = _jacobian_coloring(
        f,
        input_shape,
        input_shapes=input_shapes,
        argnums=argnums,
        mode=mode,
        symmetric=symmetric,
    )
    return value_and_jacobian_from_coloring(f, coloring, output_format=output_format)


@overload
def hessian(
    f: Callable[[ArrayLike], ArrayLike],
    input_shape: int | tuple[int, ...],
    *,
    mode: HessianMode | None = None,
    symmetric: bool = True,
    output_format: Literal["bcoo"] = ...,
) -> Callable[[ArrayLike], BCOO]: ...
@overload
def hessian(
    f: Callable[[ArrayLike], ArrayLike],
    input_shape: int | tuple[int, ...],
    *,
    mode: HessianMode | None = None,
    symmetric: bool = True,
    output_format: Literal["dense"],
) -> Callable[[ArrayLike], jax.Array]: ...
@overload
def hessian(
    f: Callable[..., Any],
    *,
    input_shapes: Any,
    argnums: int | Sequence[int] | None = None,
    mode: HessianMode | None = None,
    symmetric: bool = True,
    output_format: OutputFormat = ...,
) -> Callable[..., Any]: ...
def hessian(
    f: Callable[..., Any],
    input_shape: int | tuple[int, ...] | None = None,
    *,
    input_shapes: Any = None,
    argnums: int | Sequence[int] | None = None,
    mode: HessianMode | None = None,
    symmetric: bool = True,
    output_format: OutputFormat = "bcoo",
) -> Callable[..., Any]:
    """Detect sparsity, color, and return a function computing sparse Hessians.

    Combines [`hessian_coloring`][asdex.hessian_coloring]
    and [`hessian_from_coloring`][asdex.hessian_from_coloring]
    in one call.

    If ``f`` returns a squeezable shape like ``(1,)`` or ``(1, 1)``,
    it is automatically squeezed to scalar.

    Args:
        f: Scalar-valued function taking an array.
            Input may be multi-dimensional.
        input_shape: Shape of the input array (single-input mode).
            Mutually exclusive with ``input_shapes``.
        input_shapes: Pytree of shapes (multi-input mode).
            Mutually exclusive with ``input_shape``.
        argnums: Positional arguments to differentiate with respect to,
            mirroring ``jax.grad``.
            Only supported with multi-positional ``input_shapes``.
        mode: AD composition strategy for Hessian-vector products.
            ``"fwd_over_rev"`` uses forward-over-reverse,
            ``"rev_over_fwd"`` uses reverse-over-forward,
            ``"rev_over_rev"`` uses reverse-over-reverse.
            Defaults to ``"fwd_over_rev"``.
        symmetric: Whether to use symmetric (star) coloring.
            Defaults to True (exploits H = H^T for fewer colors).
        output_format: Type of the output matrix.
            ``"bcoo"`` returns a sparse matrix of type ``jax.experimental.sparse.BCOO`` (default),
            ``"dense"`` returns a dense matrix of type ``jax.Array``.

    Returns:
        A function that takes an input array and returns
            the Hessian of shape ``(n, n)``
            where ``n = x.size``.
            The output type depends on ``output_format``:
            a sparse ``jax.experimental.sparse.BCOO`` (default)
            or a dense ``jax.Array``.
    """
    coloring = _hessian_coloring(
        f,
        input_shape,
        input_shapes=input_shapes,
        argnums=argnums,
        mode=mode,
        symmetric=symmetric,
    )
    return hessian_from_coloring(f, coloring, output_format=output_format)


@overload
def value_and_hessian(
    f: Callable[[ArrayLike], ArrayLike],
    input_shape: int | tuple[int, ...],
    *,
    mode: HessianMode | None = None,
    symmetric: bool = True,
    output_format: Literal["bcoo"] = ...,
) -> Callable[[ArrayLike], tuple[jax.Array, BCOO]]: ...
@overload
def value_and_hessian(
    f: Callable[[ArrayLike], ArrayLike],
    input_shape: int | tuple[int, ...],
    *,
    mode: HessianMode | None = None,
    symmetric: bool = True,
    output_format: Literal["dense"],
) -> Callable[[ArrayLike], tuple[jax.Array, jax.Array]]: ...
@overload
def value_and_hessian(
    f: Callable[..., Any],
    *,
    input_shapes: Any,
    argnums: int | Sequence[int] | None = None,
    mode: HessianMode | None = None,
    symmetric: bool = True,
    output_format: OutputFormat = ...,
) -> Callable[..., tuple[Any, Any]]: ...
def value_and_hessian(
    f: Callable[..., Any],
    input_shape: int | tuple[int, ...] | None = None,
    *,
    input_shapes: Any = None,
    argnums: int | Sequence[int] | None = None,
    mode: HessianMode | None = None,
    symmetric: bool = True,
    output_format: OutputFormat = "bcoo",
) -> Callable[..., tuple[Any, Any]]:
    """Detect sparsity, color, and return a function computing value and sparse Hessian.

    Like [`hessian`][asdex.hessian],
    but can also return the primal value ``f(x)``
    without an extra forward pass.

    If ``f`` returns a squeezable shape like ``(1,)`` or ``(1, 1)``,
    it is automatically squeezed to scalar.

    Args:
        f: Scalar-valued function taking an array.
            Input may be multi-dimensional.
        input_shape: Shape of the input array (single-input mode).
            Mutually exclusive with ``input_shapes``.
        input_shapes: Pytree of shapes (multi-input mode).
            Mutually exclusive with ``input_shape``.
        argnums: Positional arguments to differentiate with respect to,
            mirroring ``jax.grad``.
            Only supported with multi-positional ``input_shapes``.
        mode: AD composition strategy for Hessian-vector products.
            ``"fwd_over_rev"`` uses forward-over-reverse,
            ``"rev_over_fwd"`` uses reverse-over-forward,
            ``"rev_over_rev"`` uses reverse-over-reverse.
            Defaults to ``"fwd_over_rev"``.
        symmetric: Whether to use symmetric (star) coloring.
            Defaults to True (exploits H = H^T for fewer colors).
        output_format: Type of the output matrix.
            ``"bcoo"`` returns a sparse matrix of type ``jax.experimental.sparse.BCOO`` (default),
            ``"dense"`` returns a dense matrix of type ``jax.Array``.

    Returns:
        A function that takes an input array and returns
            ``(f(x), H)`` where ``H`` is the Hessian
            of shape ``(n, n)``
            where ``n = x.size``.
            The output type depends on ``output_format``:
            a sparse ``jax.experimental.sparse.BCOO`` (default)
            or a dense ``jax.Array``.
    """
    coloring = _hessian_coloring(
        f,
        input_shape,
        input_shapes=input_shapes,
        argnums=argnums,
        mode=mode,
        symmetric=symmetric,
    )
    return value_and_hessian_from_coloring(f, coloring, output_format=output_format)


@overload
def jacobian_from_coloring(
    f: Callable[[ArrayLike], ArrayLike],
    coloring: ColoredPattern,
    output_format: Literal["bcoo"] = ...,
) -> Callable[[ArrayLike], BCOO]: ...
@overload
def jacobian_from_coloring(
    f: Callable[[ArrayLike], ArrayLike],
    coloring: ColoredPattern,
    output_format: Literal["dense"],
) -> Callable[[ArrayLike], jax.Array]: ...
@overload
def jacobian_from_coloring(
    f: Callable[..., Any],
    coloring: ColoredPattern,
    output_format: OutputFormat = ...,
) -> Callable[..., Any]: ...
def jacobian_from_coloring(
    f: Callable[..., Any],
    coloring: ColoredPattern,
    output_format: OutputFormat = "bcoo",
) -> Callable[..., Any]:
    """Build a sparse Jacobian function from a pre-computed coloring.

    Uses row coloring + VJPs or column coloring + JVPs,
    depending on which needs fewer colors.

    Args:
        f: Function taking an array and returning an array.
            Input and output may be multi-dimensional.
        coloring: Pre-computed [`ColoredPattern`][asdex.ColoredPattern]
            from [`jacobian_coloring`][asdex.jacobian_coloring].
        output_format: Type of the output matrix.
            ``"bcoo"`` returns a sparse matrix of type ``jax.experimental.sparse.BCOO`` (default),
            ``"dense"`` returns a dense matrix of type ``jax.Array``.

    Returns:
        A function that takes an input array and returns
            the Jacobian of shape ``(m, n)``
            where ``n = x.size`` and ``m = prod(output_shape)``.
            The output type depends on ``output_format``:
            a sparse ``jax.experimental.sparse.BCOO`` (default)
            or a dense ``jax.Array``.
    """
    _assert_output_format(output_format)

    if coloring.sparsity.is_multi_input:

        def jac_fn_multi(*args: Any) -> Any:
            return _eval_jacobian_multi(f, args, coloring, output_format)

        return jac_fn_multi

    def jac_fn(x: ArrayLike) -> BCOO | jax.Array:
        return _eval_jacobian(f, jnp.asarray(x), coloring, output_format)

    return jac_fn


@overload
def hessian_from_coloring(
    f: Callable[[ArrayLike], ArrayLike],
    coloring: ColoredPattern,
    output_format: Literal["bcoo"] = ...,
) -> Callable[[ArrayLike], BCOO]: ...
@overload
def hessian_from_coloring(
    f: Callable[[ArrayLike], ArrayLike],
    coloring: ColoredPattern,
    output_format: Literal["dense"],
) -> Callable[[ArrayLike], jax.Array]: ...
@overload
def hessian_from_coloring(
    f: Callable[..., Any],
    coloring: ColoredPattern,
    output_format: OutputFormat = ...,
) -> Callable[..., Any]: ...
def hessian_from_coloring(
    f: Callable[..., Any],
    coloring: ColoredPattern,
    output_format: OutputFormat = "bcoo",
) -> Callable[..., Any]:
    """Build a sparse Hessian function from a pre-computed coloring.

    Uses symmetric (star) coloring and Hessian-vector products by default.

    If ``f`` returns a squeezable shape like ``(1,)`` or ``(1, 1)``,
    it is automatically squeezed to scalar.

    Args:
        f: Scalar-valued function taking an array.
            Input may be multi-dimensional.
        coloring: Pre-computed [`ColoredPattern`][asdex.ColoredPattern]
            from [`hessian_coloring`][asdex.hessian_coloring].
        output_format: Type of the output matrix.
            ``"bcoo"`` returns a sparse matrix of type ``jax.experimental.sparse.BCOO`` (default),
            ``"dense"`` returns a dense matrix of type ``jax.Array``.

    Returns:
        A function that takes an input array and returns
            the Hessian of shape ``(n, n)``
            where ``n = x.size``.
            The output type depends on ``output_format``:
            a sparse ``jax.experimental.sparse.BCOO`` (default)
            or a dense ``jax.Array``.
    """
    _assert_output_format(output_format)

    if coloring.sparsity.is_multi_input:

        def hess_fn_multi(*args: Any) -> Any:
            return _eval_hessian_multi(f, args, coloring, output_format)

        return hess_fn_multi

    def hess_fn(x: ArrayLike) -> BCOO | jax.Array:
        return _eval_hessian(f, jnp.asarray(x), coloring, output_format)

    return hess_fn


@overload
def value_and_jacobian_from_coloring(
    f: Callable[[ArrayLike], ArrayLike],
    coloring: ColoredPattern,
    output_format: Literal["bcoo"] = ...,
) -> Callable[[ArrayLike], tuple[jax.Array, BCOO]]: ...
@overload
def value_and_jacobian_from_coloring(
    f: Callable[[ArrayLike], ArrayLike],
    coloring: ColoredPattern,
    output_format: Literal["dense"],
) -> Callable[[ArrayLike], tuple[jax.Array, jax.Array]]: ...
@overload
def value_and_jacobian_from_coloring(
    f: Callable[..., Any],
    coloring: ColoredPattern,
    output_format: OutputFormat = ...,
) -> Callable[..., tuple[Any, Any]]: ...
def value_and_jacobian_from_coloring(
    f: Callable[..., Any],
    coloring: ColoredPattern,
    output_format: OutputFormat = "bcoo",
) -> Callable[..., tuple[Any, Any]]:
    """Build a function computing value and sparse Jacobian from a pre-computed coloring.

    Like [`jacobian_from_coloring`][asdex.jacobian_from_coloring],
    but also returns the primal value ``f(x)`` without an extra forward pass.

    Args:
        f: Function taking an array and returning an array.
            Input and output may be multi-dimensional.
        coloring: Pre-computed [`ColoredPattern`][asdex.ColoredPattern]
            from [`jacobian_coloring`][asdex.jacobian_coloring].
        output_format: Type of the output matrix.
            ``"bcoo"`` returns a sparse matrix of type ``jax.experimental.sparse.BCOO`` (default),
            ``"dense"`` returns a dense matrix of type ``jax.Array``.

    Returns:
        A function that takes an input array and returns
            ``(f(x), J)`` where ``J`` is the Jacobian
            of shape ``(m, n)``
            where ``n = x.size`` and ``m = prod(output_shape)``.
            The output type depends on ``output_format``:
            a sparse ``jax.experimental.sparse.BCOO`` (default)
            or a dense ``jax.Array``.
    """
    _assert_output_format(output_format)

    if coloring.sparsity.is_multi_input:

        def val_jac_fn_multi(*args: Any) -> tuple[Any, Any]:
            return _eval_value_and_jacobian_multi(f, args, coloring, output_format)

        return val_jac_fn_multi

    def val_jac_fn(x: ArrayLike) -> tuple[jax.Array, BCOO | jax.Array]:
        return _eval_value_and_jacobian(f, jnp.asarray(x), coloring, output_format)

    return val_jac_fn


@overload
def value_and_hessian_from_coloring(
    f: Callable[[ArrayLike], ArrayLike],
    coloring: ColoredPattern,
    output_format: Literal["bcoo"] = ...,
) -> Callable[[ArrayLike], tuple[jax.Array, BCOO]]: ...
@overload
def value_and_hessian_from_coloring(
    f: Callable[[ArrayLike], ArrayLike],
    coloring: ColoredPattern,
    output_format: Literal["dense"],
) -> Callable[[ArrayLike], tuple[jax.Array, jax.Array]]: ...
@overload
def value_and_hessian_from_coloring(
    f: Callable[..., Any],
    coloring: ColoredPattern,
    output_format: OutputFormat = ...,
) -> Callable[..., tuple[Any, Any]]: ...
def value_and_hessian_from_coloring(
    f: Callable[..., Any],
    coloring: ColoredPattern,
    output_format: OutputFormat = "bcoo",
) -> Callable[..., tuple[Any, Any]]:
    """Build a function computing value and sparse Hessian from a pre-computed coloring.

    Like [`hessian_from_coloring`][asdex.hessian_from_coloring],
    but can also return the primal value ``f(x)`` without an extra forward pass.

    If ``f`` returns a squeezable shape like ``(1,)`` or ``(1, 1)``,
    it is automatically squeezed to scalar.

    Args:
        f: Scalar-valued function taking an array.
            Input may be multi-dimensional.
        coloring: Pre-computed [`ColoredPattern`][asdex.ColoredPattern]
            from [`hessian_coloring`][asdex.hessian_coloring].
        output_format: Type of the output matrix.
            ``"bcoo"`` returns a sparse matrix of type ``jax.experimental.sparse.BCOO`` (default),
            ``"dense"`` returns a dense matrix of type ``jax.Array``.

    Returns:
        A function that takes an input array and returns
            ``(f(x), H)`` where ``H`` is the Hessian
            of shape ``(n, n)``
            where ``n = x.size``.
            The output type depends on ``output_format``:
            a sparse ``jax.experimental.sparse.BCOO`` (default)
            or a dense ``jax.Array``.
    """
    _assert_output_format(output_format)

    if coloring.sparsity.is_multi_input:

        def val_hess_fn_multi(*args: Any) -> tuple[Any, Any]:
            return _eval_value_and_hessian_multi(f, args, coloring, output_format)

        return val_hess_fn_multi

    def val_hess_fn(x: ArrayLike) -> tuple[jax.Array, BCOO | jax.Array]:
        return _eval_value_and_hessian(f, jnp.asarray(x), coloring, output_format)

    return val_hess_fn


# Internal evaluation logic


def _eval_jacobian(
    f: Callable[[ArrayLike], ArrayLike],
    x: jax.Array,
    coloring: ColoredPattern,
    output_format: OutputFormat = "bcoo",
) -> BCOO | jax.Array:
    """Evaluate the sparse Jacobian of f at x."""
    n = x.size

    expected = coloring.sparsity.input_shape
    if x.shape != expected:
        raise ValueError(
            f"Input shape {x.shape} does not match the colored pattern, "
            f"which expects shape {expected}."
        )

    sparsity = coloring.sparsity
    m = sparsity.m
    out_shape = jax.eval_shape(f, jnp.zeros_like(x)).shape

    # Handle edge case: no outputs
    if m == 0:
        return _empty_result((0, n), output_format)

    # Handle edge case: all-zero Jacobian
    if sparsity.nnz == 0:
        return _empty_result((m, n), output_format)

    _assert_jacobian_mode(coloring.mode)
    match coloring.mode:
        case "rev":
            return _jacobian_rows(f, x, coloring, out_shape, output_format)
        case "fwd":
            return _jacobian_cols(f, x, coloring, output_format)
        case _ as unreachable:
            assert_never(unreachable)  # ty: ignore[type-assertion-failure]


def _eval_hessian(
    f: Callable[[ArrayLike], ArrayLike],
    x: jax.Array,
    coloring: ColoredPattern,
    output_format: OutputFormat = "bcoo",
) -> BCOO | jax.Array:
    """Evaluate the sparse Hessian of f at x.

    If ``f`` returns a squeezable shape like ``(1,)``,
    it is automatically squeezed to scalar.
    """
    f = _ensure_scalar(f, x.shape)
    n = x.size

    expected = coloring.sparsity.input_shape
    if x.shape != expected:
        raise ValueError(
            f"Input shape {x.shape} does not match the colored pattern, "
            f"which expects shape {expected}."
        )

    sparsity = coloring.sparsity

    # Handle edge case: all-zero Hessian
    if sparsity.nnz == 0:
        return _empty_result((n, n), output_format)

    grads = _compute_hvps(f, x, coloring)
    return _decompress(grads, coloring, output_format)


def _eval_value_and_jacobian(
    f: Callable[[ArrayLike], ArrayLike],
    x: jax.Array,
    coloring: ColoredPattern,
    output_format: OutputFormat = "bcoo",
) -> tuple[jax.Array, BCOO | jax.Array]:
    """Evaluate f(x) and the sparse Jacobian of f at x."""
    n = x.size

    expected = coloring.sparsity.input_shape
    if x.shape != expected:
        raise ValueError(
            f"Input shape {x.shape} does not match the colored pattern, "
            f"which expects shape {expected}."
        )

    sparsity = coloring.sparsity
    m = sparsity.m
    out_shape = jax.eval_shape(f, jnp.zeros_like(x)).shape

    # Handle edge case: no outputs
    if m == 0:
        y = jnp.asarray(f(x))
        return y, _empty_result((0, n), output_format)

    # Handle edge case: all-zero Jacobian
    if sparsity.nnz == 0:
        y = jnp.asarray(f(x))
        return y, _empty_result((m, n), output_format)

    _assert_jacobian_mode(coloring.mode)
    match coloring.mode:
        case "rev":
            return _value_and_jacobian_rows(f, x, coloring, out_shape, output_format)
        case "fwd":
            return _value_and_jacobian_cols(f, x, coloring, output_format)
        case _ as unreachable:
            assert_never(unreachable)  # ty: ignore[type-assertion-failure]


def _eval_value_and_hessian(
    f: Callable[[ArrayLike], ArrayLike],
    x: jax.Array,
    coloring: ColoredPattern,
    output_format: OutputFormat = "bcoo",
) -> tuple[jax.Array, BCOO | jax.Array]:
    """Evaluate f(x) and the sparse Hessian of f at x.

    If ``f`` returns a squeezable shape like ``(1,)``,
    it is automatically squeezed to scalar.
    """
    f = _ensure_scalar(f, x.shape)
    n = x.size

    expected = coloring.sparsity.input_shape
    if x.shape != expected:
        raise ValueError(
            f"Input shape {x.shape} does not match the colored pattern, "
            f"which expects shape {expected}."
        )

    sparsity = coloring.sparsity

    # Handle edge case: all-zero Hessian
    if sparsity.nnz == 0:
        y = jnp.asarray(f(x))
        return y, _empty_result((n, n), output_format)

    value, grads = _value_and_compute_hvps(f, x, coloring)
    return value, _decompress(grads, coloring, output_format)


# Private helpers: Jacobian


def _jacobian_rows(
    f: Callable[[ArrayLike], ArrayLike],
    x: jax.Array,
    coloring: ColoredPattern,
    out_shape: tuple[int, ...],
    output_format: OutputFormat = "bcoo",
) -> BCOO | jax.Array:
    """Compute sparse Jacobian via row coloring + VJPs."""
    seeds = jnp.asarray(coloring._seed_matrix, dtype=x.dtype)
    _, vjp_fn = jax.vjp(f, x)

    def single_vjp(seed: jax.Array) -> jax.Array:
        (grad,) = vjp_fn(seed.reshape(out_shape))
        return grad.ravel()

    compressed_jacobian = jax.vmap(single_vjp)(seeds)
    return _decompress(compressed_jacobian, coloring, output_format)


def _value_and_jacobian_rows(
    f: Callable[[ArrayLike], ArrayLike],
    x: jax.Array,
    coloring: ColoredPattern,
    out_shape: tuple[int, ...],
    output_format: OutputFormat = "bcoo",
) -> tuple[jax.Array, BCOO | jax.Array]:
    """Compute value and sparse Jacobian via row coloring + VJPs.

    The primal is free from the VJP forward pass.
    """
    seeds = jnp.asarray(coloring._seed_matrix, dtype=x.dtype)
    y, vjp_fn = jax.vjp(f, x)

    def single_vjp(seed: jax.Array) -> jax.Array:
        (grad,) = vjp_fn(seed.reshape(out_shape))
        return grad.ravel()

    compressed_jacobian = jax.vmap(single_vjp)(seeds)
    return y, _decompress(compressed_jacobian, coloring, output_format)


def _jacobian_cols(
    f: Callable[[ArrayLike], ArrayLike],
    x: jax.Array,
    coloring: ColoredPattern,
    output_format: OutputFormat = "bcoo",
) -> BCOO | jax.Array:
    """Compute sparse Jacobian via column coloring + JVPs."""
    seeds = jnp.asarray(coloring._seed_matrix, dtype=x.dtype)

    _, jvp_fn = jax.linearize(f, x)

    def single_jvp(seed: jax.Array) -> jax.Array:
        return jvp_fn(seed.reshape(x.shape)).ravel()

    compressed_jacobian = jax.vmap(single_jvp)(seeds)
    return _decompress(compressed_jacobian, coloring, output_format)


def _value_and_jacobian_cols(
    f: Callable[[ArrayLike], ArrayLike],
    x: jax.Array,
    coloring: ColoredPattern,
    output_format: OutputFormat = "bcoo",
) -> tuple[jax.Array, BCOO | jax.Array]:
    """Compute value and sparse Jacobian via column coloring + JVPs.

    Uses ``jax.linearize`` so the nonlinear forward pass runs only once.
    """
    seeds = jnp.asarray(coloring._seed_matrix, dtype=x.dtype)

    y, jvp_fn = jax.linearize(f, x)

    def single_jvp(seed: jax.Array) -> jax.Array:
        return jvp_fn(seed.reshape(x.shape)).ravel()

    compressed_jacobian = jax.vmap(single_jvp)(seeds)
    return y, _decompress(compressed_jacobian, coloring, output_format)


# Private helpers: Hessian


def _compute_hvps(
    f: Callable[[ArrayLike], ArrayLike],
    x: jax.Array,
    coloring: ColoredPattern,
) -> jax.Array:
    """Compute one HVP per color using pre-computed seed matrix.

    Returns ``hvps`` of shape ``(num_colors, n)``.
    """
    seeds = jnp.asarray(coloring._seed_matrix, dtype=x.dtype)

    _assert_hessian_mode(coloring.mode)
    match coloring.mode:
        case "fwd_over_rev":
            _, hvp_fn = jax.linearize(jax.grad(f), x)

            def single_hvp(v: jax.Array) -> jax.Array:
                return hvp_fn(v.reshape(x.shape)).ravel()

        case "rev_over_fwd":

            def single_hvp(v: jax.Array) -> jax.Array:
                return jax.grad(lambda p: jax.jvp(f, (p,), (v.reshape(x.shape),))[1])(
                    x
                ).ravel()

        case "rev_over_rev":
            _, hvp_fn = jax.vjp(jax.grad(f), x)

            def single_hvp(v: jax.Array) -> jax.Array:
                (hvp,) = hvp_fn(v.reshape(x.shape))
                return hvp.ravel()

        case _ as unreachable:
            assert_never(unreachable)  # ty: ignore[type-assertion-failure]

    return jax.vmap(single_hvp)(seeds)


def _value_and_compute_hvps(
    f: Callable[[ArrayLike], ArrayLike],
    x: jax.Array,
    coloring: ColoredPattern,
) -> tuple[jax.Array, jax.Array]:
    """Compute ``f(x)`` and one HVP per color using pre-computed seed matrix.

    Returns ``(f(x), hvps)`` where ``hvps`` has shape ``(num_colors, n)``.
    The primal is free for ``fwd_over_rev`` and ``rev_over_rev``;
    ``rev_over_fwd`` computes it with a separate ``f(x)`` call.
    """
    seeds = jnp.asarray(coloring._seed_matrix, dtype=x.dtype)

    _assert_hessian_mode(coloring.mode)
    match coloring.mode:
        case "fwd_over_rev":
            (value, _grad_at_x), hvp_fn = jax.linearize(jax.value_and_grad(f), x)

            def single_hvp(v: jax.Array) -> jax.Array:
                _tangent_of_value, hvp = hvp_fn(v.reshape(x.shape))
                return hvp.ravel()

        case "rev_over_fwd":
            value = jnp.asarray(f(x))

            def single_hvp(v: jax.Array) -> jax.Array:
                return jax.grad(lambda p: jax.jvp(f, (p,), (v.reshape(x.shape),))[1])(
                    x
                ).ravel()

        case "rev_over_rev":
            # TODO: f(x) is redundant with the forward pass inside grad(f).
            # Using value_and_grad + vjp would avoid it, but inflates every
            # VJP application with dead zero-cotangents for the value path.
            # Revisit if XLA reliably DCEs the zero branch.
            value = jnp.asarray(f(x))
            _, hvp_fn = jax.vjp(jax.grad(f), x)

            def single_hvp(v: jax.Array) -> jax.Array:
                (hvp,) = hvp_fn(v.reshape(x.shape))
                return hvp.ravel()

        case _ as unreachable:
            assert_never(unreachable)  # ty: ignore[type-assertion-failure]

    return value, jax.vmap(single_hvp)(seeds)


# Private helpers: decompression


def _decompress_data(coloring: ColoredPattern, compressed: jax.Array) -> jax.Array:
    """Extract sparse data values from compressed gradient rows.

    Uses pre-computed gather indices on the ``ColoredPattern``
    to vectorize the decompression step
    (no Python loop over nnz entries).

    Args:
        coloring: Colored sparsity pattern with cached indices.
        compressed: JAX array of shape ``(num_colors, vector_len)``,
            one row per color.

    Returns:
        Data array of shape ``(nnz,)`` in sparsity-pattern order.
    """
    return jax.lax.gather(
        compressed,
        coloring._gather_indices,
        dimension_numbers=jax.lax.GatherDimensionNumbers(
            offset_dims=(),
            collapsed_slice_dims=(0, 1),
            start_index_map=(0, 1),
        ),
        slice_sizes=(1, 1),
        unique_indices=True,
        mode=jax.lax.GatherScatterMode.PROMISE_IN_BOUNDS,
    )


def _decompress(
    compressed: jax.Array,
    coloring: ColoredPattern,
    output_format: OutputFormat = "bcoo",
) -> BCOO | jax.Array:
    """Extract sparse entries from compressed gradient rows.

    Calls :func:`_decompress_data` for the gather,
    then wraps the result as BCOO or scatters into a dense array
    depending on ``output_format``.

    Args:
        compressed: JAX array of shape ``(num_colors, vector_len)``,
            one row per color.
        coloring: Colored sparsity pattern with cached indices.
        output_format: Type of the output matrix.
            ``"bcoo"`` returns a sparse matrix of type ``jax.experimental.sparse.BCOO`` (default),
            ``"dense"`` returns a dense matrix of type ``jax.Array``.

    Returns:
        Matrix of shape ``coloring.sparsity.shape``.
            The output type depends on ``output_format``:
            a sparse ``jax.experimental.sparse.BCOO`` (default)
            or a dense ``jax.Array``.
    """
    data = _decompress_data(coloring, compressed)
    match output_format:
        case "bcoo":
            return coloring.sparsity.to_bcoo(data=data)
        case "dense":
            return _scatter_dense(coloring, data)
        case _ as unreachable:
            assert_never(unreachable)


def _scatter_dense(coloring: ColoredPattern, data: jax.Array) -> jax.Array:
    """Scatter sparse data values into a dense zero array.

    Args:
        coloring: Colored sparsity pattern with precomputed indices.
        data: Data array of shape ``(nnz,)``.

    Returns:
        Dense array of shape ``coloring.sparsity.shape``.
    """
    sparsity = coloring.sparsity
    indices = sparsity._bcoo_indices  # (nnz, 2)
    result = jnp.zeros(sparsity.shape, dtype=data.dtype)
    return result.at[indices[:, 0], indices[:, 1]].set(data)


def _empty_result(
    shape: tuple[int, ...], output_format: OutputFormat
) -> BCOO | jax.Array:
    """Return an all-zero matrix in the requested format."""
    match output_format:
        case "bcoo":
            return BCOO(
                (jnp.array([]), jnp.zeros((0, 2), dtype=jnp.int32)), shape=shape
            )
        case "dense":
            return jnp.zeros(shape)
        case _ as unreachable:
            assert_never(unreachable)


# Multi-input evaluation


def _eval_jacobian_multi(
    f: Callable[..., Any],
    args: tuple[Any, ...],
    coloring: ColoredPattern,
    output_format: OutputFormat,
) -> Any:
    """Evaluate a sparse Jacobian for a multi-input function at ``args``."""
    sparsity = coloring.sparsity
    _validate_multi_input_args(args, sparsity)

    m = sparsity.m
    n_selected = sparsity.n
    out_shape = _eval_out_shape_multi(f, args, sparsity)

    if m == 0 or sparsity.nnz == 0:
        dense = jnp.zeros((m, n_selected))
        return _pack_jacobian_blocks(dense, sparsity, output_format)

    _assert_jacobian_mode(coloring.mode)
    match coloring.mode:
        case "rev":
            compressed = _jacobian_rows_multi(f, args, coloring, out_shape)
        case "fwd":
            compressed = _jacobian_cols_multi(f, args, coloring)
        case _ as unreachable:
            assert_never(unreachable)  # ty: ignore[type-assertion-failure]

    data = _decompress_data(coloring, compressed)
    dense = _scatter_dense(coloring, data)
    return _pack_jacobian_blocks(dense, sparsity, output_format)


def _eval_value_and_jacobian_multi(
    f: Callable[..., Any],
    args: tuple[Any, ...],
    coloring: ColoredPattern,
    output_format: OutputFormat,
) -> tuple[Any, Any]:
    """Evaluate ``f(*args)`` and its sparse Jacobian in one pass."""
    sparsity = coloring.sparsity
    _validate_multi_input_args(args, sparsity)

    m = sparsity.m
    n_selected = sparsity.n

    if m == 0 or sparsity.nnz == 0:
        value = _call_f(f, args, sparsity)
        dense = jnp.zeros((m, n_selected))
        return value, _pack_jacobian_blocks(dense, sparsity, output_format)

    _assert_jacobian_mode(coloring.mode)
    match coloring.mode:
        case "rev":
            out_shape = _eval_out_shape_multi(f, args, sparsity)
            value, compressed = _value_and_jacobian_rows_multi(
                f, args, coloring, out_shape
            )
        case "fwd":
            value, compressed = _value_and_jacobian_cols_multi(f, args, coloring)
        case _ as unreachable:
            assert_never(unreachable)  # ty: ignore[type-assertion-failure]

    data = _decompress_data(coloring, compressed)
    dense = _scatter_dense(coloring, data)
    return value, _pack_jacobian_blocks(dense, sparsity, output_format)


def _eval_hessian_multi(
    f: Callable[..., Any],
    args: tuple[Any, ...],
    coloring: ColoredPattern,
    output_format: OutputFormat,
) -> Any:
    """Evaluate a sparse Hessian for a multi-input scalar function at ``args``."""
    sparsity = coloring.sparsity
    _validate_multi_input_args(args, sparsity)

    multi_positional = sparsity.is_multi_positional
    f_scalar = _ensure_scalar_multi(f, sparsity.input_shape, multi_positional)
    n_selected = sparsity.n

    if sparsity.nnz == 0:
        dense = jnp.zeros((n_selected, n_selected))
        return _pack_hessian_blocks(dense, sparsity, output_format)

    compressed = _compute_hvps_multi(f_scalar, args, coloring)
    data = _decompress_data(coloring, compressed)
    dense = _scatter_dense(coloring, data)
    return _pack_hessian_blocks(dense, sparsity, output_format)


def _eval_value_and_hessian_multi(
    f: Callable[..., Any],
    args: tuple[Any, ...],
    coloring: ColoredPattern,
    output_format: OutputFormat,
) -> tuple[Any, Any]:
    """Evaluate ``f(*args)`` and its sparse Hessian in one pass."""
    sparsity = coloring.sparsity
    _validate_multi_input_args(args, sparsity)

    multi_positional = sparsity.is_multi_positional
    f_scalar = _ensure_scalar_multi(f, sparsity.input_shape, multi_positional)
    n_selected = sparsity.n

    if sparsity.nnz == 0:
        value = _call_f(f_scalar, args, sparsity)
        dense = jnp.zeros((n_selected, n_selected))
        return value, _pack_hessian_blocks(dense, sparsity, output_format)

    value, compressed = _value_and_compute_hvps_multi(f_scalar, args, coloring)
    data = _decompress_data(coloring, compressed)
    dense = _scatter_dense(coloring, data)
    return value, _pack_hessian_blocks(dense, sparsity, output_format)


# Multi-input: Jacobian rows / cols


def _jacobian_rows_multi(
    f: Callable[..., Any],
    args: tuple[Any, ...],
    coloring: ColoredPattern,
    out_shape: tuple[int, ...],
) -> jax.Array:
    """Row-coloring VJPs over the combined selected input space."""
    sparsity = coloring.sparsity
    dtype = _args_dtype(args)
    _, vjp_fn = _vjp(f, args, sparsity)
    seeds = jnp.asarray(coloring._seed_matrix, dtype=dtype)

    def single_vjp(seed: jax.Array) -> jax.Array:
        cotangents = vjp_fn(seed.reshape(out_shape))
        return _flatten_selected_cotangents(cotangents, sparsity)

    return jax.vmap(single_vjp)(seeds)


def _value_and_jacobian_rows_multi(
    f: Callable[..., Any],
    args: tuple[Any, ...],
    coloring: ColoredPattern,
    out_shape: tuple[int, ...],
) -> tuple[jax.Array, jax.Array]:
    """Row-coloring VJPs, reusing the forward primal value."""
    sparsity = coloring.sparsity
    dtype = _args_dtype(args)
    value, vjp_fn = _vjp(f, args, sparsity)
    seeds = jnp.asarray(coloring._seed_matrix, dtype=dtype)

    def single_vjp(seed: jax.Array) -> jax.Array:
        cotangents = vjp_fn(seed.reshape(out_shape))
        return _flatten_selected_cotangents(cotangents, sparsity)

    return value, jax.vmap(single_vjp)(seeds)


def _jacobian_cols_multi(
    f: Callable[..., Any],
    args: tuple[Any, ...],
    coloring: ColoredPattern,
) -> jax.Array:
    """Column-coloring JVPs over the combined selected input space."""
    sparsity = coloring.sparsity
    dtype = _args_dtype(args)
    _, jvp_fn = _linearize(f, args, sparsity)
    seeds = jnp.asarray(coloring._seed_matrix, dtype=dtype)

    def single_jvp(seed: jax.Array) -> jax.Array:
        tangents = _build_tangents_from_seed(seed, args, sparsity)
        if sparsity.is_multi_positional:
            return jvp_fn(*tangents).ravel()
        return jvp_fn(tangents[0]).ravel()

    return jax.vmap(single_jvp)(seeds)


def _value_and_jacobian_cols_multi(
    f: Callable[..., Any],
    args: tuple[Any, ...],
    coloring: ColoredPattern,
) -> tuple[jax.Array, jax.Array]:
    """Column-coloring JVPs, reusing the forward primal value."""
    sparsity = coloring.sparsity
    dtype = _args_dtype(args)
    value, jvp_fn = _linearize(f, args, sparsity)
    seeds = jnp.asarray(coloring._seed_matrix, dtype=dtype)

    def single_jvp(seed: jax.Array) -> jax.Array:
        tangents = _build_tangents_from_seed(seed, args, sparsity)
        if sparsity.is_multi_positional:
            return jvp_fn(*tangents).ravel()
        return jvp_fn(tangents[0]).ravel()

    return value, jax.vmap(single_jvp)(seeds)


# Multi-input: Hessian HVPs


def _compute_hvps_multi(
    f: Callable[..., Any],
    args: tuple[Any, ...],
    coloring: ColoredPattern,
) -> jax.Array:
    """One HVP per color for a multi-input scalar function."""
    sparsity = coloring.sparsity
    dtype = _args_dtype(args)
    grad_argnums = _grad_argnums_for_sparsity(sparsity)

    seeds = jnp.asarray(coloring._seed_matrix, dtype=dtype)
    _assert_hessian_mode(coloring.mode)
    match coloring.mode:
        case "fwd_over_rev":
            grad_fn = jax.grad(f, argnums=grad_argnums)
            _, hvp_fn = _linearize(grad_fn, args, sparsity)

            def single_hvp(v: jax.Array) -> jax.Array:
                tangents = _build_tangents_from_seed(v, args, sparsity)
                if sparsity.is_multi_positional:
                    tangent_out = hvp_fn(*tangents)
                else:
                    tangent_out = hvp_fn(tangents[0])
                return _flatten_grad_output(tangent_out, sparsity)

        case "rev_over_fwd":

            def single_hvp(v: jax.Array) -> jax.Array:
                tangents = _build_tangents_from_seed(v, args, sparsity)

                def inner(*primals: Any) -> jax.Array:
                    if sparsity.is_multi_positional:
                        _, out_tangent = jax.jvp(f, primals, tangents)
                    else:
                        _, out_tangent = jax.jvp(f, (primals[0],), (tangents[0],))
                    return out_tangent

                grads = jax.grad(inner, argnums=grad_argnums)(*args)
                return _flatten_grad_output(grads, sparsity)

        case "rev_over_rev":
            grad_fn = jax.grad(f, argnums=grad_argnums)
            _, hvp_fn = _vjp(grad_fn, args, sparsity)

            def single_hvp(v: jax.Array) -> jax.Array:
                cotangent_out = _build_grad_output_from_seed(v, args, sparsity)
                cotangents = hvp_fn(cotangent_out)
                return _flatten_selected_cotangents(cotangents, sparsity)

        case _ as unreachable:
            assert_never(unreachable)  # ty: ignore[type-assertion-failure]

    return jax.vmap(single_hvp)(seeds)


def _value_and_compute_hvps_multi(
    f: Callable[..., Any],
    args: tuple[Any, ...],
    coloring: ColoredPattern,
) -> tuple[jax.Array, jax.Array]:
    """``f(*args)`` and one HVP per color for a multi-input scalar function."""
    sparsity = coloring.sparsity
    dtype = _args_dtype(args)
    grad_argnums = _grad_argnums_for_sparsity(sparsity)

    seeds = jnp.asarray(coloring._seed_matrix, dtype=dtype)
    _assert_hessian_mode(coloring.mode)
    match coloring.mode:
        case "fwd_over_rev":
            val_and_grad = jax.value_and_grad(f, argnums=grad_argnums)
            (value, _g), hvp_fn = _linearize(val_and_grad, args, sparsity)

            def single_hvp(v: jax.Array) -> jax.Array:
                tangents = _build_tangents_from_seed(v, args, sparsity)
                if sparsity.is_multi_positional:
                    _value_tangent, tangent_out = hvp_fn(*tangents)
                else:
                    _value_tangent, tangent_out = hvp_fn(tangents[0])
                return _flatten_grad_output(tangent_out, sparsity)

        case "rev_over_fwd":
            value = jnp.asarray(_call_f(f, args, sparsity))

            def single_hvp(v: jax.Array) -> jax.Array:
                tangents = _build_tangents_from_seed(v, args, sparsity)

                def inner(*primals: Any) -> jax.Array:
                    if sparsity.is_multi_positional:
                        _, out_tangent = jax.jvp(f, primals, tangents)
                    else:
                        _, out_tangent = jax.jvp(f, (primals[0],), (tangents[0],))
                    return out_tangent

                grads = jax.grad(inner, argnums=grad_argnums)(*args)
                return _flatten_grad_output(grads, sparsity)

        case "rev_over_rev":
            value = jnp.asarray(_call_f(f, args, sparsity))
            grad_fn = jax.grad(f, argnums=grad_argnums)
            _, hvp_fn = _vjp(grad_fn, args, sparsity)

            def single_hvp(v: jax.Array) -> jax.Array:
                cotangent_out = _build_grad_output_from_seed(v, args, sparsity)
                cotangents = hvp_fn(cotangent_out)
                return _flatten_selected_cotangents(cotangents, sparsity)

        case _ as unreachable:
            assert_never(unreachable)  # ty: ignore[type-assertion-failure]

    return value, jax.vmap(single_hvp)(seeds)


# Multi-input: plumbing helpers


def _validate_multi_input_args(
    args: tuple[Any, ...], sparsity: SparsityPattern
) -> None:
    """Check args match the pattern's declared pytree structure and leaf shapes."""
    if sparsity.is_multi_positional:
        user_tree = jax.tree_util.tree_structure(args)
    else:
        if len(args) != 1:
            raise ValueError(
                f"Expected a single pytree argument, got {len(args)} args."
            )
        user_tree = jax.tree_util.tree_structure(args[0])
    if user_tree != sparsity.input_treedef:
        raise ValueError(
            f"Input pytree structure {user_tree} does not match the colored "
            f"pattern, which expects {sparsity.input_treedef}."
        )
    flat = _flatten_user_args(args, sparsity)
    for i, (leaf, expected) in enumerate(zip(flat, sparsity.leaf_shapes, strict=True)):
        leaf_shape = tuple(getattr(leaf, "shape", ()))
        if leaf_shape != tuple(expected):
            raise ValueError(
                f"Input leaf {i} shape {leaf_shape} does not match expected "
                f"{tuple(expected)}."
            )


def _flatten_user_args(args: tuple[Any, ...], sparsity: SparsityPattern) -> list[Any]:
    """Flatten the user-passed args into the leaf order used by detection."""
    if sparsity.is_multi_positional:
        return jax.tree_util.tree_leaves(args)
    return jax.tree_util.tree_leaves(args[0])


def _args_dtype(args: tuple[Any, ...]) -> Any:
    """Dtype for seed arrays, taken from the first leaf we can find."""
    for leaf in jax.tree_util.tree_leaves(args):
        dtype = getattr(leaf, "dtype", None)
        if dtype is not None:
            return dtype
    return jnp.float32


def _call_f(
    f: Callable[..., Any], args: tuple[Any, ...], sparsity: SparsityPattern
) -> jax.Array:
    """Call ``f`` with the appropriate calling convention for ``sparsity``."""
    if sparsity.is_multi_positional:
        return jnp.asarray(f(*args))
    return jnp.asarray(f(args[0]))


def _eval_out_shape_multi(
    f: Callable[..., Any],
    args: tuple[Any, ...],
    sparsity: SparsityPattern,
) -> tuple[int, ...]:
    """Output shape of ``f`` applied to ``args``."""
    if sparsity.is_multi_positional:
        return jax.eval_shape(f, *args).shape
    return jax.eval_shape(f, args[0]).shape


def _vjp(
    f: Callable[..., Any],
    args: tuple[Any, ...],
    sparsity: SparsityPattern,
) -> tuple[Any, Callable[..., Any]]:
    """``jax.vjp`` with the right calling convention for ``sparsity``."""
    if sparsity.is_multi_positional:
        return jax.vjp(f, *args)
    return jax.vjp(f, args[0])


def _linearize(
    f: Callable[..., Any],
    args: tuple[Any, ...],
    sparsity: SparsityPattern,
) -> tuple[Any, Callable[..., Any]]:
    """``jax.linearize`` with the right calling convention for ``sparsity``."""
    if sparsity.is_multi_positional:
        return jax.linearize(f, *args)
    return jax.linearize(f, args[0])


def _grad_argnums_for_sparsity(sparsity: SparsityPattern) -> int | tuple[int, ...]:
    """`argnums` to pass to ``jax.grad`` for Hessian HVPs."""
    if not sparsity.is_multi_positional:
        return 0
    selected = tuple(i for i, sel in enumerate(sparsity.resolved_selected_mask) if sel)
    if len(selected) == 1:
        return selected[0]
    return selected


def _build_tangents_from_seed(
    seed: jax.Array,
    args: tuple[Any, ...],
    sparsity: SparsityPattern,
) -> tuple[Any, ...]:
    """Split a ``(n_selected,)`` seed into a tangent pytree matching ``args``.

    Non-selected leaves get zero tangents so they have no effect on the JVP.
    """
    leaf_shapes = sparsity.leaf_shapes
    leaf_sizes = sparsity.leaf_sizes
    selected_mask = sparsity.resolved_selected_mask
    selected_sizes = [
        s for s, sel in zip(leaf_sizes, selected_mask, strict=True) if sel
    ]

    chunks: list[jax.Array] = []
    offset = 0
    for size in selected_sizes:
        chunks.append(seed[offset : offset + size])
        offset += size

    full_leaves = _flatten_user_args(args, sparsity)
    tangent_leaves: list[jax.Array] = []
    chunk_idx = 0
    for leaf, shape, sel in zip(full_leaves, leaf_shapes, selected_mask, strict=True):
        if sel:
            tangent_leaves.append(chunks[chunk_idx].reshape(tuple(shape)))
            chunk_idx += 1
        else:
            tangent_leaves.append(jnp.zeros(tuple(shape), dtype=seed.dtype))
            del leaf

    if sparsity.is_multi_positional:
        tangent_args = jax.tree_util.tree_unflatten(
            sparsity.input_treedef, tangent_leaves
        )
        return tuple(tangent_args)
    tangent_pytree = jax.tree_util.tree_unflatten(
        sparsity.input_treedef, tangent_leaves
    )
    return (tangent_pytree,)


def _flatten_selected_cotangents(
    cotangents: Any, sparsity: SparsityPattern
) -> jax.Array:
    """Flatten selected cotangent leaves into a ``(n_selected,)`` vector.

    ``jax.vjp(f, *xs)`` returns a tuple of cotangents matching the primals.
    Non-selected leaves are ignored; selected leaves are raveled and concatenated.
    """
    if sparsity.is_multi_positional:
        leaves = jax.tree_util.tree_leaves(cotangents)
    else:
        # `cotangents` is a 1-tuple containing the arg's cotangent pytree.
        leaves = jax.tree_util.tree_leaves(cotangents[0])

    selected_mask = sparsity.resolved_selected_mask
    parts = [
        leaf.ravel() for leaf, sel in zip(leaves, selected_mask, strict=True) if sel
    ]
    if not parts:
        return jnp.zeros((0,))
    return jnp.concatenate(parts)


def _flatten_grad_output(out: Any, sparsity: SparsityPattern) -> jax.Array:
    """Flatten a gradient tangent/output into ``(n_selected,)``.

    The output of ``jax.grad(f, argnums=...)`` matches only the selected
    positions, so every leaf we see here contributes to the flat vector.
    """
    leaves = jax.tree_util.tree_leaves(out)
    if not leaves:
        return jnp.zeros((0,))
    return jnp.concatenate([leaf.ravel() for leaf in leaves])


def _build_grad_output_from_seed(
    seed: jax.Array,
    args: tuple[Any, ...],
    sparsity: SparsityPattern,
) -> Any:
    """Build a gradient-shaped cotangent pytree from a ``(n_selected,)`` seed.

    Mirrors ``_flatten_grad_output`` in reverse: used as the seed cotangent
    passed into the outer VJP in ``rev_over_rev`` Hessian mode.
    """
    leaf_shapes = sparsity.leaf_shapes
    leaf_sizes = sparsity.leaf_sizes
    selected_mask = sparsity.resolved_selected_mask

    selected_shapes = [
        tuple(shape)
        for shape, sel in zip(leaf_shapes, selected_mask, strict=True)
        if sel
    ]
    selected_sizes = [
        s for s, sel in zip(leaf_sizes, selected_mask, strict=True) if sel
    ]

    chunks: list[jax.Array] = []
    offset = 0
    for size, shape in zip(selected_sizes, selected_shapes, strict=True):
        chunks.append(seed[offset : offset + size].reshape(shape))
        offset += size

    del args
    if sparsity.is_multi_positional:
        grad_argnums = _grad_argnums_for_sparsity(sparsity)
        if isinstance(grad_argnums, int):
            assert len(chunks) == 1
            return chunks[0]
        return tuple(chunks)

    # Single-pytree: reconstruct the pytree over all leaves.
    return jax.tree_util.tree_unflatten(sparsity.input_treedef, list(chunks))


# Multi-input: block packing


def _selected_block_info(
    sparsity: SparsityPattern,
) -> list[tuple[int, tuple[int, ...], int]]:
    """Triples of ``(leaf_index, leaf_shape, leaf_size)`` for selected leaves."""
    selected_mask = sparsity.resolved_selected_mask
    leaf_shapes = sparsity.leaf_shapes
    leaf_sizes = sparsity.leaf_sizes
    return [
        (i, tuple(leaf_shapes[i]), leaf_sizes[i])
        for i, sel in enumerate(selected_mask)
        if sel
    ]


def _pack_jacobian_blocks(
    dense: jax.Array,
    sparsity: SparsityPattern,
    output_format: OutputFormat,
) -> Any:
    """Split a ``(m, n_selected)`` dense matrix into per-leaf Jacobian blocks."""
    selected = _selected_block_info(sparsity)
    m = sparsity.m

    blocks: list[jax.Array | BCOO] = []
    offset = 0
    for _, shape, size in selected:
        chunk = dense[:, offset : offset + size]
        block: jax.Array | BCOO = chunk.reshape((m, *shape))
        if output_format == "bcoo":
            block = BCOO.fromdense(block)
        blocks.append(block)
        offset += size

    return _pack_by_argnums(blocks, sparsity)


def _pack_hessian_blocks(
    dense: jax.Array,
    sparsity: SparsityPattern,
    output_format: OutputFormat,
) -> Any:
    """Split a ``(n_sel, n_sel)`` dense matrix into a ``k x k`` block grid."""
    selected = _selected_block_info(sparsity)

    rows: list[list[jax.Array | BCOO]] = []
    row_offset = 0
    for _, row_shape, row_size in selected:
        col_offset = 0
        row_blocks: list[jax.Array | BCOO] = []
        for _, col_shape, col_size in selected:
            chunk = dense[
                row_offset : row_offset + row_size,
                col_offset : col_offset + col_size,
            ]
            block: jax.Array | BCOO = chunk.reshape(row_shape + col_shape)
            if output_format == "bcoo":
                block = BCOO.fromdense(block)
            row_blocks.append(block)
            col_offset += col_size
        rows.append(row_blocks)
        row_offset += row_size

    def pack_single(inner_leaves: list[Any]) -> Any:
        # Outer pytree wraps each already-packed row.
        return jax.tree_util.tree_unflatten(sparsity.input_treedef, inner_leaves)

    if sparsity.is_multi_positional:
        argnums = sparsity.argnums
        selected_mask = sparsity.resolved_selected_mask
        if argnums is None:
            all_selected = all(selected_mask)
            if all_selected and sum(selected_mask) == 1:
                return rows[0][0]
        if isinstance(argnums, int):
            assert len(rows) == 1
            assert len(rows[0]) == 1
            return rows[0][0]
        return tuple(tuple(row) for row in rows)

    # Single-pytree: build a pytree of pytrees (inner axis nested under outer).
    inner_pytrees = [pack_single(row) for row in rows]
    return jax.tree_util.tree_unflatten(sparsity.input_treedef, inner_pytrees)


def _pack_by_argnums(
    blocks: Sequence[Any],
    sparsity: SparsityPattern,
) -> Any:
    """Pack Jacobian blocks according to ``argnums`` / pytree structure."""
    if sparsity.is_multi_positional:
        argnums = sparsity.argnums
        if isinstance(argnums, int):
            assert len(blocks) == 1
            return blocks[0]
        return tuple(blocks)
    return jax.tree_util.tree_unflatten(sparsity.input_treedef, list(blocks))
