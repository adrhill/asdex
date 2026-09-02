"""High-level Jacobian and Hessian coloring entry points.

Mirrors the role of ``interface.jl`` in SparseMatrixColorings.jl: mode
resolution, empty-pattern shortcuts, and dense-baseline warnings on top of
the algorithm modules.

Algorithms adapted from SparseMatrixColorings.jl (MIT license)
Copyright (c) 2024 Guillaume Dalle, Alexis Montoison, and contributors
https://github.com/gdalle/SparseMatrixColorings.jl
See also: Dalle & Montoison (2025), https://arxiv.org/abs/2505.07308

- https://github.com/gdalle/SparseMatrixColorings.jl/blob/main/src/interface.jl
"""

import warnings
from collections.abc import Callable, Sequence
from typing import Any, assert_never

import numpy as np
from jax.experimental.sparse import BCOO
from numpy.typing import NDArray

from asdex._defaults import (
    _DEFAULT_ARGNUMS,
    _DEFAULT_HAS_AUX,
    _DEFAULT_HESSIAN_MODE,
    _DEFAULT_MODE,
    _DEFAULT_SYMMETRIC_HESSIAN,
    _DEFAULT_SYMMETRIC_JACOBIAN,
    _DEFAULT_SYMMETRIC_JACOBIAN_MODE,
)
from asdex._docstrings import _fill_doc
from asdex._errors import DenseColoringWarning
from asdex._pattern import ColoredPattern, SparsityPattern
from asdex._types import (
    HessianMode,
    JacobianMode,
    _assert_hessian_mode,
    _assert_jacobian_mode,
)
from asdex.coloring._color_greedy import color_cols, color_rows
from asdex.coloring._color_symmetric import color_symmetric
from asdex.detection import hessian_sparsity as _detect_hessian_sparsity
from asdex.detection import jacobian_sparsity as _detect_jacobian_sparsity


@_fill_doc
def jacobian_coloring(
    f: Callable,
    *args: Any,
    argnums: int | Sequence[int] = _DEFAULT_ARGNUMS,
    has_aux: bool = _DEFAULT_HAS_AUX,
    mode: JacobianMode | None = _DEFAULT_MODE,
    symmetric: bool = _DEFAULT_SYMMETRIC_JACOBIAN,
    **kwargs: Any,
) -> ColoredPattern:
    """Detect Jacobian sparsity and color in one step.

    Args:
        f: {f_jac}
        *args: {sample_args}
        argnums: {argnums}
        has_aux: {has_aux_detect}
        mode: {mode_jac_coloring}
        symmetric: {symmetric_jac}
        **kwargs: {sample_kwargs_detect}

    Returns:
        A [`ColoredPattern`][asdex.ColoredPattern] ready for [`jacobian_from_coloring`][asdex.jacobian_from_coloring].
    """
    sparsity = _detect_jacobian_sparsity(
        f, *args, argnums=argnums, has_aux=has_aux, **kwargs
    )
    return jacobian_coloring_from_sparsity(sparsity, symmetric=symmetric, mode=mode)


@_fill_doc
def hessian_coloring(
    f: Callable,
    *args: Any,
    argnums: int | Sequence[int] = _DEFAULT_ARGNUMS,
    has_aux: bool = _DEFAULT_HAS_AUX,
    mode: HessianMode | None = _DEFAULT_MODE,
    symmetric: bool = _DEFAULT_SYMMETRIC_HESSIAN,
    **kwargs: Any,
) -> ColoredPattern:
    """Detect Hessian sparsity and color in one step.

    Args:
        f: {f_hess}
        *args: {sample_args}
        argnums: {argnums}
        has_aux: {has_aux_detect}
        mode: {mode_hess}
        symmetric: {symmetric_hess}
        **kwargs: {sample_kwargs_detect}

    Returns:
        A [`ColoredPattern`][asdex.ColoredPattern] ready for [`hessian_from_coloring`][asdex.hessian_from_coloring].
    """
    sparsity = _detect_hessian_sparsity(
        f, *args, argnums=argnums, has_aux=has_aux, **kwargs
    )
    return hessian_coloring_from_sparsity(sparsity, symmetric=symmetric, mode=mode)


@_fill_doc
def jacobian_coloring_from_sparsity(
    sparsity: SparsityPattern | NDArray | BCOO,
    *,
    mode: JacobianMode | None = _DEFAULT_MODE,
    symmetric: bool = _DEFAULT_SYMMETRIC_JACOBIAN,
) -> ColoredPattern:
    """Color a sparsity pattern for sparse Jacobian computation.

    Assigns colors so that same-colored rows (or columns) can be
    computed together in a single VJP (or JVP).

    Args:
        sparsity: {sparsity_jac}
        mode: {mode_jac_coloring}
        symmetric: {symmetric_jac}

    Returns:
        A [`ColoredPattern`][asdex.ColoredPattern] ready for [`jacobian_from_coloring`][asdex.jacobian_from_coloring].
    """
    sparsity = _coerce_sparsity(sparsity, "jacobian")

    if mode is not None:
        _assert_jacobian_mode(mode)

    if symmetric:
        return _color_jacobian_symmetric(
            sparsity,
            mode if mode is not None else _DEFAULT_SYMMETRIC_JACOBIAN_MODE,
        )

    # Nothing to compute when there are no nonzeros.
    if sparsity.nnz == 0:
        return _empty_jacobian_pattern(sparsity, mode)

    match mode:
        case "rev":
            colors_arr, num = color_rows(sparsity)
            result = ColoredPattern(
                sparsity,
                colors=colors_arr,
                num_colors=num,
                symmetric=False,
                mode="rev",
            )
            _warn_if_dense(num, sparsity.m, "Jacobian", sparsity.shape)
            return result

        case "fwd":
            colors_arr, num = color_cols(sparsity)
            result = ColoredPattern(
                sparsity,
                colors=colors_arr,
                num_colors=num,
                symmetric=False,
                mode="fwd",
            )
            _warn_if_dense(num, sparsity.n, "Jacobian", sparsity.shape)
            return result

        case None:
            # Pick whichever uses fewer colors.
            # Ties go to fwd (JVPs are cheaper than VJPs).
            row_colors, num_row = color_rows(sparsity)
            col_colors, num_col = color_cols(sparsity)

            if num_col <= num_row:
                result = ColoredPattern(
                    sparsity,
                    colors=col_colors,
                    num_colors=num_col,
                    symmetric=False,
                    mode="fwd",
                )
                _warn_if_dense(num_col, sparsity.n, "Jacobian", sparsity.shape)
                return result
            result = ColoredPattern(
                sparsity,
                colors=row_colors,
                num_colors=num_row,
                symmetric=False,
                mode="rev",
            )
            _warn_if_dense(num_row, sparsity.m, "Jacobian", sparsity.shape)
            return result

        case _ as unreachable:
            assert_never(unreachable)


@_fill_doc
def hessian_coloring_from_sparsity(
    sparsity: SparsityPattern | NDArray | BCOO,
    *,
    mode: HessianMode | None = _DEFAULT_MODE,
    symmetric: bool = _DEFAULT_SYMMETRIC_HESSIAN,
) -> ColoredPattern:
    """Color a sparsity pattern for sparse Hessian computation.

    Args:
        sparsity: {sparsity_hess}
        mode: {mode_hess}
        symmetric: {symmetric_hess}

    Returns:
        A [`ColoredPattern`][asdex.ColoredPattern] ready for [`hessian_from_coloring`][asdex.hessian_from_coloring].
    """
    sparsity = _coerce_sparsity(sparsity, "hessian")

    if sparsity.m != sparsity.n:
        msg = f"Hessian sparsity pattern must be square, got shape {sparsity.shape}."
        raise ValueError(msg)

    resolved_mode: HessianMode = mode if mode is not None else _DEFAULT_HESSIAN_MODE
    if mode is not None:
        _assert_hessian_mode(mode)

    if sparsity.nnz == 0:
        return _empty_hessian_pattern(sparsity, symmetric=symmetric, mode=resolved_mode)

    if symmetric:
        colors_arr, num, star_set = color_symmetric(sparsity)
        result = ColoredPattern(
            sparsity,
            colors=colors_arr,
            num_colors=num,
            symmetric=True,
            mode=resolved_mode,
            star_set=star_set,
        )
        _warn_if_dense(num, sparsity.n, "Hessian", sparsity.shape)
        return result

    # Non-symmetric: use column coloring (HVPs seed the input space)
    colors_arr, num = color_cols(sparsity)
    result = ColoredPattern(
        sparsity,
        colors=colors_arr,
        num_colors=num,
        symmetric=False,
        mode=resolved_mode,
    )
    _warn_if_dense(num, sparsity.n, "Hessian", sparsity.shape)
    return result


def _coerce_sparsity(
    sparsity: SparsityPattern | NDArray | BCOO, caller: str
) -> SparsityPattern:
    """Convert a sparsity-like input to a SparsityPattern.

    Args:
        sparsity: A SparsityPattern, NumPy array, or JAX BCOO matrix.
        caller: ``"jacobian"`` or ``"hessian"``, used in error messages.
    """
    if isinstance(sparsity, SparsityPattern):
        return sparsity
    if isinstance(sparsity, np.ndarray):
        return SparsityPattern.from_dense(sparsity)
    if isinstance(sparsity, BCOO):
        return SparsityPattern.from_bcoo(sparsity)
    msg = (
        f"Expected a SparsityPattern, NumPy array, or JAX BCOO matrix, "
        f"got {type(sparsity).__name__}. "
        f"Use {caller}_sparsity() to detect the sparsity pattern first."
    )
    raise TypeError(msg)


def _color_jacobian_symmetric(
    sparsity: SparsityPattern,
    mode: JacobianMode,
) -> ColoredPattern:
    """Color a Jacobian pattern using symmetric (star) coloring.

    Args:
        sparsity: Sparsity pattern (must be square).
        mode: The resolved AD mode.
    """
    if sparsity.nnz == 0:
        if sparsity.m != sparsity.n:
            msg = f"Symmetric coloring requires a square pattern, got shape {sparsity.shape}"
            raise ValueError(msg)
        return ColoredPattern(
            sparsity,
            colors=np.full(sparsity.n, -1, dtype=np.int32),
            num_colors=0,
            symmetric=True,
            mode=mode,
        )
    colors_arr, num, star_set = color_symmetric(sparsity)
    result = ColoredPattern(
        sparsity,
        colors=colors_arr,
        num_colors=num,
        symmetric=True,
        mode=mode,
        star_set=star_set,
    )
    _warn_if_dense(num, sparsity.n, "Jacobian", sparsity.shape)
    return result


def _empty_jacobian_pattern(
    sparsity: SparsityPattern,
    mode: JacobianMode | None,
) -> ColoredPattern:
    """Build a ``ColoredPattern`` for an all-zero Jacobian sparsity pattern.

    Args:
        sparsity: Sparsity pattern with ``nnz == 0``.
        mode: AD mode (``None`` defaults to ``"fwd"``).

    Returns:
        A ``ColoredPattern`` with zero colors.
    """
    match mode:
        case "rev":
            n_vertices = sparsity.m
            mode: JacobianMode = "rev"
        case "fwd" | None:
            n_vertices = sparsity.n
            mode = "fwd"
        case _ as unreachable:
            assert_never(unreachable)
    return ColoredPattern(
        sparsity,
        colors=np.full(n_vertices, -1, dtype=np.int32),
        num_colors=0,
        symmetric=False,
        mode=mode,
    )


def _empty_hessian_pattern(
    sparsity: SparsityPattern,
    *,
    symmetric: bool,
    mode: HessianMode,
) -> ColoredPattern:
    """Build a ``ColoredPattern`` for an all-zero Hessian sparsity pattern."""
    return ColoredPattern(
        sparsity,
        colors=np.full(sparsity.n, -1, dtype=np.int32),
        num_colors=0,
        symmetric=symmetric,
        mode=mode,
    )


def _warn_if_dense(
    num_colors: int,
    dense_baseline: int,
    kind: str,
    shape: tuple[int, int],
) -> None:
    """Warn if coloring uses as many colors as the dense baseline."""
    if num_colors >= dense_baseline:
        m, n = shape
        warnings.warn(
            f"Coloring used {num_colors} colors for a {m}\u00d7{n} {kind}"
            f" (same as the dense case).\n"
            f"No speedup over dense differentiation.\n"
            f"Suppress this warning with:"
            f' warnings.filterwarnings("ignore",'
            f" category=asdex.DenseColoringWarning)",
            category=DenseColoringWarning,
            stacklevel=3,
        )
