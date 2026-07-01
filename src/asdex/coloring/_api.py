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
    _DEFAULT_MODE,
    _DEFAULT_POSTPROCESS,
    _DEFAULT_SYMMETRIC_HESSIAN,
    _DEFAULT_SYMMETRIC_JACOBIAN,
)
from asdex._docstrings import _fill_doc
from asdex._modes import (
    HessianMode,
    JacobianMode,
    _assert_hessian_mode,
    _assert_jacobian_mode,
)
from asdex._pattern import ColoredPattern, SparsityPattern
from asdex.coloring._color_greedy import color_cols, color_rows
from asdex.coloring._color_symmetric import color_symmetric
from asdex.coloring._types import DenseColoringWarning
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
    postprocess: bool = _DEFAULT_POSTPROCESS,
    **kwargs: Any,
) -> ColoredPattern:
    """Detect Jacobian sparsity and color in one step.

    Args:
        f: {f_jac}
        *args: {sample_args}
        argnums: {argnums}
        has_aux: {has_aux_detect}
        mode: AD mode.
            ``"fwd"`` uses JVPs (forward-mode AD),
            ``"rev"`` uses VJPs (reverse-mode AD),
            ``None`` picks whichever of fwd/rev needs fewer colors
            (unless ``symmetric`` is True, in which case defaults to ``"fwd"``).
        symmetric: {symmetric}
            Requires a square Jacobian.
        postprocess: Only read when ``symmetric=True``.
            Prune colors never used as hubs and compact the remaining ones
            (reduces the number of VJPs/JVPs during decompression).
            Defaults to ``False``, matching SparseMatrixColorings.jl.
        **kwargs: {sample_kwargs_detect}

    Returns:
        A [`ColoredPattern`][asdex.ColoredPattern] ready for [`jacobian_from_coloring`][asdex.jacobian_from_coloring].
    """
    sparsity = _detect_jacobian_sparsity(
        f, *args, argnums=argnums, has_aux=has_aux, **kwargs
    )
    return jacobian_coloring_from_sparsity(
        sparsity, symmetric=symmetric, mode=mode, postprocess=postprocess
    )


@_fill_doc
def hessian_coloring(
    f: Callable,
    *args: Any,
    argnums: int | Sequence[int] = _DEFAULT_ARGNUMS,
    has_aux: bool = _DEFAULT_HAS_AUX,
    mode: HessianMode | None = _DEFAULT_MODE,
    symmetric: bool = _DEFAULT_SYMMETRIC_HESSIAN,
    postprocess: bool = _DEFAULT_POSTPROCESS,
    **kwargs: Any,
) -> ColoredPattern:
    """Detect Hessian sparsity and color in one step.

    Args:
        f: Scalar-valued function taking one or more positional arrays.
        *args: {sample_args}
        argnums: {argnums}
        has_aux: {has_aux_detect}
        mode: AD composition strategy for Hessian-vector products.
            ``"fwd_over_rev"`` uses forward-over-reverse,
            ``"rev_over_fwd"`` uses reverse-over-forward,
            ``"rev_over_rev"`` uses reverse-over-reverse.
            Defaults to ``"fwd_over_rev"``.
        symmetric: {symmetric}
            Defaults to True (exploits H = H^T for fewer colors).
        postprocess: Only read when ``symmetric=True``.
            Prune colors never used as hubs and compact the remaining ones
            (reduces the number of HVPs during decompression).
            Defaults to ``False``, matching SparseMatrixColorings.jl.
        **kwargs: {sample_kwargs_detect}

    Returns:
        A [`ColoredPattern`][asdex.ColoredPattern] ready for [`hessian_from_coloring`][asdex.hessian_from_coloring].
    """
    sparsity = _detect_hessian_sparsity(
        f, *args, argnums=argnums, has_aux=has_aux, **kwargs
    )
    return hessian_coloring_from_sparsity(
        sparsity, symmetric=symmetric, mode=mode, postprocess=postprocess
    )


@_fill_doc
def jacobian_coloring_from_sparsity(
    sparsity: SparsityPattern | NDArray | BCOO,
    *,
    mode: JacobianMode | None = _DEFAULT_MODE,
    symmetric: bool = _DEFAULT_SYMMETRIC_JACOBIAN,
    postprocess: bool = _DEFAULT_POSTPROCESS,
) -> ColoredPattern:
    """Color a sparsity pattern for sparse Jacobian computation.

    Assigns colors so that same-colored rows (or columns) can be
    computed together in a single VJP (or JVP).

    Args:
        sparsity: A [`SparsityPattern`][asdex.SparsityPattern], NumPy array,
            or JAX BCOO matrix of shape ``(m, n)``.
        mode: AD mode.
            ``"fwd"`` uses JVPs (column coloring),
            ``"rev"`` uses VJPs (row coloring).
            ``None`` picks whichever of fwd/rev needs fewer colors
            (unless ``symmetric`` is True, in which case defaults to ``"fwd"``).
        symmetric: {symmetric}
            Requires a square pattern.
        postprocess: Only read when ``symmetric=True``.
            Prune colors never used as hubs and compact the remaining ones
            (reduces the number of VJPs/JVPs during decompression).
            Defaults to ``False``, matching SparseMatrixColorings.jl.

    Returns:
        A [`ColoredPattern`][asdex.ColoredPattern] ready for [`jacobian_from_coloring`][asdex.jacobian_from_coloring].
    """
    sparsity = _coerce_sparsity(sparsity, "jacobian")

    if mode is not None:
        _assert_jacobian_mode(mode)

    if symmetric:
        return _color_jacobian_symmetric(
            sparsity,
            mode if mode is not None else "fwd",
            postprocess=postprocess,
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
    postprocess: bool = _DEFAULT_POSTPROCESS,
) -> ColoredPattern:
    """Color a sparsity pattern for sparse Hessian computation.

    Args:
        sparsity: A [`SparsityPattern`][asdex.SparsityPattern], NumPy array,
            or JAX BCOO matrix of shape ``(n, n)``.
        mode: AD composition strategy for Hessian-vector products.
            ``"fwd_over_rev"`` uses forward-over-reverse,
            ``"rev_over_fwd"`` uses reverse-over-forward,
            ``"rev_over_rev"`` uses reverse-over-reverse.
            Defaults to ``"fwd_over_rev"``.
        symmetric: {symmetric}
            Defaults to True (exploits Hessian symmetry for fewer colors).
        postprocess: Only read when ``symmetric=True``.
            Prune colors never used as hubs and compact the remaining ones
            (reduces the number of HVPs during decompression).
            Pruned vertices get the neutral color ``-1`` in the output
            (no HVP is computed for them).
            Defaults to ``False``, matching SparseMatrixColorings.jl.

    Returns:
        A [`ColoredPattern`][asdex.ColoredPattern] ready for [`hessian_from_coloring`][asdex.hessian_from_coloring].
    """
    sparsity = _coerce_sparsity(sparsity, "hessian")

    if sparsity.m != sparsity.n:
        msg = f"Hessian sparsity pattern must be square, got shape {sparsity.shape}."
        raise ValueError(msg)

    resolved_mode: HessianMode = mode if mode is not None else "fwd_over_rev"
    if mode is not None:
        _assert_hessian_mode(mode)

    if sparsity.nnz == 0:
        return _empty_hessian_pattern(sparsity, symmetric=symmetric, mode=resolved_mode)

    if symmetric:
        colors_arr, num, star_set = color_symmetric(sparsity, postprocess=postprocess)
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
    *,
    postprocess: bool,
) -> ColoredPattern:
    """Color a Jacobian pattern using symmetric (star) coloring.

    Args:
        sparsity: Sparsity pattern (must be square).
        mode: The resolved AD mode.
        postprocess: Whether to prune unused colors after star coloring.
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
    colors_arr, num, star_set = color_symmetric(sparsity, postprocess=postprocess)
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
