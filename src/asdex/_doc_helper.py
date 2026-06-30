"""Shared docstring fragments for the public API in ``decompression._api``.

The argument descriptions and the ``jax.jit`` note below are identical across
many entry points in ``decompression._api``. They are written once here and
interpolated into each ``{placeholder}`` by the ``@_fill_doc`` decorator, so a
wording fix lands in one place. Only the *description* is interpolated. The
``argname:`` prefix stays literal in each docstring so pydocstyle's D417 still
sees every argument documented.
Fragments are canonical (dedented): continuation lines sit at 0 spaces for the
top-level ``{jit}`` note and 8 for the ``Args:`` descriptions. ``_fill_doc``
runs ``inspect.cleandoc`` first so placeholders land at those columns
regardless of the per-version docstring dedenting (which changed in 3.13).
"""

from __future__ import annotations

import inspect
from collections.abc import Callable
from typing import Any, TypeVar

_JIT = """For repeated evaluation, wrap the returned function in ``jax.jit``:
each unjitted call re-traces ``f``,
which can cost far more than the differentiation itself.
The ``"numpy_dense"`` and scipy output formats cannot be jitted
since they produce non-JAX arrays."""

# Input function and coloring

_F_JAC = "Function whose Jacobian is to be computed."

_F_HESS = "Scalar-valued function whose Hessian is to be computed."

_COLORING = "Pre-computed colored sparsity pattern."

_COLORING_COMPRESSED = "The colored pattern that produced ``compressed``."

# Sample inputs and differentiation options

_SAMPLE_ARGS = """Sample arguments of ``f``.
        Only structure and dtypes are used, values are ignored."""

_ARGNUMS = """Specifies which positional argument(s) to differentiate
        with respect to (default ``0``)."""

_HAS_AUX = "Whether ``f`` returns ``(output, auxiliary_data)``."

_HOLOMORPHIC = "Whether ``f`` is promised to be holomorphic."

_ALLOW_INT_JAC = "Whether to allow differentiating with respect to integer inputs."

_ALLOW_INT_HESS = """Unsupported for Hessians; passing ``True`` raises ``TypeError``
        (integer inputs cannot be differentiated twice, matching ``jax.hessian``)."""

_MODE_HESS = "AD mode for Hessian computation."

_SYMMETRIC = "Whether to use symmetric (star) coloring."

_CHUNK_SIZE = """Maximum number of colors to process in parallel.
        When ``None`` (default), all colors are processed in a single vmapped batch.
        When specified, colors are processed in chunks of this size to reduce
        peak memory usage."""

_SAMPLE_KWARGS = """Sample keyword arguments of ``f``.
        Merged with ``sample_args`` based on ``f``'s signature."""

# Output format

_FORMAT_HEAD = """Type of the output matrix.
        ``"bcoo"`` returns ``jax.experimental.sparse.BCOO`` (default),
        ``"dense"`` returns ``jax.Array``,
        ``"numpy_dense"`` returns ``numpy.ndarray``,
        ``"scipy_coo"`` returns ``scipy.sparse.coo_array``,
        ``"scipy_csr"`` returns ``scipy.sparse.csr_array``,
        ``"scipy_csc"`` returns ``scipy.sparse.csc_array``."""

_FORMAT_JAC = (
    _FORMAT_HEAD
    + "\n        SciPy formats require scipy and only support 2D Jacobians:"
    + "\n        the input and output must each be a single flat (1D) array"
    + "\n        (scalar outputs are not supported)."
)
_FORMAT_HESS = (
    _FORMAT_HEAD
    + "\n        SciPy formats require scipy and only support 2D Hessians:"
    + "\n        the input must be a single flat (1D) array."
)
_FORMAT_FLAT = _FORMAT_HEAD + "\n        SciPy formats require scipy."

_F = TypeVar("_F", bound=Callable[..., Any])


def _fill_doc(fn: _F) -> _F:
    """Interpolate the shared docstring fragments into ``fn.__doc__``.

    Returns ``fn`` unchanged apart from its docstring, so the signature stays
    visible to type checkers and ``mkdocstrings``.
    ``inspect.cleandoc`` normalizes the indentation first so the result is
    identical whether or not the interpreter already dedented ``__doc__``
    (auto-dedenting landed in Python 3.13).
    """
    if fn.__doc__ is not None:
        fn.__doc__ = inspect.cleandoc(fn.__doc__).format(
            jit=_JIT,
            f_jac=_F_JAC,
            f_hess=_F_HESS,
            coloring=_COLORING,
            coloring_compressed=_COLORING_COMPRESSED,
            sample_args=_SAMPLE_ARGS,
            argnums=_ARGNUMS,
            has_aux=_HAS_AUX,
            holomorphic=_HOLOMORPHIC,
            allow_int_jac=_ALLOW_INT_JAC,
            allow_int_hess=_ALLOW_INT_HESS,
            mode_hess=_MODE_HESS,
            symmetric=_SYMMETRIC,
            chunk_size=_CHUNK_SIZE,
            sample_kwargs=_SAMPLE_KWARGS,
            format_jac=_FORMAT_JAC,
            format_hess=_FORMAT_HESS,
            format_flat=_FORMAT_FLAT,
        )
    return fn
