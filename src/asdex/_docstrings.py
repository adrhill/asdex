"""Shared docstring fragments for the public API.

The argument descriptions and the ``jax.jit`` note below are identical across
many entry points in the public API. They are written once here and
interpolated into each ``{placeholder}`` by the ``@_fill_doc`` decorator,
so a wording fix lands in one place. Only the *description* is interpolated.
A few fragments use f-strings to interpolate default values from
``asdex._defaults`` at import time,
so a documented default can never drift from the signature default it describes.
That value interpolation is separate from the ``{placeholder}`` fragment
substitution ``_fill_doc`` performs later.
The ``argname:`` prefix stays literal in each docstring
so pydocstyle's D417 still sees every argument documented.
Substitution matches only registered ``{placeholder}`` tokens (see ``_PLACEHOLDER``),
so ordinary braces in a docstring pass through untouched
and an unregistered placeholder fails loudly at import
instead of corrupting the rendered text.
Fragments are canonical (dedented):
continuation lines sit at 0 spaces for the top-level ``{jit}`` note
and 8 for the ``Args:`` descriptions.
``_fill_doc`` runs ``inspect.cleandoc`` first so placeholders land at those columns
regardless of the per-version docstring dedenting (which changed in 3.13).
"""

from __future__ import annotations

import inspect
import re
from collections.abc import Callable
from typing import Any, TypeVar

from asdex._defaults import (
    _DEFAULT_ARGNUMS,
    _DEFAULT_CHUNK_SIZE,
    _DEFAULT_DENSE_TOL,
    _DEFAULT_HAS_AUX,
    _DEFAULT_HESSIAN_MODE,
    _DEFAULT_HOLOMORPHIC,
    _DEFAULT_MATVEC_TOL,
    _DEFAULT_NUM_PROBES,
    _DEFAULT_POSTPROCESS,
    _DEFAULT_SYMMETRIC_HESSIAN,
    _DEFAULT_SYMMETRIC_JACOBIAN,
    _DEFAULT_SYMMETRIC_JACOBIAN_MODE,
    _DEFAULT_VERIFY_METHOD,
)

_JIT = """For repeated evaluation, wrap the returned function in ``jax.jit``:
each unjitted call re-traces ``f``,
which can cost far more than the differentiation itself.
The ``"numpy_dense"`` and scipy output formats cannot be jitted
since they produce non-JAX arrays."""

# Input function and coloring

_F_JAC = "Function whose Jacobian is to be computed."

_F_HESS = "Scalar-valued function whose Hessian is to be computed."

_COLORING = "Pre-computed colored sparsity pattern of type ``ColoredPattern``."

_COLORING_COMPRESSED = "The ``ColoredPattern`` that produced ``compressed``."

_COMPRESSED = "The compressed matrix ``B`` of shape ``(num_colors, dim)``."

# Sample inputs and differentiation options

_SAMPLE_ARGS = """Sample arguments of ``f``.
        Only structure and dtypes are used, values are ignored."""

_X_MULTI_INPUT = """For multi-input functions (where ``argnums`` is a tuple),
        pass a tuple of all positional arguments."""

_ARGNUMS = f"""Specifies which positional argument(s) to differentiate
        with respect to.
        Defaults to ``{_DEFAULT_ARGNUMS}``."""

_HAS_AUX = f"""Whether ``f`` returns ``(output, auxiliary_data)``.
        Defaults to ``{_DEFAULT_HAS_AUX}``."""

_HAS_AUX_DETECT = """Whether ``f`` returns ``(output, auxiliary_data)``.
        When True, the auxiliary output is ignored
        and only ``output`` is analyzed for sparsity."""

_HOLOMORPHIC = f"""Whether ``f`` is promised to be holomorphic.
        Defaults to ``{_DEFAULT_HOLOMORPHIC}``."""

_ALLOW_INT_JAC = "Whether to allow differentiating with respect to integer inputs."

_ALLOW_INT_HESS = """Unsupported for Hessians; passing ``True`` raises ``TypeError``
        (integer inputs cannot be differentiated twice, matching ``jax.hessian``)."""

_MODE_JAC = f"""AD mode for Jacobian computation.
        ``"fwd"`` uses JVPs (forward-mode AD),
        ``"rev"`` uses VJPs (reverse-mode AD).
        Defaults to picking whichever of fwd/rev needs fewer colors
        (unless ``symmetric`` is True, in which case defaults to ``"{_DEFAULT_SYMMETRIC_JACOBIAN_MODE}"``)."""

_MODE_JAC_COLORING = f"""AD mode.
        ``"fwd"`` uses JVPs (column coloring),
        ``"rev"`` uses VJPs (row coloring).
        Defaults to picking whichever of fwd/rev needs fewer colors
        (unless ``symmetric`` is True, in which case defaults to ``"{_DEFAULT_SYMMETRIC_JACOBIAN_MODE}"``)."""

_MODE_HESS = f"""AD composition strategy for Hessian-vector products.
        ``"fwd_over_rev"`` uses forward-over-reverse,
        ``"rev_over_fwd"`` uses reverse-over-forward,
        ``"rev_over_rev"`` uses reverse-over-reverse.
        Defaults to ``"{_DEFAULT_HESSIAN_MODE}"``."""

_SYMMETRIC_JAC = f"""Whether to use symmetric coloring.
        Defaults to ``{_DEFAULT_SYMMETRIC_JACOBIAN}``."""

_SYMMETRIC_HESS = f"""Whether to use symmetric coloring.
        Defaults to ``{_DEFAULT_SYMMETRIC_HESSIAN}``."""

_POSTPROCESS_JAC = f"""Only read when ``symmetric=True``.
        Prune colors never used as hubs and compact the remaining ones
        (reduces the number of VJPs/JVPs during decompression).
        Defaults to ``{_DEFAULT_POSTPROCESS}``."""

_POSTPROCESS_HESS = f"""Only read when ``symmetric=True``.
        Prune colors never used as hubs and compact the remaining ones
        (reduces the number of HVPs during decompression).
        Defaults to ``{_DEFAULT_POSTPROCESS}``."""

_CHUNK_SIZE = f"""Maximum number of colors to process in parallel.
        Defaults to ``{_DEFAULT_CHUNK_SIZE}``, processing all colors in a single vmapped batch.
        When specified, colors are processed in chunks of this size to reduce
        peak memory usage."""

_SAMPLE_KWARGS = """Sample keyword arguments of ``f``.
        Merged with ``sample_args`` based on ``f``'s signature."""

_SAMPLE_KWARGS_DETECT = """Sample keyword arguments of ``f``.
        Non-traceable values (bools, strings, ints) are bound statically."""

# Verification

_VERIFY_METHOD = f"""Verification method.
        ``"matvec"`` uses randomized matrix-vector products,
        which is O(k) in the number of probes.
        ``"dense"`` materializes the full dense matrix,
        which is O(n^2).
        Defaults to ``"{_DEFAULT_VERIFY_METHOD}"``."""

_NUM_PROBES = f"""Number of random probe vectors (only used by ``"matvec"``).
        Defaults to ``{_DEFAULT_NUM_PROBES}``."""

_SEED = 'PRNG seed for reproducibility (only used by ``"matvec"``).'

_RTOL = f"""Relative tolerance for comparison.
        Defaults to {_DEFAULT_MATVEC_TOL} for ``"matvec"`` and {_DEFAULT_DENSE_TOL} for ``"dense"``."""

_ATOL = f"""Absolute tolerance for comparison.
        Defaults to {_DEFAULT_MATVEC_TOL} for ``"matvec"`` and {_DEFAULT_DENSE_TOL} for ``"dense"``."""

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

# The registered fragments, keyed by the ``{placeholder}`` name they fill.
_FRAGMENTS: dict[str, str] = {
    "jit": _JIT,
    "f_jac": _F_JAC,
    "f_hess": _F_HESS,
    "coloring": _COLORING,
    "coloring_compressed": _COLORING_COMPRESSED,
    "compressed": _COMPRESSED,
    "sample_args": _SAMPLE_ARGS,
    "x_multi_input": _X_MULTI_INPUT,
    "argnums": _ARGNUMS,
    "has_aux": _HAS_AUX,
    "has_aux_detect": _HAS_AUX_DETECT,
    "holomorphic": _HOLOMORPHIC,
    "allow_int_jac": _ALLOW_INT_JAC,
    "allow_int_hess": _ALLOW_INT_HESS,
    "mode_jac": _MODE_JAC,
    "mode_jac_coloring": _MODE_JAC_COLORING,
    "mode_hess": _MODE_HESS,
    "symmetric_jac": _SYMMETRIC_JAC,
    "symmetric_hess": _SYMMETRIC_HESS,
    "postprocess_jac": _POSTPROCESS_JAC,
    "postprocess_hess": _POSTPROCESS_HESS,
    "chunk_size": _CHUNK_SIZE,
    "sample_kwargs": _SAMPLE_KWARGS,
    "sample_kwargs_detect": _SAMPLE_KWARGS_DETECT,
    "verify_method": _VERIFY_METHOD,
    "num_probes": _NUM_PROBES,
    "seed": _SEED,
    "rtol": _RTOL,
    "atol": _ATOL,
    "format_jac": _FORMAT_JAC,
    "format_hess": _FORMAT_HESS,
    "format_flat": _FORMAT_FLAT,
}

# A placeholder is a fragment key wrapped in braces, e.g. ``{f_jac}``.
# Restricting the token to a lowercase identifier means ordinary braces in a
# docstring (dict literals, empty ``{}``, numeric format specs like ``{0:.2f}``)
# never match, so ``str.format``'s "every brace is a field" fragility is gone:
# only a genuine ``{placeholder}`` is ever touched.
_PLACEHOLDER = re.compile(r"\{([a-z_][a-z0-9_]*)\}")


def _fill_doc(fn: _F) -> _F:
    """Interpolate the shared docstring fragments into ``fn.__doc__``.

    Returns ``fn`` unchanged apart from its docstring, so the signature stays
    visible to type checkers and ``mkdocstrings``.
    ``inspect.cleandoc`` normalizes the indentation first so the result is
    identical whether or not the interpreter already dedented ``__doc__``
    (auto-dedenting landed in Python 3.13).

    Only registered ``{placeholder}`` tokens are substituted.
    An unregistered one raises ``KeyError`` here at import time,
    naming the offending function so the typo is caught immediately.
    """
    if fn.__doc__ is None:
        return fn

    def replace(match: re.Match[str]) -> str:
        key = match.group(1)
        try:
            return _FRAGMENTS[key]
        except KeyError:
            raise KeyError(
                f"Unknown docstring placeholder '{{{key}}}' in "
                f"{getattr(fn, '__qualname__', fn)}. "
                f"Known fragments: {sorted(_FRAGMENTS)}."
            ) from None

    fn.__doc__ = _PLACEHOLDER.sub(replace, inspect.cleandoc(fn.__doc__))
    return fn
