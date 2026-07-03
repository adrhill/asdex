"""Type aliases for AD modes and output formats, plus AD-mode and output-format validators."""

from typing import Literal, TypeAlias, get_args

JacobianMode: TypeAlias = Literal["fwd", "rev"]
"""AD mode for Jacobian computation.

``"fwd"`` uses JVPs (forward-mode AD),
``"rev"`` uses VJPs (reverse-mode AD).
"""

HessianMode: TypeAlias = Literal["fwd_over_rev", "rev_over_fwd", "rev_over_rev"]
"""AD composition strategy for Hessian-vector products.

``"fwd_over_rev"`` uses forward-over-reverse,
``"rev_over_fwd"`` uses reverse-over-forward,
``"rev_over_rev"`` uses reverse-over-reverse.
"""

ColoringMode: TypeAlias = JacobianMode | HessianMode
"""AD mode that a coloring was computed for."""

JaxOutputFormat: TypeAlias = Literal["bcoo", "dense"]
"""JAX-native output formats."""

NumpyOutputFormat: TypeAlias = Literal["numpy_dense"]
"""NumPy output formats."""

ScipyOutputFormat: TypeAlias = Literal["scipy_coo", "scipy_csr", "scipy_csc"]
"""SciPy sparse output formats (require scipy)."""

OutputFormat: TypeAlias = JaxOutputFormat | NumpyOutputFormat | ScipyOutputFormat
"""Output format for materialized Jacobians and Hessians.

``"bcoo"`` returns ``jax.experimental.sparse.BCOO`` (default),
``"dense"`` returns ``jax.Array``,
``"numpy_dense"`` returns ``numpy.ndarray``,
``"scipy_coo"`` returns ``scipy.sparse.coo_array``,
``"scipy_csr"`` returns ``scipy.sparse.csr_array``,
``"scipy_csc"`` returns ``scipy.sparse.csc_array``.
SciPy formats require scipy and only support 2D arrays:
the input (and, for Jacobians, the output) must be a single flat (1D) array.
PyTree inputs/outputs and scalar-output Jacobians raise ValueError.
BCOO and SciPy outputs mirror the detected sparsity pattern:
structural non-zeros that are numerically zero at the evaluation point
are kept as explicit entries,
so the structure is independent of the input value.
"""

# get_args on a ``|``-union of Literals returns the nested Literal types,
# not their string values, so flatten one level to recover the values.
_OUTPUT_FORMATS = tuple(
    value for literal in get_args(OutputFormat) for value in get_args(literal)
)

# Output formats backed by host (non-JAX) arrays.
# These cannot be returned from a caller-side ``jax.jit``.
_HOST_FORMATS = ("numpy_dense", "scipy_coo", "scipy_csr", "scipy_csc")


def _assert_output_format(output_format: str) -> None:
    """Raise if *output_format* is not a valid, usable ``OutputFormat``.

    Raises:
        ValueError: If *output_format* is not a valid ``OutputFormat``.
        ImportError: If *output_format* is a scipy format and scipy is not installed.
            Checked here so that requesting a scipy format fails at construction time
            rather than at the first call.
    """
    if output_format not in _OUTPUT_FORMATS:
        raise ValueError(
            f"Unknown output_format {output_format!r}. "
            "Expected 'bcoo', 'dense', 'numpy_dense', 'scipy_coo', 'scipy_csr', or 'scipy_csc'."
        )
    if output_format in get_args(ScipyOutputFormat):
        _assert_scipy_installed(output_format)


def _assert_scipy_installed(output_format: str) -> None:
    """Raise ``ImportError`` if scipy is not installed.

    The hint points at the optional dependency
    so a scipy format fails with an actionable message rather than a bare import error.
    """
    try:
        import scipy.sparse  # noqa: PLC0415, F401
    except ImportError as e:
        raise ImportError(
            f"scipy is required for output_format={output_format!r}. "
            "Install it with: pip install 'asdex[scipy]'"
        ) from e


def _assert_jacobian_mode(mode: str) -> None:
    """Raise ``ValueError`` if *mode* is not a valid ``JacobianMode``."""
    if mode not in get_args(JacobianMode):
        raise ValueError(f"Unknown mode {mode!r}. Expected 'fwd' or 'rev'.")


def _assert_hessian_mode(mode: str) -> None:
    """Raise ``ValueError`` if *mode* is not a valid ``HessianMode``."""
    if mode not in get_args(HessianMode):
        raise ValueError(
            f"Unknown mode {mode!r}. "
            "Expected 'fwd_over_rev', 'rev_over_fwd', or 'rev_over_rev'."
        )


def _assert_coloring_mode(mode: str) -> None:
    """Raise ``ValueError`` if *mode* is not a valid ``ColoringMode``."""
    if mode not in (*get_args(JacobianMode), *get_args(HessianMode)):
        raise ValueError(f"Unknown mode {mode!r}.")
