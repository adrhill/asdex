"""Type aliases and resolution for AD mode selection."""

from typing import Literal, get_args

JacobianMode = Literal["fwd", "rev"]
"""AD mode for Jacobian computation.

``"fwd"`` uses JVPs (forward-mode AD),
``"rev"`` uses VJPs (reverse-mode AD).
"""

HessianMode = Literal["fwd_over_rev", "rev_over_fwd", "rev_over_rev"]
"""AD composition strategy for Hessian-vector products.

``"fwd_over_rev"`` uses forward-over-reverse,
``"rev_over_fwd"`` uses reverse-over-forward,
``"rev_over_rev"`` uses reverse-over-reverse.
"""

ColoringMode = JacobianMode | HessianMode
"""AD mode that a coloring was computed for."""

_JaxOutputFormat = Literal["bcoo", "dense"]
"""JAX-native output formats."""

_NumpyOutputFormat = Literal["numpy_dense"]
"""NumPy output formats."""

_ScipyOutputFormat = Literal["scipy_coo", "scipy_csr", "scipy_csc"]
"""SciPy sparse output formats (require scipy)."""

OutputFormat = _JaxOutputFormat | _NumpyOutputFormat | _ScipyOutputFormat
"""Output format for materialized Jacobians and Hessians.

``"bcoo"`` returns ``jax.experimental.sparse.BCOO`` (default),
``"dense"`` returns ``jax.Array``,
``"numpy_dense"`` returns ``numpy.ndarray``,
``"scipy_coo"`` returns ``scipy.sparse.coo_array``,
``"scipy_csr"`` returns ``scipy.sparse.csr_array``,
``"scipy_csc"`` returns ``scipy.sparse.csc_array``.
SciPy formats require scipy and only support 2D arrays
(flat inputs and outputs); PyTree inputs/outputs raise ValueError.
"""


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


def _assert_output_format(output_format: str) -> None:
    """Raise ``ValueError`` if *output_format* is not a valid ``OutputFormat``."""
    if output_format not in get_args(OutputFormat):
        raise ValueError(
            f"Unknown output_format {output_format!r}. "
            "Expected 'bcoo', 'dense', 'numpy_dense', 'scipy_coo', 'scipy_csr', or 'scipy_csc'."
        )
