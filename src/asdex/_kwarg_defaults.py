"""Default keyword-argument values shared across the public API.

Centralizing the defaults keeps the many ``jacobian`` / ``hessian`` /
``compressed_*`` entry points in sync:
each signature references a constant here instead of repeating a literal.
Jacobians default to non-symmetric coloring,
Hessians to symmetric (star) coloring.
"""

from __future__ import annotations

from asdex.modes import OutputFormat

_DEFAULT_ARGNUMS: int = 0
_DEFAULT_HAS_AUX: bool = False
_DEFAULT_HOLOMORPHIC: bool = False
_DEFAULT_ALLOW_INT: bool = False
_DEFAULT_MODE: None = None
_DEFAULT_SYMMETRIC_JACOBIAN: bool = False
_DEFAULT_SYMMETRIC_HESSIAN: bool = True
_DEFAULT_OUTPUT_FORMAT: OutputFormat = "bcoo"
_DEFAULT_CHUNK_SIZE: int | None = None
