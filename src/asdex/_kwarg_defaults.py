"""Default keyword-argument values shared across the public API.

Centralizing the defaults keeps the many public entry points in sync:
each signature references a constant here instead of repeating a literal.
The detection, coloring, differentiation, and verification entry points
all draw their shared defaults from here.
Jacobians default to non-symmetric coloring,
Hessians to symmetric (star) coloring.
"""

from __future__ import annotations

from typing import Literal

from asdex._modes import OutputFormat

# Detection, coloring, and differentiation
_DEFAULT_ARGNUMS: int = 0
_DEFAULT_HAS_AUX: bool = False
_DEFAULT_HOLOMORPHIC: bool = False
_DEFAULT_ALLOW_INT: bool = False
_DEFAULT_MODE: None = None
_DEFAULT_SYMMETRIC_JACOBIAN: bool = False
_DEFAULT_SYMMETRIC_HESSIAN: bool = True
_DEFAULT_POSTPROCESS: bool = False
_DEFAULT_OUTPUT_FORMAT: OutputFormat = "bcoo"
_DEFAULT_CHUNK_SIZE: int | None = None

# Verification
_DEFAULT_VERIFY_METHOD: Literal["matvec", "dense"] = "matvec"
_DEFAULT_NUM_PROBES: int = 25
_DEFAULT_SEED: int = 0
_DEFAULT_TOL: float | None = None
