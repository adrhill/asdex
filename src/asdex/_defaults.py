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

from asdex._types import OutputFormat

# When changing a default here, check its documentation in ``_docstrings.py``.
# Some fragments interpolate these constants and stay in sync automatically,
# but fragments whose prose depends on what the value means (the ``mode`` fragments)
# quote it literally and must be updated by hand.

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
# Resolved tolerance fallbacks used when rtol/atol are left as ``_DEFAULT_TOL``.
# The looser matvec tolerance reflects randomized probing,
# the tighter dense tolerance reflects an exact element-wise comparison.
_DEFAULT_MATVEC_TOL: float = 1e-5
_DEFAULT_DENSE_TOL: float = 1e-7
