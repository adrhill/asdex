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

from asdex._types import HessianMode, JacobianMode, OutputFormat

# When changing a default here, check its documentation in ``_docstrings.py``.
# The fragments interpolate these constants at import time,
# so a documented default stays in sync automatically.

# Detection, coloring, and differentiation
_DEFAULT_ARGNUMS: int = 0
_DEFAULT_HAS_AUX: bool = False
_DEFAULT_HOLOMORPHIC: bool = False
_DEFAULT_ALLOW_INT: bool = False
_DEFAULT_MODE: None = None
_DEFAULT_SYMMETRIC_JACOBIAN: bool = False
_DEFAULT_SYMMETRIC_HESSIAN: bool = True
_DEFAULT_OUTPUT_FORMAT: OutputFormat = "bcoo"
_DEFAULT_CHUNK_SIZE: int | None = None
# Resolved AD modes chosen when ``mode`` is left as ``_DEFAULT_MODE`` (None).
# Hessians fall back to forward-over-reverse HVPs;
# symmetric Jacobian coloring falls back to forward mode (JVPs).
_DEFAULT_HESSIAN_MODE: HessianMode = "fwd_over_rev"
_DEFAULT_SYMMETRIC_JACOBIAN_MODE: JacobianMode = "fwd"

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
