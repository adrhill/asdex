"""Shared layout fact about the compressed matrix ``B``.

Both compress (building ``B``) and decompress (consuming ``B``) need to know
the length of the dimension that compression preserves, ``B``'s second axis.
It lives here, in a leaf module of the package,
so each stage reaches it without importing the other
and their independence over the shared compressed matrix is kept.
"""

from __future__ import annotations

from typing import assert_never

from asdex.pattern import ColoredPattern


def _expected_compressed_dim(coloring: ColoredPattern) -> int:
    """Second-axis length of the compressed matrix ``B`` for ``coloring``.

    ``B`` has shape ``(num_colors, dim)``,
    where ``dim`` is the space that compression preserves,
    the opposite of the seeded space.
    For ``"fwd"`` the seed lives in the input space,
    so ``B``'s columns are the output space of size ``m``.
    For ``"rev"`` and the Hessian modes the seed lives in the output
    or cotangent space, so ``B``'s columns are the selected input space of size ``n``.

    Both equal the space the gather's ``elem_idx`` indexes,
    so this is the dimension ``decompress_data`` validates against.
    """
    sparsity = coloring.sparsity
    match coloring.mode:
        case "fwd":
            return sparsity.m
        case "rev" | "fwd_over_rev" | "rev_over_fwd" | "rev_over_rev":
            return sparsity.n
        case _ as unreachable:
            assert_never(unreachable)
