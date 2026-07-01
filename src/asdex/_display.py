"""Text rendering primitives for sparse matrix visualizations.

Turns the shape and ``(rows, cols)`` coordinates of a sparse ``(m, n)`` grid
into dot, braille, and side-by-side/stacked string visualizations.
These helpers know nothing about the pattern data structures:
callers pass the raw shape and coordinate arrays,
and [`asdex._pattern`][] builds the ``SparsityPattern`` / ``ColoredPattern``
string representations on top of them.

Adapted from SparseArrays.jl (MIT license)
Copyright (c) 2018-2024 SparseArrays.jl contributors:
https://github.com/JuliaSparse/SparseArrays.jl/contributors
https://github.com/JuliaSparse/SparseArrays.jl/
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Thresholds for switching from dot display to braille (Julia-style heuristics)
_SMALL_ROWS = 16
_SMALL_COLS = 40


def _render(m: int, n: int, rows: NDArray[np.int32], cols: NDArray[np.int32]) -> str:
    """Render visualization without header.

    Uses dot display (●/⋅) for small matrices, braille for large ones.
    """
    if m <= _SMALL_ROWS and n <= _SMALL_COLS:
        return _render_dots(m, n, rows, cols)

    braille = _render_braille(m, n, rows, cols)
    braille_lines = braille.split("\n")
    if braille_lines and braille_lines[0] != "(empty)":
        n_lines = len(braille_lines)
        bordered = []
        for i, line in enumerate(braille_lines):
            if i == 0:
                bordered.append("⎡" + line + "⎤")
            elif i == n_lines - 1:
                bordered.append("⎣" + line + "⎦")
            else:
                bordered.append("⎢" + line + "⎥")
        return "\n".join(bordered)
    return braille


def _render_dots(
    m: int, n: int, rows: NDArray[np.int32], cols: NDArray[np.int32]
) -> str:
    """Render small matrix using dots and bullets.

    Uses '⋅' for zeros and '●' for non-zeros.
    """
    if m == 0 or n == 0:
        return "(empty)"

    nonzeros = {(int(i), int(j)) for i, j in zip(rows, cols, strict=True)}
    lines = []
    for i in range(m):
        row_chars = ["●" if (i, j) in nonzeros else "⋅" for j in range(n)]
        lines.append(" ".join(row_chars))
    return "\n".join(lines)


def _render_braille(
    m: int,
    n: int,
    rows: NDArray[np.int32],
    cols: NDArray[np.int32],
    max_height: int = 20,
    max_width: int = 40,
) -> str:
    """Render sparsity pattern using Unicode braille characters.

    Each braille character represents a 4x2 block of the matrix.
    Large matrices are downsampled by linearly interpolating each
    non-zero position to the output grid.
    """
    if m == 0 or n == 0:
        return "(empty)"

    # Uniform scale preserving the aspect ratio (matches Julia's SparseArrays).
    # Pick the tighter constraint so neither dimension overflows.
    if m > 4 * max_height or n > 2 * max_width:
        s = min(2 * max_width / n, 4 * max_height / m)
        scale_height = max(int(s * m), 8)
        scale_width = max(int(s * n), 4)
    else:
        scale_height = max(m, 8)
        scale_width = max(n, 4)

    # Output braille grid dimensions
    out_rows = (scale_height - 1) // 4 + 1
    out_cols = (scale_width - 1) // 2 + 1

    # Braille dot bits: index = (col_offset % 2) * 4 + (row_offset % 4)
    braille_bits = [0x01, 0x02, 0x04, 0x40, 0x08, 0x10, 0x20, 0x80]

    grid = [[0] * out_cols for _ in range(out_rows)]

    # Scale each non-zero to the output grid via linear interpolation
    row_denom = max(m - 1, 1)
    col_denom = max(n - 1, 1)
    for i, j in zip(rows, cols, strict=True):
        si = round(int(i) * (scale_height - 1) / row_denom)
        sj = round(int(j) * (scale_width - 1) / col_denom)
        grid[si // 4][sj // 2] |= braille_bits[(sj % 2) * 4 + (si % 4)]

    lines = ["".join(chr(0x2800 + bits) for bits in row) for row in grid]
    return "\n".join(lines)


def _render_side_by_side(left_lines: list[str], right_lines: list[str]) -> str:
    """Join two visualizations side-by-side with ``→`` on the middle line."""
    max_left = max((len(line) for line in left_lines), default=0)
    n_lines = max(len(left_lines), len(right_lines))
    mid = n_lines // 2

    result = []
    for i in range(n_lines):
        left = left_lines[i] if i < len(left_lines) else ""
        right = right_lines[i] if i < len(right_lines) else ""
        sep = " → " if i == mid else "   "
        result.append(f"{left:<{max_left}}{sep}{right}")
    return "\n".join(result)


def _render_stacked(top_lines: list[str], bottom_lines: list[str]) -> str:
    """Join two visualizations stacked with centered ``↓`` between them."""
    top_width = max((len(line) for line in top_lines), default=0)
    bottom_width = max((len(line) for line in bottom_lines), default=0)
    full_width = max(top_width, bottom_width)

    result = list(top_lines)
    pad = full_width // 2
    result.append(" " * pad + "↓")
    result.extend(bottom_lines)
    return "\n".join(result)
