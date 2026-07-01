"""Pretty-printing for SparsityPattern and ColoredPattern.

Adapted from SparseArrays.jl (MIT license)
Copyright (c) 2018-2024 SparseArrays.jl contributors:
https://github.com/JuliaSparse/SparseArrays.jl/contributors
https://github.com/JuliaSparse/SparseArrays.jl/
"""

from __future__ import annotations

from asdex._pattern import ColoredPattern, SparsityPattern

# Thresholds for switching from dot display to braille (Julia-style heuristics)
_SMALL_ROWS = 16
_SMALL_COLS = 40

# Human-readable AD primitive names for display
_MODE_PRIMITIVE: dict[str, str] = {
    "fwd": "JVP",
    "rev": "VJP",
    "fwd_over_rev": "HVP",
    "rev_over_fwd": "HVP",
    "rev_over_rev": "HVP",
}


# SparsityPattern display


def sparsity_str(pattern: SparsityPattern) -> str:
    """Full string representation with header and visualization."""
    header = (
        f"SparsityPattern({pattern.m}×{pattern.n}, "
        f"nnz={pattern.nnz}, sparsity={1 - pattern.density:.1%})"
    )
    return f"{header}\n{_render(pattern)}"


def sparsity_repr(pattern: SparsityPattern) -> str:
    """Compact single-line representation."""
    return f"SparsityPattern(shape={pattern.shape}, nnz={pattern.nnz})"


# ColoredPattern display


def colored_repr(colored: ColoredPattern) -> str:
    """Compact single-line representation."""
    sp = colored.sparsity
    m, n = sp.shape
    c = colored.num_colors
    primitive = _MODE_PRIMITIVE[colored.mode]
    return (
        f"ColoredPattern({m}×{n}, nnz={sp.nnz}, sparsity={1 - sp.density:.1%}, "
        f"{primitive}, {c} {'color' if c == 1 else 'colors'})"
    )


def colored_str(colored: ColoredPattern) -> str:
    """Full string representation with AD savings summary and visualization.

    Column compression (fwd/symmetric) shows side-by-side with ``→``.
    Row compression (rev) shows stacked with ``↓``.
    """
    m, n = colored.sparsity.shape
    c = colored.num_colors
    primitive = _MODE_PRIMITIVE[colored.mode]
    s = "" if c == 1 else "s"

    def _plural(count: int, word: str) -> str:
        return f"{count} {word}" if count == 1 else f"{count} {word}s"

    if colored.symmetric:
        instead = f"instead of {_plural(n, 'HVP')}"
    else:
        instead = f"instead of {_plural(m, 'VJP')} or {_plural(n, 'JVP')}"
    header = f"{colored_repr(colored)}\n  {c} {primitive}{s} ({instead})"

    compressed = _compressed_pattern(colored)
    left_lines = _render(colored.sparsity).split("\n")
    right_lines = _render(compressed).split("\n")

    if colored._compresses_columns:
        viz = _render_side_by_side(left_lines, right_lines)
    else:
        viz = _render_stacked(left_lines, right_lines)

    return f"{header}\n{viz}"


def _compressed_pattern(colored: ColoredPattern) -> SparsityPattern:
    """Build the compressed sparsity pattern after coloring.

    For column compression (JVP/HVP, shape ``(m, num_colors)``):
    entry ``(i, c)`` is present iff any column ``j``
    with ``colors[j] == c`` has a nonzero at ``(i, j)``.

    For row compression (VJP, shape ``(num_colors, n)``):
    entry ``(c, j)`` is present iff any row ``i``
    with ``colors[i] == c`` has a nonzero at ``(i, j)``.
    """
    cls = type(colored.sparsity)
    comp_rows: list[int] = []
    comp_cols: list[int] = []

    if colored._compresses_columns:
        # Compress columns: (m, n) → (m, num_colors)
        seen: set[tuple[int, int]] = set()
        for i, j in zip(colored.sparsity.rows, colored.sparsity.cols, strict=True):
            c = int(colored.colors[j])
            entry = (int(i), c)
            if entry not in seen:
                seen.add(entry)
                comp_rows.append(entry[0])
                comp_cols.append(entry[1])
        shape = (colored.sparsity.m, colored.num_colors)
    else:
        # Compress rows: (m, n) → (num_colors, n)
        seen = set()
        for i, j in zip(colored.sparsity.rows, colored.sparsity.cols, strict=True):
            c = int(colored.colors[i])
            entry = (c, int(j))
            if entry not in seen:
                seen.add(entry)
                comp_rows.append(entry[0])
                comp_cols.append(entry[1])
        shape = (colored.num_colors, colored.sparsity.n)

    return cls.from_coo(comp_rows, comp_cols, shape)


# Rendering helpers


def _render(pattern: SparsityPattern) -> str:
    """Render visualization without header.

    Uses dot display (●/⋅) for small matrices, braille for large ones.
    """
    if pattern.m <= _SMALL_ROWS and pattern.n <= _SMALL_COLS:
        return _render_dots(pattern)

    braille = _render_braille(pattern)
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


def _render_dots(pattern: SparsityPattern) -> str:
    """Render small matrix using dots and bullets.

    Uses '⋅' for zeros and '●' for non-zeros.
    """
    if pattern.m == 0 or pattern.n == 0:
        return "(empty)"

    dense = pattern.todense()
    lines = []
    for i in range(pattern.m):
        row_chars = ["●" if dense[i, j] else "⋅" for j in range(pattern.n)]
        lines.append(" ".join(row_chars))
    return "\n".join(lines)


def _render_braille(
    pattern: SparsityPattern,
    max_height: int = 20,
    max_width: int = 40,
) -> str:
    """Render sparsity pattern using Unicode braille characters.

    Each braille character represents a 4x2 block of the matrix.
    Large matrices are downsampled by linearly interpolating each
    non-zero position to the output grid.
    """
    if pattern.m == 0 or pattern.n == 0:
        return "(empty)"

    # Uniform scale preserving the aspect ratio (matches Julia's SparseArrays).
    # Pick the tighter constraint so neither dimension overflows.
    if pattern.m > 4 * max_height or pattern.n > 2 * max_width:
        s = min(2 * max_width / pattern.n, 4 * max_height / pattern.m)
        scale_height = max(int(s * pattern.m), 8)
        scale_width = max(int(s * pattern.n), 4)
    else:
        scale_height = max(pattern.m, 8)
        scale_width = max(pattern.n, 4)

    # Output braille grid dimensions
    out_rows = (scale_height - 1) // 4 + 1
    out_cols = (scale_width - 1) // 2 + 1

    # Braille dot bits: index = (col_offset % 2) * 4 + (row_offset % 4)
    braille_bits = [0x01, 0x02, 0x04, 0x40, 0x08, 0x10, 0x20, 0x80]

    grid = [[0] * out_cols for _ in range(out_rows)]

    # Scale each non-zero to the output grid via linear interpolation
    row_denom = max(pattern.m - 1, 1)
    col_denom = max(pattern.n - 1, 1)
    for i, j in zip(pattern.rows, pattern.cols, strict=True):
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
